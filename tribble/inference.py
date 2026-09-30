"""Frame upscaling engine: tiling, padding, blending and model chaining."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import JobConfig, ModelStep, ProcessingSettings
from .models import LoadedModel, load_model, resolve_device
from .models.plugins import load_plugins
from .models.registry import plugin_dirs

log = logging.getLogger(__name__)

MIN_AUTO_TILE = 64


@dataclass
class ChainStep:
    model: LoadedModel
    strength: float = 1.0
    tile_size: Optional[int] = None


class Upscaler:
    """Runs a chain of models over single frames."""

    def __init__(self, steps: Sequence[ChainStep], settings: Optional[ProcessingSettings] = None):
        if not steps:
            raise ValueError("At least one model is required")
        self.steps = list(steps)
        self.settings = settings or ProcessingSettings()
        # Tile sizes can shrink at runtime after an out-of-memory error.
        self._tile = [self._initial_tile(s) for s in self.steps]

    # ---- construction -------------------------------------------------
    @classmethod
    def from_config(cls, cfg: JobConfig, log_fn: Callable[[str], None] = log.info) -> "Upscaler":
        steps = [s for s in cfg.models if s.enabled]
        if not steps:
            raise ValueError("No enabled models in the chain")
        p = cfg.processing
        plugins = load_plugins(plugin_dirs())
        if plugins:
            log_fn(f"Loaded arch plugins: {', '.join(plugins)}")
        chain = []
        for step in steps:
            m = load_model(
                step.model,
                device=p.device,
                half=p.half_precision,
                allow_unsafe_pickle=p.allow_unsafe_pickle,
                onnx_providers=p.onnx_providers or None,
                plugins=plugins,
            )
            log_fn(f"Loaded model: {m.describe()}")
            chain.append(ChainStep(m, step.strength, step.tile_size))
        return cls(chain, p)

    @property
    def total_scale(self) -> float:
        return math.prod(s.model.scale for s in self.steps)

    @property
    def device(self) -> torch.device:
        return self.steps[0].model.device

    def _initial_tile(self, step: ChainStep) -> int:
        if step.tile_size is not None:
            return int(step.tile_size)
        if step.model.tile_size:
            return int(step.model.tile_size)
        return int(self.settings.tile_size)

    # ---- frame I/O helpers -------------------------------------------
    @staticmethod
    def to_tensor(frame: np.ndarray) -> torch.Tensor:
        """HWC uint8/uint16 RGB -> 1CHW float32 in [0, 1]."""
        maxval = 65535.0 if frame.dtype == np.uint16 else 255.0
        t = torch.from_numpy(np.array(frame, dtype=np.float32, copy=True)).div_(maxval)
        return t.permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def to_numpy(t: torch.Tensor, bit_depth: int = 8) -> np.ndarray:
        t = t.squeeze(0).clamp_(0, 1).permute(1, 2, 0)
        if bit_depth == 16:
            return t.mul_(65535.0).round_().to(torch.int32).cpu().numpy().astype(np.uint16)
        return t.mul_(255.0).round_().to(torch.uint8).cpu().numpy()

    # ---- processing ---------------------------------------------------
    @torch.inference_mode()
    def upscale_array(self, frame: np.ndarray) -> np.ndarray:
        bit_depth = 16 if frame.dtype == np.uint16 else 8
        return self.to_numpy(self.upscale(self.to_tensor(frame)), bit_depth)

    @torch.inference_mode()
    def upscale(self, x: torch.Tensor) -> torch.Tensor:
        for i, step in enumerate(self.steps):
            x = self._run_step(i, step, x)
        return x

    def _run_step(self, idx: int, step: ChainStep, x: torch.Tensor) -> torch.Tensor:
        model = step.model
        x = x.to(model.device)
        while True:
            try:
                y = run_tiled(model, x, self._tile[idx], self.settings.tile_overlap, self._pad_multiple(model))
                break
            except RuntimeError as exc:
                if not _is_oom(exc):
                    raise
                self._free()
                h, w = x.shape[-2:]
                cur = self._tile[idx] or max(h, w)
                new = _next_pow2_below(cur)
                if new < MIN_AUTO_TILE:
                    raise RuntimeError(
                        f"Out of memory even at tile size {cur}. Try CPU, fp16 or a smaller model."
                    ) from exc
                log.warning("Out of memory at tile %s; retrying with tile %s", cur, new)
                self._tile[idx] = new

        if step.strength < 1.0:
            base = F.interpolate(x.float(), size=y.shape[-2:], mode="bicubic", align_corners=False)
            y = torch.lerp(base.to(y.device), y, float(step.strength)).clamp_(0, 1)
        return y

    def _pad_multiple(self, model: LoadedModel) -> int:
        a, b = max(model.pad_multiple, 1), max(self.settings.tile_pad_multiple, 1)
        return a * b // math.gcd(a, b)

    def _free(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def effective_tiles(self) -> List[int]:
        return list(self._tile)


def _is_oom(exc: Exception) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "failed to allocate" in msg


def _next_pow2_below(n: int) -> int:
    p = 1 << max(int(n - 1).bit_length() - 1, 0)
    return p if p < n else n // 2


# --------------------------------------------------------------------------
# tiling
# --------------------------------------------------------------------------


def _pad_to(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    ph, pw = h - x.shape[-2], w - x.shape[-1]
    if ph <= 0 and pw <= 0:
        return x
    return F.pad(x, (0, max(pw, 0), 0, max(ph, 0)), mode="replicate")


def _round_up(v: int, m: int) -> int:
    return -(-v // m) * m if m > 1 else v


def _run_padded(model: LoadedModel, x: torch.Tensor, pad_multiple: int, size: Optional[Tuple[int, int]] = None):
    """Run the model on x after padding to its requirements; crop the result."""
    h, w = x.shape[-2:]
    th, tw = size if size else (_round_up(h, pad_multiple), _round_up(w, pad_multiple))
    y = model(_pad_to(x, th, tw))
    s = y.shape[-1] / tw
    return y[..., : round(h * s), : round(w * s)], s


def _positions(length: int, tile: int, overlap: int) -> List[int]:
    if length <= tile:
        return [0]
    step = max(tile - overlap, 1)
    pos = list(range(0, length - tile, step))
    pos.append(length - tile)  # last tile flush with the edge
    return pos


def _ramp(n: int, lo_blend: int, hi_blend: int, device) -> torch.Tensor:
    """1D weight: rises over lo_blend px, flat, falls over hi_blend px."""
    w = torch.ones(n, device=device)
    if lo_blend > 0:
        w[:lo_blend] = torch.linspace(1.0 / (lo_blend + 1), 1.0, lo_blend + 1, device=device)[:lo_blend]
    if hi_blend > 0:
        w[n - hi_blend :] = torch.linspace(1.0, 1.0 / (hi_blend + 1), hi_blend + 1, device=device)[1:]
    return w


def run_tiled(
    model: LoadedModel,
    x: torch.Tensor,
    tile: int,
    overlap: int = 16,
    pad_multiple: int = 1,
) -> torch.Tensor:
    """Upscale a 1CHW tensor, optionally in overlapping tiles blended with linear ramps."""
    _, _, h, w = x.shape
    fixed = model.fixed_size

    if fixed is None and (tile <= 0 or (h <= tile and w <= tile)):
        y, _ = _run_padded(model, x, pad_multiple)
        return y.float()

    if fixed is not None:
        th, tw = fixed
    else:
        th = tw = _round_up(tile, pad_multiple)
    overlap = max(0, min(overlap, min(th, tw) // 2))
    ys, xs = _positions(h, th, overlap), _positions(w, tw, overlap)

    out = weight = None
    scale = model.scale
    for y0 in ys:
        for x0 in xs:
            patch = x[..., y0 : y0 + th, x0 : x0 + tw]
            ph, pw = patch.shape[-2:]
            size = fixed if fixed else (_round_up(ph, pad_multiple), _round_up(pw, pad_multiple))
            res, scale = _run_padded(model, patch, pad_multiple, size)
            res = res.float()
            if out is None:
                out = torch.zeros(x.shape[0], res.shape[1], round(h * scale), round(w * scale), device=res.device)
                weight = torch.zeros(1, 1, out.shape[2], out.shape[3], device=res.device)

            oy, ox = round(y0 * scale), round(x0 * scale)
            rh = min(res.shape[-2], out.shape[-2] - oy)
            rw = min(res.shape[-1], out.shape[-1] - ox)
            res = res[..., :rh, :rw]
            ob = round(overlap * scale)
            wy = _ramp(rh, ob if y0 > 0 else 0, ob if y0 + th < h else 0, res.device)
            wx = _ramp(rw, ob if x0 > 0 else 0, ob if x0 + tw < w else 0, res.device)
            mask = (wy[:, None] * wx[None, :])[None, None]
            out[..., oy : oy + rh, ox : ox + rw] += res * mask
            weight[..., oy : oy + rh, ox : ox + rw] += mask

    return out / weight.clamp_min(1e-8)


def build_upscaler(cfg: JobConfig, log_fn: Callable[[str], None] = log.info) -> Upscaler:
    return Upscaler.from_config(cfg, log_fn)


__all__ = ["ChainStep", "Upscaler", "build_upscaler", "run_tiled", "resolve_device", "ModelStep"]
