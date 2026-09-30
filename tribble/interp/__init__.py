"""Frame interpolation: model loading and frame-rate conversion.

Supported interpolation models:

* RIFE v4.2 - v4.26 checkpoints (``flownet*.pkl`` / ``.pth`` / ``.safetensors``)
* ``builtin:blend``: linear cross-fade (no motion; fast, mostly for testing)
* ``builtin:minterpolate``: ffmpeg's motion-compensated interpolation
  (handled by ffmpeg filters, not by :class:`FrameRateConverter`)
* user plugins declaring ``KIND = "interpolation"`` whose ``build()`` returns a
  module called as ``model(img0, img1, t)``
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from ..models.loader import ModelLoadError, _read_state_dict, resolve_device
from ..models.plugins import ArchPlugin, detect_plugin, load_plugins
from ..models.registry import plugin_dirs, read_sidecar
from .rife import build_rife, clean_state_dict, is_rife

log = logging.getLogger(__name__)

FFMPEG_METHOD = "builtin:minterpolate"
BUILTIN_INTERPOLATORS = ["builtin:blend", FFMPEG_METHOD]


@dataclass
class InterpModel:
    """An interpolation model: ``prepare`` a frame once, then ``infer`` in-betweens."""

    name: str
    backend: str
    device: torch.device
    dtype: torch.dtype
    pad_multiple: int
    _infer: Callable[[Any, Any, float], torch.Tensor]
    _prepare: Callable[[torch.Tensor], Any]
    info: Dict[str, Any] = field(default_factory=dict)

    def prepare(self, frame: torch.Tensor) -> Any:
        """1CHW float [0,1] (any device) -> backend state reused across timesteps."""
        return self._prepare(frame.to(self.device, self.dtype))

    def infer(self, s0: Any, s1: Any, t: float) -> torch.Tensor:
        return self._infer(s0, s1, t)

    def describe(self) -> str:
        parts = [self.name, self.backend, str(self.device)]
        if self.dtype == torch.float16:
            parts.append("fp16")
        return " | ".join(parts)


def _padded(x: torch.Tensor, m: int):
    h, w = x.shape[-2:]
    ph, pw = -(-h // m) * m, -(-w // m) * m
    if (ph, pw) != (h, w):
        x = F.pad(x, (0, pw - w, 0, ph - h), mode="replicate")
    return x, (h, w)


def load_interpolator(
    spec: str,
    device: str | torch.device = "auto",
    half: bool = False,
    flow_scale: float = 1.0,
    ensemble: bool = False,
    allow_unsafe_pickle: bool = False,
    plugins: Optional[Dict[str, ArchPlugin]] = None,
) -> InterpModel:
    dev = device if isinstance(device, torch.device) else resolve_device(device)
    use_half = bool(half) and dev.type in ("cuda", "mps")

    if spec == "builtin:blend":
        return InterpModel(
            "blend", "builtin", dev, torch.float32, 1,
            _infer=lambda a, b, t: torch.lerp(a, b, float(t)),
            _prepare=lambda x: x,
        )
    if spec == FFMPEG_METHOD:
        raise ModelLoadError("builtin:minterpolate runs inside ffmpeg and has no Python model")
    if spec.startswith("builtin:"):
        raise ModelLoadError(f"Unknown builtin interpolator {spec!r}; use one of {BUILTIN_INTERPOLATORS}")

    path = Path(spec).expanduser()
    if not path.is_file():
        raise ModelLoadError(f"Interpolation model not found: {spec}")
    sidecar = read_sidecar(path)
    if sidecar.get("half") is False:
        use_half = False
    name = sidecar.get("name") or path.stem

    obj = _read_state_dict(path, allow_unsafe_pickle)
    if not isinstance(obj, dict):
        raise ModelLoadError(f"{path.name}: expected a state dict for an interpolation model")
    sd = clean_state_dict(obj)

    if plugins is None:
        plugins = load_plugins(plugin_dirs())
    interp_plugins = {k: p for k, p in plugins.items() if p.kind == "interpolation"}

    forced = sidecar.get("arch")
    if forced and forced != "rife":
        if forced not in interp_plugins:
            raise ModelLoadError(f"Sidecar asks for interpolation plugin {forced!r}, which isn't installed")
        return _plugin_model(interp_plugins[forced], sd, sidecar, name, dev, use_half)

    if forced == "rife" or is_rife(sd):
        return _rife_model(sd, sidecar, name, dev, use_half, flow_scale, ensemble)

    for plugin in detect_plugin(interp_plugins, sd):
        return _plugin_model(plugin, sd, sidecar, name, dev, use_half)

    raise ModelLoadError(
        f"{path.name} isn't a recognised interpolation model (RIFE v4.2+). "
        "Write a plugin with KIND = \"interpolation\" to use other architectures."
    )


def _rife_model(sd, sidecar, name, dev, half, flow_scale, ensemble) -> InterpModel:
    try:
        net = build_rife(sd, sidecar.get("scale_list"))
    except Exception as exc:
        raise ModelLoadError(f"Could not load RIFE model {name}: {exc}") from exc
    if ensemble and net.block0.out_channels > 6:
        # versions that pass features between blocks (v4.21+, except v4.24) have no ensemble mode
        log.warning("Ensemble mode is not supported by this RIFE version; disabled")
        ensemble = False
    dtype = torch.float16 if half else torch.float32
    net = net.to(dev, dtype)
    flow_scale = float(flow_scale or 1.0)
    pad = max(net.pad_multiple, int(net.pad_multiple / flow_scale))

    @torch.inference_mode()
    def prepare(x):
        xp, size = _padded(x, pad)
        return xp, net.features(xp), size

    @torch.inference_mode()
    def infer(s0, s1, t):
        (x0, f0, (h, w)), (x1, f1, _) = s0, s1
        out = net(x0, x1, t, f0, f1, flow_scale=flow_scale, ensemble=ensemble)
        return out[..., :h, :w]

    return InterpModel(
        name, "rife", dev, dtype, pad, infer, prepare,
        info={"blocks": net.n_blocks, "encoder": net.encode is not None, "flow_scale": flow_scale,
              "ensemble": ensemble},
    )


def _plugin_model(plugin: ArchPlugin, sd, sidecar, name, dev, half) -> InterpModel:
    result = plugin.build(sd, dict(sidecar.get("options", {})))
    meta: dict = {}
    if isinstance(result, dict):
        meta, result = result, result["model"]
    if meta.get("supports_half") is False:
        half = False
    dtype = torch.float16 if half else torch.float32
    module = result.to(dev, dtype)
    try:
        module.eval()
    except NotImplementedError:
        pass
    pad = int(meta.get("pad_multiple", 1) or 1)

    @torch.inference_mode()
    def prepare(x):
        return _padded(x, pad)

    @torch.inference_mode()
    def infer(s0, s1, t):
        (x0, (h, w)), (x1, _) = s0, s1
        out = module(x0, x1, float(t))
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out[..., :h, :w]

    return InterpModel(name, f"plugin:{plugin.name}", dev, dtype, pad, infer, prepare)


# --------------------------------------------------------------------------
# frame-rate conversion
# --------------------------------------------------------------------------


def rate_fraction(value: float | str) -> Fraction:
    """Parse "24000/1001", 23.976, 59.94 ... into an exact fraction (NTSC-aware)."""
    if isinstance(value, str) and "/" in value:
        n, d = value.split("/")
        return Fraction(int(n), int(d))
    v = float(value)
    ntsc = v * 1.001
    if abs(ntsc - round(ntsc)) < 0.005 and abs(v - round(v)) > 0.005:
        return Fraction(round(ntsc) * 1000, 1001)
    return Fraction(v).limit_denominator(1000)


def fraction_str(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}"


def output_rate(src: Fraction, mode: str, factor: float, target_fps: float) -> Fraction:
    if mode == "fps":
        if target_fps <= 0:
            raise ValueError("Target frame rate must be positive")
        return rate_fraction(target_fps)
    if factor <= 0:
        raise ValueError("Interpolation factor must be positive")
    return src * Fraction(factor).limit_denominator(1000)


def scene_changed(a: torch.Tensor, b: torch.Tensor, threshold: float) -> bool:
    """Mean absolute luma difference of small thumbnails, in [0, 1]."""
    if threshold <= 0:
        return False
    h, w = a.shape[-2:]
    size = (max(1, round(64 * h / max(h, w))), max(1, round(64 * w / max(h, w))))
    wts = torch.tensor([0.299, 0.587, 0.114], device=a.device).view(1, 3, 1, 1)
    ta = F.interpolate((a[:, :3].float() * wts).sum(1, keepdim=True), size=size, mode="area")
    tb = F.interpolate((b[:, :3].float() * wts).sum(1, keepdim=True), size=size, mode="area")
    return float((ta - tb).abs().mean()) > threshold


class FrameRateConverter:
    """Streams frames in, yields frames at ``ratio`` times the input rate.

    Output frame *k* sits at source position ``k / ratio``; positions between
    two source frames are synthesised by the model (or the earlier frame is
    repeated across a scene cut). The final source frame is held for its full
    duration so the output length matches the input.
    """

    def __init__(self, model: InterpModel, ratio: Fraction, scene_threshold: float = 0.0):
        if ratio <= 0:
            raise ValueError("ratio must be positive")
        self.model = model
        self.ratio = Fraction(ratio)
        self.scene_threshold = scene_threshold
        self._prev: Optional[torch.Tensor] = None
        self._prev_state: Any = None
        self._index = -1  # source index of _prev
        self.scene_cuts = 0

    def expected_outputs(self, n_inputs: int) -> int:
        return math.ceil(n_inputs * self.ratio)

    def _range(self, i: int) -> range:
        return range(math.ceil(i * self.ratio), math.ceil((i + 1) * self.ratio))

    def push(self, frame: torch.Tensor) -> List[torch.Tensor]:
        if self._prev is None:
            self._prev, self._prev_state, self._index = frame, None, 0
            return []
        out: List[torch.Tensor] = []
        i, prev = self._index, self._prev
        cut = scene_changed(prev, frame, self.scene_threshold)
        if cut:
            self.scene_cuts += 1
        cur_state = None
        for k in self._range(i):
            t = k / self.ratio - i
            if t == 0 or cut:
                out.append(prev)
                continue
            if self._prev_state is None:
                self._prev_state = self.model.prepare(prev)
            if cur_state is None:
                cur_state = self.model.prepare(frame)
            y = self.model.infer(self._prev_state, cur_state, float(t))
            out.append(y.float().clamp_(0, 1).to(prev.device))
        self._prev, self._prev_state, self._index = frame, cur_state, i + 1
        return out

    def flush(self) -> List[torch.Tensor]:
        if self._prev is None:
            return []
        out = [self._prev for _ in self._range(self._index)]
        self._prev = self._prev_state = None
        return out


def minterpolate_filter(rate: Fraction, extra: str = "") -> str:
    base = f"minterpolate=fps={fraction_str(rate)}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
    return f"{base}:{extra}" if extra else base
