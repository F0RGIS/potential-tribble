"""Backend-neutral wrapper around a loaded super-resolution model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import torch


@dataclass
class LoadedModel:
    """A callable image model: ``(B, C, H, W)`` float in [0, 1] -> upscaled tensor.

    ``forward`` receives a tensor already on ``device`` and in ``dtype`` and
    must return a tensor (any device/dtype; the caller normalises it).
    """

    name: str
    forward: Callable[[torch.Tensor], torch.Tensor]
    scale: float
    backend: str  # spandrel | torchscript | export | pickle | onnx | plugin | builtin
    device: torch.device
    dtype: torch.dtype = torch.float32
    in_channels: int = 3
    out_channels: int = 3
    arch: str = ""
    # Model can't take arbitrary sizes: (h, w) it requires. Forces tiling.
    fixed_size: Optional[Tuple[int, int]] = None
    # Input dims must be a multiple of this (padding is applied automatically).
    pad_multiple: int = 1
    # Preferred tile size from the model sidecar (0 = no preference).
    tile_size: int = 0
    # Channel order the model expects. Frames arrive as RGB.
    channel_order: str = "rgb"
    info: dict = field(default_factory=dict)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, self.dtype)
        x = _adapt_channels(x, self.in_channels)
        if self.channel_order == "bgr" and x.shape[1] >= 3:
            x = x[:, [2, 1, 0]]
        y = self.forward(x)
        y = y.to(self.device, torch.float32)
        if self.channel_order == "bgr" and y.shape[1] >= 3:
            y = y[:, [2, 1, 0]]
        return _adapt_channels(y, 3)

    def describe(self) -> str:
        parts = [self.name, f"x{self.scale:g}", self.backend]
        if self.arch:
            parts.append(self.arch)
        parts.append(str(self.device))
        if self.dtype == torch.float16:
            parts.append("fp16")
        return " | ".join(parts)


def _adapt_channels(x: torch.Tensor, channels: int) -> torch.Tensor:
    c = x.shape[1]
    if c == channels:
        return x
    if channels == 1:
        # ITU-R BT.601 luma
        w = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x[:, :3] * w).sum(1, keepdim=True)
    if c == 1:
        return x.expand(-1, channels, -1, -1) if channels <= 3 else torch.cat(
            [x.expand(-1, 3, -1, -1), torch.ones_like(x).expand(-1, channels - 3, -1, -1)], 1
        )
    if c > channels:
        return x[:, :channels]
    # e.g. RGB -> RGBA: add an opaque alpha channel
    pad = torch.ones(x.shape[0], channels - c, *x.shape[2:], device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], 1)
