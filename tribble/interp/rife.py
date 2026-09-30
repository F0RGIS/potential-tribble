"""RIFE (Real-time Intermediate Flow Estimation) v4.x, one implementation for all versions.

Based on the MIT-licensed RIFE by hzwer (https://github.com/hzwer/Practical-RIFE)
and its per-version ports in vs-rife (https://github.com/HolyWu/vs-rife).
Rather than one file per release, the network layout (block count and widths,
encoder type, feature passing) is read from the checkpoint itself, so v4.2
through v4.26 (including lite/heavy variants) load from the same code.

Checkpoints: ``flownet.pkl`` / ``flownet_v4.X.pkl`` state dicts.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn


def _conv(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.LeakyReLU(0.2, True))


class ResConv(nn.Module):
    def __init__(self, c: int, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1, groups=groups)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    """``legacy`` = v4.2-v4.4 layout: plain convs and a half-resolution ConvTranspose head."""

    def __init__(self, in_planes: int, c: int, out_channels: int, depth: int = 8, groups: int = 1,
                 legacy: bool = False):
        super().__init__()
        self.conv0 = nn.Sequential(_conv(in_planes, c // 2, 2), _conv(c // 2, c, 2))
        self.legacy = legacy
        if legacy:
            self.convblock = nn.Sequential(*[_conv(c, c) for _ in range(depth)])
            self.lastconv = nn.ConvTranspose2d(c, out_channels, 4, 2, 1)
        else:
            self.convblock = nn.Sequential(*[ResConv(c, groups) for _ in range(depth)])
            self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, out_channels * 4, 4, 2, 1), nn.PixelShuffle(2))
        self.out_channels = out_channels

    def forward(self, x, flow=None, scale: float = 1.0):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear") / scale
            x = torch.cat((x, flow), 1)
        feat = self.convblock(self.conv0(x))
        up = scale * 2 if self.legacy else scale
        tmp = F.interpolate(self.lastconv(feat), scale_factor=up, mode="bilinear")
        extra = tmp[:, 5:] if self.out_channels > 6 else None
        return tmp[:, :4] * up, tmp[:, 4:5], extra


class Head(nn.Module):
    """Feature encoder used from v4.13 on (cnn0..cnn3)."""

    def __init__(self, c: int, out: int):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, c, 3, 2, 1)
        self.cnn1 = nn.Conv2d(c, c, 3, 1, 1)
        self.cnn2 = nn.Conv2d(c, c, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(c, out, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.relu(self.cnn0(x.clamp(0.0, 1.0)))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


def _sequential_encoder(sd: Dict[str, torch.Tensor]) -> nn.Sequential:
    """v4.7-v4.12 encoders: nn.Sequential of convs, LeakyReLU at odd indices."""
    idx = sorted({int(m.group(1)) for k in sd if (m := re.match(r"encode\.(\d+)\.weight$", k))})
    layers: List[nn.Module] = []
    for pos in range(idx[-1] + 1):
        if pos in idx:
            w = sd[f"encode.{pos}.weight"]
            if w.shape[-1] == 4:  # ConvTranspose2d weight: (in, out, k, k)
                layers.append(nn.ConvTranspose2d(w.shape[0], w.shape[1], 4, 2, 1))
            else:
                layers.append(nn.Conv2d(w.shape[1], w.shape[0], 3, 2 if pos == 0 else 1, 1))
        else:
            layers.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*layers)


def _warp(img: torch.Tensor, flow: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    h, w = img.shape[-2:]
    flow = flow.float()
    flow = torch.cat([flow[:, 0:1] / ((w - 1.0) / 2.0), flow[:, 1:2] / ((h - 1.0) / 2.0)], 1)
    g = (grid + flow).permute(0, 2, 3, 1)
    return F.grid_sample(img.float(), g, mode="bilinear", padding_mode="border", align_corners=True).to(img.dtype)


class RIFE(nn.Module):
    def __init__(self, blocks: List[IFBlock], encode: Optional[nn.Module], scale_list: List[float],
                 accumulate_mask: bool):
        super().__init__()
        for i, b in enumerate(blocks):
            self.add_module(f"block{i}", b)
        self.n_blocks = len(blocks)
        self.encode = encode
        self.base_scales = scale_list
        self.accumulate_mask = accumulate_mask
        self._grid_cache: Dict[tuple, torch.Tensor] = {}

    @property
    def pad_multiple(self) -> int:
        return int(4 * max(self.base_scales))  # 32 for 4 blocks, 64/128 for 5

    def _grid(self, h: int, w: int, device, dtype) -> torch.Tensor:
        key = (h, w, device, dtype)
        g = self._grid_cache.get(key)
        if g is None:
            gx = torch.linspace(-1.0, 1.0, w, device=device).view(1, 1, 1, w).expand(-1, -1, h, -1)
            gy = torch.linspace(-1.0, 1.0, h, device=device).view(1, 1, h, 1).expand(-1, -1, -1, w)
            g = torch.cat([gx, gy], 1)
            self._grid_cache = {key: g}
        return g

    def features(self, img: torch.Tensor) -> Optional[torch.Tensor]:
        return self.encode(img) if self.encode is not None else None

    def forward(self, img0, img1, timestep: float, f0=None, f1=None, flow_scale: float = 1.0,
                ensemble: bool = False):
        img0, img1 = img0.clamp(0.0, 1.0), img1.clamp(0.0, 1.0)
        n, _, h, w = img0.shape
        grid = self._grid(h, w, img0.device, torch.float32)
        t = torch.full((n, 1, h, w), float(timestep), dtype=img0.dtype, device=img0.device)
        feats = f0 is not None
        blocks = [getattr(self, f"block{i}") for i in range(self.n_blocks)]
        scales = [s / flow_scale for s in self.base_scales]

        def swap(fl):
            return torch.cat((fl[:, 2:4], fl[:, :2]), 1)

        wimg0, wimg1 = img0, img1
        flow = mask = extra = None
        for i, block in enumerate(blocks):
            if flow is None:
                parts = (img0, img1, f0, f1, t) if feats else (img0, img1, t)
                flow, mask, extra = block(torch.cat(parts, 1), None, scales[i])
                if ensemble:
                    parts = (img1, img0, f1, f0, 1 - t) if feats else (img1, img0, 1 - t)
                    fr, mr, _ = block(torch.cat(parts, 1), None, scales[i])
                    flow, mask = (flow + swap(fr)) / 2, (mask - mr) / 2
            else:
                if feats:
                    wf0, wf1 = _warp(f0, flow[:, :2], grid), _warp(f1, flow[:, 2:4], grid)
                    parts = [wimg0, wimg1, wf0, wf1, t, mask]
                    rparts = [wimg1, wimg0, wf1, wf0, 1 - t, -mask]
                else:
                    parts = [wimg0, wimg1, t, mask]
                    rparts = [wimg1, wimg0, 1 - t, -mask]
                if extra is not None:
                    parts.append(extra)
                fd, m0, extra = block(torch.cat(parts, 1), flow, scales[i])
                if ensemble:
                    fr, mr, _ = block(torch.cat(rparts, 1), swap(flow), scales[i])
                    fd, m0 = (fd + swap(fr)) / 2, (m0 - mr) / 2
                    mask = mask + m0 if self.accumulate_mask else m0
                else:
                    mask = mask + m0 if self.accumulate_mask else m0
                flow = flow + fd
            wimg0, wimg1 = _warp(img0, flow[:, :2], grid), _warp(img1, flow[:, 2:4], grid)
        m = torch.sigmoid(mask)
        return wimg0 * m + wimg1 * (1 - m)


# --------------------------------------------------------------------------
# detection / construction
# --------------------------------------------------------------------------


def clean_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    for key in ("state_dict", "model", "params"):
        if isinstance(sd.get(key), dict):
            sd = sd[key]
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}


def is_rife(sd: Dict[str, torch.Tensor]) -> bool:
    return (
        "block0.conv0.0.0.weight" in sd
        and "block1.conv0.0.0.weight" in sd
        and ("block0.lastconv.0.weight" in sd or "block0.lastconv.weight" in sd)
    )


def build_rife(sd: Dict[str, torch.Tensor], scale_list: Optional[List[float]] = None) -> RIFE:
    sd = clean_state_dict(sd)
    if not is_rife(sd):
        raise ValueError("not a RIFE v4 checkpoint")
    if "block0.conv0.0.1.weight" in sd:  # PReLU activations
        raise ValueError("RIFE v4.0/v4.1 checkpoints are not supported; use v4.2 or newer")
    legacy = "block0.lastconv.weight" in sd

    n_blocks = 0
    while f"block{n_blocks}.conv0.0.0.weight" in sd:
        n_blocks += 1
    blocks = []
    for i in range(n_blocks):
        p = f"block{i}."
        in_planes = sd[p + "conv0.0.0.weight"].shape[1]
        c = sd[p + "conv0.1.0.weight"].shape[0]
        depth = len({k.split(".")[2] for k in sd if k.startswith(p + "convblock.")})
        if legacy:
            out_ch, groups = sd[p + "lastconv.weight"].shape[1], 1
        else:
            out_ch = sd[p + "lastconv.0.weight"].shape[1] // 4
            groups = c // sd[p + "convblock.0.conv.weight"].shape[1]
        blocks.append(IFBlock(in_planes, c, out_ch, depth, groups, legacy))

    encode: Optional[nn.Module] = None
    if "encode.cnn0.weight" in sd:
        encode = Head(sd["encode.cnn0.weight"].shape[0], sd["encode.cnn3.weight"].shape[1])
    elif any(k.startswith("encode.") for k in sd):
        encode = _sequential_encoder(sd)

    if scale_list is None:
        if n_blocks == 4:
            scale_list = [8, 4, 2, 1]
        elif n_blocks == 5:
            # v4.25.lite uses a coarser pyramid than v4.25/v4.26
            lite = blocks[-1].conv0[1][0].out_channels <= 24
            scale_list = [32, 16, 8, 4, 1] if lite else [16, 8, 4, 2, 1]
        else:
            scale_list = [2 ** (n_blocks - 1 - i) for i in range(n_blocks)]

    model = RIFE(blocks, encode, [float(s) for s in scale_list], accumulate_mask=encode is None)
    # Checkpoints may carry training-only modules (teacher, caltime, ...): ignore
    # extras, but every weight the network uses must be present.
    missing, _ = model.load_state_dict(sd, strict=False)
    if missing:
        raise ValueError(f"checkpoint is missing {len(missing)} weights, e.g. {missing[0]}")
    return model.eval()
