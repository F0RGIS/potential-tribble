"""Example architecture plugin: classic SRCNN (Dong et al., 2014).

Copy this file as a starting point for any architecture spandrel doesn't
support. Drop plugins into ./plugins or ~/.tribble/plugins.

SRCNN works on an image that has already been bicubic-upscaled, so the
wrapper below does that upsampling itself; the model's "scale" therefore
comes from the sidecar options (default 2).

Sidecar example (``my_srcnn.pth.json`` next to the weights)::

    {"arch": "srcnn", "options": {"scale": 3, "channels": 3}}
"""

import torch
import torch.nn.functional as F
from torch import nn

NAME = "srcnn"


class SRCNN(nn.Module):
    def __init__(self, channels: int = 3, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(channels, 64, 9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, channels, 5, padding=2)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x).clamp(0, 1)


def detect(state_dict) -> bool:
    """Return True if these weights look like SRCNN. Optional."""
    keys = {"conv1.weight", "conv2.weight", "conv3.weight"}
    if not keys.issubset(state_dict) or len(state_dict) != 6:
        return False
    return tuple(state_dict["conv1.weight"].shape[1:]) in ((1, 9, 9), (3, 9, 9))


def build(state_dict, options):
    """Build the model and load weights. Must return a Module or a dict."""
    channels = state_dict["conv1.weight"].shape[1]
    scale = int(options.get("scale", 2))
    model = SRCNN(channels=channels, scale=scale)
    model.load_state_dict(state_dict)
    return {
        "model": model,
        "scale": scale,
        "in_channels": channels,  # 1-channel models get luma in, gray out
        "out_channels": channels,
        "supports_half": True,
    }
