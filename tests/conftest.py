import subprocess

import pytest
import torch
from torch import nn


class TinySR(nn.Module):
    """Small local-receptive-field x2 model (conv + pixel shuffle)."""

    def __init__(self, scale: int = 2, nf: int = 8):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, nf, 3, padding=1), nn.ReLU(), nn.Conv2d(nf, 3 * scale * scale, 3, padding=1)
        )
        self.up = nn.PixelShuffle(scale)
        self.scale = scale

    def forward(self, x):
        base = nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")
        return (base + 0.1 * self.up(self.body(x))).clamp(0, 1)


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return TinySR().eval()


@pytest.fixture(scope="session")
def ffmpeg():
    try:
        from tribble.video import find_ffmpeg

        return find_ffmpeg()
    except Exception:
        pytest.skip("ffmpeg not available")


@pytest.fixture
def sample_video(tmp_path, ffmpeg):
    path = tmp_path / "in.mp4"
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440",
         "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
        check=True,
    )
    return path
