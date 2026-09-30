import os
from fractions import Fraction
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tribble.config import JobConfig, ModelStep
from tribble.interp import (
    FrameRateConverter,
    load_interpolator,
    output_rate,
    rate_fraction,
    scene_changed,
)
from tribble.interp.rife import RIFE, Head, IFBlock, build_rife
from tribble.models.plugins import load_plugins
from tribble.pipeline import run_job
from tribble.video import probe


# --------------------------------------------------------------------------
# rates
# --------------------------------------------------------------------------


def test_rate_fraction():
    assert rate_fraction("24000/1001") == Fraction(24000, 1001)
    assert rate_fraction(23.976) == Fraction(24000, 1001)
    assert rate_fraction(59.94) == Fraction(60000, 1001)
    assert rate_fraction(60) == 60
    assert rate_fraction(12.5) == Fraction(25, 2)


def test_output_rate():
    src = Fraction(24000, 1001)
    assert output_rate(src, "factor", 2, 0) == Fraction(48000, 1001)
    assert output_rate(src, "fps", 0, 60) == 60
    with pytest.raises(ValueError):
        output_rate(src, "fps", 0, 0)


# --------------------------------------------------------------------------
# frame-rate conversion
# --------------------------------------------------------------------------


def blend_model():
    return load_interpolator("builtin:blend", "cpu")


def frames(n, h=8, w=8):
    return [torch.full((1, 3, h, w), i / 10.0) for i in range(n)]


def run_converter(conv, fs):
    out = []
    for f in fs:
        out += conv.push(f)
    return out + conv.flush()


def test_double_rate_positions():
    fs = frames(5)
    out = run_converter(FrameRateConverter(blend_model(), Fraction(2)), fs)
    assert len(out) == 10
    vals = [round(float(o.mean()), 4) for o in out]
    # originals on even outputs, midpoints between, last frame held
    assert vals == [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.4]


def test_24_to_60():
    conv = FrameRateConverter(blend_model(), Fraction(60) / Fraction(24))
    out = run_converter(conv, frames(8))
    assert len(out) == conv.expected_outputs(8) == 20
    # output k sits at source position k * 24/60
    for k in (1, 3, 7):
        pos = k * 24 / 60
        assert float(out[k].mean()) == pytest.approx(pos / 10, abs=1e-5)


def test_rate_reduction_drops_frames():
    out = run_converter(FrameRateConverter(blend_model(), Fraction(1, 2)), frames(6))
    assert [round(float(o.mean()), 3) for o in out] == [0.0, 0.2, 0.4]


def test_scene_cut_repeats_frame():
    a = torch.zeros(1, 3, 16, 16)
    b = torch.ones(1, 3, 16, 16)
    assert scene_changed(a, b, 0.12) and not scene_changed(a, a + 0.01, 0.12)
    conv = FrameRateConverter(blend_model(), Fraction(2), scene_threshold=0.12)
    out = run_converter(conv, [a, b])
    assert conv.scene_cuts == 1
    assert float(out[1].mean()) == 0.0  # repeated, not a 50% cross-fade


# --------------------------------------------------------------------------
# RIFE layout detection across versions
# --------------------------------------------------------------------------


def _variant(name):
    """Mimic the layout of several published RIFE versions (random weights)."""
    torch.manual_seed(0)
    if name == "v4.2":  # legacy head, no encoder
        blocks = [IFBlock(7, 64, 5, legacy=True)] + [IFBlock(12, 32, 5, legacy=True) for _ in range(3)]
        return RIFE(blocks, None, [8, 4, 2, 1], True)
    if name == "v4.6":
        blocks = [IFBlock(7, 64, 6)] + [IFBlock(12, c, 6) for c in (48, 32, 32)]
        return RIFE(blocks, None, [8, 4, 2, 1], True)
    if name == "v4.9":  # small sequential encoder
        enc = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, 2, 1), torch.nn.ConvTranspose2d(16, 4, 4, 2, 1))
        blocks = [IFBlock(15, 64, 6)] + [IFBlock(20, 32, 6) for _ in range(3)]
        return RIFE(blocks, enc, [8, 4, 2, 1], False)
    if name == "v4.14-lite":  # grouped ResConv
        blocks = [IFBlock(7 + 16, 64, 6, groups=2)] + [IFBlock(8 + 4 + 16, 32, 6, groups=2) for _ in range(3)]
        return RIFE(blocks, Head(32, 8), [8, 4, 2, 1], False)
    if name == "v4.22":  # Head encoder + feature passing
        blocks = [IFBlock(7 + 16, 64, 13)] + [IFBlock(8 + 4 + 16 + 8, 32, 13) for _ in range(3)]
        return RIFE(blocks, Head(32, 8), [8, 4, 2, 1], False)
    if name == "v4.26":  # five blocks
        blocks = [IFBlock(7 + 8, 64, 13)] + [IFBlock(8 + 4 + 8 + 8, c, 13) for c in (48, 48, 32, 32)]
        return RIFE(blocks, Head(16, 4), [16, 8, 4, 2, 1], False)
    raise KeyError(name)


@pytest.mark.parametrize("name", ["v4.2", "v4.6", "v4.9", "v4.14-lite", "v4.22", "v4.26"])
def test_rife_rebuild_from_state_dict(name):
    ref = _variant(name).eval()
    sd = {f"module.{k}": v for k, v in ref.state_dict().items()}  # as saved by RIFE training
    sd["module.teacher.dummy"] = torch.zeros(1)  # training-only extras are ignored
    model = build_rife(sd)
    assert model.n_blocks == ref.n_blocks and model.base_scales == ref.base_scales
    x0, x1 = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        a = ref(x0, x1, 0.3, ref.features(x0), ref.features(x1))
        b = model(x0, x1, 0.3, model.features(x0), model.features(x1))
    assert torch.equal(a, b)


def test_rife_missing_weights_rejected():
    sd = _variant("v4.6").state_dict()
    del sd["block3.lastconv.0.bias"]
    with pytest.raises(ValueError, match="missing"):
        build_rife(sd)


def test_rife_loader_pads_and_crops(tmp_path):
    p = tmp_path / "flownet_test.pkl"
    torch.save({f"module.{k}": v for k, v in _variant("v4.22").state_dict().items()}, p)
    m = load_interpolator(str(p), "cpu", ensemble=True)  # ensemble unsupported -> disabled
    assert m.backend == "rife" and m.pad_multiple == 32 and m.info["ensemble"] is False
    a, b = torch.rand(1, 3, 45, 70), torch.rand(1, 3, 45, 70)
    assert m.infer(m.prepare(a), m.prepare(b), 0.5).shape == (1, 3, 45, 70)
    m2 = load_interpolator(str(p), "cpu", flow_scale=0.5)
    assert m2.pad_multiple == 64


def _real_rife() -> list:
    dirs = [os.environ.get("TRIBBLE_TEST_RIFE_DIR", ""), str(Path.home() / ".tribble" / "models")]
    return sorted({str(p) for d in dirs if d and Path(d).is_dir() for p in Path(d).glob("flownet*.pkl")})


@pytest.mark.parametrize("path", _real_rife() or [None])
def test_real_rife_beats_crossfade_on_pan(path):
    if path is None:
        pytest.skip("no RIFE checkpoint found (set TRIBBLE_TEST_RIFE_DIR)")
    torch.manual_seed(3)
    big = F.interpolate(torch.rand(1, 3, 60, 90), size=(240, 360), mode="bicubic", align_corners=False).clamp(0, 1)
    crop = lambda dx: big[..., 20:212, 10 + dx : 330 + dx]  # noqa: E731
    a, mid, b = crop(0), crop(5), crop(10)
    psnr = lambda x, y: float(10 * torch.log10(1 / ((x - y) ** 2).mean()))  # noqa: E731
    m = load_interpolator(path, "cpu")
    assert psnr(m.infer(m.prepare(a), m.prepare(b), 0.5), mid) > psnr((a + b) / 2, mid) + 15


# --------------------------------------------------------------------------
# plugins
# --------------------------------------------------------------------------


def test_interpolation_plugin(tmp_path):
    d = tmp_path / "plugins"
    d.mkdir()
    (d / "avg.py").write_text(
        "import torch\n"
        "KIND = 'interpolation'\n"
        "NAME = 'avgnet'\n"
        "class Avg(torch.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__(); self.w = torch.nn.Parameter(torch.zeros(1))\n"
        "    def forward(self, a, b, t):\n"
        "        return a * (1 - t) + b * t + self.w\n"
        "def detect(sd): return set(sd) == {'w'}\n"
        "def build(sd, options):\n"
        "    m = Avg(); m.load_state_dict(sd); return {'model': m, 'pad_multiple': 8}\n"
    )
    plugins = load_plugins([d])
    assert plugins["avgnet"].kind == "interpolation"
    w = tmp_path / "avg_interp.pth"
    torch.save({"w": torch.zeros(1)}, w)
    m = load_interpolator(str(w), "cpu", plugins=plugins)
    assert m.backend == "plugin:avgnet"
    out = m.infer(m.prepare(torch.zeros(1, 3, 10, 10)), m.prepare(torch.ones(1, 3, 10, 10)), 0.25)
    assert out.shape == (1, 3, 10, 10) and float(out.mean()) == pytest.approx(0.25)


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


def _job(**interp):
    cfg = JobConfig(models=[ModelStep("builtin:bicubic@2")])
    cfg.interpolation.enabled = True
    for k, v in interp.items():
        setattr(cfg.interpolation, k, v)
    return cfg


@pytest.mark.parametrize("order", ["before", "after"])
def test_job_doubles_frame_rate(sample_video, tmp_path, order):
    cfg = _job(model="builtin:blend", order=order)
    out = run_job(sample_video, tmp_path / f"{order}.mkv", cfg, log_fn=lambda s: None)
    info = probe(out)
    assert (info.width, info.height) == (128, 96)
    assert info.fps == pytest.approx(20)
    assert 19 <= info.frame_count <= 21 and info.has_audio


def test_job_target_fps_interpolation_only(sample_video, tmp_path):
    cfg = _job(model="builtin:blend", mode="fps", target_fps=25)
    cfg.models = []
    out = run_job(sample_video, tmp_path / "o.mp4", cfg, log_fn=lambda s: None)
    info = probe(out)
    assert (info.width, info.height) == (64, 48)
    assert info.fps == pytest.approx(25) and 24 <= info.frame_count <= 26


@pytest.mark.parametrize("order", ["before", "after"])
def test_job_ffmpeg_minterpolate(sample_video, tmp_path, order):
    cfg = _job(model="builtin:minterpolate", order=order)
    out = run_job(sample_video, tmp_path / f"m_{order}.mkv", cfg, log_fn=lambda s: None)
    info = probe(out)
    assert info.fps == pytest.approx(20) and (info.width, info.height) == (128, 96)


def test_job_rife(sample_video, tmp_path):
    p = tmp_path / "flownet_tiny.pkl"
    torch.save(_variant("v4.26").state_dict(), p)
    cfg = _job(model=str(p), factor=3)
    out = run_job(sample_video, tmp_path / "r.mkv", cfg, log_fn=lambda s: None)
    info = probe(out)
    assert info.fps == pytest.approx(30) and 29 <= info.frame_count <= 31


def test_cli_interp_flags(sample_video, tmp_path):
    from tribble.cli import main

    preset = tmp_path / "p.json"
    rc = main([
        "upscale", str(sample_video), "--interp", "builtin:blend", "--interp-fps", "15",
        "--interp-order", "before", "--scene-threshold", "0", "--save-preset", str(preset),
        "-o", str(tmp_path / "c.mkv"),
    ])
    assert rc == 0
    cfg = JobConfig.load(preset)
    assert cfg.interpolation.enabled and cfg.interpolation.mode == "fps"
    assert cfg.interpolation.target_fps == 15 and cfg.models == []
    assert probe(tmp_path / "c.mkv").fps == pytest.approx(15)
