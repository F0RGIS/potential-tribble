import json

import numpy as np
import pytest
import torch

from tribble.config import JobConfig, ModelStep, OutputSettings, ProcessingSettings
from tribble.inference import ChainStep, Upscaler, run_tiled
from tribble.models import load_model
from tribble.models.base import LoadedModel
from tribble.models.plugins import load_plugins
from tribble.video import quality_args, target_size


def wrap(module, **kw) -> LoadedModel:
    return LoadedModel("tiny", lambda x: module(x), 2.0, "test", torch.device("cpu"), **kw)


def test_config_roundtrip(tmp_path):
    cfg = JobConfig(models=[ModelStep("a.pth", 0.5), ModelStep("builtin:bicubic@2")])
    cfg.processing.tile_size = 256
    cfg.output.codec = "libx265"
    p = tmp_path / "preset.json"
    cfg.save(p)
    assert JobConfig.load(p) == cfg


def test_config_tolerates_unknown_and_string_models():
    cfg = JobConfig.from_dict({"models": ["x.pth"], "output": {"codec": "ffv1", "future_key": 1}})
    assert cfg.models[0].model == "x.pth" and cfg.output.codec == "ffv1"


@pytest.mark.parametrize("tile,overlap", [(16, 4), (24, 8), (32, 0), (20, 6)])
def test_tiled_matches_full_frame(tiny_model, tile, overlap):
    m = wrap(tiny_model)
    x = torch.rand(1, 3, 45, 70)
    full = run_tiled(m, x, 0)
    tiled = run_tiled(m, x, tile, overlap)
    assert tiled.shape == full.shape == (1, 3, 90, 140)
    # The model's receptive field is 2px, so overlapping tiles must agree
    # everywhere except within that halo of each seam.
    tol = 0.05 if overlap < 4 else 0.02
    assert (tiled - full).abs().max() < tol


def test_padding_for_pad_multiple(tiny_model):
    m = wrap(tiny_model, pad_multiple=16)
    out = run_tiled(m, torch.rand(1, 3, 37, 21), 0, pad_multiple=16)
    assert out.shape[-2:] == (74, 42)


def test_fixed_size_model_is_tiled(tiny_model):
    seen = []

    def fwd(x):
        seen.append(tuple(x.shape[-2:]))
        return tiny_model(x)

    m = LoadedModel("fixed", fwd, 2.0, "test", torch.device("cpu"), fixed_size=(16, 16))
    out = run_tiled(m, torch.rand(1, 3, 40, 23), 0, overlap=4)
    assert out.shape[-2:] == (80, 46)
    assert set(seen) == {(16, 16)}


def test_chain_and_strength(tiny_model):
    a = wrap(tiny_model)
    b = load_model("builtin:bicubic@2", "cpu")
    up = Upscaler([ChainStep(a, 0.5), ChainStep(b)], ProcessingSettings(tile_size=0))
    frame = (np.random.rand(20, 30, 3) * 255).astype(np.uint8)
    out = up.upscale_array(frame)
    assert out.shape == (80, 120, 3) and out.dtype == np.uint8
    assert up.total_scale == 4

    frame16 = (np.random.rand(10, 12, 3) * 65535).astype(np.uint16)
    assert up.upscale_array(frame16).dtype == np.uint16


def test_oom_fallback_shrinks_tile(tiny_model):
    calls = []

    def fwd(x):
        calls.append(x.shape[-1])
        if x.shape[-1] > 64:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return tiny_model(x)

    m = LoadedModel("oom", fwd, 2.0, "test", torch.device("cpu"))
    up = Upscaler([ChainStep(m)], ProcessingSettings(tile_size=0, tile_overlap=8))
    out = up.upscale(torch.rand(1, 3, 100, 200))
    assert out.shape[-2:] == (200, 400)
    assert up.effective_tiles[0] <= 64


def test_builtins():
    x = torch.rand(1, 3, 8, 8)
    assert load_model("builtin:identity")(x).shape == x.shape
    assert load_model("builtin:nearest@3")(x).shape[-1] == 24
    with pytest.raises(Exception):
        load_model("builtin:magic@2")


def test_plugin_with_sidecar(tmp_path, tiny_model):
    plugdir = tmp_path / "plugins"
    plugdir.mkdir()
    (plugdir / "tiny_arch.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(__import__('pathlib').Path(__file__).parent)!r})\n"
        "from conftest import TinySR\n"
        "NAME = 'tiny'\n"
        "def detect(sd):\n"
        "    return 'body.0.weight' in sd and 'up' not in sd\n"
        "def build(sd, options):\n"
        "    m = TinySR(scale=options.get('scale', 2))\n"
        "    m.load_state_dict(sd)\n"
        "    return {'model': m, 'scale': m.scale}\n"
    )
    weights = tmp_path / "custom.pth"
    torch.save(tiny_model.state_dict(), weights)
    plugins = load_plugins([plugdir])
    assert "tiny" in plugins

    # auto-detected through detect()
    m = load_model(str(weights), "cpu", plugins=plugins)
    assert m.backend == "plugin" and m.arch == "tiny" and m.scale == 2

    # forced via sidecar, with a custom display name
    (tmp_path / "custom.pth.json").write_text(json.dumps({"arch": "tiny", "name": "My Tiny", "options": {}}))
    m = load_model(str(weights), "cpu", plugins=plugins)
    assert m.name == "My Tiny"
    x = torch.rand(1, 3, 10, 10)
    assert torch.allclose(m(x), tiny_model(x), atol=1e-6)


def test_bundled_example_plugin(tmp_path):
    from pathlib import Path

    plugins = load_plugins([Path(__file__).parent.parent / "plugins"])
    srcnn = plugins["srcnn"]
    net = srcnn.build.__globals__["SRCNN"](channels=1, scale=3)
    p = tmp_path / "srcnn_y.pth"
    torch.save(net.state_dict(), p)
    (tmp_path / "srcnn_y.pth.json").write_text(json.dumps({"options": {"scale": 3}}))
    m = load_model(str(p), "cpu", plugins=plugins)
    assert m.arch == "srcnn" and m.scale == 3 and m.in_channels == 1
    assert m(torch.rand(1, 3, 10, 10)).shape == (1, 3, 30, 30)


def test_unknown_architecture_errors(tmp_path):
    p = tmp_path / "junk.pth"
    torch.save({"foo.weight": torch.zeros(3)}, p)
    with pytest.raises(Exception, match="architecture"):
        load_model(str(p), "cpu", plugins={})


def test_torchscript_and_scale_probe(tmp_path, tiny_model):
    p = tmp_path / "ts.pt"
    torch.jit.trace(tiny_model, torch.rand(1, 3, 16, 16)).save(str(p))
    m = load_model(str(p), "cpu", plugins={})
    assert m.backend == "torchscript" and m.scale == 2


def test_pickled_module_requires_opt_in(tmp_path, tiny_model):
    p = tmp_path / "full.pth"
    torch.save(tiny_model, p)
    with pytest.raises(Exception, match="unsafe"):
        load_model(str(p), "cpu", plugins={})
    m = load_model(str(p), "cpu", allow_unsafe_pickle=True, plugins={})
    assert m.backend == "pickle" and m.scale == 2


def test_spandrel_compact_model(tmp_path):
    spandrel = pytest.importorskip("spandrel")
    from spandrel.architectures.Compact import Compact  # noqa: F401 - arch present

    net = Compact(num_in_ch=3, num_out_ch=3, num_feat=16, num_conv=4, upscale=2)
    p = tmp_path / "compact.pth"
    torch.save(net.state_dict(), p)
    m = load_model(str(p), "cpu", plugins={})
    assert m.backend == "spandrel" and m.scale == 2
    assert m(torch.rand(1, 3, 12, 12)).shape[-2:] == (24, 24)


@pytest.mark.parametrize("fixed", [False, True])
def test_onnx_model(tmp_path, tiny_model, fixed):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    p = tmp_path / "m.onnx"
    dyn = None if fixed else {"input": {2: "h", 3: "w"}, "output": {2: "H", 3: "W"}}
    torch.onnx.export(
        tiny_model, torch.rand(1, 3, 32, 32), str(p), input_names=["input"], output_names=["output"],
        dynamic_axes=dyn, opset_version=17, dynamo=False,
    )
    m = load_model(str(p), "cpu")
    assert m.backend == "onnx" and m.scale == 2
    assert (m.fixed_size is not None) == fixed
    out = run_tiled(m, torch.rand(1, 3, 50, 41), 0, overlap=8)
    assert out.shape[-2:] == (100, 82)


def test_torch_export_pt2(tmp_path, tiny_model):
    p = tmp_path / "m.pt2"
    h, w = torch.export.Dim("h", min=4, max=512), torch.export.Dim("w", min=4, max=512)
    ep = torch.export.export(tiny_model, (torch.rand(1, 3, 16, 16),), dynamic_shapes={"x": {2: h, 3: w}})
    torch.export.save(ep, str(p))
    m = load_model(str(p), "cpu", plugins={})
    assert m.backend == "export" and m.scale == 2
    assert m(torch.rand(1, 3, 20, 24)).shape[-2:] == (40, 48)


def test_target_size():
    o = OutputSettings(size_mode="model")
    assert target_size(o, 400, 300, 100, 75) is None
    o = OutputSettings(size_mode="scale", scale=2)
    assert target_size(o, 400, 300, 100, 75) == (200, 150)
    o = OutputSettings(size_mode="width", width=1001)
    assert target_size(o, 400, 300, 100, 75) == (1000, 750)
    o = OutputSettings(size_mode="height", height=720)
    assert target_size(o, 1920, 1080, 100, 75) == (1280, 720)


def test_quality_args():
    assert quality_args("libx264", 18, "slow") == ["-crf", "18", "-preset", "slow"]
    assert "-cq" in quality_args("hevc_nvenc", 20, "p7")
    assert quality_args("ffv1", 18, "slow") == []
