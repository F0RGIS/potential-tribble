"""Load any supported model file into a :class:`LoadedModel`.

Resolution order for weight files (``.pth``/``.pt``/``.ckpt``/``.safetensors``):

1. sidecar ``"arch"`` naming a user plugin -> that plugin
2. spandrel auto-detection (ESRGAN, Real-ESRGAN, SwinIR, HAT, DAT, OmniSR,
   SPAN, Compact/SRVGG, RealCUGAN, ... ~40 architectures)
3. user plugins whose ``detect()`` accepts the state dict
4. TorchScript archive
5. fully pickled ``nn.Module`` (only with ``allow_unsafe_pickle``)

``.onnx`` files go to ONNX Runtime and ``builtin:*`` specs are simple
interpolation "models" useful for testing and for chaining.

Sidecar JSON (``model.pth.json`` or ``model.json``) keys, all optional::

    {
      "name": "My model",
      "arch": "plugin-name",       # force a plugin
      "options": {...},            # passed to the plugin's build()
      "scale": 4,                  # skip scale probing
      "half": false,               # never run in fp16
      "channel_order": "bgr",      # model expects BGR input
      "pad_multiple": 16,
      "tile_size": 256,            # preferred tile size
      "fixed_size": [256, 256],    # model only accepts this input size
      "in_channels": 3, "out_channels": 3
    }
"""

from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .base import LoadedModel
from .plugins import ArchPlugin, detect_plugin, load_plugins
from .registry import plugin_dirs, read_sidecar

log = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    pass


def resolve_device(name: str = "auto") -> torch.device:
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def available_devices() -> list:
    devs = ["auto", "cpu"]
    if torch.cuda.is_available():
        devs += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        devs.append("mps")
    return devs


def load_model(
    spec: str,
    device: str | torch.device = "auto",
    half: bool = False,
    allow_unsafe_pickle: bool = False,
    onnx_providers: Optional[list] = None,
    plugins: Optional[Dict[str, ArchPlugin]] = None,
) -> LoadedModel:
    dev = device if isinstance(device, torch.device) else resolve_device(device)
    # fp16 on CPU is slow or unsupported for most ops.
    use_half = bool(half) and dev.type in ("cuda", "mps")

    if spec.startswith("builtin:"):
        return _load_builtin(spec, dev)

    path = Path(spec).expanduser()
    if not path.is_file():
        raise ModelLoadError(f"Model file not found: {spec}")
    sidecar = read_sidecar(path)
    if sidecar.get("half") is False:
        use_half = False

    if path.suffix.lower() == ".onnx":
        model = _load_onnx(path, dev, onnx_providers)
    else:
        if plugins is None:
            plugins = load_plugins(plugin_dirs())
        model = _load_torch(path, dev, use_half, allow_unsafe_pickle, sidecar, plugins)

    _apply_sidecar(model, sidecar, path)
    if not model.scale:
        model.scale = _probe_scale(model)
    return model


# --------------------------------------------------------------------------
# builtin
# --------------------------------------------------------------------------

_BUILTIN_RE = re.compile(r"^builtin:(identity|nearest|bilinear|bicubic)(?:@(\d+(?:\.\d+)?))?$")


def _load_builtin(spec: str, dev: torch.device) -> LoadedModel:
    m = _BUILTIN_RE.match(spec)
    if not m:
        raise ModelLoadError(
            f"Unknown builtin model {spec!r}. Use builtin:identity or "
            "builtin:<nearest|bilinear|bicubic>@<scale>"
        )
    mode, scale = m.group(1), float(m.group(2) or 1)
    if mode == "identity":
        scale = 1.0

    def fwd(x: torch.Tensor) -> torch.Tensor:
        if scale == 1:
            return x
        kw = {} if mode == "nearest" else {"align_corners": False}
        return F.interpolate(x, scale_factor=scale, mode=mode, **kw).clamp(0, 1)

    return LoadedModel(spec, fwd, scale, "builtin", dev, torch.float32, arch=mode)


# --------------------------------------------------------------------------
# torch weights
# --------------------------------------------------------------------------


def _read_state_dict(path: Path, allow_unsafe: bool):
    """Returns a state dict, a TorchScript module, or a pickled nn.Module."""
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ModelLoadError("Install 'safetensors' to load .safetensors models") from exc
        return load_file(str(path), device="cpu")

    if suffix in (".jit", ".torchscript"):
        return torch.jit.load(str(path), map_location="cpu")

    if suffix == ".pt2":  # torch.export archive
        return torch.export.load(str(path)).module()

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            if any(n.endswith("constants.pkl") for n in zf.namelist()):  # TorchScript archive
                return torch.jit.load(str(path), map_location="cpu")

    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as first_err:
        try:
            return torch.jit.load(str(path), map_location="cpu")
        except Exception:
            pass
        if allow_unsafe:
            log.warning("Loading %s with full pickle (unsafe mode enabled)", path.name)
            return torch.load(str(path), map_location="cpu", weights_only=False)
        raise ModelLoadError(
            f"{path.name} is not a plain weights file ({first_err}). If you trust it, "
            "enable 'allow unsafe pickle' to load a fully pickled model."
        ) from first_err


def _unwrap_state_dict(obj):
    """Common checkpoint layouts: {"params_ema": sd}, {"state_dict": sd}, ..."""
    if not isinstance(obj, dict):
        return obj
    for key in ("params_ema", "params", "state_dict", "model", "net_g", "generator"):
        inner = obj.get(key)
        if isinstance(inner, dict) and inner and all(isinstance(k, str) for k in inner):
            if any(isinstance(v, torch.Tensor) for v in inner.values()):
                return _strip_prefix(inner)
    return _strip_prefix(obj)


def _strip_prefix(sd: dict) -> dict:
    for prefix in ("module.", "model."):
        if sd and all(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):]: v for k, v in sd.items()}
    return sd


def _load_torch(path, dev, half, allow_unsafe, sidecar, plugins) -> LoadedModel:
    obj = _read_state_dict(path, allow_unsafe)
    name = sidecar.get("name") or path.stem

    if isinstance(obj, torch.jit.ScriptModule):
        return _wrap_module(obj, name, "torchscript", dev, half)
    if path.suffix.lower() == ".pt2" and isinstance(obj, torch.nn.Module):
        return _wrap_module(obj, name, "export", dev, False)
    if isinstance(obj, torch.nn.Module):
        return _wrap_module(obj, name, "pickle", dev, half)
    if not isinstance(obj, dict):
        raise ModelLoadError(f"Unrecognised checkpoint contents in {path.name}: {type(obj).__name__}")

    # 1. explicit plugin from sidecar
    forced = sidecar.get("arch")
    if forced:
        if forced not in plugins:
            raise ModelLoadError(f"Sidecar asks for arch plugin {forced!r}, which isn't installed")
        return _build_plugin(plugins[forced], _unwrap_state_dict(obj), sidecar, name, dev, half)

    # 2. spandrel
    spandrel_err = None
    try:
        return _load_spandrel(obj, name, dev, half)
    except ImportError as exc:
        spandrel_err = exc
    except Exception as exc:
        spandrel_err = exc
        log.debug("spandrel could not load %s: %s", path.name, exc)

    # 3. plugin detection
    sd = _unwrap_state_dict(obj)
    for plugin in detect_plugin(plugins, sd):
        try:
            return _build_plugin(plugin, sd, sidecar, name, dev, half)
        except Exception as exc:
            log.warning("Plugin %s matched %s but failed to build: %s", plugin.name, path.name, exc)

    raise ModelLoadError(
        f"Could not identify the architecture of {path.name}: {spandrel_err}. "
        "Write an arch plugin (see plugins/README.md) or add a sidecar JSON with \"arch\"."
    )


_spandrel_extras_installed = False


def _load_spandrel(state_dict, name, dev, half) -> LoadedModel:
    global _spandrel_extras_installed
    import spandrel

    if not _spandrel_extras_installed:
        _spandrel_extras_installed = True
        try:  # optional: extra (non-commercial licensed) architectures
            import spandrel_extra_arches

            spandrel_extra_arches.install()
        except Exception:
            pass

    desc = spandrel.ModelLoader(device="cpu").load_from_state_dict(state_dict)
    if not isinstance(desc, spandrel.ImageModelDescriptor):
        raise ModelLoadError(f"{name} is a {desc.purpose} model, not a single-image model")

    use_half = half and getattr(desc, "supports_half", False)
    desc.to(dev)
    desc.eval()
    dtype = torch.float16 if use_half else torch.float32
    if use_half:
        desc.half()

    req = desc.size_requirements
    pad_multiple = max(int(getattr(req, "multiple_of", 1) or 1), 1)

    @torch.inference_mode()
    def fwd(x):
        return desc.model(x)

    return LoadedModel(
        name=name,
        forward=fwd,
        scale=float(desc.scale),
        backend="spandrel",
        device=dev,
        dtype=dtype,
        in_channels=desc.input_channels,
        out_channels=desc.output_channels,
        arch=desc.architecture.name,
        pad_multiple=pad_multiple,
        info={"tags": list(getattr(desc, "tags", []))},
    )


def _wrap_module(module, name, backend, dev, half) -> LoadedModel:
    module = module.to(dev)
    try:
        module.eval()
    except NotImplementedError:  # torch.export modules are already in inference form
        pass
    dtype = torch.float32
    if half:
        try:
            module = module.half()
            dtype = torch.float16
        except Exception:
            pass

    @torch.inference_mode()
    def fwd(x):
        out = module(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    return LoadedModel(name, fwd, 0.0, backend, dev, dtype, arch=type(module).__name__)


def _build_plugin(plugin: ArchPlugin, sd, sidecar, name, dev, half) -> LoadedModel:
    result = plugin.build(sd, dict(sidecar.get("options", {})))
    meta = {}
    if isinstance(result, dict):
        meta = result
        result = result["model"]
    if meta.get("supports_half") is False:
        half = False
    lm = _wrap_module(result, name, "plugin", dev, half)
    lm.arch = plugin.name
    lm.scale = float(meta.get("scale", 0) or 0)
    lm.in_channels = int(meta.get("in_channels", 3))
    lm.out_channels = int(meta.get("out_channels", 3))
    lm.pad_multiple = int(meta.get("pad_multiple", 1))
    return lm


# --------------------------------------------------------------------------
# ONNX
# --------------------------------------------------------------------------


def _load_onnx(path: Path, dev: torch.device, providers: Optional[list]) -> LoadedModel:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise ModelLoadError("Install 'onnxruntime' (or onnxruntime-gpu) to use .onnx models") from exc

    available = ort.get_available_providers()
    if providers:
        chosen = [p for p in providers if p in available] or ["CPUExecutionProvider"]
    else:
        preferred = ["CPUExecutionProvider"]
        if dev.type == "cuda":
            preferred = ["TensorrtExecutionProvider", "CUDAExecutionProvider"] + preferred
        preferred = ["DmlExecutionProvider", "CoreMLExecutionProvider"] + preferred
        chosen = [p for p in preferred if p in available]
        if dev.type == "cuda":
            chosen = [p for p in chosen if p not in ("DmlExecutionProvider", "CoreMLExecutionProvider")]

    sess = ort.InferenceSession(str(path), providers=chosen)
    inp = sess.get_inputs()[0]
    out_name = sess.get_outputs()[0].name
    np_dtype = np.float16 if "float16" in inp.type else np.float32

    shape = list(inp.shape)  # e.g. [1, 3, 'h', 'w'] or [1, 3, 256, 256]
    in_ch = shape[1] if len(shape) == 4 and isinstance(shape[1], int) else 3
    fixed = None
    if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
        fixed = (shape[2], shape[3])
    fixed_batch = len(shape) == 4 and isinstance(shape[0], int)

    def fwd(x: torch.Tensor) -> torch.Tensor:
        arr = x.detach().float().cpu().numpy().astype(np_dtype)
        if fixed_batch and arr.shape[0] != shape[0]:
            outs = [sess.run([out_name], {inp.name: arr[i : i + 1]})[0] for i in range(arr.shape[0])]
            y = np.concatenate(outs, 0)
        else:
            y = sess.run([out_name], {inp.name: arr})[0]
        return torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

    # ONNX runs on its own provider; keep tensors on CPU around it.
    return LoadedModel(
        name=path.stem,
        forward=fwd,
        scale=0.0,
        backend="onnx",
        device=torch.device("cpu"),
        dtype=torch.float32,
        in_channels=in_ch,
        arch=",".join(p.replace("ExecutionProvider", "") for p in sess.get_providers()),
        fixed_size=fixed,
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _apply_sidecar(model: LoadedModel, sc: dict, path: Path) -> None:
    if not sc:
        return
    if sc.get("name"):
        model.name = str(sc["name"])
    if sc.get("scale"):
        model.scale = float(sc["scale"])
    if sc.get("channel_order") in ("rgb", "bgr"):
        model.channel_order = sc["channel_order"]
    if sc.get("pad_multiple"):
        model.pad_multiple = int(sc["pad_multiple"])
    if sc.get("tile_size"):
        model.tile_size = int(sc["tile_size"])
    if sc.get("fixed_size"):
        h, w = sc["fixed_size"]
        model.fixed_size = (int(h), int(w))
    if sc.get("in_channels"):
        model.in_channels = int(sc["in_channels"])
    if sc.get("out_channels"):
        model.out_channels = int(sc["out_channels"])
    model.info["sidecar"] = str(path)


def _probe_scale(model: LoadedModel) -> float:
    h, w = model.fixed_size or (32, 32)
    if model.pad_multiple > 1:
        m = model.pad_multiple
        h, w = -(-h // m) * m, -(-w // m) * m
    x = torch.rand(1, 3, h, w)
    with torch.inference_mode():
        y = model(x)
    scale = y.shape[-1] / w
    if abs(scale - round(scale)) < 1e-3:
        scale = float(round(scale))
    log.info("Probed scale of %s: x%g", model.name, scale)
    return scale
