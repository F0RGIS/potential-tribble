"""User architecture plugins.

A plugin is any ``.py`` file in a plugin directory. It lets you run model
architectures that spandrel doesn't know about. A plugin module defines::

    NAME = "my-arch"                       # optional, defaults to file stem

    def detect(state_dict) -> bool:        # optional: auto-detect from weights
        return "my_arch.head.weight" in state_dict

    def build(state_dict, options) -> torch.nn.Module | dict:
        model = MyArch(**options)
        model.load_state_dict(state_dict)
        return model                       # or {"model": m, "scale": 4, ...}

``options`` is the ``"options"`` object of the model's JSON sidecar (or ``{}``).
Set ``KIND = "interpolation"`` for frame-interpolation models; their module is
called as ``model(img0, img1, t)`` and returns the in-between frame.
When ``build`` returns a dict it may contain ``model`` plus any of ``scale``,
``in_channels``, ``out_channels``, ``pad_multiple``, ``supports_half``.
Anything missing is probed automatically.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Iterable, List, Optional

log = logging.getLogger(__name__)


@dataclass
class ArchPlugin:
    name: str
    path: Path
    build: Callable
    detect: Optional[Callable] = None
    kind: str = "upscale"  # upscale | interpolation


def load_plugins(dirs: Iterable[Path]) -> Dict[str, ArchPlugin]:
    plugins: Dict[str, ArchPlugin] = {}
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                mod = _import(path)
            except Exception as exc:  # a broken plugin must not break the app
                log.warning("Failed to import plugin %s: %s", path, exc)
                continue
            build = getattr(mod, "build", None)
            if not callable(build):
                log.warning("Plugin %s has no build() function; skipped", path)
                continue
            name = str(getattr(mod, "NAME", path.stem))
            kind = str(getattr(mod, "KIND", "upscale")).lower()
            plugins[name] = ArchPlugin(name, path, build, getattr(mod, "detect", None), kind)
    return plugins


def detect_plugin(plugins: Dict[str, ArchPlugin], state_dict) -> List[ArchPlugin]:
    hits = []
    for p in plugins.values():
        if p.detect is None:
            continue
        try:
            if p.detect(state_dict):
                hits.append(p)
        except Exception as exc:
            log.debug("Plugin %s detect() raised: %s", p.name, exc)
    return hits


def _import(path: Path) -> ModuleType:
    mod_name = f"tribble_plugin_{path.stem}_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod
