"""Model / plugin folder discovery and JSON sidecars."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

MODEL_EXTENSIONS = {".pth", ".pt", ".ckpt", ".bin", ".safetensors", ".onnx", ".jit", ".torchscript", ".pt2"}

BUILTIN_MODELS = [
    "builtin:bicubic@2",
    "builtin:bicubic@4",
    "builtin:bilinear@2",
    "builtin:nearest@2",
    "builtin:identity",
]

USER_DIR = Path(os.environ.get("TRIBBLE_HOME", Path.home() / ".tribble"))


def _env_dirs(var: str) -> List[Path]:
    return [Path(p) for p in os.environ.get(var, "").split(os.pathsep) if p]


def model_dirs(extra: Optional[List[str]] = None) -> List[Path]:
    dirs = [Path(p) for p in (extra or [])] + _env_dirs("TRIBBLE_MODELS")
    dirs += [Path.cwd() / "models", USER_DIR / "models"]
    return _dedupe(dirs)


def plugin_dirs(extra: Optional[List[str]] = None) -> List[Path]:
    dirs = [Path(p) for p in (extra or [])] + _env_dirs("TRIBBLE_PLUGINS")
    dirs += [Path.cwd() / "plugins", USER_DIR / "plugins"]
    return _dedupe(dirs)


def preset_dirs() -> List[Path]:
    return _dedupe([Path.cwd() / "presets", USER_DIR / "presets"])


def _dedupe(dirs: List[Path]) -> List[Path]:
    seen, out = set(), []
    for d in dirs:
        key = str(d.expanduser().resolve())
        if key not in seen:
            seen.add(key)
            out.append(d.expanduser())
    return out


@dataclass
class ModelEntry:
    path: Path
    name: str
    size_bytes: int
    sidecar: Dict = field(default_factory=dict)

    @property
    def spec(self) -> str:
        return str(self.path)


def find_sidecar(path: Path) -> Optional[Path]:
    """``foo.pth.json`` takes precedence over ``foo.json``."""
    for cand in (path.with_name(path.name + ".json"), path.with_suffix(".json")):
        if cand.is_file() and cand != path:
            return cand
    return None


def read_sidecar(path: Path) -> Dict:
    sc = find_sidecar(path)
    if sc is None:
        return {}
    try:
        data = json.loads(sc.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def scan_models(dirs: Optional[List[Path]] = None) -> List[ModelEntry]:
    entries: List[ModelEntry] = []
    for d in dirs if dirs is not None else model_dirs():
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in MODEL_EXTENSIONS:
                sc = read_sidecar(p)
                entries.append(ModelEntry(p, sc.get("name") or p.stem, p.stat().st_size, sc))
    return entries
