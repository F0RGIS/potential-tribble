"""A few well-known, freely downloadable models for a quick start.

Any other model (OpenModelDB, your own training runs, ...) works by simply
dropping the file into a models folder.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

from .models.registry import USER_DIR

_RE = "https://github.com/xinntao/Real-ESRGAN/releases/download"


@dataclass(frozen=True)
class CatalogModel:
    key: str
    filename: str
    url: str
    scale: int
    description: str


CATALOG: Dict[str, CatalogModel] = {
    m.key: m
    for m in [
        CatalogModel("realesrgan-x4plus", "RealESRGAN_x4plus.pth", f"{_RE}/v0.1.0/RealESRGAN_x4plus.pth", 4,
                     "Real-ESRGAN general photo/real-world x4 (RRDB, heavy, BSD-3)"),
        CatalogModel("realesrgan-x2plus", "RealESRGAN_x2plus.pth", f"{_RE}/v0.2.1/RealESRGAN_x2plus.pth", 2,
                     "Real-ESRGAN general x2 (RRDB, BSD-3)"),
        CatalogModel("realesrgan-x4plus-anime", "RealESRGAN_x4plus_anime_6B.pth",
                     f"{_RE}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth", 4,
                     "Real-ESRGAN anime illustrations x4 (6-block RRDB, BSD-3)"),
        CatalogModel("realesr-animevideov3", "realesr-animevideov3.pth", f"{_RE}/v0.2.5.0/realesr-animevideov3.pth",
                     4, "Real-ESRGAN anime video x4 (tiny SRVGG, very fast, BSD-3)"),
        CatalogModel("realesr-general-x4v3", "realesr-general-x4v3.pth", f"{_RE}/v0.2.5.0/realesr-general-x4v3.pth",
                     4, "Real-ESRGAN general x4 v3 (compact SRVGG, fast, BSD-3)"),
        CatalogModel("realesr-general-wdn-x4v3", "realesr-general-wdn-x4v3.pth",
                     f"{_RE}/v0.2.5.0/realesr-general-wdn-x4v3.pth", 4,
                     "Real-ESRGAN general x4 v3, stronger denoise (compact SRVGG, BSD-3)"),
    ]
}


def download(
    key: str,
    dest_dir: Optional[Path] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    overwrite: bool = False,
) -> Path:
    if key not in CATALOG:
        raise KeyError(f"Unknown model {key!r}. Available: {', '.join(CATALOG)}")
    m = CATALOG[key]
    dest_dir = Path(dest_dir) if dest_dir else USER_DIR / "models"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / m.filename
    if dest.exists() and not overwrite:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(m.url, headers={"User-Agent": "tribble-upscaler"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if progress:
                progress(got, total)
    tmp.replace(dest)
    return dest
