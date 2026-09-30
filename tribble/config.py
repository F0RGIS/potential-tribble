"""Job configuration shared by the CLI, the GUI and preset files.

Everything a run needs is described by :class:`JobConfig`, which round-trips
to plain JSON so presets can be saved, edited by hand and shared.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelStep:
    """One model in the upscaling chain.

    ``model`` is a path to a model file or a builtin id such as
    ``builtin:bicubic@2``. ``strength`` blends the model output with a plain
    bicubic upscale of its input (1.0 = model only, 0.0 = bicubic only).
    """

    model: str
    strength: float = 1.0
    tile_size: Optional[int] = None  # overrides ProcessingSettings.tile_size
    enabled: bool = True


@dataclass
class ProcessingSettings:
    device: str = "auto"  # auto | cpu | cuda | cuda:N | mps
    half_precision: bool = True  # fp16 on GPUs that support it
    tile_size: int = 0  # 0 = whole frame; otherwise tile edge in input pixels
    tile_overlap: int = 16  # overlap between tiles, input pixels
    tile_pad_multiple: int = 8  # pad tiles to a multiple of this (some arches need it)
    bit_depth: int = 8  # 8 or 16: precision of the ffmpeg frame pipes
    queue_size: int = 8  # frames buffered between decoder/model/encoder
    allow_unsafe_pickle: bool = False  # permit loading fully pickled nn.Modules
    onnx_providers: List[str] = field(default_factory=list)  # empty = best available


@dataclass
class InputSettings:
    start_time: Optional[float] = None  # seconds
    end_time: Optional[float] = None  # seconds
    max_frames: Optional[int] = None
    pre_scale: float = 1.0  # downscale before the model (e.g. 0.5 for noisy sources)
    pre_filters: str = ""  # raw ffmpeg -vf chain applied on decode (e.g. "yadif,hqdn3d")


@dataclass
class OutputSettings:
    # Final size. "model" keeps whatever the chain produces; otherwise use one of
    # scale / width / height (aspect kept when only one of width/height is set).
    size_mode: str = "model"  # model | scale | width | height | exact
    scale: float = 2.0
    width: int = 0
    height: int = 0
    resize_filter: str = "lanczos"  # ffmpeg scale flags
    post_filters: str = ""  # raw ffmpeg -vf chain applied before encoding

    container: str = "mkv"
    codec: str = "libx264"
    crf: Optional[int] = 18
    bitrate: str = ""  # e.g. "20M"; used instead of CRF when set
    preset: str = "slow"
    pixel_format: str = "yuv420p"
    fps: Optional[float] = None  # override output frame rate
    color_matrix: str = "auto"  # auto | bt709 | bt601 | bt2020 (YUV output matrix + tags)
    copy_audio: bool = True
    audio_codec: str = "copy"  # copy | aac | libopus | flac | ...
    audio_bitrate: str = ""  # e.g. "192k" when re-encoding
    copy_subtitles: bool = True
    extra_args: str = ""  # raw args appended to the encoder command
    image_sequence: bool = False  # write PNG frames instead of a video
    overwrite: bool = False


@dataclass
class JobConfig:
    models: List[ModelStep] = field(default_factory=list)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    input: InputSettings = field(default_factory=InputSettings)
    output: OutputSettings = field(default_factory=OutputSettings)

    # ---- serialisation -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobConfig":
        data = dict(data or {})
        models = []
        for m in data.get("models", []):
            if isinstance(m, str):
                m = {"model": m}
            models.append(ModelStep(**_known(ModelStep, m)))
        return cls(
            models=models,
            processing=ProcessingSettings(**_known(ProcessingSettings, data.get("processing", {}))),
            input=InputSettings(**_known(InputSettings, data.get("input", {}))),
            output=OutputSettings(**_known(OutputSettings, data.get("output", {}))),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "JobConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _known(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keys a dataclass doesn't know so old/new presets still load."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in (data or {}).items() if k in names}
