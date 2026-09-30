"""ffmpeg-based decoding, encoding and probing."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import InputSettings, OutputSettings

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv", ".ts", ".m2ts", ".mpg", ".mpeg", ".vob", ".gif",
}

CODECS = [
    "libx264", "libx265", "libsvtav1", "libaom-av1", "libvpx-vp9", "prores_ks", "ffv1",
    "h264_nvenc", "hevc_nvenc", "av1_nvenc", "h264_qsv", "hevc_qsv", "av1_qsv",
    "h264_amf", "hevc_amf", "h264_videotoolbox", "hevc_videotoolbox",
]
PIXEL_FORMATS = ["yuv420p", "yuv420p10le", "yuv422p10le", "yuv444p", "yuv444p10le", "p010le", "nv12", "rgb24"]
CONTAINERS = ["mkv", "mp4", "mov", "webm", "avi"]

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


class FFmpegError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# binaries
# --------------------------------------------------------------------------


def find_ffmpeg() -> str:
    env = os.environ.get("TRIBBLE_FFMPEG")
    if env:
        return env
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise FFmpegError("ffmpeg not found. Install it, put it on PATH, or set TRIBBLE_FFMPEG.")


def find_ffprobe() -> Optional[str]:
    env = os.environ.get("TRIBBLE_FFPROBE")
    if env:
        return env
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    try:
        ff = Path(find_ffmpeg())
        cand = ff.with_name(ff.name.replace("ffmpeg", "ffprobe"))
        if cand.is_file() and cand != ff:
            return str(cand)
    except FFmpegError:
        pass
    return None


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------


@dataclass
class MediaInfo:
    path: str
    width: int
    height: int
    fps: float
    fps_str: str  # exact rational for ffmpeg, e.g. "24000/1001"
    duration: float
    frame_count: int
    has_audio: bool
    has_subtitles: bool
    color_space: str = ""
    pix_fmt: str = ""
    codec: str = ""

    @property
    def is_image(self) -> bool:
        return Path(self.path).suffix.lower() in IMAGE_EXTENSIONS

    def summary(self) -> str:
        if self.is_image:
            return f"{self.width}x{self.height} image"
        extra = [s for s, on in (("audio", self.has_audio), ("subs", self.has_subtitles)) if on]
        return (
            f"{self.width}x{self.height} {self.codec} {self.fps:.3f} fps, "
            f"{_fmt_time(self.duration)}, ~{self.frame_count} frames"
            + (f" (+{', '.join(extra)})" if extra else "")
        )


def _fmt_time(sec: float) -> str:
    sec = max(sec, 0)
    return f"{int(sec // 3600):d}:{int(sec % 3600 // 60):02d}:{sec % 60:05.2f}"


def _parse_rate(s: str) -> float:
    try:
        if "/" in s:
            n, d = s.split("/")
            return float(n) / float(d) if float(d) else 0.0
        return float(s)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path: str | Path) -> MediaInfo:
    path = str(path)
    if not Path(path).exists():
        raise FFmpegError(f"Input not found: {path}")
    ffprobe = find_ffprobe()
    if ffprobe:
        try:
            return _probe_ffprobe(ffprobe, path)
        except Exception as exc:
            log.debug("ffprobe failed (%s); falling back to ffmpeg -i", exc)
    return _probe_ffmpeg(path)


def _probe_ffprobe(ffprobe: str, path: str) -> MediaInfo:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
        capture_output=True, check=True, creationflags=_NO_WINDOW,
    ).stdout
    data = json.loads(out)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise FFmpegError(f"No video stream in {path}")
    fps_str = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/0"
    fps = _parse_rate(fps_str)
    if fps <= 0:
        fps_str = v.get("r_frame_rate", "25/1")
        fps = _parse_rate(fps_str) or 25.0
    duration = float(v.get("duration") or data.get("format", {}).get("duration") or 0)
    frames = int(v.get("nb_frames") or 0) or int(round(duration * fps))
    return MediaInfo(
        path=path,
        width=int(v["width"]),
        height=int(v["height"]),
        fps=fps,
        fps_str=fps_str,
        duration=duration,
        frame_count=max(frames, 1),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        has_subtitles=any(s.get("codec_type") == "subtitle" for s in streams),
        color_space=v.get("color_space", ""),
        pix_fmt=v.get("pix_fmt", ""),
        codec=v.get("codec_name", ""),
    )


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VID_RE = re.compile(r"Stream #\S+.*?Video:\s*(\w+)(.*)")
_SIZE_RE = re.compile(r"\b(\d{2,5})x(\d{2,5})\b")
_FPS_RE = re.compile(r"([\d.]+)\s*(?:fps|tbr)")


def _probe_ffmpeg(path: str) -> MediaInfo:
    proc = subprocess.run(
        [find_ffmpeg(), "-hide_banner", "-i", path], capture_output=True, text=True,
        errors="replace", creationflags=_NO_WINDOW,
    )
    text = proc.stderr
    vm = _VID_RE.search(text)
    if not vm:
        raise FFmpegError(f"Could not read video info from {path}:\n{text[-500:]}")
    rest = vm.group(2)
    sm = _SIZE_RE.search(rest)
    if not sm:
        raise FFmpegError(f"Could not read frame size from {path}")
    fm = _FPS_RE.search(rest)
    fps = float(fm.group(1)) if fm else 25.0
    fps_str = {23.98: "24000/1001", 29.97: "30000/1001", 59.94: "60000/1001"}.get(round(fps, 2), f"{fps:g}")
    dm = _DUR_RE.search(text)
    duration = int(dm.group(1)) * 3600 + int(dm.group(2)) * 60 + float(dm.group(3)) if dm else 0.0
    pix = re.search(r",\s*(yuv\w+|rgb\w+|gbr\w+|nv12|p010\w*|gray\w*|pal8|bgr\w+)", rest)
    cs = re.search(r"\((?:tv|pc)?,?\s*(bt709|bt470bg|smpte170m|bt2020nc)", rest)
    return MediaInfo(
        path=path,
        width=int(sm.group(1)),
        height=int(sm.group(2)),
        fps=fps,
        fps_str=fps_str,
        duration=duration,
        frame_count=max(int(round(duration * fps)), 1),
        has_audio="Audio:" in text,
        has_subtitles="Subtitle:" in text,
        color_space=cs.group(1) if cs else "",
        pix_fmt=pix.group(1) if pix else "",
        codec=vm.group(1),
    )


# --------------------------------------------------------------------------
# colour handling
# --------------------------------------------------------------------------

# scale filter matrix name -> (-colorspace, -color_primaries, -color_trc)
_MATRIX_TAGS = {
    "bt709": ("bt709", "bt709", "bt709"),
    "bt601": ("smpte170m", "smpte170m", "smpte170m"),
    "bt2020": ("bt2020nc", "bt2020", "bt709"),
}


def input_matrix(info: MediaInfo) -> str:
    cs = (info.color_space or "").lower()
    if cs == "bt709":
        return "bt709"
    if cs in ("smpte170m", "bt470bg", "bt601"):
        return "bt601"
    if cs.startswith("bt2020"):
        return "bt2020"
    return "bt709" if info.height >= 720 else "bt601"


def output_matrix(setting: str, height: int) -> str:
    if setting in _MATRIX_TAGS:
        return setting
    return "bt709" if height >= 720 else "bt601"


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------


class FrameReader:
    """Decodes frames as RGB numpy arrays (uint8 or uint16) through an ffmpeg pipe.

    Frames are transported as PPM so each one carries its own size, which
    keeps things robust when user pre-filters crop/pad/scale.
    """

    def __init__(self, info: MediaInfo, settings: InputSettings, bit_depth: int = 8):
        self.info = info
        self.bit_depth = 16 if bit_depth == 16 else 8
        cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin"]
        if settings.start_time:
            cmd += ["-ss", f"{settings.start_time:.6f}"]
        cmd += ["-i", info.path]
        if settings.end_time:
            dur = settings.end_time - (settings.start_time or 0)
            if dur > 0:
                cmd += ["-t", f"{dur:.6f}"]
        if settings.max_frames:
            cmd += ["-frames:v", str(int(settings.max_frames))]
        cmd += ["-map", "0:v:0", "-an", "-sn"]

        vf = []
        if settings.pre_filters.strip():
            vf.append(settings.pre_filters.strip())
        if settings.pre_scale and abs(settings.pre_scale - 1.0) > 1e-6:
            s = settings.pre_scale
            vf.append(f"scale=trunc(iw*{s}/2)*2:trunc(ih*{s}/2)*2:flags=area")
        pix = "rgb48be" if self.bit_depth == 16 else "rgb24"
        vf.append(f"scale=in_color_matrix={input_matrix(info)}:flags=bicubic+accurate_rnd+full_chroma_int")
        vf.append(f"format={pix}")
        cmd += ["-vf", ",".join(vf), "-f", "image2pipe", "-c:v", "ppm", "-"]
        self.cmd = cmd
        log.debug("decoder: %s", shlex.join(cmd))
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW, bufsize=0
        )
        self._stdout = self.proc.stdout

    def _read_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._stdout.read(n - len(buf))
            if not chunk:
                break
            buf += chunk
        return bytes(buf)

    def _read_header(self) -> Optional[Tuple[int, int, int]]:
        tokens: List[bytes] = []
        cur = b""
        while len(tokens) < 4:
            c = self._stdout.read(1)
            if not c:
                return None
            if c.isspace():
                if cur:
                    tokens.append(cur)
                    cur = b""
            else:
                cur += c
        if tokens[0] != b"P6":
            raise FFmpegError(f"Unexpected frame header from decoder: {tokens[0]!r}")
        return int(tokens[1]), int(tokens[2]), int(tokens[3])

    def read(self) -> Optional[np.ndarray]:
        hdr = self._read_header()
        if hdr is None:
            return None
        w, h, maxval = hdr
        bpc = 2 if maxval > 255 else 1
        data = self._read_exact(w * h * 3 * bpc)
        if len(data) < w * h * 3 * bpc:
            return None
        if bpc == 2:
            return np.frombuffer(data, dtype=">u2").reshape(h, w, 3).astype(np.uint16)
        return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)

    def __iter__(self):
        while True:
            f = self.read()
            if f is None:
                return
            yield f

    def close(self) -> str:
        err = b""
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            _, err = self.proc.communicate(timeout=5)
        except Exception:
            pass
        return (err or b"").decode(errors="replace")


def read_frame_at(path: str, t: float, settings: Optional[InputSettings] = None, bit_depth: int = 8) -> np.ndarray:
    """Grab a single (pre-filtered) frame at time ``t`` for previews."""
    info = probe(path)
    base = settings or InputSettings()
    s = InputSettings(
        start_time=None if info.is_image else max(t, 0.0),
        max_frames=1,
        pre_scale=base.pre_scale,
        pre_filters=base.pre_filters,
    )
    reader = FrameReader(info, s, bit_depth)
    frame = reader.read()
    err = reader.close()
    if frame is None:
        raise FFmpegError(f"Could not decode a frame at {t:.2f}s: {err.strip()[-400:]}")
    return frame


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------


def quality_args(codec: str, crf: Optional[int], preset: str) -> List[str]:
    args: List[str] = []
    c = codec.lower()
    if crf is not None:
        if c in ("libx264", "libx265", "libsvtav1"):
            args += ["-crf", str(crf)]
        elif c in ("libaom-av1", "libvpx-vp9"):
            args += ["-crf", str(crf), "-b:v", "0"]
        elif c.endswith("_nvenc"):
            args += ["-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
        elif c.endswith("_qsv"):
            args += ["-global_quality", str(crf)]
        elif c.endswith("_amf"):
            args += ["-rc", "cqp", "-qp_i", str(crf), "-qp_p", str(crf)]
        elif c.endswith("_videotoolbox"):
            args += ["-q:v", str(max(1, min(100, 100 - crf * 2)))]
    if preset:
        if c in ("libx264", "libx265") or c.endswith("_nvenc") or c.endswith("_qsv"):
            args += ["-preset", preset]
        elif c == "libsvtav1" and preset.isdigit():
            args += ["-preset", preset]
        elif c in ("libaom-av1", "libvpx-vp9") and preset.isdigit():
            args += ["-cpu-used", preset]
    if c == "prores_ks" and "-profile:v" not in args:
        args += ["-profile:v", "3"]
    return args


def target_size(
    out: OutputSettings, model_w: int, model_h: int, src_w: int, src_h: int
) -> Optional[Tuple[int, int]]:
    """Final output size, or None to keep the model's output size."""
    mode = out.size_mode
    if mode == "scale" and out.scale > 0:
        w, h = src_w * out.scale, src_h * out.scale
    elif mode == "width" and out.width > 0:
        w, h = out.width, out.width * model_h / model_w
    elif mode == "height" and out.height > 0:
        w, h = out.height * model_w / model_h, out.height
    elif mode == "exact" and out.width > 0 and out.height > 0:
        w, h = out.width, out.height
    else:
        return None
    even = lambda v: max(2, int(round(v / 2)) * 2)  # noqa: E731
    return even(w), even(h)


class FrameWriter:
    """Encodes RGB frames piped from Python, muxing audio/subs from the source."""

    def __init__(
        self,
        output: str,
        width: int,
        height: int,
        fps_str: str,
        settings: OutputSettings,
        bit_depth: int = 8,
        source: Optional[MediaInfo] = None,
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
        final_size: Optional[Tuple[int, int]] = None,
    ):
        self.width, self.height = width, height
        self.bit_depth = 16 if bit_depth == 16 else 8
        pix_in = "rgb48le" if self.bit_depth == 16 else "rgb24"
        out_path = Path(output)
        is_image = out_path.suffix.lower() in IMAGE_EXTENSIONS and not settings.image_sequence
        fw, fh = final_size or (width, height)

        cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin"]
        cmd += ["-y" if settings.overwrite else "-n"]
        cmd += ["-f", "rawvideo", "-pix_fmt", pix_in, "-s", f"{width}x{height}"]
        cmd += ["-r", fps_str, "-i", "-"]

        use_audio = bool(
            source and not is_image and not settings.image_sequence and settings.copy_audio and source.has_audio
        )
        container = out_path.suffix.lower().lstrip(".")
        sub_codec = {"mkv": "copy", "mp4": "mov_text", "mov": "mov_text", "m4v": "mov_text", "webm": "webvtt"}.get(
            container
        )
        use_subs = bool(
            source and not is_image and not settings.image_sequence and settings.copy_subtitles
            and source.has_subtitles and sub_codec
        )
        if use_audio or use_subs:
            if start_time:
                cmd += ["-ss", f"{start_time:.6f}"]
            if duration:
                cmd += ["-t", f"{duration:.6f}"]
            cmd += ["-i", source.path]
        cmd += ["-map", "0:v:0"]
        if use_audio:
            cmd += ["-map", "1:a?"]
        if use_subs:
            cmd += ["-map", "1:s?"]

        vf = []
        if final_size and final_size != (width, height):
            vf.append(f"scale={fw}:{fh}:flags={settings.resize_filter or 'lanczos'}")
        if settings.post_filters.strip():
            vf.append(settings.post_filters.strip())

        if is_image:
            cmd += ["-vf", ",".join(vf)] if vf else []
            cmd += ["-frames:v", "1", "-update", "1"]
        elif settings.image_sequence:
            cmd += ["-vf", ",".join(vf)] if vf else []
            cmd += ["-c:v", "png", "-pix_fmt", "rgb48be" if self.bit_depth == 16 else "rgb24"]
        else:
            pix_out = settings.pixel_format or "yuv420p"
            if not pix_out.startswith(("rgb", "gbr", "bgr")):
                m = output_matrix(settings.color_matrix, fh)
                if pix_out.startswith(("yuv420", "yuv422", "nv12", "p010")):
                    vf.append("crop=trunc(iw/2)*2:trunc(ih/2)*2")
                vf.append(f"scale=out_color_matrix={m}:out_range=tv:flags=bicubic+accurate_rnd+full_chroma_int")
                cs, prim, trc = _MATRIX_TAGS[m]
                cmd += ["-vf", ",".join(vf), "-colorspace", cs, "-color_primaries", prim, "-color_trc", trc,
                        "-color_range", "tv"]
            elif vf:
                cmd += ["-vf", ",".join(vf)]
            cmd += ["-c:v", settings.codec, "-pix_fmt", pix_out]
            if settings.bitrate:
                cmd += ["-b:v", settings.bitrate]
                cmd += quality_args(settings.codec, None, settings.preset)
            else:
                cmd += quality_args(settings.codec, settings.crf, settings.preset)
            if settings.fps:
                cmd += ["-r", f"{settings.fps:g}"]
            if use_audio:
                cmd += ["-c:a", settings.audio_codec or "copy"]
                if settings.audio_bitrate and settings.audio_codec != "copy":
                    cmd += ["-b:a", settings.audio_bitrate]
            if use_subs:
                cmd += ["-c:s", sub_codec]
            if container in ("mp4", "mov", "m4v"):
                cmd += ["-movflags", "+faststart"]
                if settings.codec in ("libx265", "hevc_nvenc", "hevc_qsv", "hevc_amf", "hevc_videotoolbox"):
                    cmd += ["-tag:v", "hvc1"]
            if use_audio or use_subs:
                cmd += ["-shortest"]
        if settings.extra_args.strip():
            cmd += shlex.split(settings.extra_args, posix=(os.name != "nt"))

        if settings.image_sequence:
            out_path.mkdir(parents=True, exist_ok=True)
            cmd += [str(out_path / "frame_%06d.png")]
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cmd += [str(out_path)]

        self.cmd = cmd
        log.debug("encoder: %s", shlex.join(cmd))
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self.height, self.width):
            raise FFmpegError(
                f"Frame size changed mid-stream: {frame.shape[1]}x{frame.shape[0]} vs {self.width}x{self.height}"
            )
        if self.bit_depth == 16:
            frame = frame.astype("<u2", copy=False)
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise FFmpegError(f"Encoder exited early: {self._stderr()}") from exc

    def _stderr(self) -> str:
        try:
            self.proc.wait(timeout=10)
            return self.proc.stderr.read().decode(errors="replace").strip()[-2000:]
        except Exception:
            return "(no output)"

    def close(self, abort: bool = False) -> None:
        if abort:
            self.proc.kill()
            self.proc.wait()
            return
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        err = self.proc.stderr.read().decode(errors="replace")
        code = self.proc.wait()
        if code != 0:
            raise FFmpegError(f"ffmpeg encoder failed (exit {code}):\n{err.strip()[-2000:]}")
