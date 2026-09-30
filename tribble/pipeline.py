"""End-to-end job execution: decode -> upscale -> encode, with threads.

Decoding and encoding run in background threads (ffmpeg does the heavy
lifting in subprocesses) while the model runs on the calling thread, so the
GPU stays busy and I/O overlaps with inference.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .config import JobConfig
from .inference import Upscaler
from .video import IMAGE_EXTENSIONS, FFmpegError, FrameReader, FrameWriter, MediaInfo, probe, target_size

log = logging.getLogger(__name__)

_END = object()


class Cancelled(Exception):
    pass


@dataclass
class Progress:
    frame: int
    total: int
    fps: float
    elapsed: float
    eta: float

    @property
    def fraction(self) -> float:
        return min(self.frame / self.total, 1.0) if self.total else 0.0


ProgressFn = Callable[[Progress], None]
LogFn = Callable[[str], None]


def default_output_path(input_path: str | Path, cfg: JobConfig, out_dir: Optional[str] = None) -> Path:
    src = Path(input_path)
    folder = Path(out_dir) if out_dir else src.parent
    if src.suffix.lower() in IMAGE_EXTENSIONS:
        return folder / f"{src.stem}_upscaled.png"
    if cfg.output.image_sequence:
        return folder / f"{src.stem}_upscaled_frames"
    return folder / f"{src.stem}_upscaled.{cfg.output.container.lstrip('.')}"


def expected_frames(info: MediaInfo, cfg: JobConfig) -> int:
    if info.is_image:
        return 1
    total = info.frame_count
    start = cfg.input.start_time or 0.0
    if cfg.input.end_time or start:
        end = cfg.input.end_time or info.duration
        if end > start:
            total = max(int(round((end - start) * info.fps)), 1)
    if cfg.input.max_frames:
        total = min(total, int(cfg.input.max_frames))
    return total


def run_job(
    input_path: str | Path,
    output_path: str | Path,
    cfg: JobConfig,
    upscaler: Optional[Upscaler] = None,
    progress: Optional[ProgressFn] = None,
    log_fn: LogFn = log.info,
    cancel: Optional[threading.Event] = None,
) -> Path:
    cancel = cancel or threading.Event()
    input_path, output_path = str(input_path), Path(output_path)
    if output_path.exists() and not cfg.output.overwrite and not cfg.output.image_sequence:
        raise FileExistsError(f"Output exists (enable overwrite): {output_path}")
    if output_path.resolve() == Path(input_path).resolve():
        raise ValueError("Output path must differ from the input path")

    info = probe(input_path)
    log_fn(f"Input: {Path(input_path).name} - {info.summary()}")
    if upscaler is None:
        upscaler = Upscaler.from_config(cfg, log_fn)

    bit_depth = 16 if cfg.processing.bit_depth == 16 else 8
    total = expected_frames(info, cfg)
    reader = FrameReader(info, cfg.input, bit_depth)

    in_q: "queue.Queue" = queue.Queue(maxsize=max(cfg.processing.queue_size, 1))
    out_q: "queue.Queue" = queue.Queue(maxsize=max(cfg.processing.queue_size, 1))
    errors: list = []

    def decode():
        try:
            for frame in reader:
                while not cancel.is_set():
                    try:
                        in_q.put(frame, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if cancel.is_set():
                    return
        except Exception as exc:  # surfaced on the main thread
            errors.append(exc)
        finally:
            _put_end(in_q, cancel)

    writer: Optional[FrameWriter] = None

    def encode():
        try:
            while True:
                item = out_q.get()
                if item is _END:
                    return
                writer.write(item)
        except Exception as exc:
            errors.append(exc)
            cancel.set()

    dec_thread = threading.Thread(target=decode, name="tribble-decode", daemon=True)
    dec_thread.start()
    enc_thread: Optional[threading.Thread] = None

    t0 = time.perf_counter()
    done = 0
    ok = False
    try:
        while True:
            if cancel.is_set():
                break
            try:
                frame = in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame is _END:
                break
            out = upscaler.upscale_array(frame)

            if writer is None:
                oh, ow = out.shape[:2]
                final = target_size(cfg.output, ow, oh, info.width, info.height)
                log_fn(
                    f"Model output {frame.shape[1]}x{frame.shape[0]} -> {ow}x{oh}"
                    + (f", resized to {final[0]}x{final[1]}" if final else "")
                )
                duration = None
                if cfg.input.end_time:
                    duration = cfg.input.end_time - (cfg.input.start_time or 0)
                elif cfg.input.max_frames:
                    duration = cfg.input.max_frames / info.fps
                writer = FrameWriter(
                    str(output_path), ow, oh, info.fps_str, cfg.output, bit_depth,
                    source=info, start_time=cfg.input.start_time, duration=duration, final_size=final,
                )
                enc_thread = threading.Thread(target=encode, name="tribble-encode", daemon=True)
                enc_thread.start()

            try:
                _put(out_q, out, cancel)
            except Cancelled:
                break
            done += 1
            if progress:
                el = time.perf_counter() - t0
                fps = done / el if el > 0 else 0.0
                total = max(total, done)
                progress(Progress(done, total, fps, el, (total - done) / fps if fps else 0.0))

        if errors:
            raise errors[0]
        if cancel.is_set():
            raise Cancelled()
        if writer is None:
            raise FFmpegError(f"No frames decoded from {input_path}: {reader.close().strip()[-500:]}")

        _put(out_q, _END, cancel, force=True)
        enc_thread.join()
        if errors:
            raise errors[0]
        writer.close()
        ok = True
        el = time.perf_counter() - t0
        log_fn(f"Done: {done} frames in {el:.1f}s ({done / el if el else 0:.2f} fps) -> {output_path}")
        return output_path
    finally:
        if not ok:
            cancel.set()
        reader.close()
        if writer is not None and not ok:
            _drain(out_q)
            writer.close(abort=True)
            if enc_thread is not None:
                out_q.put(_END)
                enc_thread.join(timeout=5)
        dec_thread.join(timeout=5)


def _put(q: "queue.Queue", item, cancel: threading.Event, force: bool = False) -> None:
    while True:
        if cancel.is_set() and not force:
            raise Cancelled()
        try:
            q.put(item, timeout=0.2)
            return
        except queue.Full:
            continue


def _put_end(q: "queue.Queue", cancel: threading.Event) -> None:
    while True:
        try:
            q.put(_END, timeout=0.2)
            return
        except queue.Full:
            if cancel.is_set():
                _drain(q)


def _drain(q: "queue.Queue") -> None:
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass
