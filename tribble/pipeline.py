"""End-to-end job execution: decode -> upscale -> encode, with threads.

Decoding and encoding run in background threads (ffmpeg does the heavy
lifting in subprocesses) while the model runs on the calling thread, so the
GPU stays busy and I/O overlaps with inference.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

from .config import JobConfig
from .inference import Upscaler
from .interp import (
    FFMPEG_METHOD,
    FrameRateConverter,
    InterpModel,
    fraction_str,
    load_interpolator,
    minterpolate_filter,
    output_rate,
    rate_fraction,
)
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


@dataclass
class RatePlan:
    """How (and whether) the frame rate changes for a job."""

    src_rate: Fraction
    out_rate: Fraction
    converter: Optional[FrameRateConverter] = None  # AI / blend interpolation in Python
    ffmpeg_filter: str = ""  # minterpolate filter string
    order: str = "after"

    @property
    def ratio(self) -> Fraction:
        return self.out_rate / self.src_rate

    @property
    def active(self) -> bool:
        return self.converter is not None or bool(self.ffmpeg_filter)


def load_interpolator_for(cfg: JobConfig, log_fn: LogFn = log.info) -> Optional[InterpModel]:
    it = cfg.interpolation
    if not cfg.interpolating or it.model == FFMPEG_METHOD:
        return None
    p = cfg.processing
    model = load_interpolator(
        it.model, device=p.device, half=p.half_precision, flow_scale=it.flow_scale,
        ensemble=it.ensemble, allow_unsafe_pickle=p.allow_unsafe_pickle,
    )
    log_fn(f"Loaded interpolation model: {model.describe()}")
    return model


def plan_rate(info: MediaInfo, cfg: JobConfig, interpolator: Optional[InterpModel]) -> RatePlan:
    src = rate_fraction(info.fps_str)
    it = cfg.interpolation
    if not cfg.interpolating or info.is_image:
        return RatePlan(src, src)
    out = output_rate(src, it.mode, it.factor, it.target_fps)
    order = it.order if it.order in ("before", "after") else "after"
    if it.model == FFMPEG_METHOD:
        return RatePlan(src, out, ffmpeg_filter=minterpolate_filter(out), order=order)
    if interpolator is None:
        raise ValueError("No interpolation model loaded")
    conv = FrameRateConverter(interpolator, out / src, it.scene_threshold)
    return RatePlan(src, out, converter=conv, order=order)


def run_job(
    input_path: str | Path,
    output_path: str | Path,
    cfg: JobConfig,
    upscaler: Optional[Upscaler] = None,
    progress: Optional[ProgressFn] = None,
    log_fn: LogFn = log.info,
    cancel: Optional[threading.Event] = None,
    interpolator: Optional[InterpModel] = None,
) -> Path:
    cancel = cancel or threading.Event()
    input_path, output_path = str(input_path), Path(output_path)
    if output_path.exists() and not cfg.output.overwrite and not cfg.output.image_sequence:
        raise FileExistsError(f"Output exists (enable overwrite): {output_path}")
    if output_path.resolve() == Path(input_path).resolve():
        raise ValueError("Output path must differ from the input path")
    if not cfg.has_work():
        raise ValueError("Nothing to do: enable an upscaling model or frame interpolation")

    info = probe(input_path)
    log_fn(f"Input: {Path(input_path).name} - {info.summary()}")
    if upscaler is None:
        upscaler = Upscaler.from_config(cfg, log_fn)
    if interpolator is None and cfg.interpolating and not info.is_image:
        interpolator = load_interpolator_for(cfg, log_fn)
    plan = plan_rate(info, cfg, interpolator)
    if plan.active:
        how = "ffmpeg minterpolate" if plan.ffmpeg_filter else plan.converter.model.name
        log_fn(
            f"Interpolating {float(plan.src_rate):.3f} -> {float(plan.out_rate):.3f} fps "
            f"({how}, {plan.order} upscaling)"
        )

    bit_depth = 16 if cfg.processing.bit_depth == 16 else 8
    total = expected_frames(info, cfg)
    if plan.converter is not None or (plan.ffmpeg_filter and plan.order == "before"):
        total = math.ceil(total * plan.ratio)
    in_settings = cfg.input
    if plan.ffmpeg_filter and plan.order == "before":
        in_settings = replace(cfg.input, pre_filters=",".join(
            f for f in (cfg.input.pre_filters.strip(), plan.ffmpeg_filter) if f
        ))
    # Rate of the frames we hand to the encoder.
    pipe_rate = plan.out_rate if (plan.converter or plan.order == "before") else plan.src_rate
    reader = FrameReader(info, in_settings, bit_depth)

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

    @torch.inference_mode()
    def process(frame: Optional[np.ndarray]) -> List[np.ndarray]:
        """One decoded frame in (None = end of stream), zero or more output frames out."""
        conv = plan.converter
        if conv is None:
            return [] if frame is None else [upscaler.to_numpy(upscaler.upscale(upscaler.to_tensor(frame)), bit_depth)]
        if plan.order == "before":
            outs = conv.flush() if frame is None else conv.push(upscaler.to_tensor(frame))
            return [upscaler.to_numpy(upscaler.upscale(o), bit_depth) for o in outs]
        outs = conv.flush() if frame is None else conv.push(upscaler.upscale(upscaler.to_tensor(frame)))
        return [upscaler.to_numpy(o, bit_depth) for o in outs]

    dec_thread = threading.Thread(target=decode, name="tribble-decode", daemon=True)
    dec_thread.start()
    enc_thread: Optional[threading.Thread] = None

    t0 = time.perf_counter()
    done = 0
    ok = False
    src_shape = None
    try:
        finished = False
        while not finished:
            if cancel.is_set():
                break
            try:
                frame = in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame is _END:
                frame, finished = None, True
            elif src_shape is None:
                src_shape = frame.shape
            outs = process(frame)

            for out in outs:
                if writer is None:
                    oh, ow = out.shape[:2]
                    final = target_size(cfg.output, ow, oh, info.width, info.height)
                    log_fn(
                        f"Model output {src_shape[1]}x{src_shape[0]} -> {ow}x{oh}"
                        + (f", resized to {final[0]}x{final[1]}" if final else "")
                    )
                    duration = None
                    if cfg.input.end_time:
                        duration = cfg.input.end_time - (cfg.input.start_time or 0)
                    elif cfg.input.max_frames:
                        duration = cfg.input.max_frames / info.fps
                    writer = FrameWriter(
                        str(output_path), ow, oh, fraction_str(pipe_rate), cfg.output, bit_depth,
                        source=info, start_time=cfg.input.start_time, duration=duration, final_size=final,
                        pre_filters=plan.ffmpeg_filter if plan.order == "after" else "",
                    )
                    enc_thread = threading.Thread(target=encode, name="tribble-encode", daemon=True)
                    enc_thread.start()
                try:
                    _put(out_q, out, cancel)
                except Cancelled:
                    finished = True
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
        extra = ""
        if plan.converter is not None and plan.converter.scene_cuts:
            extra = f", {plan.converter.scene_cuts} scene cut(s) kept sharp"
        log_fn(f"Done: {done} frames in {el:.1f}s ({done / el if el else 0:.2f} fps){extra} -> {output_path}")
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
