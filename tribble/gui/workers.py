"""Background workers so the UI never blocks on models or ffmpeg."""

from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal

from ..config import JobConfig
from ..inference import Upscaler
from ..pipeline import Cancelled, Progress, run_job
from ..video import read_frame_at


class UpscalerCache:
    """Keeps the last loaded model chain so previews and jobs don't reload it."""

    def __init__(self):
        self._key: Optional[str] = None
        self._upscaler: Optional[Upscaler] = None
        self._lock = threading.Lock()

    @staticmethod
    def key_for(cfg: JobConfig) -> str:
        d = cfg.to_dict()
        return json.dumps({"models": d["models"], "processing": d["processing"]}, sort_keys=True)

    def get(self, cfg: JobConfig, log_fn) -> Upscaler:
        key = self.key_for(cfg)
        with self._lock:
            if self._upscaler is None or key != self._key:
                self._upscaler = None
                self._upscaler = Upscaler.from_config(cfg, log_fn)
                self._key = key
            return self._upscaler

    def clear(self) -> None:
        with self._lock:
            self._upscaler, self._key = None, None


class JobWorker(QObject):
    log = Signal(str)
    progress = Signal(object)  # Progress
    file_started = Signal(int)
    file_finished = Signal(int, bool, str)  # index, ok, message
    finished = Signal()

    def __init__(self, jobs: List[Tuple[str, str]], cfg: JobConfig, cache: UpscalerCache):
        super().__init__()
        self.jobs, self.cfg, self.cache = jobs, cfg, cache
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            try:
                upscaler = self.cache.get(self.cfg, self.log.emit)
            except Exception as exc:
                self.log.emit(f"ERROR loading models: {exc}")
                for i in range(len(self.jobs)):
                    self.file_finished.emit(i, False, str(exc))
                return
            for i, (src, dst) in enumerate(self.jobs):
                if self.cancel_event.is_set():
                    self.file_finished.emit(i, False, "cancelled")
                    continue
                self.file_started.emit(i)
                try:
                    run_job(src, dst, self.cfg, upscaler, self.progress.emit, self.log.emit, self.cancel_event)
                    self.file_finished.emit(i, True, dst)
                except Cancelled:
                    self.log.emit(f"Cancelled: {Path(src).name}")
                    self.file_finished.emit(i, False, "cancelled")
                except Exception as exc:
                    self.log.emit(f"ERROR {Path(src).name}: {exc}")
                    self.log.emit(traceback.format_exc(limit=3))
                    self.file_finished.emit(i, False, str(exc))
        finally:
            self.finished.emit()


class PreviewWorker(QObject):
    log = Signal(str)
    ready = Signal(object, object, float)  # before (np), after (np), seconds
    failed = Signal(str)
    finished = Signal()

    def __init__(self, path: str, t: float, cfg: JobConfig, cache: UpscalerCache):
        super().__init__()
        self.path, self.t, self.cfg, self.cache = path, t, cfg, cache

    def run(self) -> None:
        import time

        try:
            upscaler = self.cache.get(self.cfg, self.log.emit)
            before = read_frame_at(self.path, self.t, self.cfg.input, self.cfg.processing.bit_depth)
            t0 = time.perf_counter()
            after = upscaler.upscale_array(before)
            self.ready.emit(_to8(before), _to8(after), time.perf_counter() - t0)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class DownloadWorker(QObject):
    progress = Signal(int, int)
    done = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, key: str, dest: Optional[Path] = None):
        super().__init__()
        self.key, self.dest = key, dest

    def run(self) -> None:
        from ..catalog import download

        try:
            path = download(self.key, self.dest, lambda g, t: self.progress.emit(g, t))
            self.done.emit(str(path))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


def _to8(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint16:
        return (a >> 8).astype(np.uint8)
    return a
