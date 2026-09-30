"""Tribble GUI (PySide6)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QSettings, Qt, QThread, QTimer, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QKeySequence, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..catalog import CATALOG
from ..config import InputSettings, JobConfig, OutputSettings, ProcessingSettings
from ..models import available_devices, model_dirs, scan_models
from ..models.registry import USER_DIR, plugin_dirs, preset_dirs
from ..pipeline import default_output_path
from ..video import CODECS, CONTAINERS, IMAGE_EXTENSIONS, PIXEL_FORMATS, VIDEO_EXTENSIONS, probe
from .widgets import CompareView, FileQueue, ModelChainTable
from .workers import DownloadWorker, JobWorker, PreviewWorker, UpscalerCache

PRE_FILTER_EXAMPLES = [
    "",
    "bwdif",  # deinterlace (good quality)
    "yadif",  # deinterlace (fast)
    "hqdn3d=2:1:3:3",  # light temporal denoise
    "nlmeans=s=2",  # strong spatial denoise (slow)
    "deblock",  # reduce block artifacts
    "crop=iw-16:ih-16",  # trim borders
    "fps=24000/1001",  # change frame rate before upscaling
]
POST_FILTER_EXAMPLES = [
    "",
    "cas=0.4",  # contrast-adaptive sharpening
    "unsharp=5:5:0.6",
    "noise=alls=3:allf=t",  # add film grain back
    "eq=saturation=1.08:contrast=1.03",
    "deband",
]
RESIZE_FILTERS = ["lanczos", "spline", "bicubic", "bilinear", "area", "neighbor"]
AUDIO_CODECS = ["copy", "aac", "libopus", "flac", "ac3", "pcm_s16le"]
ENC_PRESETS = ["", "veryslow", "slower", "slow", "medium", "fast", "p7", "p5", "p4", "4", "6", "8"]
SIZE_MODES = [
    ("model", "Model scale (native)"),
    ("scale", "Scale factor vs source"),
    ("width", "Fixed width"),
    ("height", "Fixed height"),
    ("exact", "Exact width x height"),
]
MEDIA_FILTER = "Media ({});;All files (*)".format(
    " ".join(f"*{e}" for e in sorted(VIDEO_EXTENSIONS | IMAGE_EXTENSIONS))
)


def _combo(items, editable=False, current: Optional[str] = None) -> QComboBox:
    cb = QComboBox()
    cb.setEditable(editable)
    for it in items:
        if isinstance(it, tuple):
            cb.addItem(it[1], it[0])
        else:
            cb.addItem(it, it)
    if current is not None:
        _set_combo(cb, current)
    return cb


def _set_combo(cb: QComboBox, value) -> None:
    idx = cb.findData(value)
    if idx < 0:
        idx = cb.findText(str(value))
    if idx >= 0:
        cb.setCurrentIndex(idx)
    elif cb.isEditable():
        cb.setEditText(str(value))


def _combo_value(cb: QComboBox) -> str:
    if cb.isEditable():
        text = cb.currentText().strip()
        idx = cb.findText(text)
        if idx >= 0 and cb.itemData(idx) is not None:
            return str(cb.itemData(idx))
        return text
    return str(cb.currentData())


def _spin(lo, hi, value, step=1, special: Optional[str] = None, suffix="") -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(value)
    if special:
        s.setSpecialValueText(special)
    if suffix:
        s.setSuffix(suffix)
    return s


def _dspin(lo, hi, value, step=0.1, decimals=2, special: Optional[str] = None, suffix="") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    if special:
        s.setSpecialValueText(special)
    if suffix:
        s.setSuffix(suffix)
    return s


def _filter_combo(examples: List[str]) -> QComboBox:
    cb = QComboBox()
    cb.setEditable(True)
    cb.addItems(examples)
    cb.lineEdit().setPlaceholderText("none (raw ffmpeg -vf syntax, comma separated)")
    return cb


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Tribble Upscaler {__version__}")
        self.resize(1400, 900)
        self.settings = QSettings("tribble", "tribble-upscaler")
        self.cache = UpscalerCache()
        self._threads: list = []
        self._job: Optional[JobWorker] = None
        self._job_outputs: List[str] = []
        self._preview_busy = False
        self._media_duration = 0.0

        self._build_ui()
        self._build_menu()
        self.refresh_models()
        self.refresh_presets()
        self._restore()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # ---- left: settings ------------------------------------------
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)

        g = QGroupBox("Inputs")
        gv = QVBoxLayout(g)
        self.queue = FileQueue()
        self.queue.setMinimumHeight(110)
        self.queue.currentItemChanged.connect(lambda *_: self._on_input_selected())
        gv.addWidget(self.queue)
        row = QHBoxLayout()
        for text, slot in (
            ("Add files…", self.add_files),
            ("Add folder…", self.add_folder),
            ("Remove", self.queue.remove_selected),
            ("Clear", self.queue.clear),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        gv.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Output folder:"))
        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText("same folder as each input")
        row.addWidget(self.out_dir, 1)
        b = QToolButton()
        b.setText("…")
        b.clicked.connect(self.pick_out_dir)
        row.addWidget(b)
        gv.addLayout(row)
        self.input_info = QLabel("")
        self.input_info.setWordWrap(True)
        self.input_info.setStyleSheet("color: gray")
        gv.addWidget(self.input_info)
        lv.addWidget(g)

        g = QGroupBox("Model chain")
        gv = QVBoxLayout(g)
        self.chain = ModelChainTable()
        self.chain.setMinimumHeight(120)
        self.chain.changed.connect(self._on_chain_changed)
        gv.addWidget(self.chain)
        row = QHBoxLayout()
        for text, tip, slot in (
            ("+", "Add model step", lambda: self.chain.add_step()),
            ("−", "Remove selected step", self.chain.remove_selected),
            ("▲", "Move up", lambda: self.chain.move_selected(-1)),
            ("▼", "Move down", lambda: self.chain.move_selected(1)),
        ):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch()
        for text, slot in (
            ("Model info", self.show_model_info),
            ("Refresh", self.refresh_models),
            ("Download…", self.open_download_dialog),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        gv.addLayout(row)
        lv.addWidget(g)

        tabs = QTabWidget()
        tabs.addTab(self._processing_tab(), "Processing")
        tabs.addTab(self._input_tab(), "Input")
        tabs.addTab(self._output_tab(), "Output")
        lv.addWidget(tabs, 1)

        g = QGroupBox("Presets")
        row = QHBoxLayout(g)
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumContentsLength(14)
        row.addWidget(self.preset_combo, 1)
        for text, slot in (("Load", self.load_selected_preset), ("Save…", self.save_preset),
                           ("Import…", self.import_preset), ("Delete", self.delete_preset)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        lv.addWidget(g)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setMinimumWidth(540)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        splitter.addWidget(scroll)

        # ---- right: preview + run ------------------------------------
        right = QSplitter(Qt.Vertical)
        pv = QWidget()
        pl = QVBoxLayout(pv)
        pl.setContentsMargins(6, 6, 6, 0)
        bar = QHBoxLayout()
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 1000)
        self.time_slider.valueChanged.connect(self._update_time_label)
        self.time_label = QLabel("0:00.00")
        self.time_label.setMinimumWidth(70)
        self.preview_btn = QPushButton("Preview frame")
        self.preview_btn.setShortcut(QKeySequence("F5"))
        self.preview_btn.setToolTip("Upscale the frame at the slider position (F5)")
        self.preview_btn.clicked.connect(self.run_preview)
        self.mode_combo = _combo(CompareView.MODES)
        self.mode_combo.currentTextChanged.connect(lambda m: self.view.set_mode(m))
        bar.addWidget(QLabel("Time:"))
        bar.addWidget(self.time_slider, 1)
        bar.addWidget(self.time_label)
        bar.addWidget(self.preview_btn)
        bar.addWidget(self.mode_combo)
        for text, tip, slot in (
            ("Fit", "Fit to window (double-click)", lambda: self.view.reset_view()),
            ("1:1", "Actual pixels", lambda: self.view.set_actual_size()),
            ("Save…", "Save the upscaled preview frame", self.save_preview),
        ):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            bar.addWidget(b)
        pl.addLayout(bar)
        self.view = CompareView()
        self.view.setToolTip("Drag: move split • Wheel: zoom • Right-drag: pan • Double-click: fit")
        pl.addWidget(self.view, 1)
        self.preview_info = QLabel("")
        self.preview_info.setStyleSheet("color: gray")
        pl.addWidget(self.preview_info)
        right.addWidget(pv)

        run = QWidget()
        rl = QVBoxLayout(run)
        rl.setContentsMargins(6, 0, 6, 6)
        row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m frames")
        self.progress.setValue(0)
        row.addWidget(self.progress, 1)
        self.start_btn = QPushButton("Start")
        self.start_btn.setShortcut(QKeySequence("Ctrl+Return"))
        self.start_btn.setToolTip("Process all queued inputs (Ctrl+Enter)")
        self.start_btn.setStyleSheet("font-weight: bold; padding: 4px 18px")
        self.start_btn.clicked.connect(self.start_jobs)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_jobs)
        row.addWidget(self.start_btn)
        row.addWidget(self.cancel_btn)
        rl.addLayout(row)
        self.status = QLabel("Ready")
        rl.addWidget(self.status)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setPlaceholderText("Log")
        rl.addWidget(self.log_view, 1)
        right.addWidget(run)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([560, 840])
        self.splitter = splitter
        self.tabs = tabs

    def _processing_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.device = _combo(available_devices(), editable=True)
        self.device.setToolTip("auto picks CUDA, then Apple MPS, then CPU")
        f.addRow("Device:", self.device)
        self.half = QCheckBox("Half precision (fp16) on GPU")
        self.half.setToolTip("Faster and uses less VRAM. Disabled automatically for models that don't support it.")
        f.addRow(self.half)
        self.tile = _spin(0, 8192, 0, 64, special="Off (whole frame)")
        self.tile.setToolTip("Split frames into tiles to save VRAM. Shrinks automatically on out-of-memory.")
        f.addRow("Tile size:", self.tile)
        self.overlap = _spin(0, 256, 16, 4, suffix=" px")
        self.overlap.setToolTip("Tiles overlap and are feather-blended to hide seams")
        f.addRow("Tile overlap:", self.overlap)
        self.pad_mult = _spin(1, 256, 8)
        self.pad_mult.setToolTip("Tiles are padded to a multiple of this (combined with the model's own requirement)")
        f.addRow("Pad to multiple of:", self.pad_mult)
        self.bit_depth = _combo([("8", "8-bit"), ("16", "16-bit (for 10-bit output / HDR-ish sources)")])
        f.addRow("Pipe precision:", self.bit_depth)
        self.queue_size = _spin(1, 256, 8)
        self.queue_size.setToolTip("Frames buffered between decoder, model and encoder")
        f.addRow("Frame buffer:", self.queue_size)
        self.onnx_providers = QLineEdit()
        self.onnx_providers.setPlaceholderText("auto (e.g. CUDAExecutionProvider,CPUExecutionProvider)")
        f.addRow("ONNX providers:", self.onnx_providers)
        self.unsafe = QCheckBox("Allow fully pickled models (only for files you trust)")
        f.addRow(self.unsafe)
        return w

    def _input_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.start_time = _dspin(0, 1e6, 0, 1, 3, special="beginning", suffix=" s")
        f.addRow("Start at:", self.start_time)
        self.end_time = _dspin(0, 1e6, 0, 1, 3, special="end", suffix=" s")
        f.addRow("End at:", self.end_time)
        self.max_frames = _spin(0, 10_000_000, 0, 1, special="all")
        f.addRow("Max frames:", self.max_frames)
        self.pre_scale = _dspin(0.05, 1.0, 1.0, 0.05, special="")
        self.pre_scale.setToolTip("Downscale before the model — helps with noisy / soft sources")
        f.addRow("Pre-scale:", self.pre_scale)
        self.pre_filters = _filter_combo(PRE_FILTER_EXAMPLES)
        self.pre_filters.setToolTip("ffmpeg filters applied while decoding (deinterlace, denoise, crop …)")
        f.addRow("Pre-filters:", self.pre_filters)
        return w

    def _output_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.size_mode = _combo(SIZE_MODES)
        self.size_mode.currentIndexChanged.connect(self._update_size_widgets)
        f.addRow("Output size:", self.size_mode)
        self.out_scale = _dspin(0.1, 16, 2.0, 0.5, suffix=" ×")
        f.addRow("Scale:", self.out_scale)
        row = QHBoxLayout()
        self.out_w = _spin(0, 16384, 3840, 2, special="auto")
        self.out_h = _spin(0, 16384, 2160, 2, special="auto")
        row.addWidget(self.out_w)
        row.addWidget(QLabel("×"))
        row.addWidget(self.out_h)
        f.addRow("Width × height:", row)
        self.resize_filter = _combo(RESIZE_FILTERS, editable=True)
        f.addRow("Resize filter:", self.resize_filter)
        self.post_filters = _filter_combo(POST_FILTER_EXAMPLES)
        self.post_filters.setToolTip("ffmpeg filters applied before encoding (sharpen, grain, color …)")
        f.addRow("Post-filters:", self.post_filters)

        self.container = _combo(CONTAINERS, editable=True)
        f.addRow("Container:", self.container)
        self.codec = _combo(CODECS, editable=True)
        self.codec.setToolTip("Any ffmpeg video encoder name works")
        f.addRow("Video codec:", self.codec)
        row = QHBoxLayout()
        self.crf = _spin(-1, 63, 18, 1, special="codec default")
        self.crf.setToolTip("Quality (CRF / CQ). Lower = better. Ignored when a bitrate is set.")
        row.addWidget(self.crf)
        self.bitrate = QLineEdit()
        self.bitrate.setPlaceholderText("bitrate, e.g. 25M (overrides CRF)")
        row.addWidget(self.bitrate)
        f.addRow("Quality:", row)
        self.enc_preset = _combo(ENC_PRESETS, editable=True)
        f.addRow("Encoder preset:", self.enc_preset)
        self.pix_fmt = _combo(PIXEL_FORMATS, editable=True)
        f.addRow("Pixel format:", self.pix_fmt)
        self.color_matrix = _combo(["auto", "bt709", "bt601", "bt2020"])
        f.addRow("Color matrix:", self.color_matrix)
        self.out_fps = _dspin(0, 480, 0, 1, 3, special="same as source")
        f.addRow("Frame rate:", self.out_fps)
        row = QHBoxLayout()
        self.copy_audio = QCheckBox("Audio")
        self.audio_codec = _combo(AUDIO_CODECS, editable=True)
        self.audio_bitrate = QLineEdit()
        self.audio_bitrate.setPlaceholderText("e.g. 192k")
        self.copy_subs = QCheckBox("Subtitles")
        row.addWidget(self.copy_audio)
        row.addWidget(self.audio_codec)
        row.addWidget(self.audio_bitrate)
        row.addWidget(self.copy_subs)
        f.addRow("Keep:", row)
        self.extra_args = QLineEdit()
        self.extra_args.setPlaceholderText('e.g. -x265-params "aq-mode=3" -tune film')
        f.addRow("Extra encoder args:", self.extra_args)
        self.png_seq = QCheckBox("Write PNG image sequence instead of a video")
        f.addRow(self.png_seq)
        self.overwrite = QCheckBox("Overwrite existing outputs")
        f.addRow(self.overwrite)
        self._update_size_widgets()
        return w

    def _build_menu(self) -> None:
        m = self.menuBar().addMenu("&File")
        self._action(m, "Add files…", self.add_files, "Ctrl+O")
        self._action(m, "Add folder…", self.add_folder)
        m.addSeparator()
        self._action(m, "Import preset…", self.import_preset)
        self._action(m, "Export settings…", self.export_preset)
        m.addSeparator()
        self._action(m, "Quit", self.close, "Ctrl+Q")

        m = self.menuBar().addMenu("&Models")
        self._action(m, "Download models…", self.open_download_dialog)
        self._action(m, "Refresh model list", self.refresh_models)
        self._action(m, "Open models folder", lambda: self._open_folder(USER_DIR / "models"))
        self._action(m, "Open plugins folder", lambda: self._open_folder(USER_DIR / "plugins"))
        self._action(m, "Add model folder…", self.add_model_folder)
        self._action(m, "Unload models (free memory)", self.unload_models)

        m = self.menuBar().addMenu("&View")
        self.dark_action = self._action(m, "Dark theme", self._toggle_theme)
        self.dark_action.setCheckable(True)

        m = self.menuBar().addMenu("&Help")
        self._action(m, "About", self.about)

    def _action(self, menu, text, slot, shortcut=None) -> QAction:
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    # ------------------------------------------------------------------
    # config <-> UI
    # ------------------------------------------------------------------
    def config_from_ui(self) -> JobConfig:
        providers = [p.strip() for p in self.onnx_providers.text().split(",") if p.strip()]
        crf = self.crf.value()
        return JobConfig(
            models=self.chain.steps(),
            processing=ProcessingSettings(
                device=_combo_value(self.device) or "auto",
                half_precision=self.half.isChecked(),
                tile_size=self.tile.value(),
                tile_overlap=self.overlap.value(),
                tile_pad_multiple=self.pad_mult.value(),
                bit_depth=int(self.bit_depth.currentData()),
                queue_size=self.queue_size.value(),
                allow_unsafe_pickle=self.unsafe.isChecked(),
                onnx_providers=providers,
            ),
            input=InputSettings(
                start_time=self.start_time.value() or None,
                end_time=self.end_time.value() or None,
                max_frames=self.max_frames.value() or None,
                pre_scale=self.pre_scale.value(),
                pre_filters=self.pre_filters.currentText().strip(),
            ),
            output=OutputSettings(
                size_mode=self.size_mode.currentData(),
                scale=self.out_scale.value(),
                width=self.out_w.value(),
                height=self.out_h.value(),
                resize_filter=_combo_value(self.resize_filter),
                post_filters=self.post_filters.currentText().strip(),
                container=_combo_value(self.container),
                codec=_combo_value(self.codec),
                crf=None if crf < 0 else crf,
                bitrate=self.bitrate.text().strip(),
                preset=_combo_value(self.enc_preset),
                pixel_format=_combo_value(self.pix_fmt),
                fps=self.out_fps.value() or None,
                color_matrix=self.color_matrix.currentData(),
                copy_audio=self.copy_audio.isChecked(),
                audio_codec=_combo_value(self.audio_codec) or "copy",
                audio_bitrate=self.audio_bitrate.text().strip(),
                copy_subtitles=self.copy_subs.isChecked(),
                extra_args=self.extra_args.text().strip(),
                image_sequence=self.png_seq.isChecked(),
                overwrite=self.overwrite.isChecked(),
            ),
        )

    def apply_config(self, cfg: JobConfig) -> None:
        self.chain.set_steps(cfg.models)
        p, i, o = cfg.processing, cfg.input, cfg.output
        _set_combo(self.device, p.device)
        self.half.setChecked(p.half_precision)
        self.tile.setValue(p.tile_size)
        self.overlap.setValue(p.tile_overlap)
        self.pad_mult.setValue(p.tile_pad_multiple)
        _set_combo(self.bit_depth, str(p.bit_depth))
        self.queue_size.setValue(p.queue_size)
        self.unsafe.setChecked(p.allow_unsafe_pickle)
        self.onnx_providers.setText(",".join(p.onnx_providers))

        self.start_time.setValue(i.start_time or 0)
        self.end_time.setValue(i.end_time or 0)
        self.max_frames.setValue(i.max_frames or 0)
        self.pre_scale.setValue(i.pre_scale or 1.0)
        self.pre_filters.setEditText(i.pre_filters)

        _set_combo(self.size_mode, o.size_mode)
        self.out_scale.setValue(o.scale)
        self.out_w.setValue(o.width)
        self.out_h.setValue(o.height)
        _set_combo(self.resize_filter, o.resize_filter)
        self.post_filters.setEditText(o.post_filters)
        _set_combo(self.container, o.container)
        _set_combo(self.codec, o.codec)
        self.crf.setValue(-1 if o.crf is None else o.crf)
        self.bitrate.setText(o.bitrate)
        _set_combo(self.enc_preset, o.preset)
        _set_combo(self.pix_fmt, o.pixel_format)
        self.out_fps.setValue(o.fps or 0)
        _set_combo(self.color_matrix, o.color_matrix)
        self.copy_audio.setChecked(o.copy_audio)
        _set_combo(self.audio_codec, o.audio_codec)
        self.audio_bitrate.setText(o.audio_bitrate)
        self.copy_subs.setChecked(o.copy_subtitles)
        self.extra_args.setText(o.extra_args)
        self.png_seq.setChecked(o.image_sequence)
        self.overwrite.setChecked(o.overwrite)
        self._update_size_widgets()

    def _update_size_widgets(self) -> None:
        mode = self.size_mode.currentData()
        self.out_scale.setEnabled(mode == "scale")
        self.out_w.setEnabled(mode in ("width", "exact"))
        self.out_h.setEnabled(mode in ("height", "exact"))

    # ------------------------------------------------------------------
    # inputs
    # ------------------------------------------------------------------
    def add_files(self) -> None:
        start = self.settings.value("last_input_dir", "")
        paths, _ = QFileDialog.getOpenFileNames(self, "Add videos or images", start, MEDIA_FILTER)
        if paths:
            self.settings.setValue("last_input_dir", str(Path(paths[0]).parent))
            self.queue.add_files(paths)
            if self.queue.currentRow() < 0:
                self.queue.setCurrentRow(0)

    def add_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Add all media in folder")
        if d:
            self.queue.add_files([d])

    def pick_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.out_dir.text())
        if d:
            self.out_dir.setText(d)

    def _on_input_selected(self) -> None:
        path = self.queue.current_path()
        if not path:
            self.input_info.setText("")
            return
        try:
            info = probe(path)
            self._media_duration = 0.0 if info.is_image else info.duration
            self.input_info.setText(info.summary())
            self.time_slider.setEnabled(not info.is_image)
        except Exception as exc:
            self._media_duration = 0.0
            self.input_info.setText(f"Could not read: {exc}")
        self._update_time_label()

    def _preview_time(self) -> float:
        return self._media_duration * self.time_slider.value() / 1000.0

    def _update_time_label(self) -> None:
        t = self._preview_time()
        self.time_label.setText(f"{int(t // 60)}:{t % 60:05.2f}")

    # ------------------------------------------------------------------
    # models
    # ------------------------------------------------------------------
    def _extra_model_dirs(self) -> List[str]:
        v = self.settings.value("extra_model_dirs", [])
        if isinstance(v, str):
            v = [v]
        return list(v or [])

    def refresh_models(self) -> None:
        entries = scan_models(model_dirs(self._extra_model_dirs()))
        self.chain.set_available(entries)
        self.log(f"Found {len(entries)} model file(s) in: "
                 + ", ".join(str(d) for d in model_dirs(self._extra_model_dirs()) if d.is_dir()))

    def add_model_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Add a folder to scan for models")
        if d:
            dirs = self._extra_model_dirs()
            if d not in dirs:
                dirs.append(d)
                self.settings.setValue("extra_model_dirs", dirs)
            self.refresh_models()

    def _on_chain_changed(self) -> None:
        pass  # cache keys on config, so nothing to invalidate eagerly

    def show_model_info(self) -> None:
        steps = self.chain.steps()
        row = max(self.chain.currentRow(), 0)
        if not steps:
            return
        from ..models import load_model

        cfg = self.config_from_ui()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            m = load_model(steps[row].model, device="cpu", allow_unsafe_pickle=cfg.processing.allow_unsafe_pickle)
            lines = [
                f"<b>{m.name}</b>",
                f"Backend: {m.backend}",
                f"Architecture: {m.arch or '?'}",
                f"Scale: ×{m.scale:g}",
                f"Channels: {m.in_channels} → {m.out_channels}",
                f"Pad multiple: {m.pad_multiple}",
            ]
            if m.fixed_size:
                lines.append(f"Fixed input size: {m.fixed_size[1]}×{m.fixed_size[0]}")
            for k, v in m.info.items():
                lines.append(f"{k}: {v}")
            QMessageBox.information(self, "Model info", "<br>".join(lines))
        except Exception as exc:
            QMessageBox.warning(self, "Model info", f"Could not load model:\n{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def unload_models(self) -> None:
        self.cache.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self.log("Models unloaded")

    def open_download_dialog(self) -> None:
        dlg = DownloadDialog(self)
        dlg.exec()
        self.refresh_models()

    # ------------------------------------------------------------------
    # presets
    # ------------------------------------------------------------------
    def _preset_save_dir(self) -> Path:
        d = USER_DIR / "presets"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def refresh_presets(self) -> None:
        self.preset_combo.clear()
        seen = set()
        for d in preset_dirs():
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.json")):
                if p.stem in seen:
                    continue
                seen.add(p.stem)
                self.preset_combo.addItem(p.stem, str(p))
                self.preset_combo.setItemData(self.preset_combo.count() - 1, str(p), Qt.ToolTipRole)

    def load_selected_preset(self) -> None:
        path = self.preset_combo.currentData()
        if path:
            self._load_preset_file(path)

    def _load_preset_file(self, path: str) -> None:
        try:
            cfg = JobConfig.load(path)
        except Exception as exc:
            QMessageBox.warning(self, "Preset", f"Could not load preset:\n{exc}")
            return
        if not cfg.models:  # presets without models keep the current chain
            cfg.models = self.chain.steps()
        self.apply_config(cfg)
        self.log(f"Loaded preset {Path(path).name}")

    def save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:", text=self.preset_combo.currentText())
        if not ok or not name.strip():
            return
        safe = "".join(c for c in name.strip() if c.isalnum() or c in " -_.").strip()
        path = self._preset_save_dir() / f"{safe}.json"
        self.config_from_ui().save(path)
        self.refresh_presets()
        _set_combo(self.preset_combo, str(path))
        self.log(f"Saved preset {path}")

    def import_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import preset", "", "Preset (*.json)")
        if path:
            self._load_preset_file(path)

    def export_preset(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export settings", "tribble-preset.json", "Preset (*.json)")
        if path:
            self.config_from_ui().save(path)

    def delete_preset(self) -> None:
        path = self.preset_combo.currentData()
        if not path:
            return
        if QMessageBox.question(self, "Delete preset", f"Delete {Path(path).name}?") == QMessageBox.Yes:
            try:
                Path(path).unlink()
            except OSError as exc:
                QMessageBox.warning(self, "Delete preset", str(exc))
            self.refresh_presets()

    # ------------------------------------------------------------------
    # preview
    # ------------------------------------------------------------------
    def run_preview(self) -> None:
        if self._preview_busy or self._job is not None:
            return
        path = self.queue.current_path()
        if not path:
            self.view.set_message("Add an input file first")
            return
        cfg = self.config_from_ui()
        if not [s for s in cfg.models if s.enabled]:
            self.view.set_message("Add at least one model to the chain")
            return
        self._preview_busy = True
        self.preview_btn.setEnabled(False)
        self.view.set_message("Upscaling preview…")
        self.status.setText("Rendering preview…")
        w = PreviewWorker(path, self._preview_time(), cfg, self.cache)
        w.log.connect(self.log)
        w.ready.connect(self._preview_ready)
        w.failed.connect(self._preview_failed)
        w.finished.connect(self._preview_done)
        self._start_worker(w)

    def _preview_ready(self, before, after, seconds) -> None:
        self._last_preview = after
        self.view.set_images(before, after)
        self.preview_info.setText(
            f"{before.shape[1]}×{before.shape[0]} → {after.shape[1]}×{after.shape[0]} "
            f"in {seconds:.2f}s (≈{1 / seconds if seconds else 0:.2f} fps, before encoding/resizing)"
        )

    def _preview_failed(self, msg: str) -> None:
        self.view.set_message(f"Preview failed: {msg}")
        self.log(f"Preview failed: {msg}")

    def _preview_done(self) -> None:
        self._preview_busy = False
        self.preview_btn.setEnabled(self._job is None)
        self.status.setText("Ready")

    def save_preview(self) -> None:
        img = getattr(self, "_last_preview", None)
        if img is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save preview frame", "preview.png", "PNG (*.png)")
        if path:
            from .widgets import array_to_qimage

            array_to_qimage(img).save(path)

    # ------------------------------------------------------------------
    # jobs
    # ------------------------------------------------------------------
    def start_jobs(self) -> None:
        if self._job is not None:
            return
        inputs = self.queue.paths()
        if not inputs:
            QMessageBox.information(self, "Nothing to do", "Add at least one input file.")
            return
        cfg = self.config_from_ui()
        if not [s for s in cfg.models if s.enabled]:
            QMessageBox.information(self, "No model", "Add at least one model to the chain.")
            return
        out_dir = self.out_dir.text().strip() or None
        jobs = [(p, str(default_output_path(p, cfg, out_dir))) for p in inputs]
        existing = [o for _, o in jobs if Path(o).exists()]
        if existing and not cfg.output.overwrite and not cfg.output.image_sequence:
            ans = QMessageBox.question(
                self, "Overwrite?", f"{len(existing)} output file(s) already exist, e.g.\n{existing[0]}\n\nOverwrite?"
            )
            if ans != QMessageBox.Yes:
                return
            cfg.output.overwrite = True

        self._job_outputs = [o for _, o in jobs]
        for i in range(len(jobs)):
            self.queue.set_status(i, "queued")
        self.start_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setValue(0)
        self._persist()

        w = JobWorker(jobs, cfg, self.cache)
        w.log.connect(self.log)
        w.progress.connect(self._on_progress)
        w.file_started.connect(self._on_file_started)
        w.file_finished.connect(self._on_file_finished)
        w.finished.connect(self._on_jobs_done)
        self._job = w
        self._start_worker(w)

    def cancel_jobs(self) -> None:
        if self._job is not None:
            self._job.cancel()
            self.status.setText("Cancelling…")

    def _on_file_started(self, i: int) -> None:
        self.queue.set_status(i, "running")
        self.status.setText(f"Processing {Path(self.queue.paths()[i]).name} ({i + 1}/{self.queue.count()})")

    def _on_file_finished(self, i: int, ok: bool, msg: str) -> None:
        self.queue.set_status(i, "done" if ok else ("cancelled" if msg == "cancelled" else "failed"))

    def _on_progress(self, pr) -> None:
        self.progress.setMaximum(max(pr.total, 1))
        self.progress.setValue(pr.frame)
        eta = f"{int(pr.eta // 3600)}:{int(pr.eta % 3600 // 60):02d}:{int(pr.eta % 60):02d}"
        self.status.setText(f"{pr.frame}/{pr.total} frames • {pr.fps:.2f} fps • ETA {eta}")

    def _on_jobs_done(self) -> None:
        self._job = None
        self.start_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status.setText("Finished")

    def _start_worker(self, worker) -> None:
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        thread.finished.connect(lambda: self._forget_thread(thread))
        thread.finished.connect(thread.deleteLater)
        self._threads.append((thread, worker))
        thread.start()

    def _forget_thread(self, thread) -> None:
        self._threads = [(t, w) for t, w in self._threads if t is not thread]

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------
    def log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg.rstrip())

    def _open_folder(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _toggle_theme(self) -> None:
        apply_theme(QApplication.instance(), self.dark_action.isChecked())
        self.settings.setValue("dark", self.dark_action.isChecked())

    def about(self) -> None:
        dirs = "<br>".join(str(d) for d in model_dirs(self._extra_model_dirs()))
        pdirs = "<br>".join(str(d) for d in plugin_dirs())
        QMessageBox.about(
            self,
            "About Tribble",
            f"<b>Tribble Upscaler {__version__}</b><br>Local AI video upscaling.<br><br>"
            f"<b>Model folders</b><br>{dirs}<br><br><b>Plugin folders</b><br>{pdirs}",
        )

    def _persist(self) -> None:
        self.settings.setValue("config", json.dumps(self.config_from_ui().to_dict()))
        self.settings.setValue("out_dir", self.out_dir.text())
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("splitter", self.splitter.saveState())

    def _restore(self) -> None:
        raw = self.settings.value("config", "")
        cfg = None
        if raw:
            try:
                cfg = JobConfig.from_dict(json.loads(raw))
            except Exception:
                cfg = None
        if cfg is None:
            cfg = JobConfig()
        if not cfg.models:
            self.chain.add_step()
            cfg.models = self.chain.steps()
        self.apply_config(cfg)
        self.out_dir.setText(self.settings.value("out_dir", "") or "")
        geo = self.settings.value("geometry")
        if geo:
            self.restoreGeometry(geo)
        st = self.settings.value("splitter")
        if st:
            self.splitter.restoreState(st)
        dark = str(self.settings.value("dark", "false")).lower() == "true"
        self.dark_action.setChecked(dark)
        apply_theme(QApplication.instance(), dark)

    def closeEvent(self, e):
        if self._job is not None:
            if QMessageBox.question(self, "Quit", "A job is running. Cancel it and quit?") != QMessageBox.Yes:
                e.ignore()
                return
            self._job.cancel()
        self._persist()
        for thread, _ in list(self._threads):
            thread.quit()
            thread.wait(10000)
        super().closeEvent(e)


class DownloadDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Download models")
        self.resize(640, 360)
        v = QVBoxLayout(self)
        v.addWidget(QLabel(
            f"Downloads go to <code>{USER_DIR / 'models'}</code>. Any other model (e.g. from "
            "<a href='https://openmodeldb.info'>OpenModelDB</a>) works by dropping the file into a models folder."
        ))
        v.itemAt(0).widget().setOpenExternalLinks(True)
        v.itemAt(0).widget().setWordWrap(True)
        self.list = QListWidget()
        for m in CATALOG.values():
            exists = (USER_DIR / "models" / m.filename).exists()
            it = QListWidgetItem(f"{m.key}  (×{m.scale}){'  ✓ installed' if exists else ''}\n    {m.description}")
            it.setData(Qt.UserRole, m.key)
            self.list.addItem(it)
        v.addWidget(self.list, 1)
        self.bar = QProgressBar()
        self.bar.setValue(0)
        v.addWidget(self.bar)
        bb = QDialogButtonBox()
        self.dl_btn = bb.addButton("Download", QDialogButtonBox.AcceptRole)
        bb.addButton(QDialogButtonBox.Close)
        self.dl_btn.clicked.connect(self.download)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self._thread = None

    def download(self) -> None:
        item = self.list.currentItem()
        if item is None or self._thread is not None:
            return
        self.dl_btn.setEnabled(False)
        w = DownloadWorker(item.data(Qt.UserRole))
        t = QThread(self)
        w.moveToThread(t)
        t.started.connect(w.run)
        w.progress.connect(self._progress)
        w.done.connect(self._done)
        w.failed.connect(self._failed)
        w.finished.connect(t.quit)
        t.finished.connect(self._finished)
        self._thread, self._worker = t, w
        t.start()

    def _progress(self, got: int, total: int) -> None:
        self.bar.setMaximum(total or 0)
        self.bar.setValue(got)

    def _done(self, path: str) -> None:
        item = self.list.currentItem()
        if item is not None and "installed" not in item.text():
            first, rest = item.text().split("\n", 1)
            item.setText(f"{first}  ✓ installed\n{rest}")

    def _failed(self, msg: str) -> None:
        QMessageBox.warning(self, "Download failed", msg)

    def _finished(self) -> None:
        self._thread = None
        self.dl_btn.setEnabled(True)

    def reject(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(100)
        super().reject()


def apply_theme(app: QApplication, dark: bool) -> None:
    app.setStyle("Fusion")
    if not dark:
        app.setPalette(app.style().standardPalette())
        return
    p = QPalette()
    base, alt, text = QColor(37, 37, 40), QColor(48, 48, 52), QColor(225, 225, 228)
    p.setColor(QPalette.Window, alt)
    p.setColor(QPalette.WindowText, text)
    p.setColor(QPalette.Base, base)
    p.setColor(QPalette.AlternateBase, alt)
    p.setColor(QPalette.ToolTipBase, base)
    p.setColor(QPalette.ToolTipText, text)
    p.setColor(QPalette.Text, text)
    p.setColor(QPalette.Button, alt)
    p.setColor(QPalette.ButtonText, text)
    p.setColor(QPalette.Highlight, QColor(76, 130, 220))
    p.setColor(QPalette.HighlightedText, Qt.white)
    p.setColor(QPalette.Link, QColor(110, 160, 240))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(120, 120, 120))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 120, 120))
    app.setPalette(p)


def main(argv: Optional[List[str]] = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv if argv is None else ["tribble-gui", *argv])
    app.setApplicationName("Tribble Upscaler")
    win = MainWindow()
    files = [a for a in (sys.argv[1:] if argv is None else argv) if os.path.exists(a)]
    if files:
        win.queue.add_files(files)
        win.queue.setCurrentRow(0)
    win.show()
    QTimer.singleShot(0, win._on_input_selected)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
