"""Custom widgets: before/after comparison, model chain editor, file queue."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QTableWidget,
    QToolButton,
    QWidget,
)

from ..config import ModelStep
from ..models import BUILTIN_MODELS, ModelEntry
from ..video import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

MODEL_FILTER = "Models (*.pth *.pt *.pt2 *.ckpt *.bin *.safetensors *.onnx *.jit *.torchscript);;All files (*)"


def array_to_qimage(a: np.ndarray) -> QImage:
    a = np.ascontiguousarray(a)
    h, w = a.shape[:2]
    return QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy()


# --------------------------------------------------------------------------
# comparison view
# --------------------------------------------------------------------------


class CompareView(QWidget):
    """Before/after viewer.

    * drag with the left button to move the split line
    * mouse wheel zooms around the cursor, right/middle drag pans
    * double-click resets to fit
    """

    MODES = ("Split", "Side by side", "After only", "Before only")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 200)
        self.setMouseTracking(True)
        self.before: Optional[QImage] = None
        self.after: Optional[QImage] = None
        self.split = 0.5
        self.mode = "Split"
        self.zoom: Optional[float] = None  # None = fit
        self.offset = QPointF(0, 0)
        self._pan_start: Optional[QPointF] = None
        self._dragging_split = False
        self.message = "Pick an input file and press Preview"

    def set_images(self, before: np.ndarray, after: np.ndarray) -> None:
        self.before, self.after = array_to_qimage(before), array_to_qimage(after)
        self.message = ""
        self.update()

    def set_message(self, text: str) -> None:
        self.message = text
        self.update()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.update()

    def reset_view(self) -> None:
        self.zoom, self.offset = None, QPointF(0, 0)
        self.update()

    def set_actual_size(self) -> None:
        self.zoom, self.offset = 1.0, QPointF(0, 0)
        self.update()

    # geometry --------------------------------------------------------
    def _canvas_size(self):
        w, h = self.after.width(), self.after.height()
        if self.mode == "Side by side":
            return w * 2 + 8, h
        return w, h

    def _scale(self) -> float:
        if self.zoom is not None:
            return self.zoom
        cw, ch = self._canvas_size()
        return min(self.width() / cw, self.height() / ch)

    def _origin(self) -> QPointF:
        cw, ch = self._canvas_size()
        s = self._scale()
        return QPointF((self.width() - cw * s) / 2, (self.height() - ch * s) / 2) + self.offset

    def _image_rect(self, index: int = 0) -> QRectF:
        s = self._scale()
        o = self._origin()
        w, h = self.after.width() * s, self.after.height() * s
        x = o.x() + index * (w + 8 * s)
        return QRectF(x, o.y(), w, h)

    # painting --------------------------------------------------------
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), self.palette().window().color().darker(130))
        if self.after is None or self.before is None:
            p.setPen(self.palette().text().color())
            p.drawText(self.rect(), Qt.AlignCenter, self.message or "")
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform, self._scale() < 1.0)
        r = self._image_rect()
        if self.mode == "After only":
            p.drawImage(r, self.after)
        elif self.mode == "Before only":
            p.drawImage(r, self.before)
        elif self.mode == "Side by side":
            p.drawImage(r, self.before)
            p.drawImage(self._image_rect(1), self.after)
            self._label(p, r, "Before", Qt.AlignLeft)
            self._label(p, self._image_rect(1), "After", Qt.AlignLeft)
        else:
            sx = r.left() + r.width() * self.split
            p.save()
            p.setClipRect(QRectF(r.left(), r.top(), sx - r.left(), r.height()))
            p.drawImage(r, self.before)
            p.restore()
            p.save()
            p.setClipRect(QRectF(sx, r.top(), r.right() - sx, r.height()))
            p.drawImage(r, self.after)
            p.restore()
            p.setPen(QPen(QColor(255, 255, 255, 220), 2))
            p.drawLine(QPointF(sx, r.top()), QPointF(sx, r.bottom()))
            self._label(p, r, "Before", Qt.AlignLeft)
            self._label(p, r, "After", Qt.AlignRight)
        if self.message:
            self._label(p, QRectF(self.rect()), self.message, Qt.AlignHCenter)

    def _label(self, p: QPainter, r: QRectF, text: str, align) -> None:
        fm = p.fontMetrics()
        tw, th = fm.horizontalAdvance(text) + 12, fm.height() + 6
        top = max(r.top(), 0) + 6
        if align == Qt.AlignRight:
            x = min(r.right(), self.width()) - tw - 6
        elif align == Qt.AlignHCenter:
            x = r.center().x() - tw / 2
        else:
            x = max(r.left(), 0) + 6
        box = QRectF(x, top, tw, th)
        p.fillRect(box, QColor(0, 0, 0, 150))
        p.setPen(Qt.white)
        p.drawText(box, Qt.AlignCenter, text)

    # interaction -----------------------------------------------------
    def mousePressEvent(self, e):
        if self.after is None:
            return
        if e.button() == Qt.LeftButton and self.mode == "Split":
            self._dragging_split = True
            self._move_split(e.position())
        elif e.button() in (Qt.RightButton, Qt.MiddleButton):
            self._pan_start = e.position()

    def mouseMoveEvent(self, e):
        if self._dragging_split:
            self._move_split(e.position())
        elif self._pan_start is not None:
            if self.zoom is None:
                self.zoom = self._scale()
            self.offset += e.position() - self._pan_start
            self._pan_start = e.position()
            self.update()

    def mouseReleaseEvent(self, e):
        self._dragging_split = False
        self._pan_start = None

    def mouseDoubleClickEvent(self, e):
        self.reset_view()

    def _move_split(self, pos: QPointF) -> None:
        r = self._image_rect()
        if r.width() > 0:
            self.split = min(max((pos.x() - r.left()) / r.width(), 0.0), 1.0)
            self.update()

    def wheelEvent(self, e):
        if self.after is None:
            return
        old = self._scale()
        factor = 1.25 if e.angleDelta().y() > 0 else 0.8
        new = min(max(old * factor, 0.05), 32.0)
        # keep the point under the cursor fixed
        pos = e.position()
        o = self._origin()
        img_pt = (pos - o) / old
        self.zoom = new
        self.offset = QPointF(0, 0)
        o2 = self._origin()
        self.offset = pos - img_pt * new - o2
        self.update()


# --------------------------------------------------------------------------
# model chain
# --------------------------------------------------------------------------


class ModelChainTable(QTableWidget):
    """Ordered list of models with per-step strength and tile override."""

    changed = Signal()
    COLS = ("On", "Model", "", "Strength", "Tile")

    def __init__(self, parent=None):
        super().__init__(0, len(self.COLS), parent)
        self.setHorizontalHeaderLabels(self.COLS)
        self.verticalHeader().setVisible(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        h = self.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.available: List[ModelEntry] = []
        self.setToolTip(
            "Models run top to bottom. Strength blends each model's output with a plain\n"
            "bicubic upscale (lower = subtler). Tile overrides the global tile size for\n"
            "that model ('global' = use the Processing setting)."
        )

    def set_available(self, entries: List[ModelEntry]) -> None:
        self.available = entries
        steps = self.steps()
        self.setRowCount(0)
        for s in steps:
            self.add_step(s)

    def _combo(self, value: str) -> QComboBox:
        cb = QComboBox()
        cb.setEditable(True)
        cb.setInsertPolicy(QComboBox.NoInsert)
        cb.setMinimumContentsLength(18)
        cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        for e in self.available:
            cb.addItem(e.name, e.spec)
            cb.setItemData(cb.count() - 1, str(e.path), Qt.ToolTipRole)
        for b in BUILTIN_MODELS:
            cb.addItem(b, b)
        self._select(cb, value)
        cb.currentIndexChanged.connect(lambda *_: self.changed.emit())
        cb.editTextChanged.connect(lambda *_: self.changed.emit())
        return cb

    @staticmethod
    def _select(cb: QComboBox, value: str) -> None:
        if not value:
            return
        idx = cb.findData(value)
        if idx < 0:
            # match by resolved path, then add as a custom entry
            for i in range(cb.count()):
                d = cb.itemData(i)
                try:
                    if d and Path(d).expanduser().resolve() == Path(value).expanduser().resolve():
                        idx = i
                        break
                except OSError:
                    pass
        if idx < 0:
            cb.addItem(Path(value).stem if not value.startswith("builtin:") else value, value)
            cb.setItemData(cb.count() - 1, value, Qt.ToolTipRole)
            idx = cb.count() - 1
        cb.setCurrentIndex(idx)

    def add_step(self, step: Optional[ModelStep] = None) -> None:
        if step is None:
            default = self.available[0].spec if self.available else BUILTIN_MODELS[0]
            step = ModelStep(default)
        row = self.rowCount()
        self.insertRow(row)

        on = QCheckBox()
        on.setChecked(step.enabled)
        on.toggled.connect(lambda *_: self.changed.emit())
        wrap = QWidget()
        lay = QHBoxLayout(wrap)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.addWidget(on)
        self.setCellWidget(row, 0, wrap)

        cb = self._combo(step.model)
        self.setCellWidget(row, 1, cb)

        browse = QToolButton()
        browse.setText("…")
        browse.setToolTip("Browse for a model file")
        browse.clicked.connect(lambda _=False, c=cb: self._browse(c))
        self.setCellWidget(row, 2, browse)

        st = QDoubleSpinBox()
        st.setRange(0.0, 1.0)
        st.setSingleStep(0.05)
        st.setDecimals(2)
        st.setValue(step.strength)
        st.valueChanged.connect(lambda *_: self.changed.emit())
        self.setCellWidget(row, 3, st)

        tile = QSpinBox()
        tile.setRange(-1, 8192)
        tile.setSingleStep(64)
        tile.setSpecialValueText("global")
        tile.setValue(-1 if step.tile_size is None else step.tile_size)
        tile.setToolTip("Per-model tile size. 'global' uses the Processing tab; 0 = whole frame")
        tile.valueChanged.connect(lambda *_: self.changed.emit())
        self.setCellWidget(row, 4, tile)
        self.changed.emit()

    def _browse(self, cb: QComboBox) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose model", "", MODEL_FILTER)
        if path:
            self._select(cb, path)

    def remove_selected(self) -> None:
        row = self.currentRow()
        if row >= 0:
            self.removeRow(row)
            self.changed.emit()

    def move_selected(self, delta: int) -> None:
        row = self.currentRow()
        steps = self.steps()
        new = row + delta
        if row < 0 or not (0 <= new < len(steps)):
            return
        steps[row], steps[new] = steps[new], steps[row]
        self.set_steps(steps)
        self.selectRow(new)

    def set_steps(self, steps: List[ModelStep]) -> None:
        self.setRowCount(0)
        for s in steps:
            self.add_step(s)
        self.changed.emit()

    def steps(self) -> List[ModelStep]:
        out = []
        for r in range(self.rowCount()):
            on = self.cellWidget(r, 0).findChild(QCheckBox)
            cb: QComboBox = self.cellWidget(r, 1)
            spec = cb.currentData()
            text = cb.currentText().strip()
            if spec is None or (text and text != cb.itemText(cb.currentIndex())):
                spec = text  # user typed a path / builtin spec
            tile = self.cellWidget(r, 4).value()
            out.append(
                ModelStep(
                    model=str(spec),
                    strength=self.cellWidget(r, 3).value(),
                    tile_size=None if tile < 0 else tile,
                    enabled=on.isChecked(),
                )
            )
        return out


# --------------------------------------------------------------------------
# file queue
# --------------------------------------------------------------------------


class FileQueue(QListWidget):
    """Input list with drag & drop and per-file status."""

    files_changed = Signal()
    STATUS_ROLE = Qt.UserRole + 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setToolTip("Drop videos or images here")
        self.model().rowsMoved.connect(lambda *_: self.files_changed.emit())

    def add_files(self, paths: List[str]) -> None:
        existing = set(self.paths())
        for p in paths:
            path = Path(p)
            if path.is_dir():
                self.add_files(
                    [str(c) for c in sorted(path.iterdir())
                     if c.suffix.lower() in VIDEO_EXTENSIONS | IMAGE_EXTENSIONS]
                )
                continue
            if str(path) in existing:
                continue
            item = QListWidgetItem(path.name)
            item.setData(Qt.UserRole, str(path))
            item.setToolTip(str(path))
            self.addItem(item)
            existing.add(str(path))
        self.files_changed.emit()

    def paths(self) -> List[str]:
        return [self.item(i).data(Qt.UserRole) for i in range(self.count())]

    def remove_selected(self) -> None:
        for item in self.selectedItems():
            self.takeItem(self.row(item))
        self.files_changed.emit()

    def set_status(self, index: int, status: str) -> None:
        item = self.item(index)
        if item is None:
            return
        name = Path(item.data(Qt.UserRole)).name
        item.setText(f"{name}   [{status}]" if status else name)

    def current_path(self) -> Optional[str]:
        item = self.currentItem() or (self.item(0) if self.count() else None)
        return item.data(Qt.UserRole) if item else None

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        if e.mimeData().hasUrls():
            self.add_files([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])
            e.acceptProposedAction()
        else:
            super().dropEvent(e)
