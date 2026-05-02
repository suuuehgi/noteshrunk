#!/usr/bin/env python3
"""GUI frontend for noteshrunk.py.

This module provides a PySide6-based GUI that wraps the existing CLI tool
``noteshrunk.py``. It never imports the script directly; instead it invokes it
via ``subprocess`` using ``sys.executable``.

The GUI is intentionally decoupled and focuses on:
- Page management (multi-file input, ordering, state badges)
- Visual preview (original vs processed with draggable split, zoom, pan)
- Settings mapping 1:1 to noteshrunk CLI arguments
- Background processing using QThread to keep the UI responsive
- Config persistence and a simple live log pane
"""

from __future__ import annotations

import configparser
import os
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets


# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

APP_NAME = "Noteshrunk GUI"
CONFIG_PATH = Path(os.path.expanduser("~")) / ".config" / "noteshrunk_gui.conf"
NOTESHRUNK_SCRIPT = str((Path(__file__).parent / "noteshrunk.py").resolve())
VALID_OUTPUT_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class BadgeState:
    IDLE = "idle"
    PROCESSING = "processing"
    DONE = "done"
    SKIPPED = "skipped"


@dataclass
class RenderRequest:
    """Container for a render job description."""

    input_files: List[Path]
    output_path: Optional[Path]
    args: List[str]
    preview_mode: bool


# ---------------------------------------------------------------------------
# Worker thread for subprocess invocation
# ---------------------------------------------------------------------------

class RenderWorker(QtCore.QThread):
    """Run noteshrunk in a background thread.

    Emits:
        finished(path: str): emitted with output path when successful
        error(msg: str): emitted with a human-readable error message
        progress(done: int, total: int): for export runs (pages processed)
        log(text: str): for streaming stdout/stderr to the log pane
    """

    finished = QtCore.Signal(str)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    log = QtCore.Signal(str)

    def __init__(self, request: RenderRequest, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._request = request
        self._process: Optional[QtCore.QProcess] = None
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation of the running subprocess.

        This tries to terminate the underlying process gracefully and, if that
        fails, kills it. The QThread will then finish shortly after.
        """

        self._cancelled = True
        if self._process is not None:
            self._process.terminate()
            if not self._process.waitForFinished(1000):
                self._process.kill()

    def run(self) -> None:  # type: ignore[override]
        request = self._request

        if not request.input_files:
            self.error.emit("No input files provided.")
            return

        program = sys.executable
        base_args = [NOTESHRUNK_SCRIPT]

        if request.preview_mode:
            assert request.output_path is not None
            cmd = [program, *base_args, str(request.input_files[0]), "-o", str(request.output_path), *request.args]
            total_pages = 1
        else:
            assert request.output_path is not None
            files = [str(p) for p in request.input_files]
            cmd = [program, *base_args, *files, "-o", str(request.output_path), *request.args]
            total_pages = len(files)

        self.log.emit(f"Running: {' '.join(shlex.quote(c) for c in cmd)}\n")

        process = QtCore.QProcess()
        self._process = process

        # Merge stderr into stdout for simpler logging
        process.setProgram(program)
        process.setArguments(cmd[1:])
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        process.readyReadStandardOutput.connect(lambda: self._on_ready_read(process, total_pages))

        process.start()
        if not process.waitForStarted(5000):
            self.error.emit("Failed to start noteshrunk process.")
            return

        process.waitForFinished(-1)

        exit_code = process.exitCode()
        if self._cancelled:
            self.error.emit("Render cancelled by user.")
            return

        if exit_code != 0:
            out = process.readAllStandardOutput().data().decode("utf-8", errors="replace")
            self.error.emit(f"noteshrunk failed (exit code {exit_code}).\n{out}")
            return

        if request.output_path is not None:
            self.finished.emit(str(request.output_path))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_ready_read(self, process: QtCore.QProcess, total_pages: int) -> None:
        """Handle streaming output and naive progress updates."""

        data = process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        if not data:
            return
        self.log.emit(data)

        # Very simple heuristic: count lines containing "Processing image".
        done = 0
        for line in data.splitlines():
            if "Processing image" in line:
                done += 1
        if done:
            self.progress.emit(done, total_pages)


# ---------------------------------------------------------------------------
# Page thumbnails and strip
# ---------------------------------------------------------------------------

class PageThumbnailWidget(QtWidgets.QWidget):
    """Thumbnail widget with an overlaid badge for the page state."""

    def __init__(self, path: Path, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.path = path
        self._badge_state = BadgeState.IDLE
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.thumbnail_label = QtWidgets.QLabel(self)
        self.thumbnail_label.setAlignment(QtCore.Qt.AlignCenter)
        self.thumbnail_label.setMinimumSize(80, 80)
        self.thumbnail_label.setFrameShape(QtWidgets.QFrame.Box)

        self.badge_label = QtWidgets.QLabel(self)
        self.badge_label.setAlignment(QtCore.Qt.AlignCenter)
        self.badge_label.setStyleSheet("QLabel { background: #444; color: white; border-radius: 8px; padding: 2px 6px; }")
        self.badge_label.setText(self._badge_state)

        name_label = QtWidgets.QLabel(self.path.name, self)
        name_label.setAlignment(QtCore.Qt.AlignCenter)
        name_label.setWordWrap(True)

        badge_layout = QtWidgets.QHBoxLayout()
        badge_layout.addStretch(1)
        badge_layout.addWidget(self.badge_label)

        layout.addLayout(badge_layout)
        layout.addWidget(self.thumbnail_label)
        layout.addWidget(name_label)

        pixmap = QtGui.QPixmap(str(self.path))
        if not pixmap.isNull():
            self.thumbnail_label.setPixmap(pixmap.scaled(120, 120, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def set_badge(self, state: str) -> None:
        self._badge_state = state
        self.badge_label.setText(state)


class PageStrip(QtWidgets.QListWidget):
    """Scrollable strip of pages with drag-reorder and multi-file add."""

    fileSelectionChanged = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setViewMode(QtWidgets.QListView.IconMode)
        self.setResizeMode(QtWidgets.QListView.Adjust)
        self.setMovement(QtWidgets.QListView.Snap)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setSpacing(6)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)

        self.itemSelectionChanged.connect(self.fileSelectionChanged)

    # Public API --------------------------------------------------------

    def add_files(self, paths: Iterable[Path]) -> None:
        """Append files as new pages."""

        for path in paths:
            if not path.exists():
                continue
            item = QtWidgets.QListWidgetItem(self)
            widget = PageThumbnailWidget(path)
            item.setSizeHint(widget.sizeHint())
            self.addItem(item)
            self.setItemWidget(item, widget)

        if self.count() and not self.selectedItems():
            self.setCurrentRow(0)

    def clear_files(self) -> None:
        self.clear()

    def current_file(self) -> Optional[Path]:
        item = self.currentItem()
        if not item:
            return None
        widget = self.itemWidget(item)
        if isinstance(widget, PageThumbnailWidget):
            return widget.path
        return None

    def ordered_files(self) -> List[Path]:
        files: List[Path] = []
        for i in range(self.count()):
            item = self.item(i)
            widget = self.itemWidget(item)
            if isinstance(widget, PageThumbnailWidget):
                files.append(widget.path)
        return files

    def set_badge(self, index: int, state: str) -> None:
        if not (0 <= index < self.count()):
            return
        item = self.item(index)
        widget = self.itemWidget(item)
        if isinstance(widget, PageThumbnailWidget):
            widget.set_badge(state)

    # Drag & drop -------------------------------------------------------

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            paths = [Path(u.toLocalFile()) for u in event.mimeData().urls()]
            self.add_files(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


# ---------------------------------------------------------------------------
# Preview widget
# ---------------------------------------------------------------------------

class PreviewWidget(QtWidgets.QWidget):
    """Composite preview of processed | original images with draggable split."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._original: Optional[QtGui.QPixmap] = None
        self._processed: Optional[QtGui.QPixmap] = None
        self._split_ratio = 0.5
        self._zoom = 1.0
        self._pan = QtCore.QPointF(0, 0)
        self._dragging = False
        self._last_mouse_pos = QtCore.QPoint()
        self._show_spinner = False
        self.setMouseTracking(True)

    def set_original(self, pixmap: Optional[QtGui.QPixmap]) -> None:
        self._original = pixmap
        self.update()

    def set_processed(self, pixmap: Optional[QtGui.QPixmap]) -> None:
        self._processed = pixmap
        self.update()

    def show_spinner(self, visible: bool) -> None:
        self._show_spinner = visible
        self.update()

    def fit_to_window(self) -> None:
        if not self._original:
            return
        size = self.size()
        pix_size = self._original.size()
        if pix_size.isEmpty():
            return
        scale = min(size.width() / pix_size.width(), size.height() / pix_size.height())
        self._zoom = max(scale, 0.01)
        self._pan = QtCore.QPointF(0, 0)
        self.update()

    # Painting ----------------------------------------------------------

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), self.palette().color(QtGui.QPalette.Base))

        if not (self._original and self._processed):
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "No preview available")
            return

        center = QtCore.QPointF(self.width() / 2, self.height() / 2)

        painter.save()
        painter.translate(center + self._pan)
        painter.scale(self._zoom, self._zoom)

        orig_rect = QtCore.QRectF(-self._original.width(), -self._original.height() / 2,
                                  self._original.width(), self._original.height())
        proc_rect = QtCore.QRectF(0, -self._processed.height() / 2,
                                  self._processed.width(), self._processed.height())

        painter.drawPixmap(orig_rect.topLeft(), self._original)
        painter.drawPixmap(proc_rect.topLeft(), self._processed)

        # Split line between processed and original, slider effect could be
        # added later; for now this is a simple dividing line.
        split_x = 0
        painter.setPen(QtGui.QPen(QtGui.QColor("#ff8800"), 2))
        painter.drawLine(split_x, orig_rect.top(), split_x, orig_rect.bottom())

        painter.restore()

        if self._show_spinner:
            painter.setOpacity(0.6)
            painter.fillRect(self.rect(), self.palette().window())
            painter.setOpacity(1.0)
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "Processing…")

    # Interaction -------------------------------------------------------

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # type: ignore[override]
        angle = event.angleDelta().y()
        factor = 1.25 if angle > 0 else 0.8
        self._zoom = max(0.1, min(self._zoom * factor, 10.0))
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.buttons() & QtCore.Qt.LeftButton:
            self._dragging = True
            self._last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if self._dragging:
            delta = event.pos() - self._last_mouse_pos
            self._pan += QtCore.QPointF(delta.x(), delta.y())
            self._last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        self._dragging = False


# ---------------------------------------------------------------------------
# Settings panel
# ---------------------------------------------------------------------------

class SettingsPanel(QtWidgets.QScrollArea):
    """Scrollable panel mapping directly to noteshrunk CLI arguments."""

    settingChanged = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        container = QtWidgets.QWidget(self)
        self.setWidget(container)

        self._widgets: dict[str, QtWidgets.QWidget] = {}

        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_output_group())
        layout.addWidget(self._build_background_group())
        layout.addWidget(self._build_palette_group())
        layout.addWidget(self._build_filtering_group())
        layout.addWidget(self._build_binarize_group())
        layout.addWidget(self._build_thresholds_group())
        layout.addWidget(self._build_skip_empty_group())
        layout.addWidget(self._build_advanced_group())

        layout.addStretch(1)

    # Group builders ----------------------------------------------------

    def _group(self, title: str) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(title, self)
        box.setCheckable(False)
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(4)
        return box

    def _build_output_group(self) -> QtWidgets.QGroupBox:
        box = self._group("Output")
        layout = box.layout()

        ncolors = QtWidgets.QSpinBox(box)
        ncolors.setRange(2, 256)
        ncolors.setValue(8)
        ncolors.valueChanged.connect(self.settingChanged)

        dpi = QtWidgets.QSpinBox(box)
        dpi.setRange(72, 1200)
        dpi.setValue(300)
        dpi.valueChanged.connect(self.settingChanged)

        quality = QtWidgets.QSpinBox(box)
        quality.setRange(1, 100)
        quality.setValue(75)
        quality.valueChanged.connect(self.settingChanged)

        layout.addWidget(self._labeled("Number of colours", ncolors))
        layout.addWidget(self._labeled("DPI", dpi))
        layout.addWidget(self._labeled("JPEG/PDF quality", quality))

        self._widgets["n_colors"] = ncolors
        self._widgets["dpi"] = dpi
        self._widgets["quality"] = quality

        return box

    def _build_background_group(self) -> QtWidgets.QGroupBox:
        box = self._group("Background")
        layout = box.layout()

        white_bg = QtWidgets.QCheckBox("White background", box)
        black = QtWidgets.QCheckBox("Force black", box)
        normalize = QtWidgets.QCheckBox("Normalize / contrast stretch", box)
        saturate = QtWidgets.QCheckBox("Maximise saturation", box)

        for w in (white_bg, black, normalize, saturate):
            w.stateChanged.connect(self.settingChanged)
            layout.addWidget(w)

        self._widgets["white_background"] = white_bg
        self._widgets["black"] = black
        self._widgets["normalize"] = normalize
        self._widgets["saturate"] = saturate

        return box

    def _build_palette_group(self) -> QtWidgets.QGroupBox:
        box = self._group("Palette")
        layout = box.layout()

        self.local_palette_chk = QtWidgets.QCheckBox("Local palette per page", box)
        self.local_palette_chk.setChecked(False)
        self.local_palette_chk.stateChanged.connect(self.settingChanged)

        self.percentage_spin = QtWidgets.QDoubleSpinBox(box)
        self.percentage_spin.setRange(1.0, 100.0)
        self.percentage_spin.setSingleStep(1.0)
        self.percentage_spin.setValue(100.0)
        self.percentage_spin.valueChanged.connect(self.settingChanged)

        # Palette input: start with simple QLineEdit, can be extended with
        # swatches in the future.
        self.palette_line = QtWidgets.QLineEdit(box)
        self.palette_line.setPlaceholderText("#FFFFFF,#FF0000,#000000")
        self.palette_line.textChanged.connect(self.settingChanged)

        layout.addWidget(self.local_palette_chk)
        layout.addWidget(self._labeled("Sampling percentage", self.percentage_spin))
        layout.addWidget(self._labeled("Custom palette (comma-separated hex)", self.palette_line))

        info = QtWidgets.QLabel(
            "Preview uses per-page palette. Final export uses global palette when local palette is disabled.",
            box,
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(info)
        self.global_palette_info = info
        self.global_palette_info.setVisible(False)

        self._widgets["local_palette"] = self.local_palette_chk
        self._widgets["percentage"] = self.percentage_spin
        self._widgets["palette"] = self.palette_line

        return box

    def _build_filtering_group(self) -> QtWidgets.QGroupBox:
        box = self._group("Filtering")
        layout = box.layout()

        denoise_median = QtWidgets.QCheckBox("Median denoise", box)
        denoise_closing = QtWidgets.QCheckBox("Morphological closing", box)
        denoise_opening = QtWidgets.QCheckBox("Morphological opening", box)
        unsharp_mask = QtWidgets.QCheckBox("Unsharp mask", box)

        for w in (denoise_median, denoise_closing, denoise_opening, unsharp_mask):
            w.stateChanged.connect(self.settingChanged)
            layout.addWidget(w)

        median_strength = QtWidgets.QSpinBox(box)
        median_strength.setRange(1, 15)
        median_strength.setValue(3)
        median_strength.valueChanged.connect(self.settingChanged)

        closing_strength = QtWidgets.QDoubleSpinBox(box)
        closing_strength.setRange(0.0, 20.0)
        closing_strength.setValue(3.0)
        closing_strength.setSingleStep(0.5)
        closing_strength.valueChanged.connect(self.settingChanged)

        opening_strength = QtWidgets.QDoubleSpinBox(box)
        opening_strength.setRange(0.0, 20.0)
        opening_strength.setValue(3.0)
        opening_strength.setSingleStep(0.5)
        opening_strength.valueChanged.connect(self.settingChanged)

        unsharp_amount = QtWidgets.QDoubleSpinBox(box)
        unsharp_amount.setRange(0.1, 10.0)
        unsharp_amount.setValue(2.0)
        unsharp_amount.setSingleStep(0.1)
        unsharp_amount.valueChanged.connect(self.settingChanged)

        unsharp_radius = QtWidgets.QDoubleSpinBox(box)
        unsharp_radius.setRange(0.1, 50.0)
        unsharp_radius.setValue(5.0)
        unsharp_radius.setSingleStep(0.5)
        unsharp_radius.valueChanged.connect(self.settingChanged)

        layout.addWidget(self._labeled("Median strength", median_strength))
        layout.addWidget(self._labeled("Closing strength", closing_strength))
        layout.addWidget(self._labeled("Opening strength", opening_strength))
        layout.addWidget(self._labeled("Unsharp amount", unsharp_amount))
        layout.addWidget(self._labeled("Unsharp radius", unsharp_radius))

        self._widgets["denoise_median"] = denoise_median
        self._widgets["denoise_closing"] = denoise_closing
        self._widgets["denoise_opening"] = denoise_opening
        self._widgets["unsharp_mask"] = unsharp_mask
        self._widgets["median_strength"] = median_strength
        self._widgets["closing_strength"] = closing_strength
        self._widgets["opening_strength"] = opening_strength
        self._widgets["unsharp_amount"] = unsharp_amount
        self._widgets["unsharp_radius"] = unsharp_radius

        return box

    def _build_binarize_group(self) -> QtWidgets.QGroupBox:
        box = self._group("Binarize")
        layout = box.layout()

        binarize = QtWidgets.QCheckBox("Binarise image", box)
        binarize.stateChanged.connect(self.settingChanged)

        threshold_binarize = QtWidgets.QSpinBox(box)
        threshold_binarize.setRange(0, 100)
        threshold_binarize.setSpecialValueText("auto")
        threshold_binarize.setValue(0)
        threshold_binarize.valueChanged.connect(self.settingChanged)

        layout.addWidget(binarize)
        layout.addWidget(self._labeled("Threshold (percent, 0 = auto)", threshold_binarize))

        self._widgets["binarize"] = binarize
        self._widgets["threshold_binarize"] = threshold_binarize

        return box

    def _build_thresholds_group(self) -> QtWidgets.QGroupBox:
        box = self._group("Thresholds")
        layout = box.layout()

        threshold_saturation = QtWidgets.QSpinBox(box)
        threshold_saturation.setRange(0, 100)
        threshold_saturation.setValue(15)
        threshold_saturation.valueChanged.connect(self.settingChanged)

        threshold_value = QtWidgets.QSpinBox(box)
        threshold_value.setRange(0, 100)
        threshold_value.setValue(20)
        threshold_value.valueChanged.connect(self.settingChanged)

        threshold_empty = QtWidgets.QSpinBox(box)
        threshold_empty.setRange(0, 1000)
        threshold_empty.setValue(2)
        threshold_empty.valueChanged.connect(self.settingChanged)

        layout.addWidget(self._labeled("Saturation threshold", threshold_saturation))
        layout.addWidget(self._labeled("Value threshold", threshold_value))
        layout.addWidget(self._labeled("Empty-page threshold (‰)", threshold_empty))

        self._widgets["threshold_saturation"] = threshold_saturation
        self._widgets["threshold_value"] = threshold_value
        self._widgets["threshold_empty"] = threshold_empty

        return box

    def _build_skip_empty_group(self) -> QtWidgets.QGroupBox:
        box = self._group("Skip empty")
        layout = box.layout()

        skip_empty = QtWidgets.QCheckBox("Skip empty pages", box)
        skip_empty.stateChanged.connect(self.settingChanged)
        layout.addWidget(skip_empty)

        self._widgets["skip_empty"] = skip_empty
        return box

    def _build_advanced_group(self) -> QtWidgets.QGroupBox:
        box = self._group("Advanced")
        layout = box.layout()

        jobs = QtWidgets.QSpinBox(box)
        jobs.setRange(1, max(1, os.cpu_count() or 1))
        jobs.setValue(max(1, os.cpu_count() or 1))
        jobs.valueChanged.connect(self.settingChanged)

        verbose = QtWidgets.QSpinBox(box)
        verbose.setRange(0, 2)
        verbose.setValue(0)
        verbose.valueChanged.connect(self.settingChanged)

        keep_intermediate = QtWidgets.QCheckBox("Keep intermediate PDFs", box)
        keep_intermediate.stateChanged.connect(self.settingChanged)

        layout.addWidget(self._labeled("Jobs", jobs))
        layout.addWidget(self._labeled("Verbose level (0-2)", verbose))
        layout.addWidget(keep_intermediate)

        self._widgets["jobs"] = jobs
        self._widgets["verbose"] = verbose
        self._widgets["keep_intermediate"] = keep_intermediate

        return box

    # Utility -----------------------------------------------------------

    @staticmethod
    def _labeled(label: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        wrapper = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(QtWidgets.QLabel(label))
        layout.addWidget(widget, 1)
        return wrapper

    # Public API --------------------------------------------------------

    def set_global_palette_info_visible(self, visible: bool) -> None:
        self.global_palette_info.setVisible(visible)

    def to_args(self) -> List[str]:
        """Convert current widget state to a list of CLI arguments."""

        args: List[str] = []

        ncolors: QtWidgets.QSpinBox = self._widgets["n_colors"]  # type: ignore[assignment]
        dpi: QtWidgets.QSpinBox = self._widgets["dpi"]  # type: ignore[assignment]
        quality: QtWidgets.QSpinBox = self._widgets["quality"]  # type: ignore[assignment]
        args += ["-c", str(ncolors.value())]
        args += ["-d", str(dpi.value())]
        args += ["-q", str(quality.value())]

        if self._widgets["black"].isChecked():  # type: ignore[union-attr]
            args.append("--black")
        if self._widgets["white_background"].isChecked():  # type: ignore[union-attr]
            args.append("--white_background")
        if self._widgets["normalize"].isChecked():  # type: ignore[union-attr]
            args.append("--normalize")
        if self._widgets["saturate"].isChecked():  # type: ignore[union-attr]
            args.append("--saturate")

        percentage: QtWidgets.QDoubleSpinBox = self._widgets["percentage"]  # type: ignore[assignment]
        args += ["--percentage", f"{percentage.value():.2f}"]

        palette_line: QtWidgets.QLineEdit = self._widgets["palette"]  # type: ignore[assignment]
        palette_text = palette_line.text().strip()
        if palette_text:
            args += ["--palette", palette_text]
        else:
            if self._widgets["local_palette"].isChecked():  # type: ignore[union-attr]
                args.append("--local_palette")

        if self._widgets["denoise_median"].isChecked():  # type: ignore[union-attr]
            args.append("--denoise_median")
        if self._widgets["denoise_closing"].isChecked():  # type: ignore[union-attr]
            args.append("--denoise_closing")
        if self._widgets["denoise_opening"].isChecked():  # type: ignore[union-attr]
            args.append("--denoise_opening")
        if self._widgets["unsharp_mask"].isChecked():  # type: ignore[union-attr]
            args.append("--unsharp_mask")

        median_strength: QtWidgets.QSpinBox = self._widgets["median_strength"]  # type: ignore[assignment]
        closing_strength: QtWidgets.QDoubleSpinBox = self._widgets["closing_strength"]  # type: ignore[assignment]
        opening_strength: QtWidgets.QDoubleSpinBox = self._widgets["opening_strength"]  # type: ignore[assignment]
        unsharp_amount: QtWidgets.QDoubleSpinBox = self._widgets["unsharp_amount"]  # type: ignore[assignment]
        unsharp_radius: QtWidgets.QDoubleSpinBox = self._widgets["unsharp_radius"]  # type: ignore[assignment]

        args += ["--median_strength", str(median_strength.value())]
        args += ["--closing_strength", f"{closing_strength.value():.2f}"]
        args += ["--opening_strength", f"{opening_strength.value():.2f}"]
        args += ["--unsharp_amount", f"{unsharp_amount.value():2f}"]
        args += ["--unsharp_radius", f"{unsharp_radius.value():2f}"]

        if self._widgets["binarize"].isChecked():  # type: ignore[union-attr]
            args.append("--binarize")

        threshold_binarize: QtWidgets.QSpinBox = self._widgets["threshold_binarize"]  # type: ignore[assignment]
        if threshold_binarize.value() > 0:
            args += ["--threshold_binarize", str(threshold_binarize.value())]

        threshold_saturation: QtWidgets.QSpinBox = self._widgets["threshold_saturation"]  # type: ignore[assignment]
        threshold_value: QtWidgets.QSpinBox = self._widgets["threshold_value"]  # type: ignore[assignment]
        threshold_empty: QtWidgets.QSpinBox = self._widgets["threshold_empty"]  # type: ignore[assignment]

        args += ["--threshold_saturation", str(threshold_saturation.value())]
        args += ["--threshold_value", str(threshold_value.value())]
        args += ["--threshold_empty", str(threshold_empty.value())]

        if self._widgets["skip_empty"].isChecked():  # type: ignore[union-attr]
            args.append("--skip_empty")

        jobs: QtWidgets.QSpinBox = self._widgets["jobs"]  # type: ignore[assignment]
        verbose: QtWidgets.QSpinBox = self._widgets["verbose"]  # type: ignore[assignment]
        if jobs.value() > 0:
            args += ["--jobs", str(jobs.value())]
        if verbose.value() > 0:
            args += ["-" + "v" * verbose.value()]

        if self._widgets["keep_intermediate"].isChecked():  # type: ignore[union-attr]
            args.append("--keep_intermediate")

        # Always overwrite target files: the GUI controls output paths.
        args.append("--overwrite")

        return args

    def to_config(self) -> dict[str, str]:
        cfg: dict[str, str] = {}
        for name, widget in self._widgets.items():
            if isinstance(widget, QtWidgets.QCheckBox):
                cfg[name] = "1" if widget.isChecked() else "0"
            elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                cfg[name] = str(widget.value())
            elif isinstance(widget, QtWidgets.QLineEdit):
                cfg[name] = widget.text()
        return cfg

    def from_config(self, cfg: dict[str, str]) -> None:
        for name, widget in self._widgets.items():
            if name not in cfg:
                continue
            value = cfg[name]
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(value == "1")
            elif isinstance(widget, QtWidgets.QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QtWidgets.QLineEdit):
                widget.setText(value)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    """Main window assembling the three-panel layout and bottom bar."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1200, 800)

        self.page_strip = PageStrip(self)
        self.preview_widget = PreviewWidget(self)
        self.settings_panel = SettingsPanel(self)

        self.status_label = QtWidgets.QLabel("Ready", self)
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)

        self.preview_button = QtWidgets.QPushButton("Preview", self)
        self.run_button = QtWidgets.QPushButton("Run", self)
        self.run_button.setDefault(True)

        self.auto_preview_chk = QtWidgets.QCheckBox("Auto-preview", self)
        self.auto_preview_chk.setChecked(True)

        self.log_dock = self._build_log_panel()

        self._debounce_timer = QtCore.QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(300)

        self._current_worker: Optional[RenderWorker] = None

        self._build_menu()
        self._build_central_layout()
        self._build_bottom_bar()

        self._connect_signals()
        self._load_config()

    # UI construction ---------------------------------------------------

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")

        open_action = QtGui.QAction("Open…", self)
        open_action.triggered.connect(self._on_file_open)
        file_menu.addAction(open_action)

        import_action = QtGui.QAction("Import…", self)
        import_action.triggered.connect(self._on_file_import)
        file_menu.addAction(import_action)

        save_as_action = QtGui.QAction("Save As…", self)
        save_as_action.triggered.connect(self._on_save_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        quit_action = QtGui.QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _build_central_layout(self) -> None:
        splitter = QtWidgets.QSplitter(self)
        splitter.setOrientation(QtCore.Qt.Horizontal)
        splitter.addWidget(self.page_strip)
        splitter.addWidget(self.preview_widget)
        splitter.addWidget(self.settings_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        self.setCentralWidget(splitter)
        self._splitter = splitter

    def _build_bottom_bar(self) -> None:
        bottom = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(bottom)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        layout.addWidget(self.status_label, 1)
        layout.addWidget(self.progress_bar, 1)
        layout.addWidget(self.auto_preview_chk)
        layout.addWidget(self.preview_button)
        layout.addWidget(self.run_button)

        container = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(self.centralWidget())
        vbox.addWidget(bottom)

        self.setCentralWidget(container)

    def _build_log_panel(self) -> QtWidgets.QDockWidget:
        dock = QtWidgets.QDockWidget("Log", self)
        dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        text = QtWidgets.QTextEdit(dock)
        text.setReadOnly(True)
        text.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        dock.setWidget(text)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
        dock.hide()

        self.log_text = text
        return dock

    # Signal connections ------------------------------------------------

    def _connect_signals(self) -> None:
        self.page_strip.fileSelectionChanged.connect(self._on_page_selected)
        self.settings_panel.settingChanged.connect(self._on_setting_changed)

        self.preview_button.clicked.connect(self._on_preview_requested)
        self.run_button.clicked.connect(self._on_run_or_cancel)

        self.auto_preview_chk.stateChanged.connect(self._on_auto_preview_toggled)
        self._debounce_timer.timeout.connect(self._on_preview_requested)

    # Config ------------------------------------------------------------

    def _load_config(self) -> None:
        cfg = configparser.ConfigParser()
        if CONFIG_PATH.exists():
            cfg.read(CONFIG_PATH)

        if cfg.has_section("window"):
            wcfg = cfg["window"]
            try:
                geom = bytes.fromhex(wcfg.get("geometry", ""))
                if geom:
                    self.restoreGeometry(geom)
                state = bytes.fromhex(wcfg.get("window_state", ""))
                if state:
                    self.restoreState(state)
            except Exception:
                pass

        if cfg.has_section("settings"):
            scfg = dict(cfg["settings"])
            self.settings_panel.from_config(scfg)

        if cfg.has_section("ui"):
            ucfg = cfg["ui"]
            auto_preview = ucfg.get("auto_preview", "1") == "1"
            self.auto_preview_chk.setChecked(auto_preview)
            log_visible = ucfg.get("log_visible", "0") == "1"
            self.log_dock.setVisible(log_visible)

    def _save_config(self) -> None:
        cfg = configparser.ConfigParser()

        cfg["window"] = {
            "geometry": self.saveGeometry().hex(),
            "window_state": self.saveState().hex(),
        }
        cfg["settings"] = self.settings_panel.to_config()
        cfg["ui"] = {
            "auto_preview": "1" if self.auto_preview_chk.isChecked() else "0",
            "log_visible": "1" if self.log_dock.isVisible() else "0",
        }

        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CONFIG_PATH.open("w", encoding="utf-8") as fh:
            cfg.write(fh)

    # Events ------------------------------------------------------------

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        if self._current_worker is not None:
            self._current_worker.cancel()
            self._current_worker.wait(2000)
        self._save_config()
        super().closeEvent(event)

    # Slots -------------------------------------------------------------

    def _on_page_selected(self) -> None:
        path = self.page_strip.current_file()
        if not path:
            return
        pixmap = QtGui.QPixmap(str(path))
        self.preview_widget.set_original(pixmap)
        if self.auto_preview_chk.isChecked():
            self._debounce_timer.start()

    def _on_setting_changed(self) -> None:
        if self.auto_preview_chk.isChecked():
            self._debounce_timer.start()

    def _on_auto_preview_toggled(self, state: int) -> None:
        enabled = state == QtCore.Qt.Checked
        self.preview_button.setEnabled(not enabled)
        if enabled:
            self._debounce_timer.start()

    def _on_preview_requested(self) -> None:
        files = self.page_strip.ordered_files()
        if not files:
            self.status_label.setText("No input files loaded.")
            return

        current = self.page_strip.current_file() or files[0]
        self._start_render(preview_mode=True, input_files=[current])

    def _on_run_or_cancel(self) -> None:
        if self._current_worker is not None:
            self._current_worker.cancel()
            return
        files = self.page_strip.ordered_files()
        if not files:
            self.status_label.setText("No input files loaded.")
            return
        self._start_render(preview_mode=False, input_files=files)

    def _on_file_open(self) -> None:
        if self.page_strip.count() > 0:
            res = QtWidgets.QMessageBox.question(
                self,
                "Replace pages",
                "Replace the current file list with new files?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if res != QtWidgets.QMessageBox.Yes:
                return

        paths = self._ask_for_files()
        if not paths:
            return
        self.page_strip.clear_files()
        self.page_strip.add_files(paths)

    def _on_file_import(self) -> None:
        paths = self._ask_for_files()
        if not paths:
            return
        self.page_strip.add_files(paths)

    def _ask_for_files(self) -> List[Path]:
        dlg = QtWidgets.QFileDialog(self, "Select input images")
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        dlg.setNameFilter("Images (*.pdf *.png *.jpg *.jpeg *.tif *.tiff)")
        if not dlg.exec():  # 0 if cancelled
            return []
        files = [Path(f) for f in dlg.selectedFiles()]
        return files

    def _on_save_as(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save output", "output.pdf",
                                                         "PDF (*.pdf);;PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff)")
        if not path:
            return
        out = Path(path)
        if out.suffix.lower() not in VALID_OUTPUT_SUFFIXES:
            QtWidgets.QMessageBox.warning(self, "Invalid extension",
                                          "Unsupported output file extension. Please use PDF, PNG, JPEG or TIFF.")
            return
        self._last_output_path = out
        self.status_label.setText(f"Output will be saved to {out}")

    # Rendering ---------------------------------------------------------

    def _start_render(self, preview_mode: bool, input_files: List[Path]) -> None:
        if preview_mode:
            suffix = ".png"
            tmp_fd, tmp_name = tempfile.mkstemp(suffix=suffix, prefix="noteshrunk-preview-")
            os.close(tmp_fd)
            output = Path(tmp_name)
        else:
            output = getattr(self, "_last_output_path", None)
            if output is None:
                self._on_save_as()
                output = getattr(self, "_last_output_path", None)
                if output is None:
                    return

        args = self.settings_panel.to_args()

        request = RenderRequest(input_files=input_files, output_path=output, args=args, preview_mode=preview_mode)
        worker = RenderWorker(request, self)
        worker.finished.connect(self._on_render_finished)
        worker.error.connect(self._on_render_error)
        worker.progress.connect(self._on_render_progress)
        worker.log.connect(self._append_log)
        worker.finished.connect(lambda _: self._cleanup_worker())
        worker.error.connect(lambda _: self._cleanup_worker())

        self._current_worker = worker
        self._set_running_state(True, preview_mode=preview_mode)
        worker.start()

    def _cleanup_worker(self) -> None:
        self._current_worker = None
        self._set_running_state(False)

    def _set_running_state(self, running: bool, preview_mode: bool = False) -> None:
        self.run_button.setText("Cancel" if running and not preview_mode else "Run")
        self.preview_button.setEnabled(not running and not self.auto_preview_chk.isChecked())
        self.progress_bar.setVisible(running)
        if running:
            self.progress_bar.setRange(0, 0)
            self.preview_widget.show_spinner(True)
        else:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            self.preview_widget.show_spinner(False)

    def _on_render_finished(self, output_path: str) -> None:
        self.status_label.setText(f"Finished: {output_path}")
        path = Path(output_path)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            pixmap = QtGui.QPixmap(output_path)
            if not pixmap.isNull():
                self.preview_widget.set_processed(pixmap)

    def _on_render_error(self, msg: str) -> None:
        self.status_label.setText(msg.splitlines()[0])
        QtWidgets.QMessageBox.critical(self, "Error", msg)

    def _on_render_progress(self, done: int, total: int) -> None:
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)
        self.status_label.setText(f"Processing page {done}/{total}")

    def _append_log(self, text: str) -> None:
        if not self.log_dock.isVisible():
            self.log_dock.show()
        self.log_text.moveCursor(QtGui.QTextCursor.End)
        self.log_text.insertPlainText(text)
        self.log_text.moveCursor(QtGui.QTextCursor.End)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
