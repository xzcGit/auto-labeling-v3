# Plan 4: Canvas Widget

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the annotation canvas widget — the central interactive area where users view images, draw/edit bounding boxes and keypoints, zoom/pan, and manage annotation selections.

**Architecture:** A single `AnnotationCanvas(QWidget)` backed by a coordinate transform helper that converts between normalized [0,1] annotation coords and pixel screen coords. The canvas paints the image, then overlays annotations using QPainter. Mouse events handle tool modes (select, draw_bbox, draw_keypoint). Signals notify the parent of annotation changes. A `ClassPickerPopup` appears after drawing to assign a class.

**Tech Stack:** Python 3.10+, PyQt5 (QWidget, QPainter, QPen, QBrush, QTransform), pytest

---

## File Structure

```
auto-labeling-v3/
├── src/
│   └── ui/
│       ├── canvas.py              # AnnotationCanvas widget
│       └── class_picker.py        # ClassPickerPopup dialog
├── tests/
│   └── ui/
│       ├── __init__.py
│       ├── test_canvas.py         # Canvas logic tests
│       └── test_class_picker.py   # Class picker tests
```

---

## Task 1: Canvas Core — Image Display, Zoom, Pan

**Files:**
- Create: `src/ui/canvas.py`
- Create: `tests/ui/__init__.py`
- Create: `tests/ui/test_canvas.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for AnnotationCanvas."""
from pathlib import Path

import pytest
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QImage, QColor


def _make_test_image(path: Path, width: int = 200, height: int = 150) -> None:
    img = QImage(width, height, QImage.Format_RGB32)
    img.fill(QColor(Qt.blue))
    img.save(str(path), "PNG")


class TestCanvasCoordinates:
    """Test coordinate transformations between normalized and pixel space."""

    def test_norm_to_pixel_identity(self, qapp):
        from src.ui.canvas import AnnotationCanvas

        canvas = AnnotationCanvas()
        canvas.resize(400, 300)
        # Load an image so transforms are defined
        canvas._image_w = 200
        canvas._image_h = 150
        canvas._scale = 1.0
        canvas._offset_x = 0.0
        canvas._offset_y = 0.0

        # Center of image at scale=1 offset=0
        px, py = canvas.norm_to_pixel(0.5, 0.5)
        assert abs(px - 100.0) < 1.0
        assert abs(py - 75.0) < 1.0

    def test_pixel_to_norm_roundtrip(self, qapp):
        from src.ui.canvas import AnnotationCanvas

        canvas = AnnotationCanvas()
        canvas._image_w = 200
        canvas._image_h = 150
        canvas._scale = 2.0
        canvas._offset_x = 50.0
        canvas._offset_y = 30.0

        # Roundtrip
        nx, ny = 0.3, 0.7
        px, py = canvas.norm_to_pixel(nx, ny)
        nx2, ny2 = canvas.pixel_to_norm(px, py)
        assert abs(nx2 - nx) < 0.001
        assert abs(ny2 - ny) < 0.001

    def test_norm_to_pixel_with_scale(self, qapp):
        from src.ui.canvas import AnnotationCanvas

        canvas = AnnotationCanvas()
        canvas._image_w = 100
        canvas._image_h = 100
        canvas._scale = 2.0
        canvas._offset_x = 10.0
        canvas._offset_y = 20.0

        px, py = canvas.norm_to_pixel(0.0, 0.0)
        assert abs(px - 10.0) < 0.01
        assert abs(py - 20.0) < 0.01

        px, py = canvas.norm_to_pixel(1.0, 1.0)
        assert abs(px - 210.0) < 0.01
        assert abs(py - 220.0) < 0.01


class TestCanvasState:
    def test_initial_tool_mode(self, qapp):
        from src.ui.canvas import AnnotationCanvas

        canvas = AnnotationCanvas()
        assert canvas.tool_mode == "select"

    def test_set_tool_mode(self, qapp):
        from src.ui.canvas import AnnotationCanvas

        canvas = AnnotationCanvas()
        canvas.set_tool_mode("draw_bbox")
        assert canvas.tool_mode == "draw_bbox"
        canvas.set_tool_mode("draw_keypoint")
        assert canvas.tool_mode == "draw_keypoint"

    def test_load_image(self, qapp, tmp_path):
        from src.ui.canvas import AnnotationCanvas

        canvas = AnnotationCanvas()
        canvas.resize(400, 300)
        img_path = tmp_path / "test.png"
        _make_test_image(img_path, 200, 150)
        canvas.load_image(str(img_path))
        assert canvas._image is not None
        assert canvas._image_w == 200
        assert canvas._image_h == 150

    def test_load_image_fit_to_window(self, qapp, tmp_path):
        from src.ui.canvas import AnnotationCanvas

        canvas = AnnotationCanvas()
        canvas.resize(400, 300)
        img_path = tmp_path / "big.png"
        _make_test_image(img_path, 800, 600)
        canvas.load_image(str(img_path))
        # Scale should be set so image fits in widget
        assert canvas._scale <= 1.0
        assert canvas._scale > 0

    def test_set_annotations(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation

        canvas = AnnotationCanvas()
        canvas._image_w = 100
        canvas._image_h = 100

        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4))
        canvas.set_annotations([ann])
        assert len(canvas._annotations) == 1
        assert canvas._annotations[0].class_name == "cat"

    def test_clear_resets_state(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation

        canvas = AnnotationCanvas()
        canvas._image_w = 100
        canvas._image_h = 100
        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4))
        canvas.set_annotations([ann])
        canvas.select_annotation(ann.id)
        canvas.clear()
        assert canvas._annotations == []
        assert canvas._selected_id is None
        assert canvas._image is None


class TestCanvasSelection:
    def test_select_annotation(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation

        canvas = AnnotationCanvas()
        canvas._image_w = 100
        canvas._image_h = 100
        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4))
        canvas.set_annotations([ann])
        canvas.select_annotation(ann.id)
        assert canvas._selected_id == ann.id

    def test_select_none_deselects(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation

        canvas = AnnotationCanvas()
        canvas._image_w = 100
        canvas._image_h = 100
        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4))
        canvas.set_annotations([ann])
        canvas.select_annotation(ann.id)
        canvas.select_annotation(None)
        assert canvas._selected_id is None

    def test_hit_test_bbox(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation

        canvas = AnnotationCanvas()
        canvas._image_w = 200
        canvas._image_h = 200
        canvas._scale = 1.0
        canvas._offset_x = 0.0
        canvas._offset_y = 0.0

        # bbox at center (0.5, 0.5) size (0.4, 0.4) → pixel: x1=60,y1=60,x2=140,y2=140
        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.4, 0.4))
        canvas.set_annotations([ann])

        # Click inside bbox
        result = canvas.hit_test(100.0, 100.0)
        assert result == ann.id

        # Click outside bbox
        result = canvas.hit_test(10.0, 10.0)
        assert result is None
```

Write to `tests/ui/test_canvas.py`. Also create empty `tests/ui/__init__.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n yolov8 pytest tests/ui/test_canvas.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement canvas core**

```python
"""Annotation canvas widget for image display and annotation editing."""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtWidgets import QWidget, QMenu, QAction
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import (
    QPainter,
    QPen,
    QBrush,
    QColor,
    QPixmap,
    QImage,
    QImageReader,
    QFont,
    QCursor,
    QWheelEvent,
    QMouseEvent,
    QPaintEvent,
    QResizeEvent,
)

from src.core.annotation import Annotation, Keypoint

logger = logging.getLogger(__name__)

# Visual constants
HANDLE_SIZE = 6
KEYPOINT_RADIUS = 5
LABEL_FONT_SIZE = 11
LABEL_PADDING = 3
MIN_SCALE = 0.1
MAX_SCALE = 20.0
ZOOM_FACTOR = 1.15


class AnnotationCanvas(QWidget):
    """Canvas widget for displaying images and editing annotations.

    Signals:
        annotation_created(Annotation): New annotation drawn by user.
        annotation_modified(str): Annotation with given ID was moved/resized.
        annotation_selected(str): Annotation with given ID was selected (or None).
        annotation_deleted(str): Annotation with given ID should be deleted.
        class_requested(float, float): Request class picker at pixel position (after drawing).
        annotations_changed(): Any change to annotations occurred.
    """

    annotation_created = pyqtSignal(object)   # Annotation
    annotation_modified = pyqtSignal(str)     # annotation id
    annotation_selected = pyqtSignal(object)  # annotation id or None
    annotation_deleted = pyqtSignal(str)      # annotation id
    class_requested = pyqtSignal(float, float)
    annotations_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(200, 200)

        # Image state
        self._image: QPixmap | None = None
        self._image_w: int = 0
        self._image_h: int = 0

        # View transform
        self._scale: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0

        # Tool mode: "select", "draw_bbox", "draw_keypoint"
        self.tool_mode: str = "select"

        # Annotations
        self._annotations: list[Annotation] = []
        self._selected_id: str | None = None
        self._class_colors: dict[str, str] = {}

        # Drawing state
        self._drawing: bool = False
        self._draw_start: tuple[float, float] | None = None  # normalized
        self._draw_current: tuple[float, float] | None = None  # normalized

        # Dragging state (move/resize)
        self._dragging: bool = False
        self._drag_type: str = ""  # "move", "resize_tl", "resize_br", etc., "move_kp"
        self._drag_ann_id: str | None = None
        self._drag_kp_idx: int = -1
        self._drag_start_norm: tuple[float, float] | None = None
        self._drag_ann_snapshot: dict | None = None

        # Panning state
        self._panning: bool = False
        self._pan_start: tuple[float, float] | None = None

    # ── Coordinate transforms ──────────────────────────────────

    def norm_to_pixel(self, nx: float, ny: float) -> tuple[float, float]:
        """Convert normalized [0,1] image coords to widget pixel coords."""
        px = nx * self._image_w * self._scale + self._offset_x
        py = ny * self._image_h * self._scale + self._offset_y
        return px, py

    def pixel_to_norm(self, px: float, py: float) -> tuple[float, float]:
        """Convert widget pixel coords to normalized [0,1] image coords."""
        if self._image_w == 0 or self._image_h == 0 or self._scale == 0:
            return 0.0, 0.0
        nx = (px - self._offset_x) / (self._image_w * self._scale)
        ny = (py - self._offset_y) / (self._image_h * self._scale)
        return nx, ny

    def _clamp_norm(self, nx: float, ny: float) -> tuple[float, float]:
        """Clamp normalized coords to [0, 1]."""
        return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    # ── Public API ─────────────────────────────────────────────

    def load_image(self, path: str) -> None:
        """Load and display an image, fit to window."""
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        qimage = reader.read()
        if qimage.isNull():
            logger.warning("Failed to load image: %s", path)
            return
        self._image = QPixmap.fromImage(qimage)
        self._image_w = self._image.width()
        self._image_h = self._image.height()
        self._fit_to_window()
        self.update()

    def set_annotations(self, annotations: list[Annotation]) -> None:
        """Set the annotations to display."""
        self._annotations = list(annotations)
        self._selected_id = None
        self.update()

    def set_class_colors(self, colors: dict[str, str]) -> None:
        """Set class name → hex color mapping."""
        self._class_colors = colors
        self.update()

    def select_annotation(self, ann_id: str | None) -> None:
        """Select an annotation by ID, or deselect with None."""
        self._selected_id = ann_id
        self.annotation_selected.emit(ann_id)
        self.update()

    def set_tool_mode(self, mode: str) -> None:
        """Set tool mode: 'select', 'draw_bbox', 'draw_keypoint'."""
        self.tool_mode = mode
        self._drawing = False
        self._draw_start = None
        self._draw_current = None
        if mode == "select":
            self.setCursor(Qt.ArrowCursor)
        elif mode == "draw_bbox":
            self.setCursor(Qt.CrossCursor)
        elif mode == "draw_keypoint":
            self.setCursor(Qt.CrossCursor)

    def clear(self) -> None:
        """Clear image and annotations."""
        self._image = None
        self._image_w = 0
        self._image_h = 0
        self._annotations = []
        self._selected_id = None
        self._drawing = False
        self._draw_start = None
        self._draw_current = None
        self.update()

    def get_selected_annotation(self) -> Annotation | None:
        """Return the currently selected annotation."""
        if self._selected_id is None:
            return None
        for ann in self._annotations:
            if ann.id == self._selected_id:
                return ann
        return None

    def hit_test(self, px: float, py: float) -> str | None:
        """Find annotation at pixel position. Returns annotation ID or None.
        Tests in reverse order (topmost first). Checks keypoints first, then bboxes.
        """
        nx, ny = self.pixel_to_norm(px, py)

        # Check keypoints first (smaller targets, higher priority)
        kp_radius_norm_x = KEYPOINT_RADIUS / (self._image_w * self._scale) if self._image_w * self._scale > 0 else 0
        kp_radius_norm_y = KEYPOINT_RADIUS / (self._image_h * self._scale) if self._image_h * self._scale > 0 else 0

        for ann in reversed(self._annotations):
            for kp in ann.keypoints:
                if abs(kp.x - nx) < kp_radius_norm_x * 2 and abs(kp.y - ny) < kp_radius_norm_y * 2:
                    return ann.id

        # Check bboxes
        for ann in reversed(self._annotations):
            if ann.bbox:
                cx, cy, w, h = ann.bbox
                x1, y1 = cx - w / 2, cy - h / 2
                x2, y2 = cx + w / 2, cy + h / 2
                if x1 <= nx <= x2 and y1 <= ny <= y2:
                    return ann.id

        return None

    def _fit_to_window(self) -> None:
        """Scale and offset so image fits in widget."""
        if self._image_w == 0 or self._image_h == 0:
            return
        ww, wh = self.width(), self.height()
        sx = ww / self._image_w
        sy = wh / self._image_h
        self._scale = min(sx, sy)
        # Center the image
        self._offset_x = (ww - self._image_w * self._scale) / 2
        self._offset_y = (wh - self._image_h * self._scale) / 2

    # ── Paint ──────────────────────────────────────────────────

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor("#1e1e2e"))

        if self._image is None:
            painter.setPen(QColor("#6c7086"))
            painter.drawText(self.rect(), Qt.AlignCenter, "无图片")
            painter.end()
            return

        # Draw image
        dest = QRectF(
            self._offset_x, self._offset_y,
            self._image_w * self._scale, self._image_h * self._scale,
        )
        painter.drawPixmap(dest.toRect(), self._image)

        # Draw annotations
        for ann in self._annotations:
            is_selected = ann.id == self._selected_id
            color = QColor(self._class_colors.get(ann.class_name, "#89b4fa"))
            self._paint_annotation(painter, ann, color, is_selected)

        # Draw in-progress bbox
        if self._drawing and self._draw_start and self._draw_current:
            self._paint_drawing_preview(painter)

        painter.end()

    def _paint_annotation(
        self, painter: QPainter, ann: Annotation, color: QColor, selected: bool
    ) -> None:
        """Paint a single annotation (bbox + keypoints + label)."""
        if ann.bbox:
            cx, cy, w, h = ann.bbox
            x1, y1 = self.norm_to_pixel(cx - w / 2, cy - h / 2)
            x2, y2 = self.norm_to_pixel(cx + w / 2, cy + h / 2)

            pen = QPen(color, 2)
            if not ann.confirmed:
                pen.setStyle(Qt.DashLine)
            if selected:
                pen.setWidth(3)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

            # Label background
            label_text = ann.class_name
            if not ann.confirmed:
                label_text += " ⚡"
            font = QFont()
            font.setPixelSize(LABEL_FONT_SIZE)
            painter.setFont(font)
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(label_text) + LABEL_PADDING * 2
            th = fm.height() + LABEL_PADDING * 2
            label_rect = QRectF(x1, y1 - th, tw, th)
            if label_rect.top() < 0:
                label_rect.moveTop(y1)
            bg_color = QColor(color)
            bg_color.setAlpha(200)
            painter.fillRect(label_rect, bg_color)
            painter.setPen(QColor("#1e1e2e"))
            painter.drawText(label_rect, Qt.AlignCenter, label_text)

            # Control handles when selected
            if selected:
                self._paint_handles(painter, x1, y1, x2, y2)

        # Keypoints
        for i, kp in enumerate(ann.keypoints):
            px, py = self.norm_to_pixel(kp.x, kp.y)
            r = KEYPOINT_RADIUS + (2 if selected else 0)

            if kp.visible == 0:
                painter.setPen(QPen(QColor("#6c7086"), 1))
                painter.setBrush(Qt.NoBrush)
            elif kp.visible == 1:
                painter.setPen(QPen(color, 1))
                painter.setBrush(Qt.NoBrush)
            else:
                painter.setPen(QPen(color, 1))
                painter.setBrush(QBrush(color))

            painter.drawEllipse(QPointF(px, py), r, r)

            # Label for keypoint
            if selected:
                painter.setPen(QColor("#cdd6f4"))
                font = QFont()
                font.setPixelSize(10)
                painter.setFont(font)
                painter.drawText(int(px + r + 2), int(py - 2), kp.label)

    def _paint_handles(self, painter: QPainter, x1: float, y1: float, x2: float, y2: float) -> None:
        """Paint resize handles on selected bbox corners."""
        painter.setPen(QPen(QColor("#cdd6f4"), 1))
        painter.setBrush(QBrush(QColor("#89b4fa")))
        hs = HANDLE_SIZE
        for hx, hy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            painter.drawRect(QRectF(hx - hs, hy - hs, hs * 2, hs * 2))

    def _paint_drawing_preview(self, painter: QPainter) -> None:
        """Paint the bbox being drawn."""
        sx, sy = self.norm_to_pixel(*self._draw_start)
        ex, ey = self.norm_to_pixel(*self._draw_current)
        painter.setPen(QPen(QColor("#89b4fa"), 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        x = min(sx, ex)
        y = min(sy, ey)
        w = abs(ex - sx)
        h = abs(ey - sy)
        painter.drawRect(QRectF(x, y, w, h))

    # ── Mouse events ───────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        px, py = event.x(), event.y()

        # Middle button → pan
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = (px, py)
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() != Qt.LeftButton:
            return

        if self.tool_mode == "draw_bbox":
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            self._drawing = True
            self._draw_start = (nx, ny)
            self._draw_current = (nx, ny)

        elif self.tool_mode == "draw_keypoint":
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            # Emit class_requested to get class info, then create keypoint
            self.class_requested.emit(px, py)
            # Store the norm coords for when class is assigned
            self._draw_start = (nx, ny)

        elif self.tool_mode == "select":
            # Check if clicking a handle first (for selected bbox)
            handle = self._hit_test_handle(px, py)
            if handle:
                self._dragging = True
                self._drag_type = handle
                self._drag_ann_id = self._selected_id
                self._drag_start_norm = self.pixel_to_norm(px, py)
                ann = self.get_selected_annotation()
                if ann:
                    self._drag_ann_snapshot = ann.to_dict()
                return

            # Check if clicking a keypoint to drag
            kp_hit = self._hit_test_keypoint(px, py)
            if kp_hit:
                ann_id, kp_idx = kp_hit
                self._dragging = True
                self._drag_type = "move_kp"
                self._drag_ann_id = ann_id
                self._drag_kp_idx = kp_idx
                self._drag_start_norm = self.pixel_to_norm(px, py)
                return

            # Hit test annotations
            hit_id = self.hit_test(px, py)
            if hit_id:
                self.select_annotation(hit_id)
                # Start move drag
                self._dragging = True
                self._drag_type = "move"
                self._drag_ann_id = hit_id
                self._drag_start_norm = self.pixel_to_norm(px, py)
                ann = self.get_selected_annotation()
                if ann:
                    self._drag_ann_snapshot = ann.to_dict()
            else:
                self.select_annotation(None)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        px, py = event.x(), event.y()

        if self._panning and self._pan_start:
            dx = px - self._pan_start[0]
            dy = py - self._pan_start[1]
            self._offset_x += dx
            self._offset_y += dy
            self._pan_start = (px, py)
            self.update()
            return

        if self._drawing and self._draw_start:
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            self._draw_current = (nx, ny)
            self.update()
            return

        if self._dragging and self._drag_ann_id:
            nx, ny = self.pixel_to_norm(px, py)
            self._handle_drag(nx, ny)
            self.update()
            return

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        px, py = event.x(), event.y()

        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor if self.tool_mode == "select" else Qt.CrossCursor)
            return

        if event.button() != Qt.LeftButton:
            return

        if self._drawing and self._draw_start and self.tool_mode == "draw_bbox":
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            sx, sy = self._draw_start
            w = abs(nx - sx)
            h = abs(ny - sy)
            # Minimum size check
            if w > 0.01 and h > 0.01:
                cx = (sx + nx) / 2
                cy = (sy + ny) / 2
                # Request class assignment
                self._draw_current = (nx, ny)
                self.class_requested.emit(px, py)
            self._drawing = False
            self._draw_start = None
            self._draw_current = None
            self.update()
            return

        if self._dragging:
            if self._drag_type in ("move", "resize_tl", "resize_tr", "resize_bl", "resize_br"):
                self.annotation_modified.emit(self._drag_ann_id)
            elif self._drag_type == "move_kp":
                self.annotation_modified.emit(self._drag_ann_id)
            self._dragging = False
            self._drag_type = ""
            self._drag_ann_id = None
            self._drag_ann_snapshot = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.ControlModifier:
            # Zoom centered on mouse
            old_nx, old_ny = self.pixel_to_norm(event.x(), event.y())
            if event.angleDelta().y() > 0:
                self._scale = min(self._scale * ZOOM_FACTOR, MAX_SCALE)
            else:
                self._scale = max(self._scale / ZOOM_FACTOR, MIN_SCALE)
            # Adjust offset so point under cursor stays fixed
            new_px = old_nx * self._image_w * self._scale + self._offset_x
            new_py = old_ny * self._image_h * self._scale + self._offset_y
            self._offset_x += event.x() - new_px
            self._offset_y += event.y() - new_py
            self.update()

    def contextMenuEvent(self, event) -> None:
        """Show right-click context menu."""
        px, py = event.x(), event.y()
        hit_id = self.hit_test(px, py)
        if not hit_id:
            return

        self.select_annotation(hit_id)
        ann = self.get_selected_annotation()
        if not ann:
            return

        menu = QMenu(self)

        if ann.confirmed:
            unconfirm = menu.addAction("取消确认")
            unconfirm.triggered.connect(lambda: self._toggle_confirm(ann, False))
        else:
            confirm = menu.addAction("确认")
            confirm.triggered.connect(lambda: self._toggle_confirm(ann, True))

        delete = menu.addAction("删除")
        delete.triggered.connect(lambda: self.annotation_deleted.emit(ann.id))

        menu.exec_(event.globalPos())

    def resizeEvent(self, event: QResizeEvent) -> None:
        if self._image:
            self._fit_to_window()
        super().resizeEvent(event)

    # ── Drag helpers ───────────────────────────────────────────

    def _handle_drag(self, nx: float, ny: float) -> None:
        """Handle ongoing drag operation."""
        if not self._drag_ann_id or not self._drag_start_norm:
            return

        ann = None
        for a in self._annotations:
            if a.id == self._drag_ann_id:
                ann = a
                break
        if ann is None:
            return

        dx = nx - self._drag_start_norm[0]
        dy = ny - self._drag_start_norm[1]

        if self._drag_type == "move" and ann.bbox and self._drag_ann_snapshot:
            orig = self._drag_ann_snapshot
            orig_bbox = orig["bbox"]
            new_cx = orig_bbox[0] + dx
            new_cy = orig_bbox[1] + dy
            w, h = orig_bbox[2], orig_bbox[3]
            # Clamp to image bounds
            new_cx = max(w / 2, min(1.0 - w / 2, new_cx))
            new_cy = max(h / 2, min(1.0 - h / 2, new_cy))
            ann.bbox = (new_cx, new_cy, w, h)
            # Move keypoints by same offset
            if "keypoints" in orig:
                for i, kp_dict in enumerate(orig["keypoints"]):
                    if i < len(ann.keypoints):
                        ann.keypoints[i].x = max(0, min(1, kp_dict["x"] + dx))
                        ann.keypoints[i].y = max(0, min(1, kp_dict["y"] + dy))
            if not ann.confirmed:
                ann.confirmed = True
            self.annotations_changed.emit()

        elif self._drag_type == "move_kp":
            if 0 <= self._drag_kp_idx < len(ann.keypoints):
                ann.keypoints[self._drag_kp_idx].x = max(0.0, min(1.0, nx))
                ann.keypoints[self._drag_kp_idx].y = max(0.0, min(1.0, ny))
                if not ann.confirmed:
                    ann.confirmed = True
                self.annotations_changed.emit()

        elif self._drag_type.startswith("resize_") and ann.bbox and self._drag_ann_snapshot:
            orig_bbox = self._drag_ann_snapshot["bbox"]
            ocx, ocy, ow, oh = orig_bbox
            ox1, oy1 = ocx - ow / 2, ocy - oh / 2
            ox2, oy2 = ocx + ow / 2, ocy + oh / 2

            if "tl" in self._drag_type:
                ox1 = max(0.0, min(ox2 - 0.01, ox1 + dx))
                oy1 = max(0.0, min(oy2 - 0.01, oy1 + dy))
            elif "tr" in self._drag_type:
                ox2 = max(ox1 + 0.01, min(1.0, ox2 + dx))
                oy1 = max(0.0, min(oy2 - 0.01, oy1 + dy))
            elif "bl" in self._drag_type:
                ox1 = max(0.0, min(ox2 - 0.01, ox1 + dx))
                oy2 = max(oy1 + 0.01, min(1.0, oy2 + dy))
            elif "br" in self._drag_type:
                ox2 = max(ox1 + 0.01, min(1.0, ox2 + dx))
                oy2 = max(oy1 + 0.01, min(1.0, oy2 + dy))

            ann.bbox = ((ox1 + ox2) / 2, (oy1 + oy2) / 2, ox2 - ox1, oy2 - oy1)
            if not ann.confirmed:
                ann.confirmed = True
            self.annotations_changed.emit()

    def _hit_test_handle(self, px: float, py: float) -> str | None:
        """Check if pixel pos hits a resize handle on the selected bbox."""
        if not self._selected_id:
            return None
        ann = self.get_selected_annotation()
        if not ann or not ann.bbox:
            return None

        cx, cy, w, h = ann.bbox
        corners = {
            "resize_tl": (cx - w / 2, cy - h / 2),
            "resize_tr": (cx + w / 2, cy - h / 2),
            "resize_bl": (cx - w / 2, cy + h / 2),
            "resize_br": (cx + w / 2, cy + h / 2),
        }
        for handle_name, (nx, ny) in corners.items():
            hpx, hpy = self.norm_to_pixel(nx, ny)
            if abs(px - hpx) <= HANDLE_SIZE + 2 and abs(py - hpy) <= HANDLE_SIZE + 2:
                return handle_name
        return None

    def _hit_test_keypoint(self, px: float, py: float) -> tuple[str, int] | None:
        """Check if pixel pos hits a keypoint. Returns (ann_id, kp_index) or None."""
        for ann in reversed(self._annotations):
            for i, kp in enumerate(ann.keypoints):
                kpx, kpy = self.norm_to_pixel(kp.x, kp.y)
                if abs(px - kpx) <= KEYPOINT_RADIUS + 4 and abs(py - kpy) <= KEYPOINT_RADIUS + 4:
                    return ann.id, i
        return None

    def _toggle_confirm(self, ann: Annotation, confirmed: bool) -> None:
        ann.confirmed = confirmed
        self.annotations_changed.emit()
        self.update()

    # ── Public helpers for external bbox/kp creation ───────────

    def create_bbox_from_draw(self, class_name: str, class_id: int) -> Annotation | None:
        """Create a bbox annotation from the last draw operation."""
        if not self._draw_start or not self._draw_current:
            return None
        sx, sy = self._draw_start
        ex, ey = self._draw_current
        w = abs(ex - sx)
        h = abs(ey - sy)
        if w < 0.01 or h < 0.01:
            return None
        cx = (sx + ex) / 2
        cy = (sy + ey) / 2
        ann = Annotation(
            class_name=class_name,
            class_id=class_id,
            bbox=(cx, cy, w, h),
            confirmed=True,
            source="manual",
        )
        self._annotations.append(ann)
        self.select_annotation(ann.id)
        self.annotation_created.emit(ann)
        self.annotations_changed.emit()
        self._draw_start = None
        self._draw_current = None
        self.update()
        return ann

    def create_keypoint_at(
        self, class_name: str, class_id: int, label: str = "point"
    ) -> Annotation | None:
        """Create a keypoint annotation at the stored draw position."""
        if not self._draw_start:
            return None
        nx, ny = self._draw_start
        kp = Keypoint(x=nx, y=ny, visible=2, label=label)
        ann = Annotation(
            class_name=class_name,
            class_id=class_id,
            keypoints=[kp],
            confirmed=True,
            source="manual",
        )
        self._annotations.append(ann)
        self.select_annotation(ann.id)
        self.annotation_created.emit(ann)
        self.annotations_changed.emit()
        self._draw_start = None
        self.update()
        return ann
```

Write to `src/ui/canvas.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n yolov8 pytest tests/ui/test_canvas.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ui/canvas.py tests/ui/__init__.py tests/ui/test_canvas.py
git commit -m "feat: annotation canvas with image display, zoom/pan, bbox/keypoint rendering"
```

---

## Task 2: Class Picker Popup

**Files:**
- Create: `src/ui/class_picker.py`
- Create: `tests/ui/test_class_picker.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for ClassPickerPopup."""
import pytest
from PyQt5.QtCore import Qt


class TestClassPickerPopup:
    def test_creates_with_classes(self, qapp):
        from src.ui.class_picker import ClassPickerPopup

        picker = ClassPickerPopup(
            classes=["cat", "dog", "bird"],
            colors={"cat": "#a6e3a1", "dog": "#89b4fa", "bird": "#f38ba8"},
        )
        assert picker._list.count() == 3

    def test_default_selection(self, qapp):
        from src.ui.class_picker import ClassPickerPopup

        picker = ClassPickerPopup(
            classes=["cat", "dog"],
            colors={},
            default_class="dog",
        )
        assert picker._list.currentRow() == 1

    def test_default_first_if_no_default(self, qapp):
        from src.ui.class_picker import ClassPickerPopup

        picker = ClassPickerPopup(
            classes=["cat", "dog"],
            colors={},
        )
        assert picker._list.currentRow() == 0

    def test_get_selected_class(self, qapp):
        from src.ui.class_picker import ClassPickerPopup

        picker = ClassPickerPopup(
            classes=["cat", "dog", "bird"],
            colors={},
        )
        picker._list.setCurrentRow(2)
        assert picker.get_selected_class() == "bird"
        assert picker.get_selected_index() == 2

    def test_empty_classes(self, qapp):
        from src.ui.class_picker import ClassPickerPopup

        picker = ClassPickerPopup(classes=[], colors={})
        assert picker._list.count() == 0
        assert picker.get_selected_class() is None
```

Write to `tests/ui/test_class_picker.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n yolov8 pytest tests/ui/test_class_picker.py -v`
Expected: FAIL

- [ ] **Step 3: Implement class picker**

```python
"""Class picker popup for annotation class assignment."""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor


class ClassPickerPopup(QDialog):
    """Popup dialog for selecting an annotation class.

    Shows a list of classes with color indicators.
    User can click or press Enter to confirm selection.
    """

    def __init__(
        self,
        classes: list[str],
        colors: dict[str, str],
        default_class: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("选择类别")
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setMinimumWidth(160)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        header = QLabel("选择类别:")
        header.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 2px;")
        layout.addWidget(header)

        self._list = QListWidget()
        self._list.setMaximumHeight(200)

        default_row = 0
        for i, cls_name in enumerate(classes):
            item = QListWidgetItem(cls_name)
            color = colors.get(cls_name, "#89b4fa")
            item.setForeground(QColor(color))
            self._list.addItem(item)
            if cls_name == default_class:
                default_row = i

        if classes:
            self._list.setCurrentRow(default_row)

        self._list.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self._list)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.accept()
        elif event.key() == Qt.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)

    def get_selected_class(self) -> str | None:
        """Return the selected class name, or None."""
        item = self._list.currentItem()
        return item.text() if item else None

    def get_selected_index(self) -> int:
        """Return the selected class index, or -1."""
        return self._list.currentRow()
```

Write to `src/ui/class_picker.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n yolov8 pytest tests/ui/test_class_picker.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ui/class_picker.py tests/ui/test_class_picker.py
git commit -m "feat: class picker popup for annotation class assignment"
```

---

## Task 3: Run Full Test Suite

- [ ] **Step 1: Run full test suite**

Run: `conda run -n yolov8 pytest -v --tb=short`
Expected: all tests PASS. No regressions.

- [ ] **Step 2: Commit if any fixups needed**
