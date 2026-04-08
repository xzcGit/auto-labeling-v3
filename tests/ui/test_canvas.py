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
        canvas._image_w = 200
        canvas._image_h = 150
        canvas._scale = 1.0
        canvas._offset_x = 0.0
        canvas._offset_y = 0.0

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

        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.4, 0.4))
        canvas.set_annotations([ann])

        # Click inside bbox
        result = canvas.hit_test(100.0, 100.0)
        assert result == ann.id

        # Click outside bbox
        result = canvas.hit_test(10.0, 10.0)
        assert result is None


class TestCanvasLock:
    def test_set_locked(self, qapp):
        from src.ui.canvas import AnnotationCanvas

        canvas = AnnotationCanvas()
        assert canvas._locked is False
        canvas.set_locked(True)
        assert canvas._locked is True
        canvas.set_locked(False)
        assert canvas._locked is False

    def test_clear_resets_lock(self, qapp):
        from src.ui.canvas import AnnotationCanvas

        canvas = AnnotationCanvas()
        canvas.set_locked(True)
        canvas.clear()
        assert canvas._locked is False

    def test_lock_blocks_draw_mode(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from PyQt5.QtGui import QMouseEvent
        from PyQt5.QtCore import QEvent, QPoint

        canvas = AnnotationCanvas()
        canvas._image_w = 200
        canvas._image_h = 200
        canvas._scale = 1.0
        canvas._offset_x = 0.0
        canvas._offset_y = 0.0
        canvas.set_tool_mode("draw_bbox")
        canvas.set_locked(True)

        # Simulate left click — should not start drawing
        event = QMouseEvent(QEvent.MouseButtonPress, QPoint(100, 100),
                            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        canvas.mousePressEvent(event)
        assert canvas._drawing is False


class TestCanvasViewportCulling:
    def test_ann_in_viewport_visible(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation

        canvas = AnnotationCanvas()
        canvas._image_w = 200
        canvas._image_h = 200
        canvas._scale = 1.0
        canvas._offset_x = 0.0
        canvas._offset_y = 0.0

        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.3))
        assert canvas._ann_in_viewport(ann, 0, 0, 200, 200) is True

    def test_ann_in_viewport_outside(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation

        canvas = AnnotationCanvas()
        canvas._image_w = 200
        canvas._image_h = 200
        canvas._scale = 1.0
        canvas._offset_x = -500.0  # image scrolled far left
        canvas._offset_y = 0.0

        # Annotation centered at 0.5 → pixel 100, but offset pushes it to -400
        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.3))
        assert canvas._ann_in_viewport(ann, 0, 0, 200, 200) is False

    def test_keypoint_only_always_in_viewport(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation, Keypoint

        canvas = AnnotationCanvas()
        kp_ann = Annotation(class_name="pt", class_id=0, keypoints=[
            Keypoint(x=0.5, y=0.5, visible=2, label="nose"),
        ])
        # No bbox, should always return True
        assert canvas._ann_in_viewport(kp_ann, 0, 0, 200, 200) is True
