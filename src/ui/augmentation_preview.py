"""Augmentation preview dialog — shows effect of augmentation params on a sample image."""
from __future__ import annotations

import random
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGridLayout,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QColor, QTransform


class AugmentationPreviewDialog(QDialog):
    """Shows augmented versions of a sample image using current parameters."""

    def __init__(self, image_path: Path | str, params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("数据增强预览")
        self.setMinimumSize(700, 500)
        self._image_path = str(image_path)
        self._params = params
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Parameter summary
        param_text = "  |  ".join(
            f"{k}: {v}" for k, v in self._params.items() if v != 0
        )
        summary = QLabel(f"增强参数: {param_text}")
        summary.setStyleSheet("color: #a6adc8; font-size: 11px;")
        summary.setWordWrap(True)
        layout.addWidget(summary)

        # Load original image
        original = QImage(self._image_path)
        if original.isNull():
            layout.addWidget(QLabel("无法加载图片"))
            return

        # Generate augmented versions
        grid = QGridLayout()
        thumb_size = 280

        # Original
        grid.addWidget(self._make_preview("原图", original, thumb_size), 0, 0)

        # Generate 3 random augmented versions
        for i in range(3):
            aug = self._apply_augmentation(original)
            grid.addWidget(self._make_preview(f"增强 #{i+1}", aug, thumb_size), i // 2, (i % 2) + 1 if i < 2 else 0)

        # Reorganize: 2x2 grid
        grid_widget_list = []
        grid_widget_list.append(self._make_preview("原图", original, thumb_size))
        for i in range(3):
            aug = self._apply_augmentation(original)
            grid_widget_list.append(self._make_preview(f"增强 #{i+1}", aug, thumb_size))

        grid2 = QGridLayout()
        for idx, w in enumerate(grid_widget_list):
            grid2.addWidget(w, idx // 2, idx % 2)

        layout.addLayout(grid2)

        # Refresh button
        btn_refresh = QPushButton("重新生成")
        btn_refresh.clicked.connect(self._refresh)
        layout.addWidget(btn_refresh)

    def _refresh(self) -> None:
        """Rebuild the dialog with new random augmentations."""
        # Clear and rebuild
        while self.layout().count() > 0:
            item = self.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())
        self._init_ui()

    def _clear_layout(self, layout) -> None:
        while layout.count() > 0:
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

    def _make_preview(self, title: str, image: QImage, max_size: int) -> QWidget:
        """Create a labeled image preview widget."""
        from PyQt5.QtWidgets import QWidget, QVBoxLayout
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #89b4fa; font-weight: bold;")
        lbl_title.setAlignment(Qt.AlignCenter)
        vl.addWidget(lbl_title)

        pixmap = QPixmap.fromImage(image).scaled(
            max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        lbl_img = QLabel()
        lbl_img.setPixmap(pixmap)
        lbl_img.setAlignment(Qt.AlignCenter)
        vl.addWidget(lbl_img)
        return w

    def _apply_augmentation(self, image: QImage) -> QImage:
        """Apply random augmentations based on current params."""
        result = image.copy()

        # HSV augmentation
        hsv_h = self._params.get("hsv_h", 0)
        hsv_s = self._params.get("hsv_s", 0)
        hsv_v = self._params.get("hsv_v", 0)
        if hsv_h > 0 or hsv_s > 0 or hsv_v > 0:
            result = self._augment_hsv(result, hsv_h, hsv_s, hsv_v)

        # Horizontal flip
        fliplr = self._params.get("fliplr", 0)
        if fliplr > 0 and random.random() < fliplr:
            result = result.mirrored(True, False)

        # Vertical flip
        flipud = self._params.get("flipud", 0)
        if flipud > 0 and random.random() < flipud:
            result = result.mirrored(False, True)

        # Rotation
        degrees = self._params.get("degrees", 0)
        if degrees > 0:
            angle = random.uniform(-degrees, degrees)
            transform = QTransform().rotate(angle)
            result = result.transformed(transform, Qt.SmoothTransformation)

        return result

    def _augment_hsv(self, image: QImage, h_gain: float, s_gain: float, v_gain: float) -> QImage:
        """Apply HSV color jitter to image."""
        result = image.convertToFormat(QImage.Format_RGB32)
        h_delta = random.uniform(-h_gain, h_gain) * 180
        s_factor = random.uniform(max(0, 1 - s_gain), 1 + s_gain)
        v_factor = random.uniform(max(0, 1 - v_gain), 1 + v_gain)

        w, h = result.width(), result.height()
        # Sample-based approach for performance (process every 4th pixel for preview)
        for y in range(0, h, 2):
            for x in range(0, w, 2):
                c = QColor(result.pixel(x, y))
                hue = c.hueF() * 360 + h_delta
                sat = max(0, min(1, c.saturationF() * s_factor))
                val = max(0, min(1, c.valueF() * v_factor))
                c.setHsvF((hue % 360) / 360, sat, val)
                rgb = c.rgb()
                result.setPixel(x, y, rgb)
                if x + 1 < w:
                    result.setPixel(x + 1, y, rgb)
                if y + 1 < h:
                    result.setPixel(x, y + 1, rgb)
                    if x + 1 < w:
                        result.setPixel(x + 1, y + 1, rgb)
        return result
