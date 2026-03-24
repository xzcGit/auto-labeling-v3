"""Tests for LabelPanel assembly."""
from pathlib import Path

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QColor


def _make_test_project(tmp_path):
    """Create a minimal project with 3 images."""
    from src.core.project import ProjectManager

    pm = ProjectManager.create(tmp_path / "proj", "test", classes=["cat", "dog"])
    img_dir = pm.project_dir / pm.config.image_dir
    for i in range(3):
        img = QImage(100, 80, QImage.Format_RGB32)
        img.fill(QColor(Qt.blue))
        img.save(str(img_dir / f"img{i}.png"), "PNG")
    return pm


class TestLabelPanel:
    def test_creates(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        assert panel._file_list.count() == 3

    def test_has_toolbar(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        assert panel._toolbar is not None

    def test_has_canvas(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        assert panel._canvas is not None

    def test_has_annotation_panel(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        assert panel._ann_panel is not None

    def test_tool_mode_buttons(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        # Should have select, bbox, keypoint tool buttons
        assert panel._btn_select is not None
        assert panel._btn_bbox is not None
        assert panel._btn_keypoint is not None
