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

    def test_rescan_images_finds_new_files(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        assert panel._file_list.count() == 3

        img_dir = pm.project_dir / pm.config.image_dir
        img = QImage(100, 80, QImage.Format_RGB32)
        img.fill(QColor(Qt.green))
        img.save(str(img_dir / "img_new.png"), "PNG")

        added = panel.rescan_images()
        assert added == 1
        assert panel._file_list.count() == 4

    def test_rescan_images_returns_zero_when_nothing_new(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        assert panel.rescan_images() == 0

    def test_rescan_images_returns_zero_when_no_project(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        panel = LabelPanel(config_path=tmp_path / "config.json")
        assert panel.rescan_images() == 0

    def test_refresh_button_disabled_initially(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        panel = LabelPanel(config_path=tmp_path / "config.json")
        assert panel._refresh_btn.isEnabled() is False

    def test_refresh_button_enabled_after_set_project(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        assert panel._refresh_btn.isEnabled() is True

    def test_refresh_button_click_finds_new_images(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        msgs: list[str] = []
        panel.status_changed.connect(msgs.append)
        panel.set_project(pm)

        img_dir = pm.project_dir / pm.config.image_dir
        img = QImage(100, 80, QImage.Format_RGB32)
        img.fill(QColor(Qt.green))
        img.save(str(img_dir / "img_new.png"), "PNG")

        panel._refresh_btn.click()

        assert panel._file_list.count() == 4
        assert any("发现 1 张新图片" in m for m in msgs)

    def test_refresh_button_click_zero_message(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        msgs: list[str] = []
        panel.status_changed.connect(msgs.append)
        panel.set_project(pm)

        panel._refresh_btn.click()

        assert any("未发现新图片" in m for m in msgs)

    def test_f5_triggers_rescan(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel
        from PyQt5.QtCore import QEvent
        from PyQt5.QtGui import QKeyEvent

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)

        img_dir = pm.project_dir / pm.config.image_dir
        img = QImage(100, 80, QImage.Format_RGB32)
        img.fill(QColor(Qt.magenta))
        img.save(str(img_dir / "img_f5.png"), "PNG")

        ev = QKeyEvent(QEvent.KeyPress, Qt.Key_F5, Qt.NoModifier)
        panel.keyPressEvent(ev)

        assert panel._file_list.count() == 4

    def test_rescan_images_updates_project_stats_total_images(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        assert panel._ann_panel._project_total_label.text() == "总图片: 3"

        img_dir = pm.project_dir / pm.config.image_dir
        img = QImage(100, 80, QImage.Format_RGB32)
        img.fill(QColor(Qt.yellow))
        img.save(str(img_dir / "img_new.png"), "PNG")

        panel.rescan_images()

        assert panel._ann_panel._project_total_label.text() == "总图片: 4"

    def test_dropped_images_update_project_stats_total_images(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        assert panel._ann_panel._project_total_label.text() == "总图片: 3"

        external = tmp_path / "external.png"
        img = QImage(100, 80, QImage.Format_RGB32)
        img.fill(QColor(Qt.green))
        img.save(str(external), "PNG")

        panel._on_images_dropped([external])

        assert panel._file_list.count() == 4
        assert panel._ann_panel._project_total_label.text() == "总图片: 4"
