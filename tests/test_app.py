"""Tests for MainWindow."""
import json
import pytest
from pathlib import Path


def _create_project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "images").mkdir()
    (proj / "labels").mkdir()
    config = {
        "name": "test", "image_dir": "images", "label_dir": "labels",
        "classes": ["cat", "dog"], "version": "1.0", "created_at": "2026-01-01",
        "class_colors": {}, "keypoint_templates": {},
        "default_model": "", "auto_label_conf": 0.5, "auto_label_iou": 0.45,
    }
    (proj / "project.json").write_text(json.dumps(config), encoding="utf-8")
    return proj


class TestMainWindow:
    def test_window_creates(self, qapp, tmp_path):
        from src.app import MainWindow

        win = MainWindow(config_path=tmp_path / "config.json")
        assert win.windowTitle() == "AutoLabel V3"
        win.close()

    def test_has_tab_widget(self, qapp, tmp_path):
        from src.app import MainWindow

        win = MainWindow(config_path=tmp_path / "config.json")
        assert win.tab_widget is not None
        assert win.tab_widget.count() >= 1  # at least welcome tab
        win.close()

    def test_has_status_bar(self, qapp, tmp_path):
        from src.app import MainWindow

        win = MainWindow(config_path=tmp_path / "config.json")
        sb = win.statusBar()
        assert sb is not None
        win.close()

    def test_has_menu_bar(self, qapp, tmp_path):
        from src.app import MainWindow

        win = MainWindow(config_path=tmp_path / "config.json")
        mb = win.menuBar()
        assert mb is not None
        # Check key menus exist
        menus = [a.text() for a in mb.actions()]
        assert any("文件" in m for m in menus)
        win.close()

    def test_open_project_sets_title(self, qapp, tmp_path):
        from src.app import MainWindow
        from src.core.project import ProjectManager

        pm = ProjectManager.create(tmp_path / "proj", "test_proj", classes=["cat"])
        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)
        assert "test_proj" in win.windowTitle()
        win.close()

    def test_welcome_page_is_first_tab(self, qapp, tmp_path):
        from src.app import MainWindow

        win = MainWindow(config_path=tmp_path / "config.json")
        assert win.tab_widget.tabText(0) == "欢迎"
        win.close()

    def test_open_project_creates_panels(self, qapp, tmp_path):
        from src.app import MainWindow
        from src.core.project import ProjectManager

        proj = _create_project(tmp_path)
        pm = ProjectManager.open(proj)
        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)
        assert win._label_panel is not None
        assert win._train_panel is not None
        assert win._model_panel is not None
        assert win.tab_widget.count() == 4  # welcome + label + train + model
        win.close()

    def test_open_project_shows_status(self, qapp, tmp_path):
        from src.app import MainWindow
        from src.core.project import ProjectManager

        proj = _create_project(tmp_path)
        pm = ProjectManager.open(proj)
        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)
        status = win._status_label.text()
        assert "test" in status
        assert "类别" in status
        win.close()

    def test_close_saves_config(self, qapp, tmp_path):
        from src.app import MainWindow

        config_path = tmp_path / "config.json"
        win = MainWindow(config_path=config_path)
        win.close()
        # Config should have been saved
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert "window_geometry" in data

    def test_tab_switch_to_label_triggers_rescan(self, qapp, tmp_path):
        from src.app import MainWindow
        from src.core.project import ProjectManager
        from PyQt5.QtGui import QImage, QColor
        from PyQt5.QtCore import Qt

        pm = ProjectManager.create(tmp_path / "proj", "t", classes=["a"])
        img_dir = pm.project_dir / pm.config.image_dir
        for i in range(2):
            img = QImage(40, 40, QImage.Format_RGB32)
            img.fill(QColor(Qt.red))
            img.save(str(img_dir / f"i{i}.png"), "PNG")

        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)
        assert win._label_panel is not None
        assert win._label_panel._file_list.count() == 2

        img = QImage(40, 40, QImage.Format_RGB32)
        img.fill(QColor(Qt.blue))
        img.save(str(img_dir / "new.png"), "PNG")

        welcome_idx = win.tab_widget.indexOf(win._welcome)
        label_idx = win.tab_widget.indexOf(win._label_panel)
        win.tab_widget.setCurrentIndex(welcome_idx)
        win.tab_widget.setCurrentIndex(label_idx)

        assert win._label_panel._file_list.count() == 3
        assert "发现 1 张新图片" in win._status_label.text()
        win.close()

    def test_tab_switch_to_label_no_message_when_nothing_new(self, qapp, tmp_path):
        from src.app import MainWindow
        from src.core.project import ProjectManager
        from PyQt5.QtGui import QImage, QColor
        from PyQt5.QtCore import Qt

        pm = ProjectManager.create(tmp_path / "proj", "t", classes=["a"])
        img_dir = pm.project_dir / pm.config.image_dir
        img = QImage(40, 40, QImage.Format_RGB32)
        img.fill(QColor(Qt.red))
        img.save(str(img_dir / "i0.png"), "PNG")

        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)
        win._status_label.setText("sentinel")

        welcome_idx = win.tab_widget.indexOf(win._welcome)
        label_idx = win.tab_widget.indexOf(win._label_panel)
        win.tab_widget.setCurrentIndex(welcome_idx)
        win.tab_widget.setCurrentIndex(label_idx)

        assert win._status_label.text() == "sentinel"
        win.close()

    def test_new_project_refreshes_welcome_recent_list(self, qapp, tmp_path, monkeypatch):
        from src.app import MainWindow
        from src.core.project import ProjectManager

        pm = ProjectManager.create(tmp_path / "proj", "test_proj", classes=["cat"])
        win = MainWindow(config_path=tmp_path / "config.json")
        assert win._welcome.recent_list.count() == 0

        def fake_create_project():
            win._app_config.add_recent_project(str(pm.project_dir))
            return pm

        monkeypatch.setattr(win._project_ctrl, "create_project", fake_create_project)

        win._on_new_project()

        assert win._welcome.recent_list.count() == 1
        assert win._welcome.recent_list.item(0).text() == str(pm.project_dir)
        win.close()

    def test_open_project_shows_project_dir_in_status_bar_left(self, qapp, tmp_path):
        from src.app import MainWindow
        from src.core.project import ProjectManager

        pm = ProjectManager.create(tmp_path / "proj", "test_proj", classes=["cat"])
        win = MainWindow(config_path=tmp_path / "config.json")

        win.open_project(pm)

        assert str(pm.project_dir) in win._project_dir_label.text()
        assert win._project_dir_label.toolTip() == str(pm.project_dir)
        win.close()

    def test_controllers_initialized(self, qapp, tmp_path):
        from src.app import MainWindow

        win = MainWindow(config_path=tmp_path / "config.json")
        assert win._project_ctrl is not None
        assert win._model_ctrl is not None
        assert win._train_ctrl is not None
        win.close()
