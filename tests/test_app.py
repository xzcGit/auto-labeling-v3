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

    def test_controllers_initialized(self, qapp, tmp_path):
        from src.app import MainWindow

        win = MainWindow(config_path=tmp_path / "config.json")
        assert win._project_ctrl is not None
        assert win._model_ctrl is not None
        assert win._train_ctrl is not None
        win.close()
