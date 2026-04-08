"""Tests for controllers — project, model, train."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PyQt5.QtWidgets import QWidget


def _create_test_project(tmp_path: Path) -> Path:
    """Create a minimal project for testing."""
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


class TestProjectController:
    def test_open_project(self, qapp, tmp_path):
        from src.controllers.project import ProjectController
        from src.core.config import AppConfig

        proj = _create_test_project(tmp_path)
        config = AppConfig()
        ctrl = ProjectController(config, tmp_path / "cfg.json", QWidget())
        pm = ctrl.open_project(proj)
        assert pm is not None
        assert pm.config.name == "test"

    def test_open_nonexistent_returns_none(self, qapp, tmp_path):
        from src.controllers.project import ProjectController
        from src.core.config import AppConfig

        config = AppConfig()
        ctrl = ProjectController(config, tmp_path / "cfg.json", QWidget())
        with patch("src.controllers.project.QMessageBox"):
            pm = ctrl.open_project(tmp_path / "nonexistent")
        assert pm is None

    def test_backup_manager_initialized_on_open(self, qapp, tmp_path):
        from src.controllers.project import ProjectController
        from src.core.config import AppConfig

        proj = _create_test_project(tmp_path)
        config = AppConfig()
        ctrl = ProjectController(config, tmp_path / "cfg.json", QWidget())
        ctrl.open_project(proj)
        assert ctrl.backup_manager is not None

    def test_create_backup(self, qapp, tmp_path):
        from src.controllers.project import ProjectController
        from src.core.config import AppConfig

        proj = _create_test_project(tmp_path)
        config = AppConfig()
        ctrl = ProjectController(config, tmp_path / "cfg.json", QWidget())
        ctrl.open_project(proj)
        result = ctrl.create_backup()
        assert result is not None
        assert result.exists()

    def test_list_backups_empty(self, qapp, tmp_path):
        from src.controllers.project import ProjectController
        from src.core.config import AppConfig

        proj = _create_test_project(tmp_path)
        config = AppConfig()
        ctrl = ProjectController(config, tmp_path / "cfg.json", QWidget())
        ctrl.open_project(proj)
        assert ctrl.list_backups() == []


class TestTrainController:
    def test_validate_returns_none_when_user_cancels(self, qapp, tmp_path):
        from src.controllers.train import TrainController
        from src.core.project import ProjectManager

        proj = _create_test_project(tmp_path)
        pm = ProjectManager.open(proj)
        ctrl = TrainController(QWidget())

        with patch("src.controllers.train.QMessageBox") as mock_mb:
            mock_mb.question.return_value = mock_mb.No
            mock_mb.Yes = 0x00004000
            mock_mb.No = 0x00010000
            result = ctrl.validate_and_prepare(pm, "detect", 0.2)
            assert result is None

    def test_stop_without_worker(self, qapp):
        from src.controllers.train import TrainController

        ctrl = TrainController(QWidget())
        ctrl.stop()  # should not raise


class TestModelController:
    def test_predict_single_without_model(self, qapp):
        from src.controllers.model import ModelController

        ctrl = ModelController(QWidget())
        with patch("src.controllers.model.QMessageBox"):
            result = ctrl.predict_single(Path("/fake.jpg"), ["cat"])
        assert result == []

    def test_load_model_without_context(self, qapp):
        from src.controllers.model import ModelController

        ctrl = ModelController(QWidget())
        assert ctrl.load_model("fake-id") is False

    def test_delete_model_without_registry(self, qapp):
        from src.controllers.model import ModelController

        ctrl = ModelController(QWidget())
        assert ctrl.delete_model("fake-id") is False
