"""Tests for global app configuration."""
from pathlib import Path

from src.core.config import AppConfig


class TestAppConfig:
    def test_default_values(self):
        cfg = AppConfig()
        assert cfg.recent_projects == []
        assert cfg.theme == "dark"
        assert cfg.auto_save is True
        assert cfg.default_conf_threshold == 0.5

    def test_save_and_load(self, tmp_path):
        config_path = tmp_path / "config.json"
        cfg = AppConfig(recent_projects=["/path/a", "/path/b"])
        cfg.save(config_path)
        loaded = AppConfig.load(config_path)
        assert loaded.recent_projects == ["/path/a", "/path/b"]

    def test_load_missing_returns_default(self, tmp_path):
        cfg = AppConfig.load(tmp_path / "missing.json")
        assert cfg.recent_projects == []

    def test_add_recent_project(self):
        cfg = AppConfig()
        cfg.add_recent_project("/proj/a")
        cfg.add_recent_project("/proj/b")
        cfg.add_recent_project("/proj/a")  # move to front
        assert cfg.recent_projects[0] == "/proj/a"
        assert cfg.recent_projects[1] == "/proj/b"
        assert len(cfg.recent_projects) == 2

    def test_recent_projects_max_10(self):
        cfg = AppConfig()
        for i in range(15):
            cfg.add_recent_project(f"/proj/{i}")
        assert len(cfg.recent_projects) == 10
        assert cfg.recent_projects[0] == "/proj/14"
