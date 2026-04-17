"""Global application configuration (~/.autolabel/config.json)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    """Global app settings persisted across sessions."""

    recent_projects: list[str] = field(default_factory=list)
    theme: str = "dark"
    auto_save: bool = True
    default_conf_threshold: float = 0.5
    default_iou_threshold: float = 0.45
    overlap_iou_threshold: float = 0.5
    window_geometry: dict[str, int] = field(
        default_factory=lambda: {"x": 100, "y": 100, "width": 1400, "height": 900}
    )

    def to_dict(self) -> dict:
        return {
            "recent_projects": self.recent_projects,
            "theme": self.theme,
            "auto_save": self.auto_save,
            "default_conf_threshold": self.default_conf_threshold,
            "default_iou_threshold": self.default_iou_threshold,
            "overlap_iou_threshold": self.overlap_iou_threshold,
            "window_geometry": self.window_geometry,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AppConfig:
        return cls(
            recent_projects=d.get("recent_projects", []),
            theme=d.get("theme", "dark"),
            auto_save=d.get("auto_save", True),
            default_conf_threshold=d.get("default_conf_threshold", 0.5),
            default_iou_threshold=d.get("default_iou_threshold", 0.45),
            overlap_iou_threshold=d.get("overlap_iou_threshold", 0.5),
            window_geometry=d.get("window_geometry", {"x": 100, "y": 100, "width": 1400, "height": 900}),
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> AppConfig:
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return cls()

    def add_recent_project(self, project_path: str) -> None:
        """Add a project to recent list (most recent first, max 10)."""
        if project_path in self.recent_projects:
            self.recent_projects.remove(project_path)
        self.recent_projects.insert(0, project_path)
        self.recent_projects = self.recent_projects[:10]
