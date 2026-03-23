"""Model registry and management."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ModelInfo:
    """Metadata for a registered model."""

    name: str
    path: str  # relative to project dir
    task: str  # "detect", "classify", "pose"
    base_model: str
    classes: list[str]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metrics: dict[str, float] = field(default_factory=dict)
    trained_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    epochs: int = 0
    dataset_size: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "task": self.task,
            "base_model": self.base_model,
            "classes": self.classes,
            "metrics": self.metrics,
            "trained_at": self.trained_at,
            "epochs": self.epochs,
            "dataset_size": self.dataset_size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelInfo:
        return cls(
            id=d["id"],
            name=d["name"],
            path=d["path"],
            task=d["task"],
            base_model=d["base_model"],
            classes=d["classes"],
            metrics=d.get("metrics", {}),
            trained_at=d.get("trained_at", ""),
            epochs=d.get("epochs", 0),
            dataset_size=d.get("dataset_size", 0),
        )


class ModelRegistry:
    """Manages the model registry (models/registry.json)."""

    def __init__(self, models_dir: Path | str):
        self.models_dir = Path(models_dir)
        self._models: list[ModelInfo] = []

    @property
    def _registry_path(self) -> Path:
        return self.models_dir / "registry.json"

    def load(self) -> None:
        """Load registry from disk."""
        if not self._registry_path.exists():
            self._models = []
            return
        try:
            data = json.loads(self._registry_path.read_text(encoding="utf-8"))
            self._models = [ModelInfo.from_dict(m) for m in data.get("models", [])]
        except (json.JSONDecodeError, KeyError):
            self._models = []

    def save(self) -> None:
        """Save registry to disk."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        data = {"models": [m.to_dict() for m in self._models]}
        self._registry_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def register(self, model_info: ModelInfo) -> None:
        """Register a model."""
        self._models.append(model_info)

    def remove(self, model_id: str) -> None:
        """Remove a model by ID."""
        self._models = [m for m in self._models if m.id != model_id]

    def get(self, model_id: str) -> ModelInfo | None:
        """Get a model by ID."""
        for m in self._models:
            if m.id == model_id:
                return m
        return None

    def list_models(self, task: str | None = None) -> list[ModelInfo]:
        """List models, optionally filtered by task."""
        if task:
            return [m for m in self._models if m.task == task]
        return list(self._models)
