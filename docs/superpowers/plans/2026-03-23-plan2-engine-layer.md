# Plan 2: Engine Layer & Utilities

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the engine layer (model management, inference, training, dataset preparation) and utility modules (undo/redo) — the business logic that bridges core data models and the UI.

**Architecture:** Model manager handles a JSON registry of trained models. Predictor wraps ultralytics YOLO.predict and converts results to internal Annotation objects. Dataset preparer handles stratified train/val splitting with symlinks and data.yaml generation. Trainer wraps YOLO.train with callback support for metrics reporting. Undo stack is a generic command-pattern implementation. All modules are independent of PyQt (QThread workers belong in the UI plan).

**Tech Stack:** Python 3.9+, ultralytics, pathlib, random, shutil, pytest (with monkeypatch/mock for ultralytics)

---

## File Structure

```
auto-labeling-v3/
├── src/
│   ├── engine/
│   │   ├── __init__.py              # (exists)
│   │   ├── model_manager.py         # ModelInfo dataclass, ModelRegistry CRUD
│   │   ├── predictor.py             # Predictor: wraps YOLO.predict → list[Annotation]
│   │   ├── dataset.py               # DatasetPreparer: stratified split, symlinks, data.yaml
│   │   └── trainer.py               # Trainer: wraps YOLO.train with callbacks
│   └── utils/
│       ├── __init__.py              # (exists)
│       └── undo.py                  # UndoStack: command pattern per-image
├── tests/
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── test_model_manager.py
│   │   ├── test_predictor.py
│   │   ├── test_dataset.py
│   │   └── test_trainer.py
│   └── utils/
│       └── test_undo.py
```

---

## Task 1: Test Infrastructure for Engine

**Files:**
- Create: `tests/engine/__init__.py`

- [ ] **Step 1: Create test directory**

Create empty `tests/engine/__init__.py`.

- [ ] **Step 2: Verify pytest still collects all tests**

Run: `conda run -n yolov8 python -m pytest tests/ --co -q`
Expected: 56 tests collected (from Plan 1), no errors

- [ ] **Step 3: Commit**

```bash
git add tests/engine/__init__.py
git commit -m "chore: add engine test directory"
```

---

## Task 2: Undo/Redo Stack

**Files:**
- Create: `src/utils/undo.py`
- Create: `tests/utils/test_undo.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for undo/redo stack."""
from src.utils.undo import UndoStack


class TestUndoStack:
    def test_initial_state(self):
        stack = UndoStack(max_depth=50)
        assert not stack.can_undo
        assert not stack.can_redo

    def test_push_and_undo(self):
        stack = UndoStack()
        state = {"annotations": []}
        stack.push(state)
        state2 = {"annotations": [{"id": "1"}]}
        stack.push(state2)
        assert stack.can_undo
        restored = stack.undo()
        assert restored == state

    def test_redo_after_undo(self):
        stack = UndoStack()
        s1 = {"a": 1}
        s2 = {"a": 2}
        stack.push(s1)
        stack.push(s2)
        stack.undo()
        assert stack.can_redo
        restored = stack.redo()
        assert restored == s2

    def test_push_clears_redo(self):
        stack = UndoStack()
        stack.push({"a": 1})
        stack.push({"a": 2})
        stack.undo()
        assert stack.can_redo
        stack.push({"a": 3})
        assert not stack.can_redo

    def test_max_depth(self):
        stack = UndoStack(max_depth=3)
        for i in range(5):
            stack.push({"v": i})
        # Should only keep last 3 states
        count = 0
        while stack.can_undo:
            stack.undo()
            count += 1
        assert count == 2  # 3 states means 2 undos (current→prev→prev)

    def test_undo_empty_returns_none(self):
        stack = UndoStack()
        assert stack.undo() is None

    def test_redo_empty_returns_none(self):
        stack = UndoStack()
        assert stack.redo() is None

    def test_clear(self):
        stack = UndoStack()
        stack.push({"a": 1})
        stack.push({"a": 2})
        stack.clear()
        assert not stack.can_undo
        assert not stack.can_redo

    def test_deep_copy_isolation(self):
        """Ensure pushed states are deep-copied to avoid external mutation."""
        stack = UndoStack()
        data = {"items": [1, 2, 3]}
        stack.push(data)
        data["items"].append(4)  # mutate original
        stack.push(data)
        restored = stack.undo()
        assert restored["items"] == [1, 2, 3]  # should be the original
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n yolov8 python -m pytest tests/utils/test_undo.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement undo.py**

```python
"""Undo/redo stack using command pattern with deep-copied snapshots."""
from __future__ import annotations

import copy


class UndoStack:
    """Per-image undo/redo stack with configurable max depth."""

    def __init__(self, max_depth: int = 50):
        self._max_depth = max_depth
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 1

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def push(self, state: dict) -> None:
        """Push a new state snapshot (deep-copied). Clears redo stack."""
        self._undo_stack.append(copy.deepcopy(state))
        self._redo_stack.clear()
        # Enforce max depth
        while len(self._undo_stack) > self._max_depth:
            self._undo_stack.pop(0)

    def undo(self) -> dict | None:
        """Undo to previous state. Returns the restored state or None."""
        if not self.can_undo:
            return None
        current = self._undo_stack.pop()
        self._redo_stack.append(current)
        return copy.deepcopy(self._undo_stack[-1])

    def redo(self) -> dict | None:
        """Redo to next state. Returns the restored state or None."""
        if not self.can_redo:
            return None
        state = self._redo_stack.pop()
        self._undo_stack.append(state)
        return copy.deepcopy(state)

    def clear(self) -> None:
        """Clear all undo/redo history."""
        self._undo_stack.clear()
        self._redo_stack.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n yolov8 python -m pytest tests/utils/test_undo.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add src/utils/undo.py tests/utils/test_undo.py
git commit -m "feat: undo/redo stack with deep-copy isolation"
```

---

## Task 3: Model Manager

**Files:**
- Create: `src/engine/model_manager.py`
- Create: `tests/engine/test_model_manager.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for model registry and management."""
import json
from pathlib import Path

from src.engine.model_manager import ModelInfo, ModelRegistry


class TestModelInfo:
    def test_create(self):
        info = ModelInfo(
            name="yolov8n-custom",
            path="models/yolov8n-custom/best.pt",
            task="detect",
            base_model="yolov8n.pt",
            classes=["person", "car"],
        )
        assert info.name == "yolov8n-custom"
        assert info.task == "detect"
        assert info.id  # should have a UUID

    def test_to_dict_roundtrip(self):
        info = ModelInfo(
            name="test",
            path="models/test/best.pt",
            task="pose",
            base_model="yolov8n-pose.pt",
            classes=["person"],
            metrics={"mAP50": 0.89},
            epochs=100,
            dataset_size=500,
        )
        d = info.to_dict()
        restored = ModelInfo.from_dict(d)
        assert restored.name == "test"
        assert restored.metrics == {"mAP50": 0.89}
        assert restored.id == info.id


class TestModelRegistry:
    def test_create_empty(self, tmp_path):
        registry = ModelRegistry(tmp_path / "models")
        assert registry.list_models() == []

    def test_register_and_list(self, tmp_path):
        registry = ModelRegistry(tmp_path / "models")
        info = ModelInfo(
            name="model-a",
            path="models/model-a/best.pt",
            task="detect",
            base_model="yolov8n.pt",
            classes=["cat"],
        )
        registry.register(info)
        models = registry.list_models()
        assert len(models) == 1
        assert models[0].name == "model-a"

    def test_save_and_load(self, tmp_path):
        registry = ModelRegistry(tmp_path / "models")
        info = ModelInfo(
            name="model-a",
            path="models/model-a/best.pt",
            task="detect",
            base_model="yolov8n.pt",
            classes=["cat"],
        )
        registry.register(info)
        registry.save()
        # Load fresh
        registry2 = ModelRegistry(tmp_path / "models")
        registry2.load()
        assert len(registry2.list_models()) == 1
        assert registry2.list_models()[0].name == "model-a"

    def test_remove_model(self, tmp_path):
        registry = ModelRegistry(tmp_path / "models")
        info = ModelInfo(
            name="model-a",
            path="models/model-a/best.pt",
            task="detect",
            base_model="yolov8n.pt",
            classes=["cat"],
        )
        registry.register(info)
        registry.remove(info.id)
        assert registry.list_models() == []

    def test_get_by_id(self, tmp_path):
        registry = ModelRegistry(tmp_path / "models")
        info = ModelInfo(
            name="model-a",
            path="models/model-a/best.pt",
            task="detect",
            base_model="yolov8n.pt",
            classes=["cat"],
        )
        registry.register(info)
        found = registry.get(info.id)
        assert found is not None
        assert found.name == "model-a"
        assert registry.get("nonexistent") is None

    def test_load_nonexistent_is_empty(self, tmp_path):
        registry = ModelRegistry(tmp_path / "models")
        registry.load()  # no file yet
        assert registry.list_models() == []

    def test_list_by_task(self, tmp_path):
        registry = ModelRegistry(tmp_path / "models")
        registry.register(ModelInfo(name="det", path="a.pt", task="detect", base_model="yolov8n.pt", classes=["a"]))
        registry.register(ModelInfo(name="pose", path="b.pt", task="pose", base_model="yolov8n-pose.pt", classes=["a"]))
        assert len(registry.list_models(task="detect")) == 1
        assert len(registry.list_models(task="pose")) == 1
        assert len(registry.list_models()) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n yolov8 python -m pytest tests/engine/test_model_manager.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement model_manager.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n yolov8 python -m pytest tests/engine/test_model_manager.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add src/engine/model_manager.py tests/engine/test_model_manager.py
git commit -m "feat: model registry with CRUD operations"
```

---

## Task 4: Dataset Preparer

**Files:**
- Create: `src/engine/dataset.py`
- Create: `tests/engine/test_dataset.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for dataset preparation (train/val split + data.yaml generation)."""
import json
import yaml
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation
from src.core.label_io import save_annotation
from src.core.project import ProjectManager
from src.engine.dataset import DatasetPreparer


def _make_project_with_images(tmp_path, num_images=10, classes=None):
    """Helper: create a project with fake images and annotations."""
    classes = classes or ["cat", "dog"]
    pm = ProjectManager.create(
        project_dir=tmp_path / "proj",
        name="test",
        image_dir="images",
        classes=classes,
    )
    img_dir = tmp_path / "proj" / "images"
    for i in range(num_images):
        # Create fake image files
        (img_dir / f"img_{i:03d}.jpg").write_bytes(b"fake")
        # Create annotations
        cls_name = classes[i % len(classes)]
        cls_id = classes.index(cls_name)
        ia = ImageAnnotation(
            image_path=f"img_{i:03d}.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(
                    class_name=cls_name,
                    class_id=cls_id,
                    bbox=(0.5, 0.5, 0.3, 0.4),
                    confirmed=True,
                ),
            ],
        )
        save_annotation(ia, pm.label_path_for(img_dir / f"img_{i:03d}.jpg"))
    return pm


class TestDatasetPreparer:
    def test_prepare_detection_creates_structure(self, tmp_path):
        pm = _make_project_with_images(tmp_path, num_images=10)
        preparer = DatasetPreparer(pm)
        output_dir = tmp_path / "dataset"
        data_yaml = preparer.prepare(output_dir, task="detect", val_ratio=0.2)

        assert data_yaml.exists()
        assert (output_dir / "train" / "images").is_dir()
        assert (output_dir / "train" / "labels").is_dir()
        assert (output_dir / "val" / "images").is_dir()
        assert (output_dir / "val" / "labels").is_dir()

    def test_prepare_detection_data_yaml(self, tmp_path):
        pm = _make_project_with_images(tmp_path, num_images=10)
        preparer = DatasetPreparer(pm)
        output_dir = tmp_path / "dataset"
        data_yaml = preparer.prepare(output_dir, task="detect", val_ratio=0.2)

        data = yaml.safe_load(data_yaml.read_text())
        assert data["nc"] == 2
        assert data["names"] == ["cat", "dog"]
        assert "train" in data
        assert "val" in data

    def test_train_val_split_ratio(self, tmp_path):
        pm = _make_project_with_images(tmp_path, num_images=20)
        preparer = DatasetPreparer(pm)
        output_dir = tmp_path / "dataset"
        preparer.prepare(output_dir, task="detect", val_ratio=0.2)

        train_imgs = list((output_dir / "train" / "images").iterdir())
        val_imgs = list((output_dir / "val" / "images").iterdir())
        total = len(train_imgs) + len(val_imgs)
        assert total == 20
        # Allow some variance due to stratification rounding
        assert len(val_imgs) >= 2
        assert len(val_imgs) <= 6

    def test_uses_symlinks(self, tmp_path):
        pm = _make_project_with_images(tmp_path, num_images=4)
        preparer = DatasetPreparer(pm)
        output_dir = tmp_path / "dataset"
        preparer.prepare(output_dir, task="detect", val_ratio=0.5)

        # Check that image files are symlinks
        for img_path in (output_dir / "train" / "images").iterdir():
            assert img_path.is_symlink()

    def test_only_confirmed_annotations(self, tmp_path):
        """Unconfirmed annotations should be excluded."""
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="test",
            image_dir="images",
            classes=["a"],
        )
        img_dir = tmp_path / "proj" / "images"
        (img_dir / "img.jpg").write_bytes(b"fake")
        ia = ImageAnnotation(
            image_path="img.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(class_name="a", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), confirmed=True),
                Annotation(class_name="a", class_id=0, bbox=(0.2, 0.2, 0.1, 0.1), confirmed=False, source="auto", confidence=0.7),
            ],
        )
        save_annotation(ia, pm.label_path_for(img_dir / "img.jpg"))

        preparer = DatasetPreparer(pm)
        output_dir = tmp_path / "dataset"
        preparer.prepare(output_dir, task="detect", val_ratio=0.0)

        # Find the exported txt label
        label_files = list((output_dir / "train" / "labels").glob("*.txt"))
        assert len(label_files) == 1
        lines = label_files[0].read_text().strip().split("\n")
        assert len(lines) == 1  # only confirmed annotation

    def test_skip_unlabeled_images(self, tmp_path):
        """Images with no confirmed annotations should be excluded."""
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="test",
            image_dir="images",
            classes=["a"],
        )
        img_dir = tmp_path / "proj" / "images"
        (img_dir / "labeled.jpg").write_bytes(b"fake")
        (img_dir / "unlabeled.jpg").write_bytes(b"fake")
        ia = ImageAnnotation(
            image_path="labeled.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(class_name="a", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), confirmed=True),
            ],
        )
        save_annotation(ia, pm.label_path_for(img_dir / "labeled.jpg"))

        preparer = DatasetPreparer(pm)
        output_dir = tmp_path / "dataset"
        preparer.prepare(output_dir, task="detect", val_ratio=0.0)

        train_imgs = list((output_dir / "train" / "images").iterdir())
        assert len(train_imgs) == 1

    def test_prepare_pose_includes_keypoints(self, tmp_path):
        from src.core.annotation import Keypoint
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="test",
            image_dir="images",
            classes=["person"],
        )
        img_dir = tmp_path / "proj" / "images"
        (img_dir / "pose.jpg").write_bytes(b"fake")
        ia = ImageAnnotation(
            image_path="pose.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(
                    class_name="person",
                    class_id=0,
                    bbox=(0.5, 0.5, 0.3, 0.6),
                    keypoints=[
                        Keypoint(0.45, 0.3, 2, "nose"),
                        Keypoint(0.50, 0.35, 1, "left_eye"),
                    ],
                    confirmed=True,
                ),
            ],
        )
        save_annotation(ia, pm.label_path_for(img_dir / "pose.jpg"))

        preparer = DatasetPreparer(pm)
        output_dir = tmp_path / "dataset"
        data_yaml = preparer.prepare(output_dir, task="pose", val_ratio=0.0, kpt_shape=[2, 3])

        label_files = list((output_dir / "train" / "labels").glob("*.txt"))
        assert len(label_files) == 1
        parts = label_files[0].read_text().strip().split()
        # class_id + 4 bbox + 2 kpts * 3 dims = 11
        assert len(parts) == 11

        data = yaml.safe_load(data_yaml.read_text())
        assert data["kpt_shape"] == [2, 3]

    def test_prepare_classify_creates_symlink_dirs(self, tmp_path):
        pm = _make_project_with_images(tmp_path, num_images=6, classes=["cat", "dog"])
        preparer = DatasetPreparer(pm)
        output_dir = tmp_path / "dataset"
        preparer.prepare(output_dir, task="classify", val_ratio=0.0)

        # Classification structure: train/class_name/img.jpg
        assert (output_dir / "train" / "cat").is_dir()
        assert (output_dir / "train" / "dog").is_dir()
        cat_imgs = list((output_dir / "train" / "cat").iterdir())
        dog_imgs = list((output_dir / "train" / "dog").iterdir())
        assert len(cat_imgs) + len(dog_imgs) == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n yolov8 python -m pytest tests/engine/test_dataset.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement dataset.py**

```python
"""Dataset preparation for YOLO training (train/val split, symlinks, data.yaml)."""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import yaml

from src.core.annotation import ImageAnnotation
from src.core.label_io import load_annotation
from src.core.project import ProjectManager


class DatasetPreparer:
    """Prepares a YOLO-compatible dataset from a project."""

    def __init__(self, project_manager: ProjectManager):
        self.pm = project_manager

    def prepare(
        self,
        output_dir: Path | str,
        task: str = "detect",
        val_ratio: float = 0.2,
        seed: int = 42,
        kpt_shape: list[int] | None = None,
    ) -> Path:
        """Prepare dataset and return path to data.yaml.

        Args:
            output_dir: Where to create the dataset structure.
            task: "detect", "classify", or "pose".
            val_ratio: Fraction of images for validation (0.0-1.0).
            seed: Random seed for reproducible splits.
            kpt_shape: [num_keypoints, dims] for pose task.

        Returns:
            Path to the generated data.yaml file.
        """
        output_dir = Path(output_dir)
        classes = self.pm.config.classes

        # Collect labeled images with confirmed annotations
        labeled: list[tuple[Path, ImageAnnotation]] = []
        for img_path in self.pm.list_images():
            label_path = self.pm.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia is None:
                continue
            confirmed = [a for a in ia.annotations if a.confirmed]
            if not confirmed:
                continue
            # Replace with only confirmed
            ia.annotations = confirmed
            labeled.append((img_path, ia))

        # Stratified split by primary class
        train_set, val_set = self._stratified_split(labeled, val_ratio, seed)

        if task == "classify":
            self._export_classify(output_dir, train_set, val_set, classes)
        else:
            self._export_detection_or_pose(output_dir, train_set, val_set, classes, task, kpt_shape)

        # Generate data.yaml
        data_yaml_path = output_dir / "data.yaml"
        data = self._build_data_yaml(output_dir, classes, task, kpt_shape)
        data_yaml_path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
        return data_yaml_path

    def _stratified_split(
        self,
        items: list[tuple[Path, ImageAnnotation]],
        val_ratio: float,
        seed: int,
    ) -> tuple[list[tuple[Path, ImageAnnotation]], list[tuple[Path, ImageAnnotation]]]:
        """Split items into train/val using stratified sampling by primary class."""
        if val_ratio <= 0:
            return items, []
        if val_ratio >= 1:
            return [], items

        # Group by primary class (first annotation's class)
        by_class: dict[str, list[tuple[Path, ImageAnnotation]]] = defaultdict(list)
        for item in items:
            primary_class = item[1].annotations[0].class_name
            by_class[primary_class].append(item)

        rng = random.Random(seed)
        train, val = [], []
        for cls_items in by_class.values():
            rng.shuffle(cls_items)
            n_val = max(1, round(len(cls_items) * val_ratio))
            if n_val >= len(cls_items):
                n_val = max(0, len(cls_items) - 1)
            val.extend(cls_items[:n_val])
            train.extend(cls_items[n_val:])

        return train, val

    def _export_detection_or_pose(
        self,
        output_dir: Path,
        train_set: list[tuple[Path, ImageAnnotation]],
        val_set: list[tuple[Path, ImageAnnotation]],
        classes: list[str],
        task: str,
        kpt_shape: list[int] | None,
    ) -> None:
        """Export to YOLO detection/pose directory structure with symlinks."""
        for split_name, split_data in [("train", train_set), ("val", val_set)]:
            if not split_data:
                continue
            img_dir = output_dir / split_name / "images"
            lbl_dir = output_dir / split_name / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_path, ia in split_data:
                # Symlink image
                link = img_dir / img_path.name
                if not link.exists():
                    link.symlink_to(img_path.resolve())

                # Write YOLO label
                lines = []
                for ann in ia.annotations:
                    if ann.bbox is None:
                        continue
                    cid = classes.index(ann.class_name) if ann.class_name in classes else ann.class_id
                    cx, cy, w, h = ann.bbox
                    parts = [f"{cid}", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
                    if task == "pose" and ann.keypoints:
                        for kp in ann.keypoints:
                            parts.extend([f"{kp.x:.6f}", f"{kp.y:.6f}", f"{kp.visible}"])
                    lines.append(" ".join(parts))
                (lbl_dir / (img_path.stem + ".txt")).write_text(
                    "\n".join(lines) + "\n" if lines else "", encoding="utf-8"
                )

    def _export_classify(
        self,
        output_dir: Path,
        train_set: list[tuple[Path, ImageAnnotation]],
        val_set: list[tuple[Path, ImageAnnotation]],
        classes: list[str],
    ) -> None:
        """Export to YOLO classification directory structure (class_name/img.jpg symlinks)."""
        for split_name, split_data in [("train", train_set), ("val", val_set)]:
            if not split_data:
                continue
            for img_path, ia in split_data:
                # Use image_tags if available, else first annotation class
                if ia.image_tags:
                    cls_name = ia.image_tags[0]
                else:
                    cls_name = ia.annotations[0].class_name
                cls_dir = output_dir / split_name / cls_name
                cls_dir.mkdir(parents=True, exist_ok=True)
                link = cls_dir / img_path.name
                if not link.exists():
                    link.symlink_to(img_path.resolve())

    def _build_data_yaml(
        self,
        output_dir: Path,
        classes: list[str],
        task: str,
        kpt_shape: list[int] | None,
    ) -> dict:
        """Build data.yaml content dict."""
        data = {
            "path": str(output_dir.resolve()),
            "train": "train/images",
            "val": "val/images",
            "nc": len(classes),
            "names": classes,
        }
        if task == "pose" and kpt_shape:
            data["kpt_shape"] = kpt_shape
        return data
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n yolov8 python -m pytest tests/engine/test_dataset.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add src/engine/dataset.py tests/engine/test_dataset.py
git commit -m "feat: dataset preparer with stratified split and symlinks"
```

---

## Task 5: Predictor (Inference Engine)

**Files:**
- Create: `src/engine/predictor.py`
- Create: `tests/engine/test_predictor.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for inference engine (predictor)."""
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.annotation import Annotation
from src.engine.predictor import Predictor


class FakeBox:
    """Mock ultralytics box result."""
    def __init__(self, cls, conf, xywhn):
        self.cls = cls
        self.conf = conf
        self.xywhn = xywhn


class FakeKeypoints:
    """Mock ultralytics keypoints result."""
    def __init__(self, xyn, conf):
        self.xyn = xyn
        self.conf = conf


class FakeResult:
    """Mock ultralytics prediction result."""
    def __init__(self, boxes=None, keypoints=None, orig_shape=(480, 640)):
        self.boxes = boxes
        self.keypoints = keypoints
        self.orig_shape = orig_shape  # (height, width)
        self.names = {0: "person", 1: "car"}


class TestPredictor:
    def test_predict_returns_annotations(self):
        import torch
        # Mock YOLO model
        mock_model = MagicMock()
        boxes = MagicMock()
        boxes.cls = torch.tensor([0, 1])
        boxes.conf = torch.tensor([0.95, 0.80])
        boxes.xywhn = torch.tensor([[0.5, 0.4, 0.3, 0.6], [0.2, 0.3, 0.1, 0.2]])
        result = MagicMock()
        result.boxes = boxes
        result.keypoints = None
        result.orig_shape = (480, 640)
        mock_model.names = {0: "person", 1: "car"}
        mock_model.predict.return_value = [result]

        predictor = Predictor(mock_model)
        annotations = predictor.predict("test.jpg", conf=0.5, iou=0.45)

        assert len(annotations) == 2
        assert annotations[0].class_name == "person"
        assert annotations[0].confidence == 0.95
        assert annotations[0].confirmed is False
        assert annotations[0].source == "auto"
        assert annotations[0].bbox[0] == 0.5  # cx

    def test_predict_filters_by_project_classes(self):
        import torch
        mock_model = MagicMock()
        boxes = MagicMock()
        boxes.cls = torch.tensor([0, 1])
        boxes.conf = torch.tensor([0.90, 0.85])
        boxes.xywhn = torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.2, 0.2, 0.1, 0.1]])
        result = MagicMock()
        result.boxes = boxes
        result.keypoints = None
        result.orig_shape = (480, 640)
        mock_model.names = {0: "person", 1: "car"}
        mock_model.predict.return_value = [result]

        predictor = Predictor(mock_model)
        annotations = predictor.predict(
            "test.jpg", conf=0.5, iou=0.45,
            project_classes=["person"],  # only keep "person"
        )

        assert len(annotations) == 1
        assert annotations[0].class_name == "person"

    def test_predict_with_keypoints(self):
        import torch
        mock_model = MagicMock()
        boxes = MagicMock()
        boxes.cls = torch.tensor([0])
        boxes.conf = torch.tensor([0.92])
        boxes.xywhn = torch.tensor([[0.5, 0.5, 0.3, 0.6]])

        kpts = MagicMock()
        kpts.xyn = torch.tensor([[[0.45, 0.3], [0.50, 0.35]]])
        kpts.conf = torch.tensor([[0.9, 0.8]])

        result = MagicMock()
        result.boxes = boxes
        result.keypoints = kpts
        result.orig_shape = (480, 640)
        mock_model.names = {0: "person"}
        mock_model.predict.return_value = [result]

        predictor = Predictor(mock_model)
        annotations = predictor.predict(
            "test.jpg", conf=0.5, iou=0.45,
            kpt_labels=["nose", "left_eye"],
        )

        assert len(annotations) == 1
        assert len(annotations[0].keypoints) == 2
        assert annotations[0].keypoints[0].label == "nose"
        assert annotations[0].keypoints[0].x == 0.45

    def test_predict_empty_result(self):
        import torch
        mock_model = MagicMock()
        boxes = MagicMock()
        boxes.cls = torch.tensor([])
        boxes.conf = torch.tensor([])
        boxes.xywhn = torch.zeros((0, 4))
        result = MagicMock()
        result.boxes = boxes
        result.keypoints = None
        result.orig_shape = (480, 640)
        mock_model.names = {0: "person"}
        mock_model.predict.return_value = [result]

        predictor = Predictor(mock_model)
        annotations = predictor.predict("test.jpg", conf=0.5, iou=0.45)

        assert annotations == []

    def test_image_size_from_result(self):
        import torch
        mock_model = MagicMock()
        boxes = MagicMock()
        boxes.cls = torch.tensor([0])
        boxes.conf = torch.tensor([0.9])
        boxes.xywhn = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
        result = MagicMock()
        result.boxes = boxes
        result.keypoints = None
        result.orig_shape = (1080, 1920)  # (h, w)
        mock_model.names = {0: "person"}
        mock_model.predict.return_value = [result]

        predictor = Predictor(mock_model)
        annotations, img_size = predictor.predict_with_size("test.jpg", conf=0.5, iou=0.45)

        assert img_size == (1920, 1080)  # (w, h)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n yolov8 python -m pytest tests/engine/test_predictor.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement predictor.py**

```python
"""Inference engine — wraps ultralytics YOLO.predict."""
from __future__ import annotations

import logging
from pathlib import Path

from src.core.annotation import Annotation, Keypoint

logger = logging.getLogger(__name__)


class Predictor:
    """Wraps a YOLO model for inference, converting results to Annotations."""

    def __init__(self, model):
        """Initialize with a loaded ultralytics YOLO model instance."""
        self.model = model

    def predict(
        self,
        image_path: str | Path,
        conf: float = 0.5,
        iou: float = 0.45,
        project_classes: list[str] | None = None,
        kpt_labels: list[str] | None = None,
    ) -> list[Annotation]:
        """Run inference and return list of Annotations.

        Args:
            image_path: Path to the image.
            conf: Confidence threshold.
            iou: IoU threshold for NMS.
            project_classes: If provided, only keep detections whose class
                name is in this list.
            kpt_labels: Labels for keypoints (for pose models).

        Returns:
            List of Annotation objects (unconfirmed, source="auto").
        """
        annotations, _ = self._run(image_path, conf, iou, project_classes, kpt_labels)
        return annotations

    def predict_with_size(
        self,
        image_path: str | Path,
        conf: float = 0.5,
        iou: float = 0.45,
        project_classes: list[str] | None = None,
        kpt_labels: list[str] | None = None,
    ) -> tuple[list[Annotation], tuple[int, int]]:
        """Run inference and return annotations + image size (w, h)."""
        return self._run(image_path, conf, iou, project_classes, kpt_labels)

    def _run(
        self,
        image_path: str | Path,
        conf: float,
        iou: float,
        project_classes: list[str] | None,
        kpt_labels: list[str] | None,
    ) -> tuple[list[Annotation], tuple[int, int]]:
        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            verbose=False,
        )
        if not results:
            return [], (0, 0)

        result = results[0]
        h, w = result.orig_shape
        img_size = (w, h)
        names = self.model.names
        annotations = []

        boxes = result.boxes
        if boxes is None or len(boxes.cls) == 0:
            return [], img_size

        has_kpts = result.keypoints is not None

        for i in range(len(boxes.cls)):
            cls_id = int(boxes.cls[i].item())
            confidence = round(float(boxes.conf[i].item()), 4)
            class_name = names.get(cls_id, str(cls_id))

            # Filter by project classes
            if project_classes and class_name not in project_classes:
                continue

            cx = round(float(boxes.xywhn[i][0].item()), 6)
            cy = round(float(boxes.xywhn[i][1].item()), 6)
            bw = round(float(boxes.xywhn[i][2].item()), 6)
            bh = round(float(boxes.xywhn[i][3].item()), 6)

            keypoints = []
            if has_kpts and result.keypoints.xyn is not None:
                kpts_xy = result.keypoints.xyn[i]
                kpts_conf = result.keypoints.conf[i] if result.keypoints.conf is not None else None
                for j in range(len(kpts_xy)):
                    kx = round(float(kpts_xy[j][0].item()), 6)
                    ky = round(float(kpts_xy[j][1].item()), 6)
                    kc = float(kpts_conf[j].item()) if kpts_conf is not None else 1.0
                    visible = 2 if kc > 0.5 else (1 if kc > 0 else 0)
                    label = kpt_labels[j] if kpt_labels and j < len(kpt_labels) else f"kp_{j}"
                    keypoints.append(Keypoint(x=kx, y=ky, visible=visible, label=label))

            # Resolve class_id relative to project classes if provided
            if project_classes and class_name in project_classes:
                resolved_id = project_classes.index(class_name)
            else:
                resolved_id = cls_id

            annotations.append(Annotation(
                class_name=class_name,
                class_id=resolved_id,
                bbox=(cx, cy, bw, bh),
                keypoints=keypoints,
                confidence=confidence,
                confirmed=False,
                source="auto",
            ))

        return annotations, img_size
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n yolov8 python -m pytest tests/engine/test_predictor.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add src/engine/predictor.py tests/engine/test_predictor.py
git commit -m "feat: predictor wrapping YOLO inference with class filtering"
```

---

## Task 6: Trainer

**Files:**
- Create: `src/engine/trainer.py`
- Create: `tests/engine/test_trainer.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for training engine."""
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from src.engine.trainer import TrainConfig, Trainer


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig(
            data_yaml="/path/data.yaml",
            model="yolov8n.pt",
            task="detect",
        )
        assert cfg.epochs == 100
        assert cfg.batch == 16
        assert cfg.imgsz == 640
        assert cfg.device == ""
        assert cfg.optimizer == "auto"

    def test_to_train_args(self):
        cfg = TrainConfig(
            data_yaml="/path/data.yaml",
            model="yolov8n.pt",
            task="detect",
            epochs=50,
            batch=8,
            project="/out",
            name="run1",
        )
        args = cfg.to_train_args()
        assert args["data"] == "/path/data.yaml"
        assert args["epochs"] == 50
        assert args["batch"] == 8
        assert args["project"] == "/out"
        assert args["name"] == "run1"

    def test_to_train_args_excludes_empty(self):
        cfg = TrainConfig(
            data_yaml="/path/data.yaml",
            model="yolov8n.pt",
            task="detect",
        )
        args = cfg.to_train_args()
        # Empty device should not be in args (let ultralytics auto-detect)
        assert "device" not in args


class TestTrainer:
    def test_train_calls_yolo(self):
        mock_yolo_cls = MagicMock()
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        mock_model.train.return_value = MagicMock()

        cfg = TrainConfig(
            data_yaml="/data.yaml",
            model="yolov8n.pt",
            task="detect",
            epochs=10,
            project="/out",
            name="test",
        )

        trainer = Trainer(yolo_cls=mock_yolo_cls)
        trainer.train(cfg)

        mock_yolo_cls.assert_called_once_with("yolov8n.pt")
        mock_model.train.assert_called_once()
        train_kwargs = mock_model.train.call_args[1]
        assert train_kwargs["data"] == "/data.yaml"
        assert train_kwargs["epochs"] == 10

    def test_train_with_callback(self):
        mock_yolo_cls = MagicMock()
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        mock_model.train.return_value = MagicMock()

        callback_data = []

        def on_epoch(metrics: dict):
            callback_data.append(metrics)

        cfg = TrainConfig(
            data_yaml="/data.yaml",
            model="yolov8n.pt",
            task="detect",
            epochs=5,
        )

        trainer = Trainer(yolo_cls=mock_yolo_cls)
        trainer.train(cfg, on_epoch_end=on_epoch)

        # Verify callback was registered
        mock_model.add_callback.assert_called()

    def test_train_resume(self):
        mock_yolo_cls = MagicMock()
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        mock_model.train.return_value = MagicMock()

        cfg = TrainConfig(
            data_yaml="/data.yaml",
            model="/out/test/weights/last.pt",
            task="detect",
            resume=True,
        )

        trainer = Trainer(yolo_cls=mock_yolo_cls)
        trainer.train(cfg)

        train_kwargs = mock_model.train.call_args[1]
        assert train_kwargs["resume"] is True

    def test_get_best_metrics(self):
        mock_yolo_cls = MagicMock()
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        # Simulate trainer.metrics after training
        mock_model.trainer = MagicMock()
        mock_model.trainer.best_fitness = 0.85
        mock_model.trainer.metrics = {
            "metrics/mAP50(B)": 0.89,
            "metrics/mAP50-95(B)": 0.67,
        }
        mock_model.train.return_value = MagicMock()

        cfg = TrainConfig(
            data_yaml="/data.yaml",
            model="yolov8n.pt",
            task="detect",
        )

        trainer = Trainer(yolo_cls=mock_yolo_cls)
        trainer.train(cfg)
        metrics = trainer.get_best_metrics()

        assert metrics["mAP50"] == 0.89
        assert metrics["mAP50-95"] == 0.67
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n yolov8 python -m pytest tests/engine/test_trainer.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement trainer.py**

```python
"""Training engine — wraps ultralytics YOLO.train."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration."""

    data_yaml: str
    model: str  # base model path or resume checkpoint
    task: str  # "detect", "classify", "pose"

    # Basic hyperparameters
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    device: str = ""  # "" = auto, "0", "cpu", etc.
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1

    # Data augmentation
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0

    # Pose-specific
    pose: float = 12.0
    kobj: float = 1.0

    # Output
    project: str = ""
    name: str = ""
    resume: bool = False

    def to_train_args(self) -> dict:
        """Convert to kwargs dict for YOLO.train(), excluding empty/default values."""
        args = {
            "data": self.data_yaml,
            "epochs": self.epochs,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "warmup_momentum": self.warmup_momentum,
            "warmup_bias_lr": self.warmup_bias_lr,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "perspective": self.perspective,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "copy_paste": self.copy_paste,
        }
        if self.device:
            args["device"] = self.device
        if self.project:
            args["project"] = self.project
        if self.name:
            args["name"] = self.name
        if self.resume:
            args["resume"] = True
        if self.task == "pose":
            args["pose"] = self.pose
            args["kobj"] = self.kobj
        return args


class Trainer:
    """Wraps ultralytics YOLO training with callback support."""

    def __init__(self, yolo_cls=None):
        """Initialize trainer.

        Args:
            yolo_cls: The YOLO class (default: ultralytics.YOLO).
                      Accepts injection for testing.
        """
        if yolo_cls is None:
            from ultralytics import YOLO
            yolo_cls = YOLO
        self._yolo_cls = yolo_cls
        self._model = None

    def train(
        self,
        config: TrainConfig,
        on_epoch_end: Callable[[dict], None] | None = None,
    ) -> None:
        """Start training.

        Args:
            config: Training configuration.
            on_epoch_end: Optional callback invoked after each epoch with metrics dict.
        """
        self._model = self._yolo_cls(config.model)

        if on_epoch_end:
            def _epoch_callback(trainer_obj):
                metrics = {}
                if hasattr(trainer_obj, "metrics") and trainer_obj.metrics:
                    metrics = dict(trainer_obj.metrics)
                if hasattr(trainer_obj, "loss") and trainer_obj.loss is not None:
                    metrics["train_loss"] = float(trainer_obj.loss.mean().item())
                metrics["epoch"] = trainer_obj.epoch
                on_epoch_end(metrics)

            self._model.add_callback("on_fit_epoch_end", _epoch_callback)

        train_args = config.to_train_args()
        self._model.train(**train_args)

    def get_best_metrics(self) -> dict[str, float]:
        """Extract best metrics after training completes."""
        if self._model is None or not hasattr(self._model, "trainer"):
            return {}

        raw = getattr(self._model.trainer, "metrics", {})
        if not raw:
            return {}

        # Normalize metric names
        result = {}
        for key, value in raw.items():
            clean_key = key.replace("metrics/", "").replace("(B)", "").replace("(P)", "")
            result[clean_key] = round(float(value), 4)
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n yolov8 python -m pytest tests/engine/test_trainer.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add src/engine/trainer.py tests/engine/test_trainer.py
git commit -m "feat: trainer wrapping YOLO.train with callback support"
```

---

## Task 7: Full Test Suite & Integration

**Files:**
- No new files — run all tests, verify everything works together

- [ ] **Step 1: Run the full test suite**

Run: `conda run -n yolov8 python -m pytest tests/ -v --tb=short`
Expected: All tests pass (Plan 1's 56 + Plan 2's new tests)

- [ ] **Step 2: Fix any failures**

Address any issues found during full suite run.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: Plan 2 complete — engine layer with full test coverage"
```

---

## Summary

| Task | Module | Key Deliverables |
|------|--------|-----------------|
| 1 | Test infra | tests/engine/ directory |
| 2 | Undo/Redo | UndoStack with deep-copy isolation, max depth |
| 3 | Model Manager | ModelInfo, ModelRegistry CRUD + JSON persistence |
| 4 | Dataset | DatasetPreparer: stratified split, symlinks, data.yaml (detect/pose/classify) |
| 5 | Predictor | Predictor: YOLO.predict → list[Annotation], class filtering, keypoints |
| 6 | Trainer | TrainConfig, Trainer: YOLO.train with epoch callbacks, metrics extraction |
| 7 | Integration | Full test suite verification |

**Next:** Plan 3 will cover the UI layer (MainWindow, Canvas, panels, dialogs) with PyQt5.
