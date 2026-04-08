# Plan 1: Core Data Layer Implementation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundational data models, project management, label I/O, and format converters — the pure-Python core that all UI and engine layers depend on.

**Architecture:** Dataclass-based annotation models with JSON serialization. Project manager handles directory structure and config. Label I/O reads/writes the internal JSON format. Format converters (YOLO, COCO, labelme) import/export through the common Annotation model. All code is pure Python with no UI or ML dependencies, fully testable with pytest.

**Tech Stack:** Python 3.10+, dataclasses, uuid, json, pathlib, pytest

---

## File Structure

```
auto-labeling-v3/
├── main.py                          # Entry point (placeholder for now)
├── requirements.txt                 # Dependencies
├── pyproject.toml                   # Project config + pytest settings
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── annotation.py            # Keypoint, Annotation, ImageAnnotation dataclasses
│   │   ├── project.py               # ProjectConfig, ProjectManager
│   │   ├── config.py                # AppConfig (global ~/.autolabel/config.json)
│   │   ├── label_io.py              # Read/write internal JSON format
│   │   └── formats/
│   │       ├── __init__.py
│   │       ├── yolo.py              # YOLO txt + data.yaml import/export
│   │       ├── coco.py              # COCO JSON import/export
│   │       └── labelme.py           # labelme JSON import/export
│   ├── engine/                      # (Plan 2)
│   │   └── __init__.py
│   ├── ui/                          # (Plan 3-4)
│   │   └── __init__.py
│   └── utils/
│       ├── __init__.py
│       └── colors.py                # Catppuccin color palette + auto-assignment
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Shared fixtures
│   ├── core/
│   │   ├── __init__.py
│   │   ├── test_annotation.py
│   │   ├── test_project.py
│   │   ├── test_config.py
│   │   ├── test_label_io.py
│   │   └── formats/
│   │       ├── __init__.py
│   │       ├── test_yolo.py
│   │       ├── test_coco.py
│   │       └── test_labelme.py
│   └── utils/
│       ├── __init__.py
│       └── test_colors.py
└── resources/
    └── icons/                       # (empty for now)
```

---

## Task 1: Project Skeleton & Tooling

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `main.py`
- Create: all `__init__.py` files

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "auto-labeling-v3"
version = "0.1.0"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Create requirements.txt**

```
PyQt5>=5.15
pyqtgraph>=0.13
ultralytics==8.2.69
pyyaml>=6.0
```

- [ ] **Step 3: Create main.py placeholder**

```python
"""AutoLabel V3 — Entry point."""
import sys


def main():
    print("AutoLabel V3 — not yet implemented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Create all directory and __init__.py files**

Create these empty files:
- `src/__init__.py`
- `src/core/__init__.py`
- `src/core/formats/__init__.py`
- `src/engine/__init__.py`
- `src/ui/__init__.py`
- `src/utils/__init__.py`
- `tests/__init__.py`
- `tests/core/__init__.py`
- `tests/core/formats/__init__.py`
- `tests/utils/__init__.py`
- `resources/icons/.gitkeep`

- [ ] **Step 5: Create tests/conftest.py with shared fixtures**

```python
"""Shared test fixtures."""
import json
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project directory structure."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    return tmp_path


@pytest.fixture
def sample_project_config():
    """Return a minimal project config dict."""
    return {
        "name": "test_project",
        "image_dir": "images",
        "label_dir": "labels",
        "classes": ["person", "car", "dog"],
        "class_colors": {},
        "keypoint_templates": {},
        "default_model": "",
        "auto_label_conf": 0.5,
        "auto_label_iou": 0.45,
        "created_at": "2026-03-23T10:00:00",
        "version": "1.0",
    }
```

- [ ] **Step 6: Verify pytest runs with no tests collected**

Run: `cd /home/xzc/projects/auto-labeling-v3 && python -m pytest --co -q`
Expected: "no tests ran" or empty collection, exit 0 (no import errors)

- [ ] **Step 7: Commit**

```bash
git init
git add -A
git commit -m "chore: project skeleton with pytest config"
```

---

## Task 2: Color Palette Utility

**Files:**
- Create: `src/utils/colors.py`
- Create: `tests/utils/test_colors.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for Catppuccin color palette utility."""
from src.utils.colors import CATPPUCCIN_PALETTE, assign_color


class TestCatppuccinPalette:
    def test_palette_has_20_colors(self):
        assert len(CATPPUCCIN_PALETTE) == 20

    def test_colors_are_hex(self):
        for color in CATPPUCCIN_PALETTE:
            assert color.startswith("#")
            assert len(color) == 7

    def test_assign_color_by_index(self):
        assert assign_color(0) == CATPPUCCIN_PALETTE[0]
        assert assign_color(5) == CATPPUCCIN_PALETTE[5]

    def test_assign_color_wraps_around(self):
        assert assign_color(20) == CATPPUCCIN_PALETTE[0]
        assert assign_color(25) == CATPPUCCIN_PALETTE[5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/utils/test_colors.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement**

```python
"""Catppuccin Mocha color palette and auto-assignment."""

# 20 distinct colors from Catppuccin Mocha
CATPPUCCIN_PALETTE = [
    "#a6e3a1",  # green
    "#89b4fa",  # blue
    "#f38ba8",  # red
    "#fab387",  # peach
    "#cba6f7",  # mauve
    "#f9e2af",  # yellow
    "#94e2d5",  # teal
    "#f5c2e7",  # pink
    "#89dceb",  # sky
    "#eba0ac",  # maroon
    "#74c7ec",  # sapphire
    "#b4befe",  # lavender
    "#a6adc8",  # subtext0
    "#f2cdcd",  # flamingo
    "#e6c384",  # gold (custom)
    "#c6a0f6",  # violet (custom)
    "#8caaee",  # blue2 (custom)
    "#e78284",  # red2 (custom)
    "#a5adce",  # overlay (custom)
    "#81c8be",  # teal2 (custom)
]


def assign_color(index: int) -> str:
    """Assign a color from the palette by index, wrapping around."""
    return CATPPUCCIN_PALETTE[index % len(CATPPUCCIN_PALETTE)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/utils/test_colors.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/utils/colors.py tests/utils/test_colors.py
git commit -m "feat: add Catppuccin color palette utility"
```

---

## Task 3: Annotation Data Models

**Files:**
- Create: `src/core/annotation.py`
- Create: `tests/core/test_annotation.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for annotation data models."""
import uuid

from src.core.annotation import Annotation, ImageAnnotation, Keypoint


class TestKeypoint:
    def test_create_keypoint(self):
        kp = Keypoint(x=0.5, y=0.3, visible=2, label="nose")
        assert kp.x == 0.5
        assert kp.y == 0.3
        assert kp.visible == 2
        assert kp.label == "nose"

    def test_to_dict(self):
        kp = Keypoint(x=0.25, y=0.15, visible=2, label="left_eye")
        d = kp.to_dict()
        assert d == {"x": 0.25, "y": 0.15, "visible": 2, "label": "left_eye"}

    def test_from_dict(self):
        d = {"x": 0.25, "y": 0.15, "visible": 2, "label": "left_eye"}
        kp = Keypoint.from_dict(d)
        assert kp.x == 0.25
        assert kp.label == "left_eye"

    def test_clamp_coordinates(self):
        kp = Keypoint(x=1.2, y=-0.1, visible=2, label="test")
        kp.clamp()
        assert kp.x == 1.0
        assert kp.y == 0.0


class TestAnnotation:
    def test_create_manual_bbox(self):
        ann = Annotation(
            class_name="person",
            class_id=0,
            bbox=(0.5, 0.4, 0.3, 0.6),
        )
        assert ann.class_name == "person"
        assert ann.bbox == (0.5, 0.4, 0.3, 0.6)
        assert ann.keypoints == []
        assert ann.confidence == 1.0
        assert ann.confirmed is True
        assert ann.source == "manual"
        # id should be a valid UUID
        uuid.UUID(ann.id)

    def test_create_auto_bbox(self):
        ann = Annotation(
            class_name="car",
            class_id=1,
            bbox=(0.6, 0.3, 0.25, 0.35),
            confidence=0.87,
            confirmed=False,
            source="auto",
        )
        assert ann.confirmed is False
        assert ann.source == "auto"
        assert ann.confidence == 0.87

    def test_to_dict_roundtrip(self):
        ann = Annotation(
            class_name="person",
            class_id=0,
            bbox=(0.5, 0.4, 0.3, 0.6),
            keypoints=[Keypoint(0.25, 0.15, 2, "nose")],
        )
        d = ann.to_dict()
        restored = Annotation.from_dict(d)
        assert restored.class_name == ann.class_name
        assert restored.bbox == ann.bbox
        assert len(restored.keypoints) == 1
        assert restored.keypoints[0].label == "nose"
        assert restored.id == ann.id

    def test_annotation_without_bbox(self):
        ann = Annotation(
            class_name="point",
            class_id=0,
            bbox=None,
            keypoints=[Keypoint(0.5, 0.5, 2, "center")],
        )
        assert ann.bbox is None
        d = ann.to_dict()
        assert d["bbox"] is None
        restored = Annotation.from_dict(d)
        assert restored.bbox is None

    def test_clamp_bbox(self):
        ann = Annotation(
            class_name="test",
            class_id=0,
            bbox=(0.5, 0.5, 1.2, 0.8),  # width exceeds
        )
        ann.clamp()
        x, y, w, h = ann.bbox
        # After clamp, bbox should not extend beyond [0,1]
        assert x - w / 2 >= 0.0
        assert x + w / 2 <= 1.0
        assert y - h / 2 >= 0.0
        assert y + h / 2 <= 1.0


class TestImageAnnotation:
    def test_create_empty(self):
        ia = ImageAnnotation(
            image_path="img_001.jpg",
            image_size=(1920, 1080),
        )
        assert ia.annotations == []
        assert ia.image_tags == []

    def test_to_dict_roundtrip(self):
        ann = Annotation(class_name="person", class_id=0, bbox=(0.5, 0.4, 0.3, 0.6))
        ia = ImageAnnotation(
            image_path="img_001.jpg",
            image_size=(1920, 1080),
            annotations=[ann],
            image_tags=["outdoor"],
        )
        d = ia.to_dict()
        restored = ImageAnnotation.from_dict(d)
        assert restored.image_path == "img_001.jpg"
        assert restored.image_size == (1920, 1080)
        assert len(restored.annotations) == 1
        assert restored.image_tags == ["outdoor"]

    def test_confirmed_count(self):
        a1 = Annotation(class_name="a", class_id=0, bbox=(0.5, 0.5, 0.1, 0.1), confirmed=True)
        a2 = Annotation(class_name="b", class_id=1, bbox=(0.2, 0.2, 0.1, 0.1), confirmed=False, source="auto", confidence=0.8)
        ia = ImageAnnotation(
            image_path="test.jpg",
            image_size=(100, 100),
            annotations=[a1, a2],
        )
        assert ia.confirmed_count == 1
        assert ia.unconfirmed_count == 1

    def test_status(self):
        # Empty
        ia = ImageAnnotation(image_path="a.jpg", image_size=(100, 100))
        assert ia.status == "unlabeled"

        # All confirmed
        ia.annotations = [
            Annotation(class_name="a", class_id=0, bbox=(0.5, 0.5, 0.1, 0.1), confirmed=True),
        ]
        assert ia.status == "confirmed"

        # Has unconfirmed
        ia.annotations.append(
            Annotation(class_name="b", class_id=1, bbox=(0.2, 0.2, 0.1, 0.1), confirmed=False, source="auto", confidence=0.8),
        )
        assert ia.status == "pending"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/test_annotation.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement annotation.py**

```python
"""Annotation data models for AutoLabel V3."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class Keypoint:
    """A single keypoint with normalized coordinates."""

    x: float
    y: float
    visible: int  # 0=invisible, 1=occluded, 2=visible
    label: str

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "visible": self.visible, "label": self.label}

    @classmethod
    def from_dict(cls, d: dict) -> Keypoint:
        return cls(x=d["x"], y=d["y"], visible=d["visible"], label=d["label"])

    def clamp(self) -> None:
        """Clamp coordinates to [0, 1]."""
        self.x = max(0.0, min(1.0, self.x))
        self.y = max(0.0, min(1.0, self.y))


@dataclass
class Annotation:
    """A single annotation (bbox, keypoints, or both)."""

    class_name: str
    class_id: int
    bbox: tuple[float, float, float, float] | None = None  # (cx, cy, w, h) normalized
    keypoints: list[Keypoint] = field(default_factory=list)
    confidence: float = 1.0
    confirmed: bool = True
    source: str = "manual"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "bbox": list(self.bbox) if self.bbox else None,
            "keypoints": [kp.to_dict() for kp in self.keypoints],
            "confidence": self.confidence,
            "confirmed": self.confirmed,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Annotation:
        bbox = tuple(d["bbox"]) if d.get("bbox") else None
        keypoints = [Keypoint.from_dict(kp) for kp in d.get("keypoints", [])]
        return cls(
            id=d["id"],
            class_name=d["class_name"],
            class_id=d["class_id"],
            bbox=bbox,
            keypoints=keypoints,
            confidence=d.get("confidence", 1.0),
            confirmed=d.get("confirmed", True),
            source=d.get("source", "manual"),
        )

    def clamp(self) -> None:
        """Clamp bbox and keypoints to [0, 1] image bounds.

        Clamps edges to [0, 1] then recomputes center and size,
        preserving as much of the original box as possible.
        """
        if self.bbox:
            cx, cy, w, h = self.bbox
            # Compute edges, clamp to [0, 1], recompute center/size
            x1 = max(0.0, cx - w / 2)
            y1 = max(0.0, cy - h / 2)
            x2 = min(1.0, cx + w / 2)
            y2 = min(1.0, cy + h / 2)
            self.bbox = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        for kp in self.keypoints:
            kp.clamp()


@dataclass
class ImageAnnotation:
    """All annotations for a single image."""

    image_path: str
    image_size: tuple[int, int]  # (width, height)
    annotations: list[Annotation] = field(default_factory=list)
    image_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "image_size": list(self.image_size),
            "image_tags": self.image_tags,
            "annotations": [ann.to_dict() for ann in self.annotations],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ImageAnnotation:
        return cls(
            image_path=d["image_path"],
            image_size=tuple(d["image_size"]),
            annotations=[Annotation.from_dict(a) for a in d.get("annotations", [])],
            image_tags=d.get("image_tags", []),
        )

    @property
    def confirmed_count(self) -> int:
        return sum(1 for a in self.annotations if a.confirmed)

    @property
    def unconfirmed_count(self) -> int:
        return sum(1 for a in self.annotations if not a.confirmed)

    @property
    def status(self) -> str:
        """Return 'unlabeled', 'confirmed', or 'pending'."""
        if not self.annotations:
            return "unlabeled"
        if all(a.confirmed for a in self.annotations):
            return "confirmed"
        return "pending"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/core/test_annotation.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add src/core/annotation.py tests/core/test_annotation.py
git commit -m "feat: annotation data models with serialization"
```

---

## Task 4: Label I/O (Internal JSON Format)

**Files:**
- Create: `src/core/label_io.py`
- Create: `tests/core/test_label_io.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for label I/O (internal JSON format)."""
import json
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation, Keypoint
from src.core.label_io import load_annotation, save_annotation


class TestSaveAnnotation:
    def test_save_creates_json_file(self, tmp_path):
        ia = ImageAnnotation(
            image_path="img_001.jpg",
            image_size=(1920, 1080),
            annotations=[
                Annotation(class_name="person", class_id=0, bbox=(0.5, 0.4, 0.3, 0.6)),
            ],
            image_tags=["outdoor"],
        )
        save_annotation(ia, tmp_path / "img_001.json")
        assert (tmp_path / "img_001.json").exists()

    def test_saved_json_structure(self, tmp_path):
        ia = ImageAnnotation(
            image_path="img_001.jpg",
            image_size=(1920, 1080),
            annotations=[
                Annotation(class_name="person", class_id=0, bbox=(0.5, 0.4, 0.3, 0.6)),
            ],
        )
        path = tmp_path / "img_001.json"
        save_annotation(ia, path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["image_path"] == "img_001.jpg"
        assert data["image_size"] == [1920, 1080]
        assert len(data["annotations"]) == 1
        assert data["annotations"][0]["class_name"] == "person"


class TestLoadAnnotation:
    def test_load_roundtrip(self, tmp_path):
        ia = ImageAnnotation(
            image_path="test.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(
                    class_name="car",
                    class_id=1,
                    bbox=(0.6, 0.3, 0.25, 0.35),
                    keypoints=[Keypoint(0.5, 0.5, 2, "center")],
                    confidence=0.87,
                    confirmed=False,
                    source="auto",
                ),
            ],
            image_tags=["daytime"],
        )
        path = tmp_path / "test.json"
        save_annotation(ia, path)
        loaded = load_annotation(path)
        assert loaded.image_path == "test.jpg"
        assert loaded.annotations[0].confidence == 0.87
        assert loaded.annotations[0].confirmed is False
        assert loaded.annotations[0].keypoints[0].label == "center"
        assert loaded.image_tags == ["daytime"]

    def test_load_nonexistent_returns_none(self, tmp_path):
        result = load_annotation(tmp_path / "missing.json")
        assert result is None

    def test_load_corrupted_returns_none(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json {{{", encoding="utf-8")
        result = load_annotation(path)
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/test_label_io.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement label_io.py**

```python
"""Read/write the internal JSON annotation format."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.core.annotation import ImageAnnotation

logger = logging.getLogger(__name__)


def save_annotation(image_annotation: ImageAnnotation, path: Path | str) -> None:
    """Save an ImageAnnotation to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = image_annotation.to_dict()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_annotation(path: Path | str) -> ImageAnnotation | None:
    """Load an ImageAnnotation from a JSON file. Returns None if file missing or corrupted."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ImageAnnotation.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("Failed to load annotation %s: %s", path, e)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/core/test_label_io.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add src/core/label_io.py tests/core/test_label_io.py
git commit -m "feat: label I/O for internal JSON format"
```

---

## Task 5: Project Configuration & Management

**Files:**
- Create: `src/core/project.py`
- Create: `tests/core/test_project.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for project configuration and management."""
import json
from pathlib import Path

from src.core.project import ProjectConfig, ProjectManager


class TestProjectConfig:
    def test_create_minimal(self):
        cfg = ProjectConfig(
            name="test",
            image_dir="images",
            label_dir="labels",
            classes=["person", "car"],
        )
        assert cfg.name == "test"
        assert cfg.classes == ["person", "car"]
        assert cfg.class_colors == {}
        assert cfg.keypoint_templates == {}
        assert cfg.version == "1.0"

    def test_to_dict_roundtrip(self):
        cfg = ProjectConfig(
            name="test",
            image_dir="images",
            label_dir="labels",
            classes=["a", "b"],
            class_colors={"a": "#ff0000"},
            keypoint_templates={
                "pose": {
                    "labels": ["nose", "eye"],
                    "skeleton": [[0, 1]],
                }
            },
        )
        d = cfg.to_dict()
        restored = ProjectConfig.from_dict(d)
        assert restored.name == cfg.name
        assert restored.classes == cfg.classes
        assert restored.class_colors == {"a": "#ff0000"}
        assert restored.keypoint_templates["pose"]["labels"] == ["nose", "eye"]

    def test_get_class_color_assigned(self):
        cfg = ProjectConfig(
            name="test",
            image_dir="images",
            label_dir="labels",
            classes=["a", "b", "c"],
            class_colors={"a": "#ff0000"},
        )
        assert cfg.get_class_color("a") == "#ff0000"
        # b and c should get auto-assigned from palette
        color_b = cfg.get_class_color("b")
        assert color_b.startswith("#")

    def test_get_class_id(self):
        cfg = ProjectConfig(
            name="test",
            image_dir="images",
            label_dir="labels",
            classes=["person", "car", "dog"],
        )
        assert cfg.get_class_id("person") == 0
        assert cfg.get_class_id("car") == 1
        assert cfg.get_class_id("dog") == 2
        assert cfg.get_class_id("unknown") == -1


class TestProjectManager:
    def test_create_project(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "my_project",
            name="my_project",
            image_dir="images",
            classes=["person", "car"],
        )
        assert (tmp_path / "my_project" / "project.json").exists()
        assert (tmp_path / "my_project" / "images").is_dir()
        assert (tmp_path / "my_project" / "labels").is_dir()
        assert pm.config.name == "my_project"

    def test_open_project(self, tmp_path):
        # Create first
        ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="proj",
            image_dir="images",
            classes=["a"],
        )
        # Open
        pm = ProjectManager.open(tmp_path / "proj")
        assert pm.config.name == "proj"
        assert pm.config.classes == ["a"]

    def test_open_nonexistent_raises(self, tmp_path):
        import pytest
        with pytest.raises(FileNotFoundError):
            ProjectManager.open(tmp_path / "nope")

    def test_save_config(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="proj",
            image_dir="images",
            classes=["a"],
        )
        pm.config.classes.append("b")
        pm.save()
        # Re-open and verify
        pm2 = ProjectManager.open(tmp_path / "proj")
        assert "b" in pm2.config.classes

    def test_list_images(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="proj",
            image_dir="images",
            classes=["a"],
        )
        img_dir = tmp_path / "proj" / "images"
        (img_dir / "a.jpg").write_bytes(b"fake")
        (img_dir / "b.png").write_bytes(b"fake")
        (img_dir / "c.txt").write_bytes(b"fake")  # not an image
        images = pm.list_images()
        names = [p.name for p in images]
        assert "a.jpg" in names
        assert "b.png" in names
        assert "c.txt" not in names

    def test_label_path_for_image(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="proj",
            image_dir="images",
            classes=["a"],
        )
        img = tmp_path / "proj" / "images" / "photo.jpg"
        label_path = pm.label_path_for(img)
        assert label_path == tmp_path / "proj" / "labels" / "photo.json"

    def test_add_class(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="proj",
            image_dir="images",
            classes=["a"],
        )
        pm.add_class("b")
        assert "b" in pm.config.classes
        # Duplicate should not add
        pm.add_class("b")
        assert pm.config.classes.count("b") == 1

    def test_remove_class(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="proj",
            image_dir="images",
            classes=["a", "b"],
        )
        pm.remove_class("a")
        assert "a" not in pm.config.classes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/test_project.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement project.py**

```python
"""Project configuration and management."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.utils.colors import assign_color

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class ProjectConfig:
    """Project configuration stored in project.json."""

    name: str
    image_dir: str  # relative to project dir
    label_dir: str  # relative to project dir
    classes: list[str]
    class_colors: dict[str, str] = field(default_factory=dict)
    keypoint_templates: dict[str, dict] = field(default_factory=dict)
    default_model: str = ""
    auto_label_conf: float = 0.5
    auto_label_iou: float = 0.45
    created_at: str = ""
    version: str = "1.0"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "image_dir": self.image_dir,
            "label_dir": self.label_dir,
            "classes": self.classes,
            "class_colors": self.class_colors,
            "keypoint_templates": self.keypoint_templates,
            "default_model": self.default_model,
            "auto_label_conf": self.auto_label_conf,
            "auto_label_iou": self.auto_label_iou,
            "created_at": self.created_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProjectConfig:
        return cls(
            name=d["name"],
            image_dir=d["image_dir"],
            label_dir=d["label_dir"],
            classes=d["classes"],
            class_colors=d.get("class_colors", {}),
            keypoint_templates=d.get("keypoint_templates", {}),
            default_model=d.get("default_model", ""),
            auto_label_conf=d.get("auto_label_conf", 0.5),
            auto_label_iou=d.get("auto_label_iou", 0.45),
            created_at=d.get("created_at", ""),
            version=d.get("version", "1.0"),
        )

    def get_class_color(self, class_name: str) -> str:
        """Get color for a class. Uses custom color if set, otherwise auto-assigns from palette."""
        if class_name in self.class_colors:
            return self.class_colors[class_name]
        idx = self.classes.index(class_name) if class_name in self.classes else 0
        return assign_color(idx)

    def get_class_id(self, class_name: str) -> int:
        """Get class index. Returns -1 if not found."""
        try:
            return self.classes.index(class_name)
        except ValueError:
            return -1


class ProjectManager:
    """Manages a project directory and its configuration."""

    def __init__(self, project_dir: Path, config: ProjectConfig):
        self.project_dir = project_dir
        self.config = config

    @classmethod
    def create(
        cls,
        project_dir: Path | str,
        name: str,
        image_dir: str = "images",
        classes: list[str] | None = None,
    ) -> ProjectManager:
        """Create a new project."""
        project_dir = Path(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / image_dir).mkdir(exist_ok=True)
        label_dir = "labels"
        (project_dir / label_dir).mkdir(exist_ok=True)

        config = ProjectConfig(
            name=name,
            image_dir=image_dir,
            label_dir=label_dir,
            classes=classes or [],
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        pm = cls(project_dir, config)
        pm.save()
        return pm

    @classmethod
    def open(cls, project_dir: Path | str) -> ProjectManager:
        """Open an existing project."""
        project_dir = Path(project_dir)
        config_path = project_dir / "project.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No project.json found in {project_dir}")
        data = json.loads(config_path.read_text(encoding="utf-8"))
        config = ProjectConfig.from_dict(data)
        return cls(project_dir, config)

    def save(self) -> None:
        """Save project config to project.json."""
        path = self.project_dir / "project.json"
        path.write_text(
            json.dumps(self.config.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def list_images(self) -> list[Path]:
        """List all image files in the image directory, sorted by name."""
        img_dir = self.project_dir / self.config.image_dir
        if not img_dir.exists():
            return []
        return sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

    def label_path_for(self, image_path: Path | str) -> Path:
        """Get the label JSON path for a given image."""
        image_path = Path(image_path)
        return self.project_dir / self.config.label_dir / (image_path.stem + ".json")

    def add_class(self, class_name: str) -> None:
        """Add a class if it doesn't exist."""
        if class_name not in self.config.classes:
            self.config.classes.append(class_name)

    def remove_class(self, class_name: str) -> None:
        """Remove a class."""
        if class_name in self.config.classes:
            self.config.classes.remove(class_name)
            self.config.class_colors.pop(class_name, None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/core/test_project.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add src/core/project.py tests/core/test_project.py
git commit -m "feat: project config and management"
```

---

## Task 6: App Config (Global Settings)

**Files:**
- Create: `src/core/config.py`
- Create: `tests/core/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/test_config.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement config.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/core/test_config.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add src/core/config.py tests/core/test_config.py
git commit -m "feat: global app config with recent projects"
```

---

## Task 7: YOLO Format Import/Export

**Files:**
- Create: `src/core/formats/yolo.py`
- Create: `tests/core/formats/test_yolo.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for YOLO format import/export."""
import yaml
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation, Keypoint
from src.core.formats.yolo import (
    export_yolo_detection,
    import_yolo_detection,
    export_yolo_pose,
    import_yolo_pose,
)


class TestYoloDetectionExport:
    def test_export_single_image(self, tmp_path):
        ia = ImageAnnotation(
            image_path="img.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(class_name="person", class_id=0, bbox=(0.5, 0.4, 0.3, 0.6), confirmed=True),
                Annotation(class_name="car", class_id=1, bbox=(0.2, 0.3, 0.1, 0.2), confirmed=True),
            ],
        )
        out_dir = tmp_path / "output"
        export_yolo_detection([ia], out_dir, classes=["person", "car"])
        txt_path = out_dir / "labels" / "img.txt"
        assert txt_path.exists()
        lines = txt_path.read_text().strip().split("\n")
        assert len(lines) == 2
        parts = lines[0].split()
        assert parts[0] == "0"  # class_id
        assert len(parts) == 5  # id cx cy w h

    def test_export_generates_data_yaml(self, tmp_path):
        ia = ImageAnnotation(image_path="a.jpg", image_size=(100, 100), annotations=[])
        out_dir = tmp_path / "output"
        export_yolo_detection([ia], out_dir, classes=["person", "car"])
        yaml_path = out_dir / "data.yaml"
        assert yaml_path.exists()
        data = yaml.safe_load(yaml_path.read_text())
        assert data["nc"] == 2
        assert data["names"] == ["person", "car"]

    def test_export_only_confirmed(self, tmp_path):
        ia = ImageAnnotation(
            image_path="img.jpg",
            image_size=(100, 100),
            annotations=[
                Annotation(class_name="a", class_id=0, bbox=(0.5, 0.5, 0.1, 0.1), confirmed=True),
                Annotation(class_name="b", class_id=1, bbox=(0.2, 0.2, 0.1, 0.1), confirmed=False, source="auto", confidence=0.8),
            ],
        )
        out_dir = tmp_path / "output"
        export_yolo_detection([ia], out_dir, classes=["a", "b"], only_confirmed=True)
        lines = (out_dir / "labels" / "img.txt").read_text().strip().split("\n")
        assert len(lines) == 1


class TestYoloDetectionImport:
    def test_import_single_file(self, tmp_path):
        # Create YOLO structure
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "img.txt").write_text("0 0.5 0.4 0.3 0.6\n1 0.2 0.3 0.1 0.2\n")
        classes = ["person", "car"]
        results = import_yolo_detection(labels_dir, classes)
        assert len(results) == 1
        ia = results[0]
        assert ia.image_path == "img"
        assert len(ia.annotations) == 2
        assert ia.annotations[0].class_name == "person"
        assert ia.annotations[0].bbox == (0.5, 0.4, 0.3, 0.6)
        assert ia.annotations[0].confirmed is True
        assert ia.annotations[0].source == "manual"

    def test_import_with_data_yaml(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text(yaml.dump({"names": ["cat", "dog"], "nc": 2}))
        results = import_yolo_detection(labels_dir, classes=None, data_yaml=yaml_path)
        assert results[0].annotations[0].class_name == "cat"


class TestYoloPoseExport:
    def test_export_with_keypoints(self, tmp_path):
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
        out_dir = tmp_path / "output"
        export_yolo_pose([ia], out_dir, classes=["person"], kpt_dim=3)
        txt = (out_dir / "labels" / "pose.txt").read_text().strip()
        parts = txt.split()
        # class_id + 4 bbox + 2 keypoints * 3 dims = 11
        assert len(parts) == 11
        assert parts[0] == "0"


class TestYoloPoseImport:
    def test_import_pose(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        # class cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v
        (labels_dir / "p.txt").write_text("0 0.5 0.5 0.3 0.6 0.45 0.3 2 0.50 0.35 1\n")
        results = import_yolo_pose(
            labels_dir,
            classes=["person"],
            kpt_labels=["nose", "left_eye"],
            kpt_dim=3,
        )
        assert len(results) == 1
        ann = results[0].annotations[0]
        assert ann.bbox == (0.5, 0.5, 0.3, 0.6)
        assert len(ann.keypoints) == 2
        assert ann.keypoints[0].label == "nose"
        assert ann.keypoints[0].visible == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/formats/test_yolo.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement yolo.py**

```python
"""YOLO format import/export (detection + pose)."""
from __future__ import annotations

from pathlib import Path

import yaml

from src.core.annotation import Annotation, ImageAnnotation, Keypoint


def export_yolo_detection(
    image_annotations: list[ImageAnnotation],
    output_dir: Path | str,
    classes: list[str],
    only_confirmed: bool = False,
) -> None:
    """Export annotations to YOLO detection format (txt + data.yaml)."""
    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    for ia in image_annotations:
        stem = Path(ia.image_path).stem
        lines = []
        for ann in ia.annotations:
            if only_confirmed and not ann.confirmed:
                continue
            if ann.bbox is None:
                continue
            cid = classes.index(ann.class_name) if ann.class_name in classes else ann.class_id
            cx, cy, w, h = ann.bbox
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    # data.yaml
    data = {
        "nc": len(classes),
        "names": classes,
    }
    (output_dir / "data.yaml").write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")


def import_yolo_detection(
    labels_dir: Path | str,
    classes: list[str] | None = None,
    data_yaml: Path | str | None = None,
) -> list[ImageAnnotation]:
    """Import YOLO detection format. Provide classes or data_yaml."""
    labels_dir = Path(labels_dir)

    if classes is None and data_yaml:
        data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
        classes = data["names"]

    if classes is None:
        raise ValueError("Must provide classes or data_yaml")

    results = []
    for txt_path in sorted(labels_dir.glob("*.txt")):
        annotations = []
        for line in txt_path.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split()
            cid = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            annotations.append(Annotation(
                class_name=classes[cid] if cid < len(classes) else str(cid),
                class_id=cid,
                bbox=(cx, cy, w, h),
                confirmed=True,
                source="manual",
            ))
        results.append(ImageAnnotation(
            image_path=txt_path.stem,
            image_size=(0, 0),  # unknown without actual image
            annotations=annotations,
        ))
    return results


def export_yolo_pose(
    image_annotations: list[ImageAnnotation],
    output_dir: Path | str,
    classes: list[str],
    kpt_dim: int = 3,
    only_confirmed: bool = False,
) -> None:
    """Export annotations to YOLO pose format."""
    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    for ia in image_annotations:
        stem = Path(ia.image_path).stem
        lines = []
        for ann in ia.annotations:
            if only_confirmed and not ann.confirmed:
                continue
            if ann.bbox is None:
                continue
            cid = classes.index(ann.class_name) if ann.class_name in classes else ann.class_id
            cx, cy, w, h = ann.bbox
            parts = [f"{cid}", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
            for kp in ann.keypoints:
                parts.append(f"{kp.x:.6f}")
                parts.append(f"{kp.y:.6f}")
                if kpt_dim == 3:
                    parts.append(f"{kp.visible}")
            lines.append(" ".join(parts))
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def import_yolo_pose(
    labels_dir: Path | str,
    classes: list[str],
    kpt_labels: list[str],
    kpt_dim: int = 3,
) -> list[ImageAnnotation]:
    """Import YOLO pose format."""
    labels_dir = Path(labels_dir)
    num_kpts = len(kpt_labels)

    results = []
    for txt_path in sorted(labels_dir.glob("*.txt")):
        annotations = []
        for line in txt_path.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split()
            cid = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            keypoints = []
            kp_start = 5
            for i in range(num_kpts):
                offset = kp_start + i * kpt_dim
                kx = float(parts[offset])
                ky = float(parts[offset + 1])
                vis = int(float(parts[offset + 2])) if kpt_dim == 3 else 2
                keypoints.append(Keypoint(x=kx, y=ky, visible=vis, label=kpt_labels[i]))
            annotations.append(Annotation(
                class_name=classes[cid] if cid < len(classes) else str(cid),
                class_id=cid,
                bbox=(cx, cy, w, h),
                keypoints=keypoints,
                confirmed=True,
                source="manual",
            ))
        results.append(ImageAnnotation(
            image_path=txt_path.stem,
            image_size=(0, 0),
            annotations=annotations,
        ))
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/core/formats/test_yolo.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add src/core/formats/yolo.py tests/core/formats/test_yolo.py
git commit -m "feat: YOLO format import/export (detection + pose)"
```

---

## Task 8: COCO Format Import/Export

**Files:**
- Create: `src/core/formats/coco.py`
- Create: `tests/core/formats/test_coco.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for COCO format import/export."""
import json
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation, Keypoint
from src.core.formats.coco import export_coco, import_coco


class TestCocoExport:
    def test_export_detection_bbox_values(self, tmp_path):
        """Verify exact pixel conversion: normalized center -> COCO pixel top-left."""
        ia = ImageAnnotation(
            image_path="v.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.5, 0.5), confirmed=True),
            ],
        )
        out_path = tmp_path / "coco.json"
        export_coco([ia], out_path, classes=["person"])
        data = json.loads(out_path.read_text())
        bbox = data["annotations"][0]["bbox"]
        # cx=0.5, cy=0.5, w=0.5, h=0.5 on 640x480 -> x_tl=160, y_tl=120, pw=320, ph=240
        assert bbox[0] == 160.0
        assert bbox[1] == 120.0
        assert bbox[2] == 320.0
        assert bbox[3] == 240.0


    def test_export_detection(self, tmp_path):
        ia = ImageAnnotation(
            image_path="img_001.jpg",
            image_size=(1920, 1080),
            annotations=[
                Annotation(class_name="person", class_id=0, bbox=(0.5, 0.4, 0.3, 0.6), confirmed=True),
            ],
        )
        out_path = tmp_path / "coco.json"
        export_coco([ia], out_path, classes=["person", "car"])
        data = json.loads(out_path.read_text())
        assert len(data["images"]) == 1
        assert data["images"][0]["file_name"] == "img_001.jpg"
        assert data["images"][0]["width"] == 1920
        assert len(data["annotations"]) == 1
        ann = data["annotations"][0]
        # COCO bbox is [x_top_left, y_top_left, width, height] in pixels
        assert ann["category_id"] == 1  # COCO is 1-indexed
        assert len(ann["bbox"]) == 4
        assert len(data["categories"]) == 2

    def test_export_with_keypoints(self, tmp_path):
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
        out_path = tmp_path / "coco.json"
        export_coco([ia], out_path, classes=["person"])
        data = json.loads(out_path.read_text())
        ann = data["annotations"][0]
        assert "keypoints" in ann
        assert ann["num_keypoints"] == 2
        # COCO keypoints: [x, y, v, x, y, v, ...]
        assert len(ann["keypoints"]) == 6

    def test_export_only_confirmed(self, tmp_path):
        ia = ImageAnnotation(
            image_path="img.jpg",
            image_size=(100, 100),
            annotations=[
                Annotation(class_name="a", class_id=0, bbox=(0.5, 0.5, 0.1, 0.1), confirmed=True),
                Annotation(class_name="a", class_id=0, bbox=(0.2, 0.2, 0.1, 0.1), confirmed=False, source="auto", confidence=0.8),
            ],
        )
        out_path = tmp_path / "coco.json"
        export_coco([ia], out_path, classes=["a"], only_confirmed=True)
        data = json.loads(out_path.read_text())
        assert len(data["annotations"]) == 1


class TestCocoImport:
    def test_import_detection(self, tmp_path):
        coco_data = {
            "images": [
                {"id": 1, "file_name": "img.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [192, 96, 192, 288],  # x,y,w,h in pixels
                    "area": 55296,
                    "iscrowd": 0,
                },
            ],
            "categories": [
                {"id": 1, "name": "person"},
            ],
        }
        coco_path = tmp_path / "coco.json"
        coco_path.write_text(json.dumps(coco_data))
        results = import_coco(coco_path)
        assert len(results) == 1
        ia = results[0]
        assert ia.image_path == "img.jpg"
        assert ia.image_size == (640, 480)
        assert len(ia.annotations) == 1
        ann = ia.annotations[0]
        assert ann.class_name == "person"
        # Verify normalized bbox (center format)
        cx, cy, w, h = ann.bbox
        assert abs(cx - 0.45) < 0.01
        assert abs(cy - 0.5) < 0.01
        assert abs(w - 0.3) < 0.01
        assert abs(h - 0.6) < 0.01
        assert ann.confirmed is True

    def test_import_with_classes_resolves_class_id(self, tmp_path):
        coco_data = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 5, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            ],
            "categories": [{"id": 5, "name": "dog"}],
        }
        coco_path = tmp_path / "c.json"
        coco_path.write_text(json.dumps(coco_data))
        results = import_coco(coco_path, classes=["cat", "dog", "bird"])
        ann = results[0].annotations[0]
        assert ann.class_name == "dog"
        assert ann.class_id == 1  # index in provided classes list
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/formats/test_coco.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement coco.py**

```python
"""COCO format import/export."""
from __future__ import annotations

import json
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation, Keypoint


def export_coco(
    image_annotations: list[ImageAnnotation],
    output_path: Path | str,
    classes: list[str],
    only_confirmed: bool = False,
) -> None:
    """Export annotations to COCO JSON format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    ann_id = 1

    for img_id, ia in enumerate(image_annotations, start=1):
        w_img, h_img = ia.image_size
        images.append({
            "id": img_id,
            "file_name": ia.image_path,
            "width": w_img,
            "height": h_img,
        })

        for ann in ia.annotations:
            if only_confirmed and not ann.confirmed:
                continue
            if ann.bbox is None:
                continue

            cx, cy, bw, bh = ann.bbox
            # Convert normalized center to COCO pixel [x_tl, y_tl, w, h]
            x_tl = (cx - bw / 2) * w_img
            y_tl = (cy - bh / 2) * h_img
            pw = bw * w_img
            ph = bh * h_img

            cat_id = (classes.index(ann.class_name) + 1) if ann.class_name in classes else ann.class_id + 1

            coco_ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [round(x_tl, 2), round(y_tl, 2), round(pw, 2), round(ph, 2)],
                "area": round(pw * ph, 2),
                "iscrowd": 0,
            }

            if ann.keypoints:
                kps = []
                for kp in ann.keypoints:
                    kps.extend([round(kp.x * w_img, 2), round(kp.y * h_img, 2), kp.visible])
                coco_ann["keypoints"] = kps
                coco_ann["num_keypoints"] = len(ann.keypoints)

            annotations.append(coco_ann)
            ann_id += 1

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(classes)]

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    output_path.write_text(json.dumps(coco, indent=2, ensure_ascii=False), encoding="utf-8")


def import_coco(coco_path: Path | str, classes: list[str] | None = None) -> list[ImageAnnotation]:
    """Import annotations from COCO JSON format.

    Args:
        coco_path: Path to the COCO JSON file.
        classes: Optional class list for resolving class_id. If provided,
                 class_id is set to the index in this list. Otherwise,
                 class_id is derived from COCO category order.
    """
    coco_path = Path(coco_path)
    data = json.loads(coco_path.read_text(encoding="utf-8"))

    # Build lookups
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    img_map = {img["id"]: img for img in data["images"]}

    # Group annotations by image
    anns_by_image: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    results = []
    for img_id, img_info in sorted(img_map.items()):
        w_img = img_info["width"]
        h_img = img_info["height"]
        annotations = []

        for coco_ann in anns_by_image.get(img_id, []):
            x_tl, y_tl, pw, ph = coco_ann["bbox"]
            # Convert pixel [x_tl, y_tl, w, h] to normalized center
            cx = (x_tl + pw / 2) / w_img
            cy = (y_tl + ph / 2) / h_img
            bw = pw / w_img
            bh = ph / h_img

            cat_id = coco_ann["category_id"]
            class_name = cat_map.get(cat_id, str(cat_id))

            # Resolve class_id from provided classes list, or fallback
            if classes and class_name in classes:
                class_idx = classes.index(class_name)
            else:
                class_idx = cat_id - 1  # COCO is 1-indexed

            keypoints = []
            if "keypoints" in coco_ann:
                kps = coco_ann["keypoints"]
                for i in range(0, len(kps), 3):
                    kx = kps[i] / w_img if w_img else 0
                    ky = kps[i + 1] / h_img if h_img else 0
                    vis = int(kps[i + 2])
                    keypoints.append(Keypoint(x=kx, y=ky, visible=vis, label=f"kp_{i // 3}"))

            annotations.append(Annotation(
                class_name=class_name,
                class_id=class_idx,
                bbox=(cx, cy, bw, bh),
                keypoints=keypoints,
                confirmed=True,
                source="manual",
            ))

        results.append(ImageAnnotation(
            image_path=img_info["file_name"],
            image_size=(w_img, h_img),
            annotations=annotations,
        ))

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/core/formats/test_coco.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add src/core/formats/coco.py tests/core/formats/test_coco.py
git commit -m "feat: COCO format import/export"
```

---

## Task 9: labelme Format Import/Export

**Files:**
- Create: `src/core/formats/labelme.py`
- Create: `tests/core/formats/test_labelme.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for labelme format import/export."""
import json
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation, Keypoint
from src.core.formats.labelme import export_labelme, import_labelme


class TestLabelmeExport:
    def test_export_bbox(self, tmp_path):
        ia = ImageAnnotation(
            image_path="test.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.5, 0.5), confirmed=True),
            ],
        )
        out_dir = tmp_path / "output"
        export_labelme([ia], out_dir)
        json_path = out_dir / "test.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["imagePath"] == "test.jpg"
        assert data["imageWidth"] == 640
        assert data["imageHeight"] == 480
        assert len(data["shapes"]) == 1
        shape = data["shapes"][0]
        assert shape["label"] == "person"
        assert shape["shape_type"] == "rectangle"
        # labelme rectangle: [[x1, y1], [x2, y2]] in pixels
        assert len(shape["points"]) == 2

    def test_export_keypoints(self, tmp_path):
        ia = ImageAnnotation(
            image_path="kp.jpg",
            image_size=(100, 100),
            annotations=[
                Annotation(
                    class_name="point",
                    class_id=0,
                    bbox=None,
                    keypoints=[Keypoint(0.5, 0.3, 2, "nose")],
                    confirmed=True,
                ),
            ],
        )
        out_dir = tmp_path / "output"
        export_labelme([ia], out_dir)
        data = json.loads((out_dir / "kp.json").read_text())
        assert len(data["shapes"]) == 1
        assert data["shapes"][0]["shape_type"] == "point"
        assert data["shapes"][0]["label"] == "nose"


class TestLabelmeImport:
    def test_import_rectangle(self, tmp_path):
        labelme_data = {
            "imagePath": "img.jpg",
            "imageWidth": 640,
            "imageHeight": 480,
            "shapes": [
                {
                    "label": "car",
                    "shape_type": "rectangle",
                    "points": [[160, 120], [480, 360]],
                    "flags": {},
                },
            ],
        }
        json_path = tmp_path / "img.json"
        json_path.write_text(json.dumps(labelme_data))
        results = import_labelme(tmp_path)
        assert len(results) == 1
        ia = results[0]
        assert ia.image_path == "img.jpg"
        assert ia.image_size == (640, 480)
        ann = ia.annotations[0]
        assert ann.class_name == "car"
        # Check normalized center bbox
        cx, cy, w, h = ann.bbox
        assert abs(cx - 0.5) < 0.01
        assert abs(cy - 0.5) < 0.01
        assert abs(w - 0.5) < 0.01
        assert abs(h - 0.5) < 0.01

    def test_import_point(self, tmp_path):
        labelme_data = {
            "imagePath": "kp.jpg",
            "imageWidth": 100,
            "imageHeight": 100,
            "shapes": [
                {
                    "label": "nose",
                    "shape_type": "point",
                    "points": [[50, 30]],
                    "flags": {},
                },
            ],
        }
        (tmp_path / "kp.json").write_text(json.dumps(labelme_data))
        results = import_labelme(tmp_path)
        ia = results[0]
        assert len(ia.annotations) == 1
        ann = ia.annotations[0]
        assert ann.bbox is None
        assert len(ann.keypoints) == 1
        assert ann.keypoints[0].label == "nose"
        assert abs(ann.keypoints[0].x - 0.5) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/formats/test_labelme.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement labelme.py**

```python
"""labelme JSON format import/export."""
from __future__ import annotations

import json
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation, Keypoint


def export_labelme(
    image_annotations: list[ImageAnnotation],
    output_dir: Path | str,
    only_confirmed: bool = False,
) -> None:
    """Export annotations to labelme JSON format (one JSON per image)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ia in image_annotations:
        w_img, h_img = ia.image_size
        shapes = []

        for ann in ia.annotations:
            if only_confirmed and not ann.confirmed:
                continue

            # Export bbox as rectangle
            if ann.bbox is not None:
                cx, cy, bw, bh = ann.bbox
                x1 = (cx - bw / 2) * w_img
                y1 = (cy - bh / 2) * h_img
                x2 = (cx + bw / 2) * w_img
                y2 = (cy + bh / 2) * h_img
                shapes.append({
                    "label": ann.class_name,
                    "shape_type": "rectangle",
                    "points": [[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y2, 2)]],
                    "flags": {},
                })

            # Export keypoints as individual points
            for kp in ann.keypoints:
                shapes.append({
                    "label": kp.label,
                    "shape_type": "point",
                    "points": [[round(kp.x * w_img, 2), round(kp.y * h_img, 2)]],
                    "flags": {},
                })

        labelme_data = {
            "version": "5.0.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": ia.image_path,
            "imageData": None,
            "imageWidth": w_img,
            "imageHeight": h_img,
        }

        stem = Path(ia.image_path).stem
        out_path = output_dir / f"{stem}.json"
        out_path.write_text(json.dumps(labelme_data, indent=2, ensure_ascii=False), encoding="utf-8")


def import_labelme(input_dir: Path | str) -> list[ImageAnnotation]:
    """Import annotations from labelme JSON files in a directory."""
    input_dir = Path(input_dir)
    results = []

    for json_path in sorted(input_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        if "shapes" not in data:
            continue

        w_img = data.get("imageWidth", 0)
        h_img = data.get("imageHeight", 0)
        image_path = data.get("imagePath", json_path.stem + ".jpg")

        annotations = []
        for shape in data["shapes"]:
            shape_type = shape.get("shape_type", "")
            label = shape.get("label", "unknown")
            points = shape.get("points", [])

            if shape_type == "rectangle" and len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                cx = ((x1 + x2) / 2) / w_img if w_img else 0
                cy = ((y1 + y2) / 2) / h_img if h_img else 0
                bw = abs(x2 - x1) / w_img if w_img else 0
                bh = abs(y2 - y1) / h_img if h_img else 0
                annotations.append(Annotation(
                    class_name=label,
                    class_id=0,
                    bbox=(cx, cy, bw, bh),
                    confirmed=True,
                    source="manual",
                ))
            elif shape_type == "point" and len(points) == 1:
                px, py = points[0]
                kp = Keypoint(
                    x=px / w_img if w_img else 0,
                    y=py / h_img if h_img else 0,
                    visible=2,
                    label=label,
                )
                annotations.append(Annotation(
                    class_name=label,
                    class_id=0,
                    bbox=None,
                    keypoints=[kp],
                    confirmed=True,
                    source="manual",
                ))

        results.append(ImageAnnotation(
            image_path=image_path,
            image_size=(w_img, h_img),
            annotations=annotations,
        ))

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/core/formats/test_labelme.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add src/core/formats/labelme.py tests/core/formats/test_labelme.py
git commit -m "feat: labelme format import/export"
```

---

## Task 10: Full Test Suite & Integration Smoke Test

**Files:**
- No new files — run all tests, verify everything works together

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass (approximately 30+ tests across all modules)

- [ ] **Step 2: Fix any failures**

Address any issues found during full suite run.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: Plan 1 complete — core data layer with full test coverage"
```

---

## Summary

| Task | Module | Key Deliverables |
|------|--------|-----------------|
| 1 | Skeleton | pyproject.toml, directory structure, conftest.py |
| 2 | Colors | Catppuccin palette, assign_color() |
| 3 | Annotation | Keypoint, Annotation, ImageAnnotation dataclasses |
| 4 | Label I/O | save_annotation(), load_annotation() |
| 5 | Project | ProjectConfig, ProjectManager |
| 6 | Config | AppConfig (global settings) |
| 7 | YOLO | Detection + pose import/export |
| 8 | COCO | COCO JSON import/export |
| 9 | labelme | labelme JSON import/export |
| 10 | Integration | Full test suite verification |

**Next:** Plan 2 will cover the Engine layer (trainer, predictor, model_manager) with ultralytics integration.
