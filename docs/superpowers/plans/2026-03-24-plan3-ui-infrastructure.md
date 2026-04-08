# Plan 3: UI Infrastructure & Workers

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the UI infrastructure layer — theme stylesheet, image loading utilities, QThread workers for training and batch inference, and the MainWindow skeleton with tab structure.

**Architecture:** A Catppuccin Mocha dark theme applied as a global QSS stylesheet. Image utility provides QPixmap loading with LRU cache. Workers wrap engine layer (Trainer, Predictor) in QThread subclasses emitting Qt signals for progress/completion/error. MainWindow uses QTabWidget to host the three panels (label, train, model) with a startup welcome page. All workers are tested with mocked engine dependencies.

**Tech Stack:** Python 3.10+, PyQt5 (QThread, pyqtSignal, QPixmap), pytest (custom qapp fixture, no pytest-qt dependency)

---

## File Structure

```
auto-labeling-v3/
├── src/
│   ├── ui/
│   │   ├── __init__.py           # (exists, empty)
│   │   └── theme.py              # Catppuccin Mocha QSS stylesheet
│   ├── utils/
│   │   ├── image.py              # QPixmap loading with LRU cache
│   │   └── workers.py            # TrainWorker, BatchPredictWorker QThread classes
│   └── app.py                    # MainWindow with tab structure
├── tests/
│   ├── conftest.py               # Add qapp fixture
│   ├── utils/
│   │   ├── test_image.py         # Image loader tests
│   │   └── test_workers.py       # Worker tests with mocked engine
│   └── test_app.py               # MainWindow smoke tests
```

---

## Task 1: Theme Stylesheet

**Files:**
- Create: `src/ui/theme.py`

- [ ] **Step 1: Create theme module**

```python
"""Catppuccin Mocha dark theme stylesheet for PyQt5."""

# Catppuccin Mocha palette
MOCHA = {
    "base": "#1e1e2e",
    "mantle": "#181825",
    "crust": "#11111b",
    "surface0": "#313244",
    "surface1": "#45475a",
    "surface2": "#585b70",
    "overlay0": "#6c7086",
    "text": "#cdd6f4",
    "subtext0": "#a6adc8",
    "green": "#a6e3a1",
    "blue": "#89b4fa",
    "red": "#f38ba8",
    "peach": "#fab387",
    "yellow": "#f9e2af",
    "mauve": "#cba6f7",
    "teal": "#94e2d5",
    "sky": "#89dceb",
    "lavender": "#b4befe",
}

STYLESHEET = f"""
QWidget {{
    background-color: {MOCHA['base']};
    color: {MOCHA['text']};
    font-size: 13px;
}}

QMainWindow {{
    background-color: {MOCHA['base']};
}}

QTabWidget::pane {{
    border: 1px solid {MOCHA['surface1']};
    background-color: {MOCHA['base']};
}}

QTabBar::tab {{
    background-color: {MOCHA['surface0']};
    color: {MOCHA['subtext0']};
    padding: 8px 20px;
    border: 1px solid {MOCHA['surface1']};
    border-bottom: none;
    margin-right: 2px;
}}

QTabBar::tab:selected {{
    background-color: {MOCHA['base']};
    color: {MOCHA['text']};
    border-bottom: 2px solid {MOCHA['blue']};
}}

QTabBar::tab:hover {{
    background-color: {MOCHA['surface1']};
}}

QPushButton {{
    background-color: {MOCHA['surface0']};
    color: {MOCHA['text']};
    border: 1px solid {MOCHA['surface1']};
    border-radius: 4px;
    padding: 6px 16px;
}}

QPushButton:hover {{
    background-color: {MOCHA['surface1']};
}}

QPushButton:pressed {{
    background-color: {MOCHA['surface2']};
}}

QPushButton:disabled {{
    color: {MOCHA['overlay0']};
    background-color: {MOCHA['mantle']};
}}

QListWidget {{
    background-color: {MOCHA['mantle']};
    border: 1px solid {MOCHA['surface0']};
    border-radius: 4px;
    outline: none;
}}

QListWidget::item {{
    padding: 4px 8px;
}}

QListWidget::item:selected {{
    background-color: {MOCHA['surface0']};
    color: {MOCHA['text']};
}}

QListWidget::item:hover {{
    background-color: {MOCHA['surface0']};
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {MOCHA['mantle']};
    color: {MOCHA['text']};
    border: 1px solid {MOCHA['surface1']};
    border-radius: 4px;
    padding: 4px 8px;
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {MOCHA['blue']};
}}

QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}

QTextEdit, QPlainTextEdit {{
    background-color: {MOCHA['mantle']};
    color: {MOCHA['text']};
    border: 1px solid {MOCHA['surface0']};
    border-radius: 4px;
}}

QLabel {{
    background-color: transparent;
}}

QGroupBox {{
    border: 1px solid {MOCHA['surface1']};
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 16px;
    font-weight: bold;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    padding: 0 6px;
    color: {MOCHA['subtext0']};
}}

QScrollBar:vertical {{
    background-color: {MOCHA['mantle']};
    width: 10px;
    border: none;
}}

QScrollBar::handle:vertical {{
    background-color: {MOCHA['surface1']};
    border-radius: 5px;
    min-height: 20px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {MOCHA['surface2']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: {MOCHA['mantle']};
    height: 10px;
    border: none;
}}

QScrollBar::handle:horizontal {{
    background-color: {MOCHA['surface1']};
    border-radius: 5px;
    min-width: 20px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {MOCHA['surface2']};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QStatusBar {{
    background-color: {MOCHA['mantle']};
    color: {MOCHA['subtext0']};
}}

QMenuBar {{
    background-color: {MOCHA['mantle']};
    color: {MOCHA['text']};
}}

QMenuBar::item:selected {{
    background-color: {MOCHA['surface0']};
}}

QMenu {{
    background-color: {MOCHA['surface0']};
    color: {MOCHA['text']};
    border: 1px solid {MOCHA['surface1']};
}}

QMenu::item:selected {{
    background-color: {MOCHA['surface1']};
}}

QProgressBar {{
    background-color: {MOCHA['surface0']};
    border: none;
    border-radius: 4px;
    text-align: center;
    color: {MOCHA['text']};
}}

QProgressBar::chunk {{
    background-color: {MOCHA['blue']};
    border-radius: 4px;
}}

QToolTip {{
    background-color: {MOCHA['surface0']};
    color: {MOCHA['text']};
    border: 1px solid {MOCHA['surface1']};
    padding: 4px;
}}

QSplitter::handle {{
    background-color: {MOCHA['surface0']};
}}

QSplitter::handle:hover {{
    background-color: {MOCHA['surface1']};
}}

QHeaderView::section {{
    background-color: {MOCHA['surface0']};
    color: {MOCHA['text']};
    border: 1px solid {MOCHA['surface1']};
    padding: 4px;
}}

QCheckBox {{
    color: {MOCHA['text']};
    spacing: 6px;
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid {MOCHA['surface2']};
    background-color: {MOCHA['mantle']};
}}

QCheckBox::indicator:checked {{
    background-color: {MOCHA['blue']};
    border-color: {MOCHA['blue']};
}}

QSlider::groove:horizontal {{
    background-color: {MOCHA['surface0']};
    height: 6px;
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background-color: {MOCHA['blue']};
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}}
"""


def apply_theme(app) -> None:
    """Apply the Catppuccin Mocha dark theme to a QApplication."""
    app.setStyleSheet(STYLESHEET)
```

Write this to `src/ui/theme.py`.

- [ ] **Step 2: Commit**

```bash
git add src/ui/theme.py
git commit -m "feat: Catppuccin Mocha dark theme stylesheet"
```

---

## Task 2: Image Loading Utility

**Files:**
- Create: `src/utils/image.py`
- Create: `tests/utils/test_image.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for image loading utility."""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def _make_test_image(path: Path, width: int = 100, height: int = 80) -> None:
    """Create a minimal valid PNG file for testing."""
    from PyQt5.QtGui import QImage, QColor
    from PyQt5.QtCore import Qt

    img = QImage(width, height, QImage.Format_RGB32)
    img.fill(QColor(Qt.red))
    img.save(str(path), "PNG")


class TestLoadPixmap:
    def test_load_valid_image(self, qapp, tmp_path):
        from src.utils.image import load_pixmap

        img_path = tmp_path / "test.png"
        _make_test_image(img_path)
        pixmap = load_pixmap(img_path)
        assert pixmap is not None
        assert not pixmap.isNull()
        assert pixmap.width() == 100
        assert pixmap.height() == 80

    def test_load_nonexistent_returns_none(self, qapp, tmp_path):
        from src.utils.image import load_pixmap

        pixmap = load_pixmap(tmp_path / "nope.png")
        assert pixmap is None

    def test_load_with_max_size(self, qapp, tmp_path):
        from src.utils.image import load_pixmap

        img_path = tmp_path / "big.png"
        _make_test_image(img_path, 2000, 1000)
        pixmap = load_pixmap(img_path, max_size=500)
        assert pixmap is not None
        assert pixmap.width() <= 500
        assert pixmap.height() <= 500


class TestImageCache:
    def test_cache_returns_same_object(self, qapp, tmp_path):
        from src.utils.image import ImageCache

        img_path = tmp_path / "cached.png"
        _make_test_image(img_path)
        cache = ImageCache(max_size=5)
        p1 = cache.get(img_path)
        p2 = cache.get(img_path)
        assert p1 is not None
        assert p1 is p2  # same cached object

    def test_cache_evicts_oldest(self, qapp, tmp_path):
        from src.utils.image import ImageCache

        cache = ImageCache(max_size=2)
        paths = []
        for i in range(3):
            p = tmp_path / f"img{i}.png"
            _make_test_image(p)
            paths.append(p)

        cache.get(paths[0])
        cache.get(paths[1])
        cache.get(paths[2])  # should evict paths[0]
        assert str(paths[0]) not in cache._cache

    def test_cache_clear(self, qapp, tmp_path):
        from src.utils.image import ImageCache

        img_path = tmp_path / "clear.png"
        _make_test_image(img_path)
        cache = ImageCache(max_size=5)
        cache.get(img_path)
        assert len(cache._cache) == 1
        cache.clear()
        assert len(cache._cache) == 0


class TestGetImageSize:
    def test_returns_width_height(self, qapp, tmp_path):
        from src.utils.image import get_image_size

        img_path = tmp_path / "size.png"
        _make_test_image(img_path, 320, 240)
        w, h = get_image_size(img_path)
        assert w == 320
        assert h == 240

    def test_nonexistent_returns_zero(self, qapp, tmp_path):
        from src.utils.image import get_image_size

        w, h = get_image_size(tmp_path / "nope.png")
        assert w == 0
        assert h == 0
```

Write to `tests/utils/test_image.py`.

- [ ] **Step 2: Append qapp fixture to existing conftest.py**

Append the following to the existing `tests/conftest.py` (which already has `import pytest`, `tmp_project` and `sample_project_config` fixtures):

```python
@pytest.fixture(scope="session")
def qapp():
    """Provide a QApplication instance for the entire test session."""
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
```

Do NOT overwrite the existing file — only append this fixture.

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/utils/test_image.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.image'`

- [ ] **Step 4: Implement image utility**

```python
"""Image loading utilities with caching."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImageReader


def load_pixmap(
    path: Path | str, max_size: int | None = None
) -> QPixmap | None:
    """Load image as QPixmap. Returns None if file doesn't exist or fails to load."""
    path = Path(path)
    if not path.exists():
        return None

    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    image = reader.read()
    if image.isNull():
        return None

    pixmap = QPixmap.fromImage(image)
    if max_size and (pixmap.width() > max_size or pixmap.height() > max_size):
        pixmap = pixmap.scaled(
            max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
    return pixmap


def get_image_size(path: Path | str) -> tuple[int, int]:
    """Get image dimensions (width, height) without fully loading. Returns (0, 0) on failure."""
    reader = QImageReader(str(path))
    size = reader.size()
    if size.isValid():
        return size.width(), size.height()
    return 0, 0


class ImageCache:
    """LRU cache for loaded QPixmaps."""

    def __init__(self, max_size: int = 10):
        self._max_size = max_size
        self._cache: OrderedDict[str, QPixmap] = OrderedDict()

    def get(self, path: Path | str, pixmap_max_size: int | None = None) -> QPixmap | None:
        """Get pixmap from cache, loading if necessary."""
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        pixmap = load_pixmap(path, max_size=pixmap_max_size)
        if pixmap is None:
            return None

        self._cache[key] = pixmap
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return pixmap

    def clear(self) -> None:
        """Clear all cached pixmaps."""
        self._cache.clear()

    def invalidate(self, path: Path | str) -> None:
        """Remove a specific path from cache."""
        self._cache.pop(str(path), None)
```

Write to `src/utils/image.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/utils/test_image.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/utils/image.py tests/utils/test_image.py tests/conftest.py
git commit -m "feat: image loading utility with LRU cache"
```

---

## Task 3: QThread Workers

**Files:**
- Create: `src/utils/workers.py`
- Create: `tests/utils/test_workers.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for QThread workers."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PyQt5.QtCore import QCoreApplication


def _process_events():
    """Process pending Qt events."""
    QCoreApplication.processEvents()


class TestTrainWorker:
    def test_emits_epoch_signal(self, qapp):
        from src.utils.workers import TrainWorker
        from src.engine.trainer import TrainConfig

        config = TrainConfig(data_yaml="data.yaml", model="yolov8n.pt", task="detect", epochs=2)

        # Mock the Trainer
        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance

        # Capture on_epoch_end callback and call it during train
        def fake_train(cfg, on_epoch_end=None):
            if on_epoch_end:
                on_epoch_end({"epoch": 0, "train_loss": 1.5})
                on_epoch_end({"epoch": 1, "train_loss": 0.8})

        mock_trainer_instance.train.side_effect = fake_train
        mock_trainer_instance.get_best_metrics.return_value = {"mAP50": 0.85}

        worker = TrainWorker(config, trainer_cls=mock_trainer_cls)
        epochs = []
        worker.epoch_update.connect(lambda d: epochs.append(d))
        finished_data = []
        worker.finished_ok.connect(lambda d: finished_data.append(d))

        worker.run()  # call run() directly, not start()
        assert len(epochs) == 2
        assert epochs[0]["epoch"] == 0
        assert len(finished_data) == 1
        assert finished_data[0]["mAP50"] == 0.85

    def test_emits_error_on_exception(self, qapp):
        from src.utils.workers import TrainWorker
        from src.engine.trainer import TrainConfig

        config = TrainConfig(data_yaml="data.yaml", model="yolov8n.pt", task="detect")
        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance
        mock_trainer_instance.train.side_effect = RuntimeError("CUDA OOM")

        worker = TrainWorker(config, trainer_cls=mock_trainer_cls)
        errors = []
        worker.error.connect(lambda msg: errors.append(msg))

        worker.run()
        assert len(errors) == 1
        assert "CUDA OOM" in errors[0]


class TestBatchPredictWorker:
    def test_emits_progress_and_results(self, qapp):
        from src.utils.workers import BatchPredictWorker
        from src.core.annotation import Annotation

        mock_predictor = MagicMock()
        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), confirmed=False, source="auto")
        mock_predictor.predict_with_size.return_value = ([ann], (640, 480))

        image_paths = [Path(f"/imgs/img{i}.jpg") for i in range(3)]
        worker = BatchPredictWorker(
            predictor=mock_predictor,
            image_paths=image_paths,
            conf=0.5,
            iou=0.45,
        )

        progress_values = []
        worker.progress.connect(lambda cur, total: progress_values.append((cur, total)))
        results = []
        worker.image_done.connect(lambda path, anns, size: results.append((path, anns, size)))
        finished = []
        worker.finished_ok.connect(lambda: finished.append(True))

        worker.run()
        assert len(progress_values) == 3
        assert progress_values[-1] == (3, 3)
        assert len(results) == 3
        assert results[0][1][0].class_name == "cat"
        assert len(finished) == 1

    def test_cancel_stops_processing(self, qapp):
        from src.utils.workers import BatchPredictWorker

        mock_predictor = MagicMock()
        mock_predictor.predict_with_size.return_value = ([], (640, 480))

        image_paths = [Path(f"/imgs/img{i}.jpg") for i in range(10)]
        worker = BatchPredictWorker(
            predictor=mock_predictor,
            image_paths=image_paths,
            conf=0.5,
            iou=0.45,
        )

        results = []
        worker.image_done.connect(lambda path, anns, size: results.append(path))

        # Cancel after first call
        def cancel_after_first(*args, **kwargs):
            if mock_predictor.predict_with_size.call_count >= 2:
                worker.cancel()
            return ([], (640, 480))

        mock_predictor.predict_with_size.side_effect = cancel_after_first
        worker.run()
        assert len(results) < 10

    def test_emits_error_on_exception(self, qapp):
        from src.utils.workers import BatchPredictWorker

        mock_predictor = MagicMock()
        mock_predictor.predict_with_size.side_effect = RuntimeError("model error")

        worker = BatchPredictWorker(
            predictor=mock_predictor,
            image_paths=[Path("/img.jpg")],
            conf=0.5,
            iou=0.45,
        )
        errors = []
        worker.error.connect(lambda msg: errors.append(msg))

        worker.run()
        assert len(errors) == 1
        assert "model error" in errors[0]
```

Write to `tests/utils/test_workers.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/utils/test_workers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.workers'`

- [ ] **Step 3: Implement workers**

```python
"""QThread workers for training and batch inference."""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

from src.core.annotation import Annotation
from src.engine.trainer import TrainConfig, Trainer

logger = logging.getLogger(__name__)


class TrainWorker(QThread):
    """Runs YOLO training in a background thread.

    Signals:
        epoch_update(dict): Emitted after each epoch with metrics dict.
        finished_ok(dict): Emitted on successful completion with best metrics.
        error(str): Emitted if training fails with error message.
    """

    epoch_update = pyqtSignal(dict)
    finished_ok = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, config: TrainConfig, trainer_cls=None, parent=None):
        super().__init__(parent)
        self._config = config
        self._trainer_cls = trainer_cls or Trainer

    def run(self) -> None:
        try:
            trainer = self._trainer_cls()
            trainer.train(self._config, on_epoch_end=self._on_epoch)
            metrics = trainer.get_best_metrics()
            self.finished_ok.emit(metrics)
        except Exception as e:
            logger.exception("Training failed")
            self.error.emit(str(e))

    def _on_epoch(self, metrics: dict) -> None:
        self.epoch_update.emit(metrics)


class BatchPredictWorker(QThread):
    """Runs batch inference in a background thread.

    Signals:
        progress(int, int): Emitted with (current, total) after each image.
        image_done(str, object, object): Emitted with (image_path, annotations, image_size).
        finished_ok(): Emitted when all images are processed (not emitted on cancel).
        error(str): Emitted if inference fails with error message.
    """

    progress = pyqtSignal(int, int)
    image_done = pyqtSignal(str, object, object)  # (path, annotations, image_size)
    finished_ok = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        predictor,
        image_paths: list[Path],
        conf: float = 0.5,
        iou: float = 0.45,
        project_classes: list[str] | None = None,
        kpt_labels: list[str] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._predictor = predictor
        self._image_paths = image_paths
        self._conf = conf
        self._iou = iou
        self._project_classes = project_classes
        self._kpt_labels = kpt_labels
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation of batch processing."""
        self._cancelled = True

    def run(self) -> None:
        total = len(self._image_paths)
        try:
            for i, img_path in enumerate(self._image_paths):
                if self._cancelled:
                    break
                annotations, img_size = self._predictor.predict_with_size(
                    img_path,
                    conf=self._conf,
                    iou=self._iou,
                    project_classes=self._project_classes,
                    kpt_labels=self._kpt_labels,
                )
                self.image_done.emit(str(img_path), annotations, img_size)
                self.progress.emit(i + 1, total)
            if not self._cancelled:
                self.finished_ok.emit()
        except Exception as e:
            logger.exception("Batch inference failed")
            self.error.emit(str(e))
```

Write to `src/utils/workers.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/utils/test_workers.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/workers.py tests/utils/test_workers.py
git commit -m "feat: QThread workers for training and batch inference"
```

---

## Task 4: MainWindow Skeleton

**Files:**
- Create: `src/app.py`
- Create: `tests/test_app.py`
- Modify: `main.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for MainWindow."""
import pytest
from pathlib import Path


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
```

Write to `tests/test_app.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_app.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.app'`

- [ ] **Step 3: Implement MainWindow**

```python
"""Main application window."""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QMenuBar,
    QAction,
    QStatusBar,
)
from PyQt5.QtCore import Qt

from src.core.config import AppConfig
from src.core.project import ProjectManager
from src.ui.theme import apply_theme

logger = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / ".autolabel" / "config.json"


class WelcomePage(QWidget):
    """Startup welcome page with recent projects and create/open buttons."""

    def __init__(self, app_config: AppConfig, parent=None):
        super().__init__(parent)
        self._config = app_config
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 40, 60, 40)

        # Title
        title = QLabel("AutoLabel V3")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #89b4fa;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("图像标注 · 模型训练 · 自动标注")
        subtitle.setStyleSheet("font-size: 14px; color: #a6adc8;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        layout.addSpacing(30)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_new = QPushButton("新建项目")
        self.btn_new.setMinimumWidth(120)
        btn_layout.addWidget(self.btn_new)

        self.btn_open = QPushButton("打开项目")
        self.btn_open.setMinimumWidth(120)
        btn_layout.addWidget(self.btn_open)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        layout.addSpacing(20)

        # Recent projects
        recent_label = QLabel("最近项目")
        recent_label.setStyleSheet("font-size: 14px; color: #a6adc8;")
        layout.addWidget(recent_label)

        self.recent_list = QListWidget()
        self.recent_list.setMaximumHeight(200)
        for project_path in self._config.recent_projects:
            item = QListWidgetItem(project_path)
            self.recent_list.addItem(item)
        layout.addWidget(self.recent_list)

        layout.addStretch()


class MainWindow(QMainWindow):
    """Application main window with tab-based layout."""

    def __init__(self, config_path: Path | str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AutoLabel V3")

        # Load app config
        self._config_path = Path(config_path) if config_path else CONFIG_PATH
        self._app_config = AppConfig.load(self._config_path)
        geo = self._app_config.window_geometry
        self.setGeometry(geo["x"], geo["y"], geo["width"], geo["height"])

        # State
        self._project: ProjectManager | None = None

        # Central widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Welcome page
        self._welcome = WelcomePage(self._app_config)
        self._welcome.btn_new.clicked.connect(self._on_new_project)
        self._welcome.btn_open.clicked.connect(self._on_open_project)
        self._welcome.recent_list.itemDoubleClicked.connect(self._on_recent_clicked)
        self.tab_widget.addTab(self._welcome, "欢迎")

        # Menu bar
        self._setup_menus()

        # Status bar
        self._status_label = QLabel("就绪")
        self.statusBar().addPermanentWidget(self._status_label)

    def _setup_menus(self) -> None:
        mb = self.menuBar()

        # File menu
        file_menu = mb.addMenu("文件")

        new_action = QAction("新建项目", self)
        new_action.triggered.connect(self._on_new_project)
        file_menu.addAction(new_action)

        open_action = QAction("打开项目", self)
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def open_project(self, project_manager: ProjectManager) -> None:
        """Open a project and switch to labeling workspace."""
        self._project = project_manager
        self.setWindowTitle(f"AutoLabel V3 — {project_manager.config.name}")

        # Update recent projects
        self._app_config.add_recent_project(str(project_manager.project_dir))
        self._app_config.save(self._config_path)

        self._status_label.setText(
            f"项目: {project_manager.config.name} | "
            f"图片: {len(project_manager.list_images())} | "
            f"类别: {len(project_manager.config.classes)}"
        )

    def _on_new_project(self) -> None:
        """Handle new project creation."""
        # Placeholder — will be connected to NewProjectDialog in Plan 5
        pass

    def _on_open_project(self) -> None:
        """Handle open project via file dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self, "打开项目", "", "项目文件 (project.json)"
        )
        if path:
            project_dir = Path(path).parent
            try:
                pm = ProjectManager.open(project_dir)
                self.open_project(pm)
            except Exception as e:
                logger.error("Failed to open project: %s", e)

    def _on_recent_clicked(self, item: QListWidgetItem) -> None:
        """Handle double-click on recent project."""
        project_dir = Path(item.text())
        try:
            pm = ProjectManager.open(project_dir)
            self.open_project(pm)
        except Exception as e:
            logger.error("Failed to open recent project: %s", e)

    def closeEvent(self, event) -> None:
        """Save config on close."""
        geo = self.geometry()
        self._app_config.window_geometry = {
            "x": geo.x(), "y": geo.y(),
            "width": geo.width(), "height": geo.height(),
        }
        self._app_config.save(self._config_path)
        super().closeEvent(event)
```

Write to `src/app.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_app.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Update main.py entry point**

```python
"""AutoLabel V3 — entry point."""
import sys

from PyQt5.QtWidgets import QApplication

from src.app import MainWindow
from src.ui.theme import apply_theme


def main() -> int:
    app = QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
```

Write to `main.py`.

- [ ] **Step 6: Run all tests to verify nothing is broken**

Run: `pytest -v`
Expected: all tests PASS (including existing core/ engine/ utils/ tests).

- [ ] **Step 7: Commit**

```bash
git add src/app.py tests/test_app.py main.py
git commit -m "feat: MainWindow skeleton with welcome page and tab structure"
```

---

## Task 5: Verify Full Test Suite

- [ ] **Step 1: Run full test suite**

Run: `pytest -v --tb=short`
Expected: all tests PASS. No regressions from existing core/ and engine/ tests.

- [ ] **Step 2: Quick smoke test of the app**

Run: `python main.py`
Expected: Window opens with dark theme, welcome page shows. Close manually.
