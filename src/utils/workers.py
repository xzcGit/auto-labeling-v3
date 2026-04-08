"""QThread workers for training and batch inference."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal, QMutex

from src.core.annotation import Annotation
from src.engine.trainer import TrainConfig, Trainer

logger = logging.getLogger(__name__)


class TrainWorker(QThread):
    """Runs YOLO training in a background thread.

    Signals:
        epoch_update(dict): Emitted after each epoch with metrics dict.
        finished_ok(dict): Emitted on successful completion with best metrics.
        cancelled(): Emitted when training is cancelled by user.
        error(str): Emitted if training fails with error message.
    """

    epoch_update = pyqtSignal(dict)
    finished_ok = pyqtSignal(dict)
    cancelled = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, config: TrainConfig, trainer_cls=None, parent=None):
        super().__init__(parent)
        self._config = config
        self._trainer_cls = trainer_cls or Trainer
        self._trainer: Trainer | None = None
        self._trainer_mutex = QMutex()

    def cancel(self) -> None:
        """Request graceful cancellation of training."""
        self._trainer_mutex.lock()
        try:
            if self._trainer:
                self._trainer.request_cancel()
        finally:
            self._trainer_mutex.unlock()

    def run(self) -> None:
        try:
            trainer = self._trainer_cls()
            self._trainer_mutex.lock()
            try:
                self._trainer = trainer
            finally:
                self._trainer_mutex.unlock()
            trainer.train(self._config, on_epoch_end=self._on_epoch)
            if trainer.cancelled:
                self.cancelled.emit()
            else:
                metrics = trainer.get_best_metrics()
                self.finished_ok.emit(metrics)
        except Exception as e:
            # Broad catch intentional: uncaught exceptions in QThread silently kill the thread
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
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Request cancellation of batch processing."""
        self._cancelled.set()

    def run(self) -> None:
        total = len(self._image_paths)
        try:
            for i, img_path in enumerate(self._image_paths):
                if self._cancelled.is_set():
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
            if not self._cancelled.is_set():
                self.finished_ok.emit()
        except Exception as e:
            # Broad catch intentional: uncaught exceptions in QThread silently kill the thread
            logger.exception("Batch inference failed")
            self.error.emit(str(e))
