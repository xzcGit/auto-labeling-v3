"""Training controller — start, stop, register trained models."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import QWidget, QMessageBox

from src.core.project import ProjectManager
from src.core.label_io import load_annotation
from src.engine.dataset import DatasetPreparer
from src.engine.model_manager import ModelRegistry, ModelInfo
from src.engine.trainer import TrainConfig
from src.utils.workers import TrainWorker

logger = logging.getLogger(__name__)


class TrainController:
    """Handles training lifecycle: validation, start, stop, model registration."""

    def __init__(self, parent_widget: QWidget):
        self._parent = parent_widget
        self._worker: TrainWorker | None = None
        self._run_name: str = ""
        self._dataset_size: int = 0

    @property
    def worker(self) -> TrainWorker | None:
        return self._worker

    @property
    def dataset_size(self) -> int:
        return self._dataset_size

    def validate_and_prepare(
        self, project: ProjectManager, task: str, val_ratio: float,
        kpt_shape: list[int] | None = None,
    ) -> str | None:
        """Validate dataset and prepare for training. Returns data_yaml path or None."""
        confirmed_count = 0
        class_counts: dict[str, int] = {}
        for img_path in project.list_images():
            label_path = project.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia is None:
                continue
            for ann in ia.annotations:
                if ann.confirmed:
                    confirmed_count += 1
                    class_counts[ann.class_name] = class_counts.get(ann.class_name, 0) + 1

        self._dataset_size = confirmed_count

        if confirmed_count < 10:
            reply = QMessageBox.question(
                self._parent, "标注数量不足",
                f"仅有 {confirmed_count} 个已确认标注，建议至少 10 个。\n是否仍然继续训练？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return None

        if class_counts:
            max_c = max(class_counts.values())
            min_c = min(class_counts.values())
            if min_c > 0 and max_c / min_c > 10:
                imbalanced = ", ".join(f"{k}: {v}" for k, v in sorted(class_counts.items()))
                reply = QMessageBox.question(
                    self._parent, "类别不均衡",
                    f"类别分布严重不均衡:\n{imbalanced}\n是否仍然继续训练？",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return None

        preparer = DatasetPreparer(project)
        output_dir = project.project_dir / "datasets" / "current"
        data_yaml = preparer.prepare(output_dir, task=task, val_ratio=val_ratio, kpt_shape=kpt_shape)
        return str(data_yaml)

    def start(self, config: TrainConfig, project: ProjectManager, task: str) -> TrainWorker:
        """Create and start a training worker. Returns the worker."""
        config.project = str(project.project_dir / "models")
        run_name = f"{task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        config.name = run_name
        self._run_name = run_name

        self._worker = TrainWorker(config)
        self._worker.start()
        logger.info("Training started: %s | %s | %d epochs", task, config.model, config.epochs)
        return self._worker

    def stop(self) -> None:
        """Request graceful stop of current training."""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()

    def register_model(
        self, registry: ModelRegistry, project: ProjectManager,
        task: str, base_model: str, epochs: int, metrics: dict,
    ) -> ModelInfo:
        """Register the trained model in the registry."""
        model_info = ModelInfo(
            name=f"{task}-{len(registry.list_models()) + 1}",
            path=f"models/{self._run_name}/weights/best.pt",
            task=task,
            base_model=base_model,
            classes=project.config.classes,
            metrics=metrics,
            epochs=epochs,
            dataset_size=self._dataset_size,
        )
        registry.register(model_info)
        registry.save()
        logger.info("Registered trained model: %s", model_info.name)
        return model_info
