"""Model controller — load, delete, import, inference."""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtWidgets import QWidget, QFileDialog, QInputDialog, QMessageBox

from src.core.project import ProjectManager
from src.core.annotation import ImageAnnotation
from src.core.label_io import load_annotation, save_annotation
from src.engine.model_manager import ModelRegistry, ModelInfo
from src.engine.predictor import Predictor
from src.utils.workers import BatchPredictWorker

logger = logging.getLogger(__name__)


class ModelController:
    """Handles model lifecycle: load, delete, import, auto-label."""

    def __init__(self, parent_widget: QWidget):
        self._parent = parent_widget
        self._predictor: Predictor | None = None
        self._registry: ModelRegistry | None = None
        self._project: ProjectManager | None = None
        self._batch_worker: BatchPredictWorker | None = None

    @property
    def predictor(self) -> Predictor | None:
        return self._predictor

    @property
    def registry(self) -> ModelRegistry | None:
        return self._registry

    def set_context(self, project: ProjectManager, registry: ModelRegistry) -> None:
        self._project = project
        self._registry = registry

    def load_model(self, model_id: str) -> bool:
        """Load a model for inference. Returns True on success."""
        if not self._registry or not self._project:
            return False
        model_info = self._registry.get(model_id)
        if not model_info:
            return False
        try:
            model_path = Path(model_info.path)
            if not model_path.is_absolute():
                model_path = self._project.project_dir / model_path
            if not model_path.exists():
                QMessageBox.warning(self._parent, "错误", f"模型文件不存在: {model_path}")
                return False
            from ultralytics import YOLO
            yolo_model = YOLO(str(model_path))
            self._predictor = Predictor(yolo_model)
            logger.info("Loaded model: %s from %s", model_info.name, model_path)
            return True
        except (RuntimeError, FileNotFoundError, OSError) as e:
            logger.error("Failed to load model: %s", e, exc_info=True)
            QMessageBox.warning(self._parent, "加载失败", f"模型加载失败: {e}")
            return False

    def delete_model(self, model_id: str) -> bool:
        """Delete model from registry (not file). Returns True if deleted."""
        if not self._registry:
            return False
        model_info = self._registry.get(model_id)
        if not model_info:
            return False
        reply = QMessageBox.question(
            self._parent, "确认删除",
            f"确定要删除模型 \"{model_info.name}\" 吗？\n（模型文件不会被删除）",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._registry.remove(model_id)
            self._registry.save()
            return True
        return False

    def rename_model(self, model_id: str) -> bool:
        """Rename a model's display name via dialog. Returns True if renamed."""
        if not self._registry:
            return False
        model_info = self._registry.get(model_id)
        if not model_info:
            return False
        new_name, ok = QInputDialog.getText(
            self._parent, "重命名模型", "请输入新的模型名称:",
            text=model_info.name,
        )
        if not ok or not new_name.strip() or new_name.strip() == model_info.name:
            return False
        self._registry.rename(model_id, new_name.strip())
        self._registry.save()
        logger.info("Renamed model %s -> %s", model_id, new_name.strip())
        return True

    def import_model(self) -> ModelInfo | None:
        """Import an external .pt model. Returns ModelInfo or None."""
        if not self._registry or not self._project:
            return None
        file_path, _ = QFileDialog.getOpenFileName(
            self._parent, "选择模型文件", "", "PyTorch 模型 (*.pt);;所有文件 (*)"
        )
        if not file_path:
            return None
        name, ok = QInputDialog.getText(self._parent, "模型名称", "请输入模型名称:")
        if not ok or not name.strip():
            return None
        tasks = ["detect", "classify", "pose"]
        task, ok = QInputDialog.getItem(self._parent, "任务类型", "选择任务类型:", tasks, 0, False)
        if not ok:
            return None
        p = Path(file_path)
        try:
            rel = p.relative_to(self._project.project_dir)
            model_path = str(rel)
        except ValueError:
            model_path = str(p)
        model_info = ModelInfo(
            name=name.strip(),
            path=model_path,
            task=task,
            base_model="imported",
            classes=self._project.config.classes,
        )
        self._registry.register(model_info)
        self._registry.save()
        logger.info("Imported model: %s", name.strip())
        return model_info

    def predict_single(self, img_path: Path, classes: list[str],
                       conf: float = 0.5, iou: float = 0.45) -> list:
        """Run single-image prediction. Returns annotations list."""
        if not self._predictor:
            QMessageBox.information(self._parent, "提示", "请先在模型面板中加载一个模型")
            return []
        try:
            return self._predictor.predict(
                img_path, conf=conf, iou=iou, project_classes=classes,
            )
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Auto-label failed: %s", e, exc_info=True)
            QMessageBox.warning(self._parent, "自动标注失败", str(e))
            return []
