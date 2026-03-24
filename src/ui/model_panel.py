"""Model management panel — model list, load/switch, auto-label settings."""
from __future__ import annotations

import logging

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QDoubleSpinBox,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

from src.engine.model_manager import ModelInfo

logger = logging.getLogger(__name__)


class ModelPanel(QWidget):
    """Model management panel.

    Signals:
        model_load_requested(str): Request to load model by ID.
        model_delete_requested(str): Request to delete model by ID.
    """

    model_load_requested = pyqtSignal(str)
    model_delete_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._models: list[ModelInfo] = []
        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        # Model list
        list_group = QGroupBox("已注册模型")
        list_layout = QVBoxLayout(list_group)

        self._model_list = QListWidget()
        list_layout.addWidget(self._model_list)

        btn_layout = QHBoxLayout()
        self._btn_load = QPushButton("加载模型")
        self._btn_load.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e;")
        self._btn_delete = QPushButton("删除模型")
        self._btn_delete.setStyleSheet("background-color: #f38ba8; color: #1e1e2e;")
        btn_layout.addWidget(self._btn_load)
        btn_layout.addWidget(self._btn_delete)
        list_layout.addLayout(btn_layout)

        layout.addWidget(list_group)

        # Model details
        self._detail_group = QGroupBox("模型详情")
        detail_layout = QFormLayout(self._detail_group)
        self._detail_name = QLabel("")
        self._detail_task = QLabel("")
        self._detail_base = QLabel("")
        self._detail_classes = QLabel("")
        self._detail_metrics = QLabel("")
        self._detail_trained = QLabel("")
        self._detail_epochs = QLabel("")
        self._detail_dataset = QLabel("")

        for label_name, widget in [
            ("名称:", self._detail_name),
            ("任务:", self._detail_task),
            ("基础模型:", self._detail_base),
            ("类别:", self._detail_classes),
            ("指标:", self._detail_metrics),
            ("训练时间:", self._detail_trained),
            ("Epochs:", self._detail_epochs),
            ("数据集:", self._detail_dataset),
        ]:
            widget.setStyleSheet("color: #cdd6f4;")
            widget.setWordWrap(True)
            detail_layout.addRow(label_name, widget)

        layout.addWidget(self._detail_group)

        # Auto-label settings
        auto_group = QGroupBox("自动标注设置")
        auto_form = QFormLayout(auto_group)

        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.01, 1.0)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.5)
        auto_form.addRow("置信度阈值:", self._conf_spin)

        self._iou_spin = QDoubleSpinBox()
        self._iou_spin.setRange(0.01, 1.0)
        self._iou_spin.setDecimals(2)
        self._iou_spin.setSingleStep(0.05)
        self._iou_spin.setValue(0.45)
        auto_form.addRow("IoU 阈值:", self._iou_spin)

        layout.addWidget(auto_group)

        # Current model indicator
        self._current_label = QLabel("当前模型: 无")
        self._current_label.setStyleSheet("color: #f9e2af; font-size: 13px; font-weight: bold;")
        layout.addWidget(self._current_label)

        layout.addStretch()

    def _connect_signals(self) -> None:
        self._model_list.currentRowChanged.connect(self._on_model_selected)
        self._btn_load.clicked.connect(self._on_load)
        self._btn_delete.clicked.connect(self._on_delete)

    def set_models(self, models: list[ModelInfo]) -> None:
        """Update the model list."""
        self._models = list(models)
        self._model_list.blockSignals(True)
        self._model_list.clear()

        task_colors = {"detect": "#89b4fa", "classify": "#a6e3a1", "pose": "#cba6f7"}
        for model in models:
            color = task_colors.get(model.task, "#cdd6f4")
            item = QListWidgetItem(f"[{model.task}] {model.name}")
            item.setData(Qt.UserRole, model.id)
            item.setForeground(QColor(color))
            self._model_list.addItem(item)

        self._model_list.blockSignals(False)

    def set_current_model_name(self, name: str) -> None:
        """Display the currently loaded model name."""
        self._current_label.setText(f"当前模型: {name}")

    def get_conf_threshold(self) -> float:
        return self._conf_spin.value()

    def get_iou_threshold(self) -> float:
        return self._iou_spin.value()

    def _on_model_selected(self, row: int) -> None:
        if 0 <= row < len(self._models):
            model = self._models[row]
            self._detail_name.setText(model.name)
            self._detail_task.setText(model.task)
            self._detail_base.setText(model.base_model)
            self._detail_classes.setText(", ".join(model.classes))
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in model.metrics.items())
            self._detail_metrics.setText(metrics_str or "无")
            self._detail_trained.setText(model.trained_at)
            self._detail_epochs.setText(str(model.epochs))
            self._detail_dataset.setText(f"{model.dataset_size} 张")

    def _get_selected_id(self) -> str | None:
        item = self._model_list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def _on_load(self) -> None:
        model_id = self._get_selected_id()
        if model_id:
            self.model_load_requested.emit(model_id)

    def _on_delete(self) -> None:
        model_id = self._get_selected_id()
        if model_id:
            self.model_delete_requested.emit(model_id)
