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
    QDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

from src.engine.model_manager import ModelInfo
from src.ui.icons import icon

logger = logging.getLogger(__name__)


class ModelPanel(QWidget):
    """Model management panel.

    Signals:
        model_load_requested(str): Request to load model by ID.
        model_delete_requested(str): Request to delete model by ID.
    """

    model_load_requested = pyqtSignal(str)
    model_delete_requested = pyqtSignal(str)
    model_rename_requested = pyqtSignal(str)
    model_import_requested = pyqtSignal()  # Request to import external .pt file

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
        self._model_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        list_layout.addWidget(self._model_list)

        btn_layout = QHBoxLayout()
        self._btn_load = QPushButton(icon("load_model"), "加载模型")
        self._btn_load.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e;")
        self._btn_load.setToolTip("加载选中模型用于自动标注")
        self._btn_delete = QPushButton(icon("delete"), "删除模型")
        self._btn_delete.setStyleSheet("background-color: #f38ba8; color: #1e1e2e;")
        self._btn_delete.setToolTip("从注册表中删除选中模型")
        self._btn_rename = QPushButton("重命名")
        self._btn_rename.setToolTip("修改选中模型的显示名称")
        self._btn_import = QPushButton(icon("import"), "导入模型")
        self._btn_import.setToolTip("从文件导入外部 .pt 模型")
        self._btn_compare = QPushButton("对比模型")
        self._btn_compare.setToolTip("对比选中模型的指标")
        btn_layout.addWidget(self._btn_load)
        btn_layout.addWidget(self._btn_delete)
        btn_layout.addWidget(self._btn_rename)
        btn_layout.addWidget(self._btn_import)
        btn_layout.addWidget(self._btn_compare)
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

        self._overlap_iou_spin = QDoubleSpinBox()
        self._overlap_iou_spin.setRange(0.01, 1.0)
        self._overlap_iou_spin.setDecimals(2)
        self._overlap_iou_spin.setSingleStep(0.05)
        self._overlap_iou_spin.setValue(0.5)
        self._overlap_iou_spin.setToolTip("预测框与已确认框重叠超过此阈值时触发冲突二选一")
        auto_form.addRow("重叠 IoU 阈值:", self._overlap_iou_spin)

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
        self._btn_rename.clicked.connect(self._on_rename)
        self._btn_import.clicked.connect(lambda: self.model_import_requested.emit())
        self._btn_compare.clicked.connect(self._on_compare)

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
        logger.info("Model list updated: %d models", len(models))

    def set_current_model_name(self, name: str) -> None:
        """Display the currently loaded model name."""
        self._current_label.setText(f"当前模型: {name}")

    def get_conf_threshold(self) -> float:
        return self._conf_spin.value()

    def get_iou_threshold(self) -> float:
        return self._iou_spin.value()

    def get_overlap_iou_threshold(self) -> float:
        return self._overlap_iou_spin.value()

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

    def _on_rename(self) -> None:
        model_id = self._get_selected_id()
        if model_id:
            self.model_rename_requested.emit(model_id)

    def _on_compare(self) -> None:
        """Show comparison dialog for selected models."""
        selected_ids = []
        for item in self._model_list.selectedItems():
            mid = item.data(Qt.UserRole)
            if mid:
                selected_ids.append(mid)
        if len(selected_ids) < 2:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "提示", "请选择至少 2 个模型进行对比")
            return
        models = [m for m in self._models if m.id in selected_ids]
        dlg = ModelCompareDialog(models, self)
        dlg.exec_()


class ModelCompareDialog(QDialog):
    """Dialog showing a table comparison of model metrics."""

    def __init__(self, models: list[ModelInfo], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"模型对比 ({len(models)} 个模型)")
        self.setMinimumSize(600, 400)
        self._models = models
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Collect all metric keys across all models
        all_keys: list[str] = []
        for m in self._models:
            for k in m.metrics:
                if k not in all_keys:
                    all_keys.append(k)

        # Fixed columns + metric columns
        fixed_cols = ["名称", "任务", "基础模型", "Epochs", "数据集"]
        headers = fixed_cols + all_keys
        n_rows = len(self._models)
        n_cols = len(headers)

        table = QTableWidget(n_rows, n_cols)
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setAlternatingRowColors(True)

        for row, model in enumerate(self._models):
            table.setItem(row, 0, QTableWidgetItem(model.name))
            table.setItem(row, 1, QTableWidgetItem(model.task))
            table.setItem(row, 2, QTableWidgetItem(model.base_model))
            table.setItem(row, 3, QTableWidgetItem(str(model.epochs)))
            table.setItem(row, 4, QTableWidgetItem(str(model.dataset_size)))

            for ci, key in enumerate(all_keys):
                val = model.metrics.get(key)
                text = f"{val:.4f}" if val is not None else "-"
                item = QTableWidgetItem(text)
                # Highlight best value in each metric column
                if val is not None:
                    item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row, len(fixed_cols) + ci, item)

        # Highlight best values per metric column
        for ci, key in enumerate(all_keys):
            col_idx = len(fixed_cols) + ci
            values = []
            for row in range(n_rows):
                val = self._models[row].metrics.get(key)
                values.append(val)
            # Best = max for most metrics (mAP, precision, recall), min for loss
            is_loss = "loss" in key.lower()
            best_val = None
            for v in values:
                if v is not None:
                    if best_val is None:
                        best_val = v
                    elif is_loss and v < best_val:
                        best_val = v
                    elif not is_loss and v > best_val:
                        best_val = v
            if best_val is not None:
                for row in range(n_rows):
                    if values[row] == best_val:
                        item = table.item(row, col_idx)
                        if item:
                            item.setForeground(QColor("#a6e3a1"))

        layout.addWidget(table)

        # Summary
        summary = QLabel(f"共 {len(self._models)} 个模型 | 绿色标记为各指标最优值")
        summary.setStyleSheet("color: #a6adc8; font-size: 11px;")
        layout.addWidget(summary)
