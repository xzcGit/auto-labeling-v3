"""Training panel — parameter config, training curves, and log display."""
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
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QPlainTextEdit,
    QSplitter,
    QCheckBox,
)
from PyQt5.QtCore import Qt, pyqtSignal

from src.engine.trainer import TrainConfig

logger = logging.getLogger(__name__)


class TrainPanel(QWidget):
    """Training panel with parameter configuration and monitoring.

    Signals:
        start_requested(TrainConfig): User clicked start training.
        stop_requested(): User clicked stop training.
    """

    start_requested = pyqtSignal(object)  # TrainConfig
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)

        # Left: parameter config
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)

        # Task type
        task_group = QGroupBox("任务配置")
        task_form = QFormLayout(task_group)
        self._task_combo = QComboBox()
        self._task_combo.addItems(["detect", "classify", "pose"])
        task_form.addRow("任务类型:", self._task_combo)

        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"])
        task_form.addRow("基础模型:", self._model_combo)

        self._device_combo = QComboBox()
        self._device_combo.setEditable(True)
        self._device_combo.addItems(["", "0", "1", "cpu"])
        task_form.addRow("设备:", self._device_combo)

        left_layout.addWidget(task_group)

        # Basic hyperparameters
        hyper_group = QGroupBox("训练参数")
        hyper_form = QFormLayout(hyper_group)

        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 10000)
        self._epochs_spin.setValue(100)
        hyper_form.addRow("Epochs:", self._epochs_spin)

        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 256)
        self._batch_spin.setValue(16)
        hyper_form.addRow("Batch:", self._batch_spin)

        self._imgsz_spin = QSpinBox()
        self._imgsz_spin.setRange(32, 4096)
        self._imgsz_spin.setSingleStep(32)
        self._imgsz_spin.setValue(640)
        hyper_form.addRow("ImgSz:", self._imgsz_spin)

        self._optimizer_combo = QComboBox()
        self._optimizer_combo.addItems(["auto", "SGD", "Adam", "AdamW"])
        hyper_form.addRow("优化器:", self._optimizer_combo)

        self._lr0_spin = QDoubleSpinBox()
        self._lr0_spin.setRange(0.0001, 1.0)
        self._lr0_spin.setDecimals(4)
        self._lr0_spin.setSingleStep(0.001)
        self._lr0_spin.setValue(0.01)
        hyper_form.addRow("学习率:", self._lr0_spin)

        self._val_ratio_spin = QDoubleSpinBox()
        self._val_ratio_spin.setRange(0.05, 0.5)
        self._val_ratio_spin.setDecimals(2)
        self._val_ratio_spin.setSingleStep(0.05)
        self._val_ratio_spin.setValue(0.2)
        hyper_form.addRow("验证集比例:", self._val_ratio_spin)

        left_layout.addWidget(hyper_group)

        # Data augmentation
        aug_group = QGroupBox("数据增强")
        aug_form = QFormLayout(aug_group)

        self._mosaic_spin = QDoubleSpinBox()
        self._mosaic_spin.setRange(0, 1)
        self._mosaic_spin.setDecimals(1)
        self._mosaic_spin.setValue(1.0)
        aug_form.addRow("Mosaic:", self._mosaic_spin)

        self._fliplr_spin = QDoubleSpinBox()
        self._fliplr_spin.setRange(0, 1)
        self._fliplr_spin.setDecimals(1)
        self._fliplr_spin.setValue(0.5)
        aug_form.addRow("FlipLR:", self._fliplr_spin)

        self._flipud_spin = QDoubleSpinBox()
        self._flipud_spin.setRange(0, 1)
        self._flipud_spin.setDecimals(1)
        self._flipud_spin.setValue(0.0)
        aug_form.addRow("FlipUD:", self._flipud_spin)

        left_layout.addWidget(aug_group)

        # Pose-specific params
        self._pose_group = QGroupBox("Pose 参数")
        pose_form = QFormLayout(self._pose_group)

        self._kpt_num_spin = QSpinBox()
        self._kpt_num_spin.setRange(1, 100)
        self._kpt_num_spin.setValue(17)
        pose_form.addRow("关键点数:", self._kpt_num_spin)

        self._pose_weight_spin = QDoubleSpinBox()
        self._pose_weight_spin.setRange(0, 100)
        self._pose_weight_spin.setValue(12.0)
        pose_form.addRow("Pose权重:", self._pose_weight_spin)

        self._kobj_spin = QDoubleSpinBox()
        self._kobj_spin.setRange(0, 10)
        self._kobj_spin.setValue(1.0)
        pose_form.addRow("Kobj权重:", self._kobj_spin)

        self._pose_group.setVisible(False)
        left_layout.addWidget(self._pose_group)

        # Resume checkbox
        self._resume_check = QCheckBox("断点续训 (Resume)")
        left_layout.addWidget(self._resume_check)

        # Buttons
        btn_layout = QHBoxLayout()
        self._btn_start = QPushButton("开始训练")
        self._btn_start.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; font-weight: bold;")
        self._btn_stop = QPushButton("停止训练")
        self._btn_stop.setEnabled(False)
        btn_layout.addWidget(self._btn_start)
        btn_layout.addWidget(self._btn_stop)
        left_layout.addLayout(btn_layout)

        left_layout.addStretch()

        # Right: curves + log
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)

        # Placeholder for pyqtgraph curves
        self._curve_label = QLabel("训练曲线 (训练开始后显示)")
        self._curve_label.setAlignment(Qt.AlignCenter)
        self._curve_label.setStyleSheet("color: #6c7086; font-size: 14px; min-height: 200px; border: 1px solid #313244; border-radius: 4px;")
        right_layout.addWidget(self._curve_label)

        # Try to create pyqtgraph widget
        self._plot_widget = None
        try:
            import pyqtgraph as pg
            pg.setConfigOptions(background="#1e1e2e", foreground="#cdd6f4")
            self._plot_widget = pg.PlotWidget(title="Loss / mAP")
            self._plot_widget.setLabel("bottom", "Epoch")
            self._plot_widget.addLegend()
            self._train_loss_curve = self._plot_widget.plot([], [], pen=pg.mkPen("#f38ba8", width=2), name="Train Loss")
            self._val_loss_curve = self._plot_widget.plot([], [], pen=pg.mkPen("#89b4fa", width=2), name="Val Loss")
            self._map_curve = self._plot_widget.plot([], [], pen=pg.mkPen("#a6e3a1", width=2), name="mAP50")
            right_layout.replaceWidget(self._curve_label, self._plot_widget)
            self._curve_label.hide()
        except ImportError:
            pass

        # Epoch data for curves
        self._epoch_data: list[dict] = []

        # Log
        log_label = QLabel("训练日志")
        log_label.setStyleSheet("color: #a6adc8; font-size: 12px;")
        right_layout.addWidget(log_label)

        self._log_text = QPlainTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumBlockCount(1000)
        right_layout.addWidget(self._log_text)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([350, 600])
        layout.addWidget(splitter)

    def _connect_signals(self) -> None:
        self._task_combo.currentTextChanged.connect(self._on_task_changed)
        self._btn_start.clicked.connect(self._on_start)
        self._btn_stop.clicked.connect(self._on_stop)

    def _on_task_changed(self, task: str) -> None:
        self._pose_group.setVisible(task == "pose")

    def _on_start(self) -> None:
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._log_text.clear()
        self._epoch_data.clear()

    def _on_stop(self) -> None:
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self.stop_requested.emit()

    def get_train_config(self, data_yaml: str, model: str | None = None) -> TrainConfig:
        """Build TrainConfig from current UI values."""
        return TrainConfig(
            data_yaml=data_yaml,
            model=model or self._model_combo.currentText(),
            task=self._task_combo.currentText(),
            epochs=self._epochs_spin.value(),
            batch=self._batch_spin.value(),
            imgsz=self._imgsz_spin.value(),
            device=self._device_combo.currentText(),
            optimizer=self._optimizer_combo.currentText(),
            lr0=self._lr0_spin.value(),
            mosaic=self._mosaic_spin.value(),
            fliplr=self._fliplr_spin.value(),
            flipud=self._flipud_spin.value(),
            pose=self._pose_weight_spin.value(),
            kobj=self._kobj_spin.value(),
            resume=self._resume_check.isChecked(),
        )

    def get_val_ratio(self) -> float:
        """Get validation split ratio."""
        return self._val_ratio_spin.value()

    def append_log(self, text: str) -> None:
        """Append text to training log."""
        self._log_text.appendPlainText(text)

    def update_epoch(self, metrics: dict) -> None:
        """Update curves and log with epoch metrics."""
        self._epoch_data.append(metrics)
        epoch = metrics.get("epoch", len(self._epoch_data) - 1)

        # Log line
        parts = [f"Epoch {epoch}"]
        for k, v in metrics.items():
            if k != "epoch":
                parts.append(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        self.append_log(" | ".join(parts))

        # Update curves
        if self._plot_widget:
            epochs = list(range(len(self._epoch_data)))
            train_losses = [d.get("train_loss", 0) for d in self._epoch_data]
            val_losses = [d.get("val_loss", 0) for d in self._epoch_data]
            maps = [d.get("metrics/mAP50(B)", d.get("mAP50", 0)) for d in self._epoch_data]
            self._train_loss_curve.setData(epochs, train_losses)
            self._val_loss_curve.setData(epochs, val_losses)
            self._map_curve.setData(epochs, maps)

    def on_training_finished(self, metrics: dict) -> None:
        """Handle training completion."""
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self.append_log("--- 训练完成 ---")
        if metrics:
            for k, v in metrics.items():
                self.append_log(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    def on_training_error(self, error_msg: str) -> None:
        """Handle training error."""
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self.append_log(f"--- 训练失败: {error_msg} ---")
