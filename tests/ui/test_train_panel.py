"""Tests for TrainPanel."""
import pytest
from PyQt5.QtCore import Qt


class TestTrainPanel:
    def test_creates(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        assert panel is not None

    def test_has_task_selector(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        assert panel._task_combo is not None
        # Should have detect, classify, pose
        items = [panel._task_combo.itemText(i) for i in range(panel._task_combo.count())]
        assert "detect" in items
        assert "classify" in items
        assert "pose" in items

    def test_has_hyperparameters(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        assert panel._epochs_spin is not None
        assert panel._batch_spin is not None
        assert panel._imgsz_spin is not None
        assert panel._lr0_spin is not None

    def test_has_start_stop_buttons(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        assert panel._btn_start is not None
        assert panel._btn_stop is not None

    def test_has_log_display(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        assert panel._log_text is not None

    def test_get_train_config(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel._task_combo.setCurrentText("detect")
        panel._epochs_spin.setValue(50)
        panel._batch_spin.setValue(8)
        panel._imgsz_spin.setValue(320)

        config = panel.get_train_config(data_yaml="/tmp/data.yaml", model="yolov8n.pt")
        assert config.task == "detect"
        assert config.epochs == 50
        assert config.batch == 8
        assert config.imgsz == 320
        assert config.data_yaml == "/tmp/data.yaml"

    def test_append_log(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel.append_log("Epoch 1/50")
        assert "Epoch 1/50" in panel._log_text.toPlainText()

    def test_update_epoch_metrics(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel.update_epoch({"epoch": 0, "train_loss": 1.5})
        assert "epoch: 0" in panel._log_text.toPlainText().lower() or "Epoch" in panel._log_text.toPlainText()

    def test_pose_params_visible_on_pose_task(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel._task_combo.setCurrentText("pose")
        assert not panel._pose_group.isHidden()

    def test_pose_params_hidden_on_detect_task(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel._task_combo.setCurrentText("detect")
        assert panel._pose_group.isHidden()

    def test_has_preset_combo(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        assert panel._preset_combo is not None
        assert panel._preset_combo.count() >= 3

    def test_preset_fast_changes_epochs(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel._preset_combo.setCurrentText("快速验证")
        assert panel._epochs_spin.value() == 10
        assert panel._batch_spin.value() == 32

    def test_preset_accurate_changes_epochs(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel._preset_combo.setCurrentText("高精度")
        assert panel._epochs_spin.value() == 300
        assert panel._batch_spin.value() == 8

    def test_preset_default_restores(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel._preset_combo.setCurrentText("快速验证")
        panel._preset_combo.setCurrentText("默认")
        assert panel._epochs_spin.value() == 100
        assert panel._batch_spin.value() == 16
