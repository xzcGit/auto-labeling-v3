"""Tests for ModelPanel."""
import pytest
from PyQt5.QtCore import Qt


class TestModelPanel:
    def test_creates(self, qapp):
        from src.ui.model_panel import ModelPanel

        panel = ModelPanel()
        assert panel is not None

    def test_has_model_list(self, qapp):
        from src.ui.model_panel import ModelPanel

        panel = ModelPanel()
        assert panel._model_list is not None

    def test_set_models(self, qapp):
        from src.ui.model_panel import ModelPanel
        from src.engine.model_manager import ModelInfo

        panel = ModelPanel()
        models = [
            ModelInfo(name="det-v1", path="models/det/best.pt", task="detect",
                      base_model="yolov8n.pt", classes=["cat", "dog"]),
            ModelInfo(name="pose-v1", path="models/pose/best.pt", task="pose",
                      base_model="yolov8n-pose.pt", classes=["person"]),
        ]
        panel.set_models(models)
        assert panel._model_list.count() == 2

    def test_has_threshold_controls(self, qapp):
        from src.ui.model_panel import ModelPanel

        panel = ModelPanel()
        assert panel._conf_spin is not None
        assert panel._iou_spin is not None

    def test_get_thresholds(self, qapp):
        from src.ui.model_panel import ModelPanel

        panel = ModelPanel()
        panel._conf_spin.setValue(0.6)
        panel._iou_spin.setValue(0.5)
        assert panel.get_conf_threshold() == 0.6
        assert panel.get_iou_threshold() == 0.5

    def test_has_load_delete_buttons(self, qapp):
        from src.ui.model_panel import ModelPanel

        panel = ModelPanel()
        assert panel._btn_load is not None
        assert panel._btn_delete is not None

    def test_has_compare_button(self, qapp):
        from src.ui.model_panel import ModelPanel

        panel = ModelPanel()
        assert panel._btn_compare is not None

    def test_model_list_multi_select(self, qapp):
        from src.ui.model_panel import ModelPanel
        from PyQt5.QtWidgets import QAbstractItemView

        panel = ModelPanel()
        assert panel._model_list.selectionMode() == QAbstractItemView.ExtendedSelection


class TestModelCompareDialog:
    def test_creates_with_models(self, qapp):
        from src.ui.model_panel import ModelCompareDialog
        from src.engine.model_manager import ModelInfo

        m1 = ModelInfo(name="v1", path="m1.pt", task="detect", base_model="yolov8n.pt",
                       classes=["cat"], metrics={"mAP50": 0.85, "mAP50-95": 0.65}, epochs=100)
        m2 = ModelInfo(name="v2", path="m2.pt", task="detect", base_model="yolov8s.pt",
                       classes=["cat"], metrics={"mAP50": 0.90, "mAP50-95": 0.70}, epochs=200)
        dlg = ModelCompareDialog([m1, m2])
        assert dlg.windowTitle().startswith("模型对比")

    def test_handles_missing_metrics(self, qapp):
        from src.ui.model_panel import ModelCompareDialog
        from src.engine.model_manager import ModelInfo

        m1 = ModelInfo(name="v1", path="m1.pt", task="detect", base_model="yolov8n.pt",
                       classes=["cat"], metrics={"mAP50": 0.85})
        m2 = ModelInfo(name="v2", path="m2.pt", task="detect", base_model="yolov8s.pt",
                       classes=["cat"], metrics={"recall": 0.9})
        dlg = ModelCompareDialog([m1, m2])
        # Should not crash with mismatched metric keys
        assert dlg is not None
