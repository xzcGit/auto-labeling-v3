"""Tests for dialogs."""
import pytest
from pathlib import Path
from PyQt5.QtCore import Qt


class TestNewProjectDialog:
    def test_creates(self, qapp):
        from src.ui.dialogs import NewProjectDialog

        dlg = NewProjectDialog()
        assert dlg is not None

    def test_has_name_field(self, qapp):
        from src.ui.dialogs import NewProjectDialog

        dlg = NewProjectDialog()
        assert dlg._name_edit is not None

    def test_has_classes_field(self, qapp):
        from src.ui.dialogs import NewProjectDialog

        dlg = NewProjectDialog()
        assert dlg._classes_edit is not None

    def test_get_values(self, qapp, tmp_path):
        from src.ui.dialogs import NewProjectDialog

        dlg = NewProjectDialog()
        dlg._name_edit.setText("test_project")
        dlg._dir_edit.setText(str(tmp_path))
        dlg._classes_edit.setText("cat, dog, bird")

        name, proj_dir, classes = dlg.get_values()
        assert name == "test_project"
        assert proj_dir == str(tmp_path)
        assert classes == ["cat", "dog", "bird"]


class TestExportDialog:
    def test_creates(self, qapp):
        from src.ui.dialogs import ExportDialog

        dlg = ExportDialog()
        assert dlg is not None

    def test_has_format_selector(self, qapp):
        from src.ui.dialogs import ExportDialog

        dlg = ExportDialog()
        items = [dlg._format_combo.itemText(i) for i in range(dlg._format_combo.count())]
        assert "YOLO" in items
        assert "COCO" in items
        assert "labelme" in items

    def test_get_values(self, qapp, tmp_path):
        from src.ui.dialogs import ExportDialog

        dlg = ExportDialog()
        dlg._format_combo.setCurrentText("COCO")
        dlg._dir_edit.setText(str(tmp_path))
        dlg._confirmed_only.setChecked(True)

        fmt, out_dir, only_confirmed = dlg.get_values()
        assert fmt == "COCO"
        assert out_dir == str(tmp_path)
        assert only_confirmed is True


class TestClassManagerDialog:
    def test_creates(self, qapp):
        from src.ui.dialogs import ClassManagerDialog

        dlg = ClassManagerDialog(classes=["cat", "dog"], colors={"cat": "#a6e3a1"})
        assert dlg._class_list.count() == 2

    def test_add_class(self, qapp):
        from src.ui.dialogs import ClassManagerDialog

        dlg = ClassManagerDialog(classes=["cat"], colors={})
        dlg._new_class_edit.setText("dog")
        dlg._on_add()
        assert dlg._class_list.count() == 2

    def test_get_classes(self, qapp):
        from src.ui.dialogs import ClassManagerDialog

        dlg = ClassManagerDialog(classes=["cat", "dog"], colors={})
        result = dlg.get_classes()
        assert result == ["cat", "dog"]
