"""Tests for dialogs."""
import os
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

        name, proj_dir, image_dir, classes = dlg.get_values()
        assert name == "test_project"
        assert proj_dir == str(tmp_path)
        assert image_dir == ""
        assert classes == ["cat", "dog", "bird"]

    def test_rejects_empty_name(self, qapp, tmp_path):
        from src.ui.dialogs import NewProjectDialog

        dlg = NewProjectDialog()
        dlg._name_edit.setText("")
        dlg._dir_edit.setText(str(tmp_path))
        dlg._validate_and_accept()
        assert "项目名称" in dlg._error_label.text()

    def test_rejects_invalid_chars_in_name(self, qapp, tmp_path):
        from src.ui.dialogs import NewProjectDialog

        dlg = NewProjectDialog()
        dlg._name_edit.setText("test<project")
        dlg._dir_edit.setText(str(tmp_path))
        dlg._validate_and_accept()
        assert "非法字符" in dlg._error_label.text()

    def test_rejects_long_name(self, qapp, tmp_path):
        from src.ui.dialogs import NewProjectDialog

        dlg = NewProjectDialog()
        dlg._name_edit.setText("x" * 65)
        dlg._dir_edit.setText(str(tmp_path))
        dlg._validate_and_accept()
        assert "64" in dlg._error_label.text()

    def test_rejects_nonexistent_dir(self, qapp):
        from src.ui.dialogs import NewProjectDialog

        dlg = NewProjectDialog()
        dlg._name_edit.setText("test")
        dlg._dir_edit.setText("/nonexistent/path/12345")
        dlg._validate_and_accept()
        assert "不存在" in dlg._error_label.text()

    def test_rejects_nonexistent_image_dir(self, qapp, tmp_path):
        from src.ui.dialogs import NewProjectDialog

        dlg = NewProjectDialog()
        dlg._name_edit.setText("test")
        dlg._dir_edit.setText(str(tmp_path))
        dlg._image_dir_edit.setText("/nonexistent/img/dir")
        dlg._validate_and_accept()
        assert "图片目录" in dlg._error_label.text()

    def test_accepts_valid_input(self, qapp, tmp_path):
        from src.ui.dialogs import NewProjectDialog

        dlg = NewProjectDialog()
        dlg._name_edit.setText("my_project")
        dlg._dir_edit.setText(str(tmp_path))
        dlg._validate_and_accept()
        assert dlg._error_label.text() == ""


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

    def test_rejects_nonexistent_output_dir(self, qapp):
        from src.ui.dialogs import ExportDialog

        dlg = ExportDialog()
        dlg._dir_edit.setText("/nonexistent/output/12345")
        dlg._validate_and_accept()
        assert "不存在" in dlg._error_label.text()

    def test_rejects_empty_output_dir(self, qapp):
        from src.ui.dialogs import ExportDialog

        dlg = ExportDialog()
        dlg._dir_edit.setText("")
        dlg._validate_and_accept()
        assert "输出目录" in dlg._error_label.text()

    def test_accepts_valid_dir(self, qapp, tmp_path):
        from src.ui.dialogs import ExportDialog

        dlg = ExportDialog()
        dlg._dir_edit.setText(str(tmp_path))
        dlg._validate_and_accept()
        assert dlg._error_label.text() == ""


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

    def test_rejects_long_class_name(self, qapp):
        from src.ui.dialogs import ClassManagerDialog

        dlg = ClassManagerDialog(classes=[], colors={})
        dlg._new_class_edit.setText("x" * 33)
        dlg._on_add()
        assert "32" in dlg._status_label.text()
        assert dlg._class_list.count() == 0

    def test_rejects_invalid_chars_in_class_name(self, qapp):
        from src.ui.dialogs import ClassManagerDialog

        dlg = ClassManagerDialog(classes=[], colors={})
        dlg._new_class_edit.setText("cat*dog")
        dlg._on_add()
        assert "非法字符" in dlg._status_label.text()
        assert dlg._class_list.count() == 0

    def test_rejects_duplicate_class(self, qapp):
        from src.ui.dialogs import ClassManagerDialog

        dlg = ClassManagerDialog(classes=["cat"], colors={})
        dlg._new_class_edit.setText("cat")
        dlg._on_add()
        assert "已存在" in dlg._status_label.text()
        assert dlg._class_list.count() == 1
