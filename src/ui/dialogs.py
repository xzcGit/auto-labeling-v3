"""Dialogs — new project, export, class management."""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QCheckBox,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QGroupBox,
    QDialogButtonBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor


class NewProjectDialog(QDialog):
    """Dialog for creating a new project."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("新建项目")
        self.setMinimumWidth(400)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("项目名称")
        form.addRow("项目名称:", self._name_edit)

        dir_layout = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("选择项目目录")
        dir_layout.addWidget(self._dir_edit)
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse_dir)
        dir_layout.addWidget(btn_browse)
        form.addRow("项目目录:", dir_layout)

        self._image_dir_edit = QLineEdit("images")
        form.addRow("图片目录:", self._image_dir_edit)

        self._classes_edit = QLineEdit()
        self._classes_edit.setPlaceholderText("逗号分隔，如: cat, dog, bird")
        form.addRow("初始类别:", self._classes_edit)

        layout.addLayout(form)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择项目目录")
        if path:
            self._dir_edit.setText(path)

    def get_values(self) -> tuple[str, str, list[str]]:
        """Return (name, project_dir, classes)."""
        name = self._name_edit.text().strip()
        proj_dir = self._dir_edit.text().strip()
        classes_text = self._classes_edit.text().strip()
        classes = [c.strip() for c in classes_text.split(",") if c.strip()] if classes_text else []
        return name, proj_dir, classes


class ExportDialog(QDialog):
    """Dialog for exporting annotations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导出标注")
        self.setMinimumWidth(400)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self._format_combo = QComboBox()
        self._format_combo.addItems(["YOLO", "COCO", "labelme"])
        form.addRow("导出格式:", self._format_combo)

        dir_layout = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("选择输出目录")
        dir_layout.addWidget(self._dir_edit)
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse_dir)
        dir_layout.addWidget(btn_browse)
        form.addRow("输出目录:", dir_layout)

        self._confirmed_only = QCheckBox("仅导出已确认标注")
        self._confirmed_only.setChecked(False)
        form.addRow("", self._confirmed_only)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self._dir_edit.setText(path)

    def get_values(self) -> tuple[str, str, bool]:
        """Return (format, output_dir, only_confirmed)."""
        return (
            self._format_combo.currentText(),
            self._dir_edit.text().strip(),
            self._confirmed_only.isChecked(),
        )


class ClassManagerDialog(QDialog):
    """Dialog for managing project classes."""

    def __init__(self, classes: list[str], colors: dict[str, str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("类别管理")
        self.setMinimumWidth(350)
        self._classes = list(classes)
        self._colors = dict(colors)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Class list
        self._class_list = QListWidget()
        for cls in self._classes:
            color = self._colors.get(cls, "#89b4fa")
            item = QListWidgetItem(cls)
            item.setForeground(QColor(color))
            self._class_list.addItem(item)
        layout.addWidget(self._class_list)

        # Add class
        add_layout = QHBoxLayout()
        self._new_class_edit = QLineEdit()
        self._new_class_edit.setPlaceholderText("新类别名称")
        add_layout.addWidget(self._new_class_edit)
        btn_add = QPushButton("添加")
        btn_add.clicked.connect(self._on_add)
        add_layout.addWidget(btn_add)
        layout.addLayout(add_layout)

        # Remove button
        btn_remove = QPushButton("删除选中类别")
        btn_remove.clicked.connect(self._on_remove)
        layout.addWidget(btn_remove)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_add(self) -> None:
        name = self._new_class_edit.text().strip()
        if name and name not in self._classes:
            self._classes.append(name)
            item = QListWidgetItem(name)
            item.setForeground(QColor("#89b4fa"))
            self._class_list.addItem(item)
            self._new_class_edit.clear()

    def _on_remove(self) -> None:
        row = self._class_list.currentRow()
        if row >= 0:
            cls = self._classes.pop(row)
            self._colors.pop(cls, None)
            self._class_list.takeItem(row)

    def get_classes(self) -> list[str]:
        """Return the current class list."""
        return list(self._classes)
