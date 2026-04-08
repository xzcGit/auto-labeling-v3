"""Dialogs — new project, export, class management."""
from __future__ import annotations

import os
import re
from pathlib import Path

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
    QProgressBar,
    QMessageBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

_ERROR_STYLE = "color: #f38ba8; font-size: 11px;"
_INVALID_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MAX_PROJECT_NAME_LEN = 64
_MAX_CLASS_NAME_LEN = 32


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

        img_dir_layout = QHBoxLayout()
        self._image_dir_edit = QLineEdit()
        self._image_dir_edit.setPlaceholderText("留空则在项目目录下创建 images/")
        img_dir_layout.addWidget(self._image_dir_edit)
        btn_img_browse = QPushButton("浏览...")
        btn_img_browse.clicked.connect(self._browse_image_dir)
        img_dir_layout.addWidget(btn_img_browse)
        form.addRow("图片目录:", img_dir_layout)

        self._classes_edit = QLineEdit()
        self._classes_edit.setPlaceholderText("逗号分隔，如: cat, dog, bird")
        form.addRow("初始类别:", self._classes_edit)

        layout.addLayout(form)

        # Validation error label
        self._error_label = QLabel("")
        self._error_label.setStyleSheet(_ERROR_STYLE)
        self._error_label.setWordWrap(True)
        layout.addWidget(self._error_label)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate_and_accept(self) -> None:
        """Validate inputs before accepting."""
        name = self._name_edit.text().strip()
        proj_dir = self._dir_edit.text().strip()
        if not name:
            self._error_label.setText("请输入项目名称")
            self._name_edit.setFocus()
            return
        if len(name) > _MAX_PROJECT_NAME_LEN:
            self._error_label.setText(f"项目名称不能超过{_MAX_PROJECT_NAME_LEN}个字符")
            self._name_edit.setFocus()
            return
        if _INVALID_NAME_CHARS.search(name):
            self._error_label.setText("项目名称包含非法字符")
            self._name_edit.setFocus()
            return
        if not proj_dir:
            self._error_label.setText("请选择项目目录")
            self._dir_edit.setFocus()
            return
        proj_path = Path(proj_dir)
        if not proj_path.exists():
            self._error_label.setText("项目目录不存在")
            self._dir_edit.setFocus()
            return
        if not os.access(str(proj_path), os.W_OK):
            self._error_label.setText("项目目录没有写入权限")
            self._dir_edit.setFocus()
            return
        # Validate image dir if provided
        image_dir = self._image_dir_edit.text().strip()
        if image_dir:
            img_path = Path(image_dir)
            if not img_path.exists():
                self._error_label.setText("图片目录不存在")
                self._image_dir_edit.setFocus()
                return
        self._error_label.setText("")
        self.accept()

    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择项目目录")
        if path:
            self._dir_edit.setText(path)

    def _browse_image_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if path:
            self._image_dir_edit.setText(path)

    def get_values(self) -> tuple[str, str, str, list[str]]:
        """Return (name, project_dir, image_dir, classes)."""
        name = self._name_edit.text().strip()
        proj_dir = self._dir_edit.text().strip()
        image_dir = self._image_dir_edit.text().strip()
        classes_text = self._classes_edit.text().strip()
        classes = [c.strip() for c in classes_text.split(",") if c.strip()] if classes_text else []
        return name, proj_dir, image_dir, classes


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
        from src.core.formats import get_export_registry
        self._format_combo.addItems(get_export_registry().list_names())
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

        # Validation error label
        self._error_label = QLabel("")
        self._error_label.setStyleSheet(_ERROR_STYLE)
        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate_and_accept(self) -> None:
        """Validate inputs before accepting."""
        out_dir = self._dir_edit.text().strip()
        if not out_dir:
            self._error_label.setText("请选择输出目录")
            self._dir_edit.setFocus()
            return
        out_path = Path(out_dir)
        if not out_path.exists():
            self._error_label.setText("输出目录不存在")
            self._dir_edit.setFocus()
            return
        if not os.access(out_dir, os.W_OK):
            self._error_label.setText("输出目录没有写入权限")
            self._dir_edit.setFocus()
            return
        self._error_label.setText("")
        self.accept()

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
        self._new_class_edit.returnPressed.connect(self._on_add)
        add_layout.addWidget(self._new_class_edit)
        btn_add = QPushButton("添加")
        btn_add.clicked.connect(self._on_add)
        add_layout.addWidget(btn_add)
        layout.addLayout(add_layout)

        # Status label for duplicate warning
        self._status_label = QLabel("")
        self._status_label.setStyleSheet(_ERROR_STYLE)
        layout.addWidget(self._status_label)

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
        if not name:
            return
        if len(name) > _MAX_CLASS_NAME_LEN:
            self._status_label.setText(f"类别名称不能超过{_MAX_CLASS_NAME_LEN}个字符")
            return
        if _INVALID_NAME_CHARS.search(name):
            self._status_label.setText("类别名称包含非法字符")
            return
        if name in self._classes:
            self._status_label.setText(f"类别 \"{name}\" 已存在")
            return
        self._status_label.setText("")
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
            self._status_label.setText("")

    def get_classes(self) -> list[str]:
        """Return the current class list."""
        return list(self._classes)


class BatchProgressDialog(QDialog):
    """Progress dialog for batch operations with cancel support."""

    def __init__(self, title: str, total: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        self.setModal(True)
        self._cancelled = False

        layout = QVBoxLayout(self)

        self._info_label = QLabel(f"处理中: 0/{total}")
        layout.addWidget(self._info_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, total)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        self._detail_label = QLabel("")
        self._detail_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        layout.addWidget(self._detail_label)

        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self._on_cancel)
        layout.addWidget(btn_cancel)

    def update_progress(self, current: int, total: int) -> None:
        """Update progress bar and label."""
        self._progress.setMaximum(total)
        self._progress.setValue(current)
        self._info_label.setText(f"处理中: {current}/{total}")

    def set_detail(self, text: str) -> None:
        """Set detail text (e.g. current file name)."""
        self._detail_label.setText(text)

    def _on_cancel(self) -> None:
        self._cancelled = True
        self._info_label.setText("正在取消...")

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled
