"""Class picker popup for annotation class assignment."""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor


class ClassPickerPopup(QDialog):
    """Popup dialog for selecting an annotation class.

    Shows a list of classes with color indicators.
    User can click or press Enter to confirm selection.
    """

    def __init__(
        self,
        classes: list[str],
        colors: dict[str, str],
        default_class: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("选择类别")
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setMinimumWidth(160)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        header = QLabel("选择类别:")
        header.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 2px;")
        layout.addWidget(header)

        self._list = QListWidget()
        self._list.setMaximumHeight(200)

        default_row = 0
        for i, cls_name in enumerate(classes):
            item = QListWidgetItem(cls_name)
            color = colors.get(cls_name, "#89b4fa")
            item.setForeground(QColor(color))
            self._list.addItem(item)
            if cls_name == default_class:
                default_row = i

        if classes:
            self._list.setCurrentRow(default_row)

        self._list.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self._list)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.accept()
        elif event.key() == Qt.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)

    def get_selected_class(self) -> str | None:
        """Return the selected class name, or None."""
        item = self._list.currentItem()
        return item.text() if item else None

    def get_selected_index(self) -> int:
        """Return the selected class index, or -1."""
        return self._list.currentRow()
