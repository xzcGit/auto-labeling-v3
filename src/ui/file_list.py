"""File list widget for image navigation and status display."""
from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QListWidget, QListWidgetItem
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

# Status colors (Catppuccin Mocha)
STATUS_COLORS = {
    "confirmed": "#a6e3a1",   # green
    "pending": "#f9e2af",     # yellow
    "unlabeled": "#6c7086",   # overlay0 (gray)
}

STATUS_ICONS = {
    "confirmed": "\u2713",    # ✓
    "pending": "\u26a1",      # ⚡
    "unlabeled": "\u25cb",    # ○
}


class FileListWidget(QListWidget):
    """Image file list with status indicators and filtering.

    Signals:
        image_selected(Path): Emitted when user clicks a different image.
    """

    image_selected = pyqtSignal(object)  # Path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paths: list[Path] = []
        self._statuses: dict[str, str] = {}  # path_str -> status
        self._filter: str | None = None  # None = show all

        self.currentRowChanged.connect(self._on_row_changed)

    def set_image_paths(self, paths: list[Path]) -> None:
        """Set the list of image paths to display."""
        self.blockSignals(True)
        self.clear()
        self._paths = list(paths)
        for path in paths:
            status = self._statuses.get(str(path), "unlabeled")
            icon = STATUS_ICONS.get(status, "○")
            item = QListWidgetItem(f"{icon} {path.name}")
            item.setData(Qt.UserRole, str(path))
            item.setForeground(QColor(STATUS_COLORS.get(status, "#6c7086")))
            self.addItem(item)
        self._apply_filter()
        self.blockSignals(False)

    def set_status(self, path: Path, status: str) -> None:
        """Update the status of an image file."""
        self._statuses[str(path)] = status
        # Update the item display
        for i in range(self.count()):
            item = self.item(i)
            if item.data(Qt.UserRole) == str(path):
                icon = STATUS_ICONS.get(status, "○")
                item.setText(f"{icon} {path.name}")
                item.setForeground(QColor(STATUS_COLORS.get(status, "#6c7086")))
                break

    def set_filter(self, status: str | None) -> None:
        """Filter items by status. None shows all."""
        self._filter = status
        self._apply_filter()

    def _apply_filter(self) -> None:
        """Apply current filter to items."""
        for i in range(self.count()):
            item = self.item(i)
            path_str = item.data(Qt.UserRole)
            if self._filter is None:
                item.setHidden(False)
            else:
                item_status = self._statuses.get(path_str, "unlabeled")
                item.setHidden(item_status != self._filter)

    def get_current_path(self) -> Path | None:
        """Get the currently selected image path."""
        item = self.currentItem()
        if item is None:
            return None
        path_str = item.data(Qt.UserRole)
        return Path(path_str) if path_str else None

    def get_index_info(self) -> tuple[int, int]:
        """Get current 1-based index and total count."""
        row = self.currentRow()
        return (row + 1 if row >= 0 else 0), len(self._paths)

    def go_next(self) -> None:
        """Navigate to next image."""
        row = self.currentRow()
        if row < self.count() - 1:
            self.setCurrentRow(row + 1)

    def go_prev(self) -> None:
        """Navigate to previous image."""
        row = self.currentRow()
        if row > 0:
            self.setCurrentRow(row - 1)

    def refresh_paths(self, paths: list[Path]) -> None:
        """Refresh file list, preserving statuses and selection."""
        current = self.get_current_path()
        self._paths = list(paths)
        self.set_image_paths(paths)
        # Restore selection
        if current:
            for i in range(self.count()):
                if self.item(i).data(Qt.UserRole) == str(current):
                    self.setCurrentRow(i)
                    break

    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            path = self.get_current_path()
            if path:
                self.image_selected.emit(path)
