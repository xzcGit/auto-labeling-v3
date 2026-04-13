"""File list widget for image navigation and status display."""
from __future__ import annotations

import shutil
from pathlib import Path

from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QMenu, QAction
from PyQt5.QtCore import Qt, pyqtSignal, QUrl
from PyQt5.QtGui import QColor, QDragEnterEvent, QDropEvent

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

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
    """Image file list with status indicators, filtering, and drag-and-drop.

    Signals:
        image_selected(Path): Emitted when user clicks a different image.
        images_dropped(list[Path]): Emitted when image files are dropped onto the list.
        batch_confirm_requested(list): Emitted with list of Paths to batch confirm.
        batch_delete_requested(list): Emitted with list of Paths to batch delete annotations.
    """

    image_selected = pyqtSignal(object)  # Path
    images_dropped = pyqtSignal(list)    # list[Path]
    batch_confirm_requested = pyqtSignal(list)   # list[Path]
    batch_delete_requested = pyqtSignal(list)    # list[Path]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paths: list[Path] = []
        self._statuses: dict[str, str] = {}  # path_str -> status
        self._image_classes: dict[str, set[str]] = {}  # path_str -> set of class names
        self._filter: str | None = None  # None = show all
        self._class_filter: str | None = None  # None = show all classes

        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.currentRowChanged.connect(self._on_row_changed)

    # ── Drag and drop ─────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        if not event.mimeData().hasUrls():
            super().dropEvent(event)
            return
        image_paths = []
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(path)
            elif path.is_dir():
                for ext in IMAGE_EXTENSIONS:
                    image_paths.extend(path.glob(f"*{ext}"))
        if image_paths:
            self.images_dropped.emit(image_paths)
        event.acceptProposedAction()

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

    def set_class_filter(self, class_name: str | None) -> None:
        """Filter items by class name. None shows all."""
        self._class_filter = class_name
        self._apply_filter()

    def set_image_classes(self, path: Path, classes: set[str]) -> None:
        """Set the class names present in an image's annotations."""
        self._image_classes[str(path)] = classes

    def _apply_filter(self) -> None:
        """Apply current status and class filters to items."""
        for i in range(self.count()):
            item = self.item(i)
            path_str = item.data(Qt.UserRole)
            hidden = False
            # Status filter
            if self._filter is not None:
                item_status = self._statuses.get(path_str, "unlabeled")
                if item_status != self._filter:
                    hidden = True
            # Class filter
            if not hidden and self._class_filter is not None:
                img_classes = self._image_classes.get(path_str, set())
                if self._class_filter not in img_classes:
                    hidden = True
            item.setHidden(hidden)

    def get_current_path(self) -> Path | None:
        """Get the currently selected image path."""
        item = self.currentItem()
        if item is None:
            return None
        path_str = item.data(Qt.UserRole)
        return Path(path_str) if path_str else None

    def get_index_info(self) -> tuple[int, int]:
        """Get current 1-based index and visible count (respects active filter)."""
        row = self.currentRow()
        visible_count = sum(1 for i in range(self.count()) if not self.item(i).isHidden())
        if row < 0:
            return 0, visible_count
        # Compute 1-based visible index
        visible_idx = 0
        for i in range(row + 1):
            if not self.item(i).isHidden():
                visible_idx += 1
        return visible_idx, visible_count

    def go_next(self) -> None:
        """Navigate to next visible image."""
        row = self.currentRow()
        for i in range(row + 1, self.count()):
            if not self.item(i).isHidden():
                self.setCurrentRow(i)
                return

    def go_prev(self) -> None:
        """Navigate to previous visible image."""
        row = self.currentRow()
        for i in range(row - 1, -1, -1):
            if not self.item(i).isHidden():
                self.setCurrentRow(i)
                return

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

    def get_selected_paths(self) -> list[Path]:
        """Get all selected image paths."""
        paths = []
        for item in self.selectedItems():
            path_str = item.data(Qt.UserRole)
            if path_str:
                paths.append(Path(path_str))
        return paths

    def get_visible_paths(self) -> list[Path]:
        """Get paths of all currently visible (non-hidden) items."""
        paths = []
        for i in range(self.count()):
            item = self.item(i)
            if not item.isHidden():
                path_str = item.data(Qt.UserRole)
                if path_str:
                    paths.append(Path(path_str))
        return paths

    def contextMenuEvent(self, event) -> None:
        """Show right-click context menu for file list items."""
        item = self.itemAt(event.pos())
        if not item:
            return
        path_str = item.data(Qt.UserRole)
        if not path_str:
            return

        selected = self.get_selected_paths()
        menu = QMenu(self)

        # Batch operations (when multiple selected)
        if len(selected) > 1:
            batch_confirm = menu.addAction(f"批量确认 ({len(selected)} 张)")
            batch_confirm.triggered.connect(lambda: self.batch_confirm_requested.emit(selected))

            batch_delete = menu.addAction(f"批量删除标注 ({len(selected)} 张)")
            batch_delete.triggered.connect(lambda: self.batch_delete_requested.emit(selected))

            menu.addSeparator()

        open_folder = menu.addAction("在文件管理器中打开")
        open_folder.triggered.connect(lambda: self._open_in_explorer(Path(path_str)))

        copy_path = menu.addAction("复制文件路径")
        copy_path.triggered.connect(lambda: self._copy_path(path_str))

        menu.exec_(event.globalPos())

    def _open_in_explorer(self, path: Path) -> None:
        """Open the containing folder in the system file manager."""
        import subprocess, sys
        folder = str(path.parent)
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", str(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)])
        else:
            subprocess.Popen(["xdg-open", folder])

    def _copy_path(self, path_str: str) -> None:
        """Copy file path to clipboard."""
        from PyQt5.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(path_str)
