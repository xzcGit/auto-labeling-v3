"""Tests for FileListWidget."""
from pathlib import Path

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QColor


def _make_test_image(path: Path, width: int = 100, height: int = 80) -> None:
    img = QImage(width, height, QImage.Format_RGB32)
    img.fill(QColor(Qt.red))
    img.save(str(path), "PNG")


class TestFileListWidget:
    def test_set_image_paths(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path(f"/imgs/img{i}.jpg") for i in range(5)]
        widget.set_image_paths(paths)
        assert widget.count() == 5

    def test_get_current_path(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path("/imgs/a.jpg"), Path("/imgs/b.jpg")]
        widget.set_image_paths(paths)
        widget.setCurrentRow(1)
        assert widget.get_current_path() == paths[1]

    def test_get_current_path_none_when_empty(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        assert widget.get_current_path() is None

    def test_set_status(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path("/imgs/a.jpg"), Path("/imgs/b.jpg")]
        widget.set_image_paths(paths)

        widget.set_status(paths[0], "confirmed")
        widget.set_status(paths[1], "pending")

        assert widget._statuses[str(paths[0])] == "confirmed"
        assert widget._statuses[str(paths[1])] == "pending"

    def test_filter_by_status(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path(f"/imgs/img{i}.jpg") for i in range(4)]
        widget.set_image_paths(paths)
        widget.set_status(paths[0], "confirmed")
        widget.set_status(paths[1], "pending")
        widget.set_status(paths[2], "unlabeled")
        widget.set_status(paths[3], "confirmed")

        widget.set_filter("confirmed")
        visible = [i for i in range(widget.count()) if not widget.item(i).isHidden()]
        assert len(visible) == 2

    def test_filter_all_shows_everything(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path(f"/imgs/img{i}.jpg") for i in range(3)]
        widget.set_image_paths(paths)
        widget.set_status(paths[0], "confirmed")
        widget.set_filter("confirmed")
        widget.set_filter(None)  # show all
        visible = [i for i in range(widget.count()) if not widget.item(i).isHidden()]
        assert len(visible) == 3

    def test_navigate_next_prev(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path(f"/imgs/img{i}.jpg") for i in range(3)]
        widget.set_image_paths(paths)
        widget.setCurrentRow(0)

        widget.go_next()
        assert widget.currentRow() == 1
        widget.go_next()
        assert widget.currentRow() == 2
        widget.go_next()
        assert widget.currentRow() == 2  # stays at end

        widget.go_prev()
        assert widget.currentRow() == 1
        widget.go_prev()
        assert widget.currentRow() == 0
        widget.go_prev()
        assert widget.currentRow() == 0  # stays at start

    def test_current_index_info(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path(f"/imgs/img{i}.jpg") for i in range(5)]
        widget.set_image_paths(paths)
        widget.setCurrentRow(2)
        idx, total = widget.get_index_info()
        assert idx == 3  # 1-based
        assert total == 5

    def test_extended_selection_mode(self, qapp):
        from src.ui.file_list import FileListWidget
        from PyQt5.QtWidgets import QAbstractItemView

        widget = FileListWidget()
        assert widget.selectionMode() == QAbstractItemView.ExtendedSelection

    def test_get_selected_paths(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path(f"/imgs/img{i}.jpg") for i in range(5)]
        widget.set_image_paths(paths)
        # Select multiple items
        widget.item(1).setSelected(True)
        widget.item(3).setSelected(True)
        selected = widget.get_selected_paths()
        assert len(selected) == 2

    def test_batch_confirm_signal(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        results = []
        widget.batch_confirm_requested.connect(lambda p: results.append(p))
        # Simulate signal emission
        widget.batch_confirm_requested.emit([Path("/img1.jpg")])
        assert len(results) == 1

    def test_batch_delete_signal(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        results = []
        widget.batch_delete_requested.connect(lambda p: results.append(p))
        widget.batch_delete_requested.emit([Path("/img1.jpg")])
        assert len(results) == 1

    def test_class_filter(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path(f"/imgs/img{i}.jpg") for i in range(4)]
        widget.set_image_paths(paths)

        # Set classes for images
        widget.set_image_classes(paths[0], {"cat", "dog"})
        widget.set_image_classes(paths[1], {"cat"})
        widget.set_image_classes(paths[2], {"dog"})
        widget.set_image_classes(paths[3], set())  # no annotations

        # Filter by "cat"
        widget.set_class_filter("cat")
        visible = [i for i in range(widget.count()) if not widget.item(i).isHidden()]
        assert len(visible) == 2  # img0 and img1

        # Filter by "dog"
        widget.set_class_filter("dog")
        visible = [i for i in range(widget.count()) if not widget.item(i).isHidden()]
        assert len(visible) == 2  # img0 and img2

        # Clear filter
        widget.set_class_filter(None)
        visible = [i for i in range(widget.count()) if not widget.item(i).isHidden()]
        assert len(visible) == 4

    def test_combined_status_and_class_filter(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path(f"/imgs/img{i}.jpg") for i in range(3)]
        widget.set_image_paths(paths)

        widget.set_status(paths[0], "confirmed")
        widget.set_status(paths[1], "pending")
        widget.set_status(paths[2], "confirmed")
        widget.set_image_classes(paths[0], {"cat"})
        widget.set_image_classes(paths[1], {"cat"})
        widget.set_image_classes(paths[2], {"dog"})

        # Filter: confirmed + cat
        widget.set_filter("confirmed")
        widget.set_class_filter("cat")
        visible = [i for i in range(widget.count()) if not widget.item(i).isHidden()]
        assert len(visible) == 1  # only img0

    def test_get_paths_returns_copy(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path("/imgs/a.jpg"), Path("/imgs/b.jpg"), Path("/imgs/c.jpg")]
        widget.set_image_paths(paths)

        got = widget.get_paths()
        assert got == paths
        got.append(Path("/imgs/x.jpg"))
        assert widget.get_paths() == paths
