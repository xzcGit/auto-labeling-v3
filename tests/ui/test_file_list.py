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
