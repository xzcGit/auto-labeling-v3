"""Image loading utilities with caching."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImageReader


def load_pixmap(
    path: Path | str, max_size: int | None = None
) -> QPixmap | None:
    """Load image as QPixmap. Returns None if file doesn't exist or fails to load."""
    path = Path(path)
    if not path.exists():
        return None

    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    image = reader.read()
    if image.isNull():
        return None

    pixmap = QPixmap.fromImage(image)
    if max_size and (pixmap.width() > max_size or pixmap.height() > max_size):
        pixmap = pixmap.scaled(
            max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
    return pixmap


def get_image_size(path: Path | str) -> tuple[int, int]:
    """Get image dimensions (width, height) without fully loading. Returns (0, 0) on failure."""
    reader = QImageReader(str(path))
    size = reader.size()
    if size.isValid():
        return size.width(), size.height()
    return 0, 0


class ImageCache:
    """LRU cache for loaded QPixmaps."""

    def __init__(self, max_size: int = 10):
        self._max_size = max_size
        self._cache: OrderedDict[str, QPixmap] = OrderedDict()

    def get(self, path: Path | str, pixmap_max_size: int | None = None) -> QPixmap | None:
        """Get pixmap from cache, loading if necessary."""
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        pixmap = load_pixmap(path, max_size=pixmap_max_size)
        if pixmap is None:
            return None

        self._cache[key] = pixmap
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return pixmap

    def clear(self) -> None:
        """Clear all cached pixmaps."""
        self._cache.clear()

    def invalidate(self, path: Path | str) -> None:
        """Remove a specific path from cache."""
        self._cache.pop(str(path), None)
