"""Label panel — main annotation workspace assembling file list, canvas, and properties."""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QPushButton,
    QToolBar,
    QLabel,
    QComboBox,
    QAction,
)
from PyQt5.QtCore import Qt, pyqtSignal

from src.core.annotation import Annotation, ImageAnnotation
from src.core.label_io import save_annotation, load_annotation
from src.core.project import ProjectManager
from src.ui.canvas import AnnotationCanvas
from src.ui.file_list import FileListWidget
from src.ui.properties import AnnotationPanel
from src.ui.class_picker import ClassPickerPopup
from src.utils.image import get_image_size
from src.utils.undo import UndoStack

logger = logging.getLogger(__name__)


class LabelPanel(QWidget):
    """Main annotation workspace.

    Layout: toolbar (top) | file_list (left) | canvas (center) | properties (right)
    """

    def __init__(self, config_path=None, parent=None):
        super().__init__(parent)
        self._project: ProjectManager | None = None
        self._current_image_path: Path | None = None
        self._current_annotation: ImageAnnotation | None = None
        self._undo_stacks: dict[str, UndoStack] = {}  # per-image undo
        self._last_class: str | None = None
        self._config_path = config_path

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        self._toolbar = QToolBar()
        self._toolbar.setMovable(False)
        self._toolbar.setStyleSheet("QToolBar { spacing: 4px; padding: 4px; }")

        self._btn_select = QPushButton("V 移动")
        self._btn_select.setCheckable(True)
        self._btn_select.setChecked(True)
        self._btn_bbox = QPushButton("W 矩形框")
        self._btn_bbox.setCheckable(True)
        self._btn_keypoint = QPushButton("K 关键点")
        self._btn_keypoint.setCheckable(True)

        for btn in [self._btn_select, self._btn_bbox, self._btn_keypoint]:
            btn.setMinimumWidth(80)
            self._toolbar.addWidget(btn)

        self._toolbar.addSeparator()

        self._btn_confirm_all = QPushButton("全部确认")
        self._toolbar.addWidget(self._btn_confirm_all)

        self._toolbar.addSeparator()

        # Filter combo
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["全部", "已确认", "待确认", "未标注"])
        self._filter_combo.setMinimumWidth(80)
        self._toolbar.addWidget(QLabel(" 筛选: "))
        self._toolbar.addWidget(self._filter_combo)

        layout.addWidget(self._toolbar)

        # Main splitter: file_list | canvas | properties
        self._splitter = QSplitter(Qt.Horizontal)

        self._file_list = FileListWidget()
        self._file_list.setMaximumWidth(250)
        self._splitter.addWidget(self._file_list)

        self._canvas = AnnotationCanvas()
        self._splitter.addWidget(self._canvas)

        self._ann_panel = AnnotationPanel()
        self._ann_panel.setMaximumWidth(280)
        self._splitter.addWidget(self._ann_panel)

        self._splitter.setStretchFactor(0, 0)  # file list fixed
        self._splitter.setStretchFactor(1, 1)  # canvas stretches
        self._splitter.setStretchFactor(2, 0)  # properties fixed
        self._splitter.setSizes([200, 800, 250])

        layout.addWidget(self._splitter)

    def _connect_signals(self) -> None:
        # Tool buttons
        self._btn_select.clicked.connect(lambda: self._set_tool("select"))
        self._btn_bbox.clicked.connect(lambda: self._set_tool("draw_bbox"))
        self._btn_keypoint.clicked.connect(lambda: self._set_tool("draw_keypoint"))

        # File list
        self._file_list.image_selected.connect(self._on_image_selected)

        # Canvas signals
        self._canvas.annotation_selected.connect(self._on_annotation_selected)
        self._canvas.annotation_created.connect(self._on_annotation_created)
        self._canvas.annotation_modified.connect(self._on_annotation_modified)
        self._canvas.annotation_deleted.connect(self._on_annotation_deleted)
        self._canvas.class_requested.connect(self._on_class_requested)
        self._canvas.annotations_changed.connect(self._on_annotations_changed)

        # Properties panel
        self._ann_panel.annotation_clicked.connect(self._canvas.select_annotation)

        # Confirm all
        self._btn_confirm_all.clicked.connect(self._confirm_all)

        # Filter
        self._filter_combo.currentTextChanged.connect(self._on_filter_changed)

    def set_project(self, project: ProjectManager) -> None:
        """Set the current project and populate file list."""
        self._project = project
        self._undo_stacks.clear()

        # Set colors
        colors = {}
        for cls in project.config.classes:
            colors[cls] = project.config.get_class_color(cls)
        self._canvas.set_class_colors(colors)
        self._ann_panel.set_class_colors(colors)
        self._ann_panel.set_classes(project.config.classes)

        # Load file list
        images = project.list_images()
        self._file_list.set_image_paths(images)

        # Update statuses
        for img_path in images:
            label_path = project.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia:
                self._file_list.set_status(img_path, ia.status)

        # Select first image
        if images:
            self._file_list.setCurrentRow(0)

    # ── Tool management ────────────────────────────────────────

    def _set_tool(self, mode: str) -> None:
        self._btn_select.setChecked(mode == "select")
        self._btn_bbox.setChecked(mode == "draw_bbox")
        self._btn_keypoint.setChecked(mode == "draw_keypoint")
        self._canvas.set_tool_mode(mode)

    # ── Image switching ────────────────────────────────────────

    def _on_image_selected(self, path: Path) -> None:
        """Handle switching to a new image."""
        # Save current
        self._save_current()

        self._current_image_path = path
        self._canvas.load_image(str(path))

        # Load annotations
        if self._project:
            label_path = self._project.label_path_for(path)
            ia = load_annotation(label_path)
            if ia:
                self._current_annotation = ia
            else:
                w, h = get_image_size(path)
                self._current_annotation = ImageAnnotation(
                    image_path=path.name,
                    image_size=(w, h),
                )

            self._canvas.set_annotations(list(self._current_annotation.annotations))
            self._ann_panel.set_annotations(list(self._current_annotation.annotations))
            self._ann_panel.set_image_tags(self._current_annotation.image_tags)

            # Init undo stack if needed
            key = str(path)
            if key not in self._undo_stacks:
                self._undo_stacks[key] = UndoStack()
                self._undo_stacks[key].push(self._current_annotation.to_dict())

    def _save_current(self) -> None:
        """Save current image's annotations to disk."""
        if not self._project or not self._current_image_path or not self._current_annotation:
            return
        # Sync canvas annotations back
        self._current_annotation.annotations = list(self._canvas._annotations)
        self._current_annotation.image_tags = self._ann_panel.get_image_tags()
        label_path = self._project.label_path_for(self._current_image_path)
        save_annotation(self._current_annotation, label_path)
        # Update file list status
        self._file_list.set_status(self._current_image_path, self._current_annotation.status)

    # ── Annotation events ──────────────────────────────────────

    def _on_annotation_selected(self, ann_id) -> None:
        self._ann_panel.select_annotation(ann_id)

    def _on_annotation_created(self, ann) -> None:
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_annotation_modified(self, ann_id: str) -> None:
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_annotation_deleted(self, ann_id: str) -> None:
        self._canvas._annotations = [a for a in self._canvas._annotations if a.id != ann_id]
        self._canvas._selected_id = None
        self._canvas.update()
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_annotations_changed(self) -> None:
        self._sync_annotations_to_panel()

    def _on_class_requested(self, px: float, py: float) -> None:
        """Show class picker and create annotation."""
        if not self._project:
            return
        classes = self._project.config.classes
        if not classes:
            return
        colors = {}
        for cls in classes:
            colors[cls] = self._project.config.get_class_color(cls)

        picker = ClassPickerPopup(
            classes=classes,
            colors=colors,
            default_class=self._last_class,
            parent=self,
        )
        picker.move(self.mapToGlobal(self._canvas.mapTo(self, self._canvas.pos())))
        if picker.exec_():
            cls_name = picker.get_selected_class()
            cls_id = picker.get_selected_index()
            if cls_name is not None:
                self._last_class = cls_name
                if self._canvas.tool_mode == "draw_bbox":
                    self._canvas.create_bbox_from_draw(cls_name, cls_id)
                elif self._canvas.tool_mode == "draw_keypoint":
                    self._canvas.create_keypoint_at(cls_name, cls_id)
        else:
            # Cancelled — clear draw state
            self._canvas._draw_start = None
            self._canvas._draw_current = None
            self._canvas.update()

    def _sync_annotations_to_panel(self) -> None:
        self._ann_panel.set_annotations(list(self._canvas._annotations))

    def _confirm_all(self) -> None:
        """Confirm all annotations on current image."""
        for ann in self._canvas._annotations:
            ann.confirmed = True
        self._push_undo()
        self._canvas.update()
        self._sync_annotations_to_panel()

    # ── Undo/Redo ──────────────────────────────────────────────

    def _push_undo(self) -> None:
        if not self._current_image_path or not self._current_annotation:
            return
        self._current_annotation.annotations = list(self._canvas._annotations)
        key = str(self._current_image_path)
        if key not in self._undo_stacks:
            self._undo_stacks[key] = UndoStack()
        self._undo_stacks[key].push(self._current_annotation.to_dict())

    def undo(self) -> None:
        if not self._current_image_path:
            return
        key = str(self._current_image_path)
        stack = self._undo_stacks.get(key)
        if not stack or not stack.can_undo:
            return
        state = stack.undo()
        if state:
            self._restore_state(state)

    def redo(self) -> None:
        if not self._current_image_path:
            return
        key = str(self._current_image_path)
        stack = self._undo_stacks.get(key)
        if not stack or not stack.can_redo:
            return
        state = stack.redo()
        if state:
            self._restore_state(state)

    def _restore_state(self, state: dict) -> None:
        ia = ImageAnnotation.from_dict(state)
        self._current_annotation = ia
        self._canvas.set_annotations(list(ia.annotations))
        self._ann_panel.set_annotations(list(ia.annotations))

    # ── Filter ─────────────────────────────────────────────────

    def _on_filter_changed(self, text: str) -> None:
        mapping = {"全部": None, "已确认": "confirmed", "待确认": "pending", "未标注": "unlabeled"}
        self._file_list.set_filter(mapping.get(text))

    # ── Keyboard shortcuts ─────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        key = event.key()
        mod = event.modifiers()

        if key == Qt.Key_W:
            self._set_tool("draw_bbox")
        elif key == Qt.Key_K:
            self._set_tool("draw_keypoint")
        elif key == Qt.Key_V:
            self._set_tool("select")
        elif key == Qt.Key_D or key == Qt.Key_Right:
            self._save_current()
            self._file_list.go_next()
        elif key == Qt.Key_A and not (mod & Qt.ShiftModifier) or key == Qt.Key_Left:
            self._save_current()
            self._file_list.go_prev()
        elif key == Qt.Key_Space and mod & Qt.ControlModifier:
            self._confirm_all()
        elif key == Qt.Key_Space:
            ann = self._canvas.get_selected_annotation()
            if ann:
                ann.confirmed = True
                self._push_undo()
                self._canvas.update()
                self._sync_annotations_to_panel()
        elif key == Qt.Key_Delete:
            ann = self._canvas.get_selected_annotation()
            if ann:
                self._on_annotation_deleted(ann.id)
        elif key == Qt.Key_Z and mod & Qt.ControlModifier:
            self.undo()
        elif key == Qt.Key_Y and mod & Qt.ControlModifier:
            self.redo()
        elif key == Qt.Key_S and mod & Qt.ControlModifier:
            self._save_current()
        elif key == Qt.Key_F5:
            if self._project:
                images = self._project.list_images()
                self._file_list.refresh_paths(images)
        else:
            super().keyPressEvent(event)

    def save_and_cleanup(self) -> None:
        """Save current work before closing."""
        self._save_current()
