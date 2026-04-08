"""Label panel — main annotation workspace assembling file list, canvas, and properties."""
from __future__ import annotations

import logging
import shutil
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
from PyQt5.QtCore import Qt, pyqtSignal, QPoint

from src.core.annotation import Annotation, ImageAnnotation
from src.core.label_io import save_annotation, load_annotation
from src.core.project import ProjectManager
from src.ui.canvas import AnnotationCanvas
from src.ui.file_list import FileListWidget
from src.ui.properties import AnnotationPanel
from src.ui.class_picker import ClassPickerPopup
from src.utils.image import get_image_size, ImageCache
from src.utils.undo import UndoStack
from src.ui.icons import icon

logger = logging.getLogger(__name__)


class LabelPanel(QWidget):
    """Main annotation workspace.

    Layout: toolbar (top) | file_list (left) | canvas (center) | properties (right)
    """

    auto_label_single_requested = pyqtSignal()
    auto_label_batch_requested = pyqtSignal()
    status_changed = pyqtSignal(str)  # status text for status bar

    def __init__(self, config_path=None, parent=None):
        super().__init__(parent)
        self._project: ProjectManager | None = None
        self._current_image_path: Path | None = None
        self._current_annotation: ImageAnnotation | None = None
        self._undo_stacks: dict[str, UndoStack] = {}  # per-image undo
        self._last_class: str | None = None
        self._config_path = config_path
        self._clipboard: list[dict] | None = None  # copied annotations as dicts
        self._image_cache = ImageCache(max_count=16, max_memory_mb=512.0)

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

        self._btn_select = QPushButton(icon("cursor"), "移动")
        self._btn_select.setCheckable(True)
        self._btn_select.setChecked(True)
        self._btn_select.setToolTip("选择/移动工具 (V)")
        self._btn_bbox = QPushButton(icon("bbox"), "矩形框")
        self._btn_bbox.setCheckable(True)
        self._btn_bbox.setToolTip("绘制矩形框 (W)")
        self._btn_keypoint = QPushButton(icon("keypoint"), "关键点")
        self._btn_keypoint.setCheckable(True)
        self._btn_keypoint.setToolTip("绘制关键点 (K)")

        for btn in [self._btn_select, self._btn_bbox, self._btn_keypoint]:
            btn.setMinimumWidth(80)
            self._toolbar.addWidget(btn)

        self._toolbar.addSeparator()

        self._btn_confirm_all = QPushButton(icon("check_all"), "全部确认")
        self._btn_confirm_all.setToolTip("确认当前图片所有标注 (Ctrl+Space)")
        self._toolbar.addWidget(self._btn_confirm_all)

        self._toolbar.addSeparator()

        self._btn_auto_single = QPushButton(icon("auto_label"), "自动标注")
        self._btn_auto_single.setToolTip("对当前图片执行自动标注 (Shift+A)")
        self._toolbar.addWidget(self._btn_auto_single)

        self._btn_auto_batch = QPushButton(icon("batch"), "批量标注")
        self._btn_auto_batch.setToolTip("对多张图片执行批量自动标注 (Ctrl+Shift+A)")
        self._toolbar.addWidget(self._btn_auto_batch)

        self._toolbar.addSeparator()

        self._btn_undo = QPushButton(icon("undo"), "")
        self._btn_undo.setToolTip("撤销 (Ctrl+Z)")
        self._btn_undo.setFixedWidth(36)
        self._toolbar.addWidget(self._btn_undo)

        self._btn_redo = QPushButton(icon("redo"), "")
        self._btn_redo.setToolTip("重做 (Ctrl+Y)")
        self._btn_redo.setFixedWidth(36)
        self._toolbar.addWidget(self._btn_redo)

        self._btn_save = QPushButton(icon("save"), "")
        self._btn_save.setToolTip("保存 (Ctrl+S)")
        self._btn_save.setFixedWidth(36)
        self._toolbar.addWidget(self._btn_save)

        self._toolbar.addSeparator()

        # Filter combos
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["全部", "已确认", "待确认", "未标注"])
        self._filter_combo.setMinimumWidth(80)
        self._toolbar.addWidget(QLabel(" 筛选: "))
        self._toolbar.addWidget(self._filter_combo)

        self._class_filter_combo = QComboBox()
        self._class_filter_combo.addItem("所有类别")
        self._class_filter_combo.setMinimumWidth(80)
        self._toolbar.addWidget(QLabel(" 类别: "))
        self._toolbar.addWidget(self._class_filter_combo)

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
        self._file_list.images_dropped.connect(self._on_images_dropped)
        self._file_list.batch_confirm_requested.connect(self._on_batch_confirm)
        self._file_list.batch_delete_requested.connect(self._on_batch_delete)

        # Canvas signals
        self._canvas.annotation_selected.connect(self._on_annotation_selected)
        self._canvas.annotation_created.connect(self._on_annotation_created)
        self._canvas.annotation_modified.connect(self._on_annotation_modified)
        self._canvas.annotation_deleted.connect(self._on_annotation_deleted)
        self._canvas.class_requested.connect(self._on_class_requested)
        self._canvas.class_change_requested.connect(self._on_class_change_requested)
        self._canvas.annotations_changed.connect(self._on_annotations_changed)
        self._canvas.annotation_copied.connect(self._on_annotation_copied)

        # Properties panel
        self._ann_panel.annotation_clicked.connect(self._canvas.select_annotation)

        # Confirm all
        self._btn_confirm_all.clicked.connect(self._confirm_all)

        # Auto-label buttons
        self._btn_auto_single.clicked.connect(self.auto_label_single_requested.emit)
        self._btn_auto_batch.clicked.connect(self.auto_label_batch_requested.emit)

        # Undo/Redo/Save buttons
        self._btn_undo.clicked.connect(self.undo)
        self._btn_redo.clicked.connect(self.redo)
        self._btn_save.clicked.connect(self._save_current)

        # Filter
        self._filter_combo.currentTextChanged.connect(self._on_filter_changed)
        self._class_filter_combo.currentTextChanged.connect(self._on_class_filter_changed)

    def set_project(self, project: ProjectManager) -> None:
        """Set the current project and populate file list."""
        # Clear previous state to avoid saving to wrong project
        self._current_image_path = None
        self._current_annotation = None
        self._project = project
        self._undo_stacks.clear()
        self._canvas.clear()

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
                # Track classes per image for class filtering
                classes_in_img = {a.class_name for a in ia.annotations}
                self._file_list.set_image_classes(img_path, classes_in_img)

        # Populate class filter combo
        self._class_filter_combo.blockSignals(True)
        self._class_filter_combo.clear()
        self._class_filter_combo.addItem("所有类别")
        for cls in project.config.classes:
            self._class_filter_combo.addItem(cls)
        self._class_filter_combo.blockSignals(False)

        # Select first image
        if images:
            self._file_list.setCurrentRow(0)

        logger.info("Project loaded: %s (%d images)", project.config.name, len(images))
        self._update_project_stats()

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

        # Load via cache
        pixmap = self._image_cache.get(path)
        if pixmap:
            self._canvas.set_pixmap(pixmap)
        else:
            self._canvas.load_image(str(path))
        logger.debug("Image selected: %s", path.name)

        # Preload neighbors (2 ahead, 1 behind)
        self._preload_neighbors(path)

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

            self._emit_status()

            # Init undo stack if needed
            key = str(path)
            if key not in self._undo_stacks:
                self._undo_stacks[key] = UndoStack()
                self._undo_stacks[key].push(self._current_annotation.to_dict())

            self._update_project_stats()
            self._update_lock_state()

    def _preload_neighbors(self, current: Path) -> None:
        """Preload images adjacent to current for smoother navigation."""
        if not self._project:
            return
        images = self._project.list_images()
        try:
            idx = images.index(current)
        except ValueError:
            return
        neighbors = []
        for offset in [1, 2, -1]:
            ni = idx + offset
            if 0 <= ni < len(images):
                neighbors.append(images[ni])
        if neighbors:
            self._image_cache.preload(neighbors)

    def _save_current(self) -> None:
        """Save current image's annotations to disk."""
        if not self._project or not self._current_image_path or not self._current_annotation:
            return
        # Sync canvas annotations back
        self._current_annotation.annotations = list(self._canvas._annotations)
        self._current_annotation.image_tags = self._ann_panel.get_image_tags()
        label_path = self._project.label_path_for(self._current_image_path)
        save_annotation(self._current_annotation, label_path)
        logger.debug("Saved annotations for %s", self._current_image_path.name)
        # Update file list status
        self._file_list.set_status(self._current_image_path, self._current_annotation.status)

    # ── Annotation events ──────────────────────────────────────

    def _on_annotation_selected(self, ann_id) -> None:
        self._ann_panel.select_annotation(ann_id)

    def _on_annotation_created(self, ann) -> None:
        self._push_undo()
        self._sync_annotations_to_panel()
        self._update_lock_state()

    def _on_annotation_modified(self, ann_id: str) -> None:
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_annotation_deleted(self, ann_id: str) -> None:
        self._canvas._annotations = [a for a in self._canvas._annotations if a.id != ann_id]
        self._canvas._selected_id = None
        self._canvas.update()
        self._push_undo()
        self._sync_annotations_to_panel()
        self._update_lock_state()
        self._sync_annotations_to_panel()

    def _on_annotations_changed(self) -> None:
        self._sync_annotations_to_panel()

    def _on_class_requested(self, px: float, py: float) -> None:
        """Show class picker and create annotation.

        Allows selecting existing class or typing a new one (like labelimg).
        New classes are automatically added to the project.
        """
        if not self._project:
            return
        classes = self._project.config.classes
        colors = {}
        for cls in classes:
            colors[cls] = self._project.config.get_class_color(cls)

        picker = ClassPickerPopup(
            classes=classes,
            colors=colors,
            default_class=self._last_class,
            parent=self,
        )
        global_pos = self._canvas.mapToGlobal(QPoint(int(px), int(py)))
        picker.move(global_pos)
        if picker.exec_():
            cls_name = picker.get_selected_class()
            if cls_name is None:
                self._clear_draw_state()
                return

            # If user typed a new class, add it to the project
            if picker.is_new_class():
                self._project.add_class(cls_name)
                self._project.save()
                # Refresh colors
                new_color = self._project.config.get_class_color(cls_name)
                colors[cls_name] = new_color
                self._canvas.set_class_colors(colors)
                self._ann_panel.set_class_colors(colors)
                self._ann_panel.set_classes(self._project.config.classes)

            cls_id = self._project.config.get_class_id(cls_name)
            self._last_class = cls_name
            if self._canvas.tool_mode == "draw_bbox":
                self._canvas.create_bbox_from_draw(cls_name, cls_id)
            elif self._canvas.tool_mode == "draw_keypoint":
                self._canvas.create_keypoint_at(cls_name, cls_id)
        else:
            self._clear_draw_state()

    def _clear_draw_state(self) -> None:
        """Clear canvas draw state after cancelled operation."""
        self._canvas._draw_start = None
        self._canvas._draw_current = None
        self._canvas.update()

    def _on_class_change_requested(self, ann_id: str, px: float, py: float) -> None:
        """Show class picker to change an annotation's class."""
        if not self._project:
            return
        ann = None
        for a in self._canvas._annotations:
            if a.id == ann_id:
                ann = a
                break
        if ann is None:
            return

        classes = self._project.config.classes
        colors = {}
        for cls in classes:
            colors[cls] = self._project.config.get_class_color(cls)

        picker = ClassPickerPopup(
            classes=classes,
            colors=colors,
            default_class=ann.class_name,
            parent=self,
        )
        global_pos = self._canvas.mapToGlobal(QPoint(int(px), int(py)))
        picker.move(global_pos)
        if picker.exec_():
            cls_name = picker.get_selected_class()
            if cls_name is None:
                return

            # If user typed a new class, add to project
            if picker.is_new_class():
                self._project.add_class(cls_name)
                self._project.save()
                new_color = self._project.config.get_class_color(cls_name)
                colors[cls_name] = new_color
                self._canvas.set_class_colors(colors)
                self._ann_panel.set_class_colors(colors)
                self._ann_panel.set_classes(self._project.config.classes)

            if cls_name != ann.class_name:
                ann.class_name = cls_name
                ann.class_id = self._project.config.get_class_id(cls_name)
                self._push_undo()
                self._canvas.update()
                self._sync_annotations_to_panel()

    def _sync_annotations_to_panel(self) -> None:
        self._ann_panel.set_annotations(list(self._canvas._annotations))
        self._emit_status()

    def _emit_status(self) -> None:
        """Emit status bar text with current file info."""
        if not self._current_image_path:
            return
        idx, total = self._file_list.get_index_info()
        n_ann = len(self._canvas._annotations)
        n_confirmed = sum(1 for a in self._canvas._annotations if a.confirmed)
        n_pending = n_ann - n_confirmed
        parts = [
            self._current_image_path.name,
            f"{idx}/{total}",
            f"标注: {n_ann}",
        ]
        if n_pending > 0:
            parts.append(f"确认: {n_confirmed} 待确认: {n_pending}")
        self.status_changed.emit(" | ".join(parts))

    def _compute_project_stats(self) -> dict:
        """Compute project-level annotation statistics."""
        if not self._project:
            return {}
        stats = {
            "total_images": 0,
            "labeled_images": 0,
            "confirmed_images": 0,
            "total_annotations": 0,
            "class_counts": {},
        }
        images = self._project.list_images()
        stats["total_images"] = len(images)
        for img_path in images:
            label_path = self._project.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia is None or len(ia.annotations) == 0:
                continue
            stats["labeled_images"] += 1
            all_confirmed = all(a.confirmed for a in ia.annotations)
            if all_confirmed:
                stats["confirmed_images"] += 1
            for ann in ia.annotations:
                stats["total_annotations"] += 1
                stats["class_counts"][ann.class_name] = stats["class_counts"].get(ann.class_name, 0) + 1
        return stats

    def _update_project_stats(self) -> None:
        """Recompute and update project stats in the panel."""
        self._ann_panel.set_project_stats(self._compute_project_stats())

    def _confirm_all(self) -> None:
        """Confirm all annotations on current image."""
        for ann in self._canvas._annotations:
            ann.confirmed = True
        self._push_undo()
        self._canvas.update()
        self._sync_annotations_to_panel()
        self._update_lock_state()

    # ── Copy / Paste ──────────────────────────────────────────

    def _copy_annotation(self) -> None:
        """Copy selected annotation to clipboard."""
        ann = self._canvas.get_selected_annotation()
        if ann:
            self._clipboard = [ann.to_dict()]
            logger.debug("Copied annotation: %s", ann.class_name)

    def _on_annotation_copied(self, ann_id: str) -> None:
        """Handle copy via right-click menu."""
        for ann in self._canvas._annotations:
            if ann.id == ann_id:
                self._clipboard = [ann.to_dict()]
                logger.debug("Copied annotation via menu: %s", ann.class_name)
                break

    def _paste_annotation(self) -> None:
        """Paste clipboard annotations onto current image."""
        if not self._clipboard or self._canvas._locked:
            return
        import uuid as _uuid
        for ann_dict in self._clipboard:
            new_dict = dict(ann_dict)
            new_dict["id"] = str(_uuid.uuid4())
            new_dict["confirmed"] = False
            new_ann = Annotation.from_dict(new_dict)
            self._canvas._annotations.append(new_ann)
        self._push_undo()
        self._canvas.update()
        self._sync_annotations_to_panel()
        logger.debug("Pasted %d annotations", len(self._clipboard))

    # ── Lock state ────────────────────────────────────────────

    def _update_lock_state(self) -> None:
        """Lock canvas when all annotations are confirmed."""
        if not self._current_annotation or not self._canvas:
            return
        anns = self._canvas._annotations
        locked = len(anns) > 0 and all(a.confirmed for a in anns)
        self._canvas.set_locked(locked)

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

    def _on_class_filter_changed(self, text: str) -> None:
        class_name = None if text == "所有类别" else text
        self._file_list.set_class_filter(class_name)

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
        elif key == Qt.Key_A and (mod & Qt.ControlModifier) and (mod & Qt.ShiftModifier):
            self.auto_label_batch_requested.emit()
        elif key == Qt.Key_A and (mod & Qt.ShiftModifier):
            self.auto_label_single_requested.emit()
        elif key == Qt.Key_D or key == Qt.Key_Right:
            self._save_current()
            self._file_list.go_next()
        elif (key == Qt.Key_A and not (mod & (Qt.ShiftModifier | Qt.ControlModifier))) or key == Qt.Key_Left:
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
                self._update_lock_state()
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
        elif key == Qt.Key_C and mod & Qt.ControlModifier:
            self._copy_annotation()
        elif key == Qt.Key_V and mod & Qt.ControlModifier:
            self._paste_annotation()
        elif key == Qt.Key_F5:
            if self._project:
                images = self._project.list_images()
                self._file_list.refresh_paths(images)
        elif key in (Qt.Key_Plus, Qt.Key_Equal) and mod & Qt.ControlModifier:
            self._canvas.zoom_in()
        elif key == Qt.Key_Minus and mod & Qt.ControlModifier:
            self._canvas.zoom_out()
        elif key == Qt.Key_0 and mod & Qt.ControlModifier:
            self._canvas.zoom_fit()
        else:
            super().keyPressEvent(event)

    def save_and_cleanup(self) -> None:
        """Save current work before closing."""
        self._save_current()

    def _on_images_dropped(self, paths: list[Path]) -> None:
        """Handle image files dropped onto the file list."""
        if not self._project:
            return
        image_dir = self._project.project_dir / self._project.config.image_dir
        image_dir.mkdir(parents=True, exist_ok=True)
        added = 0
        for src in paths:
            dst = image_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                added += 1
        if added > 0:
            images = self._project.list_images()
            self._file_list.refresh_paths(images)
            self.status_changed.emit(f"已导入 {added} 张图片")

    def get_current_image_path(self) -> Path | None:
        """Return the currently displayed image path."""
        return self._current_image_path

    def add_auto_annotations(self, annotations: list) -> None:
        """Add auto-label predictions to current image (unconfirmed)."""
        for ann in annotations:
            self._canvas._annotations.append(ann)
        self._push_undo()
        self._canvas.update()
        self._sync_annotations_to_panel()

    def get_unlabeled_image_paths(self) -> list[Path]:
        """Return paths of images with no annotations."""
        if not self._project:
            return []
        result = []
        for img_path in self._project.list_images():
            label_path = self._project.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia is None or len(ia.annotations) == 0:
                result.append(img_path)
        return result

    # ── Batch operations ──────────────────────────────────────

    def _on_batch_confirm(self, paths: list[Path]) -> None:
        """Batch confirm all annotations for the given images."""
        if not self._project:
            return
        self._save_current()
        count = 0
        for img_path in paths:
            label_path = self._project.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia and ia.annotations:
                for ann in ia.annotations:
                    ann.confirmed = True
                save_annotation(ia, label_path)
                self._file_list.set_status(img_path, ia.status)
                count += 1
        # Reload current image if it was in the batch
        if self._current_image_path and self._current_image_path in paths:
            self._on_image_selected(self._current_image_path)
        self._update_project_stats()
        self.status_changed.emit(f"批量确认: {count} 张图片")
        logger.info("Batch confirmed %d images", count)

    def _on_batch_delete(self, paths: list[Path]) -> None:
        """Batch delete all annotations for the given images."""
        if not self._project:
            return
        self._save_current()
        count = 0
        for img_path in paths:
            label_path = self._project.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia and ia.annotations:
                ia.annotations.clear()
                save_annotation(ia, label_path)
                self._file_list.set_status(img_path, "unlabeled")
                count += 1
        # Reload current image if it was in the batch
        if self._current_image_path and self._current_image_path in paths:
            self._on_image_selected(self._current_image_path)
        self._update_project_stats()
        self.status_changed.emit(f"批量删除标注: {count} 张图片")
        logger.info("Batch deleted annotations for %d images", count)
