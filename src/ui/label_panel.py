"""Label panel — main annotation workspace assembling file list, canvas, and properties."""
from __future__ import annotations

import logging
import shutil
from collections import OrderedDict
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QPushButton,
    QToolBar,
    QToolButton,
    QLabel,
    QComboBox,
    QAction,
    QMessageBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint

from src.core.annotation import Annotation, ImageAnnotation
from src.core.label_io import save_annotation, load_annotation
from src.core.project import ProjectManager
from src.ui.canvas import AnnotationCanvas
from src.ui.file_list import FileListWidget
from src.ui.properties import AnnotationPanel
from src.ui.class_picker import ClassPickerPopup, KeypointLabelPicker
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
        self._undo_stacks: OrderedDict[str, UndoStack] = OrderedDict()  # per-image undo, LRU
        self._last_class: str | None = None
        self._config_path = config_path
        self._clipboard: list[dict] | None = None  # copied annotations as dicts
        self._image_cache = ImageCache(max_count=16, max_memory_mb=512.0)
        self._stats_cache: dict = {}  # cached project stats
        self._prev_annotations_snapshot: list[tuple] | None = None  # immutable stats snapshot

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

        self._btn_confirm_visible = QPushButton(icon("confirm_visible"), "确认可见预标注")
        self._btn_confirm_visible.setToolTip("确认当前可见图片的所有未确认标注")
        self._toolbar.addWidget(self._btn_confirm_visible)

        self._btn_revert_visible = QPushButton(icon("revert_visible"), "撤销可见预标注")
        self._btn_revert_visible.setToolTip("删除当前可见图片的所有未确认标注")
        self._toolbar.addWidget(self._btn_revert_visible)

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

        # Left pane: refresh toolbar + file list
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)

        file_toolbar = QWidget()
        tb_layout = QHBoxLayout(file_toolbar)
        tb_layout.setContentsMargins(4, 2, 4, 2)
        tb_layout.setSpacing(4)
        self._refresh_btn = QToolButton()
        self._refresh_btn.setIcon(icon("refresh"))
        self._refresh_btn.setToolTip("刷新图像列表 (F5)")
        self._refresh_btn.setEnabled(False)
        self._refresh_btn.clicked.connect(self._on_refresh_clicked)
        tb_layout.addWidget(self._refresh_btn)
        tb_layout.addStretch(1)

        left_layout.addWidget(file_toolbar)
        left_layout.addWidget(self._file_list, 1)
        left_pane.setMaximumWidth(250)

        self._splitter.addWidget(left_pane)

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
        self._canvas.keypoint_attach_requested.connect(self._on_keypoint_attach_requested)
        self._canvas.keypoint_selected.connect(self._ann_panel.select_keypoint)

        # Properties panel
        self._ann_panel.annotation_clicked.connect(self._canvas.select_annotation)
        self._ann_panel.keypoint_clicked.connect(self._on_panel_keypoint_clicked)
        self._ann_panel.keypoint_rename_requested.connect(self._on_keypoint_rename)
        self._ann_panel.keypoint_visibility_requested.connect(self._on_keypoint_visibility)
        self._ann_panel.keypoint_delete_requested.connect(self._on_keypoint_delete)

        # Confirm all
        self._btn_confirm_all.clicked.connect(self._confirm_all)
        self._btn_confirm_visible.clicked.connect(self._batch_confirm_visible)
        self._btn_revert_visible.clicked.connect(self._batch_revert_visible)

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
        self._init_stats_cache()
        self._refresh_btn.setEnabled(True)

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
            else:
                self._undo_stacks.move_to_end(key)
            while len(self._undo_stacks) > self._UNDO_MAX_IMAGES:
                self._undo_stacks.popitem(last=False)

            # Snapshot annotations for incremental stats on save
            self._prev_annotations_snapshot = self._stats_snapshot(self._current_annotation.annotations)

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
        self._current_annotation.annotations = list(self._canvas.annotations)
        self._current_annotation.image_tags = self._ann_panel.get_image_tags()
        label_path = self._project.label_path_for(self._current_image_path)
        save_annotation(self._current_annotation, label_path)
        logger.debug("Saved annotations for %s", self._current_image_path.name)
        # Update file list status
        self._file_list.set_status(self._current_image_path, self._current_annotation.status)
        # Incremental stats update
        old_snap = self._prev_annotations_snapshot or []
        new_snap = self._stats_snapshot(self._current_annotation.annotations)
        if old_snap != new_snap:
            self._update_stats_incremental(old_snap, new_snap)
            self._prev_annotations_snapshot = new_snap

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
        self._canvas.remove_annotation(ann_id)
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_annotations_changed(self) -> None:
        self._sync_annotations_to_panel()

    def _show_class_picker(self, default_class: str | None, px: float, py: float) -> str | None:
        """Show class picker popup and handle new class creation. Returns class name or None."""
        if not self._project:
            return None
        classes = self._project.config.classes
        colors = {cls: self._project.config.get_class_color(cls) for cls in classes}

        picker = ClassPickerPopup(
            classes=classes,
            colors=colors,
            default_class=default_class,
            parent=self,
        )
        global_pos = self._canvas.mapToGlobal(QPoint(int(px), int(py)))
        picker.move(global_pos)
        if not picker.exec_():
            return None

        cls_name = picker.get_selected_class()
        if cls_name is None:
            return None

        if picker.is_new_class():
            self._project.add_class(cls_name)
            self._project.save()
            colors[cls_name] = self._project.config.get_class_color(cls_name)
            self._canvas.set_class_colors(colors)
            self._ann_panel.set_class_colors(colors)
            self._ann_panel.set_classes(self._project.config.classes)

        return cls_name

    def _on_class_requested(self, px: float, py: float) -> None:
        """Show class picker and create annotation."""
        cls_name = self._show_class_picker(self._last_class, px, py)
        if cls_name is None:
            self._clear_draw_state()
            return

        cls_id = self._project.config.get_class_id(cls_name)
        self._last_class = cls_name
        if self._canvas.tool_mode == "draw_bbox":
            self._canvas.create_bbox_from_draw(cls_name, cls_id)
        elif self._canvas.tool_mode == "draw_keypoint":
            self._canvas.create_keypoint_at(cls_name, cls_id)

    def _clear_draw_state(self) -> None:
        """Clear canvas draw state after cancelled operation."""
        self._canvas.clear_draw_state()

    def _on_class_change_requested(self, ann_id: str, px: float, py: float) -> None:
        """Show class picker to change an annotation's class."""
        ann = None
        for a in self._canvas.annotations:
            if a.id == ann_id:
                ann = a
                break
        if ann is None:
            return

        cls_name = self._show_class_picker(ann.class_name, px, py)
        if cls_name is None or cls_name == ann.class_name:
            return

        ann.class_name = cls_name
        ann.class_id = self._project.config.get_class_id(cls_name)
        self._push_undo()
        self._canvas.update()
        self._sync_annotations_to_panel()

    def _on_keypoint_attach_requested(self, ann_id: str, px: float, py: float) -> None:
        """Show keypoint label picker and attach keypoint to existing annotation."""
        from src.core.annotation import Keypoint

        ann = next((a for a in self._canvas.annotations if a.id == ann_id), None)
        if ann is None or not self._canvas._draw_start:
            self._clear_draw_state()
            return

        # Collect existing keypoint labels from current image
        existing_labels: list[str] = []
        seen: set[str] = set()
        for a in self._canvas.annotations:
            for kp in a.keypoints:
                if kp.label not in seen:
                    existing_labels.append(kp.label)
                    seen.add(kp.label)

        default_label = f"kp_{len(ann.keypoints)}"

        picker = KeypointLabelPicker(
            existing_labels=existing_labels,
            default_label=default_label,
            parent=self,
        )
        global_pos = self._canvas.mapToGlobal(QPoint(int(px), int(py)))
        picker.move(global_pos)

        if not picker.exec_():
            self._clear_draw_state()
            return

        label = picker.get_label()
        if not label:
            self._clear_draw_state()
            return

        nx, ny = self._canvas._draw_start
        kp = Keypoint(x=nx, y=ny, visible=2, label=label)
        self._canvas.add_keypoint_to_annotation(ann_id, kp)
        self._canvas._draw_start = None
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_panel_keypoint_clicked(self, ann_id: str, kp_idx: int) -> None:
        """Handle keypoint selection from the panel tree."""
        self._canvas.select_keypoint(ann_id, kp_idx)

    def _on_keypoint_rename(self, ann_id: str, kp_idx: int, new_label: str) -> None:
        """Handle keypoint rename from panel context menu."""
        self._canvas.rename_keypoint(ann_id, kp_idx, new_label)
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_keypoint_visibility(self, ann_id: str, kp_idx: int) -> None:
        """Handle keypoint visibility toggle from panel."""
        self._canvas.cycle_keypoint_visibility(ann_id, kp_idx)
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_keypoint_delete(self, ann_id: str, kp_idx: int) -> None:
        """Handle keypoint deletion from panel."""
        self._canvas.remove_keypoint(ann_id, kp_idx)
        self._push_undo()
        self._sync_annotations_to_panel()

    def _sync_annotations_to_panel(self) -> None:
        self._ann_panel.set_annotations(list(self._canvas.annotations))
        self._emit_status()

    def _emit_status(self) -> None:
        """Emit status bar text with current file info."""
        if not self._current_image_path:
            return
        idx, total = self._file_list.get_index_info()
        n_ann = len(self._canvas.annotations)
        n_confirmed = sum(1 for a in self._canvas.annotations if a.confirmed)
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
        """Compute project-level annotation statistics (full scan)."""
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

    def _init_stats_cache(self) -> None:
        """Full scan to initialize stats cache. Called once on project load."""
        self._stats_cache = self._compute_project_stats()
        self._ann_panel.set_project_stats(self._stats_cache)

    def _update_stats_incremental(self, old_snap: list[tuple], new_snap: list[tuple]) -> None:
        """Incrementally update cached stats. Each snapshot is [(class_name, confirmed), ...]."""
        if not self._stats_cache:
            return
        had_old = len(old_snap) > 0
        has_new = len(new_snap) > 0

        # labeled_images delta
        if had_old and not has_new:
            self._stats_cache["labeled_images"] -= 1
        elif not had_old and has_new:
            self._stats_cache["labeled_images"] += 1

        # confirmed_images delta
        old_all_confirmed = had_old and all(c for _, c in old_snap)
        new_all_confirmed = has_new and all(c for _, c in new_snap)
        if old_all_confirmed and not new_all_confirmed:
            self._stats_cache["confirmed_images"] -= 1
        elif not old_all_confirmed and new_all_confirmed:
            self._stats_cache["confirmed_images"] += 1

        # total_annotations and class_counts delta
        for cls, _ in old_snap:
            self._stats_cache["total_annotations"] -= 1
            self._stats_cache["class_counts"][cls] = self._stats_cache["class_counts"].get(cls, 1) - 1
            if self._stats_cache["class_counts"][cls] <= 0:
                del self._stats_cache["class_counts"][cls]

        for cls, _ in new_snap:
            self._stats_cache["total_annotations"] += 1
            self._stats_cache["class_counts"][cls] = self._stats_cache["class_counts"].get(cls, 0) + 1

        self._ann_panel.set_project_stats(self._stats_cache)

    @staticmethod
    def _stats_snapshot(anns) -> list[tuple]:
        """Capture immutable stats-relevant data from annotations."""
        return [(a.class_name, a.confirmed) for a in anns]

    def _update_project_stats(self) -> None:
        """Recompute and update project stats in the panel (full scan fallback)."""
        self._init_stats_cache()

    def _confirm_all(self) -> None:
        """Confirm all annotations on current image."""
        for ann in self._canvas.annotations:
            ann.confirmed = True
        self._push_undo()
        self._canvas.update()
        self._sync_annotations_to_panel()

    # ── Copy / Paste ──────────────────────────────────────────

    def _copy_annotation(self) -> None:
        """Copy selected annotation to clipboard."""
        ann = self._canvas.get_selected_annotation()
        if ann:
            self._clipboard = [ann.to_dict()]
            logger.debug("Copied annotation: %s", ann.class_name)

    def _on_annotation_copied(self, ann_id: str) -> None:
        """Handle copy via right-click menu."""
        for ann in self._canvas.annotations:
            if ann.id == ann_id:
                self._clipboard = [ann.to_dict()]
                logger.debug("Copied annotation via menu: %s", ann.class_name)
                break

    def _paste_annotation(self) -> None:
        """Paste clipboard annotations onto current image."""
        if not self._clipboard or self._canvas.is_locked:
            return
        import uuid as _uuid
        new_anns = []
        for ann_dict in self._clipboard:
            new_dict = dict(ann_dict)
            new_dict["id"] = str(_uuid.uuid4())
            new_dict["confirmed"] = False
            new_anns.append(Annotation.from_dict(new_dict))
        self._canvas.add_annotations(new_anns)
        self._push_undo()
        self._sync_annotations_to_panel()
        logger.debug("Pasted %d annotations", len(self._clipboard))

    # ── Undo/Redo ──────────────────────────────────────────────

    _UNDO_MAX_IMAGES = 20

    def _push_undo(self) -> None:
        if not self._current_image_path or not self._current_annotation:
            return
        self._current_annotation.annotations = list(self._canvas.annotations)
        key = str(self._current_image_path)
        if key not in self._undo_stacks:
            self._undo_stacks[key] = UndoStack()
        else:
            self._undo_stacks.move_to_end(key)
        self._undo_stacks[key].push(self._current_annotation.to_dict())
        # Evict oldest stacks if over limit
        while len(self._undo_stacks) > self._UNDO_MAX_IMAGES:
            self._undo_stacks.popitem(last=False)

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
            self.rescan_images()
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

    def rescan_images(self) -> int:
        """Re-scan project image dir; refresh list only if new files found.

        Returns:
            Number of newly discovered images (0 if none or no project).
        """
        if not self._project:
            return 0
        current = {str(p) for p in self._file_list.get_paths()}
        latest = self._project.list_images()
        added = len({str(p) for p in latest} - current)
        if added:
            self._file_list.refresh_paths(latest)
        return added

    def _on_refresh_clicked(self) -> None:
        """Handler for the refresh button in the file toolbar."""
        n = self.rescan_images()
        if n > 0:
            self.status_changed.emit(f"发现 {n} 张新图片")
        else:
            self.status_changed.emit("未发现新图片")

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

    def add_auto_annotations(self, annotations: list, overlap_iou: float = 0.5) -> None:
        """Add auto-label predictions to current image with conflict detection."""
        from src.core.annotation import find_conflicts

        existing = self._canvas.annotations
        conflicts, clean_preds = find_conflicts(existing, annotations, overlap_iou)

        # Add non-conflicting predictions directly
        self._canvas.add_annotations(clean_preds)

        # Add conflicting predictions and mark them as conflict pairs
        if conflicts:
            conflict_anns = [pred for _, pred in conflicts]
            self._canvas.add_annotations(conflict_anns)
            self._canvas.set_conflict_pairs(
                [(ex.id, pred.id) for ex, pred in conflicts])

        self._push_undo()
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
                old_snap = self._stats_snapshot(ia.annotations)
                for ann in ia.annotations:
                    ann.confirmed = True
                save_annotation(ia, label_path)
                self._file_list.set_status(img_path, ia.status)
                self._update_stats_incremental(old_snap, self._stats_snapshot(ia.annotations))
                count += 1
        # Reload current image if it was in the batch
        if self._current_image_path and self._current_image_path in paths:
            self._on_image_selected(self._current_image_path)
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
                old_snap = self._stats_snapshot(ia.annotations)
                ia.annotations.clear()
                save_annotation(ia, label_path)
                self._file_list.set_status(img_path, "unlabeled")
                self._update_stats_incremental(old_snap, [])
                count += 1
        # Reload current image if it was in the batch
        if self._current_image_path and self._current_image_path in paths:
            self._on_image_selected(self._current_image_path)
        self.status_changed.emit(f"批量删除标注: {count} 张图片")
        logger.info("Batch deleted annotations for %d images", count)

    def _collect_unconfirmed(self, visible_paths: list[Path]):
        """Load annotations for visible paths, return those with unconfirmed items.

        Returns (affected: list[(img_path, label_path, ia)], total_unconfirmed: int).
        """
        affected = []
        total = 0
        for img_path in visible_paths:
            label_path = self._project.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia:
                unconfirmed = sum(1 for a in ia.annotations if not a.confirmed)
                if unconfirmed > 0:
                    affected.append((img_path, label_path, ia))
                    total += unconfirmed
        return affected, total

    def _batch_confirm_visible(self) -> None:
        """Confirm all unconfirmed annotations in currently visible images."""
        if not self._project:
            return
        visible_paths = self._file_list.get_visible_paths()
        if not visible_paths:
            return

        affected, total = self._collect_unconfirmed(visible_paths)
        if total == 0:
            self.status_changed.emit("没有需要确认的预标注")
            return

        reply = QMessageBox.question(
            self, "确认可见预标注",
            f"将确认 {len(affected)} 张图片中的 {total} 个未确认标注，是否继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._save_current()
        count = 0
        for img_path, label_path, ia in affected:
            old_snap = self._stats_snapshot(ia.annotations)
            for ann in ia.annotations:
                if not ann.confirmed:
                    ann.confirmed = True
            save_annotation(ia, label_path)
            self._file_list.set_status(img_path, ia.status)
            self._update_stats_incremental(old_snap, self._stats_snapshot(ia.annotations))
            count += 1

        if self._current_image_path and self._current_image_path in visible_paths:
            self._on_image_selected(self._current_image_path)
        self.status_changed.emit(f"已确认可见预标注: {count} 张图片")
        logger.info("Batch confirmed visible unconfirmed annotations for %d images", count)

    def _batch_revert_visible(self) -> None:
        """Delete all unconfirmed annotations in currently visible images."""
        if not self._project:
            return
        visible_paths = self._file_list.get_visible_paths()
        if not visible_paths:
            return

        affected, total = self._collect_unconfirmed(visible_paths)
        if total == 0:
            self.status_changed.emit("没有需要撤销的预标注")
            return

        reply = QMessageBox.question(
            self, "撤销可见预标注",
            f"将删除 {len(affected)} 张图片中的 {total} 个未确认标注，此操作不可撤销，是否继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._save_current()
        count = 0
        for img_path, label_path, ia in affected:
            old_snap = self._stats_snapshot(ia.annotations)
            ia.annotations = [a for a in ia.annotations if a.confirmed]
            save_annotation(ia, label_path)
            self._file_list.set_status(img_path, ia.status)
            self._update_stats_incremental(old_snap, self._stats_snapshot(ia.annotations))
            count += 1

        if self._current_image_path and self._current_image_path in visible_paths:
            self._on_image_selected(self._current_image_path)
        self.status_changed.emit(f"已撤销可见预标注: {count} 张图片")
        logger.info("Batch reverted visible unconfirmed annotations for %d images", count)
