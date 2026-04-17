"""Annotation list and properties panel."""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QGroupBox,
    QComboBox,
    QPushButton,
    QMenu,
    QInputDialog,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

from src.core.annotation import Annotation


class AnnotationPanel(QWidget):
    """Right-side panel showing annotation tree, properties, and image tags.

    Signals:
        annotation_clicked(str): Annotation ID clicked in the tree.
        keypoint_clicked(str, int): Keypoint clicked — (ann_id, kp_index).
        keypoint_rename_requested(str, int, str): (ann_id, kp_idx, new_label).
        keypoint_visibility_requested(str, int): (ann_id, kp_idx) — cycle visibility.
        keypoint_delete_requested(str, int): (ann_id, kp_idx).
    """

    annotation_clicked = pyqtSignal(str)
    keypoint_clicked = pyqtSignal(str, int)
    keypoint_rename_requested = pyqtSignal(str, int, str)
    keypoint_visibility_requested = pyqtSignal(str, int)
    keypoint_delete_requested = pyqtSignal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._annotations: list[Annotation] = []
        self._selected_id: str | None = None
        self._selected_kp_idx: int | None = None
        self._classes: list[str] = []
        self._class_colors: dict[str, str] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Image tags section
        tags_group = QGroupBox("图片分类")
        tags_layout = QVBoxLayout(tags_group)

        tag_add_layout = QHBoxLayout()
        self._tags_combo = QComboBox()
        self._tags_combo.setPlaceholderText("选择分类标签...")
        tag_add_layout.addWidget(self._tags_combo)
        self._btn_add_tag = QPushButton("+")
        self._btn_add_tag.setFixedWidth(30)
        self._btn_add_tag.setToolTip("添加标签")
        self._btn_add_tag.clicked.connect(self._on_add_tag)
        tag_add_layout.addWidget(self._btn_add_tag)
        tags_layout.addLayout(tag_add_layout)

        self._tags_list = QListWidget()
        self._tags_list.setMaximumHeight(60)
        tags_layout.addWidget(self._tags_list)

        self._btn_remove_tag = QPushButton("移除选中标签")
        self._btn_remove_tag.setFixedHeight(24)
        self._btn_remove_tag.clicked.connect(self._on_remove_tag)
        tags_layout.addWidget(self._btn_remove_tag)

        layout.addWidget(tags_group)

        # Annotation tree
        ann_group = QGroupBox("标注列表")
        ann_layout = QVBoxLayout(ann_group)
        self._ann_tree = QTreeWidget()
        self._ann_tree.setHeaderHidden(True)
        self._ann_tree.setIndentation(16)
        self._ann_tree.currentItemChanged.connect(self._on_tree_item_changed)
        self._ann_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._ann_tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        ann_layout.addWidget(self._ann_tree)
        layout.addWidget(ann_group)

        # Properties section
        props_group = QGroupBox("属性")
        props_layout = QVBoxLayout(props_group)

        self._class_label = QLabel("")
        self._conf_label = QLabel("")
        self._status_label = QLabel("")
        self._source_label = QLabel("")
        self._bbox_label = QLabel("")

        for lbl in [self._class_label, self._conf_label, self._status_label,
                     self._source_label, self._bbox_label]:
            lbl.setStyleSheet("color: #cdd6f4; font-size: 12px;")
            props_layout.addWidget(lbl)

        layout.addWidget(props_group)

        # Stats (current image)
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        layout.addWidget(self._stats_label)

        # Project stats section
        self._stats_group = QGroupBox("项目统计")
        stats_layout = QVBoxLayout(self._stats_group)

        self._project_total_label = QLabel("总图片: 0")
        self._project_labeled_label = QLabel("已标注: 0")
        self._project_confirmed_label = QLabel("全确认: 0")
        self._project_ann_count_label = QLabel("总标注数: 0")

        for lbl in [self._project_total_label, self._project_labeled_label,
                     self._project_confirmed_label, self._project_ann_count_label]:
            lbl.setStyleSheet("color: #cdd6f4; font-size: 11px;")
            stats_layout.addWidget(lbl)

        # Class distribution
        self._class_dist_label = QLabel("类别分布:")
        self._class_dist_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        stats_layout.addWidget(self._class_dist_label)

        self._class_dist_list = QListWidget()
        self._class_dist_list.setMaximumHeight(120)
        self._class_dist_list.setStyleSheet("font-size: 11px;")
        stats_layout.addWidget(self._class_dist_list)

        layout.addWidget(self._stats_group)

        layout.addStretch()

    def set_classes(self, classes: list[str]) -> None:
        """Set available classes for image tags."""
        self._classes = classes
        self._tags_combo.clear()
        for cls in classes:
            self._tags_combo.addItem(cls)

    def set_class_colors(self, colors: dict[str, str]) -> None:
        """Set class color mapping."""
        self._class_colors = colors

    def set_annotations(self, annotations: list[Annotation]) -> None:
        """Update the annotation tree."""
        self._annotations = list(annotations)
        self._ann_tree.blockSignals(True)
        self._ann_tree.clear()

        for ann in annotations:
            color = QColor(self._class_colors.get(ann.class_name, "#89b4fa"))
            status_icon = "\u2713" if ann.confirmed else "\u26a1"
            type_hint = ""
            if ann.bbox and ann.keypoints:
                type_hint = f" [bbox+kp\u00d7{len(ann.keypoints)}]"
            elif ann.bbox:
                type_hint = " [bbox]"
            elif ann.keypoints:
                type_hint = f" [kp\u00d7{len(ann.keypoints)}]"

            top_item = QTreeWidgetItem([f"{status_icon} {ann.class_name}{type_hint}"])
            top_item.setData(0, Qt.UserRole, ann.id)
            top_item.setData(0, Qt.UserRole + 1, -1)  # -1 = annotation level
            top_item.setForeground(0, color)
            self._ann_tree.addTopLevelItem(top_item)

            # Add keypoint children
            for i, kp in enumerate(ann.keypoints):
                vis_names = ["\u25cb", "\u25d1", "\u25cf"]  # empty, half, full circle
                vis_icon = vis_names[kp.visible] if kp.visible < 3 else "?"
                child = QTreeWidgetItem([f"  {vis_icon} {kp.label}"])
                child.setData(0, Qt.UserRole, ann.id)
                child.setData(0, Qt.UserRole + 1, i)  # keypoint index
                child.setForeground(0, color)
                top_item.addChild(child)

            if ann.keypoints:
                top_item.setExpanded(True)

        self._ann_tree.blockSignals(False)
        self._update_stats()

    def select_annotation(self, ann_id: str | None) -> None:
        """Select an annotation in the tree and show its properties."""
        self._selected_id = ann_id
        self._selected_kp_idx = None
        if ann_id is None:
            self._ann_tree.clearSelection()
            self._clear_properties()
            return

        for i in range(self._ann_tree.topLevelItemCount()):
            item = self._ann_tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == ann_id:
                self._ann_tree.blockSignals(True)
                self._ann_tree.setCurrentItem(item)
                self._ann_tree.blockSignals(False)
                break

        ann = self._find_annotation(ann_id)
        if ann:
            self._show_properties(ann)

    def select_keypoint(self, ann_id: str, kp_idx: int) -> None:
        """Select a specific keypoint in the tree."""
        self._selected_id = ann_id
        self._selected_kp_idx = kp_idx

        for i in range(self._ann_tree.topLevelItemCount()):
            top = self._ann_tree.topLevelItem(i)
            if top.data(0, Qt.UserRole) == ann_id:
                top.setExpanded(True)
                if 0 <= kp_idx < top.childCount():
                    self._ann_tree.blockSignals(True)
                    self._ann_tree.setCurrentItem(top.child(kp_idx))
                    self._ann_tree.blockSignals(False)
                break

        ann = self._find_annotation(ann_id)
        if ann and 0 <= kp_idx < len(ann.keypoints):
            self._show_keypoint_properties(ann, kp_idx)

    def set_image_tags(self, tags: list[str]) -> None:
        """Set the current image's classification tags."""
        self._tags_list.clear()
        for tag in tags:
            self._tags_list.addItem(tag)

    def get_image_tags(self) -> list[str]:
        """Get the current image tags."""
        return [self._tags_list.item(i).text() for i in range(self._tags_list.count())]

    def set_project_stats(self, stats: dict) -> None:
        """Update project-level statistics."""
        self._project_total_label.setText(f"总图片: {stats.get('total_images', 0)}")
        self._project_labeled_label.setText(f"已标注: {stats.get('labeled_images', 0)}")
        self._project_confirmed_label.setText(f"全确认: {stats.get('confirmed_images', 0)}")
        self._project_ann_count_label.setText(f"总标注数: {stats.get('total_annotations', 0)}")

        self._class_dist_list.clear()
        class_counts = stats.get("class_counts", {})
        for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            color = self._class_colors.get(cls_name, "#89b4fa")
            item = QListWidgetItem(f"{cls_name}: {count}")
            item.setForeground(QColor(color))
            self._class_dist_list.addItem(item)

    def clear(self) -> None:
        """Clear all state."""
        self._annotations = []
        self._selected_id = None
        self._selected_kp_idx = None
        self._ann_tree.clear()
        self._tags_list.clear()
        self._clear_properties()
        self._stats_label.setText("")
        self._class_dist_list.clear()

    def _show_properties(self, ann: Annotation) -> None:
        self._class_label.setText(f"类别: {ann.class_name}")
        self._conf_label.setText(f"置信度: {ann.confidence:.2f}")
        self._status_label.setText(f"状态: {'已确认' if ann.confirmed else '待确认'}")
        self._source_label.setText(f"来源: {'手动' if ann.source == 'manual' else '自动'}")
        if ann.bbox:
            cx, cy, w, h = ann.bbox
            self._bbox_label.setText(f"Bbox: ({cx:.3f}, {cy:.3f}, {w:.3f}, {h:.3f})")
        elif ann.keypoints:
            self._bbox_label.setText(f"关键点: {len(ann.keypoints)} 个")
        else:
            self._bbox_label.setText("")

    def _show_keypoint_properties(self, ann: Annotation, kp_idx: int) -> None:
        kp = ann.keypoints[kp_idx]
        vis_names = ["不可见", "被遮挡", "可见"]
        vis = vis_names[kp.visible] if kp.visible < 3 else "?"
        self._class_label.setText(f"关键点: {kp.label}")
        self._conf_label.setText(f"所属: {ann.class_name}")
        self._status_label.setText(f"可见性: {vis}")
        self._source_label.setText(f"坐标: ({kp.x:.4f}, {kp.y:.4f})")
        self._bbox_label.setText(f"索引: {kp_idx}/{len(ann.keypoints)}")

    def _clear_properties(self) -> None:
        self._class_label.setText("")
        self._conf_label.setText("")
        self._status_label.setText("")
        self._source_label.setText("")
        self._bbox_label.setText("")

    def _update_stats(self) -> None:
        total = len(self._annotations)
        confirmed = sum(1 for a in self._annotations if a.confirmed)
        pending = total - confirmed
        self._stats_label.setText(f"标注: {total} | 确认: {confirmed} | 待确认: {pending}")

    def _find_annotation(self, ann_id: str) -> Annotation | None:
        for ann in self._annotations:
            if ann.id == ann_id:
                return ann
        return None

    def _on_tree_item_changed(self, current, previous) -> None:
        if current is None:
            return
        ann_id = current.data(0, Qt.UserRole)
        kp_idx = current.data(0, Qt.UserRole + 1)
        if ann_id is None:
            return
        if kp_idx is not None and kp_idx >= 0:
            self._selected_kp_idx = kp_idx
            self.keypoint_clicked.emit(ann_id, kp_idx)
            ann = self._find_annotation(ann_id)
            if ann:
                self._show_keypoint_properties(ann, kp_idx)
        else:
            self._selected_kp_idx = None
            self.annotation_clicked.emit(ann_id)
            ann = self._find_annotation(ann_id)
            if ann:
                self._show_properties(ann)

    def _on_tree_context_menu(self, pos) -> None:
        item = self._ann_tree.itemAt(pos)
        if not item:
            return
        ann_id = item.data(0, Qt.UserRole)
        kp_idx = item.data(0, Qt.UserRole + 1)
        if ann_id is None:
            return

        ann = self._find_annotation(ann_id)
        if not ann:
            return

        if kp_idx is not None and kp_idx >= 0 and kp_idx < len(ann.keypoints):
            kp = ann.keypoints[kp_idx]
            menu = QMenu(self)

            rename = menu.addAction(f"重命名 ({kp.label})")
            rename.triggered.connect(
                lambda _, aid=ann_id, ki=kp_idx, old=kp.label: self._rename_keypoint(aid, ki, old))

            vis_names = ["不可见", "被遮挡", "可见"]
            vis = vis_names[kp.visible] if kp.visible < 3 else "?"
            toggle_vis = menu.addAction(f"切换可见性 ({vis})")
            toggle_vis.triggered.connect(
                lambda _, aid=ann_id, ki=kp_idx: self.keypoint_visibility_requested.emit(aid, ki))

            delete = menu.addAction("删除关键点")
            delete.triggered.connect(
                lambda _, aid=ann_id, ki=kp_idx: self.keypoint_delete_requested.emit(aid, ki))

            menu.exec_(self._ann_tree.viewport().mapToGlobal(pos))

    def _rename_keypoint(self, ann_id: str, kp_idx: int, old_label: str) -> None:
        new_label, ok = QInputDialog.getText(self, "重命名关键点", "标签:", text=old_label)
        if ok and new_label.strip():
            self.keypoint_rename_requested.emit(ann_id, kp_idx, new_label.strip())

    def _on_add_tag(self) -> None:
        """Add selected class as image tag."""
        tag = self._tags_combo.currentText()
        if not tag:
            return
        for i in range(self._tags_list.count()):
            if self._tags_list.item(i).text() == tag:
                return
        self._tags_list.addItem(tag)

    def _on_remove_tag(self) -> None:
        """Remove selected tag from the list."""
        row = self._tags_list.currentRow()
        if row >= 0:
            self._tags_list.takeItem(row)
