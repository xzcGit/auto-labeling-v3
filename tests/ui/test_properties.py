"""Tests for AnnotationPanel."""
import pytest
from PyQt5.QtCore import Qt


class TestAnnotationPanel:
    def test_set_annotations(self, qapp):
        from src.ui.properties import AnnotationPanel
        from src.core.annotation import Annotation

        panel = AnnotationPanel()
        anns = [
            Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4)),
            Annotation(class_name="dog", class_id=1, bbox=(0.2, 0.3, 0.1, 0.2)),
        ]
        panel.set_annotations(anns)
        assert panel._ann_tree.topLevelItemCount() == 2

    def test_set_annotations_with_keypoints(self, qapp):
        from src.ui.properties import AnnotationPanel
        from src.core.annotation import Annotation, Keypoint

        panel = AnnotationPanel()
        kps = [Keypoint(x=0.1, y=0.2, visible=2, label="nose"),
               Keypoint(x=0.3, y=0.4, visible=1, label="eye")]
        ann = Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), keypoints=kps)
        panel.set_annotations([ann])
        assert panel._ann_tree.topLevelItemCount() == 1
        top = panel._ann_tree.topLevelItem(0)
        assert top.childCount() == 2
        assert top.isExpanded()

    def test_clear(self, qapp):
        from src.ui.properties import AnnotationPanel
        from src.core.annotation import Annotation

        panel = AnnotationPanel()
        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4))
        panel.set_annotations([ann])
        panel.clear()
        assert panel._ann_tree.topLevelItemCount() == 0

    def test_select_annotation_shows_properties(self, qapp):
        from src.ui.properties import AnnotationPanel
        from src.core.annotation import Annotation

        panel = AnnotationPanel()
        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), confidence=0.95)
        panel.set_annotations([ann])
        panel.select_annotation(ann.id)
        assert "cat" in panel._class_label.text()

    def test_select_none_clears_properties(self, qapp):
        from src.ui.properties import AnnotationPanel
        from src.core.annotation import Annotation

        panel = AnnotationPanel()
        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4))
        panel.set_annotations([ann])
        panel.select_annotation(ann.id)
        panel.select_annotation(None)
        assert panel._class_label.text() == ""

    def test_select_keypoint_shows_properties(self, qapp):
        from src.ui.properties import AnnotationPanel
        from src.core.annotation import Annotation, Keypoint

        panel = AnnotationPanel()
        kp = Keypoint(x=0.1, y=0.2, visible=2, label="nose")
        ann = Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), keypoints=[kp])
        panel.set_annotations([ann])
        panel.select_keypoint(ann.id, 0)
        assert "nose" in panel._class_label.text()
        assert "person" in panel._conf_label.text()

    def test_set_image_tags(self, qapp):
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        panel.set_classes(["cat", "dog", "bird"])
        panel.set_image_tags(["cat", "bird"])
        tags = panel.get_image_tags()
        assert set(tags) == {"cat", "bird"}

    def test_stats_display(self, qapp):
        from src.ui.properties import AnnotationPanel
        from src.core.annotation import Annotation

        panel = AnnotationPanel()
        anns = [
            Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), confirmed=True),
            Annotation(class_name="dog", class_id=1, bbox=(0.2, 0.3, 0.1, 0.2), confirmed=False),
            Annotation(class_name="cat", class_id=0, bbox=(0.7, 0.7, 0.2, 0.2), confirmed=True),
        ]
        panel.set_annotations(anns)
        assert "2" in panel._stats_label.text()  # 2 confirmed


class TestProjectStats:
    def test_set_project_stats_displays_totals(self, qapp):
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        stats = {
            "total_images": 100,
            "labeled_images": 80,
            "confirmed_images": 50,
            "total_annotations": 200,
            "class_counts": {"cat": 120, "dog": 80},
        }
        panel.set_project_stats(stats)
        assert "100" in panel._project_total_label.text()
        assert "80" in panel._project_labeled_label.text()
        assert "50" in panel._project_confirmed_label.text()
        assert "200" in panel._project_ann_count_label.text()
        assert panel._class_dist_list.count() == 2

    def test_set_project_stats_empty(self, qapp):
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        panel.set_project_stats({})
        assert "0" in panel._project_total_label.text()
        assert panel._class_dist_list.count() == 0

    def test_class_distribution_sorted_by_count(self, qapp):
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        stats = {
            "total_images": 10,
            "class_counts": {"rare": 5, "common": 50, "mid": 20},
        }
        panel.set_project_stats(stats)
        assert panel._class_dist_list.count() == 3
        # First item should be 'common' (highest count)
        assert "common" in panel._class_dist_list.item(0).text()
        assert "mid" in panel._class_dist_list.item(1).text()
        assert "rare" in panel._class_dist_list.item(2).text()

    def test_class_distribution_uses_colors(self, qapp):
        from src.ui.properties import AnnotationPanel

        panel = AnnotationPanel()
        panel.set_class_colors({"cat": "#a6e3a1"})
        stats = {"class_counts": {"cat": 10}}
        panel.set_project_stats(stats)
        item = panel._class_dist_list.item(0)
        assert item.foreground().color().name() == "#a6e3a1"
