"""Tests for YOLO format import/export."""
import yaml
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation, Keypoint
from src.core.formats.yolo import (
    export_yolo_detection,
    import_yolo_detection,
    export_yolo_pose,
    import_yolo_pose,
)


class TestYoloDetectionExport:
    def test_export_single_image(self, tmp_path):
        ia = ImageAnnotation(
            image_path="img.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(class_name="person", class_id=0, bbox=(0.5, 0.4, 0.3, 0.6), confirmed=True),
                Annotation(class_name="car", class_id=1, bbox=(0.2, 0.3, 0.1, 0.2), confirmed=True),
            ],
        )
        out_dir = tmp_path / "output"
        export_yolo_detection([ia], out_dir, classes=["person", "car"])
        txt_path = out_dir / "labels" / "img.txt"
        assert txt_path.exists()
        lines = txt_path.read_text().strip().split("\n")
        assert len(lines) == 2
        parts = lines[0].split()
        assert parts[0] == "0"  # class_id
        assert len(parts) == 5  # id cx cy w h

    def test_export_generates_data_yaml(self, tmp_path):
        ia = ImageAnnotation(image_path="a.jpg", image_size=(100, 100), annotations=[])
        out_dir = tmp_path / "output"
        export_yolo_detection([ia], out_dir, classes=["person", "car"])
        yaml_path = out_dir / "data.yaml"
        assert yaml_path.exists()
        data = yaml.safe_load(yaml_path.read_text())
        assert data["nc"] == 2
        assert data["names"] == ["person", "car"]

    def test_export_only_confirmed(self, tmp_path):
        ia = ImageAnnotation(
            image_path="img.jpg",
            image_size=(100, 100),
            annotations=[
                Annotation(class_name="a", class_id=0, bbox=(0.5, 0.5, 0.1, 0.1), confirmed=True),
                Annotation(class_name="b", class_id=1, bbox=(0.2, 0.2, 0.1, 0.1), confirmed=False, source="auto", confidence=0.8),
            ],
        )
        out_dir = tmp_path / "output"
        export_yolo_detection([ia], out_dir, classes=["a", "b"], only_confirmed=True)
        lines = (out_dir / "labels" / "img.txt").read_text().strip().split("\n")
        assert len(lines) == 1


class TestYoloDetectionImport:
    def test_import_single_file(self, tmp_path):
        # Create YOLO structure
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "img.txt").write_text("0 0.5 0.4 0.3 0.6\n1 0.2 0.3 0.1 0.2\n")
        classes = ["person", "car"]
        results = import_yolo_detection(labels_dir, classes)
        assert len(results) == 1
        ia = results[0]
        assert ia.image_path == "img"
        assert len(ia.annotations) == 2
        assert ia.annotations[0].class_name == "person"
        assert ia.annotations[0].bbox == (0.5, 0.4, 0.3, 0.6)
        assert ia.annotations[0].confirmed is True
        assert ia.annotations[0].source == "manual"

    def test_import_with_data_yaml(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text(yaml.dump({"names": ["cat", "dog"], "nc": 2}))
        results = import_yolo_detection(labels_dir, classes=None, data_yaml=yaml_path)
        assert results[0].annotations[0].class_name == "cat"


class TestYoloPoseExport:
    def test_export_with_keypoints(self, tmp_path):
        ia = ImageAnnotation(
            image_path="pose.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(
                    class_name="person",
                    class_id=0,
                    bbox=(0.5, 0.5, 0.3, 0.6),
                    keypoints=[
                        Keypoint(0.45, 0.3, 2, "nose"),
                        Keypoint(0.50, 0.35, 1, "left_eye"),
                    ],
                    confirmed=True,
                ),
            ],
        )
        out_dir = tmp_path / "output"
        export_yolo_pose([ia], out_dir, classes=["person"], kpt_dim=3)
        txt = (out_dir / "labels" / "pose.txt").read_text().strip()
        parts = txt.split()
        # class_id + 4 bbox + 2 keypoints * 3 dims = 11
        assert len(parts) == 11
        assert parts[0] == "0"


class TestYoloPoseImport:
    def test_import_pose(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        # class cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v
        (labels_dir / "p.txt").write_text("0 0.5 0.5 0.3 0.6 0.45 0.3 2 0.50 0.35 1\n")
        results = import_yolo_pose(
            labels_dir,
            classes=["person"],
            kpt_labels=["nose", "left_eye"],
            kpt_dim=3,
        )
        assert len(results) == 1
        ann = results[0].annotations[0]
        assert ann.bbox == (0.5, 0.5, 0.3, 0.6)
        assert len(ann.keypoints) == 2
        assert ann.keypoints[0].label == "nose"
        assert ann.keypoints[0].visible == 2
