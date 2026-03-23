"""Tests for labelme format import/export."""
import json
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation, Keypoint
from src.core.formats.labelme import export_labelme, import_labelme


class TestLabelmeExport:
    def test_export_bbox(self, tmp_path):
        ia = ImageAnnotation(
            image_path="test.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.5, 0.5), confirmed=True),
            ],
        )
        out_dir = tmp_path / "output"
        export_labelme([ia], out_dir)
        json_path = out_dir / "test.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["imagePath"] == "test.jpg"
        assert data["imageWidth"] == 640
        assert data["imageHeight"] == 480
        assert len(data["shapes"]) == 1
        shape = data["shapes"][0]
        assert shape["label"] == "person"
        assert shape["shape_type"] == "rectangle"
        # labelme rectangle: [[x1, y1], [x2, y2]] in pixels
        assert len(shape["points"]) == 2

    def test_export_keypoints(self, tmp_path):
        ia = ImageAnnotation(
            image_path="kp.jpg",
            image_size=(100, 100),
            annotations=[
                Annotation(
                    class_name="point",
                    class_id=0,
                    bbox=None,
                    keypoints=[Keypoint(0.5, 0.3, 2, "nose")],
                    confirmed=True,
                ),
            ],
        )
        out_dir = tmp_path / "output"
        export_labelme([ia], out_dir)
        data = json.loads((out_dir / "kp.json").read_text())
        assert len(data["shapes"]) == 1
        assert data["shapes"][0]["shape_type"] == "point"
        assert data["shapes"][0]["label"] == "nose"


class TestLabelmeImport:
    def test_import_rectangle(self, tmp_path):
        labelme_data = {
            "imagePath": "img.jpg",
            "imageWidth": 640,
            "imageHeight": 480,
            "shapes": [
                {
                    "label": "car",
                    "shape_type": "rectangle",
                    "points": [[160, 120], [480, 360]],
                    "flags": {},
                },
            ],
        }
        json_path = tmp_path / "img.json"
        json_path.write_text(json.dumps(labelme_data))
        results = import_labelme(tmp_path)
        assert len(results) == 1
        ia = results[0]
        assert ia.image_path == "img.jpg"
        assert ia.image_size == (640, 480)
        ann = ia.annotations[0]
        assert ann.class_name == "car"
        # Check normalized center bbox
        cx, cy, w, h = ann.bbox
        assert abs(cx - 0.5) < 0.01
        assert abs(cy - 0.5) < 0.01
        assert abs(w - 0.5) < 0.01
        assert abs(h - 0.5) < 0.01

    def test_import_point(self, tmp_path):
        labelme_data = {
            "imagePath": "kp.jpg",
            "imageWidth": 100,
            "imageHeight": 100,
            "shapes": [
                {
                    "label": "nose",
                    "shape_type": "point",
                    "points": [[50, 30]],
                    "flags": {},
                },
            ],
        }
        (tmp_path / "kp.json").write_text(json.dumps(labelme_data))
        results = import_labelme(tmp_path)
        ia = results[0]
        assert len(ia.annotations) == 1
        ann = ia.annotations[0]
        assert ann.bbox is None
        assert len(ann.keypoints) == 1
        assert ann.keypoints[0].label == "nose"
        assert abs(ann.keypoints[0].x - 0.5) < 0.01
