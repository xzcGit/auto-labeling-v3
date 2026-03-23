"""Tests for annotation data models."""
import uuid

from src.core.annotation import Annotation, ImageAnnotation, Keypoint


class TestKeypoint:
    def test_create_keypoint(self):
        kp = Keypoint(x=0.5, y=0.3, visible=2, label="nose")
        assert kp.x == 0.5
        assert kp.y == 0.3
        assert kp.visible == 2
        assert kp.label == "nose"

    def test_to_dict(self):
        kp = Keypoint(x=0.25, y=0.15, visible=2, label="left_eye")
        d = kp.to_dict()
        assert d == {"x": 0.25, "y": 0.15, "visible": 2, "label": "left_eye"}

    def test_from_dict(self):
        d = {"x": 0.25, "y": 0.15, "visible": 2, "label": "left_eye"}
        kp = Keypoint.from_dict(d)
        assert kp.x == 0.25
        assert kp.label == "left_eye"

    def test_clamp_coordinates(self):
        kp = Keypoint(x=1.2, y=-0.1, visible=2, label="test")
        kp.clamp()
        assert kp.x == 1.0
        assert kp.y == 0.0


class TestAnnotation:
    def test_create_manual_bbox(self):
        ann = Annotation(
            class_name="person",
            class_id=0,
            bbox=(0.5, 0.4, 0.3, 0.6),
        )
        assert ann.class_name == "person"
        assert ann.bbox == (0.5, 0.4, 0.3, 0.6)
        assert ann.keypoints == []
        assert ann.confidence == 1.0
        assert ann.confirmed is True
        assert ann.source == "manual"
        # id should be a valid UUID
        uuid.UUID(ann.id)

    def test_create_auto_bbox(self):
        ann = Annotation(
            class_name="car",
            class_id=1,
            bbox=(0.6, 0.3, 0.25, 0.35),
            confidence=0.87,
            confirmed=False,
            source="auto",
        )
        assert ann.confirmed is False
        assert ann.source == "auto"
        assert ann.confidence == 0.87

    def test_to_dict_roundtrip(self):
        ann = Annotation(
            class_name="person",
            class_id=0,
            bbox=(0.5, 0.4, 0.3, 0.6),
            keypoints=[Keypoint(0.25, 0.15, 2, "nose")],
        )
        d = ann.to_dict()
        restored = Annotation.from_dict(d)
        assert restored.class_name == ann.class_name
        assert restored.bbox == ann.bbox
        assert len(restored.keypoints) == 1
        assert restored.keypoints[0].label == "nose"
        assert restored.id == ann.id

    def test_annotation_without_bbox(self):
        ann = Annotation(
            class_name="point",
            class_id=0,
            bbox=None,
            keypoints=[Keypoint(0.5, 0.5, 2, "center")],
        )
        assert ann.bbox is None
        d = ann.to_dict()
        assert d["bbox"] is None
        restored = Annotation.from_dict(d)
        assert restored.bbox is None

    def test_clamp_bbox(self):
        ann = Annotation(
            class_name="test",
            class_id=0,
            bbox=(0.5, 0.5, 1.2, 0.8),  # width exceeds
        )
        ann.clamp()
        x, y, w, h = ann.bbox
        # After clamp, bbox should not extend beyond [0,1]
        assert x - w / 2 >= 0.0
        assert x + w / 2 <= 1.0
        assert y - h / 2 >= 0.0
        assert y + h / 2 <= 1.0


class TestImageAnnotation:
    def test_create_empty(self):
        ia = ImageAnnotation(
            image_path="img_001.jpg",
            image_size=(1920, 1080),
        )
        assert ia.annotations == []
        assert ia.image_tags == []

    def test_to_dict_roundtrip(self):
        ann = Annotation(class_name="person", class_id=0, bbox=(0.5, 0.4, 0.3, 0.6))
        ia = ImageAnnotation(
            image_path="img_001.jpg",
            image_size=(1920, 1080),
            annotations=[ann],
            image_tags=["outdoor"],
        )
        d = ia.to_dict()
        restored = ImageAnnotation.from_dict(d)
        assert restored.image_path == "img_001.jpg"
        assert restored.image_size == (1920, 1080)
        assert len(restored.annotations) == 1
        assert restored.image_tags == ["outdoor"]

    def test_confirmed_count(self):
        a1 = Annotation(class_name="a", class_id=0, bbox=(0.5, 0.5, 0.1, 0.1), confirmed=True)
        a2 = Annotation(class_name="b", class_id=1, bbox=(0.2, 0.2, 0.1, 0.1), confirmed=False, source="auto", confidence=0.8)
        ia = ImageAnnotation(
            image_path="test.jpg",
            image_size=(100, 100),
            annotations=[a1, a2],
        )
        assert ia.confirmed_count == 1
        assert ia.unconfirmed_count == 1

    def test_status(self):
        # Empty
        ia = ImageAnnotation(image_path="a.jpg", image_size=(100, 100))
        assert ia.status == "unlabeled"

        # All confirmed
        ia.annotations = [
            Annotation(class_name="a", class_id=0, bbox=(0.5, 0.5, 0.1, 0.1), confirmed=True),
        ]
        assert ia.status == "confirmed"

        # Has unconfirmed
        ia.annotations.append(
            Annotation(class_name="b", class_id=1, bbox=(0.2, 0.2, 0.1, 0.1), confirmed=False, source="auto", confidence=0.8),
        )
        assert ia.status == "pending"
