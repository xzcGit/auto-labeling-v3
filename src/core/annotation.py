"""Annotation data models for AutoLabel V3."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class Keypoint:
    """A single keypoint with normalized coordinates."""

    x: float
    y: float
    visible: int  # 0=invisible, 1=occluded, 2=visible
    label: str

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "visible": self.visible, "label": self.label}

    @classmethod
    def from_dict(cls, d: dict) -> Keypoint:
        return cls(x=d["x"], y=d["y"], visible=d["visible"], label=d["label"])

    def clamp(self) -> None:
        """Clamp coordinates to [0, 1]."""
        self.x = max(0.0, min(1.0, self.x))
        self.y = max(0.0, min(1.0, self.y))


@dataclass
class Annotation:
    """A single annotation (bbox, keypoints, or both)."""

    class_name: str
    class_id: int
    bbox: tuple[float, float, float, float] | None = None  # (cx, cy, w, h) normalized
    keypoints: list[Keypoint] = field(default_factory=list)
    confidence: float = 1.0
    confirmed: bool = True
    source: str = "manual"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "bbox": list(self.bbox) if self.bbox else None,
            "keypoints": [kp.to_dict() for kp in self.keypoints],
            "confidence": self.confidence,
            "confirmed": self.confirmed,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Annotation:
        bbox = tuple(d["bbox"]) if d.get("bbox") else None
        keypoints = [Keypoint.from_dict(kp) for kp in d.get("keypoints", [])]
        return cls(
            id=d["id"],
            class_name=d["class_name"],
            class_id=d["class_id"],
            bbox=bbox,
            keypoints=keypoints,
            confidence=d.get("confidence", 1.0),
            confirmed=d.get("confirmed", True),
            source=d.get("source", "manual"),
        )

    def clamp(self) -> None:
        """Clamp bbox and keypoints to [0, 1] image bounds."""
        if self.bbox:
            cx, cy, w, h = self.bbox
            x1 = max(0.0, cx - w / 2)
            y1 = max(0.0, cy - h / 2)
            x2 = min(1.0, cx + w / 2)
            y2 = min(1.0, cy + h / 2)
            self.bbox = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        for kp in self.keypoints:
            kp.clamp()


@dataclass
class ImageAnnotation:
    """All annotations for a single image."""

    image_path: str
    image_size: tuple[int, int]  # (width, height)
    annotations: list[Annotation] = field(default_factory=list)
    image_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "image_size": list(self.image_size),
            "image_tags": self.image_tags,
            "annotations": [ann.to_dict() for ann in self.annotations],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ImageAnnotation:
        return cls(
            image_path=d["image_path"],
            image_size=tuple(d["image_size"]),
            annotations=[Annotation.from_dict(a) for a in d.get("annotations", [])],
            image_tags=d.get("image_tags", []),
        )

    @property
    def confirmed_count(self) -> int:
        return sum(1 for a in self.annotations if a.confirmed)

    @property
    def unconfirmed_count(self) -> int:
        return sum(1 for a in self.annotations if not a.confirmed)

    @property
    def status(self) -> str:
        """Return 'unlabeled', 'confirmed', or 'pending'."""
        if not self.annotations:
            return "unlabeled"
        if all(a.confirmed for a in self.annotations):
            return "confirmed"
        return "pending"
