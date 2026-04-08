"""Inference engine — wraps ultralytics YOLO.predict."""
from __future__ import annotations

import logging
from pathlib import Path

from src.core.annotation import Annotation, Keypoint

logger = logging.getLogger(__name__)


class Predictor:
    """Wraps a YOLO model for inference, converting results to Annotations."""

    def __init__(self, model):
        """Initialize with a loaded ultralytics YOLO model instance."""
        self.model = model

    def predict(
        self,
        image_path: str | Path,
        conf: float = 0.5,
        iou: float = 0.45,
        project_classes: list[str] | None = None,
        kpt_labels: list[str] | None = None,
    ) -> list[Annotation]:
        """Run inference and return list of Annotations."""
        annotations, _ = self._run(image_path, conf, iou, project_classes, kpt_labels)
        return annotations

    def predict_with_size(
        self,
        image_path: str | Path,
        conf: float = 0.5,
        iou: float = 0.45,
        project_classes: list[str] | None = None,
        kpt_labels: list[str] | None = None,
    ) -> tuple[list[Annotation], tuple[int, int]]:
        """Run inference and return annotations + image size (w, h)."""
        return self._run(image_path, conf, iou, project_classes, kpt_labels)

    def _run(
        self,
        image_path: str | Path,
        conf: float,
        iou: float,
        project_classes: list[str] | None,
        kpt_labels: list[str] | None,
    ) -> tuple[list[Annotation], tuple[int, int]]:
        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            verbose=False,
        )
        logger.debug("Predict: %s (conf=%.2f, iou=%.2f)", image_path, conf, iou)
        if not results:
            return [], (0, 0)

        result = results[0]
        h, w = result.orig_shape
        img_size = (w, h)
        names = self.model.names
        annotations = []

        boxes = result.boxes
        if boxes is None or len(boxes.cls) == 0:
            return [], img_size

        has_kpts = result.keypoints is not None

        for i in range(len(boxes.cls)):
            cls_id = int(boxes.cls[i].item())
            confidence = round(float(boxes.conf[i].item()), 4)
            class_name = names.get(cls_id, str(cls_id))

            if project_classes and class_name not in project_classes:
                continue

            cx = round(float(boxes.xywhn[i][0].item()), 6)
            cy = round(float(boxes.xywhn[i][1].item()), 6)
            bw = round(float(boxes.xywhn[i][2].item()), 6)
            bh = round(float(boxes.xywhn[i][3].item()), 6)

            keypoints = []
            if has_kpts and result.keypoints.xyn is not None:
                kpts_xy = result.keypoints.xyn[i]
                kpts_conf = result.keypoints.conf[i] if result.keypoints.conf is not None else None
                for j in range(len(kpts_xy)):
                    kx = round(float(kpts_xy[j][0].item()), 6)
                    ky = round(float(kpts_xy[j][1].item()), 6)
                    kc = float(kpts_conf[j].item()) if kpts_conf is not None else 1.0
                    visible = 2 if kc > 0.5 else (1 if kc > 0 else 0)
                    label = kpt_labels[j] if kpt_labels and j < len(kpt_labels) else f"kp_{j}"
                    keypoints.append(Keypoint(x=kx, y=ky, visible=visible, label=label))

            if project_classes and class_name in project_classes:
                resolved_id = project_classes.index(class_name)
            else:
                resolved_id = cls_id

            annotations.append(Annotation(
                class_name=class_name,
                class_id=resolved_id,
                bbox=(cx, cy, bw, bh),
                keypoints=keypoints,
                confidence=confidence,
                confirmed=False,
                source="auto",
            ))

        logger.debug("Predict result: %d annotations", len(annotations))
        return annotations, img_size
