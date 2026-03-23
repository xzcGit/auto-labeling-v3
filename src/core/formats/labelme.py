"""labelme JSON format import/export."""
from __future__ import annotations

import json
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation, Keypoint


def export_labelme(
    image_annotations: list[ImageAnnotation],
    output_dir: Path | str,
    only_confirmed: bool = False,
) -> None:
    """Export annotations to labelme JSON format (one JSON per image)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ia in image_annotations:
        w_img, h_img = ia.image_size
        shapes = []

        for ann in ia.annotations:
            if only_confirmed and not ann.confirmed:
                continue

            # Export bbox as rectangle
            if ann.bbox is not None:
                cx, cy, bw, bh = ann.bbox
                x1 = (cx - bw / 2) * w_img
                y1 = (cy - bh / 2) * h_img
                x2 = (cx + bw / 2) * w_img
                y2 = (cy + bh / 2) * h_img
                shapes.append({
                    "label": ann.class_name,
                    "shape_type": "rectangle",
                    "points": [[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y2, 2)]],
                    "flags": {},
                })

            # Export keypoints as individual points
            for kp in ann.keypoints:
                shapes.append({
                    "label": kp.label,
                    "shape_type": "point",
                    "points": [[round(kp.x * w_img, 2), round(kp.y * h_img, 2)]],
                    "flags": {},
                })

        labelme_data = {
            "version": "5.0.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": ia.image_path,
            "imageData": None,
            "imageWidth": w_img,
            "imageHeight": h_img,
        }

        stem = Path(ia.image_path).stem
        out_path = output_dir / f"{stem}.json"
        out_path.write_text(json.dumps(labelme_data, indent=2, ensure_ascii=False), encoding="utf-8")


def import_labelme(input_dir: Path | str) -> list[ImageAnnotation]:
    """Import annotations from labelme JSON files in a directory."""
    input_dir = Path(input_dir)
    results = []

    for json_path in sorted(input_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        if "shapes" not in data:
            continue

        w_img = data.get("imageWidth", 0)
        h_img = data.get("imageHeight", 0)
        image_path = data.get("imagePath", json_path.stem + ".jpg")

        annotations = []
        for shape in data["shapes"]:
            shape_type = shape.get("shape_type", "")
            label = shape.get("label", "unknown")
            points = shape.get("points", [])

            if shape_type == "rectangle" and len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                cx = ((x1 + x2) / 2) / w_img if w_img else 0
                cy = ((y1 + y2) / 2) / h_img if h_img else 0
                bw = abs(x2 - x1) / w_img if w_img else 0
                bh = abs(y2 - y1) / h_img if h_img else 0
                annotations.append(Annotation(
                    class_name=label,
                    class_id=0,
                    bbox=(cx, cy, bw, bh),
                    confirmed=True,
                    source="manual",
                ))
            elif shape_type == "point" and len(points) == 1:
                px, py = points[0]
                kp = Keypoint(
                    x=px / w_img if w_img else 0,
                    y=py / h_img if h_img else 0,
                    visible=2,
                    label=label,
                )
                annotations.append(Annotation(
                    class_name=label,
                    class_id=0,
                    bbox=None,
                    keypoints=[kp],
                    confirmed=True,
                    source="manual",
                ))

        results.append(ImageAnnotation(
            image_path=image_path,
            image_size=(w_img, h_img),
            annotations=annotations,
        ))

    return results
