"""YOLO format import/export (detection + pose)."""
from __future__ import annotations

from pathlib import Path

import yaml

from src.core.annotation import Annotation, ImageAnnotation, Keypoint


def export_yolo_detection(
    image_annotations: list[ImageAnnotation],
    output_dir: Path | str,
    classes: list[str],
    only_confirmed: bool = False,
) -> None:
    """Export annotations to YOLO detection format (txt + data.yaml)."""
    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    for ia in image_annotations:
        stem = Path(ia.image_path).stem
        lines = []
        for ann in ia.annotations:
            if only_confirmed and not ann.confirmed:
                continue
            if ann.bbox is None:
                continue
            cid = classes.index(ann.class_name) if ann.class_name in classes else ann.class_id
            cx, cy, w, h = ann.bbox
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    # data.yaml
    data = {
        "nc": len(classes),
        "names": classes,
    }
    (output_dir / "data.yaml").write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")


def import_yolo_detection(
    labels_dir: Path | str,
    classes: list[str] | None = None,
    data_yaml: Path | str | None = None,
) -> list[ImageAnnotation]:
    """Import YOLO detection format. Provide classes or data_yaml."""
    labels_dir = Path(labels_dir)

    if classes is None and data_yaml:
        data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
        classes = data["names"]

    if classes is None:
        raise ValueError("Must provide classes or data_yaml")

    results = []
    for txt_path in sorted(labels_dir.glob("*.txt")):
        annotations = []
        for line in txt_path.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split()
            cid = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            annotations.append(Annotation(
                class_name=classes[cid] if cid < len(classes) else str(cid),
                class_id=cid,
                bbox=(cx, cy, w, h),
                confirmed=True,
                source="manual",
            ))
        results.append(ImageAnnotation(
            image_path=txt_path.stem,
            image_size=(0, 0),  # unknown without actual image
            annotations=annotations,
        ))
    return results


def export_yolo_pose(
    image_annotations: list[ImageAnnotation],
    output_dir: Path | str,
    classes: list[str],
    kpt_dim: int = 3,
    only_confirmed: bool = False,
) -> None:
    """Export annotations to YOLO pose format."""
    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    for ia in image_annotations:
        stem = Path(ia.image_path).stem
        lines = []
        for ann in ia.annotations:
            if only_confirmed and not ann.confirmed:
                continue
            if ann.bbox is None:
                continue
            cid = classes.index(ann.class_name) if ann.class_name in classes else ann.class_id
            cx, cy, w, h = ann.bbox
            parts = [f"{cid}", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
            for kp in ann.keypoints:
                parts.append(f"{kp.x:.6f}")
                parts.append(f"{kp.y:.6f}")
                if kpt_dim == 3:
                    parts.append(f"{kp.visible}")
            lines.append(" ".join(parts))
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def import_yolo_pose(
    labels_dir: Path | str,
    classes: list[str],
    kpt_labels: list[str],
    kpt_dim: int = 3,
) -> list[ImageAnnotation]:
    """Import YOLO pose format."""
    labels_dir = Path(labels_dir)
    num_kpts = len(kpt_labels)

    results = []
    for txt_path in sorted(labels_dir.glob("*.txt")):
        annotations = []
        for line in txt_path.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split()
            cid = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            keypoints = []
            kp_start = 5
            for i in range(num_kpts):
                offset = kp_start + i * kpt_dim
                kx = float(parts[offset])
                ky = float(parts[offset + 1])
                vis = int(float(parts[offset + 2])) if kpt_dim == 3 else 2
                keypoints.append(Keypoint(x=kx, y=ky, visible=vis, label=kpt_labels[i]))
            annotations.append(Annotation(
                class_name=classes[cid] if cid < len(classes) else str(cid),
                class_id=cid,
                bbox=(cx, cy, w, h),
                keypoints=keypoints,
                confirmed=True,
                source="manual",
            ))
        results.append(ImageAnnotation(
            image_path=txt_path.stem,
            image_size=(0, 0),
            annotations=annotations,
        ))
    return results
