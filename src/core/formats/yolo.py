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


def _find_data_yaml(labels_dir: Path) -> Path | None:
    """Search for data.yaml in the given dir, parent dir, or sibling paths."""
    for candidate in [
        labels_dir / "data.yaml",
        labels_dir.parent / "data.yaml",
    ]:
        if candidate.exists():
            return candidate
    return None


def _detect_yolo_format(labels_dir: Path) -> tuple[str, int]:
    """Detect whether YOLO labels are detection or pose by inspecting the first file.

    Returns ("detection", 0) or ("pose", num_keypoints).
    For pose, assumes kpt_dim=3 (x, y, visibility).
    """
    for txt_path in sorted(labels_dir.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        first_line = text.split("\n")[0].strip()
        parts = first_line.split()
        n = len(parts)
        if n <= 5:
            return "detection", 0
        # Assume kpt_dim=3: extra columns = num_keypoints * 3
        extra = n - 5
        if extra % 3 == 0:
            return "pose", extra // 3
        elif extra % 2 == 0:
            return "pose", extra // 2
        # Fallback: treat as detection
        return "detection", 0
    return "detection", 0


def import_yolo_auto(
    labels_dir: Path | str,
    classes: list[str] | None = None,
    data_yaml: Path | str | None = None,
    kpt_labels: list[str] | None = None,
    kpt_dim: int = 3,
) -> list[ImageAnnotation]:
    """Auto-detect YOLO format (detection or pose) and import accordingly.

    Searches for data.yaml in the directory and its parent.
    Falls back to numeric class names if no classes are available.
    """
    labels_dir = Path(labels_dir)

    # Resolve classes from data.yaml if not provided
    if classes is None and data_yaml is None:
        data_yaml = _find_data_yaml(labels_dir)

    if classes is None and data_yaml is not None:
        data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
        classes = data.get("names")

    # Detect format
    fmt, num_kpts = _detect_yolo_format(labels_dir)

    if fmt == "pose" and num_kpts > 0:
        if classes is None:
            # Infer max class id from files to generate fallback names
            classes = _infer_classes_from_files(labels_dir)
        if kpt_labels is None:
            kpt_labels = [f"kp_{i}" for i in range(num_kpts)]
        return import_yolo_pose(labels_dir, classes, kpt_labels, kpt_dim)
    else:
        if classes is None:
            classes = _infer_classes_from_files(labels_dir)
        return import_yolo_detection(labels_dir, classes)


def _infer_classes_from_files(labels_dir: Path) -> list[str]:
    """Scan all txt files to find max class_id and generate numeric class names."""
    max_id = -1
    for txt_path in labels_dir.glob("*.txt"):
        for line in txt_path.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            cid = int(line.strip().split()[0])
            if cid > max_id:
                max_id = cid
    if max_id < 0:
        return []
    return [str(i) for i in range(max_id + 1)]
