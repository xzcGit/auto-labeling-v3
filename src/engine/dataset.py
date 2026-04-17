"""Dataset preparation for YOLO training (train/val split, symlinks, data.yaml)."""
from __future__ import annotations

import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

from src.core.annotation import ImageAnnotation
from src.core.label_io import load_annotation
from src.core.project import ProjectManager

logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Prepares a YOLO-compatible dataset from a project."""

    def __init__(self, project_manager: ProjectManager):
        self.pm = project_manager

    def prepare(
        self,
        output_dir: Path | str,
        task: str = "detect",
        val_ratio: float = 0.2,
        seed: int = 42,
        kpt_shape: list[int] | None = None,
    ) -> Path:
        """Prepare dataset and return path to data.yaml."""
        output_dir = Path(output_dir)

        # Clean previous dataset to avoid stale symlinks / ultralytics .cache files
        if output_dir.exists():
            shutil.rmtree(output_dir)

        classes = self.pm.config.classes

        # Collect labeled images with confirmed annotations
        labeled: list[tuple[Path, ImageAnnotation]] = []
        for img_path in self.pm.list_images():
            label_path = self.pm.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia is None:
                continue
            confirmed = [a for a in ia.annotations if a.confirmed]
            if not confirmed:
                continue
            ia.annotations = confirmed
            labeled.append((img_path, ia))

        if not labeled:
            raise ValueError("没有找到已确认标注的图片，无法准备数据集")

        # Stratified split by primary class
        train_set, val_set = self._stratified_split(labeled, val_ratio, seed)

        if not train_set:
            raise ValueError("训练集为空，请减小验证集比例或增加标注数据")

        logger.info(
            "Dataset prepared: %d train, %d val (task=%s)",
            len(train_set), len(val_set), task,
        )

        if task == "classify":
            self._export_classify(output_dir, train_set, val_set)
        else:
            self._export_detection_or_pose(output_dir, train_set, val_set, classes, task)

        # Generate data.yaml
        data_yaml_path = output_dir / "data.yaml"
        data = self._build_data_yaml(output_dir, classes, task, kpt_shape, has_val=bool(val_set))
        data_yaml_path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
        return data_yaml_path

    def _stratified_split(
        self,
        items: list[tuple[Path, ImageAnnotation]],
        val_ratio: float,
        seed: int,
    ) -> tuple[list[tuple[Path, ImageAnnotation]], list[tuple[Path, ImageAnnotation]]]:
        """Split items into train/val using stratified sampling by primary class."""
        if val_ratio <= 0:
            return items, []
        if val_ratio >= 1:
            return [], items

        by_class: dict[str, list[tuple[Path, ImageAnnotation]]] = defaultdict(list)
        for item in items:
            primary_class = item[1].annotations[0].class_name
            by_class[primary_class].append(item)

        rng = random.Random(seed)
        train, val = [], []
        for cls_items in by_class.values():
            rng.shuffle(cls_items)
            n_val = max(1, round(len(cls_items) * val_ratio))
            if n_val >= len(cls_items):
                n_val = max(0, len(cls_items) - 1)
            val.extend(cls_items[:n_val])
            train.extend(cls_items[n_val:])

        return train, val

    def _export_detection_or_pose(
        self,
        output_dir: Path,
        train_set: list[tuple[Path, ImageAnnotation]],
        val_set: list[tuple[Path, ImageAnnotation]],
        classes: list[str],
        task: str,
    ) -> None:
        """Export to YOLO detection/pose directory structure with symlinks."""
        for split_name, split_data in [("train", train_set), ("val", val_set)]:
            if not split_data:
                continue
            img_dir = output_dir / split_name / "images"
            lbl_dir = output_dir / split_name / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_path, ia in split_data:
                link = img_dir / img_path.name
                if not link.exists():
                    link.symlink_to(img_path.resolve())

                lines = []
                for ann in ia.annotations:
                    if ann.bbox is None:
                        continue
                    cid = classes.index(ann.class_name) if ann.class_name in classes else ann.class_id
                    cx, cy, w, h = ann.bbox
                    parts = [f"{cid}", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
                    if task == "pose" and ann.keypoints:
                        for kp in ann.keypoints:
                            parts.extend([f"{kp.x:.6f}", f"{kp.y:.6f}", f"{kp.visible}"])
                    lines.append(" ".join(parts))
                (lbl_dir / (img_path.stem + ".txt")).write_text(
                    "\n".join(lines) + "\n" if lines else "", encoding="utf-8"
                )

    def _export_classify(
        self,
        output_dir: Path,
        train_set: list[tuple[Path, ImageAnnotation]],
        val_set: list[tuple[Path, ImageAnnotation]],
    ) -> None:
        """Export to YOLO classification directory structure."""
        for split_name, split_data in [("train", train_set), ("val", val_set)]:
            if not split_data:
                continue
            for img_path, ia in split_data:
                if ia.image_tags:
                    cls_name = ia.image_tags[0]
                else:
                    cls_name = ia.annotations[0].class_name
                cls_dir = output_dir / split_name / cls_name
                cls_dir.mkdir(parents=True, exist_ok=True)
                link = cls_dir / img_path.name
                if not link.exists():
                    link.symlink_to(img_path.resolve())

    def _build_data_yaml(
        self,
        output_dir: Path,
        classes: list[str],
        task: str,
        kpt_shape: list[int] | None,
        has_val: bool = True,
    ) -> dict:
        """Build data.yaml content dict."""
        if task == "classify":
            data = {
                "path": str(output_dir.resolve()),
                "train": "train",
            }
            if has_val:
                data["val"] = "val"
            return data
        data = {
            "path": str(output_dir.resolve()),
            "train": "train/images",
            "nc": len(classes),
            "names": classes,
        }
        if has_val:
            data["val"] = "val/images"
        if task == "pose" and kpt_shape:
            data["kpt_shape"] = kpt_shape
        return data
