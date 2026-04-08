"""Training engine — wraps ultralytics YOLO.train."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration."""

    data_yaml: str
    model: str
    task: str  # "detect", "classify", "pose"

    # Basic hyperparameters
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    device: str = ""
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1

    # Data augmentation
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0

    # Pose-specific
    pose: float = 12.0
    kobj: float = 1.0
    kpt_shape: list[int] | None = None  # [num_keypoints, dim] e.g. [17, 3]

    # Output
    project: str = ""
    name: str = ""
    resume: bool = False

    def to_train_args(self) -> dict:
        """Convert to kwargs dict for YOLO.train()."""
        args = {
            "data": self.data_yaml,
            "epochs": self.epochs,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "warmup_momentum": self.warmup_momentum,
            "warmup_bias_lr": self.warmup_bias_lr,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "perspective": self.perspective,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "copy_paste": self.copy_paste,
        }
        if self.device:
            args["device"] = self.device
        if self.project:
            args["project"] = self.project
        if self.name:
            args["name"] = self.name
        if self.resume:
            args["resume"] = True
        if self.task == "pose":
            args["pose"] = self.pose
            args["kobj"] = self.kobj
            if self.kpt_shape:
                args["kpt_shape"] = self.kpt_shape
        return args


# ── Training presets ───────────────────────────────────────────

TRAIN_PRESETS: dict[str, dict] = {
    "默认": {},  # Use TrainConfig defaults
    "快速验证": {
        "epochs": 10,
        "batch": 32,
        "imgsz": 320,
        "lr0": 0.01,
        "mosaic": 0.0,
        "mixup": 0.0,
        "scale": 0.0,
        "fliplr": 0.0,
    },
    "标准训练": {
        "epochs": 100,
        "batch": 16,
        "imgsz": 640,
        "lr0": 0.01,
        "mosaic": 1.0,
        "fliplr": 0.5,
        "scale": 0.5,
    },
    "高精度": {
        "epochs": 300,
        "batch": 8,
        "imgsz": 640,
        "lr0": 0.001,
        "lrf": 0.001,
        "mosaic": 1.0,
        "mixup": 0.15,
        "copy_paste": 0.1,
        "fliplr": 0.5,
        "scale": 0.9,
        "translate": 0.2,
        "degrees": 10.0,
    },
}


class Trainer:
    """Wraps ultralytics YOLO training with callback support."""

    def __init__(self, yolo_cls=None):
        if yolo_cls is None:
            from ultralytics import YOLO
            yolo_cls = YOLO
        self._yolo_cls = yolo_cls
        self._model = None
        self._cancel_requested = False

    def request_cancel(self) -> None:
        """Request graceful cancellation of training."""
        self._cancel_requested = True
        logger.info("Training cancel requested")

    @property
    def cancelled(self) -> bool:
        return self._cancel_requested

    def train(
        self,
        config: TrainConfig,
        on_epoch_end: Callable[[dict], None] | None = None,
    ) -> None:
        """Start training."""
        logger.info("Starting training: model=%s, epochs=%d", config.model, config.epochs)
        self._model = self._yolo_cls(config.model)

        def _epoch_callback(trainer_obj):
            if self._cancel_requested:
                # Tell YOLO training is done — graceful stop
                trainer_obj.epoch = trainer_obj.epochs
                return
            if on_epoch_end:
                metrics = {}
                if hasattr(trainer_obj, "metrics") and trainer_obj.metrics:
                    metrics = dict(trainer_obj.metrics)
                if hasattr(trainer_obj, "loss") and trainer_obj.loss is not None:
                    metrics["train_loss"] = float(trainer_obj.loss.mean().item())
                metrics["epoch"] = trainer_obj.epoch
                on_epoch_end(metrics)

        self._model.add_callback("on_fit_epoch_end", _epoch_callback)

        train_args = config.to_train_args()
        self._model.train(**train_args)

    def get_best_metrics(self) -> dict[str, float]:
        """Extract best metrics after training completes."""
        if self._model is None or not hasattr(self._model, "trainer"):
            return {}

        raw = getattr(self._model.trainer, "metrics", {})
        if not raw:
            return {}

        result = {}
        for key, value in raw.items():
            clean_key = key.replace("metrics/", "").replace("(B)", "").replace("(P)", "")
            result[clean_key] = round(float(value), 4)
        return result
