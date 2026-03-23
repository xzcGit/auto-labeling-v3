"""Tests for training engine."""
from unittest.mock import MagicMock

from src.engine.trainer import TrainConfig, Trainer


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig(
            data_yaml="/path/data.yaml",
            model="yolov8n.pt",
            task="detect",
        )
        assert cfg.epochs == 100
        assert cfg.batch == 16
        assert cfg.imgsz == 640
        assert cfg.device == ""
        assert cfg.optimizer == "auto"

    def test_to_train_args(self):
        cfg = TrainConfig(
            data_yaml="/path/data.yaml",
            model="yolov8n.pt",
            task="detect",
            epochs=50,
            batch=8,
            project="/out",
            name="run1",
        )
        args = cfg.to_train_args()
        assert args["data"] == "/path/data.yaml"
        assert args["epochs"] == 50
        assert args["batch"] == 8
        assert args["project"] == "/out"
        assert args["name"] == "run1"

    def test_to_train_args_excludes_empty(self):
        cfg = TrainConfig(
            data_yaml="/path/data.yaml",
            model="yolov8n.pt",
            task="detect",
        )
        args = cfg.to_train_args()
        assert "device" not in args


class TestTrainer:
    def test_train_calls_yolo(self):
        mock_yolo_cls = MagicMock()
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        mock_model.train.return_value = MagicMock()

        cfg = TrainConfig(
            data_yaml="/data.yaml",
            model="yolov8n.pt",
            task="detect",
            epochs=10,
            project="/out",
            name="test",
        )

        trainer = Trainer(yolo_cls=mock_yolo_cls)
        trainer.train(cfg)

        mock_yolo_cls.assert_called_once_with("yolov8n.pt")
        mock_model.train.assert_called_once()
        train_kwargs = mock_model.train.call_args[1]
        assert train_kwargs["data"] == "/data.yaml"
        assert train_kwargs["epochs"] == 10

    def test_train_with_callback(self):
        mock_yolo_cls = MagicMock()
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        mock_model.train.return_value = MagicMock()

        def on_epoch(metrics: dict):
            pass

        cfg = TrainConfig(
            data_yaml="/data.yaml",
            model="yolov8n.pt",
            task="detect",
            epochs=5,
        )

        trainer = Trainer(yolo_cls=mock_yolo_cls)
        trainer.train(cfg, on_epoch_end=on_epoch)

        mock_model.add_callback.assert_called()

    def test_train_resume(self):
        mock_yolo_cls = MagicMock()
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        mock_model.train.return_value = MagicMock()

        cfg = TrainConfig(
            data_yaml="/data.yaml",
            model="/out/test/weights/last.pt",
            task="detect",
            resume=True,
        )

        trainer = Trainer(yolo_cls=mock_yolo_cls)
        trainer.train(cfg)

        train_kwargs = mock_model.train.call_args[1]
        assert train_kwargs["resume"] is True

    def test_get_best_metrics(self):
        mock_yolo_cls = MagicMock()
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        mock_model.trainer = MagicMock()
        mock_model.trainer.best_fitness = 0.85
        mock_model.trainer.metrics = {
            "metrics/mAP50(B)": 0.89,
            "metrics/mAP50-95(B)": 0.67,
        }
        mock_model.train.return_value = MagicMock()

        cfg = TrainConfig(
            data_yaml="/data.yaml",
            model="yolov8n.pt",
            task="detect",
        )

        trainer = Trainer(yolo_cls=mock_yolo_cls)
        trainer.train(cfg)
        metrics = trainer.get_best_metrics()

        assert metrics["mAP50"] == 0.89
        assert metrics["mAP50-95"] == 0.67
