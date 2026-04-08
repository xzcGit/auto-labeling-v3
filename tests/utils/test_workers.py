"""Tests for QThread workers."""
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PyQt5.QtCore import QCoreApplication, QMutex


def _process_events():
    """Process pending Qt events."""
    QCoreApplication.processEvents()


class TestTrainWorker:
    def test_emits_epoch_signal(self, qapp):
        from src.utils.workers import TrainWorker
        from src.engine.trainer import TrainConfig

        config = TrainConfig(data_yaml="data.yaml", model="yolov8n.pt", task="detect", epochs=2)

        # Mock the Trainer
        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance

        # Capture on_epoch_end callback and call it during train
        def fake_train(cfg, on_epoch_end=None):
            if on_epoch_end:
                on_epoch_end({"epoch": 0, "train_loss": 1.5})
                on_epoch_end({"epoch": 1, "train_loss": 0.8})

        mock_trainer_instance.train.side_effect = fake_train
        mock_trainer_instance.get_best_metrics.return_value = {"mAP50": 0.85}
        mock_trainer_instance.cancelled = False

        worker = TrainWorker(config, trainer_cls=mock_trainer_cls)
        epochs = []
        worker.epoch_update.connect(lambda d: epochs.append(d))
        finished_data = []
        worker.finished_ok.connect(lambda d: finished_data.append(d))

        worker.run()  # call run() directly, not start()
        assert len(epochs) == 2
        assert epochs[0]["epoch"] == 0
        assert len(finished_data) == 1
        assert finished_data[0]["mAP50"] == 0.85

    def test_emits_error_on_exception(self, qapp):
        from src.utils.workers import TrainWorker
        from src.engine.trainer import TrainConfig

        config = TrainConfig(data_yaml="data.yaml", model="yolov8n.pt", task="detect")
        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance
        mock_trainer_instance.train.side_effect = RuntimeError("CUDA OOM")

        worker = TrainWorker(config, trainer_cls=mock_trainer_cls)
        errors = []
        worker.error.connect(lambda msg: errors.append(msg))

        worker.run()
        assert len(errors) == 1
        assert "CUDA OOM" in errors[0]


class TestBatchPredictWorker:
    def test_emits_progress_and_results(self, qapp):
        from src.utils.workers import BatchPredictWorker
        from src.core.annotation import Annotation

        mock_predictor = MagicMock()
        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), confirmed=False, source="auto")
        mock_predictor.predict_with_size.return_value = ([ann], (640, 480))

        image_paths = [Path(f"/imgs/img{i}.jpg") for i in range(3)]
        worker = BatchPredictWorker(
            predictor=mock_predictor,
            image_paths=image_paths,
            conf=0.5,
            iou=0.45,
        )

        progress_values = []
        worker.progress.connect(lambda cur, total: progress_values.append((cur, total)))
        results = []
        worker.image_done.connect(lambda path, anns, size: results.append((path, anns, size)))
        finished = []
        worker.finished_ok.connect(lambda: finished.append(True))

        worker.run()
        assert len(progress_values) == 3
        assert progress_values[-1] == (3, 3)
        assert len(results) == 3
        assert results[0][1][0].class_name == "cat"
        assert len(finished) == 1

    def test_cancel_stops_processing(self, qapp):
        from src.utils.workers import BatchPredictWorker

        mock_predictor = MagicMock()
        mock_predictor.predict_with_size.return_value = ([], (640, 480))

        image_paths = [Path(f"/imgs/img{i}.jpg") for i in range(10)]
        worker = BatchPredictWorker(
            predictor=mock_predictor,
            image_paths=image_paths,
            conf=0.5,
            iou=0.45,
        )

        results = []
        worker.image_done.connect(lambda path, anns, size: results.append(path))

        # Cancel after first call
        def cancel_after_first(*args, **kwargs):
            if mock_predictor.predict_with_size.call_count >= 2:
                worker.cancel()
            return ([], (640, 480))

        mock_predictor.predict_with_size.side_effect = cancel_after_first
        worker.run()
        assert len(results) < 10

    def test_emits_error_on_exception(self, qapp):
        from src.utils.workers import BatchPredictWorker

        mock_predictor = MagicMock()
        mock_predictor.predict_with_size.side_effect = RuntimeError("model error")

        worker = BatchPredictWorker(
            predictor=mock_predictor,
            image_paths=[Path("/img.jpg")],
            conf=0.5,
            iou=0.45,
        )
        errors = []
        worker.error.connect(lambda msg: errors.append(msg))

        worker.run()
        assert len(errors) == 1
        assert "model error" in errors[0]


class TestThreadSafety:
    def test_batch_worker_cancelled_is_event(self, qapp):
        from src.utils.workers import BatchPredictWorker

        mock_predictor = MagicMock()
        worker = BatchPredictWorker(
            predictor=mock_predictor,
            image_paths=[Path("/img.jpg")],
        )
        assert isinstance(worker._cancelled, threading.Event)

    def test_batch_worker_cancel_sets_event(self, qapp):
        from src.utils.workers import BatchPredictWorker

        mock_predictor = MagicMock()
        worker = BatchPredictWorker(
            predictor=mock_predictor,
            image_paths=[Path("/img.jpg")],
        )
        assert not worker._cancelled.is_set()
        worker.cancel()
        assert worker._cancelled.is_set()

    def test_train_worker_has_mutex(self, qapp):
        from src.utils.workers import TrainWorker
        from src.engine.trainer import TrainConfig

        config = TrainConfig(data_yaml="data.yaml", model="yolov8n.pt", task="detect")
        worker = TrainWorker(config)
        assert isinstance(worker._trainer_mutex, QMutex)

    def test_train_worker_cancel_before_run(self, qapp):
        """Calling cancel() before run() should not raise."""
        from src.utils.workers import TrainWorker
        from src.engine.trainer import TrainConfig

        config = TrainConfig(data_yaml="data.yaml", model="yolov8n.pt", task="detect")
        worker = TrainWorker(config)
        worker.cancel()  # Should not raise when _trainer is None

    def test_train_worker_cancel_calls_request_cancel(self, qapp):
        from src.utils.workers import TrainWorker
        from src.engine.trainer import TrainConfig

        config = TrainConfig(data_yaml="data.yaml", model="yolov8n.pt", task="detect")
        mock_trainer_cls = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.cancelled = True

        def fake_train(cfg, on_epoch_end=None):
            # Simulate cancel being called during training
            pass

        mock_trainer.train.side_effect = fake_train

        worker = TrainWorker(config, trainer_cls=mock_trainer_cls)
        worker.run()  # Sets self._trainer
        worker.cancel()
        mock_trainer.request_cancel.assert_called_once()
