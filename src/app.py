"""Main application window."""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QAction,
    QMessageBox,
)
from PyQt5.QtCore import Qt

from src.core.config import AppConfig
from src.core.project import ProjectManager
from src.core.annotation import ImageAnnotation
from src.core.label_io import load_annotation
from src.ui.label_panel import LabelPanel
from src.ui.train_panel import TrainPanel
from src.ui.model_panel import ModelPanel
from src.ui.dialogs import BatchProgressDialog
from src.engine.model_manager import ModelRegistry
from src.utils.workers import BatchPredictWorker
from src.controllers.project import ProjectController
from src.controllers.model import ModelController
from src.controllers.train import TrainController
from src.ui.icons import icon

logger = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / ".autolabel" / "config.json"


class WelcomePage(QWidget):
    """Startup welcome page with recent projects and create/open buttons."""

    def __init__(self, app_config: AppConfig, parent=None):
        super().__init__(parent)
        self._config = app_config
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 40, 60, 40)

        title = QLabel("AutoLabel V3")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #89b4fa;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("图像标注 · 模型训练 · 自动标注")
        subtitle.setStyleSheet("font-size: 14px; color: #a6adc8;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        layout.addSpacing(30)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_new = QPushButton(icon("new_project"), "新建项目")
        self.btn_new.setMinimumWidth(120)
        btn_layout.addWidget(self.btn_new)

        self.btn_open = QPushButton(icon("open_project"), "打开项目")
        self.btn_open.setMinimumWidth(120)
        btn_layout.addWidget(self.btn_open)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        layout.addSpacing(20)

        recent_label = QLabel("最近项目")
        recent_label.setStyleSheet("font-size: 14px; color: #a6adc8;")
        layout.addWidget(recent_label)

        self.recent_list = QListWidget()
        self.recent_list.setMaximumHeight(200)
        for project_path in self._config.recent_projects:
            item = QListWidgetItem(project_path)
            self.recent_list.addItem(item)
        layout.addWidget(self.recent_list)

        layout.addStretch()


class MainWindow(QMainWindow):
    """Application main window with tab-based layout.

    Business logic is delegated to controllers:
    - ProjectController: create, open, export, class management
    - ModelController: load, delete, import, single inference
    - TrainController: validate, start, stop, model registration
    """

    def __init__(self, config_path: Path | str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AutoLabel V3")

        self._config_path = Path(config_path) if config_path else CONFIG_PATH
        self._app_config = AppConfig.load(self._config_path)
        geo = self._app_config.window_geometry
        self.setGeometry(geo["x"], geo["y"], geo["width"], geo["height"])

        # Controllers
        self._project_ctrl = ProjectController(self._app_config, self._config_path, self)
        self._model_ctrl = ModelController(self)
        self._train_ctrl = TrainController(self)

        # State
        self._project: ProjectManager | None = None
        self._label_panel: LabelPanel | None = None
        self._train_panel: TrainPanel | None = None
        self._model_panel: ModelPanel | None = None
        self._model_registry: ModelRegistry | None = None
        self._batch_worker: BatchPredictWorker | None = None
        self._batch_dialog: BatchProgressDialog | None = None

        # Central widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Welcome page
        self._welcome = WelcomePage(self._app_config)
        self._welcome.btn_new.clicked.connect(self._on_new_project)
        self._welcome.btn_open.clicked.connect(self._on_open_project)
        self._welcome.recent_list.itemDoubleClicked.connect(self._on_recent_clicked)
        self.tab_widget.addTab(self._welcome, icon("welcome"), "欢迎")

        self._setup_menus()

        self._status_label = QLabel("就绪")
        self.statusBar().addPermanentWidget(self._status_label)

        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index: int) -> None:
        """Auto-rescan images when switching to the Label tab."""
        if self._label_panel is None:
            return
        if self.tab_widget.widget(index) is self._label_panel:
            n = self._label_panel.rescan_images()
            if n > 0:
                self._status_label.setText(f"发现 {n} 张新图片")

    def _setup_menus(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("文件")

        new_action = QAction(icon("new_project"), "新建项目", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._on_new_project)
        file_menu.addAction(new_action)

        open_action = QAction(icon("open_project"), "打开项目", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        export_action = QAction(icon("export"), "导出...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._on_export)
        file_menu.addAction(export_action)

        import_action = QAction(icon("import"), "导入标注...", self)
        import_action.setShortcut("Ctrl+I")
        import_action.triggered.connect(self._on_import)
        file_menu.addAction(import_action)

        file_menu.addSeparator()

        exit_action = QAction(icon("exit"), "退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = mb.addMenu("编辑")

        classes_action = QAction(icon("classes"), "类别管理...", self)
        classes_action.triggered.connect(self._on_class_manager)
        edit_menu.addAction(classes_action)

    # ── Project ──────────────────────────────────────────────

    def open_project(self, project_manager: ProjectManager) -> None:
        """Open a project and switch to labeling workspace."""
        self._project = project_manager
        self.setWindowTitle(f"AutoLabel V3 — {project_manager.config.name}")

        self._model_registry = ModelRegistry(project_manager.project_dir / "models")
        self._model_registry.load()
        self._model_ctrl.set_context(project_manager, self._model_registry)

        if self._label_panel is None:
            self._label_panel = LabelPanel(config_path=self._config_path)
            self._label_panel.auto_label_single_requested.connect(self._on_auto_label_single)
            self._label_panel.auto_label_batch_requested.connect(self._on_auto_label_batch)
            self._label_panel.status_changed.connect(self._status_label.setText)
            self.tab_widget.addTab(self._label_panel, icon("label_tab"), "标注")
        self._label_panel.set_project(project_manager)

        if self._train_panel is None:
            self._train_panel = TrainPanel()
            self._train_panel._btn_start.clicked.connect(self._on_start_training)
            self._train_panel.stop_requested.connect(self._on_stop_training)
            self._train_panel.preview_augmentation_requested.connect(self._on_preview_augmentation)
            self.tab_widget.addTab(self._train_panel, icon("train_tab"), "训练")

        if self._model_panel is None:
            self._model_panel = ModelPanel()
            self._model_panel.model_load_requested.connect(self._on_model_load)
            self._model_panel.model_delete_requested.connect(self._on_model_delete)
            self._model_panel.model_rename_requested.connect(self._on_model_rename)
            self._model_panel.model_import_requested.connect(self._on_model_import)
            self.tab_widget.addTab(self._model_panel, icon("model_tab"), "模型")
        self._model_panel.set_models(self._model_registry.list_models())
        self._train_panel.set_registered_models(self._model_registry.list_models())

        self.tab_widget.setCurrentWidget(self._label_panel)
        self._status_label.setText(
            f"项目: {project_manager.config.name} | "
            f"图片: {len(project_manager.list_images())} | "
            f"类别: {len(project_manager.config.classes)}"
        )

    def _on_new_project(self) -> None:
        pm = self._project_ctrl.create_project()
        if pm:
            self.open_project(pm)

    def _on_open_project(self) -> None:
        pm = self._project_ctrl.open_project_dialog()
        if pm:
            self.open_project(pm)

    def _on_recent_clicked(self, item: QListWidgetItem) -> None:
        pm = self._project_ctrl.open_recent(item)
        if pm:
            self.open_project(pm)
        else:
            row = self._welcome.recent_list.row(item)
            if row >= 0:
                self._welcome.recent_list.takeItem(row)

    def _on_export(self) -> None:
        if not self._project:
            return
        if self._label_panel:
            self._label_panel.save_and_cleanup()
        try:
            self._project_ctrl.export(self._project)
            self._status_label.setText("导出完成")
        except (OSError, ValueError, KeyError):
            pass  # Error already shown by controller

    def _on_import(self) -> None:
        if not self._project:
            return
        if self._label_panel:
            self._label_panel.save_and_cleanup()
        count = self._project_ctrl.import_annotations(self._project)
        if count is not None and count > 0:
            # Refresh label panel to show imported annotations
            if self._label_panel:
                self._label_panel.set_project(self._project)
            self._status_label.setText(f"导入完成: {count} 个图片")
        elif count == 0:
            self._status_label.setText("导入完成: 无匹配图片")

    def _on_class_manager(self) -> None:
        if not self._project:
            return
        if self._project_ctrl.manage_classes(self._project):
            if self._label_panel:
                self._label_panel.set_project(self._project)

    # ── Model ────────────────────────────────────────────────

    def _on_model_load(self, model_id: str) -> None:
        if self._model_ctrl.load_model(model_id):
            model_info = self._model_ctrl.registry.get(model_id)
            if self._model_panel and model_info:
                self._model_panel.set_current_model_name(model_info.name)
            self._status_label.setText(f"已加载模型: {model_info.name}" if model_info else "")

    def _on_model_delete(self, model_id: str) -> None:
        if self._model_ctrl.delete_model(model_id):
            self._refresh_model_lists()

    def _on_model_rename(self, model_id: str) -> None:
        if self._model_ctrl.rename_model(model_id):
            self._refresh_model_lists()

    def _on_model_import(self) -> None:
        model_info = self._model_ctrl.import_model()
        if model_info:
            self._refresh_model_lists()
            self._status_label.setText(f"已导入模型: {model_info.name}")

    def _refresh_model_lists(self) -> None:
        if self._model_registry:
            models = self._model_registry.list_models()
            if self._model_panel:
                self._model_panel.set_models(models)
            if self._train_panel:
                self._train_panel.set_registered_models(models)

    # ── Auto-label ───────────────────────────────────────────

    def _on_auto_label_single(self) -> None:
        if not self._label_panel or not self._project:
            return
        img_path = self._label_panel.get_current_image_path()
        if not img_path:
            return
        conf = self._model_panel.get_conf_threshold() if self._model_panel else 0.5
        iou = self._model_panel.get_iou_threshold() if self._model_panel else 0.45
        overlap_iou = self._model_panel.get_overlap_iou_threshold() if self._model_panel else 0.5
        annotations = self._model_ctrl.predict_single(
            img_path, self._project.config.classes, conf=conf, iou=iou,
        )
        if annotations:
            self._label_panel.add_auto_annotations(annotations, overlap_iou=overlap_iou)
            self._status_label.setText(f"自动标注: 检测到 {len(annotations)} 个目标")
        else:
            self._status_label.setText("自动标注: 未检测到目标")

    def _on_auto_label_batch(self) -> None:
        if not self._model_ctrl.predictor:
            QMessageBox.information(self, "提示", "请先在模型面板中加载一个模型")
            return
        if not self._label_panel or not self._project:
            return
        self._label_panel.save_and_cleanup()

        all_images = self._project.list_images()
        unlabeled = self._label_panel.get_unlabeled_image_paths()

        items = ["仅未标注图片", "全部图片"]
        from PyQt5.QtWidgets import QInputDialog
        choice, ok = QInputDialog.getItem(
            self, "批量自动标注", f"选择范围 (未标注: {len(unlabeled)} / 全部: {len(all_images)})",
            items, 0, False,
        )
        if not ok:
            return

        target_images = unlabeled if choice == items[0] else all_images
        if not target_images:
            QMessageBox.information(self, "提示", "没有需要处理的图片")
            return

        conf = self._model_panel.get_conf_threshold() if self._model_panel else 0.5
        iou = self._model_panel.get_iou_threshold() if self._model_panel else 0.45

        self._batch_worker = BatchPredictWorker(
            predictor=self._model_ctrl.predictor,
            image_paths=target_images,
            conf=conf, iou=iou,
            project_classes=self._project.config.classes,
        )
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.image_done.connect(self._on_batch_image_done)
        self._batch_worker.finished_ok.connect(self._on_batch_finished)
        self._batch_worker.error.connect(self._on_batch_error)
        self._batch_worker.start()

        self._batch_dialog = BatchProgressDialog("批量自动标注", len(target_images), self)
        self._batch_dialog.show()
        self._status_label.setText(f"批量标注进行中: 0/{len(target_images)}")

    def _on_batch_progress(self, current: int, total: int) -> None:
        self._status_label.setText(f"批量标注进行中: {current}/{total}")
        if self._batch_dialog:
            self._batch_dialog.update_progress(current, total)
            if self._batch_dialog.is_cancelled and self._batch_worker:
                self._batch_worker.cancel()

    def _on_batch_image_done(self, path_str: str, annotations, img_size) -> None:
        if not self._project:
            return
        img_path = Path(path_str)
        label_path = self._project.label_path_for(img_path)
        ia = load_annotation(label_path)
        if ia is None:
            ia = ImageAnnotation(image_path=img_path.name, image_size=img_size)
        # Filter out predictions that overlap with existing confirmed annotations
        from src.core.annotation import find_conflicts
        overlap_iou = self._model_panel.get_overlap_iou_threshold() if self._model_panel else 0.5
        _, clean_preds = find_conflicts(ia.annotations, annotations, overlap_iou)
        for ann in clean_preds:
            ia.annotations.append(ann)
        from src.core.label_io import save_annotation
        save_annotation(ia, label_path)
        if self._label_panel:
            self._label_panel._file_list.set_status(img_path, ia.status)

    def _on_batch_finished(self) -> None:
        self._status_label.setText("批量自动标注完成")
        if self._batch_dialog:
            self._batch_dialog.close()
            self._batch_dialog = None
        if self._label_panel and self._label_panel.get_current_image_path():
            self._label_panel._on_image_selected(self._label_panel.get_current_image_path())

    def _on_batch_error(self, msg: str) -> None:
        self._status_label.setText("批量标注失败")
        if self._batch_dialog:
            self._batch_dialog.close()
            self._batch_dialog = None
        QMessageBox.warning(self, "批量标注失败", msg)

    # ── Training ─────────────────────────────────────────────

    def _on_preview_augmentation(self, params: dict) -> None:
        """Show augmentation preview dialog."""
        if not self._project or not self._label_panel:
            return
        img_path = self._label_panel.get_current_image_path()
        if not img_path:
            # Use first image in project
            images = self._project.list_images()
            if not images:
                QMessageBox.information(self, "提示", "没有可用图片")
                return
            img_path = images[0]
        from src.ui.augmentation_preview import AugmentationPreviewDialog
        dlg = AugmentationPreviewDialog(img_path, params, self)
        dlg.exec_()

    def _on_start_training(self) -> None:
        if not self._project or not self._train_panel:
            return
        try:
            if self._label_panel:
                self._label_panel.save_and_cleanup()

            task = self._train_panel._task_combo.currentText()
            val_ratio = self._train_panel.get_val_ratio()
            kpt_shape = None
            if task == "pose":
                kpt_shape = [self._train_panel._kpt_num_spin.value(), self._train_panel._kpt_dim_spin.value()]
            data_yaml = self._train_ctrl.validate_and_prepare(self._project, task, val_ratio, kpt_shape=kpt_shape)
            if data_yaml is None:
                self._train_panel._btn_start.setEnabled(True)
                self._train_panel._btn_stop.setEnabled(False)
                return

            config = self._train_panel.get_train_config(data_yaml=data_yaml)
            worker = self._train_ctrl.start(config, self._project, task)
            worker.epoch_update.connect(self._train_panel.update_epoch)
            worker.finished_ok.connect(self._on_training_finished)
            worker.cancelled.connect(self._on_training_cancelled)
            worker.error.connect(self._train_panel.on_training_error)

            self._train_panel.append_log(f"开始训练: {task} | {config.model} | {config.epochs} epochs")
        except (OSError, ValueError, RuntimeError) as e:
            logger.error("Failed to start training: %s", e, exc_info=True)
            self._train_panel.on_training_error(str(e))

    def _on_stop_training(self) -> None:
        self._train_ctrl.stop()
        if self._train_panel:
            self._train_panel.append_log("正在停止训练...")
            self._train_panel._btn_stop.setEnabled(False)

    def _on_training_cancelled(self) -> None:
        if self._train_panel:
            self._train_panel._btn_start.setEnabled(True)
            self._train_panel._btn_stop.setEnabled(False)
            self._train_panel._epoch_progress.setVisible(False)
            self._train_panel.append_log("--- 训练已停止 ---")

    def _on_training_finished(self, metrics: dict) -> None:
        if self._train_panel:
            self._train_panel.on_training_finished(metrics)
        if self._model_registry and self._project:
            task = self._train_panel._task_combo.currentText() if self._train_panel else "detect"
            base_model = self._train_panel._model_combo.currentText() if self._train_panel else "yolov8n.pt"
            epochs = self._train_panel._epochs_spin.value() if self._train_panel else 0
            model_info = self._train_ctrl.register_model(
                self._model_registry, self._project, task, base_model, epochs, metrics,
            )
            self._refresh_model_lists()
            self._on_model_load(model_info.id)
            self._status_label.setText(f"训练完成，已自动加载模型: {model_info.name}")

    # ── Lifecycle ────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._label_panel:
            self._label_panel.save_and_cleanup()
        geo = self.geometry()
        self._app_config.window_geometry = {
            "x": geo.x(), "y": geo.y(),
            "width": geo.width(), "height": geo.height(),
        }
        self._app_config.save(self._config_path)
        super().closeEvent(event)
