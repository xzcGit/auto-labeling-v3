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
    QFileDialog,
    QAction,
    QMessageBox,
)
from PyQt5.QtCore import Qt

from src.core.config import AppConfig
from src.core.project import ProjectManager
from src.ui.label_panel import LabelPanel
from src.ui.train_panel import TrainPanel
from src.ui.model_panel import ModelPanel
from src.ui.dialogs import NewProjectDialog, ExportDialog, ClassManagerDialog
from src.engine.model_manager import ModelRegistry
from src.engine.dataset import DatasetPreparer
from src.utils.workers import TrainWorker

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

        # Title
        title = QLabel("AutoLabel V3")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #89b4fa;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("图像标注 · 模型训练 · 自动标注")
        subtitle.setStyleSheet("font-size: 14px; color: #a6adc8;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        layout.addSpacing(30)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_new = QPushButton("新建项目")
        self.btn_new.setMinimumWidth(120)
        btn_layout.addWidget(self.btn_new)

        self.btn_open = QPushButton("打开项目")
        self.btn_open.setMinimumWidth(120)
        btn_layout.addWidget(self.btn_open)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        layout.addSpacing(20)

        # Recent projects
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
    """Application main window with tab-based layout."""

    def __init__(self, config_path: Path | str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AutoLabel V3")

        # Load app config
        self._config_path = Path(config_path) if config_path else CONFIG_PATH
        self._app_config = AppConfig.load(self._config_path)
        geo = self._app_config.window_geometry
        self.setGeometry(geo["x"], geo["y"], geo["width"], geo["height"])

        # State
        self._project: ProjectManager | None = None
        self._label_panel: LabelPanel | None = None
        self._train_panel: TrainPanel | None = None
        self._model_panel: ModelPanel | None = None
        self._model_registry: ModelRegistry | None = None
        self._train_worker: TrainWorker | None = None

        # Central widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Welcome page
        self._welcome = WelcomePage(self._app_config)
        self._welcome.btn_new.clicked.connect(self._on_new_project)
        self._welcome.btn_open.clicked.connect(self._on_open_project)
        self._welcome.recent_list.itemDoubleClicked.connect(self._on_recent_clicked)
        self.tab_widget.addTab(self._welcome, "欢迎")

        # Menu bar
        self._setup_menus()

        # Status bar
        self._status_label = QLabel("就绪")
        self.statusBar().addPermanentWidget(self._status_label)

    def _setup_menus(self) -> None:
        mb = self.menuBar()

        # File menu
        file_menu = mb.addMenu("文件")

        new_action = QAction("新建项目", self)
        new_action.triggered.connect(self._on_new_project)
        file_menu.addAction(new_action)

        open_action = QAction("打开项目", self)
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        export_action = QAction("导出...", self)
        export_action.triggered.connect(self._on_export)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = mb.addMenu("编辑")

        classes_action = QAction("类别管理...", self)
        classes_action.triggered.connect(self._on_class_manager)
        edit_menu.addAction(classes_action)

    def open_project(self, project_manager: ProjectManager) -> None:
        """Open a project and switch to labeling workspace."""
        self._project = project_manager
        self.setWindowTitle(f"AutoLabel V3 — {project_manager.config.name}")

        # Update recent projects
        self._app_config.add_recent_project(str(project_manager.project_dir))
        self._app_config.save(self._config_path)

        # Model registry
        self._model_registry = ModelRegistry(project_manager.project_dir / "models")
        self._model_registry.load()

        # Create or update panels
        if self._label_panel is None:
            self._label_panel = LabelPanel(config_path=self._config_path)
            self.tab_widget.addTab(self._label_panel, "标注")
        self._label_panel.set_project(project_manager)

        if self._train_panel is None:
            self._train_panel = TrainPanel()
            self._train_panel._btn_start.clicked.connect(self._on_start_training)
            self._train_panel.stop_requested.connect(self._on_stop_training)
            self.tab_widget.addTab(self._train_panel, "训练")

        if self._model_panel is None:
            self._model_panel = ModelPanel()
            self.tab_widget.addTab(self._model_panel, "模型")
        self._model_panel.set_models(self._model_registry.list_models())

        self.tab_widget.setCurrentWidget(self._label_panel)

        self._status_label.setText(
            f"项目: {project_manager.config.name} | "
            f"图片: {len(project_manager.list_images())} | "
            f"类别: {len(project_manager.config.classes)}"
        )

    def _on_new_project(self) -> None:
        """Handle new project creation."""
        dlg = NewProjectDialog(self)
        if dlg.exec_():
            name, proj_dir, classes = dlg.get_values()
            if not name or not proj_dir:
                return
            try:
                pm = ProjectManager.create(proj_dir, name, classes=classes or None)
                self.open_project(pm)
            except Exception as e:
                logger.error("Failed to create project: %s", e)
                QMessageBox.warning(self, "错误", f"创建项目失败: {e}")

    def _on_open_project(self) -> None:
        """Handle open project via file dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self, "打开项目", "", "项目文件 (project.json)"
        )
        if path:
            project_dir = Path(path).parent
            try:
                pm = ProjectManager.open(project_dir)
                self.open_project(pm)
            except Exception as e:
                logger.error("Failed to open project: %s", e)

    def _on_recent_clicked(self, item: QListWidgetItem) -> None:
        """Handle double-click on recent project."""
        project_dir = Path(item.text())
        try:
            pm = ProjectManager.open(project_dir)
            self.open_project(pm)
        except Exception as e:
            logger.error("Failed to open recent project: %s", e)

    def _on_export(self) -> None:
        """Handle export dialog."""
        if not self._project:
            return
        dlg = ExportDialog(self)
        if dlg.exec_():
            fmt, out_dir, only_confirmed = dlg.get_values()
            if not out_dir:
                return
            try:
                from src.core.label_io import load_annotation
                annotations = []
                for img_path in self._project.list_images():
                    label_path = self._project.label_path_for(img_path)
                    ia = load_annotation(label_path)
                    if ia:
                        annotations.append(ia)

                if fmt == "YOLO":
                    from src.core.formats.yolo import export_yolo_detection
                    export_yolo_detection(annotations, out_dir, self._project.config.classes, only_confirmed)
                elif fmt == "COCO":
                    from src.core.formats.coco import export_coco
                    export_coco(annotations, Path(out_dir) / "coco.json", self._project.config.classes, only_confirmed)
                elif fmt == "labelme":
                    from src.core.formats.labelme import export_labelme
                    export_labelme(annotations, out_dir, only_confirmed)

                self._status_label.setText(f"导出完成: {fmt} → {out_dir}")
            except Exception as e:
                logger.error("Export failed: %s", e)
                QMessageBox.warning(self, "导出失败", str(e))

    def _on_class_manager(self) -> None:
        """Handle class management dialog."""
        if not self._project:
            return
        dlg = ClassManagerDialog(
            self._project.config.classes,
            self._project.config.class_colors,
            self,
        )
        if dlg.exec_():
            self._project.config.classes = dlg.get_classes()
            self._project.save()
            if self._label_panel:
                self._label_panel.set_project(self._project)

    def _on_start_training(self) -> None:
        """Start a training run."""
        if not self._project or not self._train_panel:
            return
        try:
            # Save current annotations first
            if self._label_panel:
                self._label_panel.save_and_cleanup()

            # Prepare dataset
            preparer = DatasetPreparer(self._project)
            task = self._train_panel._task_combo.currentText()
            val_ratio = self._train_panel.get_val_ratio()
            output_dir = self._project.project_dir / "datasets" / "current"
            data_yaml = preparer.prepare(output_dir, task=task, val_ratio=val_ratio)

            # Build config
            config = self._train_panel.get_train_config(
                data_yaml=str(data_yaml),
                model=self._train_panel._model_combo.currentText(),
            )
            config.project = str(self._project.project_dir / "models")
            config.name = f"{task}-train"

            # Launch worker
            self._train_worker = TrainWorker(config)
            self._train_worker.epoch_update.connect(self._train_panel.update_epoch)
            self._train_worker.finished_ok.connect(self._on_training_finished)
            self._train_worker.error.connect(self._train_panel.on_training_error)
            self._train_worker.start()

            self._train_panel.append_log(f"开始训练: {task} | {config.model} | {config.epochs} epochs")
        except Exception as e:
            logger.error("Failed to start training: %s", e)
            self._train_panel.on_training_error(str(e))

    def _on_stop_training(self) -> None:
        """Stop the current training run."""
        if self._train_worker and self._train_worker.isRunning():
            self._train_worker.terminate()
            self._train_panel.append_log("训练已停止")

    def _on_training_finished(self, metrics: dict) -> None:
        """Handle training completion."""
        if self._train_panel:
            self._train_panel.on_training_finished(metrics)
        # Register model
        if self._model_registry and self._project:
            from src.engine.model_manager import ModelInfo
            task = self._train_panel._task_combo.currentText() if self._train_panel else "detect"
            model_info = ModelInfo(
                name=f"{task}-{len(self._model_registry.list_models()) + 1}",
                path=f"models/{task}-train/weights/best.pt",
                task=task,
                base_model=self._train_panel._model_combo.currentText() if self._train_panel else "yolov8n.pt",
                classes=self._project.config.classes,
                metrics=metrics,
            )
            self._model_registry.register(model_info)
            self._model_registry.save()
            if self._model_panel:
                self._model_panel.set_models(self._model_registry.list_models())

    def closeEvent(self, event) -> None:
        """Save config and annotations on close."""
        if self._label_panel:
            self._label_panel.save_and_cleanup()
        geo = self.geometry()
        self._app_config.window_geometry = {
            "x": geo.x(), "y": geo.y(),
            "width": geo.width(), "height": geo.height(),
        }
        self._app_config.save(self._config_path)
        super().closeEvent(event)
