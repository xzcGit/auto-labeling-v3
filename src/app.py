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
)
from PyQt5.QtCore import Qt

from src.core.config import AppConfig
from src.core.project import ProjectManager
from src.ui.label_panel import LabelPanel

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

        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def open_project(self, project_manager: ProjectManager) -> None:
        """Open a project and switch to labeling workspace."""
        self._project = project_manager
        self.setWindowTitle(f"AutoLabel V3 — {project_manager.config.name}")

        # Update recent projects
        self._app_config.add_recent_project(str(project_manager.project_dir))
        self._app_config.save(self._config_path)

        # Create or update label panel
        if self._label_panel is None:
            self._label_panel = LabelPanel(config_path=self._config_path)
            self.tab_widget.addTab(self._label_panel, "标注")
        self._label_panel.set_project(project_manager)
        self.tab_widget.setCurrentWidget(self._label_panel)

        self._status_label.setText(
            f"项目: {project_manager.config.name} | "
            f"图片: {len(project_manager.list_images())} | "
            f"类别: {len(project_manager.config.classes)}"
        )

    def _on_new_project(self) -> None:
        """Handle new project creation."""
        # Placeholder — will be connected to NewProjectDialog in Plan 5
        pass

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
