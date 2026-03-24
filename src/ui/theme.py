"""Catppuccin Mocha dark theme stylesheet for PyQt5."""

# Catppuccin Mocha palette
MOCHA = {
    "base": "#1e1e2e",
    "mantle": "#181825",
    "crust": "#11111b",
    "surface0": "#313244",
    "surface1": "#45475a",
    "surface2": "#585b70",
    "overlay0": "#6c7086",
    "text": "#cdd6f4",
    "subtext0": "#a6adc8",
    "green": "#a6e3a1",
    "blue": "#89b4fa",
    "red": "#f38ba8",
    "peach": "#fab387",
    "yellow": "#f9e2af",
    "mauve": "#cba6f7",
    "teal": "#94e2d5",
    "sky": "#89dceb",
    "lavender": "#b4befe",
}

STYLESHEET = f"""
QWidget {{
    background-color: {MOCHA['base']};
    color: {MOCHA['text']};
    font-size: 13px;
}}

QMainWindow {{
    background-color: {MOCHA['base']};
}}

QTabWidget::pane {{
    border: 1px solid {MOCHA['surface1']};
    background-color: {MOCHA['base']};
}}

QTabBar::tab {{
    background-color: {MOCHA['surface0']};
    color: {MOCHA['subtext0']};
    padding: 8px 20px;
    border: 1px solid {MOCHA['surface1']};
    border-bottom: none;
    margin-right: 2px;
}}

QTabBar::tab:selected {{
    background-color: {MOCHA['base']};
    color: {MOCHA['text']};
    border-bottom: 2px solid {MOCHA['blue']};
}}

QTabBar::tab:hover {{
    background-color: {MOCHA['surface1']};
}}

QPushButton {{
    background-color: {MOCHA['surface0']};
    color: {MOCHA['text']};
    border: 1px solid {MOCHA['surface1']};
    border-radius: 4px;
    padding: 6px 16px;
}}

QPushButton:hover {{
    background-color: {MOCHA['surface1']};
}}

QPushButton:pressed {{
    background-color: {MOCHA['surface2']};
}}

QPushButton:disabled {{
    color: {MOCHA['overlay0']};
    background-color: {MOCHA['mantle']};
}}

QListWidget {{
    background-color: {MOCHA['mantle']};
    border: 1px solid {MOCHA['surface0']};
    border-radius: 4px;
    outline: none;
}}

QListWidget::item {{
    padding: 4px 8px;
}}

QListWidget::item:selected {{
    background-color: {MOCHA['surface0']};
    color: {MOCHA['text']};
}}

QListWidget::item:hover {{
    background-color: {MOCHA['surface0']};
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {MOCHA['mantle']};
    color: {MOCHA['text']};
    border: 1px solid {MOCHA['surface1']};
    border-radius: 4px;
    padding: 4px 8px;
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {MOCHA['blue']};
}}

QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}

QTextEdit, QPlainTextEdit {{
    background-color: {MOCHA['mantle']};
    color: {MOCHA['text']};
    border: 1px solid {MOCHA['surface0']};
    border-radius: 4px;
}}

QLabel {{
    background-color: transparent;
}}

QGroupBox {{
    border: 1px solid {MOCHA['surface1']};
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 16px;
    font-weight: bold;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    padding: 0 6px;
    color: {MOCHA['subtext0']};
}}

QScrollBar:vertical {{
    background-color: {MOCHA['mantle']};
    width: 10px;
    border: none;
}}

QScrollBar::handle:vertical {{
    background-color: {MOCHA['surface1']};
    border-radius: 5px;
    min-height: 20px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {MOCHA['surface2']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: {MOCHA['mantle']};
    height: 10px;
    border: none;
}}

QScrollBar::handle:horizontal {{
    background-color: {MOCHA['surface1']};
    border-radius: 5px;
    min-width: 20px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {MOCHA['surface2']};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QStatusBar {{
    background-color: {MOCHA['mantle']};
    color: {MOCHA['subtext0']};
}}

QMenuBar {{
    background-color: {MOCHA['mantle']};
    color: {MOCHA['text']};
}}

QMenuBar::item:selected {{
    background-color: {MOCHA['surface0']};
}}

QMenu {{
    background-color: {MOCHA['surface0']};
    color: {MOCHA['text']};
    border: 1px solid {MOCHA['surface1']};
}}

QMenu::item:selected {{
    background-color: {MOCHA['surface1']};
}}

QProgressBar {{
    background-color: {MOCHA['surface0']};
    border: none;
    border-radius: 4px;
    text-align: center;
    color: {MOCHA['text']};
}}

QProgressBar::chunk {{
    background-color: {MOCHA['blue']};
    border-radius: 4px;
}}

QToolTip {{
    background-color: {MOCHA['surface0']};
    color: {MOCHA['text']};
    border: 1px solid {MOCHA['surface1']};
    padding: 4px;
}}

QSplitter::handle {{
    background-color: {MOCHA['surface0']};
}}

QSplitter::handle:hover {{
    background-color: {MOCHA['surface1']};
}}

QHeaderView::section {{
    background-color: {MOCHA['surface0']};
    color: {MOCHA['text']};
    border: 1px solid {MOCHA['surface1']};
    padding: 4px;
}}

QCheckBox {{
    color: {MOCHA['text']};
    spacing: 6px;
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid {MOCHA['surface2']};
    background-color: {MOCHA['mantle']};
}}

QCheckBox::indicator:checked {{
    background-color: {MOCHA['blue']};
    border-color: {MOCHA['blue']};
}}

QSlider::groove:horizontal {{
    background-color: {MOCHA['surface0']};
    height: 6px;
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background-color: {MOCHA['blue']};
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}}
"""


def apply_theme(app) -> None:
    """Apply the Catppuccin Mocha dark theme to a QApplication."""
    app.setStyleSheet(STYLESHEET)
