"""Audiobook Studio Pro — Application entry point.

Features: splash screen, pre-flight FFmpeg check, single instance guard,
Catppuccin Mocha/Latte theme support, dependency checking.
"""

import logging
import os
import sys
import tempfile

from PySide6.QtWidgets import QApplication, QSplashScreen, QMessageBox
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QColor, QPainter, QFont

from config import AppConfig
# styles inlined below
from main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("audiobook")

DARK_STYLESHEET = """
/* ══════════════════════════════════════════════════════════════
   Audiobook Studio Pro — Catppuccin Mocha Dark Theme
   Palette: Catppuccin Mocha + Blue Accent (#89b4fa)
   ══════════════════════════════════════════════════════════════ */

/* ── Global ──────────────────────────────────────────────────── */
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'Inter', 'Noto Sans', sans-serif;
    font-size: 13px;
}

QToolTip {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
}

/* ── Menu & Status Bar ───────────────────────────────────────── */
QMenuBar {
    background-color: #181825;
    color: #cdd6f4;
    border-bottom: 1px solid #313244;
    padding: 2px;
}
QMenuBar::item {
    padding: 6px 14px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #313244;
}
QMenu {
    background-color: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #313244;
    color: #89b4fa;
}
QStatusBar {
    background-color: #181825;
    color: #6c7086;
    border-top: 1px solid #313244;
    font-size: 11px;
    padding: 2px 8px;
}

/* ── Group Boxes ─────────────────────────────────────────────── */
QGroupBox {
    border: 1px solid #313244;
    border-radius: 6px;
    margin-top: 14px;
    padding: 16px 10px 10px 10px;
    font-weight: 600;
    font-size: 13px;
    color: #bac2de;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #89b4fa;
}

/* ── Inputs & Dropdowns ──────────────────────────────────────── */
QLineEdit, QTextEdit, QComboBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 7px 10px;
    color: #cdd6f4;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
    border-color: #89b4fa;
}
QLineEdit:disabled, QTextEdit:disabled {
    background-color: #1e1e2e;
    color: #585b70;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #6c7086;
    margin-right: 6px;
}
QComboBox QAbstractItemView {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    selection-background-color: #45475a;
    selection-color: #cdd6f4;
    outline: none;
    padding: 4px;
}

/* ── Buttons ─────────────────────────────────────────────────── */
QPushButton {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 7px 16px;
    color: #cdd6f4;
    font-weight: 500;
    min-height: 18px;
}
QPushButton:hover {
    background-color: #45475a;
    border-color: #585b70;
}
QPushButton:pressed {
    background-color: #1e1e2e;
    border-color: #89b4fa;
}
QPushButton:disabled {
    background-color: #1e1e2e;
    color: #45475a;
    border-color: #313244;
}
QPushButton:focus {
    border-color: #89b4fa;
    outline: none;
}

/* Primary Action */
QPushButton#startBtn {
    background-color: #89b4fa;
    color: #1e1e2e;
    font-size: 14px;
    font-weight: 700;
    padding: 10px 20px;
    border: none;
    border-radius: 4px;
    letter-spacing: 0.3px;
}
QPushButton#startBtn:hover {
    background-color: #b4d0fb;
}
QPushButton#startBtn:pressed {
    background-color: #74a8f7;
}
QPushButton#startBtn:disabled {
    background-color: #313244;
    color: #585b70;
}

/* Destructive Action */
QPushButton#cancelBtn {
    background-color: transparent;
    color: #f38ba8;
    font-size: 13px;
    font-weight: 600;
    padding: 10px 16px;
    border: 1px solid #f38ba8;
    border-radius: 4px;
}
QPushButton#cancelBtn:hover {
    background-color: rgba(243, 139, 168, 0.12);
}
QPushButton#cancelBtn:pressed {
    background-color: rgba(243, 139, 168, 0.2);
}
QPushButton#cancelBtn:disabled {
    color: #45475a;
    border-color: #45475a;
    background-color: transparent;
}

/* Secondary Action */
QPushButton#previewBtn {
    background-color: transparent;
    color: #a6e3a1;
    font-weight: 600;
    padding: 10px 16px;
    border: 1px solid #a6e3a1;
    border-radius: 4px;
}
QPushButton#previewBtn:hover {
    background-color: rgba(166, 227, 161, 0.12);
}
QPushButton#previewBtn:pressed {
    background-color: rgba(166, 227, 161, 0.2);
}
QPushButton#previewBtn:disabled {
    color: #45475a;
    border-color: #45475a;
    background-color: transparent;
}

/* Accent button */
QPushButton#metaBtn {
    background-color: transparent;
    color: #cba6f7;
    font-weight: 600;
    padding: 10px 16px;
    border: 1px solid #cba6f7;
    border-radius: 4px;
}
QPushButton#metaBtn:hover {
    background-color: rgba(203, 166, 247, 0.12);
}

/* ── Sliders ─────────────────────────────────────────────────── */
QSlider::groove:horizontal {
    border: none;
    height: 4px;
    background: #45475a;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #89b4fa;
    border: none;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #b4d0fb;
}
QSlider::sub-page:horizontal {
    background: #89b4fa;
    border-radius: 2px;
}

/* ── Progress Bar ────────────────────────────────────────────── */
QProgressBar {
    border: none;
    border-radius: 4px;
    background-color: #313244;
    text-align: center;
    color: #6c7086;
    font-weight: 600;
    font-size: 11px;
    height: 20px;
}
QProgressBar::chunk {
    border-radius: 4px;
    background-color: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #89b4fa, stop:1 #74c7ec
    );
}

/* ── Tabs ────────────────────────────────────────────────────── */
QTabWidget::pane {
    border: 1px solid #313244;
    border-radius: 6px;
    padding: 4px;
    background-color: #1e1e2e;
}
QTabBar::tab {
    background: #313244;
    border: 1px solid #313244;
    border-bottom: none;
    padding: 8px 20px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: #6c7086;
    font-weight: 500;
}
QTabBar::tab:selected {
    background: #1e1e2e;
    color: #89b4fa;
    font-weight: 700;
    border-bottom: 2px solid #89b4fa;
}
QTabBar::tab:hover:!selected {
    background: #45475a;
    color: #cdd6f4;
}

/* ── Spin Boxes ──────────────────────────────────────────────── */
QSpinBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 5px 8px;
    color: #cdd6f4;
}
QSpinBox:focus {
    border-color: #89b4fa;
}
QSpinBox::up-button, QSpinBox::down-button {
    background-color: #45475a;
    border: none;
    width: 16px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {
    background-color: #585b70;
}

/* ── Checkboxes ──────────────────────────────────────────────── */
QCheckBox {
    color: #cdd6f4;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 2px solid #45475a;
    background-color: #313244;
}
QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}
QCheckBox::indicator:hover {
    border-color: #89b4fa;
}

/* ── Scroll Bars ─────────────────────────────────────────────── */
QScrollBar:vertical {
    background: #1e1e2e;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #585b70;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: #1e1e2e;
    height: 10px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background: #45475a;
    border-radius: 5px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background: #585b70;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* ── Specific Custom Widgets ─────────────────────────────────── */
QTextEdit#logConsole {
    background-color: #11111b;
    border: 1px solid #1e1e2e;
    border-radius: 4px;
    font-family: 'Cascadia Code', 'JetBrains Mono', 'Consolas', monospace;
    font-size: 11px;
    color: #6c7086;
    padding: 4px;
}
QLabel#statusLabel {
    color: #cdd6f4;
    font-weight: 700;
    font-size: 12px;
}
QLabel#statsLabel {
    color: #6c7086;
    font-size: 11px;
    padding: 2px 4px;
}

/* ── Branding Header ─────────────────────────────────────────── */
QLabel#brandLabel {
    color: #89b4fa;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: 0.5px;
}
QLabel#brandSubLabel {
    color: #6c7086;
    font-size: 11px;
    font-weight: 400;
}

/* ── Quality Preset Badge ────────────────────────────────────── */
QLabel#presetBadge {
    color: #a6e3a1;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    background-color: rgba(166, 227, 161, 0.1);
    border: 1px solid rgba(166, 227, 161, 0.3);
    border-radius: 3px;
}

/* ── Table Widgets ───────────────────────────────────────────── */
QTableWidget {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    gridline-color: #45475a;
    color: #cdd6f4;
    selection-background-color: #45475a;
}
QTableWidget::item {
    padding: 4px 8px;
}
QHeaderView::section {
    background-color: #1e1e2e;
    color: #bac2de;
    border: 1px solid #313244;
    padding: 6px 8px;
    font-weight: 600;
    font-size: 12px;
}

/* ── Dialog Boxes ────────────────────────────────────────────── */
QDialog {
    background-color: #1e1e2e;
}
QMessageBox {
    background-color: #1e1e2e;
}
QMessageBox QLabel {
    color: #cdd6f4;
    font-size: 13px;
}

/* ── Splitter ────────────────────────────────────────────────── */
QSplitter::handle {
    background-color: #313244;
}
QSplitter::handle:hover {
    background-color: #89b4fa;
}

/* ── Special Buttons ─────────────────────────────────────────── */
QPushButton#autoAssignBtn {
    background-color: transparent;
    color: #f9e2af;
    font-weight: 600;
    padding: 8px 14px;
    border: 1px solid #f9e2af;
    border-radius: 4px;
}
QPushButton#autoAssignBtn:hover {
    background-color: rgba(249, 226, 175, 0.12);
}
QPushButton#assignVoicesBtn {
    background-color: transparent;
    color: #89b4fa;
    font-weight: 600;
    padding: 8px 14px;
    border: 1px solid #89b4fa;
    border-radius: 4px;
}
QPushButton#assignVoicesBtn:hover {
    background-color: rgba(137, 180, 250, 0.12);
}
QPushButton#assignVoicesBtn:disabled {
    color: #45475a;
    border-color: #45475a;
}
QPushButton#pauseBtn {
    background-color: transparent;
    color: #f9e2af;
    font-weight: 600;
    padding: 10px 16px;
    border: 1px solid #f9e2af;
    border-radius: 4px;
}
QPushButton#pauseBtn:hover {
    background-color: rgba(249, 226, 175, 0.12);
}
QPushButton#pauseBtn:disabled {
    color: #45475a;
    border-color: #45475a;
}
QPushButton#detectBtn {
    background-color: transparent;
    color: #cba6f7;
    font-weight: 600;
    padding: 8px 14px;
    border: 1px solid #cba6f7;
    border-radius: 4px;
}
QPushButton#detectBtn:hover {
    background-color: rgba(203, 166, 247, 0.12);
}
QPushButton#resetDefaultsBtn {
    background-color: transparent;
    color: #94e2d5;
    font-weight: 600;
    padding: 8px 14px;
    border: 1px solid #94e2d5;
    border-radius: 4px;
}
QPushButton#resetDefaultsBtn:hover {
    background-color: rgba(148, 226, 213, 0.12);
}

/* ── Speaker Card ────────────────────────────────────────────── */
QFrame#speakerCard {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 8px;
}
QFrame#speakerCard:hover {
    border-color: #89b4fa;
}

/* ── Voice Preview Button ───────────────────────────────────── */
QPushButton#voicePreviewBtn {
    background-color: transparent;
    color: #74c7ec;
    font-weight: 600;
    padding: 4px 10px;
    border: 1px solid #74c7ec;
    border-radius: 3px;
    font-size: 11px;
}
QPushButton#voicePreviewBtn:hover {
    background-color: rgba(116, 199, 236, 0.12);
}

/* ── Theme Toggle ────────────────────────────────────────────── */
QPushButton#themeToggleBtn {
    background-color: transparent;
    color: #f9e2af;
    font-weight: 600;
    padding: 6px 12px;
    border: 1px solid #f9e2af;
    border-radius: 4px;
    font-size: 12px;
}
QPushButton#themeToggleBtn:hover {
    background-color: rgba(249, 226, 175, 0.12);
}

/* ── Chapter Timeline ────────────────────────────────────────── */
QFrame#chapterTimeline {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 6px;
    padding: 4px;
}
QLabel#chapterLabel {
    color: #6c7086;
    font-size: 10px;
    padding: 1px 4px;
}
QLabel#chapterLabelActive {
    color: #89b4fa;
    font-size: 10px;
    font-weight: 700;
    padding: 1px 4px;
    background-color: rgba(137, 180, 250, 0.15);
    border-radius: 2px;
}
QLabel#chapterLabelDone {
    color: #a6e3a1;
    font-size: 10px;
    font-weight: 600;
    padding: 1px 4px;
}

/* ── Export Profile Button ───────────────────────────────────── */
QPushButton#exportProfileBtn {
    background-color: transparent;
    color: #fab387;
    font-weight: 600;
    padding: 8px 14px;
    border: 1px solid #fab387;
    border-radius: 4px;
}
QPushButton#exportProfileBtn:hover {
    background-color: rgba(250, 179, 135, 0.12);
}

/* ── Template Button ─────────────────────────────────────────── */
QPushButton#templateBtn {
    background-color: transparent;
    color: #94e2d5;
    font-weight: 600;
    padding: 8px 14px;
    border: 1px solid #94e2d5;
    border-radius: 4px;
}
QPushButton#templateBtn:hover {
    background-color: rgba(148, 226, 213, 0.12);
}

/* ── Backend Description Label ──────────────────────────────── */
QLabel#backendDesc {
    color: #6c7086;
    font-size: 11px;
    font-style: italic;
    padding: 2px 4px;
}
QComboBox#backendSelector {
    font-size: 13px;
    font-weight: 600;
    padding: 8px 12px;
    min-height: 24px;
}

/* ── List Widgets ────────────────────────────────────────────── */
QListWidget {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    color: #cdd6f4;
    outline: none;
}
QListWidget::item {
    padding: 4px 8px;
    border-radius: 3px;
}
QListWidget::item:selected {
    background-color: #45475a;
    color: #89b4fa;
}
QListWidget::item:hover:!selected {
    background-color: #3b3b52;
}

/* ── History Widget ──────────────────────────────────────────── */
QLabel#historyLabel {
    color: #6c7086;
    font-size: 11px;
}
"""


# ══════════════════════════════════════════════════════════════════
#  Catppuccin Latte — Light Theme
# ══════════════════════════════════════════════════════════════════

LIGHT_STYLESHEET = """
/* ══════════════════════════════════════════════════════════════
   Audiobook Studio Pro — Catppuccin Latte Light Theme
   Palette: Catppuccin Latte + Blue Accent (#1e66f5)
   ══════════════════════════════════════════════════════════════ */

/* ── Global ──────────────────────────────────────────────────── */
QMainWindow, QWidget {
    background-color: #eff1f5;
    color: #4c4f69;
    font-family: 'Segoe UI', 'Inter', 'Noto Sans', sans-serif;
    font-size: 13px;
}

QToolTip {
    background-color: #e6e9ef;
    color: #4c4f69;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
}

/* ── Menu & Status Bar ───────────────────────────────────────── */
QMenuBar {
    background-color: #e6e9ef;
    color: #4c4f69;
    border-bottom: 1px solid #ccd0da;
    padding: 2px;
}
QMenuBar::item {
    padding: 6px 14px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #ccd0da;
}
QMenu {
    background-color: #eff1f5;
    border: 1px solid #ccd0da;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #ccd0da;
    color: #1e66f5;
}
QStatusBar {
    background-color: #e6e9ef;
    color: #7c7f93;
    border-top: 1px solid #ccd0da;
    font-size: 11px;
    padding: 2px 8px;
}

/* ── Group Boxes ─────────────────────────────────────────────── */
QGroupBox {
    border: 1px solid #ccd0da;
    border-radius: 6px;
    margin-top: 14px;
    padding: 16px 10px 10px 10px;
    font-weight: 600;
    font-size: 13px;
    color: #5c5f77;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #1e66f5;
}

/* ── Inputs & Dropdowns ──────────────────────────────────────── */
QLineEdit, QTextEdit, QComboBox {
    background-color: #e6e9ef;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    padding: 7px 10px;
    color: #4c4f69;
    selection-background-color: #1e66f5;
    selection-color: #eff1f5;
}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
    border-color: #1e66f5;
}
QLineEdit:disabled, QTextEdit:disabled {
    background-color: #eff1f5;
    color: #9ca0b0;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #7c7f93;
    margin-right: 6px;
}
QComboBox QAbstractItemView {
    background-color: #e6e9ef;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    selection-background-color: #ccd0da;
    selection-color: #4c4f69;
    outline: none;
    padding: 4px;
}

/* ── Buttons ─────────────────────────────────────────────────── */
QPushButton {
    background-color: #e6e9ef;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    padding: 7px 16px;
    color: #4c4f69;
    font-weight: 500;
    min-height: 18px;
}
QPushButton:hover {
    background-color: #ccd0da;
    border-color: #bcc0cc;
}
QPushButton:pressed {
    background-color: #dce0e8;
    border-color: #1e66f5;
}
QPushButton:disabled {
    background-color: #eff1f5;
    color: #9ca0b0;
    border-color: #e6e9ef;
}
QPushButton:focus {
    border-color: #1e66f5;
    outline: none;
}

/* Primary Action */
QPushButton#startBtn {
    background-color: #1e66f5;
    color: #eff1f5;
    font-size: 14px;
    font-weight: 700;
    padding: 10px 20px;
    border: none;
    border-radius: 4px;
    letter-spacing: 0.3px;
}
QPushButton#startBtn:hover {
    background-color: #4c83f7;
}
QPushButton#startBtn:pressed {
    background-color: #1550d4;
}
QPushButton#startBtn:disabled {
    background-color: #ccd0da;
    color: #7c7f93;
}

/* Destructive Action */
QPushButton#cancelBtn {
    background-color: transparent;
    color: #d20f39;
    font-size: 13px;
    font-weight: 600;
    padding: 10px 16px;
    border: 1px solid #d20f39;
    border-radius: 4px;
}
QPushButton#cancelBtn:hover {
    background-color: rgba(210, 15, 57, 0.08);
}
QPushButton#cancelBtn:pressed {
    background-color: rgba(210, 15, 57, 0.15);
}
QPushButton#cancelBtn:disabled {
    color: #9ca0b0;
    border-color: #9ca0b0;
}

/* Secondary Action */
QPushButton#previewBtn {
    background-color: transparent;
    color: #40a02b;
    font-weight: 600;
    padding: 10px 16px;
    border: 1px solid #40a02b;
    border-radius: 4px;
}
QPushButton#previewBtn:hover {
    background-color: rgba(64, 160, 43, 0.08);
}
QPushButton#previewBtn:pressed {
    background-color: rgba(64, 160, 43, 0.15);
}
QPushButton#previewBtn:disabled {
    color: #9ca0b0;
    border-color: #9ca0b0;
}

/* Accent button */
QPushButton#metaBtn {
    background-color: transparent;
    color: #8839ef;
    font-weight: 600;
    padding: 10px 16px;
    border: 1px solid #8839ef;
    border-radius: 4px;
}
QPushButton#metaBtn:hover {
    background-color: rgba(136, 57, 239, 0.08);
}

/* ── Sliders ─────────────────────────────────────────────────── */
QSlider::groove:horizontal {
    border: none;
    height: 4px;
    background: #ccd0da;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #1e66f5;
    border: none;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #4c83f7;
}
QSlider::sub-page:horizontal {
    background: #1e66f5;
    border-radius: 2px;
}

/* ── Progress Bar ────────────────────────────────────────────── */
QProgressBar {
    border: none;
    border-radius: 4px;
    background-color: #ccd0da;
    text-align: center;
    color: #7c7f93;
    font-weight: 600;
    font-size: 11px;
    height: 20px;
}
QProgressBar::chunk {
    border-radius: 4px;
    background-color: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #1e66f5, stop:1 #209fb5
    );
}

/* ── Tabs ────────────────────────────────────────────────────── */
QTabWidget::pane {
    border: 1px solid #ccd0da;
    border-radius: 6px;
    padding: 4px;
    background-color: #eff1f5;
}
QTabBar::tab {
    background: #e6e9ef;
    border: 1px solid #ccd0da;
    border-bottom: none;
    padding: 8px 20px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: #7c7f93;
    font-weight: 500;
}
QTabBar::tab:selected {
    background: #eff1f5;
    color: #1e66f5;
    font-weight: 700;
    border-bottom: 2px solid #1e66f5;
}
QTabBar::tab:hover:!selected {
    background: #ccd0da;
    color: #4c4f69;
}

/* ── Spin Boxes ──────────────────────────────────────────────── */
QSpinBox {
    background-color: #e6e9ef;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    padding: 5px 8px;
    color: #4c4f69;
}
QSpinBox:focus {
    border-color: #1e66f5;
}
QSpinBox::up-button, QSpinBox::down-button {
    background-color: #ccd0da;
    border: none;
    width: 16px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {
    background-color: #bcc0cc;
}

/* ── Checkboxes ──────────────────────────────────────────────── */
QCheckBox {
    color: #4c4f69;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 2px solid #ccd0da;
    background-color: #e6e9ef;
}
QCheckBox::indicator:checked {
    background-color: #1e66f5;
    border-color: #1e66f5;
}
QCheckBox::indicator:hover {
    border-color: #1e66f5;
}

/* ── Scroll Bars ─────────────────────────────────────────────── */
QScrollBar:vertical {
    background: #eff1f5;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #ccd0da;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #bcc0cc;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #eff1f5;
    height: 10px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background: #ccd0da;
    border-radius: 5px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background: #bcc0cc;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* ── Specific Custom Widgets ─────────────────────────────────── */
QTextEdit#logConsole {
    background-color: #e6e9ef;
    border: 1px solid #dce0e8;
    border-radius: 4px;
    font-family: 'Cascadia Code', 'JetBrains Mono', 'Consolas', monospace;
    font-size: 11px;
    color: #7c7f93;
    padding: 4px;
}
QLabel#statusLabel {
    color: #4c4f69;
    font-weight: 700;
    font-size: 12px;
}
QLabel#statsLabel {
    color: #7c7f93;
    font-size: 11px;
    padding: 2px 4px;
}
QLabel#brandLabel {
    color: #1e66f5;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: 0.5px;
}
QLabel#brandSubLabel {
    color: #7c7f93;
    font-size: 11px;
}
QLabel#presetBadge {
    color: #40a02b;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    background-color: rgba(64, 160, 43, 0.08);
    border: 1px solid rgba(64, 160, 43, 0.25);
    border-radius: 3px;
}

/* ── Table Widgets ───────────────────────────────────────────── */
QTableWidget {
    background-color: #e6e9ef;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    gridline-color: #ccd0da;
    color: #4c4f69;
    selection-background-color: #ccd0da;
}
QTableWidget::item { padding: 4px 8px; }
QHeaderView::section {
    background-color: #dce0e8;
    color: #5c5f77;
    border: 1px solid #ccd0da;
    padding: 6px 8px;
    font-weight: 600;
    font-size: 12px;
}

/* ── Dialog Boxes ────────────────────────────────────────────── */
QDialog { background-color: #eff1f5; }
QMessageBox { background-color: #eff1f5; }
QMessageBox QLabel { color: #4c4f69; font-size: 13px; }

/* ── Splitter ────────────────────────────────────────────────── */
QSplitter::handle { background-color: #ccd0da; }
QSplitter::handle:hover { background-color: #1e66f5; }

/* ── Special Buttons (light variants) ────────────────────────── */
QPushButton#autoAssignBtn {
    background-color: transparent;
    color: #df8e1d;
    font-weight: 600;
    padding: 8px 14px;
    border: 1px solid #df8e1d;
    border-radius: 4px;
}
QPushButton#autoAssignBtn:hover { background-color: rgba(223, 142, 29, 0.08); }
QPushButton#assignVoicesBtn {
    background-color: transparent;
    color: #1e66f5;
    font-weight: 600;
    padding: 8px 14px;
    border: 1px solid #1e66f5;
    border-radius: 4px;
}
QPushButton#assignVoicesBtn:hover { background-color: rgba(30, 102, 245, 0.08); }
QPushButton#assignVoicesBtn:disabled { color: #9ca0b0; border-color: #9ca0b0; }
QPushButton#pauseBtn {
    background-color: transparent;
    color: #df8e1d;
    font-weight: 600;
    padding: 10px 16px;
    border: 1px solid #df8e1d;
    border-radius: 4px;
}
QPushButton#pauseBtn:hover { background-color: rgba(223, 142, 29, 0.08); }
QPushButton#pauseBtn:disabled { color: #9ca0b0; border-color: #9ca0b0; }
QPushButton#detectBtn {
    background-color: transparent;
    color: #8839ef;
    font-weight: 600;
    padding: 8px 14px;
    border: 1px solid #8839ef;
    border-radius: 4px;
}
QPushButton#detectBtn:hover { background-color: rgba(136, 57, 239, 0.08); }
QPushButton#resetDefaultsBtn {
    background-color: transparent;
    color: #179299;
    font-weight: 600;
    padding: 8px 14px;
    border: 1px solid #179299;
    border-radius: 4px;
}
QPushButton#resetDefaultsBtn:hover { background-color: rgba(23, 146, 153, 0.08); }

/* ── Speaker Card (light) ───────────────────────────────────── */
QFrame#speakerCard {
    background-color: #e6e9ef;
    border: 1px solid #ccd0da;
    border-radius: 8px;
    padding: 8px;
}
QFrame#speakerCard:hover { border-color: #1e66f5; }

/* ── Voice Preview Button ───────────────────────────────────── */
QPushButton#voicePreviewBtn {
    background-color: transparent;
    color: #209fb5;
    font-weight: 600;
    padding: 4px 10px;
    border: 1px solid #209fb5;
    border-radius: 3px;
    font-size: 11px;
}
QPushButton#voicePreviewBtn:hover { background-color: rgba(32, 159, 181, 0.08); }

/* ── Theme Toggle (light) ────────────────────────────────────── */
QPushButton#themeToggleBtn {
    background-color: transparent;
    color: #df8e1d;
    font-weight: 600;
    padding: 6px 12px;
    border: 1px solid #df8e1d;
    border-radius: 4px;
    font-size: 12px;
}
QPushButton#themeToggleBtn:hover { background-color: rgba(223, 142, 29, 0.08); }

/* ── Chapter Timeline (light) ────────────────────────────────── */
QFrame#chapterTimeline {
    background-color: #e6e9ef;
    border: 1px solid #ccd0da;
    border-radius: 6px;
    padding: 4px;
}
QLabel#chapterLabel {
    color: #7c7f93;
    font-size: 10px;
    padding: 1px 4px;
}
QLabel#chapterLabelActive {
    color: #1e66f5;
    font-size: 10px;
    font-weight: 700;
    padding: 1px 4px;
    background-color: rgba(30, 102, 245, 0.1);
    border-radius: 2px;
}
QLabel#chapterLabelDone {
    color: #40a02b;
    font-size: 10px;
    font-weight: 600;
    padding: 1px 4px;
}

/* ── Export/Template buttons (light) ─────────────────────────── */
QPushButton#exportProfileBtn {
    background-color: transparent;
    color: #fe640b;
    font-weight: 600;
    padding: 8px 14px;
    border: 1px solid #fe640b;
    border-radius: 4px;
}
QPushButton#exportProfileBtn:hover { background-color: rgba(254, 100, 11, 0.08); }
QPushButton#templateBtn {
    background-color: transparent;
    color: #179299;
    font-weight: 600;
    padding: 8px 14px;
    border: 1px solid #179299;
    border-radius: 4px;
}
QPushButton#templateBtn:hover { background-color: rgba(23, 146, 153, 0.08); }

QLabel#backendDesc {
    color: #7c7f93;
    font-size: 11px;
    font-style: italic;
    padding: 2px 4px;
}
QComboBox#backendSelector {
    font-size: 13px;
    font-weight: 600;
    padding: 8px 12px;
    min-height: 24px;
}

/* ── List Widgets (light) ────────────────────────────────────── */
QListWidget {
    background-color: #e6e9ef;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    color: #4c4f69;
    outline: none;
}
QListWidget::item { padding: 4px 8px; border-radius: 3px; }
QListWidget::item:selected { background-color: #ccd0da; color: #1e66f5; }
QListWidget::item:hover:!selected { background-color: #dce0e8; }

QLabel#historyLabel { color: #7c7f93; font-size: 11px; }
"""


def get_stylesheet(theme: str = "dark") -> str:
    """Return the appropriate stylesheet for the given theme."""
    if theme == "light":
        return LIGHT_STYLESHEET


APP_VERSION = "5.0.0"


# ══════════════════════════════════════════════════════════════════
#  Splash Screen
# ══════════════════════════════════════════════════════════════════

def _create_splash() -> QSplashScreen:
    """Generate a branded splash screen."""
    pm = QPixmap(480, 280)
    pm.fill(QColor("#1e1e2e"))

    painter = QPainter(pm)

    # Title
    painter.setPen(QColor("#89b4fa"))
    painter.setFont(QFont("Segoe UI", 28, QFont.Bold))
    painter.drawText(pm.rect(), Qt.AlignCenter, "Audiobook Studio Pro")

    # Version
    painter.setPen(QColor("#6c7086"))
    painter.setFont(QFont("Segoe UI", 11))
    sub_rect = pm.rect()
    sub_rect.moveTop(sub_rect.top() + 40)
    painter.drawText(sub_rect, Qt.AlignCenter, f"Version {APP_VERSION}")

    # Loading text
    painter.setPen(QColor("#45475a"))
    painter.setFont(QFont("Segoe UI", 10))
    load_rect = pm.rect()
    load_rect.moveTop(load_rect.top() + 80)
    painter.drawText(load_rect, Qt.AlignCenter, "Loading...")

    painter.end()

    splash = QSplashScreen(pm)
    splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.SplashScreen)
    return splash


# ══════════════════════════════════════════════════════════════════
#  Pre-flight Checks
# ══════════════════════════════════════════════════════════════════

def _check_ffmpeg() -> tuple[bool, str]:
    """Check if FFmpeg is available."""
    from audio_postprocessor import AudioPostProcessor
    return AudioPostProcessor.check_ffmpeg()


def _check_dependencies() -> list[str]:
    """Check for optional dependencies and return list of missing ones."""
    missing = []
    try:
        import pypdf
    except ImportError:
        missing.append("pypdf (PDF support)")
    try:
        import docx
    except ImportError:
        missing.append("python-docx (DOCX support)")
    try:
        import bs4
    except ImportError:
        missing.append("beautifulsoup4 (HTML/EPUB support)")
    try:
        import ebooklib
    except ImportError:
        missing.append("ebooklib (EPUB support)")
    try:
        import spacy
    except ImportError:
        missing.append("spacy (NLP speaker detection)")
    try:
        import edge_tts
    except ImportError:
        missing.append("edge-tts (Edge TTS backend)")
    return missing


# ══════════════════════════════════════════════════════════════════
#  Main Entry Point
# ══════════════════════════════════════════════════════════════════

def main():
    # ── Single instance guard ───────────────────────────────────
    lock_path = os.path.join(tempfile.gettempdir(), "audiobook_studio.lock")
    lock_file = None
    try:
        import fcntl
        lock_file = open(lock_path, "w")
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (ImportError, OSError):
        pass

    config = AppConfig.load()

    # ── Apply log level ─────────────────────────────────────────
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.getLogger("audiobook").setLevel(log_level)

    # ── Create Application ─────────────────────────────────────
    app = QApplication(sys.argv)
    app.setStyleSheet(get_stylesheet(config.theme))
    app.setStyle("Fusion")
    app.setApplicationName("Audiobook Studio Pro")
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("Audiobook Studio")

    # ── Splash Screen ──────────────────────────────────────────
    splash = _create_splash()
    splash.show()
    app.processEvents()

    # ── Pre-flight Checks ──────────────────────────────────────
    splash.showMessage("Checking FFmpeg...", Qt.AlignBottom | Qt.AlignHCenter, QColor("#6c7086"))
    app.processEvents()

    ffmpeg_ok, ffmpeg_msg = _check_ffmpeg()
    if not ffmpeg_ok and config.verify_ffmpeg_on_start:
        splash.close()
        QMessageBox.critical(
            None, "FFmpeg Required",
            f"Audiobook Studio Pro requires FFmpeg for audio processing.\n\n"
            f"{ffmpeg_msg}\n\n"
            f"Please install FFmpeg and ensure it is on your system PATH.\n"
            f"Download: https://ffmpeg.org/download.html"
        )
        sys.exit(1)

    splash.showMessage("Checking dependencies...", Qt.AlignBottom | Qt.AlignHCenter, QColor("#6c7086"))
    app.processEvents()

    missing = _check_dependencies()
    if missing:
        logger.warning("Optional dependencies missing: %s", ", ".join(missing))

    # ── Create Main Window ─────────────────────────────────────
    splash.showMessage("Loading interface...", Qt.AlignBottom | Qt.AlignHCenter, QColor("#6c7086"))
    app.processEvents()

    window = MainWindow(config)

    if missing and not hasattr(app, '_deps_warned'):
        app._deps_warned = True
        QTimer.singleShot(500, lambda: QMessageBox.warning(
            window, "Optional Dependencies",
            "Some optional dependencies are missing:\n\n- " +
            "\n- ".join(missing) +
            "\n\nInstall them with pip for full functionality."
        ))

    splash.finish(window)
    window.show()

    ret = app.exec()

    window.save_state()
    config.save()

    if lock_file:
        try:
            import fcntl
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
        except Exception:
            pass

    return ret


if __name__ == "__main__":
    sys.exit(main())
