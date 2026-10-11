"""Main application window — professional audiobook production interface.

Audiobook Studio Pro: splitter-based layout with left settings panel
(tabbed: Source & Output, TTS Backend, Quality, Advanced) and right
generation panel (progress, chapter timeline, log, controls).

Major UI features:
  - Dark/Light theme toggle with persistent preference
  - Chapter timeline widget with real-time progress during generation
  - Improved TTS backend selector with status indicators & stacked settings
  - Export profile & project template quick-access toolbar buttons
  - Generation history dialog
  - Partial chapter regeneration via context menu on timeline
  - Reading-level stats display (Flesch-Kincaid, complexity, listening time)
  - Pronunciation auto-suggest
  - Live waveform preview animation during voice preview
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QFileDialog, QProgressBar,
    QMessageBox, QTextEdit, QGroupBox, QSpinBox, QSlider, QCheckBox,
    QFormLayout, QTabWidget, QStackedWidget, QSizePolicy,
    QMenu, QMenuBar, QStatusBar, QSplitter, QAction,
    QToolBar, QFrame, QScrollArea, QApplication, QDialog,
)
from PySide6.QtCore import Qt, QUrl, QThread, Signal, QSize, QTimer
from PySide6.QtGui import (
    QDragEnterEvent, QDropEvent, QAction as QActionType,
    QKeySequence, QIcon, QFont, QColor,
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from config import (
    AppConfig, EngineConfig, AudiobookProject,
    QUALITY_PRESETS, BACKEND_REGISTRY, EXPORT_PROFILES,
    PROJECT_TEMPLATES, SpeakerVoiceConfig,
)
from pipeline import AudiobookEngine, PipelineResult
from dialogs import (
    VoiceSearchDialog, LocalModelConfigDialog, MetadataDialog,
    VoiceAssignmentDialog, SpeakerMapDialog,
    PronunciationDialog, QualityPresetDialog, AboutDialog,
    ExportProfileDialog, ProjectTemplateDialog, GenerationHistoryDialog,
)
from styles import get_stylesheet
from text_processor import TextProcessor, Character
from tts_backends import TTSBackend


# ══════════════════════════════════════════════════════════════════
#  Speaker Detection Worker (runs in background thread)
# ══════════════════════════════════════════════════════════════════

class _SpeakerDetectWorker(QThread):
    """Background worker for speaker detection to avoid UI freeze."""
    finished = Signal(list, object)   # (speaker_names, character_registry)
    error = Signal(str)

    def __init__(self, source: str, is_raw_text: bool, parent=None):
        super().__init__(parent)
        self._source = source
        self._is_raw = is_raw_text

    def run(self):
        try:
            speakers, registry = TextProcessor.extract_speakers_detailed(
                self._source, self._is_raw
            )
            self.finished.emit(speakers, registry)
        except Exception as exc:
            self.error.emit(str(exc))


# ══════════════════════════════════════════════════════════════════
#  Pronunciation Suggest Worker
# ══════════════════════════════════════════════════════════════════

class _PronounSuggestWorker(QThread):
    """Background worker for pronunciation auto-suggest."""
    finished = Signal(dict)     # {word: phonetic}
    error = Signal(str)

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self._text = text

    def run(self):
        try:
            suggestions = TextProcessor.suggest_pronunciations(self._text)
            self.finished.emit(suggestions)
        except Exception as exc:
            self.error.emit(str(exc))


# ══════════════════════════════════════════════════════════════════
#  Chapter Timeline Widget
# ══════════════════════════════════════════════════════════════════

class ChapterTimelineWidget(QFrame):
    """Horizontal chapter timeline that shows generation progress per chapter.

    Each chapter is a clickable label. Active chapter is highlighted,
    completed chapters are shown in green. Right-click context menu
    allows partial chapter regeneration.
    """

    chapter_regen_requested = Signal(int)   # chapter index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("chapterTimeline")
        self._chapter_count = 0
        self._active_chapter = -1
        self._completed: set = set()
        self._labels: List[QLabel] = []

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(2)

        self._placeholder = QLabel("No chapters detected yet")
        self._placeholder.setObjectName("timelinePlaceholder")
        self._placeholder.setStyleSheet(
            "color: #6c7086; font-size: 11px; padding: 4px;"
        )
        self._layout.addWidget(self._placeholder)

        self.setMinimumHeight(40)
        self.setMaximumHeight(60)

    def set_chapters(self, count: int, titles: Optional[List[str]] = None):
        """Set the total number of chapters and rebuild timeline labels."""
        # Clear existing
        for lbl in self._labels:
            self._layout.removeWidget(lbl)
            lbl.deleteLater()
        self._labels.clear()
        self._completed.clear()
        self._active_chapter = -1
        self._chapter_count = count

        if self._placeholder:
            self._layout.removeWidget(self._placeholder)

        if count <= 0:
            self._placeholder = QLabel("No chapters detected yet")
            self._placeholder.setObjectName("timelinePlaceholder")
            self._placeholder.setStyleSheet(
                "color: #6c7086; font-size: 11px; padding: 4px;"
            )
            self._layout.addWidget(self._placeholder)
            return

        for i in range(count):
            title = (titles[i] if titles and i < len(titles)
                     else f"Ch {i + 1}")
            lbl = QLabel(title)
            lbl.setObjectName("chapterLabel")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(self._default_style())
            lbl.setToolTip(f"Chapter {i + 1}: {title}")
            lbl.setContextMenuPolicy(Qt.CustomContextMenu)
            lbl.customContextMenuRequested.connect(
                lambda pos, idx=i: self._show_context_menu(idx, pos)
            )
            lbl.setCursor(Qt.PointingHandCursor)
            lbl.setMinimumWidth(40)
            self._layout.addWidget(lbl)
            self._labels.append(lbl)

        self._layout.addStretch()

    def set_active(self, chapter_idx: int):
        """Highlight the currently processing chapter."""
        # Reset previous active
        if 0 <= self._active_chapter < len(self._labels):
            if self._active_chapter not in self._completed:
                self._labels[self._active_chapter].setStyleSheet(
                    self._default_style()
                )
        self._active_chapter = chapter_idx
        if 0 <= chapter_idx < len(self._labels):
            self._labels[chapter_idx].setStyleSheet(self._active_style())

    def mark_completed(self, chapter_idx: int):
        """Mark a chapter as completed."""
        self._completed.add(chapter_idx)
        if 0 <= chapter_idx < len(self._labels):
            self._labels[chapter_idx].setStyleSheet(self._done_style())

    def reset(self):
        """Reset all chapter states."""
        self._completed.clear()
        self._active_chapter = -1
        for lbl in self._labels:
            lbl.setStyleSheet(self._default_style())

    def _show_context_menu(self, chapter_idx: int, pos):
        """Show right-click context menu for partial chapter regeneration."""
        menu = QMenu(self)
        action = menu.addAction("🔄  Re-generate this chapter")
        action.setToolTip("Re-generate only this chapter using pipeline.run_partial()")
        act = menu.exec(self._labels[chapter_idx].mapToGlobal(pos))
        if act == action:
            self.chapter_regen_requested.emit(chapter_idx)

    # ── Styles ──────────────────────────────────────────────────

    @staticmethod
    def _default_style() -> str:
        return (
            "color: #cdd6f4; font-size: 11px; padding: 3px 8px; "
            "background-color: rgba(205,214,244,0.06); "
            "border: 1px solid rgba(205,214,244,0.15); border-radius: 4px;"
        )

    @staticmethod
    def _active_style() -> str:
        return (
            "color: #89b4fa; font-size: 11px; padding: 3px 8px; "
            "background-color: rgba(137,180,250,0.18); "
            "border: 1px solid rgba(137,180,250,0.5); border-radius: 4px; "
            "font-weight: bold;"
        )

    @staticmethod
    def _done_style() -> str:
        return (
            "color: #a6e3a1; font-size: 11px; padding: 3px 8px; "
            "background-color: rgba(166,227,161,0.12); "
            "border: 1px solid rgba(166,227,161,0.3); border-radius: 4px;"
        )


# ══════════════════════════════════════════════════════════════════
#  MainWindow
# ══════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """Primary application window for Audiobook Studio Pro.

    Layout: QSplitter with left panel (settings tabs) and right panel
    (generation: progress, chapter timeline, log, controls).
    """

    def __init__(self, config: AppConfig):
        super().__init__()
        self.setWindowTitle("Audiobook Studio Pro")
        self.setMinimumSize(config.window_width, config.window_height)
        self.setAcceptDrops(True)

        self.config = config
        self.engine: AudiobookEngine | None = None
        self._player: QMediaPlayer | None = None
        self._audio_output: QAudioOutput | None = None
        self._preview_thread: QThread | None = None
        self._detected_speakers: list = []
        self._character_registry = None
        self._current_theme: str = config.theme  # "dark" or "light"
        self._partial_engine = None  # for chapter regeneration

        self.default_output = str(Path(__file__).resolve().parent / "output")

        # ── Apply saved theme ───────────────────────────────────
        QApplication.instance().setStyleSheet(get_stylesheet(self._current_theme))

        # ── Menu Bar ────────────────────────────────────────────
        self._build_menu()

        # ── Toolbar ─────────────────────────────────────────────
        self._build_toolbar()

        # ── Status Bar ──────────────────────────────────────────
        self.statusBar().showMessage("Ready")

        # Export profile badge in status bar
        self._profile_badge = QLabel()
        self._profile_badge.setObjectName("profileBadge")
        self._profile_badge.setStyleSheet(
            "color: #f9e2af; font-size: 11px; padding: 2px 8px; "
            "background-color: rgba(249,226,175,0.1); "
            "border: 1px solid rgba(249,226,175,0.3); border-radius: 4px;"
        )
        self._update_profile_badge()
        self.statusBar().addPermanentWidget(self._profile_badge)

        # ── Central Splitter ────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # Left panel — settings tabs
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(6, 6, 3, 6)
        left_layout.setSpacing(4)

        # Branding row
        brand_row = QHBoxLayout()
        brand_label = QLabel("🎙️ Audiobook Studio Pro")
        brand_label.setObjectName("brandLabel")
        brand_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        brand_row.addWidget(brand_label)

        self.preset_badge = QLabel()
        self.preset_badge.setObjectName("presetBadge")
        self.preset_badge.setStyleSheet(
            "color: #a6e3a1; font-size: 11px; padding: 2px 8px; "
            "background-color: rgba(166,227,161,0.12); "
            "border: 1px solid rgba(166,227,161,0.3); border-radius: 4px;"
        )
        self._update_preset_badge()
        brand_row.addWidget(self.preset_badge)
        brand_row.addStretch()
        left_layout.addLayout(brand_row)

        # Tab widget
        self.tabs = QTabWidget()
        self._build_source_output_tab()
        self._build_backend_tab()
        self._build_quality_tab()
        self._build_advanced_tab()
        left_layout.addWidget(self.tabs, 1)

        splitter.addWidget(left_widget)

        # Right panel — generation
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(3, 6, 6, 6)
        right_layout.setSpacing(6)

        # Progress group
        prog_group = QGroupBox("Generation")
        prog_layout = QVBoxLayout(prog_group)

        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        prog_layout.addWidget(self.pbar)

        self.status_lbl = QLabel("Ready")
        self.status_lbl.setObjectName("statusLabel")
        self.status_lbl.setStyleSheet("font-weight: bold;")
        prog_layout.addWidget(self.status_lbl)

        # ── Chapter Timeline ────────────────────────────────────
        timeline_label = QLabel("Chapter Timeline")
        timeline_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        prog_layout.addWidget(timeline_label)

        self.chapter_timeline = ChapterTimelineWidget()
        self.chapter_timeline.chapter_regen_requested.connect(
            self._on_chapter_regen_requested
        )
        prog_layout.addWidget(self.chapter_timeline)

        # Control buttons
        ctrl_row = QHBoxLayout()

        self.btn_start = QPushButton("▶  Generate")
        self.btn_start.setObjectName("startBtn")
        self.btn_start.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.btn_start.setMinimumHeight(36)
        self.btn_start.setToolTip("Start audiobook generation (Ctrl+Enter)")
        self.btn_start.clicked.connect(self._start_generation)
        ctrl_row.addWidget(self.btn_start, 1)

        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.setObjectName("pauseBtn")
        self.btn_pause.setToolTip("Pause / Resume generation (Ctrl+Shift+P)")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._pause_generation)
        ctrl_row.addWidget(self.btn_pause)

        self.btn_cancel = QPushButton("⏹  Cancel")
        self.btn_cancel.setObjectName("cancelBtn")
        self.btn_cancel.setToolTip("Cancel generation (Escape)")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_generation)
        ctrl_row.addWidget(self.btn_cancel)

        self.btn_preview = QPushButton("🔊  Preview")
        self.btn_preview.setObjectName("previewBtn")
        self.btn_preview.setToolTip("Generate a short audio preview (Ctrl+P)")
        self.btn_preview.clicked.connect(self._preview_voice)
        ctrl_row.addWidget(self.btn_preview)

        prog_layout.addLayout(ctrl_row)
        right_layout.addWidget(prog_group)

        # ── Preview waveform animation ──────────────────────────
        self._preview_pbar = QProgressBar()
        self._preview_pbar.setObjectName("previewWaveform")
        self._preview_pbar.setMaximum(100)
        self._preview_pbar.setValue(0)
        self._preview_pbar.setVisible(False)
        self._preview_pbar.setTextVisible(False)
        self._preview_pbar.setMaximumHeight(8)
        self._preview_pbar.setStyleSheet(
            "QProgressBar#previewWaveform { "
            "  background-color: rgba(137,180,250,0.1); "
            "  border: none; border-radius: 4px; "
            "} "
            "QProgressBar#previewWaveform::chunk { "
            "  background-color: #89b4fa; border-radius: 4px; "
            "}"
        )
        right_layout.addWidget(self._preview_pbar)

        self._preview_anim_timer = QTimer(self)
        self._preview_anim_timer.setInterval(80)
        self._preview_anim_timer.timeout.connect(self._animate_preview_pbar)

        # Log console
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_box = QTextEdit()
        self.log_box.setObjectName("logConsole")
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet(
            "font-family: 'Consolas', 'Monaco', 'Courier New', monospace; "
            "font-size: 12px;"
        )
        log_layout.addWidget(self.log_box)
        right_layout.addWidget(log_group, 1)

        # Stats row
        self.gen_stats_label = QLabel()
        self.gen_stats_label.setObjectName("genStatsLabel")
        self.gen_stats_label.setStyleSheet(
            "color: #a6adc8; font-size: 11px; padding: 4px 8px; "
            "background-color: rgba(166,173,200,0.05); border-radius: 4px;"
        )
        right_layout.addWidget(self.gen_stats_label)

        splitter.addWidget(right_widget)

        # Set splitter sizes (left 55%, right 45%)
        splitter.setSizes([550, 450])
        splitter.setStretchFactor(0, 55)
        splitter.setStretchFactor(1, 45)

        # ── Initial Updates ─────────────────────────────────────
        self._auto_name()
        self._update_stats()
        self._on_backend_changed(self.backend_combo.currentData())

        # ── Restore Window Position ─────────────────────────────
        if config.window_x >= 0 and config.window_y >= 0:
            self.move(config.window_x, config.window_y)
        if config.window_maximized:
            self.showMaximized()

    # ═════════════════════════════════════════════════════════════
    # TOOLBAR
    # ═════════════════════════════════════════════════════════════

    def _build_toolbar(self):
        """Build the main toolbar with theme toggle, export profile, template."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        self.addToolBar(toolbar)

        # Theme toggle button
        self.btn_theme_toggle = QPushButton(
            "☀️  Light" if self._current_theme == "dark" else "🌙  Dark"
        )
        self.btn_theme_toggle.setObjectName("themeToggleBtn")
        self.btn_theme_toggle.setToolTip("Toggle dark/light theme")
        self.btn_theme_toggle.clicked.connect(self._toggle_theme)
        toolbar.addWidget(self.btn_theme_toggle)

        toolbar.addSeparator()

        # Export profile button
        self.btn_export_profile = QPushButton("📦  Export Profile")
        self.btn_export_profile.setObjectName("exportProfileBtn")
        self.btn_export_profile.setToolTip("Open export profile dialog")
        self.btn_export_profile.clicked.connect(self._open_export_profile)
        toolbar.addWidget(self.btn_export_profile)

        # Project template button
        self.btn_template = QPushButton("📋  Template")
        self.btn_template.setObjectName("templateBtn")
        self.btn_template.setToolTip("Open project template dialog")
        self.btn_template.clicked.connect(self._open_project_template)
        toolbar.addWidget(self.btn_template)

    # ═════════════════════════════════════════════════════════════
    # TAB 1: SOURCE & OUTPUT
    # ═════════════════════════════════════════════════════════════

    def _build_source_output_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)

        # ── Source file group ──────────────────────────────────
        src_group = QGroupBox("Source File")
        src_layout = QVBoxLayout(src_group)

        # Recent files
        if self.config.recent_files:
            recent_row = QHBoxLayout()
            recent_row.addWidget(QLabel("Recent:"))
            self.recent_combo = QComboBox()
            self.recent_combo.setMaximumWidth(350)
            self.recent_combo.addItem("-- Select recent file --")
            for rf in self.config.recent_files:
                if Path(rf).exists():
                    self.recent_combo.addItem(Path(rf).name, rf)
            self.recent_combo.currentIndexChanged.connect(self._on_recent_file)
            recent_row.addWidget(self.recent_combo, 1)
            src_layout.addLayout(recent_row)

        # File path + browse
        h_file = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setText(self.config.last_file)
        self.file_edit.setPlaceholderText(
            "PDF, TXT, DOCX, EPUB, HTML...  (drag & drop)"
        )
        self.file_edit.textChanged.connect(self._auto_name)
        self.file_edit.textChanged.connect(self._update_stats)
        self.file_edit.setToolTip(
            "Path to source document. Supports TXT, PDF, DOCX, EPUB, HTML, MD"
        )
        btn_browse = QPushButton("Browse")
        btn_browse.setToolTip("Browse for a source document file (Ctrl+O)")
        btn_browse.clicked.connect(self._load_source_file)
        h_file.addWidget(self.file_edit, 1)
        h_file.addWidget(btn_browse)
        src_layout.addLayout(h_file)

        # Drop zone label
        self.drop_label = QLabel("📂  Drop files here")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet(
            "color: #6c7086; border: 2px dashed #45475a; border-radius: 8px; "
            "padding: 12px; font-size: 13px;"
        )
        src_layout.addWidget(self.drop_label)

        # Or paste text
        src_layout.addWidget(QLabel("— or paste text below —"))
        self.raw_text = QTextEdit()
        self.raw_text.setPlaceholderText("Paste your text here...")
        self.raw_text.setMaximumHeight(120)
        self.raw_text.textChanged.connect(self._auto_name)
        self.raw_text.textChanged.connect(self._update_stats)
        self.raw_text.setToolTip("Paste raw text content for audiobook generation")
        src_layout.addWidget(self.raw_text)

        layout.addWidget(src_group)

        # ── Output group ───────────────────────────────────────
        out_group = QGroupBox("Output")
        out_form = QFormLayout(out_group)

        h_dir = QHBoxLayout()
        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText(f"Auto: {self.default_output}")
        self.out_dir.setText(self.config.output_dir)
        self.out_dir.textChanged.connect(self._auto_name)
        self.out_dir.setToolTip("Output directory. Leave blank for default.")
        btn_dir = QPushButton("Browse")
        btn_dir.setToolTip("Browse for output directory")
        btn_dir.clicked.connect(self._browse_dir)
        h_dir.addWidget(self.out_dir, 1)
        h_dir.addWidget(btn_dir)
        out_form.addRow("Directory:", h_dir)

        self.out_name = QLineEdit()
        self.out_name.setPlaceholderText("Auto-filled from source")
        self.out_name.setToolTip("Output filename (without extension)")
        out_form.addRow("Filename:", self.out_name)

        self.format_combo = QComboBox()
        self.format_combo.addItems([
            "mp3", "wav", "m4b", "both (MP3+WAV)", "all (MP3+WAV+M4B)"
        ])
        fmt = self.config.output_format
        fmt_map = {"mp3": 0, "wav": 1, "m4b": 2, "both": 3, "all": 4}
        self.format_combo.setCurrentIndex(fmt_map.get(fmt, 0))
        self.format_combo.setToolTip("Output audio format")
        out_form.addRow("Format:", self.format_combo)

        layout.addWidget(out_group)

        # ── Stats ──────────────────────────────────────────────
        self.stats_label = QLabel()
        self.stats_label.setObjectName("statsLabel")
        self.stats_label.setStyleSheet(
            "color: #cdd6f4; font-size: 12px; padding: 4px 8px; "
            "background-color: rgba(205,214,244,0.06); border-radius: 4px;"
        )
        layout.addWidget(self.stats_label)

        # Reading level stats
        self.reading_level_label = QLabel()
        self.reading_level_label.setObjectName("readingLevelLabel")
        self.reading_level_label.setStyleSheet(
            "color: #cba6f7; font-size: 11px; padding: 4px 8px; "
            "background-color: rgba(203,166,247,0.06); "
            "border: 1px solid rgba(203,166,247,0.12); border-radius: 4px;"
        )
        layout.addWidget(self.reading_level_label)

        # Quality preset badge in stats
        self.quality_preset_label = QLabel()
        self.quality_preset_label.setStyleSheet("color: #89b4fa; font-size: 11px;")
        layout.addWidget(self.quality_preset_label)
        self._update_quality_preset_label()

        layout.addStretch()
        self.tabs.addTab(tab, "📄 Source & Output")

    # ═════════════════════════════════════════════════════════════
    # TAB 2: TTS BACKEND (THE IMPROVED SELECTOR)
    # ═════════════════════════════════════════════════════════════

    def _build_backend_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)

        # ── Backend Selector Combo ────────────────────────────
        selector_group = QGroupBox("TTS Backend")
        selector_layout = QVBoxLayout(selector_group)

        # Combo row
        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel("Engine:"))
        self.backend_combo = QComboBox()
        self.backend_combo.setObjectName("backendSelector")
        for bid, info in BACKEND_REGISTRY.items():
            tag = ""
            if info.supports_ssml:
                tag += " +SSML"
            if info.supports_multi_voice:
                tag += " +Multi-Voice"
            if info.is_local:
                tag += " (Local)"
            self.backend_combo.addItem(f"{info.icon}  {info.name}{tag}", bid)
        idx = self.backend_combo.findData(self.config.backend)
        if idx >= 0:
            self.backend_combo.setCurrentIndex(idx)
        self.backend_combo.currentIndexChanged.connect(
            lambda: self._on_backend_changed(self.backend_combo.currentData())
        )
        self.backend_combo.setToolTip("Select the TTS synthesis backend")
        combo_row.addWidget(self.backend_combo, 1)

        # Configure Backend button
        self.btn_configure = QPushButton("⚙  Configure")
        self.btn_configure.setToolTip(
            "Configure the selected TTS backend (model path, API key, etc.)"
        )
        self.btn_configure.clicked.connect(self._configure_backend)
        combo_row.addWidget(self.btn_configure)
        selector_layout.addLayout(combo_row)

        # Backend description label
        self.backend_desc = QLabel()
        self.backend_desc.setObjectName("backendDesc")
        self.backend_desc.setWordWrap(True)
        self.backend_desc.setStyleSheet(
            "color: #cdd6f4; font-size: 12px; padding: 6px 10px; "
            "background-color: rgba(205,214,244,0.05); "
            "border: 1px solid rgba(205,214,244,0.1); border-radius: 6px;"
        )
        selector_layout.addWidget(self.backend_desc)

        # Backend status indicator
        self.backend_status_indicator = QLabel()
        self.backend_status_indicator.setObjectName("backendStatusIndicator")
        self.backend_status_indicator.setStyleSheet(
            "font-size: 11px; padding: 2px 8px; border-radius: 3px;"
        )
        self._update_backend_status()
        selector_layout.addWidget(self.backend_status_indicator)

        layout.addWidget(selector_group)

        # ── Backend-Specific Settings (stacked widget) ────────
        settings_group = QGroupBox("Backend Settings")
        settings_layout = QVBoxLayout(settings_group)

        self.backend_stack = QStackedWidget()

        # --- Edge TTS page ---
        edge_page = QWidget()
        ef = QFormLayout(edge_page)

        h_v = QHBoxLayout()
        self.edge_voice = QComboBox()
        self.edge_voice.setEditable(True)
        self.edge_voice.addItems([
            "en-US-AvaNeural", "en-US-AndrewNeural", "en-US-SamuelNeural",
            "en-US-MichelleNeural", "en-GB-SoniaNeural", "en-GB-RyanNeural",
        ])
        self.edge_voice.setCurrentText(self.config.edge_voice)
        self.edge_voice.setToolTip(
            "Voice for Edge TTS synthesis. Click 'Search' for the full catalog."
        )
        btn_sv = QPushButton("🔍  Search")
        btn_sv.setToolTip("Search the complete Edge TTS voice catalog (400+ voices)")
        btn_sv.clicked.connect(self._search_voices)
        h_v.addWidget(self.edge_voice, 1)
        h_v.addWidget(btn_sv)
        ef.addRow("Voice:", h_v)

        self.edge_rate, self.edge_rate_lbl = self._make_slider(
            "Speed:", self.config.edge_rate, "%"
        )
        ef.addRow(self.edge_rate_lbl)
        ef.addRow(self.edge_rate)
        self.edge_vol, self.edge_vol_lbl = self._make_slider(
            "Volume:", self.config.edge_volume, "%"
        )
        ef.addRow(self.edge_vol_lbl)
        ef.addRow(self.edge_vol)
        self.edge_pitch, self.edge_pitch_lbl = self._make_slider(
            "Pitch:", self.config.edge_pitch, "Hz"
        )
        ef.addRow(self.edge_pitch_lbl)
        ef.addRow(self.edge_pitch)

        self.chk_ssml = QCheckBox("SSML Multi-Voice Synthesis")
        self.chk_ssml.setChecked(self.config.use_ssml)
        self.chk_ssml.setToolTip(
            "Use SSML markup for multi-speaker synthesis. "
            "Enables voice switching within a single audio chunk for Edge TTS."
        )
        ef.addRow(self.chk_ssml)

        self.backend_stack.addWidget(edge_page)

        # --- Qwen Local page ---
        qwen_page = QWidget()
        qf = QFormLayout(qwen_page)
        self.qwen_voice = QComboBox()
        self.qwen_voice.setEditable(True)
        self.qwen_voice.addItems([
            "Cherry", "Serena", "Ethan", "Chelsie", "Echo", "Bonnie"
        ])
        self.qwen_voice.setCurrentText(self.config.qwen_voice)
        self.qwen_voice.setToolTip("Select the voice for Qwen TTS")
        qf.addRow("Voice:", self.qwen_voice)

        h_qpath = QHBoxLayout()
        self.qwen_path_edit = QLineEdit()
        self.qwen_path_edit.setText(self.config.qwen_model_path)
        self.qwen_path_edit.setPlaceholderText("HuggingFace repo ID or local path")
        self.qwen_path_edit.setToolTip("Model path for Qwen TTS")
        h_qpath.addWidget(self.qwen_path_edit, 1)
        qf.addRow("Model Path:", h_qpath)

        self.qwen_device = QComboBox()
        self.qwen_device.addItems(["cuda", "cpu"])
        self.qwen_device.setCurrentText(self.config.qwen_device)
        self.qwen_device.setToolTip("Compute device for Qwen TTS")
        qf.addRow("Device:", self.qwen_device)

        self.qwen_status = QLabel(
            "Model: " + (self.config.qwen_model_path or "Not Set")
        )
        self.qwen_status.setStyleSheet("color: #6c7086; font-size: 11px;")
        qf.addRow(self.qwen_status)

        self.backend_stack.addWidget(qwen_page)

        # --- OmniVoice Local page ---
        omni_page = QWidget()
        of_layout = QFormLayout(omni_page)
        self.omni_speaker = QSpinBox()
        self.omni_speaker.setRange(0, 100)
        self.omni_speaker.setValue(self.config.omnivoice_speaker_id)
        self.omni_speaker.setToolTip("Speaker ID for OmniVoice model")
        of_layout.addRow("Speaker ID:", self.omni_speaker)

        h_opath = QHBoxLayout()
        self.omni_path_edit = QLineEdit()
        self.omni_path_edit.setText(self.config.omnivoice_model_path)
        self.omni_path_edit.setPlaceholderText("HuggingFace repo ID or local path")
        self.omni_path_edit.setToolTip("Model path for OmniVoice TTS")
        h_opath.addWidget(self.omni_path_edit, 1)
        of_layout.addRow("Model Path:", h_opath)

        self.omni_device = QComboBox()
        self.omni_device.addItems(["cuda", "cpu"])
        self.omni_device.setCurrentText(self.config.omnivoice_device)
        self.omni_device.setToolTip("Compute device for OmniVoice TTS")
        of_layout.addRow("Device:", self.omni_device)

        self.omni_status = QLabel(
            "Model: " + (self.config.omnivoice_model_path or "Not Set")
        )
        self.omni_status.setStyleSheet("color: #6c7086; font-size: 11px;")
        of_layout.addRow(self.omni_status)

        self.backend_stack.addWidget(omni_page)

        # --- DramaBox page ---
        dramabox_page = QWidget()
        db_layout = QFormLayout(dramabox_page)

        self.dramabox_api_url = QLineEdit()
        self.dramabox_api_url.setText(self.config.dramabox_api_url)
        self.dramabox_api_url.setPlaceholderText(
            "https://api.dramabox.example.com"
        )
        self.dramabox_api_url.setToolTip(
            "DramaBox API endpoint URL (leave blank for local inference)"
        )
        db_layout.addRow("API URL:", self.dramabox_api_url)

        self.dramabox_api_key = QLineEdit()
        self.dramabox_api_key.setText(self.config.dramabox_api_key)
        self.dramabox_api_key.setPlaceholderText("Your DramaBox API key")
        self.dramabox_api_key.setEchoMode(QLineEdit.Password)
        self.dramabox_api_key.setToolTip(
            "API key for DramaBox service authentication"
        )
        db_layout.addRow("API Key:", self.dramabox_api_key)

        self.dramabox_style = QComboBox()
        self.dramabox_style.addItems([
            "dramatic", "natural", "expressive", "calm"
        ])
        self.dramabox_style.setCurrentText(self.config.dramabox_voice_style)
        self.dramabox_style.setToolTip("Voice style for DramaBox synthesis")
        db_layout.addRow("Voice Style:", self.dramabox_style)

        self.dramabox_model = QLineEdit()
        self.dramabox_model.setText(self.config.dramabox_model)
        self.dramabox_model.setPlaceholderText("dramabox-v1")
        self.dramabox_model.setToolTip("DramaBox model version")
        db_layout.addRow("Model:", self.dramabox_model)

        self.dramabox_status = QLabel()
        if self.config.dramabox_api_url:
            self.dramabox_status.setText(f"API: {self.config.dramabox_api_url}")
        else:
            self.dramabox_status.setText("Mode: Local inference")
        self.dramabox_status.setStyleSheet("color: #6c7086; font-size: 11px;")
        db_layout.addRow(self.dramabox_status)

        self.chk_dramabox_ssml = QCheckBox("SSML Multi-Voice Synthesis")
        self.chk_dramabox_ssml.setChecked(self.config.use_ssml)
        self.chk_dramabox_ssml.setToolTip(
            "Use SSML markup for multi-speaker synthesis with DramaBox. "
            "Supports emotional style tags and voice switching."
        )
        db_layout.addRow(self.chk_dramabox_ssml)

        self.backend_stack.addWidget(dramabox_page)

        settings_layout.addWidget(self.backend_stack)
        layout.addWidget(settings_group)

        # ── Multi-Speaker group ────────────────────────────────
        ms_group = QGroupBox("Multi-Speaker")
        ms_layout = QVBoxLayout(ms_group)

        chk_row = QHBoxLayout()
        self.chk_multi = QCheckBox("Enable Multi-Speaker")
        self.chk_multi.setChecked(self.config.multi_speaker)
        self.chk_multi.setToolTip(
            "Use NLP to detect dialogue and assign different voices to characters. "
            "Requires voice assignment configuration."
        )
        chk_row.addWidget(self.chk_multi)

        self.speaker_count_badge = QLabel()
        self.speaker_count_badge.setStyleSheet(
            "color: #f9e2af; font-size: 11px; padding: 2px 6px; "
            "background-color: rgba(249,226,175,0.1); "
            "border: 1px solid rgba(249,226,175,0.3); border-radius: 3px;"
        )
        chk_row.addWidget(self.speaker_count_badge)
        chk_row.addStretch()
        ms_layout.addLayout(chk_row)

        # Detect & assign buttons
        detect_row = QHBoxLayout()
        self.btn_detect = QPushButton("🔍  Detect Speakers")
        self.btn_detect.setToolTip("Run NLP speaker detection on the source text")
        self.btn_detect.clicked.connect(self._detect_speakers)
        detect_row.addWidget(self.btn_detect)

        self.btn_assign_voices = QPushButton("🎭  Assign Voices")
        self.btn_assign_voices.setObjectName("assignVoicesBtn")
        self.btn_assign_voices.setToolTip(
            "Open the professional voice assignment dialog with auto-assign "
            "and manual voice selection for each detected speaker"
        )
        self.btn_assign_voices.clicked.connect(self._assign_voices)
        self.btn_assign_voices.setEnabled(self.config.multi_speaker)
        detect_row.addWidget(self.btn_assign_voices)

        self.btn_speaker_map = QPushButton("📋  Speaker Map")
        self.btn_speaker_map.setToolTip("Legacy character-to-voice mapping dialog")
        self.btn_speaker_map.clicked.connect(self._open_speaker_map)
        self.btn_speaker_map.setEnabled(self.config.multi_speaker)
        detect_row.addWidget(self.btn_speaker_map)

        ms_layout.addLayout(detect_row)

        self.chk_multi.toggled.connect(self.btn_assign_voices.setEnabled)
        self.chk_multi.toggled.connect(self.btn_speaker_map.setEnabled)

        # Auto-assign
        self.chk_auto_assign = QCheckBox("Auto-assign voices on detect")
        self.chk_auto_assign.setChecked(self.config.auto_assign_on_detect)
        self.chk_auto_assign.setToolTip(
            "Automatically assign voices to newly detected speakers "
            "using the gender-match strategy"
        )
        ms_layout.addWidget(self.chk_auto_assign)

        layout.addWidget(ms_group)
        layout.addStretch()
        self.tabs.addTab(tab, "🎙️ TTS Backend")

    # ═════════════════════════════════════════════════════════════
    # TAB 3: QUALITY
    # ═════════════════════════════════════════════════════════════

    def _build_quality_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)

        # ── Quality preset ─────────────────────────────────────
        preset_group = QGroupBox("Quality Preset")
        preset_layout = QVBoxLayout(preset_group)

        h_preset = QHBoxLayout()
        self.quality_combo = QComboBox()
        for key, preset in QUALITY_PRESETS.items():
            self.quality_combo.addItem(preset.name, key)
        idx = self.quality_combo.findData(self.config.quality_preset)
        if idx >= 0:
            self.quality_combo.setCurrentIndex(idx)
        self.quality_combo.currentIndexChanged.connect(self._on_quality_changed)
        self.quality_combo.setToolTip(
            "Select audio quality preset for different distribution targets"
        )
        h_preset.addWidget(self.quality_combo, 1)

        btn_preset_cfg = QPushButton("⚙  Configure Custom")
        btn_preset_cfg.setToolTip("Open quality preset configuration dialog")
        btn_preset_cfg.clicked.connect(self._open_quality_preset)
        h_preset.addWidget(btn_preset_cfg)
        preset_layout.addLayout(h_preset)

        # Preset description
        self.preset_desc = QLabel()
        self.preset_desc.setWordWrap(True)
        self.preset_desc.setStyleSheet("color: #6c7086; font-size: 11px;")
        self._update_preset_desc()
        preset_layout.addWidget(self.preset_desc)

        layout.addWidget(preset_group)

        # ── Audio settings ─────────────────────────────────────
        audio_group = QGroupBox("Audio Processing")
        audio_form = QFormLayout(audio_group)

        self.chk_norm = QCheckBox("EBU R128 loudness normalization")
        self.chk_norm.setChecked(self.config.normalize_audio)
        self.chk_norm.setToolTip(
            "Apply two-pass EBU R128 loudness normalization. "
            "Ensures consistent volume across the entire audiobook."
        )
        audio_form.addRow(self.chk_norm)

        self.chk_trim = QCheckBox("Trim inter-chunk silence")
        self.chk_trim.setChecked(self.config.trim_silence)
        self.chk_trim.setToolTip("Remove excessive silence between audio chunks")
        audio_form.addRow(self.chk_trim)

        self.chk_chapters = QCheckBox("Chapter-aware generation")
        self.chk_chapters.setChecked(self.config.chapter_aware)
        self.chk_chapters.setToolTip(
            "Detect chapter boundaries and process each chapter separately. "
            "Enables M4B chapter markers."
        )
        audio_form.addRow(self.chk_chapters)

        self.chk_ssml_quality = QCheckBox("SSML voice switching")
        self.chk_ssml_quality.setChecked(self.config.use_ssml)
        self.chk_ssml_quality.setToolTip(
            "Use SSML markup to switch voices within a single synthesis chunk"
        )
        audio_form.addRow(self.chk_ssml_quality)

        layout.addWidget(audio_group)
        layout.addStretch()
        self.tabs.addTab(tab, "🎵 Quality")

    # ═════════════════════════════════════════════════════════════
    # TAB 4: ADVANCED
    # ═════════════════════════════════════════════════════════════

    def _build_advanced_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)

        # ── Performance group ──────────────────────────────────
        perf_group = QGroupBox("Performance")
        perf_form = QFormLayout(perf_group)

        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(1, 20)
        self.spin_workers.setValue(self.config.concurrency)
        self.spin_workers.setToolTip(
            "Number of parallel TTS synthesis workers. "
            "Higher values = faster but more resource-intensive."
        )
        perf_form.addRow("Concurrency:", self.spin_workers)

        self.spin_chunk = QSpinBox()
        self.spin_chunk.setRange(100, 5000)
        self.spin_chunk.setSingleStep(50)
        self.spin_chunk.setValue(self.config.chunk_size)
        self.spin_chunk.setToolTip(
            "Maximum characters per TTS chunk. Smaller chunks = more granular progress, "
            "larger chunks = fewer API calls."
        )
        perf_form.addRow("Chunk Size:", self.spin_chunk)

        layout.addWidget(perf_group)

        # ── Tools group ────────────────────────────────────────
        tools_group = QGroupBox("Tools")
        tools_layout = QVBoxLayout(tools_group)

        # Pronunciation dictionary + auto-suggest
        pron_row = QHBoxLayout()
        btn_pron = QPushButton("📝  Pronunciation Dictionary")
        btn_pron.setToolTip(
            "Edit the pronunciation dictionary to fix TTS mispronunciations"
        )
        btn_pron.clicked.connect(self._open_pronunciation)
        pron_row.addWidget(btn_pron)

        self.btn_suggest_pron = QPushButton("💡  Auto-Suggest")
        self.btn_suggest_pron.setObjectName("suggestPronBtn")
        self.btn_suggest_pron.setToolTip(
            "Auto-detect unusual words and suggest phonetic pronunciations"
        )
        self.btn_suggest_pron.clicked.connect(self._auto_suggest_pronunciations)
        pron_row.addWidget(self.btn_suggest_pron)
        tools_layout.addLayout(pron_row)

        # Pronunciation suggestions display area
        self.pron_suggestions_label = QLabel()
        self.pron_suggestions_label.setObjectName("pronSuggestionsLabel")
        self.pron_suggestions_label.setWordWrap(True)
        self.pron_suggestions_label.setStyleSheet(
            "color: #f9e2af; font-size: 11px; padding: 4px 8px; "
            "background-color: rgba(249,226,175,0.05); "
            "border: 1px solid rgba(249,226,175,0.12); border-radius: 4px;"
        )
        self.pron_suggestions_label.setVisible(False)
        tools_layout.addWidget(self.pron_suggestions_label)

        btn_meta = QPushButton("📖  Metadata")
        btn_meta.setToolTip("Edit audiobook metadata (title, author, cover art)")
        btn_meta.clicked.connect(self._open_metadata)
        tools_layout.addWidget(btn_meta)

        layout.addWidget(tools_group)
        layout.addStretch()
        self.tabs.addTab(tab, "⚙ Advanced")

    # ═════════════════════════════════════════════════════════════
    # MENU BAR
    # ═════════════════════════════════════════════════════════════

    def _build_menu(self):
        menubar = self.menuBar()

        # ── File menu ──────────────────────────────────────────
        file_menu = menubar.addMenu("&File")

        act_new = QActionType("New Project", self)
        act_new.setShortcut(QKeySequence("Ctrl+N"))
        act_new.setStatusTip("Start a new project")
        act_new.triggered.connect(self._new_project)
        file_menu.addAction(act_new)

        act_open = QActionType("Import File...", self)
        act_open.setShortcut(QKeySequence("Ctrl+O"))
        act_open.setStatusTip("Open a source document")
        act_open.triggered.connect(self._load_source_file)
        file_menu.addAction(act_open)

        act_raw = QActionType("Import Text...", self)
        act_raw.setShortcut(QKeySequence("Ctrl+Shift+O"))
        act_raw.setStatusTip("Open raw text in the paste area")
        act_raw.triggered.connect(self._open_raw_text)
        file_menu.addAction(act_raw)

        file_menu.addSeparator()

        act_save = QActionType("Save Project...", self)
        act_save.setShortcut(QKeySequence("Ctrl+S"))
        act_save.setStatusTip("Save current project settings")
        act_save.triggered.connect(self._save_project)
        file_menu.addAction(act_save)

        act_load = QActionType("Open Project...", self)
        act_load.setShortcut(QKeySequence("Ctrl+L"))
        act_load.setStatusTip("Load a saved project")
        act_load.triggered.connect(self._load_project)
        file_menu.addAction(act_load)

        file_menu.addSeparator()

        # Recent files submenu
        self.recent_menu = QMenu("Recent Files", self)
        self._update_recent_menu()
        file_menu.addMenu(self.recent_menu)

        file_menu.addSeparator()

        act_exit = QActionType("Exit", self)
        act_exit.setShortcut(QKeySequence("Alt+F4"))
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # ── Edit menu ──────────────────────────────────────────
        edit_menu = menubar.addMenu("&Edit")

        act_meta = QActionType("Metadata...", self)
        act_meta.triggered.connect(self._open_metadata)
        edit_menu.addAction(act_meta)

        act_pron = QActionType("Pronunciation Dictionary...", self)
        act_pron.triggered.connect(self._open_pronunciation)
        edit_menu.addAction(act_pron)

        act_voice_map = QActionType("Voice Map...", self)
        act_voice_map.triggered.connect(self._open_speaker_map)
        edit_menu.addAction(act_voice_map)

        act_quality = QActionType("Project Settings...", self)
        act_quality.triggered.connect(self._open_quality_preset)
        edit_menu.addAction(act_quality)

        # ── View menu ──────────────────────────────────────────
        view_menu = menubar.addMenu("&View")

        act_history = QActionType("Generation History...", self)
        act_history.setShortcut(QKeySequence("Ctrl+H"))
        act_history.setStatusTip("View generation history")
        act_history.triggered.connect(self._open_generation_history)
        view_menu.addAction(act_history)

        view_menu.addSeparator()

        act_theme = QActionType("Toggle Theme", self)
        act_theme.setShortcut(QKeySequence("Ctrl+T"))
        act_theme.setStatusTip("Toggle dark/light theme")
        act_theme.triggered.connect(self._toggle_theme)
        view_menu.addAction(act_theme)

        act_timeline = QActionType("Chapter Timeline", self)
        act_timeline.setCheckable(True)
        act_timeline.setChecked(True)
        act_timeline.setStatusTip("Show/hide chapter timeline")
        act_timeline.triggered.connect(
            lambda checked: self.chapter_timeline.setVisible(checked)
        )
        view_menu.addAction(act_timeline)

        # ── Tools menu ─────────────────────────────────────────
        tools_menu = menubar.addMenu("&Tools")

        act_export = QActionType("Export Profile...", self)
        act_export.triggered.connect(self._open_export_profile)
        tools_menu.addAction(act_export)

        act_template = QActionType("Project Template...", self)
        act_template.triggered.connect(self._open_project_template)
        tools_menu.addAction(act_template)

        act_batch = QActionType("Batch Re-export...", self)
        act_batch.setStatusTip("Re-export with different settings")
        act_batch.triggered.connect(self._batch_reexport)
        tools_menu.addAction(act_batch)

        tools_menu.addSeparator()

        act_detect = QActionType("Detect Speakers", self)
        act_detect.triggered.connect(self._detect_speakers)
        tools_menu.addAction(act_detect)

        # ── Help menu ──────────────────────────────────────────
        help_menu = menubar.addMenu("&Help")
        act_about = QActionType("About", self)
        act_about.triggered.connect(self._about)
        help_menu.addAction(act_about)

    # ═════════════════════════════════════════════════════════════
    # HELPERS
    # ═════════════════════════════════════════════════════════════

    def _make_slider(self, label: str, default: str, unit: str):
        """Create a slider with label for rate/volume/pitch controls."""
        sl = QSlider(Qt.Horizontal)
        sl.setRange(-100, 100)
        sl.setValue(0)
        # Parse default like "+10%", "-5Hz"
        try:
            val = int(re.sub(r'[+%Hz]', '', default))
            sl.setValue(val)
        except (ValueError, TypeError):
            pass
        lbl = QLabel(f"{label} {sl.value()}{unit}")
        lbl.setMinimumWidth(100)
        sl.valueChanged.connect(
            lambda v, l=lbl, la=label, u=unit: l.setText(f"{la} {v:+d}{u}")
        )
        return sl, lbl

    def _source_text(self) -> str:
        """Return the active source — file path or raw text."""
        raw = self.raw_text.toPlainText().strip()
        if raw:
            return raw
        return self.file_edit.text().strip()

    def _is_raw_text(self) -> bool:
        """True when the active source is the pasted text area."""
        return bool(self.raw_text.toPlainText().strip())

    def _output_path(self) -> str:
        name = self.out_name.text().strip() or "audiobook"
        d = self.out_dir.text().strip() or self.default_output
        return os.path.join(d, name)

    def _output_format(self) -> str:
        idx = self.format_combo.currentIndex()
        return ["mp3", "wav", "m4b", "both", "all"][idx]

    def _build_engine_config(self) -> EngineConfig:
        """Collect UI state into AppConfig, then produce an EngineConfig."""
        self.config.backend = self.backend_combo.currentData()
        self.config.edge_voice = self.edge_voice.currentText()
        self.config.edge_rate = f"{self.edge_rate.value():+d}%"
        self.config.edge_volume = f"{self.edge_vol.value():+d}%"
        self.config.edge_pitch = f"{self.edge_pitch.value():+d}Hz"
        self.config.qwen_voice = self.qwen_voice.currentText()
        self.config.qwen_model_path = self.qwen_path_edit.text().strip()
        self.config.qwen_device = self.qwen_device.currentText()
        self.config.omnivoice_speaker_id = self.omni_speaker.value()
        self.config.omnivoice_model_path = self.omni_path_edit.text().strip()
        self.config.omnivoice_device = self.omni_device.currentText()
        self.config.dramabox_voice_style = self.dramabox_style.currentText()
        self.config.dramabox_model = self.dramabox_model.text().strip()
        self.config.dramabox_api_url = self.dramabox_api_url.text().strip()
        self.config.dramabox_api_key = self.dramabox_api_key.text().strip()
        self.config.use_ssml = (
            self.chk_ssml.isChecked()
            or self.chk_ssml_quality.isChecked()
            or self.chk_dramabox_ssml.isChecked()
        )
        self.config.concurrency = self.spin_workers.value()
        self.config.chunk_size = self.spin_chunk.value()
        self.config.output_format = self._output_format()
        self.config.normalize_audio = self.chk_norm.isChecked()
        self.config.trim_silence = self.chk_trim.isChecked()
        self.config.chapter_aware = self.chk_chapters.isChecked()
        self.config.multi_speaker = self.chk_multi.isChecked()
        self.config.auto_assign_on_detect = self.chk_auto_assign.isChecked()
        return EngineConfig.from_app_config(self.config)

    def _create_backend(self, backend_id: str) -> TTSBackend:
        """Create a TTSBackend instance for the given backend ID.

        Builds a config dict from the current AppConfig and delegates
        to TTSBackend.create(). Supports fallback chain: if the primary
        backend fails, tries Edge TTS as a fallback.
        """
        config_dict = {
            "edge_voice": self.config.edge_voice,
            "qwen_model_path": self.config.qwen_model_path,
            "qwen_device": self.config.qwen_device,
            "qwen_voice": self.config.qwen_voice,
            "omnivoice_model_path": self.config.omnivoice_model_path,
            "omnivoice_device": self.config.omnivoice_device,
            "omnivoice_speaker_id": self.config.omnivoice_speaker_id,
            "dramabox_api_url": self.config.dramabox_api_url,
            "dramabox_api_key": self.config.dramabox_api_key,
            "dramabox_voice_style": self.config.dramabox_voice_style,
            "dramabox_model": self.config.dramabox_model,
        }
        try:
            return TTSBackend.create(backend_id, config_dict)
        except Exception:
            # Fallback chain: try edge_tts if primary is not already edge
            if backend_id != "edge_tts":
                self.log(
                    '<font color="#f9e2af">⚠ Primary backend failed, '
                    'falling back to Edge TTS</font>'
                )
                return TTSBackend.create("edge_tts", config_dict)
            raise

    def save_state(self):
        """Persist window geometry and theme to config."""
        self.config.window_width = self.width()
        self.config.window_height = self.height()
        self.config.window_x = self.x()
        self.config.window_y = self.y()
        self.config.window_maximized = self.isMaximized()
        self.config.theme = self._current_theme

    # ═════════════════════════════════════════════════════════════
    # THEME
    # ═════════════════════════════════════════════════════════════

    def _toggle_theme(self):
        """Toggle between dark and light themes and persist preference."""
        if self._current_theme == "dark":
            self._current_theme = "light"
        else:
            self._current_theme = "dark"
        QApplication.instance().setStyleSheet(
            get_stylesheet(self._current_theme)
        )
        self.btn_theme_toggle.setText(
            "☀️  Light" if self._current_theme == "dark" else "🌙  Dark"
        )
        self.config.theme = self._current_theme
        self.log(
            f'<font color="#cba6f7">Theme switched to {self._current_theme}</font>'
        )

    # ═════════════════════════════════════════════════════════════
    # UI CALLBACKS
    # ═════════════════════════════════════════════════════════════

    def _on_backend_changed(self, backend_id: str):
        """Update UI when the TTS backend selection changes."""
        # Map backend IDs to stack indices
        backend_map = {
            "edge_tts": 0,
            "qwen_local": 1,
            "omnivoice_local": 2,
            "dramabox": 3,
        }
        idx = backend_map.get(backend_id, 0)
        self.backend_stack.setCurrentIndex(idx)

        # Update description
        info = BACKEND_REGISTRY.get(backend_id)
        if info:
            parts = [f"{info.icon}  <b>{info.name}</b>"]
            parts.append(info.description)
            tags = []
            if info.is_local:
                tags.append("🖥️ Local")
            else:
                tags.append("☁️ Cloud")
            if info.supports_ssml:
                tags.append("SSML")
            if info.supports_multi_voice:
                tags.append("Multi-Voice")
            if info.requires_config:
                tags.append("⚠️ Requires config")
            parts.append("  ·  ".join(tags))
            self.backend_desc.setText("<br>".join(parts))
        else:
            self.backend_desc.setText("")

        # Show/hide Configure button for non-Edge backends
        self.btn_configure.setVisible(backend_id != "edge_tts")

        # Reset voice pools to backend defaults
        self.config.reset_voice_pools_to_backend_defaults(backend_id)

        # Update backend status indicator
        self._update_backend_status()

    def _update_backend_status(self):
        """Update the backend status indicator (online/offline/needs-config)."""
        bid = self.backend_combo.currentData() if hasattr(self, 'backend_combo') else self.config.backend
        info = BACKEND_REGISTRY.get(bid)

        if not info:
            self.backend_status_indicator.setText("⚪ Unknown")
            self.backend_status_indicator.setStyleSheet(
                "color: #6c7086; font-size: 11px; padding: 2px 8px; "
                "border-radius: 3px;"
            )
            return

        if bid == "edge_tts":
            # Edge TTS is always available (cloud, no config needed)
            self.backend_status_indicator.setText("🟢 Online — Ready")
            self.backend_status_indicator.setStyleSheet(
                "color: #a6e3a1; font-size: 11px; padding: 2px 8px; "
                "background-color: rgba(166,227,161,0.08); "
                "border: 1px solid rgba(166,227,161,0.2); border-radius: 3px;"
            )
        elif info.requires_config:
            # Check if configured
            configured = False
            if bid == "qwen_local":
                configured = bool(self.config.qwen_model_path)
            elif bid == "omnivoice_local":
                configured = bool(self.config.omnivoice_model_path)
            elif bid == "dramabox":
                configured = bool(self.config.dramabox_api_url)

            if configured:
                self.backend_status_indicator.setText("🟡 Configured — Needs verification")
                self.backend_status_indicator.setStyleSheet(
                    "color: #f9e2af; font-size: 11px; padding: 2px 8px; "
                    "background-color: rgba(249,226,175,0.08); "
                    "border: 1px solid rgba(249,226,175,0.2); border-radius: 3px;"
                )
            else:
                self.backend_status_indicator.setText("🔴 Needs Configuration")
                self.backend_status_indicator.setStyleSheet(
                    "color: #f38ba8; font-size: 11px; padding: 2px 8px; "
                    "background-color: rgba(243,139,168,0.08); "
                    "border: 1px solid rgba(243,139,168,0.2); border-radius: 3px;"
                )
        else:
            self.backend_status_indicator.setText("🟢 Available")
            self.backend_status_indicator.setStyleSheet(
                "color: #a6e3a1; font-size: 11px; padding: 2px 8px; "
                "background-color: rgba(166,227,161,0.08); "
                "border: 1px solid rgba(166,227,161,0.2); border-radius: 3px;"
            )

    def _on_quality_changed(self):
        key = self.quality_combo.currentData()
        self.config.apply_preset(key)
        self._update_preset_badge()
        self._update_preset_desc()
        self._update_quality_preset_label()

    def _update_preset_badge(self):
        key = self.config.quality_preset
        preset = QUALITY_PRESETS.get(key, QUALITY_PRESETS["acx_audible"])
        self.preset_badge.setText(preset.name)

    def _update_preset_desc(self):
        key = self.quality_combo.currentData()
        preset = QUALITY_PRESETS.get(key, QUALITY_PRESETS["acx_audible"])
        self.preset_desc.setText(
            f"{preset.description}<br>"
            f"<small>{preset.target_lufs} LUFS · {preset.true_peak} dB TP · "
            f"{preset.sample_rate} Hz · {preset.bitrate}</small>"
        )

    def _update_quality_preset_label(self):
        key = self.config.quality_preset
        preset = QUALITY_PRESETS.get(key, QUALITY_PRESETS["acx_audible"])
        self.quality_preset_label.setText(
            f"Quality: {preset.name} — {preset.target_lufs} LUFS, "
            f"{preset.sample_rate} Hz, {preset.bitrate}"
        )

    def _update_profile_badge(self):
        """Update the export profile badge in the status bar."""
        current = getattr(self.config, 'export_profile', 'acx')
        profile = EXPORT_PROFILES.get(current)
        if profile:
            self._profile_badge.setText(f"📦 {profile.name}")
        else:
            self._profile_badge.setText("📦 Custom")

    def _on_recent_file(self):
        if hasattr(self, 'recent_combo'):
            data = self.recent_combo.currentData()
            if data and Path(data).exists():
                self.file_edit.setText(data)

    def _auto_name(self):
        src = self.file_edit.text().strip()
        if src and Path(src).exists():
            self.out_name.setText(Path(src).stem)
        elif self.raw_text.toPlainText().strip():
            first = self.raw_text.toPlainText().strip().split('\n')[0][:40]
            self.out_name.setText(
                re.sub(r'[^\w\s-]', '', first).strip() or "audiobook"
            )

    def _update_stats(self):
        """Update word count, chapters, speakers, and reading level stats."""
        src = self._source_text()
        if not src:
            self.stats_label.setText("")
            self.reading_level_label.setText("")
            self.reading_level_label.setVisible(False)
            return
        try:
            stats = TextProcessor.compute_stats(src, self._is_raw_text())
            parts = [
                f"📝 {stats.word_count:,} words",
                f"⏱ ~{stats.estimated_str}",
                f"📖 {stats.chapter_count} ch.",
            ]
            if stats.speakers_detected:
                parts.append(f"🗣 {stats.speakers_detected} speakers")
            self.stats_label.setText("  ·  ".join(parts))

            # Reading level stats
            rl = TextProcessor.reading_level(
                src if self._is_raw_text() else src
            )
            self.reading_level_label.setVisible(True)
            self.reading_level_label.setText(
                f"📊 Flesch-Kincaid Grade: {rl.flesch_kincaid_grade:.1f}  ·  "
                f"Complexity: {rl.complexity_score:.0f}/100  ·  "
                f"Listening: ~{rl.estimated_listening_minutes:.0f} min"
            )

            # Update chapter timeline
            if stats.chapter_count > 0:
                self.chapter_timeline.set_chapters(stats.chapter_count)

        except Exception:
            self.stats_label.setText("")
            self.reading_level_label.setVisible(False)

    # ═════════════════════════════════════════════════════════════
    # FILE / BROWSE
    # ═════════════════════════════════════════════════════════════

    def _new_project(self):
        """Reset all fields for a new project."""
        self.file_edit.clear()
        self.raw_text.clear()
        self.out_dir.clear()
        self.out_name.clear()
        self.format_combo.setCurrentIndex(0)
        self._detected_speakers = []
        self._character_registry = None
        self.speaker_count_badge.setText("")
        self.chapter_timeline.set_chapters(0)
        self.log_box.clear()
        self.pbar.setValue(0)
        self.status_lbl.setText("Ready")
        self.statusBar().showMessage("New project started", 2000)

    def _load_source_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Open Source Document", "",
            "Documents (*.txt *.pdf *.docx *.epub *.html *.htm *.md);;"
            "All Files (*)"
        )
        if f:
            self.file_edit.setText(f)
            self.config.add_recent_file(f)
            self._update_stats()
            self._update_recent_menu()

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self.out_dir.setText(d)

    def _open_raw_text(self):
        """Open a file and put its content into the raw text area."""
        f, _ = QFileDialog.getOpenFileName(
            self, "Open Raw Text", "",
            "Text Files (*.txt *.md);;All Files (*)"
        )
        if f:
            try:
                content = Path(f).read_text(encoding='utf-8', errors='replace')
                self.raw_text.setPlainText(content)
                self.config.add_recent_file(f)
                self._update_recent_menu()
            except Exception as exc:
                QMessageBox.warning(
                    self, "Open Raw Text", f"Failed to read file:\n{exc}"
                )

    # ═════════════════════════════════════════════════════════════
    # BACKEND DIALOGS
    # ═════════════════════════════════════════════════════════════

    def _configure_backend(self):
        """Open LocalModelConfigDialog for the current backend."""
        bid = self.backend_combo.currentData()
        if bid == "edge_tts":
            QMessageBox.information(
                self, "Configure Backend",
                "Edge TTS requires no additional configuration.\n"
                "Use the Voice Search button to browse 400+ voices."
            )
            return

        dlg = LocalModelConfigDialog(self.config.to_dict(), bid, self)
        if dlg.exec() == QDialog.Accepted:
            new_cfg = dlg.get_config()
            for k, v in new_cfg.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
            # Update status labels
            if bid == "qwen_local":
                self.qwen_status.setText(
                    "Model: " + (self.config.qwen_model_path or "Not Set")
                )
                self.qwen_path_edit.setText(self.config.qwen_model_path)
                self.qwen_device.setCurrentText(self.config.qwen_device)
            elif bid == "omnivoice_local":
                self.omni_status.setText(
                    "Model: " + (self.config.omnivoice_model_path or "Not Set")
                )
                self.omni_path_edit.setText(self.config.omnivoice_model_path)
                self.omni_device.setCurrentText(self.config.omnivoice_device)
            elif bid == "dramabox":
                if self.config.dramabox_api_url:
                    self.dramabox_status.setText(
                        f"API: {self.config.dramabox_api_url}"
                    )
                else:
                    self.dramabox_status.setText("Mode: Local inference")
                self.dramabox_api_url.setText(self.config.dramabox_api_url)
                self.dramabox_api_key.setText(self.config.dramabox_api_key)
            # Refresh status indicator
            self._update_backend_status()

    def _search_voices(self):
        """Open VoiceSearchDialog for the current backend."""
        bid = self.backend_combo.currentData()
        try:
            backend = self._create_backend(bid)
            current_voice = self.edge_voice.currentText()
            dlg = VoiceSearchDialog(backend, current_voice, self)
            if dlg.exec() == QDialog.Accepted:
                selected = dlg.get_selected()
                self.edge_voice.setCurrentText(selected)
        except Exception as exc:
            QMessageBox.warning(
                self, "Voice Search", f"Failed to load voices:\n{exc}"
            )

    # ═════════════════════════════════════════════════════════════
    # SPEAKER DETECTION & VOICE ASSIGNMENT
    # ═════════════════════════════════════════════════════════════

    def _detect_speakers(self):
        """Run speaker detection in a background thread and update UI."""
        src = self._source_text()
        if not src:
            QMessageBox.information(
                self, "Detect Speakers", "Load source text first."
            )
            return

        self.btn_detect.setEnabled(False)
        self.btn_detect.setText("🔍  Detecting...")
        self.statusBar().showMessage("Running speaker detection...")

        self._detect_worker = _SpeakerDetectWorker(
            src, self._is_raw_text(), self
        )
        self._detect_worker.finished.connect(self._on_speakers_detected)
        self._detect_worker.error.connect(self._on_speaker_detect_error)
        self._detect_worker.start()

    def _on_speakers_detected(self, speakers: list, registry):
        """Handle speaker detection results."""
        self._detected_speakers = speakers
        self._character_registry = registry
        self.speaker_count_badge.setText(f"{len(speakers)} speakers")

        self.btn_detect.setEnabled(True)
        self.btn_detect.setText("🔍  Detect Speakers")
        self.statusBar().showMessage(
            f"Detected {len(speakers)} speakers", 4000
        )

        # Auto-assign voices if enabled
        if self.chk_auto_assign.isChecked() and self.chk_multi.isChecked():
            speaker_genders = []
            for name in speakers:
                char = registry.characters.get(
                    name, Character(name=name)
                )
                speaker_genders.append((name, char.gender))
            voice_map = self.config.auto_assign_voices(speaker_genders)
            self.config.voice_map = voice_map

        self.log(
            f'<font color="#a6e3a1">Detected {len(speakers)} speakers: '
            f'{", ".join(speakers[:8])}'
            f'{"..." if len(speakers) > 8 else ""}</font>'
        )

    def _on_speaker_detect_error(self, msg: str):
        self.btn_detect.setEnabled(True)
        self.btn_detect.setText("🔍  Detect Speakers")
        QMessageBox.warning(
            self, "Speaker Detection", f"Detection failed:\n{msg}"
        )

    def _assign_voices(self):
        """Open VoiceAssignmentDialog with detected speakers and registry."""
        src = self._source_text()
        if not src:
            QMessageBox.information(
                self, "Voice Assignment",
                "Load source text and detect speakers first."
            )
            return

        # Detect speakers if not already done
        if not self._detected_speakers:
            self._detect_speakers()
            if not self._detected_speakers:
                return

        bid = self.backend_combo.currentData()
        try:
            backend = self._create_backend(bid)
            dlg = VoiceAssignmentDialog(
                self.config.to_dict(),
                self._detected_speakers,
                backend,
                self._character_registry,
                self,
            )
            if dlg.exec() == QDialog.Accepted:
                new_cfg = dlg.get_config()
                for k, v in new_cfg.items():
                    if hasattr(self.config, k):
                        setattr(self.config, k, v)
                self.log(
                    f'<font color="#89b4fa">Voice assignments updated: '
                    f'{len(self.config.voice_map)} speakers mapped</font>'
                )
        except Exception as exc:
            QMessageBox.warning(
                self, "Voice Assignment",
                f"Failed to open voice assignment:\n{exc}"
            )

    def _open_speaker_map(self):
        """Open the legacy SpeakerMapDialog."""
        src = self._source_text()
        if not src:
            QMessageBox.information(
                self, "Speaker Map", "Load source text first."
            )
            return

        # Detect speakers if not already done
        if not self._detected_speakers:
            self._detect_speakers()

        bid = self.backend_combo.currentData()
        try:
            backend = self._create_backend(bid)
            dlg = SpeakerMapDialog(
                self.config.to_dict(),
                self._detected_speakers,
                backend,
                self._character_registry,
                self,
            )
            if dlg.exec() == QDialog.Accepted:
                new_cfg = dlg.get_config()
                for k, v in new_cfg.items():
                    if hasattr(self.config, k):
                        setattr(self.config, k, v)
        except Exception as exc:
            QMessageBox.warning(
                self, "Speaker Map",
                f"Failed to open speaker map:\n{exc}"
            )

    # ═════════════════════════════════════════════════════════════
    # OTHER DIALOGS
    # ═════════════════════════════════════════════════════════════

    def _open_metadata(self):
        dlg = MetadataDialog(self.config.to_dict(), self)
        if dlg.exec() == QDialog.Accepted:
            for k, v in dlg.get_config().items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)

    def _open_pronunciation(self):
        dlg = PronunciationDialog(self.config.pronunciation_dict, self)
        if dlg.exec() == QDialog.Accepted:
            self.config.pronunciation_dict = dlg.get_dict()

    def _open_quality_preset(self):
        dlg = QualityPresetDialog(self.config.to_dict(), self)
        if dlg.exec() == QDialog.Accepted:
            for k, v in dlg.get_config().items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
            idx = self.quality_combo.findData(self.config.quality_preset)
            if idx >= 0:
                self.quality_combo.setCurrentIndex(idx)
            self._update_preset_badge()
            self._update_preset_desc()
            self._update_quality_preset_label()

    def _open_export_profile(self):
        """Open the ExportProfileDialog."""
        dlg = ExportProfileDialog(self.config.to_dict(), self)
        if dlg.exec() == QDialog.Accepted:
            for k, v in dlg.get_config().items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
            self._update_profile_badge()
            self.log(
                '<font color="#f9e2af">Export profile updated</font>'
            )

    def _open_project_template(self):
        """Open the ProjectTemplateDialog."""
        dlg = ProjectTemplateDialog(self.config.to_dict(), self)
        if dlg.exec() == QDialog.Accepted:
            for k, v in dlg.get_config().items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
            # Apply template settings to UI
            self._apply_config_to_ui()
            self.log(
                '<font color="#cba6f7">Project template applied</font>'
            )

    def _open_generation_history(self):
        """Open the GenerationHistoryDialog."""
        dlg = GenerationHistoryDialog(self.config, self)
        dlg.exec()

    def _about(self):
        AboutDialog(self).exec()

    # ═════════════════════════════════════════════════════════════
    # PRONUNCIATION AUTO-SUGGEST
    # ═════════════════════════════════════════════════════════════

    def _auto_suggest_pronunciations(self):
        """Run pronunciation auto-suggest in a background thread."""
        src = self._source_text()
        if not src:
            QMessageBox.information(
                self, "Auto-Suggest", "Load source text first."
            )
            return

        self.btn_suggest_pron.setEnabled(False)
        self.btn_suggest_pron.setText("💡  Analyzing...")
        self.statusBar().showMessage("Analyzing text for pronunciation suggestions...")

        self._pron_suggest_worker = _PronounSuggestWorker(src, self)
        self._pron_suggest_worker.finished.connect(
            self._on_pron_suggestions_ready
        )
        self._pron_suggest_worker.error.connect(
            self._on_pron_suggest_error
        )
        self._pron_suggest_worker.start()

    def _on_pron_suggestions_ready(self, suggestions: Dict[str, str]):
        """Display pronunciation suggestions inline."""
        self.btn_suggest_pron.setEnabled(True)
        self.btn_suggest_pron.setText("💡  Auto-Suggest")
        self.statusBar().showMessage(
            f"Found {len(suggestions)} pronunciation suggestions", 3000
        )

        if not suggestions:
            self.pron_suggestions_label.setText(
                "✅ No unusual words detected — text looks pronunciation-friendly!"
            )
            self.pron_suggestions_label.setVisible(True)
            return

        # Show top suggestions inline
        items = list(suggestions.items())[:12]
        parts = [f"<b>{word}</b> → {phon}" for word, phon in items]
        html = "💡 <b>Suggestions:</b> " + "  ·  ".join(parts)
        if len(suggestions) > 12:
            html += f"  ·  ... and {len(suggestions) - 12} more"
        self.pron_suggestions_label.setText(html)
        self.pron_suggestions_label.setVisible(True)

        # Auto-add to pronunciation dict
        added = 0
        for word, phon in suggestions.items():
            if word not in self.config.pronunciation_dict:
                self.config.pronunciation_dict[word] = phon
                added += 1

        if added > 0:
            self.log(
                f'<font color="#f9e2af">Auto-added {added} pronunciation '
                f'entries</font>'
            )

    def _on_pron_suggest_error(self, msg: str):
        self.btn_suggest_pron.setEnabled(True)
        self.btn_suggest_pron.setText("💡  Auto-Suggest")
        QMessageBox.warning(
            self, "Auto-Suggest", f"Pronunciation analysis failed:\n{msg}"
        )

    # ═════════════════════════════════════════════════════════════
    # BATCH RE-EXPORT
    # ═════════════════════════════════════════════════════════════

    def _batch_reexport(self):
        """Re-export the last generated audiobook with different settings."""
        src = self._source_text()
        if not src:
            QMessageBox.information(
                self, "Batch Re-export", "No source loaded for re-export."
            )
            return
        self.log(
            '<font color="#f9e2af">Batch re-export initiated with current '
            'settings...</font>'
        )
        # Use the same start flow — user should adjust settings first
        self._start_generation()

    # ═════════════════════════════════════════════════════════════
    # PROJECT SAVE / LOAD
    # ═════════════════════════════════════════════════════════════

    def _save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "Audiobook Project (*.abp)"
        )
        if path:
            if not path.endswith(".abp"):
                path += ".abp"
            proj = AudiobookProject.from_state(
                name=self.out_name.text(),
                source_file=self.file_edit.text(),
                source_text=self.raw_text.toPlainText(),
                is_raw_text=self._is_raw_text(),
                output_path=self._output_path(),
                config=self.config,
            )
            proj.save(Path(path))
            self.statusBar().showMessage(f"Project saved: {path}", 3000)

    def _load_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "Audiobook Project (*.abp)"
        )
        if path:
            proj = AudiobookProject.load(Path(path))
            if proj.source_file:
                self.file_edit.setText(proj.source_file)
            if proj.source_text:
                self.raw_text.setPlainText(proj.source_text)
            if proj.config_snapshot:
                self.config = AppConfig.from_dict(proj.config_snapshot)
            self._apply_config_to_ui()
            self.statusBar().showMessage(f"Project loaded: {path}", 3000)
            self._update_stats()
            self._update_preset_badge()
            self._update_quality_preset_label()

    def _apply_config_to_ui(self):
        """Apply the current AppConfig to all UI controls."""
        # Backend
        idx = self.backend_combo.findData(self.config.backend)
        if idx >= 0:
            self.backend_combo.setCurrentIndex(idx)

        # Edge
        self.edge_voice.setCurrentText(self.config.edge_voice)
        self.qwen_voice.setCurrentText(self.config.qwen_voice)
        self.qwen_path_edit.setText(self.config.qwen_model_path)
        self.qwen_device.setCurrentText(self.config.qwen_device)
        self.omni_speaker.setValue(self.config.omnivoice_speaker_id)
        self.omni_path_edit.setText(self.config.omnivoice_model_path)
        self.omni_device.setCurrentText(self.config.omnivoice_device)
        self.dramabox_api_url.setText(self.config.dramabox_api_url)
        self.dramabox_api_key.setText(self.config.dramabox_api_key)
        self.dramabox_style.setCurrentText(self.config.dramabox_voice_style)
        self.dramabox_model.setText(self.config.dramabox_model)

        # Output
        self.out_dir.setText(self.config.output_dir)
        fmt_map = {"mp3": 0, "wav": 1, "m4b": 2, "both": 3, "all": 4}
        self.format_combo.setCurrentIndex(
            fmt_map.get(self.config.output_format, 0)
        )

        # Quality
        qidx = self.quality_combo.findData(self.config.quality_preset)
        if qidx >= 0:
            self.quality_combo.setCurrentIndex(qidx)

        # Checkboxes
        self.chk_norm.setChecked(self.config.normalize_audio)
        self.chk_trim.setChecked(self.config.trim_silence)
        self.chk_chapters.setChecked(self.config.chapter_aware)
        self.chk_ssml.setChecked(self.config.use_ssml)
        self.chk_ssml_quality.setChecked(self.config.use_ssml)
        self.chk_dramabox_ssml.setChecked(self.config.use_ssml)
        self.chk_multi.setChecked(self.config.multi_speaker)
        self.chk_auto_assign.setChecked(self.config.auto_assign_on_detect)

        # Performance
        self.spin_workers.setValue(self.config.concurrency)
        self.spin_chunk.setValue(self.config.chunk_size)

    def _update_recent_menu(self):
        """Rebuild the Recent Files submenu from config."""
        self.recent_menu.clear()
        if not self.config.recent_files:
            act = self.recent_menu.addAction("No recent files")
            act.setEnabled(False)
            return
        for rf in self.config.recent_files:
            if Path(rf).exists():
                act = self.recent_menu.addAction(Path(rf).name)
                act.setToolTip(rf)
                act.setData(rf)
                act.triggered.connect(self._on_recent_menu_action)

    def _on_recent_menu_action(self):
        act = self.sender()
        if act and act.data():
            self.file_edit.setText(act.data())
            self._update_stats()

    # ═════════════════════════════════════════════════════════════
    # DRAG & DROP
    # ═════════════════════════════════════════════════════════════

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            f = url.toLocalFile()
            if f:
                self.file_edit.setText(f)
                self.drop_label.setText(f"📂  {Path(f).name}")
                self.config.add_recent_file(f)
                self._update_recent_menu()
                self._update_stats()
                break

    # ═════════════════════════════════════════════════════════════
    # LOGGING
    # ═════════════════════════════════════════════════════════════

    def log(self, msg: str):
        self.log_box.append(msg)

    # ═════════════════════════════════════════════════════════════
    # ENGINE CONTROL — GENERATION
    # ═════════════════════════════════════════════════════════════

    def _start_generation(self):
        """Validate config, create backend, build EngineConfig, start AudiobookEngine."""
        src = self._source_text()
        if not src:
            QMessageBox.warning(
                self, "No Source",
                "Load a source document or paste text first."
            )
            return

        # Build engine config (also updates self.config from UI)
        cfg = self._build_engine_config()

        # Validate source file exists (for file-based source)
        if not self._is_raw_text() and not Path(src).exists():
            QMessageBox.warning(
                self, "Source Missing",
                f"Source file not found:\n{src}"
            )
            return

        # Create backend
        try:
            backend = self._create_backend(cfg.backend_id)
        except Exception as exc:
            QMessageBox.critical(
                self, "Backend Error",
                f"Failed to create TTS backend:\n{exc}"
            )
            return

        # Validate backend config
        errors = backend.validate_config()
        if errors:
            QMessageBox.critical(
                self, "Configuration Error", "\n".join(errors)
            )
            return

        # Ensure output directory exists
        out = self._output_path()
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

        # Create and start engine
        self.engine = AudiobookEngine(
            src, out, cfg, backend, self._is_raw_text()
        )
        self.engine.progress.connect(self._on_engine_progress)
        self.engine.log.connect(self._on_engine_log)
        self.engine.chapter_progress.connect(self._on_chapter_progress)
        self.engine.finished.connect(self._on_engine_finished)

        # Reset chapter timeline
        self.chapter_timeline.reset()

        # Build per-speaker voice configs for the engine
        # (The pipeline reads these from EngineConfig)

        # Update UI state
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        self.btn_preview.setEnabled(False)
        self.pbar.setValue(0)
        backend_name = cfg.backend_id
        info = BACKEND_REGISTRY.get(cfg.backend_id)
        if info:
            backend_name = info.name
        self.log(
            f'<font color="#89b4fa"><b>Starting generation with '
            f'{backend_name}...</b></font>'
        )

        self.engine.start()

    def _pause_generation(self):
        """Toggle pause/resume on the running engine."""
        if not self.engine:
            return
        if self.engine.is_paused:
            self.engine.resume()
            self.btn_pause.setText("⏸  Pause")
            self.log('<font color="#f9e2af">▶ Resumed generation</font>')
        else:
            self.engine.pause()
            self.btn_pause.setText("▶  Resume")
            self.log('<font color="#f9e2af">⏸ Paused generation</font>')

    def _cancel_generation(self):
        """Cancel the running engine."""
        if self.engine:
            self.engine.cancel()
            self.log('<font color="#f38ba8">⏹ Cancelled generation</font>')

    def _on_engine_progress(self, pct: int, msg: str):
        self.pbar.setValue(pct)
        self.status_lbl.setText(msg)
        self.gen_stats_label.setText(
            f"Progress: {pct}%  ·  {msg}"
        )

    def _on_engine_log(self, msg: str):
        self.log(msg)

    def _on_chapter_progress(self, current: int, total: int):
        """Update chapter timeline and status during generation."""
        self.statusBar().showMessage(f"Chapter {current}/{total}")
        if total > 0:
            # Ensure timeline has the right number of chapters
            if self.chapter_timeline._chapter_count != total:
                self.chapter_timeline.set_chapters(total)
            # Mark previous chapters as completed
            if current > 1:
                for i in range(current - 1):
                    self.chapter_timeline.mark_completed(i)
            # Mark current as active
            self.chapter_timeline.set_active(current - 1)

    def _on_engine_finished(self, ok: bool, msg: str):
        """Handle engine completion — success or failure."""
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.btn_preview.setEnabled(True)
        self.btn_pause.setText("⏸  Pause")
        self.engine = None

        if ok:
            self.pbar.setValue(100)
            self.status_lbl.setText("✅ Complete!")
            self.log(
                '<font color="#a6e3a1"><b>Audiobook produced successfully!'
                '</b></font>'
            )
            self.log(msg)
            # Mark all chapters as completed
            for i in range(self.chapter_timeline._chapter_count):
                self.chapter_timeline.mark_completed(i)
            self.gen_stats_label.setText("✅ Generation complete")
        else:
            self.log(f'<font color="red"><b>Failed:</b></font> {msg}')
            self.gen_stats_label.setText("❌ Generation failed")

    # ═════════════════════════════════════════════════════════════
    # PARTIAL CHAPTER REGENERATION
    # ═════════════════════════════════════════════════════════════

    def _on_chapter_regen_requested(self, chapter_idx: int):
        """Handle request to re-generate a single chapter."""
        src = self._source_text()
        if not src:
            QMessageBox.information(
                self, "Re-generate Chapter", "Load source text first."
            )
            return

        if self.engine and self.engine.isRunning():
            QMessageBox.warning(
                self, "Re-generate Chapter",
                "Cannot re-generate while generation is running."
            )
            return

        reply = QMessageBox.question(
            self, "Re-generate Chapter",
            f"Re-generate chapter {chapter_idx + 1}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            cfg = self._build_engine_config()
            backend = self._create_backend(cfg.backend_id)

            from pipeline import PartialRegenEngine
            self._partial_engine = PartialRegenEngine(
                source=src,
                output_path=self._output_path(),
                config=cfg,
                backend=backend,
                chapter_index=chapter_idx,
                is_raw_text=self._is_raw_text(),
                parent=self,
            )
            self._partial_engine.progress.connect(self._on_engine_progress)
            self._partial_engine.log.connect(self._on_engine_log)
            self._partial_engine.finished.connect(
                self._on_partial_regen_finished
            )

            self.chapter_timeline.set_active(chapter_idx)
            self.log(
                f'<font color="#cba6f7">Re-generating chapter '
                f'{chapter_idx + 1}...</font>'
            )
            self._partial_engine.start()

        except Exception as exc:
            QMessageBox.critical(
                self, "Re-generate Chapter",
                f"Failed to start partial regeneration:\n{exc}"
            )

    def _on_partial_regen_finished(self, ok: bool, msg: str):
        """Handle partial chapter regeneration completion."""
        if ok:
            self.log(
                '<font color="#a6e3a1">Chapter re-generation complete!</font>'
            )
        else:
            self.log(
                f'<font color="red">Chapter re-generation failed: {msg}</font>'
            )
        self._partial_engine = None
        self.status_lbl.setText("Ready")

    # ═════════════════════════════════════════════════════════════
    # PREVIEW
    # ═════════════════════════════════════════════════════════════

    def _preview_voice(self):
        """Generate a short audio preview of the first chunk with waveform animation."""
        src = self._source_text()
        if not src:
            return
        try:
            cfg = self._build_engine_config()
            backend = self._create_backend(cfg.backend_id)

            # Get first chunk
            chunks = list(TextProcessor.chunk_generator(
                src, cfg.chunk_size, self._is_raw_text(), cfg.multi_speaker,
                pronunciation_dict=(
                    cfg.pronunciation_dict if cfg.pronunciation_dict else None
                ),
            ))
            if not chunks:
                return
            chunk = chunks[0]
            out = os.path.join(tempfile.gettempdir(), "audiobook_preview.mp3")

            # Map segments to voices
            segments = [
                (text, cfg.voice_map.get(speaker, cfg.voice))
                for text, speaker in chunk.segments
            ]

            # Show waveform animation
            self._preview_pbar.setVisible(True)
            self._preview_pbar.setValue(0)
            self._preview_anim_timer.start()

            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    backend.synthesize_segments(
                        segments, out, voice=cfg.voice
                    )
                )
                if result and os.path.exists(out):
                    self._play(out)
                    self.log(
                        '<font color="#89b4fa">Preview generated</font>'
                    )
            finally:
                self._preview_anim_timer.stop()
                self._preview_pbar.setVisible(False)
                loop.run_until_complete(backend.close())
                loop.close()
        except Exception as exc:
            self._preview_anim_timer.stop()
            self._preview_pbar.setVisible(False)
            self.log(f'<font color="red">Preview error: {exc}</font>')

    def _play(self, path: str):
        """Play an audio file using QMediaPlayer."""
        try:
            if not self._player:
                self._player = QMediaPlayer()
                self._audio_output = QAudioOutput()
                self._player.setAudioOutput(self._audio_output)
            self._player.setSource(QUrl.fromLocalFile(path))
            self._audio_output.setVolume(0.8)
            self._player.play()
        except Exception as exc:
            self.log(f'<font color="red">Playback error: {exc}</font>')

    def _animate_preview_pbar(self):
        """Animate the preview progress bar to simulate waveform."""
        val = self._preview_pbar.value() + 5
        if val > 100:
            val = 0
        self._preview_pbar.setValue(val)

    # ═════════════════════════════════════════════════════════════
    # KEYBOARD SHORTCUTS
    # ═════════════════════════════════════════════════════════════

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts not covered by menu accelerators."""
        modifiers = event.modifiers()
        key = event.key()

        # Ctrl+Enter: Start generation
        if modifiers == Qt.ControlModifier and key == Qt.Key_Return:
            if self.btn_start.isEnabled():
                self._start_generation()
            return

        # Ctrl+Shift+P: Pause/Resume
        if (modifiers & Qt.ControlModifier and
                modifiers & Qt.ShiftModifier and key == Qt.Key_P):
            if self.btn_pause.isEnabled():
                self._pause_generation()
            return

        # Ctrl+P: Preview
        if modifiers == Qt.ControlModifier and key == Qt.Key_P:
            if self.btn_preview.isEnabled():
                self._preview_voice()
            return

        # Escape: Cancel
        if key == Qt.Key_Escape:
            if self.btn_cancel.isEnabled():
                self._cancel_generation()
            return

        super().keyPressEvent(event)

    # ═════════════════════════════════════════════════════════════
    # WINDOW CLOSE
    # ═════════════════════════════════════════════════════════════

    def closeEvent(self, event):
        """Save state and clean up on close."""
        self.save_state()

        # Stop engine if running
        if self.engine and self.engine.isRunning():
            self.engine.cancel()
            self.engine.wait(3000)

        # Stop partial engine if running
        if self._partial_engine and self._partial_engine.isRunning():
            self._partial_engine.cancel()
            self._partial_engine.wait(3000)

        # Stop media player
        if self._player:
            self._player.stop()

        # Stop preview animation
        self._preview_anim_timer.stop()

        super().closeEvent(event)
