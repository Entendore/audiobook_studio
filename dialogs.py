"""
Audiobook Studio Pro — Dialog Windows (v2 — Upgraded)

All dialog windows for the application: voice search, model configuration,
metadata editing, voice assignment, speaker mapping (card-based redesign),
pronunciation editing with auto-suggest, quality presets, export profiles,
project templates, generation history, voice preview, and the about box.

Catppuccin Mocha colour palette used throughout for custom widgets.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl
from PySide6.QtGui import QColor, QFont, QIcon
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from tts_backends import TTSBackend, VoiceInfo
from config import (
    QualityPreset,
    QUALITY_PRESETS,
    VoiceAssignmentStrategy,
    BACKEND_REGISTRY,
    BACKEND_VOICE_POOLS,
    EXPORT_PROFILES,
    PROJECT_TEMPLATES,
    SpeakerVoiceConfig,
    AppConfig,
    ExportProfile,
    ProjectTemplate,
    GenerationEntry,
)
from text_processor import Character


# ═══════════════════════════════════════════════════════════════════════
# Catppuccin Mocha Palette
# ═══════════════════════════════════════════════════════════════════════

class _C:
    """Catppuccin Mocha colour constants for styling."""
    BASE      = "#1e1e2e"
    MANTLE    = "#181825"
    CRUST     = "#11111b"
    SURFACE0  = "#313244"
    SURFACE1  = "#45475a"
    SURFACE2  = "#585b70"
    OVERLAY0  = "#6c7086"
    OVERLAY1  = "#7f849c"
    OVERLAY2  = "#9399b2"
    SUBTEXT0  = "#a6adc8"
    SUBTEXT1  = "#bac2de"
    TEXT      = "#cdd6f4"
    LAVENDER  = "#b4befe"
    BLUE      = "#89b4fa"
    SAPPHIRE  = "#74c7ec"
    SKY       = "#89dceb"
    TEAL      = "#94e2d5"
    GREEN     = "#a6e3a1"
    YELLOW    = "#f9e2af"
    PEACH     = "#fab387"
    MAROON    = "#eba0ac"
    RED       = "#f38ba8"
    MAUVE     = "#cba6f7"
    PINK      = "#f5c2e7"
    FLAMINGO  = "#f2cdcd"
    ROSEWATER = "#f5e0dc"


# ═══════════════════════════════════════════════════════════════════════
# Worker Threads
# ═══════════════════════════════════════════════════════════════════════

class VoiceLoadWorker(QThread):
    """Loads voices from a TTS backend asynchronously so the UI stays responsive."""

    finished = Signal(list)
    error = Signal(str)

    def __init__(self, backend: TTSBackend, parent=None):
        super().__init__(parent)
        self.backend = backend

    def run(self):
        loop = asyncio.new_event_loop()
        try:
            voices = loop.run_until_complete(self.backend.list_voices())
            self.finished.emit(voices)
        except Exception as exc:
            self.error.emit(str(exc))
            self.finished.emit([])
        finally:
            try:
                loop.run_until_complete(self.backend.close())
            except Exception:
                pass
            loop.close()


class VoicePreviewWorker(QThread):
    """Synthesises a short voice sample in a background thread using asyncio."""

    finished = Signal(str)    # path to generated audio
    error = Signal(str)

    def __init__(
        self,
        backend: TTSBackend,
        voice_id: str = "",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        text: str = "Hello, this is a sample of my voice.",
        parent=None,
    ):
        super().__init__(parent)
        self.backend = backend
        self.voice_id = voice_id
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self.text = text

    def run(self):
        loop = asyncio.new_event_loop()
        try:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False, prefix="voice_preview_"
            )
            output_path = tmp.name
            tmp.close()

            loop.run_until_complete(
                self.backend.synthesize(
                    text=self.text,
                    output_path=output_path,
                    voice=self.voice_id,
                    rate=self.rate,
                    volume=self.volume,
                    pitch=self.pitch,
                )
            )
            self.finished.emit(output_path)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            try:
                loop.run_until_complete(self.backend.close())
            except Exception:
                pass
            loop.close()


# ═══════════════════════════════════════════════════════════════════════
# Voice Search Dialog
# ═══════════════════════════════════════════════════════════════════════

class VoiceSearchDialog(QDialog):
    """Search and select a TTS voice from the full voice catalogue.

    Upgraded: added voice preview button, gender badges, and
    Catppuccin Mocha styling.
    """

    def __init__(
        self,
        backend: TTSBackend,
        current: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Search Voice")
        self.setMinimumSize(520, 580)
        self._voices: list = []
        self._current = current
        self._preview_worker: Optional[VoicePreviewWorker] = None
        self._preview_path: Optional[str] = None
        self._backend = backend
        self._player: Optional[QMediaPlayer] = None
        self._audio_output: Optional[QAudioOutput] = None

        root = QVBoxLayout(self)

        # --- search ---
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Filter:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Type to filter voices…")
        self.search_edit.textChanged.connect(self._filter)
        search_row.addWidget(self.search_edit, 1)
        root.addLayout(search_row)

        # --- gender filter ---
        gender_row = QHBoxLayout()
        gender_row.addWidget(QLabel("Gender:"))
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["All", "Male", "Female", "Neutral"])
        self.gender_combo.currentTextChanged.connect(self._filter)
        gender_row.addWidget(self.gender_combo)
        gender_row.addStretch()
        root.addLayout(gender_row)

        # --- list ---
        self.list_widget = QListWidget()
        self.list_widget.setSortingEnabled(True)
        self.list_widget.doubleClicked.connect(self.accept)
        root.addWidget(self.list_widget, 1)

        # --- status ---
        self.status_label = QLabel("Loading voices…")
        self.status_label.setStyleSheet(f"color: {_C.SUBTEXT0}; font-style: italic;")
        root.addWidget(self.status_label)

        # --- preview button ---
        preview_row = QHBoxLayout()
        self.preview_btn = QPushButton("▶ Preview Voice")
        self.preview_btn.setEnabled(False)
        self.preview_btn.setStyleSheet(
            f"background: {_C.TEAL}; color: {_C.CRUST}; font-weight: 600; "
            f"padding: 6px 16px; border-radius: 5px;"
        )
        self.preview_btn.clicked.connect(self._preview_selected)
        preview_row.addWidget(self.preview_btn)
        preview_row.addStretch()
        root.addLayout(preview_row)

        # --- buttons ---
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        root.addWidget(self.button_box)

        # --- kick off loader ---
        self._worker = VoiceLoadWorker(backend, self)
        self._worker.finished.connect(self._on_loaded)
        self._worker.start()

    # -- slots --

    def _on_loaded(self, voices: list):
        self._voices = voices
        self._populate()
        self.status_label.setText(f"{len(self._voices)} voice(s) loaded")
        self.preview_btn.setEnabled(True)

    def _populate(self):
        self.list_widget.clear()
        for v in self._voices:
            label = v.display() if hasattr(v, "display") else v.id
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, v)
            # Gender badge colouring
            gender = getattr(v, "gender", "").lower()
            colour_map = {
                "female": _C.PINK,
                "male": _C.BLUE,
                "neutral": _C.GREEN,
            }
            fg = colour_map.get(gender, _C.TEXT)
            item.setForeground(QColor(fg))
            self.list_widget.addItem(item)
            if v.id == self._current:
                self.list_widget.setCurrentItem(item)
        self._filter(self.search_edit.text())

    def _filter(self, text: str = ""):
        text = text.lower()
        gender_filter = self.gender_combo.currentText().lower()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            voice: VoiceInfo = item.data(Qt.ItemDataRole.UserRole)
            matches_text = text in item.text().lower()
            matches_gender = (
                gender_filter == "all"
                or getattr(voice, "gender", "").lower() == gender_filter
            )
            item.setHidden(not (matches_text and matches_gender))

    def _preview_selected(self):
        """Generate a short sample for the currently selected voice and play it."""
        cur = self.list_widget.currentItem()
        if cur is None:
            return
        voice: VoiceInfo = cur.data(Qt.ItemDataRole.UserRole)
        if voice is None:
            return

        self.preview_btn.setEnabled(False)
        self.preview_btn.setText("⏳ Generating…")

        self._preview_worker = VoicePreviewWorker(
            backend=self._backend,
            voice_id=voice.id,
            parent=self,
        )
        self._preview_worker.finished.connect(self._on_preview_ready)
        self._preview_worker.error.connect(self._on_preview_error)
        self._preview_worker.start()

    def _on_preview_ready(self, path: str):
        self._preview_path = path
        self.preview_btn.setEnabled(True)
        self.preview_btn.setText("▶ Preview Voice")
        # Play using QMediaPlayer
        if self._player is None:
            self._audio_output = QAudioOutput()
            self._player = QMediaPlayer()
            self._player.setAudioOutput(self._audio_output)
        self._player.setSource(QUrl.fromLocalFile(path))
        self._player.play()

    def _on_preview_error(self, msg: str):
        self.preview_btn.setEnabled(True)
        self.preview_btn.setText("▶ Preview Voice")
        self.status_label.setText(f"Preview error: {msg}")

    def get_selected(self) -> str:
        cur = self.list_widget.currentItem()
        if cur is None:
            return ""
        voice: VoiceInfo = cur.data(Qt.ItemDataRole.UserRole)
        return voice.id if voice else cur.text()


# ═══════════════════════════════════════════════════════════════════════
# Local Model Configuration Dialog
# ═══════════════════════════════════════════════════════════════════════

class LocalModelConfigDialog(QDialog):
    """Configure parameters for local / API TTS backends.

    Upgraded: added device auto-detect (cuda / cpu).
    """

    def __init__(self, config: dict, backend_id: str, parent=None):
        super().__init__(parent)
        self._backend_id = backend_id
        self._cfg = dict(config)

        self.setWindowTitle(f"Configure — {backend_id}")
        self.setMinimumWidth(560)

        root = QVBoxLayout(self)

        # --- info ---
        desc_map = {
            "qwen_local": (
                "Qwen TTS — runs locally via HF transformers.  "
                "Provide the model path and device."
            ),
            "omnivoice_local": (
                "OmniVoice — local multi-speaker model.  "
                "Set model path, device and speaker ID."
            ),
        }
        info = QLabel(desc_map.get(backend_id, f"Configuration for {backend_id}"))
        info.setWordWrap(True)
        info.setStyleSheet(
            f"color: {_C.SUBTEXT0}; font-style: italic; margin-bottom: 8px;"
        )
        root.addWidget(info)

        # --- form ---
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        if backend_id == "qwen_local":
            self._add_local_model_fields(form)
        elif backend_id == "omnivoice_local":
            self._add_omnivoice_fields(form)
        else:
            self._add_generic_fields(form)

        root.addLayout(form)

        # --- buttons ---
        bbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
        )
        bbox.accepted.connect(self._save)
        bbox.rejected.connect(self.reject)
        root.addWidget(bbox)

    # -- field builders --

    def _add_local_model_fields(self, form: QFormLayout):
        row = QHBoxLayout()
        self.model_path_edit = QLineEdit(self._cfg.get("model_path", ""))
        row.addWidget(self.model_path_edit)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_model)
        row.addWidget(browse_btn)
        form.addRow("Model path:", row)

        device_row = QHBoxLayout()
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        detected = self._auto_detect_device()
        self.device_combo.setCurrentText(self._cfg.get("device", detected))
        device_row.addWidget(self.device_combo)
        detect_btn = QPushButton("Auto-Detect")
        detect_btn.setToolTip("Detect best available compute device")
        detect_btn.setStyleSheet(
            f"background: {_C.MAUVE}; color: {_C.CRUST}; font-weight: 600; "
            f"padding: 4px 10px; border-radius: 4px;"
        )
        detect_btn.clicked.connect(self._auto_detect_and_set)
        device_row.addWidget(detect_btn)
        form.addRow("Device:", device_row)

    def _add_omnivoice_fields(self, form: QFormLayout):
        row = QHBoxLayout()
        self.model_path_edit = QLineEdit(self._cfg.get("model_path", ""))
        row.addWidget(self.model_path_edit)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_model)
        row.addWidget(browse_btn)
        form.addRow("Model path:", row)

        device_row = QHBoxLayout()
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        detected = self._auto_detect_device()
        self.device_combo.setCurrentText(self._cfg.get("device", detected))
        device_row.addWidget(self.device_combo)
        detect_btn = QPushButton("Auto-Detect")
        detect_btn.setToolTip("Detect best available compute device")
        detect_btn.setStyleSheet(
            f"background: {_C.MAUVE}; color: {_C.CRUST}; font-weight: 600; "
            f"padding: 4px 10px; border-radius: 4px;"
        )
        detect_btn.clicked.connect(self._auto_detect_and_set)
        device_row.addWidget(detect_btn)
        form.addRow("Device:", device_row)

        self.speaker_spin = QSpinBox()
        self.speaker_spin.setRange(0, 9999)
        self.speaker_spin.setValue(int(self._cfg.get("speaker_id", 0)))
        form.addRow("Speaker ID:", self.speaker_spin)

    def _add_generic_fields(self, form: QFormLayout):
        for key, val in self._cfg.items():
            le = QLineEdit(str(val))
            le.setObjectName(f"cfg_{key}")
            form.addRow(f"{key}:", le)

    # -- helpers --

    @staticmethod
    def _auto_detect_device() -> str:
        """Return 'cuda' if CUDA appears available, else 'cpu'."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        # Fallback: check nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return "cuda"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return "cpu"

    def _auto_detect_and_set(self):
        device = self._auto_detect_device()
        self.device_combo.setCurrentText(device)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "All Files (*)"
        )
        if path:
            self.model_path_edit.setText(path)

    def _save(self):
        if self._backend_id == "qwen_local":
            self._cfg["model_path"] = self.model_path_edit.text()
            self._cfg["device"] = self.device_combo.currentText()
        elif self._backend_id == "omnivoice_local":
            self._cfg["model_path"] = self.model_path_edit.text()
            self._cfg["device"] = self.device_combo.currentText()
            self._cfg["speaker_id"] = self.speaker_spin.value()
        else:
            for key in list(self._cfg.keys()):
                le = self.findChild(QLineEdit, f"cfg_{key}")
                if le:
                    self._cfg[key] = le.text()
        self.accept()

    def get_config(self) -> dict:
        return dict(self._cfg)


# ═══════════════════════════════════════════════════════════════════════
# Metadata Dialog
# ═══════════════════════════════════════════════════════════════════════

class MetadataDialog(QDialog):
    """Edit audiobook metadata: title, author, cover art.

    Upgraded: added auto-extract from source button.
    """

    def __init__(self, config: dict, source_file: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Audiobook Metadata")
        self.setMinimumWidth(520)
        self._cfg = dict(config)
        self._source_file = source_file

        root = QVBoxLayout(self)
        form = QFormLayout()

        self.title_edit = QLineEdit(self._cfg.get("title", ""))
        form.addRow("Title:", self.title_edit)

        self.author_edit = QLineEdit(self._cfg.get("author", ""))
        form.addRow("Author:", self.author_edit)

        cover_row = QHBoxLayout()
        self.cover_edit = QLineEdit(self._cfg.get("cover_art", ""))
        cover_row.addWidget(self.cover_edit, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_cover)
        cover_row.addWidget(browse)
        form.addRow("Cover Art:", cover_row)

        root.addLayout(form)

        # --- auto-extract ---
        auto_row = QHBoxLayout()
        self.auto_extract_btn = QPushButton("🔗 Auto-Extract from Source")
        self.auto_extract_btn.setToolTip(
            "Attempt to extract title, author, and cover from the source file"
        )
        self.auto_extract_btn.setStyleSheet(
            f"background: {_C.MAUVE}; color: {_C.CRUST}; font-weight: 600; "
            f"padding: 6px 14px; border-radius: 5px;"
        )
        self.auto_extract_btn.clicked.connect(self._auto_extract)
        self.auto_extract_btn.setEnabled(bool(source_file))
        auto_row.addWidget(self.auto_extract_btn)
        auto_row.addStretch()
        root.addLayout(auto_row)

        # --- buttons ---
        bbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
        )
        bbox.accepted.connect(self._save)
        bbox.rejected.connect(self.reject)
        root.addWidget(bbox)

    def _browse_cover(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Cover Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if path:
            self.cover_edit.setText(path)

    def _auto_extract(self):
        """Try to extract metadata from the source file using TextProcessor."""
        if not self._source_file:
            return
        try:
            from text_processor import TextProcessor
            tp = TextProcessor()
            meta = tp.extract_metadata(self._source_file)
            if meta.get("title"):
                self.title_edit.setText(meta["title"])
            if meta.get("author"):
                self.author_edit.setText(meta["author"])
            if meta.get("cover_art"):
                self.cover_edit.setText(meta["cover_art"])
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Auto-Extract Failed",
                f"Could not extract metadata:\n{exc}",
            )

    def _save(self):
        self._cfg["title"] = self.title_edit.text()
        self._cfg["author"] = self.author_edit.text()
        self._cfg["cover_art"] = self.cover_edit.text()
        self.accept()

    def get_config(self) -> dict:
        return dict(self._cfg)


# ═══════════════════════════════════════════════════════════════════════
# Voice Assignment Dialog — Advanced (kept for compatibility)
# ═══════════════════════════════════════════════════════════════════════

class VoiceAssignmentDialog(QDialog):
    """Professional voice-assignment dialog with auto-assign, voice pools,
    gender colour-coding, preview buttons and backend-aware defaults."""

    def __init__(
        self,
        config: dict,
        detected_speakers: list,
        backend: TTSBackend,
        character_registry=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Voice Assignment")
        self.setMinimumSize(820, 700)

        self._cfg = dict(config)
        self._speakers: list = detected_speakers
        self._backend = backend
        self._registry = character_registry or {}
        self._all_voices: list = []
        self._voice_map: dict = dict(self._cfg.get("voice_map", {}))

        self._build_ui()

        # Start loading voices in background
        self._worker = VoiceLoadWorker(backend, self)
        self._worker.finished.connect(self._on_voices_loaded)
        self._worker.start()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # ---- header ----
        header = QLabel(
            "Assign a TTS voice to each detected speaker.  Use auto-assign for a "
            "quick start, then fine-tune individual voices as needed."
        )
        header.setWordWrap(True)
        header.setStyleSheet(
            f"font-size:13px; padding:8px; background:{_C.SURFACE0}; "
            f"border-radius:6px; color:{_C.TEXT};"
        )
        root.addWidget(header)

        # ---- badge row ----
        badge_row = QHBoxLayout()
        badge_row.setSpacing(8)

        nlp_badge = QLabel("NLP Engine: 10-Pass Detection")
        nlp_badge.setStyleSheet(
            f"background:{_C.TEAL}; color:{_C.CRUST}; padding:3px 10px; "
            f"border-radius:10px; font-weight:600; font-size:11px;"
        )
        badge_row.addWidget(nlp_badge)

        backend_name = self._cfg.get("backend", "unknown")
        backend_info = BACKEND_REGISTRY.get(backend_name, {})
        display_name = backend_info.get("name", backend_info.get("display_name", backend_name))
        b_badge = QLabel(f"Backend: {display_name}")
        b_badge.setStyleSheet(
            f"background:{_C.MAUVE}; color:{_C.CRUST}; padding:3px 10px; "
            f"border-radius:10px; font-weight:600; font-size:11px;"
        )
        badge_row.addWidget(b_badge)

        if backend_info.get("supports_ssml", backend_info.get("ssml", False)):
            ssml_tag = QLabel("SSML")
            ssml_tag.setStyleSheet(
                f"background:{_C.PEACH}; color:{_C.CRUST}; padding:3px 8px; "
                f"border-radius:10px; font-weight:600; font-size:11px;"
            )
            badge_row.addWidget(ssml_tag)

        if backend_info.get("supports_multi_voice", backend_info.get("multi_voice", False)):
            mv_tag = QLabel("Multi-Voice")
            mv_tag.setStyleSheet(
                f"background:{_C.BLUE}; color:{_C.CRUST}; padding:3px 8px; "
                f"border-radius:10px; font-weight:600; font-size:11px;"
            )
            badge_row.addWidget(mv_tag)

        badge_row.addStretch()
        root.addLayout(badge_row)

        # ---- auto-assign strategy group ----
        strat_group = QGroupBox("Auto-Assign Strategy")
        strat_lay = QHBoxLayout(strat_group)

        self.strategy_combo = QComboBox()
        strategy_items = {
            VoiceAssignmentStrategy.GENDER_MATCH: "Gender Match",
            VoiceAssignmentStrategy.VOICE_POOL: "Voice Pool",
            VoiceAssignmentStrategy.NARRATOR_PLUS_ONE: "Narrator+One",
            VoiceAssignmentStrategy.MANUAL: "Manual Only",
        }
        for key, label in strategy_items.items():
            self.strategy_combo.addItem(label, key)

        current_strat = self._cfg.get("voice_assignment_strategy", VoiceAssignmentStrategy.GENDER_MATCH)
        idx = self.strategy_combo.findData(current_strat)
        if idx >= 0:
            self.strategy_combo.setCurrentIndex(idx)
        strat_lay.addWidget(QLabel("Strategy:"))
        strat_lay.addWidget(self.strategy_combo)

        auto_btn = QPushButton("Auto-Assign")
        auto_btn.setObjectName("autoAssignBtn")
        auto_btn.setStyleSheet(
            f"background:{_C.GREEN}; color:{_C.CRUST}; font-weight:600; "
            f"padding:5px 14px; border-radius:5px;"
        )
        auto_btn.clicked.connect(self._auto_assign)
        strat_lay.addWidget(auto_btn)

        reset_btn = QPushButton("Reset to Backend Defaults")
        reset_btn.setStyleSheet(
            f"background:{_C.PEACH}; color:{_C.CRUST}; font-weight:600; "
            f"padding:5px 14px; border-radius:5px;"
        )
        reset_btn.clicked.connect(self._reset_to_backend_defaults)
        strat_lay.addWidget(reset_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.setStyleSheet(
            f"background:{_C.SURFACE2}; color:{_C.TEXT}; font-weight:600; "
            f"padding:5px 14px; border-radius:5px;"
        )
        clear_btn.clicked.connect(self._clear_assignments)
        strat_lay.addWidget(clear_btn)

        strat_lay.addStretch()
        root.addWidget(strat_group)

        # ---- narrator override ----
        narrator_row = QHBoxLayout()
        narrator_row.addWidget(QLabel("Narrator Voice Override:"))
        self.narrator_combo = QComboBox()
        self.narrator_combo.setEditable(True)
        self.narrator_combo.addItem("")
        narrator_text = self._cfg.get("narrator_voice", "")
        self.narrator_combo.setCurrentText(narrator_text)
        narrator_row.addWidget(self.narrator_combo, 1)
        root.addLayout(narrator_row)

        # ---- speaker table ----
        table_group = QGroupBox("Speakers")
        table_lay = QVBoxLayout(table_group)

        self.speaker_table = QTableWidget(0, 6)
        self.speaker_table.setObjectName("speakerTable")
        self.speaker_table.setHorizontalHeaderLabels(
            ["Character", "Gender", "Lines", "Mentions", "Voice ID", "Preview"]
        )
        self.speaker_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.speaker_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Stretch
        )
        self.speaker_table.setAlternatingRowColors(True)
        self.speaker_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.speaker_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._build_speaker_table()
        table_lay.addWidget(self.speaker_table)
        root.addWidget(table_group)

        # ---- voice pool editor ----
        pool_group = QGroupBox("Voice Pool Editor  (Gender Match strategy)")
        pool_lay = QHBoxLayout(pool_group)

        for label_text, pool_key, colour in [
            ("Male", "male_pool", _C.BLUE),
            ("Female", "female_pool", _C.PINK),
            ("Neutral", "neutral_pool", _C.GREEN),
        ]:
            col = QVBoxLayout()
            lbl = QLabel(label_text)
            lbl.setStyleSheet(f"color:{colour}; font-weight:700;")
            col.addWidget(lbl)
            te = QTextEdit()
            te.setMaximumHeight(60)
            te.setPlaceholderText("One voice ID per line")
            pool_val = self._cfg.get(pool_key, [])
            if isinstance(pool_val, list):
                te.setPlainText("\n".join(pool_val))
            elif isinstance(pool_val, str):
                te.setPlainText(pool_val)
            setattr(self, f"_pool_{pool_key}", te)
            col.addWidget(te)
            pool_lay.addLayout(col)

        root.addWidget(pool_group)

        # ---- button box ----
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self._save)
        self.button_box.rejected.connect(self.reject)
        root.addWidget(self.button_box)

    # -------------------------------------------------- speaker table build

    def _build_speaker_table(self):
        self.speaker_table.setRowCount(0)
        for spk in self._speakers:
            row = self.speaker_table.rowCount()
            self.speaker_table.insertRow(row)

            name = spk.name if hasattr(spk, "name") else str(spk)
            gender = spk.gender if hasattr(spk, "gender") else "unknown"
            lines = spk.dialogue_count if hasattr(spk, "dialogue_count") else (
                spk.line_count if hasattr(spk, "line_count") else 0
            )
            mentions = spk.mention_count if hasattr(spk, "mention_count") else 0

            # character name (colour-coded by gender)
            name_item = QTableWidgetItem(name)
            colour_map = {
                "female": QColor(_C.PINK),
                "male": QColor(_C.BLUE),
                "unknown": QColor(_C.GREEN),
            }
            name_item.setForeground(colour_map.get(gender, QColor(_C.GREEN)))
            bold_font = name_item.font()
            bold_font.setBold(True)
            name_item.setFont(bold_font)
            self.speaker_table.setItem(row, 0, name_item)

            # gender
            gender_item = QTableWidgetItem(gender.capitalize())
            gender_item.setFlags(gender_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.speaker_table.setItem(row, 1, gender_item)

            # lines
            lines_item = QTableWidgetItem(str(lines))
            lines_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            lines_item.setFlags(lines_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.speaker_table.setItem(row, 2, lines_item)

            # mentions
            mentions_item = QTableWidgetItem(str(mentions))
            mentions_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            mentions_item.setFlags(mentions_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.speaker_table.setItem(row, 3, mentions_item)

            # voice id combo (editable)
            voice_combo = QComboBox()
            voice_combo.setEditable(True)
            voice_combo.addItem("")
            assigned = self._voice_map.get(name, "")
            if assigned:
                voice_combo.setCurrentText(assigned)
            self.speaker_table.setCellWidget(row, 4, voice_combo)

            # preview button
            preview_btn = QPushButton("▶ Play")
            preview_btn.setStyleSheet(
                f"padding:3px 10px; border-radius:4px; "
                f"background:{_C.SURFACE0}; color:{_C.TEXT}; font-size:11px;"
            )
            preview_btn.clicked.connect(lambda _, r=row: self._preview_voice(r))
            self.speaker_table.setCellWidget(row, 5, preview_btn)

    # -------------------------------------------------- voice loading

    def _on_voices_loaded(self, voices: list):
        self._all_voices = voices
        voice_ids = [""] + [v.id for v in voices]

        # Populate speaker table combos
        for row in range(self.speaker_table.rowCount()):
            combo = self.speaker_table.cellWidget(row, 4)
            if combo is None:
                continue
            current = combo.currentText()
            combo.clear()
            combo.addItems(voice_ids)
            if current:
                combo.setCurrentText(current)

        # Populate narrator combo
        self.narrator_combo.clear()
        self.narrator_combo.addItems(voice_ids)
        narr = self._cfg.get("narrator_voice", "")
        if narr:
            self.narrator_combo.setCurrentText(narr)

    # -------------------------------------------------- actions

    def _auto_assign(self):
        """Run auto-assignment via AppConfig helper."""
        try:
            app_cfg = AppConfig.from_dict(self._cfg)
            # Collect current pool text from editors
            for pool_key in ("male_pool", "female_pool", "neutral_pool"):
                te = getattr(self, f"_pool_{pool_key}", None)
                if te:
                    config_key = pool_key.replace("_pool", "_voice_pool")
                    setattr(app_cfg, config_key, [
                        line.strip() for line in te.toPlainText().splitlines() if line.strip()
                    ])

            self._cfg["voice_assignment_strategy"] = self.strategy_combo.currentData()

            speakers_tuples = []
            for spk in self._speakers:
                name = spk.name if hasattr(spk, "name") else str(spk)
                gender = spk.gender if hasattr(spk, "gender") else "unknown"
                speakers_tuples.append((name, gender))

            new_map = app_cfg.auto_assign_voices(speakers_tuples)
            self._voice_map.update(new_map)

            # Update combos
            for row in range(self.speaker_table.rowCount()):
                name_item = self.speaker_table.item(row, 0)
                if name_item is None:
                    continue
                combo = self.speaker_table.cellWidget(row, 4)
                if combo:
                    vid = self._voice_map.get(name_item.text(), "")
                    combo.setCurrentText(vid)
        except Exception as exc:
            print(f"[VoiceAssignmentDialog] auto-assign error: {exc}")

    def _reset_to_backend_defaults(self):
        """Reset voice pools to backend-specific defaults and refresh editors."""
        backend_name = self._cfg.get("backend", "")
        try:
            app_cfg = AppConfig.from_dict(self._cfg)
            app_cfg.reset_voice_pools_to_backend_defaults()
            pools = app_cfg.to_dict()
        except Exception:
            pools = BACKEND_VOICE_POOLS.get(backend_name, {})

        for pool_key in ("male_pool", "female_pool", "neutral_pool"):
            te = getattr(self, f"_pool_{pool_key}", None)
            if te:
                val = pools.get(pool_key, [])
                if isinstance(val, list):
                    te.setPlainText("\n".join(val))
                else:
                    te.setPlainText(str(val))

    def _clear_assignments(self):
        """Clear all voice assignment combos."""
        self._voice_map.clear()
        for row in range(self.speaker_table.rowCount()):
            combo = self.speaker_table.cellWidget(row, 4)
            if combo:
                combo.setCurrentIndex(0)
        self.narrator_combo.setCurrentIndex(0)

    def _preview_voice(self, row: int):
        """Generate and play a short sample for the voice in the given row."""
        name_item = self.speaker_table.item(row, 0)
        combo = self.speaker_table.cellWidget(row, 4)
        char_name = name_item.text() if name_item else "?"
        voice_id = combo.currentText() if combo else ""
        if not voice_id:
            return

        dialog = VoicePreviewDialog(
            backend=self._backend,
            voice_id=voice_id,
            character_name=char_name,
            parent=self,
        )
        dialog.exec()

    # -------------------------------------------------- save

    def _save(self):
        # voice map from table
        vm: dict = {}
        for row in range(self.speaker_table.rowCount()):
            name_item = self.speaker_table.item(row, 0)
            combo = self.speaker_table.cellWidget(row, 4)
            if name_item and combo:
                vid = combo.currentText()
                if vid:
                    vm[name_item.text()] = vid
        self._cfg["voice_map"] = vm

        # pools
        for pool_key in ("male_pool", "female_pool", "neutral_pool"):
            te = getattr(self, f"_pool_{pool_key}", None)
            if te:
                self._cfg[pool_key] = [
                    line.strip() for line in te.toPlainText().splitlines() if line.strip()
                ]

        self._cfg["voice_assignment_strategy"] = self.strategy_combo.currentData()
        self._cfg["narrator_voice"] = self.narrator_combo.currentText()
        self.accept()

    def get_config(self) -> dict:
        return dict(self._cfg)


# ═══════════════════════════════════════════════════════════════════════
# Speaker Map Dialog — MAJOR REDESIGN (Card-Based)
# ═══════════════════════════════════════════════════════════════════════

class SpeakerMapDialog(QDialog):
    """Card-based speaker-to-voice mapping dialog.

    Each speaker gets a visual card with:
    - Speaker name & gender label
    - Voice dropdown
    - Backend override dropdown
    - Rate / Volume / Pitch sliders
    - Voice Preview button
    - Confidence indicator badge

    Top bar: Auto-Assign Voices, Pick Voice (click-to-assign mode)
    """

    def __init__(
        self,
        speakers: list,
        registry,           # CharacterRegistry from text_processor
        config,             # AppConfig
        backend: TTSBackend,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Speaker Voice Map")
        self.setMinimumSize(780, 680)

        self._speakers: list = speakers          # List[str] of speaker names
        self._registry = registry                # CharacterRegistry
        self._config = config                    # AppConfig
        self._backend = backend
        self._all_voices: list = []
        self._cards: Dict[str, dict] = {}        # speaker_name → card widget refs
        self._pick_voice_mode: bool = False
        self._pick_voice_target: str = ""
        self._preview_worker: Optional[VoicePreviewWorker] = None
        self._player: Optional[QMediaPlayer] = None
        self._audio_output: Optional[QAudioOutput] = None

        self._build_ui()

        # Load voices in background
        self._worker = VoiceLoadWorker(backend, self)
        self._worker.finished.connect(self._on_voices_loaded)
        self._worker.start()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # ---- top bar ----
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        self.auto_assign_btn = QPushButton("⚡ Auto-Assign Voices")
        self.auto_assign_btn.setStyleSheet(
            f"background:{_C.GREEN}; color:{_C.CRUST}; font-weight:700; "
            f"padding:7px 18px; border-radius:6px; font-size:13px;"
        )
        self.auto_assign_btn.clicked.connect(self._auto_assign)
        top_bar.addWidget(self.auto_assign_btn)

        self.pick_voice_btn = QPushButton("🎯 Pick Voice (Click-to-Assign)")
        self.pick_voice_btn.setCheckable(True)
        self.pick_voice_btn.setStyleSheet(
            f"background:{_C.SURFACE1}; color:{_C.TEXT}; font-weight:600; "
            f"padding:7px 18px; border-radius:6px; font-size:13px;"
        )
        self.pick_voice_btn.toggled.connect(self._toggle_pick_mode)
        top_bar.addWidget(self.pick_voice_btn)

        top_bar.addStretch()

        # Strategy selector
        self.strategy_combo = QComboBox()
        strategy_items = {
            VoiceAssignmentStrategy.GENDER_MATCH: "Gender Match",
            VoiceAssignmentStrategy.VOICE_POOL: "Voice Pool",
            VoiceAssignmentStrategy.NARRATOR_PLUS_ONE: "Narrator+One",
            VoiceAssignmentStrategy.MANUAL: "Manual Only",
        }
        for key, label in strategy_items.items():
            self.strategy_combo.addItem(label, key)
        current_strat = getattr(self._config, "voice_assignment_strategy", VoiceAssignmentStrategy.GENDER_MATCH)
        idx = self.strategy_combo.findData(current_strat)
        if idx >= 0:
            self.strategy_combo.setCurrentIndex(idx)
        top_bar.addWidget(QLabel("Strategy:"))
        top_bar.addWidget(self.strategy_combo)
        root.addLayout(top_bar)

        # ---- pick mode indicator ----
        self.pick_indicator = QLabel("")
        self.pick_indicator.setStyleSheet(
            f"background:{_C.MAUVE}; color:{_C.CRUST}; font-weight:600; "
            f"padding:5px 12px; border-radius:5px; font-size:12px;"
        )
        self.pick_indicator.setVisible(False)
        root.addWidget(self.pick_indicator)

        # ---- scroll area for cards ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: 1px solid {_C.SURFACE0}; border-radius: 6px; }}"
        )

        self._cards_container = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_container)
        self._cards_layout.setSpacing(10)
        self._cards_layout.setContentsMargins(6, 6, 6, 6)
        self._cards_layout.addStretch()

        scroll.setWidget(self._cards_container)
        root.addWidget(scroll, 1)

        # ---- status ----
        self.status_label = QLabel("Loading voices…")
        self.status_label.setStyleSheet(f"color: {_C.SUBTEXT0}; font-style: italic;")
        root.addWidget(self.status_label)

        # ---- button box ----
        bbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
        )
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        root.addWidget(bbox)

        # Build cards
        self._build_all_cards()

    # -------------------------------------------------- card building

    def _build_all_cards(self):
        # Remove existing cards
        for speaker, refs in self._cards.items():
            card = refs.get("frame")
            if card:
                card.setParent(None)
                card.deleteLater()
        self._cards.clear()

        # Remove stretch, add cards, re-add stretch
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for speaker in self._speakers:
            char = None
            if self._registry and hasattr(self._registry, "characters"):
                chars = self._registry.characters
                char = chars.get(speaker)

            card = self._build_speaker_card(speaker, char)
            self._cards_layout.addWidget(card)

        self._cards_layout.addStretch()

    def _build_speaker_card(self, speaker: str, character) -> QFrame:
        """Build a card with: name label, gender, voice combo, backend override,
        rate/volume/pitch sliders, preview button, confidence badge."""

        card = QFrame()
        card.setObjectName("speakerCard")
        card.setStyleSheet(
            f"QFrame#speakerCard {{"
            f"  background: {_C.SURFACE0};"
            f"  border: 1px solid {_C.SURFACE1};"
            f"  border-radius: 10px;"
            f"  padding: 12px;"
            f"}}"
            f"QFrame#speakerCard:hover {{"
            f"  border-color: {_C.LAVENDER};"
            f"}}"
        )

        layout = QVBoxLayout(card)
        layout.setSpacing(6)

        # ---- Row 1: Name + Gender + Confidence Badge ----
        row1 = QHBoxLayout()

        # Speaker name (bold, gender-coloured)
        gender = getattr(character, "gender", "unknown") if character else "unknown"
        name_colour = {
            "female": _C.PINK,
            "male": _C.BLUE,
            "neutral": _C.GREEN,
        }.get(gender, _C.TEXT)

        name_label = QLabel(speaker)
        name_label.setStyleSheet(
            f"color: {name_colour}; font-size: 16px; font-weight: 700;"
        )
        row1.addWidget(name_label)

        # Gender badge
        gender_badge = QLabel(gender.capitalize())
        gender_bg = {
            "female": _C.PINK,
            "male": _C.BLUE,
            "neutral": _C.GREEN,
        }.get(gender, _C.SURFACE2)
        gender_badge.setStyleSheet(
            f"background: {gender_bg}; color: {_C.CRUST}; "
            f"padding: 2px 8px; border-radius: 8px; font-weight: 600; font-size: 11px;"
        )
        row1.addWidget(gender_badge)

        # Confidence badge
        # Determine confidence from character data
        confidence = self._compute_confidence(speaker, character)
        conf_colours = {
            "HIGH": _C.GREEN,
            "MEDIUM": _C.YELLOW,
            "LOW": _C.RED,
        }
        conf_bg = conf_colours.get(confidence, _C.SURFACE2)
        conf_badge = QLabel(f"● {confidence}")
        conf_badge.setStyleSheet(
            f"background: {conf_bg}; color: {_C.CRUST}; "
            f"padding: 2px 8px; border-radius: 8px; font-weight: 600; font-size: 11px;"
        )
        row1.addWidget(conf_badge)

        row1.addStretch()

        # Pick voice target button (for click-to-assign mode)
        pick_btn = QPushButton("🎯 Assign")
        pick_btn.setToolTip(f"Click to assign a voice to {speaker}")
        pick_btn.setStyleSheet(
            f"background: {_C.SURFACE1}; color: {_C.TEXT}; "
            f"padding: 3px 8px; border-radius: 4px; font-size: 11px;"
        )
        pick_btn.clicked.connect(lambda checked=False, s=speaker: self._on_card_pick(s))
        row1.addWidget(pick_btn)

        layout.addLayout(row1)

        # ---- Row 2: Voice combo + Backend override + Search button + Preview ----
        row2 = QHBoxLayout()

        # Voice dropdown
        voice_combo = QComboBox()
        voice_combo.setEditable(True)
        voice_combo.setMinimumWidth(180)
        # Pre-fill from config
        current_voice = ""
        if hasattr(self._config, "voice_map"):
            current_voice = self._config.voice_map.get(speaker, "")
        elif isinstance(self._config, dict):
            current_voice = self._config.get("voice_map", {}).get(speaker, "")
        voice_combo.addItem("")
        if current_voice:
            voice_combo.setCurrentText(current_voice)
        row2.addWidget(QLabel("Voice:"))
        row2.addWidget(voice_combo, 1)

        # Backend override
        backend_combo = QComboBox()
        backend_combo.addItem("(Global)")  # empty = use global backend
        for bid, binfo in BACKEND_REGISTRY.items():
            backend_combo.addItem(binfo.name, bid)
        # Check if there's a per-speaker backend override
        spk_cfg = self._get_speaker_config(speaker)
        override_backend = spk_cfg.get("backend_id", "")
        if override_backend:
            idx = backend_combo.findData(override_backend)
            if idx >= 0:
                backend_combo.setCurrentIndex(idx)
        row2.addWidget(QLabel("Backend:"))
        row2.addWidget(backend_combo)

        # Search button
        search_btn = QPushButton("🔍 Pick Voice")
        search_btn.setToolTip("Open voice search dialog")
        search_btn.setStyleSheet(
            f"background: {_C.LAVENDER}; color: {_C.CRUST}; "
            f"padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;"
        )
        search_btn.clicked.connect(
            lambda checked=False, s=speaker: self._pick_voice(s)
        )
        row2.addWidget(search_btn)

        # Preview button
        preview_btn = QPushButton("▶ Preview")
        preview_btn.setStyleSheet(
            f"background: {_C.TEAL}; color: {_C.CRUST}; "
            f"padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: 600;"
        )
        preview_btn.clicked.connect(
            lambda checked=False, s=speaker: self._preview_voice(s)
        )
        row2.addWidget(preview_btn)

        layout.addLayout(row2)

        # ---- Row 3: Rate / Volume / Pitch sliders ----
        row3 = QHBoxLayout()
        row3.setSpacing(12)

        rate_val = spk_cfg.get("rate", "+0%")
        volume_val = spk_cfg.get("volume", "+0%")
        pitch_val = spk_cfg.get("pitch", "+0Hz")

        # Rate slider (-50% to +100%)
        rate_label = QLabel(f"Rate: {rate_val}")
        rate_label.setMinimumWidth(90)
        rate_slider = QSlider(Qt.Orientation.Horizontal)
        rate_slider.setRange(-50, 100)
        rate_slider.setValue(self._parse_percent(rate_val))
        rate_slider.setTickInterval(25)
        rate_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        rate_slider.valueChanged.connect(
            lambda v, lbl=rate_label: lbl.setText(f"Rate: {v:+d}%")
        )
        row3.addWidget(rate_label)
        row3.addWidget(rate_slider, 1)

        # Volume slider (-50% to +100%)
        volume_label = QLabel(f"Vol: {volume_val}")
        volume_label.setMinimumWidth(90)
        volume_slider = QSlider(Qt.Orientation.Horizontal)
        volume_slider.setRange(-50, 100)
        volume_slider.setValue(self._parse_percent(volume_val))
        volume_slider.setTickInterval(25)
        volume_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        volume_slider.valueChanged.connect(
            lambda v, lbl=volume_label: lbl.setText(f"Vol: {v:+d}%")
        )
        row3.addWidget(volume_label)
        row3.addWidget(volume_slider, 1)

        # Pitch slider (-50Hz to +50Hz)
        pitch_label = QLabel(f"Pitch: {pitch_val}")
        pitch_label.setMinimumWidth(100)
        pitch_slider = QSlider(Qt.Orientation.Horizontal)
        pitch_slider.setRange(-50, 50)
        pitch_slider.setValue(self._parse_hz(pitch_val))
        pitch_slider.setTickInterval(10)
        pitch_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        pitch_slider.valueChanged.connect(
            lambda v, lbl=pitch_label: lbl.setText(f"Pitch: {v:+d}Hz")
        )
        row3.addWidget(pitch_label)
        row3.addWidget(pitch_slider, 1)

        layout.addLayout(row3)

        # Store references
        self._cards[speaker] = {
            "frame": card,
            "name_label": name_label,
            "gender_badge": gender_badge,
            "conf_badge": conf_badge,
            "voice_combo": voice_combo,
            "backend_combo": backend_combo,
            "rate_slider": rate_slider,
            "rate_label": rate_label,
            "volume_slider": volume_slider,
            "volume_label": volume_label,
            "pitch_slider": pitch_slider,
            "pitch_label": pitch_label,
            "preview_btn": preview_btn,
            "pick_btn": pick_btn,
        }

        return card

    # -------------------------------------------------- helpers

    @staticmethod
    def _parse_percent(val: str) -> int:
        """Parse '+0%' → 0, '-25%' → -25, '+50%' → 50."""
        try:
            return int(val.replace("%", "").replace("+", ""))
        except (ValueError, AttributeError):
            return 0

    @staticmethod
    def _parse_hz(val: str) -> int:
        """Parse '+0Hz' → 0, '-10Hz' → -10, '+20Hz' → 20."""
        try:
            return int(val.replace("Hz", "").replace("+", ""))
        except (ValueError, AttributeError):
            return 0

    def _get_speaker_config(self, speaker: str) -> dict:
        """Get per-speaker voice config from the AppConfig."""
        if hasattr(self._config, "speaker_voice_configs"):
            return self._config.speaker_voice_configs.get(speaker, {})
        if isinstance(self._config, dict):
            return self._config.get("speaker_voice_configs", {}).get(speaker, {})
        return {}

    def _compute_confidence(self, speaker: str, character) -> str:
        """Compute a confidence level based on character data."""
        if character is None:
            return "LOW"
        dialogue_count = getattr(character, "dialogue_count", 0)
        mention_count = getattr(character, "mention_count", 0)
        gender = getattr(character, "gender", "unknown")
        is_named = getattr(character, "is_named", False)

        score = 0
        if dialogue_count > 10:
            score += 2
        elif dialogue_count > 3:
            score += 1
        if mention_count > 5:
            score += 1
        if gender != "unknown":
            score += 2
        if is_named:
            score += 1

        if score >= 4:
            return "HIGH"
        elif score >= 2:
            return "MEDIUM"
        return "LOW"

    # -------------------------------------------------- voice loading

    def _on_voices_loaded(self, voices: list):
        self._all_voices = voices
        voice_ids = [""] + [v.id for v in voices]

        # Populate all card voice combos
        for speaker, refs in self._cards.items():
            combo = refs["voice_combo"]
            current = combo.currentText()
            combo.clear()
            combo.addItems(voice_ids)
            if current:
                combo.setCurrentText(current)

        self.status_label.setText(f"{len(voices)} voice(s) loaded")

    # -------------------------------------------------- actions

    def _auto_assign(self):
        """Run auto-assignment via AppConfig helper and update all cards."""
        try:
            if hasattr(self._config, "auto_assign_voices"):
                strategy = self.strategy_combo.currentData()
                self._config.voice_assignment_strategy = strategy

                speakers_tuples = []
                for speaker in self._speakers:
                    char = None
                    if self._registry and hasattr(self._registry, "characters"):
                        char = self._registry.characters.get(speaker)
                    gender = getattr(char, "gender", "unknown") if char else "unknown"
                    speakers_tuples.append((speaker, gender))

                new_map = self._config.auto_assign_voices(speakers_tuples)
            else:
                # Fallback: use config dict
                app_cfg = AppConfig.from_dict(self._config if isinstance(self._config, dict) else {})
                speakers_tuples = []
                for speaker in self._speakers:
                    char = None
                    if self._registry and hasattr(self._registry, "characters"):
                        char = self._registry.characters.get(speaker)
                    gender = getattr(char, "gender", "unknown") if char else "unknown"
                    speakers_tuples.append((speaker, gender))
                new_map = app_cfg.auto_assign_voices(speakers_tuples)

            # Update all card combos
            for speaker, refs in self._cards.items():
                combo = refs["voice_combo"]
                vid = new_map.get(speaker, "")
                if vid:
                    combo.setCurrentText(vid)

            self.status_label.setText(
                f"Auto-assigned {len(new_map)} voice(s)"
            )
        except Exception as exc:
            self.status_label.setText(f"Auto-assign error: {exc}")

    def _toggle_pick_mode(self, checked: bool):
        """Toggle the click-to-assign voice mode."""
        self._pick_voice_mode = checked
        if checked:
            self.pick_voice_btn.setStyleSheet(
                f"background:{_C.MAUVE}; color:{_C.CRUST}; font-weight:700; "
                f"padding:7px 18px; border-radius:6px; font-size:13px;"
            )
            self.pick_indicator.setText(
                "🎯 Click a speaker card, then select a voice from search"
            )
            self.pick_indicator.setVisible(True)
        else:
            self.pick_voice_btn.setStyleSheet(
                f"background:{_C.SURFACE1}; color:{_C.TEXT}; font-weight:600; "
                f"padding:7px 18px; border-radius:6px; font-size:13px;"
            )
            self.pick_indicator.setVisible(False)
            self._pick_voice_target = ""

    def _on_card_pick(self, speaker: str):
        """Handle clicking the assign button on a card (in pick mode or not)."""
        if self._pick_voice_mode:
            self._pick_voice_target = speaker
            self.pick_indicator.setText(
                f"🎯 Target: {speaker} — now click a voice in the search dialog"
            )
        # Open voice search regardless
        self._pick_voice(speaker)

    def _pick_voice(self, speaker: str):
        """Open VoiceSearchDialog for this speaker."""
        refs = self._cards.get(speaker)
        if not refs:
            return
        current = refs["voice_combo"].currentText()
        dialog = VoiceSearchDialog(
            backend=self._backend,
            current=current,
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected = dialog.get_selected()
            if selected:
                refs["voice_combo"].setCurrentText(selected)

    def _preview_voice(self, speaker: str):
        """Generate a short audio sample with this speaker's config."""
        refs = self._cards.get(speaker)
        if not refs:
            return
        voice_id = refs["voice_combo"].currentText()
        if not voice_id:
            return

        rate = f"{refs['rate_slider'].value():+d}%"
        volume = f"{refs['volume_slider'].value():+d}%"
        pitch = f"{refs['pitch_slider'].value():+d}Hz"

        refs["preview_btn"].setEnabled(False)
        refs["preview_btn"].setText("⏳…")

        # Determine backend to use (possibly overridden)
        backend = self._backend
        backend_idx = refs["backend_combo"].currentIndex()
        if backend_idx > 0:  # 0 = "(Global)"
            override_id = refs["backend_combo"].currentData()
            if override_id:
                try:
                    backend = TTSBackend.create(override_id)
                except Exception:
                    pass  # Fall back to global

        self._preview_worker = VoicePreviewWorker(
            backend=backend,
            voice_id=voice_id,
            rate=rate,
            volume=volume,
            pitch=pitch,
            parent=self,
        )
        self._preview_worker.finished.connect(
            lambda path, s=speaker: self._on_card_preview_ready(path, s)
        )
        self._preview_worker.error.connect(
            lambda msg, s=speaker: self._on_card_preview_error(msg, s)
        )
        self._preview_worker.start()

    def _on_card_preview_ready(self, path: str, speaker: str):
        refs = self._cards.get(speaker)
        if refs:
            refs["preview_btn"].setEnabled(True)
            refs["preview_btn"].setText("▶ Preview")
        # Play
        if self._player is None:
            self._audio_output = QAudioOutput()
            self._player = QMediaPlayer()
            self._player.setAudioOutput(self._audio_output)
        self._player.setSource(QUrl.fromLocalFile(path))
        self._player.play()

    def _on_card_preview_error(self, msg: str, speaker: str):
        refs = self._cards.get(speaker)
        if refs:
            refs["preview_btn"].setEnabled(True)
            refs["preview_btn"].setText("▶ Preview")
        self.status_label.setText(f"Preview error for {speaker}: {msg}")

    # -------------------------------------------------- output

    def get_voice_map(self) -> Dict[str, str]:
        """Return the current voice mapping."""
        result: Dict[str, str] = {}
        for speaker, refs in self._cards.items():
            vid = refs["voice_combo"].currentText()
            if vid:
                result[speaker] = vid
        return result

    def get_speaker_configs(self) -> Dict[str, Dict]:
        """Return per-speaker configs (voice_id, backend_id, rate, volume, pitch)."""
        result: Dict[str, Dict] = {}
        for speaker, refs in self._cards.items():
            backend_idx = refs["backend_combo"].currentIndex()
            backend_id = ""
            if backend_idx > 0:
                backend_id = refs["backend_combo"].currentData() or ""
            result[speaker] = {
                "voice_id": refs["voice_combo"].currentText(),
                "backend_id": backend_id,
                "rate": f"{refs['rate_slider'].value():+d}%",
                "volume": f"{refs['volume_slider'].value():+d}%",
                "pitch": f"{refs['pitch_slider'].value():+d}Hz",
            }
        return result


# ═══════════════════════════════════════════════════════════════════════
# Pronunciation Dialog
# ═══════════════════════════════════════════════════════════════════════

class PronunciationDialog(QDialog):
    """Edit the pronunciation dictionary (source → pronunciation).

    Upgraded: added Auto-Suggest button that calls TextProcessor.
    """

    def __init__(
        self,
        pronunciation_dict: dict,
        source_text: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Pronunciation Dictionary")
        self.setMinimumSize(600, 480)
        self._dict = dict(pronunciation_dict)
        self._source_text = source_text

        root = QVBoxLayout(self)

        # ---- main table ----
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Source", "Pronunciation"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.table.setAlternatingRowColors(True)

        for src, pron in self._dict.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(src))
            self.table.setItem(row, 1, QTableWidgetItem(pron))

        root.addWidget(self.table, 1)

        # ---- suggestion area ----
        suggest_group = QGroupBox("Auto-Suggest")
        suggest_lay = QVBoxLayout(suggest_group)

        self.suggest_list = QListWidget()
        self.suggest_list.setMaximumHeight(120)
        self.suggest_list.setVisible(False)
        self.suggest_list.itemDoubleClicked.connect(self._add_suggestion)
        suggest_lay.addWidget(self.suggest_list)

        suggest_btn_row = QHBoxLayout()
        auto_suggest_btn = QPushButton("🔍 Auto-Suggest")
        auto_suggest_btn.setToolTip(
            "Scan the source text for unusual words and suggest pronunciations"
        )
        auto_suggest_btn.setStyleSheet(
            f"background:{_C.MAUVE}; color:{_C.CRUST}; font-weight:600; "
            f"padding:5px 12px; border-radius:5px;"
        )
        auto_suggest_btn.clicked.connect(self._auto_suggest)
        suggest_btn_row.addWidget(auto_suggest_btn)

        self.suggest_status = QLabel("")
        self.suggest_status.setStyleSheet(f"color: {_C.SUBTEXT0};")
        suggest_btn_row.addWidget(self.suggest_status)
        suggest_btn_row.addStretch()
        suggest_lay.addLayout(suggest_btn_row)

        root.addWidget(suggest_group)

        # ---- button row ----
        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Entry")
        add_btn.clicked.connect(self._add_row)
        btn_row.addWidget(add_btn)
        rm_btn = QPushButton("Remove Entry")
        rm_btn.clicked.connect(self._remove_row)
        btn_row.addWidget(rm_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # ---- dialog buttons ----
        bbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
        )
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        root.addWidget(bbox)

    def _add_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(""))
        self.table.setItem(row, 1, QTableWidgetItem(""))
        self.table.editItem(self.table.item(row, 0))

    def _remove_row(self):
        rows = set(idx.row() for idx in self.table.selectedIndexes())
        for r in sorted(rows, reverse=True):
            self.table.removeRow(r)

    def _auto_suggest(self):
        """Call TextProcessor.suggest_pronunciations() on the source text."""
        if not self._source_text:
            self.suggest_status.setText("No source text available")
            return
        try:
            from text_processor import TextProcessor
            suggestions = TextProcessor.suggest_pronunciations(self._source_text)
            if not suggestions:
                self.suggest_status.setText("No suggestions found")
                return
            self.suggest_list.clear()
            self.suggest_list.setVisible(True)
            for src, pron in suggestions.items():
                # Skip already-present entries
                if src in self._dict:
                    continue
                item = QListWidgetItem(f"{src}  →  {pron}")
                item.setData(Qt.ItemDataRole.UserRole, (src, pron))
                self.suggest_list.addItem(item)
            self.suggest_status.setText(
                f"{self.suggest_list.count()} suggestion(s) — double-click to add"
            )
        except Exception as exc:
            self.suggest_status.setText(f"Error: {exc}")

    def _add_suggestion(self, item: QListWidgetItem):
        """Add a suggestion to the pronunciation table."""
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        src, pron = data
        self._dict[src] = pron
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(src))
        self.table.setItem(row, 1, QTableWidgetItem(pron))
        # Remove from suggestion list
        self.suggest_list.takeItem(self.suggest_list.row(item))

    def get_pronunciation_dict(self) -> dict:
        result: Dict[str, str] = {}
        for row in range(self.table.rowCount()):
            src_item = self.table.item(row, 0)
            pron_item = self.table.item(row, 1)
            if src_item and pron_item:
                src = src_item.text().strip()
                pron = pron_item.text().strip()
                if src:
                    result[src] = pron
        return result


# ═══════════════════════════════════════════════════════════════════════
# Quality Preset Dialog
# ═══════════════════════════════════════════════════════════════════════

class QualityPresetDialog(QDialog):
    """Configure output quality preset or custom parameters.

    Upgraded: added per-chapter normalization checkbox.
    """

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quality Preset")
        self.setMinimumWidth(560)
        self._cfg = dict(config)

        root = QVBoxLayout(self)

        # --- preset selector ---
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("custom")
        for name in QUALITY_PRESETS:
            self.preset_combo.addItem(name)
        current = self._cfg.get("quality_preset", "standard")
        idx = self.preset_combo.findText(current)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.preset_combo, 1)
        root.addLayout(preset_row)

        # --- description ---
        self.desc_label = QLabel()
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet(
            f"color:{_C.SUBTEXT0}; font-style:italic; margin-bottom:6px;"
        )
        root.addWidget(self.desc_label)
        self._update_desc()

        # --- custom parameters ---
        self.custom_group = QGroupBox("Custom Parameters")
        form = QFormLayout(self.custom_group)

        self.lufs_spin = QDoubleSpinBox()
        self.lufs_spin.setRange(-60.0, 0.0)
        self.lufs_spin.setDecimals(1)
        self.lufs_spin.setValue(float(self._cfg.get("target_lufs", -16.0)))
        self.lufs_spin.setSuffix(" LUFS")
        form.addRow("Target LUFS:", self.lufs_spin)

        self.peak_spin = QDoubleSpinBox()
        self.peak_spin.setRange(-6.0, 0.0)
        self.peak_spin.setDecimals(1)
        self.peak_spin.setValue(float(self._cfg.get("true_peak", -1.0)))
        self.peak_spin.setSuffix(" dB")
        form.addRow("True Peak:", self.peak_spin)

        self.rate_combo = QComboBox()
        self.rate_combo.addItems(["16000", "22050", "24000", "44100", "48000"])
        self.rate_combo.setCurrentText(str(self._cfg.get("sample_rate", 24000)))
        form.addRow("Sample Rate:", self.rate_combo)

        self.bitrate_combo = QComboBox()
        self.bitrate_combo.addItems([
            "64000", "96000", "128000", "192000", "256000", "320000",
        ])
        self.bitrate_combo.setCurrentText(str(self._cfg.get("bitrate", 128000)))
        form.addRow("Bitrate:", self.bitrate_combo)

        self.silence_spin = QDoubleSpinBox()
        self.silence_spin.setRange(0.0, 5.0)
        self.silence_spin.setDecimals(2)
        self.silence_spin.setValue(float(self._cfg.get("silence_threshold", 0.5)))
        self.silence_spin.setSuffix(" s")
        form.addRow("Silence Threshold:", self.silence_spin)

        self.inter_silence_spin = QDoubleSpinBox()
        self.inter_silence_spin.setRange(0.0, 5.0)
        self.inter_silence_spin.setDecimals(2)
        self.inter_silence_spin.setValue(
            float(self._cfg.get("inter_chunk_silence", 0.3))
        )
        self.inter_silence_spin.setSuffix(" s")
        form.addRow("Inter-chunk Silence:", self.inter_silence_spin)

        self.fade_spin = QDoubleSpinBox()
        self.fade_spin.setRange(0.0, 2.0)
        self.fade_spin.setDecimals(3)
        self.fade_spin.setValue(float(self._cfg.get("fade_duration", 0.05)))
        self.fade_spin.setSuffix(" s")
        form.addRow("Fade Duration:", self.fade_spin)

        # Per-chapter normalization
        self.per_chapter_norm_check = QCheckBox(
            "Per-chapter loudness normalization"
        )
        self.per_chapter_norm_check.setToolTip(
            "Apply individual loudness targets to each chapter instead of "
            "a single global target.  Useful for books with varying chapter lengths."
        )
        self.per_chapter_norm_check.setChecked(
            bool(self._cfg.get("per_chapter_normalization", False))
        )
        form.addRow(self.per_chapter_norm_check)

        root.addWidget(self.custom_group)
        self._toggle_custom()

        # --- buttons ---
        bbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
        )
        bbox.accepted.connect(self._save)
        bbox.rejected.connect(self.reject)
        root.addWidget(bbox)

    # -- helpers --

    def _update_desc(self):
        name = self.preset_combo.currentText()
        if name == "custom":
            self.desc_label.setText("Custom — adjust every parameter manually.")
        elif name in QUALITY_PRESETS:
            preset = QUALITY_PRESETS[name]
            self.desc_label.setText(getattr(preset, "description", str(preset)))
        else:
            self.desc_label.setText("")

    def _toggle_custom(self):
        is_custom = self.preset_combo.currentText() == "custom"
        self.custom_group.setVisible(is_custom)
        self.custom_group.setEnabled(is_custom)

    def _on_preset_changed(self, text: str):
        self._update_desc()
        self._toggle_custom()

        # Fill custom fields from preset values for reference
        if text in QUALITY_PRESETS:
            preset = QUALITY_PRESETS[text]
            self.lufs_spin.setValue(getattr(preset, "target_lufs", -16.0))
            self.peak_spin.setValue(getattr(preset, "true_peak", -1.0))
            sr = str(getattr(preset, "sample_rate", 24000))
            idx = self.rate_combo.findText(sr)
            if idx >= 0:
                self.rate_combo.setCurrentIndex(idx)
            br = str(getattr(preset, "bitrate", 128000))
            idx = self.bitrate_combo.findText(br)
            if idx >= 0:
                self.bitrate_combo.setCurrentIndex(idx)
            self.silence_spin.setValue(
                float(getattr(preset, "silence_threshold_db", -35)) / 10.0
                if isinstance(getattr(preset, "silence_threshold_db", -35), int)
                else float(getattr(preset, "silence_threshold", 0.5))
            )
            self.inter_silence_spin.setValue(
                getattr(preset, "inter_chunk_silence", 0.3)
            )
            self.fade_spin.setValue(getattr(preset, "fade_duration", 0.05))

    def _save(self):
        self._cfg["quality_preset"] = self.preset_combo.currentText()
        self._cfg["target_lufs"] = self.lufs_spin.value()
        self._cfg["true_peak"] = self.peak_spin.value()
        self._cfg["sample_rate"] = int(self.rate_combo.currentText())
        self._cfg["bitrate"] = int(self.bitrate_combo.currentText())
        self._cfg["silence_threshold"] = self.silence_spin.value()
        self._cfg["inter_chunk_silence"] = self.inter_silence_spin.value()
        self._cfg["fade_duration"] = self.fade_spin.value()
        self._cfg["per_chapter_normalization"] = self.per_chapter_norm_check.isChecked()
        self.accept()

    def get_config(self) -> dict:
        return dict(self._cfg)


# ═══════════════════════════════════════════════════════════════════════
# About Dialog
# ═══════════════════════════════════════════════════════════════════════

class AboutDialog(QDialog):
    """Application about box."""

    APP_NAME = "Audiobook Studio Pro"
    VERSION = "2.0.0"
    DESCRIPTION = (
        "Professional audiobook production suite with multi-speaker detection, "
        "auto voice assignment, SSML support, card-based voice mapping, "
        "export profiles, project templates, and production-quality output."
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {self.APP_NAME}")
        self.setFixedSize(480, 360)

        root = QVBoxLayout(self)

        # title
        title = QLabel(self.APP_NAME)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        root.addWidget(title)

        # version
        ver = QLabel(f"Version {self.VERSION}")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ver.setStyleSheet(f"color:{_C.SUBTEXT0}; margin-bottom:10px;")
        root.addWidget(ver)

        # description
        desc = QLabel(self.DESCRIPTION)
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("margin-bottom:12px;")
        root.addWidget(desc)

        # links
        links = QLabel(
            '<a href="https://github.com/audiobook-studio/audiobook-studio-pro">'
            "GitHub</a>  ·  "
            '<a href="https://audiobook-studio.dev/docs">Documentation</a>  ·  '
            '<a href="https://audiobook-studio.dev/changelog">Changelog</a>'
        )
        links.setAlignment(Qt.AlignmentFlag.AlignCenter)
        links.setOpenExternalLinks(True)
        root.addWidget(links)

        root.addStretch()

        # ok button
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setDefault(True)
        root.addWidget(ok_btn, alignment=Qt.AlignmentFlag.AlignCenter)


# ═══════════════════════════════════════════════════════════════════════
# NEW: Export Profile Dialog
# ═══════════════════════════════════════════════════════════════════════

class ExportProfileDialog(QDialog):
    """Select and apply an export profile (ACX, Spotify, Archive, General).

    Shows profile details and allows applying the profile to config,
    or applying and starting generation immediately.
    """

    def __init__(self, config, parent=None):
        """config may be an AppConfig or dict."""
        super().__init__(parent)
        self.setWindowTitle("Export Profile")
        self.setMinimumSize(600, 520)
        self._config = config
        self._generate_requested = False

        root = QVBoxLayout(self)

        # ---- header ----
        header = QLabel(
            "Select an export profile optimised for different distribution targets.  "
            "Each profile pre-configures format, quality, normalisation, and chapters."
        )
        header.setWordWrap(True)
        header.setStyleSheet(
            f"padding:8px; background:{_C.SURFACE0}; border-radius:6px; "
            f"color:{_C.TEXT}; font-size:12px;"
        )
        root.addWidget(header)

        # ---- profile list ----
        list_detail = QHBoxLayout()

        self.profile_list = QListWidget()
        self.profile_list.setMaximumWidth(200)
        for key, profile in EXPORT_PROFILES.items():
            item = QListWidgetItem(profile.name)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.profile_list.addItem(item)
        self.profile_list.setCurrentRow(0)
        self.profile_list.currentItemChanged.connect(self._on_profile_selected)
        list_detail.addWidget(self.profile_list)

        # ---- detail panel ----
        detail_frame = QFrame()
        detail_frame.setStyleSheet(
            f"QFrame {{ background: {_C.SURFACE0}; border-radius: 8px; padding: 12px; }}"
        )
        detail_lay = QVBoxLayout(detail_frame)

        self.detail_title = QLabel("")
        self.detail_title.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {_C.LAVENDER};"
        )
        detail_lay.addWidget(self.detail_title)

        self.detail_desc = QLabel("")
        self.detail_desc.setWordWrap(True)
        self.detail_desc.setStyleSheet(f"color: {_C.SUBTEXT1}; margin-bottom: 8px;")
        detail_lay.addWidget(self.detail_desc)

        # Detail grid
        self.detail_grid = QGridLayout()
        self.detail_grid.setSpacing(6)
        self._detail_labels: Dict[str, QLabel] = {}
        detail_fields = [
            ("Format", "output_format"),
            ("Quality Preset", "quality_preset"),
            ("Include Cover", "include_cover"),
            ("Include Chapters", "include_chapters"),
            ("Normalize Audio", "normalize"),
            ("Trim Silence", "trim_silence"),
            ("Crossfade", "crossfade_ms"),
            ("Room Tone", "room_tone"),
        ]
        for row, (label, key) in enumerate(detail_fields):
            lbl = QLabel(f"{label}:")
            lbl.setStyleSheet(f"color: {_C.OVERLAY1}; font-weight: 600;")
            val = QLabel("")
            val.setStyleSheet(f"color: {_C.TEXT};")
            self.detail_grid.addWidget(lbl, row, 0)
            self.detail_grid.addWidget(val, row, 1)
            self._detail_labels[key] = val

        detail_lay.addLayout(self.detail_grid)
        detail_lay.addStretch()

        list_detail.addWidget(detail_frame, 1)
        root.addLayout(list_detail, 1)

        # ---- buttons ----
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        apply_btn = QPushButton("Apply Profile")
        apply_btn.setStyleSheet(
            f"background:{_C.BLUE}; color:{_C.CRUST}; font-weight: 700; "
            f"padding: 8px 20px; border-radius: 6px; font-size: 13px;"
        )
        apply_btn.clicked.connect(self._apply)
        btn_row.addWidget(apply_btn)

        generate_btn = QPushButton("Apply & Generate")
        generate_btn.setStyleSheet(
            f"background:{_C.GREEN}; color:{_C.CRUST}; font-weight: 700; "
            f"padding: 8px 20px; border-radius: 6px; font-size: 13px;"
        )
        generate_btn.clicked.connect(self._apply_and_generate)
        btn_row.addWidget(generate_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(
            f"background:{_C.SURFACE2}; color:{_C.TEXT}; font-weight: 600; "
            f"padding: 8px 20px; border-radius: 6px;"
        )
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        root.addLayout(btn_row)

        # Initial selection
        self._on_profile_selected(self.profile_list.currentItem(), None)

    def _on_profile_selected(self, current, _previous):
        if current is None:
            return
        key = current.data(Qt.ItemDataRole.UserRole)
        profile = EXPORT_PROFILES.get(key)
        if profile is None:
            return
        self.detail_title.setText(profile.name)
        self.detail_desc.setText(profile.description)

        self._detail_labels["output_format"].setText(profile.output_format.upper())
        self._detail_labels["quality_preset"].setText(profile.quality_preset)
        self._detail_labels["include_cover"].setText(
            "✅ Yes" if profile.include_cover else "❌ No"
        )
        self._detail_labels["include_chapters"].setText(
            "✅ Yes" if profile.include_chapters else "❌ No"
        )
        self._detail_labels["normalize"].setText(
            "✅ Yes" if profile.normalize else "❌ No"
        )
        self._detail_labels["trim_silence"].setText(
            "✅ Yes" if profile.trim_silence else "❌ No"
        )
        self._detail_labels["crossfade_ms"].setText(
            f"{profile.crossfade_ms} ms"
        )
        self._detail_labels["room_tone"].setText(profile.room_tone.capitalize())

    def _get_selected_key(self) -> str:
        current = self.profile_list.currentItem()
        if current:
            return current.data(Qt.ItemDataRole.UserRole) or "acx"
        return "acx"

    def _apply(self):
        """Apply the selected export profile to config and accept."""
        key = self._get_selected_key()
        if hasattr(self._config, "apply_export_profile"):
            self._config.apply_export_profile(key)
        self._generate_requested = False
        self.accept()

    def _apply_and_generate(self):
        """Apply the selected export profile and signal generation request."""
        key = self._get_selected_key()
        if hasattr(self._config, "apply_export_profile"):
            self._config.apply_export_profile(key)
        self._generate_requested = True
        self.accept()

    @property
    def wants_generate(self) -> bool:
        """True if the user clicked 'Apply & Generate'."""
        return self._generate_requested


# ═══════════════════════════════════════════════════════════════════════
# NEW: Project Template Dialog
# ═══════════════════════════════════════════════════════════════════════

class ProjectTemplateDialog(QDialog):
    """Select and apply a project template.

    Templates pre-configure backend, voice strategy, quality, multi-speaker,
    SSML, and default voice map.
    """

    def __init__(self, config, parent=None):
        """config may be an AppConfig or dict."""
        super().__init__(parent)
        self.setWindowTitle("Project Template")
        self.setMinimumSize(620, 520)
        self._config = config

        root = QVBoxLayout(self)

        # ---- header ----
        header = QLabel(
            "Choose a project template to quickly set up the optimal configuration "
            "for your audiobook type.  Templates configure backend, voice strategy, "
            "quality preset, and more."
        )
        header.setWordWrap(True)
        header.setStyleSheet(
            f"padding:8px; background:{_C.SURFACE0}; border-radius:6px; "
            f"color:{_C.TEXT}; font-size:12px;"
        )
        root.addWidget(header)

        # ---- template list + detail ----
        list_detail = QHBoxLayout()

        self.template_list = QListWidget()
        self.template_list.setMaximumWidth(200)
        for key, tmpl in PROJECT_TEMPLATES.items():
            item = QListWidgetItem(tmpl.name)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.template_list.addItem(item)
        self.template_list.setCurrentRow(0)
        self.template_list.currentItemChanged.connect(self._on_template_selected)
        list_detail.addWidget(self.template_list)

        # ---- detail panel ----
        detail_frame = QFrame()
        detail_frame.setStyleSheet(
            f"QFrame {{ background: {_C.SURFACE0}; border-radius: 8px; padding: 12px; }}"
        )
        detail_lay = QVBoxLayout(detail_frame)

        self.detail_title = QLabel("")
        self.detail_title.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {_C.MAUVE};"
        )
        detail_lay.addWidget(self.detail_title)

        self.detail_desc = QLabel("")
        self.detail_desc.setWordWrap(True)
        self.detail_desc.setStyleSheet(f"color: {_C.SUBTEXT1}; margin-bottom: 8px;")
        detail_lay.addWidget(self.detail_desc)

        # Detail grid
        self.detail_grid = QGridLayout()
        self.detail_grid.setSpacing(6)
        self._detail_labels: Dict[str, QLabel] = {}
        detail_fields = [
            ("Backend", "backend"),
            ("Voice Strategy", "voice_assignment_strategy"),
            ("Quality Preset", "quality_preset"),
            ("Multi-Speaker", "multi_speaker"),
            ("Chapter Aware", "chapter_aware"),
            ("Use SSML", "use_ssml"),
            ("Default Voice Map", "default_voice_map"),
        ]
        for row, (label, key) in enumerate(detail_fields):
            lbl = QLabel(f"{label}:")
            lbl.setStyleSheet(f"color: {_C.OVERLAY1}; font-weight: 600;")
            val = QLabel("")
            val.setWordWrap(True)
            val.setStyleSheet(f"color: {_C.TEXT};")
            self.detail_grid.addWidget(lbl, row, 0)
            self.detail_grid.addWidget(val, row, 1)
            self._detail_labels[key] = val

        detail_lay.addLayout(self.detail_grid)
        detail_lay.addStretch()

        list_detail.addWidget(detail_frame, 1)
        root.addLayout(list_detail, 1)

        # ---- buttons ----
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        apply_btn = QPushButton("Apply Template")
        apply_btn.setStyleSheet(
            f"background:{_C.MAUVE}; color:{_C.CRUST}; font-weight: 700; "
            f"padding: 8px 20px; border-radius: 6px; font-size: 13px;"
        )
        apply_btn.clicked.connect(self._apply)
        btn_row.addWidget(apply_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(
            f"background:{_C.SURFACE2}; color:{_C.TEXT}; font-weight: 600; "
            f"padding: 8px 20px; border-radius: 6px;"
        )
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        root.addLayout(btn_row)

        # Initial selection
        self._on_template_selected(self.template_list.currentItem(), None)

    def _on_template_selected(self, current, _previous):
        if current is None:
            return
        key = current.data(Qt.ItemDataRole.UserRole)
        tmpl = PROJECT_TEMPLATES.get(key)
        if tmpl is None:
            return
        self.detail_title.setText(tmpl.name)
        self.detail_desc.setText(tmpl.description)

        # Backend name
        backend_info = BACKEND_REGISTRY.get(tmpl.backend, None)
        backend_display = backend_info.name if backend_info else tmpl.backend

        self._detail_labels["backend"].setText(backend_display)
        self._detail_labels["voice_assignment_strategy"].setText(
            tmpl.voice_assignment_strategy.replace("_", " ").title()
        )
        self._detail_labels["quality_preset"].setText(tmpl.quality_preset)
        self._detail_labels["multi_speaker"].setText(
            "✅ Yes" if tmpl.multi_speaker else "❌ No"
        )
        self._detail_labels["chapter_aware"].setText(
            "✅ Yes" if tmpl.chapter_aware else "❌ No"
        )
        self._detail_labels["use_ssml"].setText(
            "✅ Yes" if tmpl.use_ssml else "❌ No"
        )
        # Format voice map
        vm_str = ", ".join(
            f"{k} → {v}" for k, v in tmpl.default_voice_map.items()
        )
        self._detail_labels["default_voice_map"].setText(vm_str or "—")

    def _get_selected_key(self) -> str:
        current = self.template_list.currentItem()
        if current:
            return current.data(Qt.ItemDataRole.UserRole) or "audiobook_standard"
        return "audiobook_standard"

    def _apply(self):
        """Apply the selected project template to config and accept."""
        key = self._get_selected_key()
        if hasattr(self._config, "apply_project_template"):
            self._config.apply_project_template(key)
        self.accept()


# ═══════════════════════════════════════════════════════════════════════
# NEW: Generation History Dialog
# ═══════════════════════════════════════════════════════════════════════

class GenerationHistoryDialog(QDialog):
    """Show generation history in a table.

    Columns: Time, Source, Backend, Quality, Chunks, Duration, Status
    Double-click to open output folder.  "Clear History" button.
    """

    def __init__(self, config, parent=None):
        """config may be an AppConfig or dict with 'generation_history'."""
        super().__init__(parent)
        self.setWindowTitle("Generation History")
        self.setMinimumSize(800, 480)
        self._config = config

        root = QVBoxLayout(self)

        # ---- table ----
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Time", "Source", "Backend", "Quality", "Chunks", "Duration", "Status"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            6, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(self._open_output_folder)

        # Populate
        history = self._get_history()
        for entry in history:
            row = self.table.rowCount()
            self.table.insertRow(row)

            ts = entry.get("timestamp", "")
            source = entry.get("source_file", "")
            backend = entry.get("backend", "")
            quality = entry.get("quality_preset", "")
            total = entry.get("total_chunks", 0)
            failed = entry.get("failed_chunks", 0)
            elapsed = entry.get("elapsed_seconds", 0.0)
            duration_ms = entry.get("duration_ms", 0.0)

            self.table.setItem(row, 0, QTableWidgetItem(ts))
            self.table.setItem(row, 1, QTableWidgetItem(os.path.basename(source)))
            self.table.setItem(row, 2, QTableWidgetItem(backend))
            self.table.setItem(row, 3, QTableWidgetItem(quality))
            chunks_item = QTableWidgetItem(str(total))
            chunks_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 4, chunks_item)

            # Duration in readable format
            if elapsed >= 60:
                dur_text = f"{elapsed / 60:.1f} min"
            else:
                dur_text = f"{elapsed:.1f} s"
            dur_item = QTableWidgetItem(dur_text)
            dur_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 5, dur_item)

            # Status: OK if no failures, else show failure count
            if failed == 0:
                status = "✅ OK"
                status_colour = _C.GREEN
            else:
                status = f"⚠️ {failed} failed"
                status_colour = _C.YELLOW
            status_item = QTableWidgetItem(status)
            status_item.setForeground(QColor(status_colour))
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 6, status_item)

        root.addWidget(self.table, 1)

        if not history:
            empty_label = QLabel("No generation history yet.")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_label.setStyleSheet(
                f"color:{_C.SUBTEXT0}; font-size:14px; margin: 20px;"
            )
            root.addWidget(empty_label)

        # ---- buttons ----
        btn_row = QHBoxLayout()

        clear_btn = QPushButton("🗑️ Clear History")
        clear_btn.setStyleSheet(
            f"background:{_C.RED}; color:{_C.CRUST}; font-weight: 600; "
            f"padding: 6px 14px; border-radius: 5px;"
        )
        clear_btn.clicked.connect(self._clear_history)
        btn_row.addWidget(clear_btn)

        btn_row.addStretch()

        open_btn = QPushButton("📂 Open Output Folder")
        open_btn.clicked.connect(self._open_selected_output)
        btn_row.addWidget(open_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)

        root.addLayout(btn_row)

    def _get_history(self) -> list:
        if hasattr(self._config, "generation_history"):
            return list(self._config.generation_history)
        if isinstance(self._config, dict):
            return list(self._config.get("generation_history", []))
        return []

    def _open_output_folder(self, index):
        """Open the output folder for the double-clicked row."""
        row = index.row()
        history = self._get_history()
        if row >= len(history):
            return
        output_path = history[row].get("output_path", "")
        if output_path and os.path.exists(output_path):
            self._reveal_path(output_path)
        else:
            QMessageBox.information(
                self, "Not Found", "Output folder no longer exists."
            )

    def _open_selected_output(self):
        """Open the output folder for the currently selected row."""
        rows = set(idx.row() for idx in self.table.selectedIndexes())
        if not rows:
            return
        history = self._get_history()
        row = min(rows)
        if row >= len(history):
            return
        output_path = history[row].get("output_path", "")
        if output_path and os.path.exists(output_path):
            self._reveal_path(output_path)

    @staticmethod
    def _reveal_path(path: str):
        """Reveal a path in the system file manager."""
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", os.path.dirname(path) or path])

    def _clear_history(self):
        reply = QMessageBox.question(
            self,
            "Clear History",
            "Are you sure you want to clear all generation history?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            if hasattr(self._config, "generation_history"):
                self._config.generation_history.clear()
            elif isinstance(self._config, dict):
                self._config["generation_history"] = []
            self.table.setRowCount(0)


# ═══════════════════════════════════════════════════════════════════════
# NEW: Voice Preview Dialog
# ═══════════════════════════════════════════════════════════════════════

class VoicePreviewDialog(QDialog):
    """Small dialog for previewing a voice.

    Shows voice name, plays a short sample.  Uses QMediaPlayer for playback.
    "Generate Sample" synthesises "Hello, this is a sample of my voice."
    Mini progress bar shows generation progress.
    """

    def __init__(
        self,
        backend: TTSBackend,
        voice_id: str = "",
        character_name: str = "",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Voice Preview")
        self.setMinimumSize(360, 220)
        self._backend = backend
        self._voice_id = voice_id
        self._preview_path: Optional[str] = None
        self._preview_worker: Optional[VoicePreviewWorker] = None

        root = QVBoxLayout(self)

        # ---- voice info ----
        info_frame = QFrame()
        info_frame.setStyleSheet(
            f"QFrame {{ background: {_C.SURFACE0}; border-radius: 8px; padding: 10px; }}"
        )
        info_lay = QVBoxLayout(info_frame)

        voice_label = QLabel(voice_id or "Default Voice")
        voice_label.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {_C.LAVENDER};"
        )
        info_lay.addWidget(voice_label)

        if character_name:
            char_label = QLabel(f"Character: {character_name}")
            char_label.setStyleSheet(f"color: {_C.SUBTEXT1};")
            info_lay.addWidget(char_label)

        params_text = f"Rate: {rate}  |  Volume: {volume}  |  Pitch: {pitch}"
        params_label = QLabel(params_text)
        params_label.setStyleSheet(f"color: {_C.OVERLAY1}; font-size: 11px;")
        info_lay.addWidget(params_label)

        root.addWidget(info_frame)

        # ---- progress bar ----
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            f"QProgressBar {{ border: 1px solid {_C.SURFACE1}; border-radius: 4px; "
            f"text-align: center; background: {_C.SURFACE0}; }}"
            f"QProgressBar::chunk {{ background: {_C.TEAL}; border-radius: 3px; }}"
        )
        root.addWidget(self.progress_bar)

        # ---- playback controls ----
        play_row = QHBoxLayout()

        self.generate_btn = QPushButton("🎙️ Generate Sample")
        self.generate_btn.setStyleSheet(
            f"background:{_C.TEAL}; color:{_C.CRUST}; font-weight: 700; "
            f"padding: 8px 16px; border-radius: 6px; font-size: 13px;"
        )
        self.generate_btn.clicked.connect(self._generate_sample)
        play_row.addWidget(self.generate_btn)

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setEnabled(False)
        self.play_btn.setStyleSheet(
            f"background:{_C.GREEN}; color:{_C.CRUST}; font-weight: 600; "
            f"padding: 8px 16px; border-radius: 6px;"
        )
        self.play_btn.clicked.connect(self._play_sample)
        play_row.addWidget(self.play_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            f"background:{_C.SURFACE2}; color:{_C.TEXT}; font-weight: 600; "
            f"padding: 8px 16px; border-radius: 6px;"
        )
        self.stop_btn.clicked.connect(self._stop_sample)
        play_row.addWidget(self.stop_btn)

        root.addLayout(play_row)

        # ---- status ----
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {_C.SUBTEXT0}; font-style: italic;")
        root.addWidget(self.status_label)

        # ---- close ----
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._cleanup_and_accept)
        root.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)

        # ---- media player ----
        self._player: Optional[QMediaPlayer] = None
        self._audio_output: Optional[QAudioOutput] = None

    def _generate_sample(self):
        """Synthesise a short sample in a worker thread."""
        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Generating sample…")

        self._preview_worker = VoicePreviewWorker(
            backend=self._backend,
            voice_id=self._voice_id,
            parent=self,
        )
        self._preview_worker.finished.connect(self._on_sample_ready)
        self._preview_worker.error.connect(self._on_sample_error)
        self._preview_worker.start()

    def _on_sample_ready(self, path: str):
        self._preview_path = path
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Sample ready — click Play to listen")
        self.play_btn.setEnabled(True)
        # Auto-play
        self._play_sample()

    def _on_sample_error(self, msg: str):
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {msg}")

    def _play_sample(self):
        if not self._preview_path or not os.path.exists(self._preview_path):
            self.status_label.setText("No sample available — generate first")
            return
        if self._player is None:
            self._audio_output = QAudioOutput()
            self._player = QMediaPlayer()
            self._player.setAudioOutput(self._audio_output)
        self._player.setSource(QUrl.fromLocalFile(self._preview_path))
        self._player.play()
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._player.playbackStateChanged.connect(self._on_playback_state)

    def _stop_sample(self):
        if self._player:
            self._player.stop()
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_playback_state(self, state):
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    def _cleanup_and_accept(self):
        """Stop playback and accept."""
        if self._player:
            self._player.stop()
        self.accept()

    def reject(self):
        if self._player:
            self._player.stop()
        super().reject()
