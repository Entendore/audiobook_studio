"""Application configuration — dataclass-based with validation and persistence.

Professional Audiobook Studio Pro configuration with quality presets,
pronunciation dictionary, recent files, multi-backend TTS, project management,
project templates, export profiles, per-speaker overrides, and generation history.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("audiobook_config.json")

# ── Quality Presets ──────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class QualityPreset:
    """Audio quality preset for different distribution targets."""
    name: str
    description: str
    target_lufs: float
    true_peak: float
    sample_rate: int
    bitrate: str          # e.g. "192k"
    codec: str            # e.g. "libmp3lame", "aac"
    mono: bool            # Force mono output
    silence_threshold_db: int
    inter_chunk_silence: float
    fade_duration: float


QUALITY_PRESETS: Dict[str, QualityPreset] = {
    "acx_audible": QualityPreset(
        name="ACX / Audible",
        description="Audible/ACX compliant: -16 LUFS, -3 dB TP, 44.1 kHz mono",
        target_lufs=-16.0, true_peak=-3.0, sample_rate=44100,
        bitrate="192k", codec="libmp3lame", mono=True,
        silence_threshold_db=-35, inter_chunk_silence=0.25, fade_duration=0.015,
    ),
    "podcast": QualityPreset(
        name="Podcast Standard",
        description="Podcast optimized: -16 LUFS, -1 dB TP, 48 kHz stereo",
        target_lufs=-16.0, true_peak=-1.0, sample_rate=48000,
        bitrate="192k", codec="libmp3lame", mono=False,
        silence_threshold_db=-40, inter_chunk_silence=0.20, fade_duration=0.010,
    ),
    "high_quality": QualityPreset(
        name="High Quality",
        description="Max fidelity: -14 LUFS, 48 kHz, 320 kbps",
        target_lufs=-14.0, true_peak=-1.0, sample_rate=48000,
        bitrate="320k", codec="libmp3lame", mono=False,
        silence_threshold_db=-45, inter_chunk_silence=0.30, fade_duration=0.020,
    ),
    "archive": QualityPreset(
        name="Archive / Lossless",
        description="Lossless WAV archival: -14 LUFS, 48 kHz",
        target_lufs=-14.0, true_peak=-1.0, sample_rate=48000,
        bitrate="999k", codec="pcm_s16le", mono=False,
        silence_threshold_db=-50, inter_chunk_silence=0.35, fade_duration=0.025,
    ),
    "custom": QualityPreset(
        name="Custom",
        description="User-defined settings",
        target_lufs=-16.0, true_peak=-1.0, sample_rate=44100,
        bitrate="192k", codec="libmp3lame", mono=True,
        silence_threshold_db=-35, inter_chunk_silence=0.25, fade_duration=0.015,
    ),
}


# ── Export Profiles ─────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ExportProfile:
    """One-click export profile for different distribution targets."""
    name: str
    description: str
    output_format: str       # mp3, m4b, wav
    quality_preset: str      # key into QUALITY_PRESETS
    include_cover: bool
    include_chapters: bool
    normalize: bool
    trim_silence: bool
    crossfade_ms: int        # crossfade between chapters in ms
    room_tone: str           # "none", "subtle", "moderate", "heavy"


EXPORT_PROFILES: Dict[str, ExportProfile] = {
    "acx": ExportProfile(
        name="ACX / Audible", description="ACX-compliant audiobook for Audible distribution",
        output_format="m4b", quality_preset="acx_audible",
        include_cover=True, include_chapters=True, normalize=True,
        trim_silence=True, crossfade_ms=0, room_tone="none",
    ),
    "spotify": ExportProfile(
        name="Spotify / Podcast", description="Optimized for Spotify podcast distribution",
        output_format="mp3", quality_preset="podcast",
        include_cover=True, include_chapters=False, normalize=True,
        trim_silence=True, crossfade_ms=500, room_tone="subtle",
    ),
    "archive": ExportProfile(
        name="Archive / Lossless", description="Lossless archival copy",
        output_format="wav", quality_preset="archive",
        include_cover=True, include_chapters=True, normalize=True,
        trim_silence=False, crossfade_ms=0, room_tone="none",
    ),
    "general": ExportProfile(
        name="General MP3", description="Standard MP3 for general use",
        output_format="mp3", quality_preset="high_quality",
        include_cover=True, include_chapters=True, normalize=True,
        trim_silence=True, crossfade_ms=200, room_tone="subtle",
    ),
}


# ── Project Templates ──────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ProjectTemplate:
    """Reusable project template with preset voice maps and settings."""
    name: str
    description: str
    backend: str
    voice_assignment_strategy: str
    quality_preset: str
    multi_speaker: bool
    chapter_aware: bool
    use_ssml: bool
    default_voice_map: Dict[str, str]


PROJECT_TEMPLATES: Dict[str, ProjectTemplate] = {
    "fantasy_novel": ProjectTemplate(
        name="Fantasy Novel", description="Multi-voice fantasy with dramatic voices",
        backend="edge_tts", voice_assignment_strategy="gender_match",
        quality_preset="acx_audible", multi_speaker=True,
        chapter_aware=True, use_ssml=True,
        default_voice_map={"Narrator": "en-US-AvaNeural"},
    ),
    "podcast_mono": ProjectTemplate(
        name="Podcast (Mono)", description="Single-voice podcast production",
        backend="edge_tts", voice_assignment_strategy="manual",
        quality_preset="podcast", multi_speaker=False,
        chapter_aware=False, use_ssml=False,
        default_voice_map={"Narrator": "en-US-GuyNeural"},
    ),
    "dramatic_reading": ProjectTemplate(
        name="Dramatic Reading", description="Full dramatic cast with DramaBox",
        backend="dramabox", voice_assignment_strategy="gender_match",
        quality_preset="high_quality", multi_speaker=True,
        chapter_aware=True, use_ssml=True,
        default_voice_map={"Narrator": "narrator_calm"},
    ),
    "audiobook_standard": ProjectTemplate(
        name="Standard Audiobook", description="Standard ACX-compliant audiobook",
        backend="edge_tts", voice_assignment_strategy="narrator_plus",
        quality_preset="acx_audible", multi_speaker=True,
        chapter_aware=True, use_ssml=True,
        default_voice_map={"Narrator": "en-US-AvaNeural"},
    ),
}


# ── Backend Info ─────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class BackendInfo:
    """Static metadata about a TTS backend."""
    id: str
    name: str
    description: str
    is_local: bool        # True if runs locally (no network)
    supports_ssml: bool   # True if supports SSML voice switching
    supports_multi_voice: bool  # True if can do multi-voice in single synthesis
    requires_config: bool  # True if needs model path / API key setup
    icon: str = ""        # Emoji or icon identifier for UI display


BACKEND_REGISTRY: Dict[str, BackendInfo] = {
    "edge_tts": BackendInfo(
        id="edge_tts", name="Edge TTS",
        description="Microsoft Edge TTS — Cloud-based, 400+ voices, SSML support",
        is_local=False, supports_ssml=True, supports_multi_voice=True,
        requires_config=False, icon="🌐",
    ),
    "qwen_local": BackendInfo(
        id="qwen_local", name="Qwen TTS",
        description="Qwen Local TTS — Transformer-based, runs on GPU/CPU",
        is_local=True, supports_ssml=False, supports_multi_voice=False,
        requires_config=True, icon="🤖",
    ),
    "omnivoice_local": BackendInfo(
        id="omnivoice_local", name="OmniVoice",
        description="OmniVoice Local TTS — Multi-speaker, runs on GPU/CPU",
        is_local=True, supports_ssml=False, supports_multi_voice=True,
        requires_config=True, icon="🎙️",
    ),
    "dramabox": BackendInfo(
        id="dramabox", name="DramaBox",
        description="DramaBox TTS — Dramatic multi-voice synthesis with emotion",
        is_local=False, supports_ssml=True, supports_multi_voice=True,
        requires_config=True, icon="🎭",
    ),
}


# ── Voice Auto-Assignment Strategy ──────────────────────────────

class VoiceAssignmentStrategy:
    """Strategy for automatically assigning voices to detected speakers."""
    GENDER_MATCH = "gender_match"        # Match voice gender to character gender
    VOICE_POOL = "voice_pool"            # Round-robin from a curated voice pool
    NARRATOR_PLUS_ONE = "narrator_plus"  # Narrator voice + one alternate voice for all dialogue
    MANUAL = "manual"                    # No auto-assignment; user picks every voice


# ── Backend-Aware Voice Pools ───────────────────────────────────

BACKEND_VOICE_POOLS: Dict[str, Dict[str, Any]] = {
    "edge_tts": {
        "male": [
            "en-US-AndrewNeural", "en-US-SamuelNeural", "en-GB-RyanNeural",
            "en-US-GuyNeural", "en-AU-WilliamNeural",
        ],
        "female": [
            "en-US-AvaNeural", "en-US-MichelleNeural", "en-GB-SoniaNeural",
            "en-US-JennyNeural", "en-AU-NatashaNeural",
        ],
        "neutral": [
            "en-US-AvaNeural", "en-US-AndrewNeural",
        ],
        "narrator": "en-US-AvaNeural",
    },
    "qwen_local": {
        "male": ["Ethan", "Echo"],
        "female": ["Cherry", "Serena", "Chelsie", "Bonnie"],
        "neutral": ["Cherry", "Ethan"],
        "narrator": "Cherry",
    },
    "omnivoice_local": {
        "male": ["1", "3"],
        "female": ["0", "2", "5"],
        "neutral": ["4"],
        "narrator": "0",
    },
    "dramabox": {
        "male": ["male_dramatic", "male_calm", "male_aggressive", "elder_male", "villain_male", "child_boy"],
        "female": ["female_dramatic", "female_calm", "female_sweet", "elder_female", "villain_female", "child_girl"],
        "neutral": ["narrator_calm", "narrator_warm"],
        "narrator": "narrator_calm",
    },
}


# ── Per-Speaker Voice Override ─────────────────────────────────

@dataclass
class SpeakerVoiceConfig:
    """Fine-grained per-speaker voice configuration."""
    voice_id: str = ""
    backend_id: str = ""        # Per-speaker backend override (empty = use global)
    rate: str = "+0%"           # Speed adjustment
    volume: str = "+0%"         # Volume adjustment
    pitch: str = "+0Hz"         # Pitch adjustment
    style: str = ""             # Voice style (e.g., "cheerful", "dramatic")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SpeakerVoiceConfig:
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


# ── Generation History Entry ───────────────────────────────────

@dataclass
class GenerationEntry:
    """Record of a single audiobook generation run."""
    timestamp: str
    source_file: str
    output_path: str
    backend: str
    voice: str
    quality_preset: str
    total_chunks: int
    failed_chunks: int
    duration_ms: float
    elapsed_seconds: float
    chapter_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Room Tone Presets ──────────────────────────────────────────

ROOM_TONE_PRESETS: Dict[str, Dict[str, Any]] = {
    "none": {"name": "None", "description": "No ambient background", "file": "", "volume_db": 0},
    "subtle": {"name": "Subtle Room", "description": "Light room ambiance", "file": "", "volume_db": -35},
    "moderate": {"name": "Moderate Room", "description": "Noticeable room tone", "file": "", "volume_db": -28},
    "heavy": {"name": "Heavy Ambient", "description": "Strong atmospheric background", "file": "", "volume_db": -22},
}


@dataclass
class AppConfig:
    """Mutable application configuration persisted to disk."""

    # ── Backend selection ───────────────────────────────────────
    backend: str = "edge_tts"

    # ── Edge TTS ────────────────────────────────────────────────
    edge_voice: str = "en-US-AvaNeural"
    edge_rate: str = "+0%"
    edge_volume: str = "+0%"
    edge_pitch: str = "+0Hz"

    # ── Local Qwen TTS ──────────────────────────────────────────
    qwen_model_path: str = "Qwen/Qwen2.5-3B"
    qwen_device: str = "cuda"
    qwen_voice: str = "Cherry"

    # ── Local OmniVoice TTS ─────────────────────────────────────
    omnivoice_model_path: str = "OmniVoice/OmniVoice-TTS"
    omnivoice_device: str = "cuda"
    omnivoice_speaker_id: int = 0

    # ── DramaBox TTS ────────────────────────────────────────────
    dramabox_api_url: str = ""
    dramabox_api_key: str = ""
    dramabox_voice_style: str = "dramatic"
    dramabox_model: str = "dramabox-v1"

    # ── General ─────────────────────────────────────────────────
    concurrency: int = 5
    chunk_size: int = 500
    output_format: str = "mp3"
    output_dir: str = ""
    last_file: str = ""
    normalize_audio: bool = True
    trim_silence: bool = True
    chapter_aware: bool = True
    multi_speaker: bool = False
    voice_map: Dict[str, str] = field(default_factory=dict)

    # ── Per-Speaker Voice Configs ───────────────────────────────
    speaker_voice_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Maps speaker_name -> SpeakerVoiceConfig.to_dict()

    # ── Fallback Chain ──────────────────────────────────────────
    fallback_backends: List[str] = field(default_factory=lambda: ["edge_tts"])
    enable_fallback: bool = True

    # ── Voice Auto-Assignment ───────────────────────────────────
    voice_assignment_strategy: str = VoiceAssignmentStrategy.GENDER_MATCH
    auto_assign_on_detect: bool = True
    narrator_voice: str = ""
    male_voice_pool: List[str] = field(default_factory=lambda: [
        "en-US-AndrewNeural", "en-US-SamuelNeural", "en-GB-RyanNeural",
        "en-US-GuyNeural", "en-AU-WilliamNeural",
    ])
    female_voice_pool: List[str] = field(default_factory=lambda: [
        "en-US-AvaNeural", "en-US-MichelleNeural", "en-GB-SoniaNeural",
        "en-US-JennyNeural", "en-AU-NatashaNeural",
    ])
    neutral_voice_pool: List[str] = field(default_factory=lambda: [
        "en-US-AvaNeural", "en-US-AndrewNeural",
    ])

    # ── Quality Preset ──────────────────────────────────────────
    quality_preset: str = "acx_audible"

    # ── Custom Audio Parameters ─────────────────────────────────
    custom_target_lufs: float = -16.0
    custom_true_peak: float = -1.0
    custom_sample_rate: int = 44100
    custom_bitrate: str = "192k"
    custom_silence_threshold_db: int = -35
    custom_inter_chunk_silence: float = 0.25
    custom_fade_duration: float = 0.015

    # ── SSML ────────────────────────────────────────────────────
    use_ssml: bool = True

    # ── Pronunciation Dictionary ────────────────────────────────
    pronunciation_dict: Dict[str, str] = field(default_factory=dict)

    # ── Metadata / Cover ────────────────────────────────────────
    book_title: str = ""
    book_author: str = ""
    cover_art: str = ""

    # ── Recent Files ────────────────────────────────────────────
    recent_files: List[str] = field(default_factory=list)
    max_recent_files: int = 10

    # ── Projects ────────────────────────────────────────────────
    last_project_dir: str = ""

    # ── Window geometry ─────────────────────────────────────────
    window_width: int = 960
    window_height: int = 1080
    window_x: int = -1
    window_y: int = -1
    window_maximized: bool = False

    # ── Theme ───────────────────────────────────────────────────
    theme: str = "dark"  # "dark" or "light"

    # ── Audio Post-Processing ───────────────────────────────────
    crossfade_duration_ms: int = 0         # Crossfade between chapters
    room_tone: str = "none"                # Room tone preset key
    room_tone_file: str = ""               # Custom room tone audio file
    per_chapter_normalization: bool = False # Individual loudness targets per chapter

    # ── Generation History ──────────────────────────────────────
    generation_history: List[Dict[str, Any]] = field(default_factory=list)
    max_history_entries: int = 50

    # ── Export Profile ──────────────────────────────────────────
    last_export_profile: str = "acx"

    # ── Advanced ────────────────────────────────────────────────
    auto_extract_metadata: bool = True
    verify_ffmpeg_on_start: bool = True
    log_level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AppConfig:
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)

    def save(self, path: Path = CONFIG_PATH) -> None:
        try:
            path.write_text(
                json.dumps(self.to_dict(), indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to save config: %s", exc)

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> AppConfig:
        if path.exists():
            try:
                return cls.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                logger.warning("Failed to load config: %s — using defaults", exc)
        return cls()

    def get_quality_preset(self) -> QualityPreset:
        """Return the active quality preset (or custom built from custom fields)."""
        if self.quality_preset == "custom":
            return QualityPreset(
                name="Custom",
                description="User-defined settings",
                target_lufs=self.custom_target_lufs,
                true_peak=self.custom_true_peak,
                sample_rate=self.custom_sample_rate,
                bitrate=self.custom_bitrate,
                codec="libmp3lame",
                mono=True,
                silence_threshold_db=self.custom_silence_threshold_db,
                inter_chunk_silence=self.custom_inter_chunk_silence,
                fade_duration=self.custom_fade_duration,
            )
        return QUALITY_PRESETS.get(self.quality_preset, QUALITY_PRESETS["acx_audible"])

    def add_recent_file(self, filepath: str) -> None:
        """Add a file to the recent files list, moving it to top if exists."""
        files = [f for f in self.recent_files if f != filepath]
        files.insert(0, filepath)
        self.recent_files = files[:self.max_recent_files]

    def apply_preset(self, preset_name: str) -> None:
        """Switch to a quality preset and copy its values to custom fields."""
        self.quality_preset = preset_name
        if preset_name != "custom":
            p = QUALITY_PRESETS.get(preset_name, QUALITY_PRESETS["acx_audible"])
            self.custom_target_lufs = p.target_lufs
            self.custom_true_peak = p.true_peak
            self.custom_sample_rate = p.sample_rate
            self.custom_bitrate = p.bitrate
            self.custom_silence_threshold_db = p.silence_threshold_db
            self.custom_inter_chunk_silence = p.inter_chunk_silence
            self.custom_fade_duration = p.fade_duration

    def apply_export_profile(self, profile_key: str) -> None:
        """Apply an export profile to this config."""
        profile = EXPORT_PROFILES.get(profile_key)
        if profile is None:
            return
        self.output_format = profile.output_format
        self.apply_preset(profile.quality_preset)
        self.normalize_audio = profile.normalize
        self.trim_silence = profile.trim_silence
        self.crossfade_duration_ms = profile.crossfade_ms
        self.room_tone = profile.room_tone
        self.last_export_profile = profile_key

    def apply_project_template(self, template_key: str) -> None:
        """Apply a project template to this config."""
        template = PROJECT_TEMPLATES.get(template_key)
        if template is None:
            return
        self.backend = template.backend
        self.voice_assignment_strategy = template.voice_assignment_strategy
        self.apply_preset(template.quality_preset)
        self.multi_speaker = template.multi_speaker
        self.chapter_aware = template.chapter_aware
        self.use_ssml = template.use_ssml
        self.voice_map = dict(template.default_voice_map)

    def get_voice_pools_for_backend(self, backend_id: str = "") -> Dict[str, List[str]]:
        """Get voice pools appropriate for the current backend."""
        bid = backend_id or self.backend
        default_pools = BACKEND_VOICE_POOLS.get(bid, BACKEND_VOICE_POOLS["edge_tts"])

        result: Dict[str, List[str]] = {
            "male": self.male_voice_pool if self.male_voice_pool else default_pools.get("male", []),
            "female": self.female_voice_pool if self.female_voice_pool else default_pools.get("female", []),
            "neutral": self.neutral_voice_pool if self.neutral_voice_pool else default_pools.get("neutral", []),
        }
        return result

    def get_default_voice(self, backend_id: str = "") -> str:
        """Get the default voice for the current or specified backend."""
        bid = backend_id or self.backend
        if bid == "edge_tts":
            return self.edge_voice
        elif bid == "qwen_local":
            return self.qwen_voice
        elif bid == "omnivoice_local":
            return str(self.omnivoice_speaker_id)
        elif bid == "dramabox":
            return self.dramabox_voice_style
        return self.edge_voice

    def reset_voice_pools_to_backend_defaults(self, backend_id: str = "") -> None:
        """Reset voice pools to the default pools for the current backend."""
        bid = backend_id or self.backend
        default_pools = BACKEND_VOICE_POOLS.get(bid, BACKEND_VOICE_POOLS["edge_tts"])
        self.male_voice_pool = list(default_pools.get("male", []))
        self.female_voice_pool = list(default_pools.get("female", []))
        self.neutral_voice_pool = list(default_pools.get("neutral", []))

    def get_speaker_voice_config(self, speaker: str) -> SpeakerVoiceConfig:
        """Get the per-speaker voice config, creating a default if not found."""
        if speaker in self.speaker_voice_configs:
            return SpeakerVoiceConfig.from_dict(self.speaker_voice_configs[speaker])
        return SpeakerVoiceConfig(voice_id=self.voice_map.get(speaker, ""))

    def set_speaker_voice_config(self, speaker: str, config: SpeakerVoiceConfig) -> None:
        """Set the per-speaker voice config."""
        self.speaker_voice_configs[speaker] = config.to_dict()
        if config.voice_id:
            self.voice_map[speaker] = config.voice_id

    def add_generation_entry(self, entry: GenerationEntry) -> None:
        """Add an entry to generation history, trimming to max size."""
        self.generation_history.insert(0, entry.to_dict())
        self.generation_history = self.generation_history[:self.max_history_entries]

    def auto_assign_voices(
        self,
        speakers: List[Tuple[str, str]],  # (name, gender) pairs
    ) -> Dict[str, str]:
        """Automatically assign TTS voices to detected speakers."""
        strategy = self.voice_assignment_strategy
        voice_map: Dict[str, str] = {}

        if strategy == VoiceAssignmentStrategy.MANUAL:
            return self.voice_map.copy()

        pools = self.get_voice_pools_for_backend()
        narrator = self.narrator_voice or self.get_default_voice()

        if strategy == VoiceAssignmentStrategy.NARRATOR_PLUS_ONE:
            voice_map["Narrator"] = narrator
            for name, gender in speakers:
                if name != "Narrator":
                    pool = pools.get("female" if gender == "female" else "male", [])
                    voice_map[name] = pool[0] if pool else narrator
            return voice_map

        if strategy == VoiceAssignmentStrategy.GENDER_MATCH:
            used_voices: set = set()
            voice_map["Narrator"] = narrator
            used_voices.add(narrator)

            for name, gender in speakers:
                if name == "Narrator":
                    continue
                if name in self.voice_map and self.voice_map[name]:
                    voice_map[name] = self.voice_map[name]
                    used_voices.add(self.voice_map[name])
                    continue
                if gender == "female":
                    pool = pools.get("female", [])
                elif gender == "male":
                    pool = pools.get("male", [])
                else:
                    pool = pools.get("neutral", [])
                assigned = False
                for v in pool:
                    if v not in used_voices:
                        voice_map[name] = v
                        used_voices.add(v)
                        assigned = True
                        break
                if not assigned and pool:
                    voice_map[name] = pool[0]
                    used_voices.add(pool[0])
            return voice_map

        if strategy == VoiceAssignmentStrategy.VOICE_POOL:
            all_voices = list(set(pools.get("male", []) + pools.get("female", [])))
            voice_map["Narrator"] = narrator
            idx = 0
            for name, gender in speakers:
                if name == "Narrator":
                    continue
                if name in self.voice_map and self.voice_map[name]:
                    voice_map[name] = self.voice_map[name]
                    continue
                if all_voices:
                    voice_map[name] = all_voices[idx % len(all_voices)]
                    idx += 1
            return voice_map

        return voice_map


# ── Engine Config (immutable snapshot for pipeline) ────────────

@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Immutable snapshot passed to the pipeline.  Thread-safe."""

    backend_id: str
    voice: str
    rate: str
    volume: str
    pitch: str
    concurrency: int
    chunk_size: int
    output_format: str
    normalize: bool
    trim_silence: bool
    chapter_aware: bool
    multi_speaker: bool
    voice_map: Dict[str, str]
    title: str = "Audiobook"
    author: str = "Unknown"
    cover_art: str = ""
    # Audio quality
    quality_preset: str = "acx_audible"
    target_lufs: float = -16.0
    true_peak: float = -3.0
    sample_rate: int = 44100
    bitrate: str = "192k"
    silence_threshold_db: int = -35
    inter_chunk_silence: float = 0.25
    fade_duration: float = 0.015
    # SSML
    use_ssml: bool = True
    # Pronunciation
    pronunciation_dict: Dict[str, str] = field(default_factory=dict)
    # Per-speaker voice configs
    speaker_voice_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Fallback chain
    fallback_backends: Tuple[str, ...] = ("edge_tts",)
    enable_fallback: bool = True
    # Local Model Configs
    qwen_model_path: str = ""
    qwen_device: str = "cuda"
    qwen_voice: str = "Cherry"
    omnivoice_model_path: str = ""
    omnivoice_device: str = "cuda"
    omnivoice_speaker_id: int = 0
    # DramaBox Config
    dramabox_api_url: str = ""
    dramabox_api_key: str = ""
    dramabox_voice_style: str = "dramatic"
    dramabox_model: str = "dramabox-v1"
    # Audio post-processing
    crossfade_duration_ms: int = 0
    room_tone: str = "none"
    room_tone_file: str = ""
    per_chapter_normalization: bool = False

    @classmethod
    def from_app_config(cls, app: AppConfig) -> EngineConfig:
        bid = app.backend
        if bid == "qwen_local":
            voice = app.qwen_voice
            rate, volume, pitch = "+0%", "+0%", "+0Hz"
        elif bid == "omnivoice_local":
            voice = str(app.omnivoice_speaker_id)
            rate, volume, pitch = "+0%", "+0%", "+0Hz"
        elif bid == "dramabox":
            voice = app.dramabox_voice_style
            rate, volume, pitch = "+0%", "+0%", "+0Hz"
        else:
            voice = app.edge_voice
            rate = app.edge_rate
            volume = app.edge_volume
            pitch = app.edge_pitch

        preset = app.get_quality_preset()

        return cls(
            backend_id=bid, voice=voice, rate=rate, volume=volume, pitch=pitch,
            concurrency=app.concurrency, chunk_size=app.chunk_size,
            output_format=app.output_format, normalize=app.normalize_audio,
            trim_silence=app.trim_silence, chapter_aware=app.chapter_aware,
            multi_speaker=app.multi_speaker, voice_map=app.voice_map,
            title=app.book_title or "Audiobook", author=app.book_author or "Unknown",
            cover_art=app.cover_art,
            quality_preset=app.quality_preset,
            target_lufs=preset.target_lufs, true_peak=preset.true_peak,
            sample_rate=preset.sample_rate, bitrate=preset.bitrate,
            silence_threshold_db=preset.silence_threshold_db,
            inter_chunk_silence=preset.inter_chunk_silence,
            fade_duration=preset.fade_duration,
            use_ssml=app.use_ssml,
            pronunciation_dict=app.pronunciation_dict,
            speaker_voice_configs=dict(app.speaker_voice_configs),
            fallback_backends=tuple(app.fallback_backends),
            enable_fallback=app.enable_fallback,
            qwen_model_path=app.qwen_model_path, qwen_device=app.qwen_device,
            qwen_voice=app.qwen_voice,
            omnivoice_model_path=app.omnivoice_model_path, omnivoice_device=app.omnivoice_device,
            omnivoice_speaker_id=app.omnivoice_speaker_id,
            dramabox_api_url=app.dramabox_api_url, dramabox_api_key=app.dramabox_api_key,
            dramabox_voice_style=app.dramabox_voice_style, dramabox_model=app.dramabox_model,
            crossfade_duration_ms=app.crossfade_duration_ms,
            room_tone=app.room_tone, room_tone_file=app.room_tone_file,
            per_chapter_normalization=app.per_chapter_normalization,
        )


# ── Project File ────────────────────────────────────────────────

@dataclass
class AudiobookProject:
    """Serializable project state for save/load/resume."""
    name: str = ""
    source_file: str = ""
    source_text: str = ""
    is_raw_text: bool = False
    output_path: str = ""
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    completed_chunks: List[int] = field(default_factory=list)
    total_chunks: int = 0
    chapter_count: int = 0
    template: str = ""

    def save(self, path: Path) -> None:
        try:
            path.write_text(
                json.dumps(asdict(self), indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to save project: %s", exc)

    @classmethod
    def load(cls, path: Path) -> AudiobookProject:
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Failed to load project: %s", exc)
            return cls()

    @classmethod
    def from_state(
        cls,
        name: str,
        source_file: str,
        source_text: str,
        is_raw_text: bool,
        output_path: str,
        config: AppConfig,
        completed_chunks: Optional[List[int]] = None,
        total_chunks: int = 0,
        chapter_count: int = 0,
        template: str = "",
    ) -> AudiobookProject:
        return cls(
            name=name,
            source_file=source_file,
            source_text=source_text if is_raw_text else "",
            is_raw_text=is_raw_text,
            output_path=output_path,
            config_snapshot=config.to_dict(),
            completed_chunks=completed_chunks or [],
            total_chunks=total_chunks,
            chapter_count=chapter_count,
            template=template,
        )
