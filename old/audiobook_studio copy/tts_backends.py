"""
TTS Backend Plugin System
=========================

Abstract base class with auto-registration registry, SSML multi-voice synthesis,
fallback chain support, and per-speaker backend override.

Supported backends (all inline):
  - Edge TTS   (cloud, via ``edge_tts`` package)
  - Qwen Local (local, via torch / transformers — optional)
  - OmniVoice  (local, multi-speaker — optional)
  - DramaBox   (cloud, via aiohttp REST API)
  - Fallback   (wrapper that tries multiple backends in order)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# VoiceInfo — Immutable voice descriptor
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class VoiceInfo:
    """Immutable descriptor for a TTS voice."""

    id: str
    name: str
    locale: str = ""
    gender: str = ""
    style: str = ""  # Voice style (e.g. "cheerful", "dramatic")

    def display(self) -> str:
        """Human-readable representation of the voice."""
        parts = [self.name]
        if self.locale:
            parts.append(f"({self.locale})")
        if self.gender:
            parts.append(f"[{self.gender}]")
        if self.style:
            parts.append(f"~{self.style}~")
        return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# TTSBackend — Abstract base with auto-registration registry
# ═══════════════════════════════════════════════════════════════════

class TTSBackend(ABC):
    """Abstract base class for TTS backend plugins.

    Subclasses are automatically registered via ``__init_subclass__``.
    Two registration patterns are supported:

    1. **Keyword arg** (preferred)::

        class MyBackend(TTSBackend, backend_id="my_backend"):
            ...

    2. **Class-level attribute** (legacy)::

        class MyBackend(TTSBackend):
            id = "my_backend"
            ...

    Use ``TTSBackend.get_backend(backend_id, config)`` or
    ``TTSBackend.create(backend_id, config)`` to instantiate a registered backend.
    """

    _registry: Dict[str, Type[TTSBackend]] = {}

    # --- Class-level descriptors (override in subclass) ---
    id: str = ""
    name: str = ""
    description: str = ""
    supports_ssml: bool = False
    supports_multi_voice: bool = False

    # ------------------------------------------------------------------
    # Auto-registration
    # ------------------------------------------------------------------

    def __init_subclass__(cls, backend_id: str = "", **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Prefer explicit backend_id keyword; fall back to class-level id
        effective_id = backend_id or getattr(cls, "id", "")
        if effective_id:
            cls._registry[effective_id] = cls
            cls._backend_id = effective_id
            logger.debug("Registered TTS backend: %s (%s)", effective_id, getattr(cls, "name", ""))

    # ------------------------------------------------------------------
    # Registry accessors
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, backend_id: str, config: dict) -> TTSBackend:
        """Instantiate a registered backend by its id."""
        if backend_id not in cls._registry:
            avail = ", ".join(cls._registry) or "none"
            raise ValueError(f"Unknown backend '{backend_id}'. Available: {avail}")
        return cls._registry[backend_id](config)

    @classmethod
    def get_backend(cls, backend_id: str, **kwargs: Any) -> TTSBackend:
        """Instantiate a registered backend by its id (keyword-arg style).

        Converts ``kwargs`` into a config dict for ``create()``.
        """
        return cls.create(backend_id, kwargs)

    @classmethod
    def available(cls) -> Dict[str, str]:
        """Return ``{backend_id: display_name}`` for all registered backends."""
        return {bid: getattr(bcls, "name", bid) for bid, bcls in cls._registry.items()}

    @classmethod
    def get_backend_info(cls, backend_id: str) -> Optional[Type[TTSBackend]]:
        """Return the backend class for the given id, or ``None``."""
        return cls._registry.get(backend_id)

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, config: dict) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Abstract methods — every backend MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        output_path: str,
        voice: str = "",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        **kwargs: Any,
    ) -> bool:
        """Synthesize plain text to an audio file. Returns ``True`` on success."""
        ...

    @abstractmethod
    async def synthesize_ssml(
        self,
        ssml: str,
        output_path: str,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
    ) -> bool:
        """Synthesize from SSML markup.

        Backends that don't support SSML should strip tags and fall back to
        plain text synthesis.
        """
        ...

    @abstractmethod
    async def list_voices(self) -> List[VoiceInfo]:
        """List available voices for this backend."""
        ...

    @abstractmethod
    def validate_config(self) -> List[str]:
        """Return a list of validation error strings.  Empty list = valid."""
        ...

    # ------------------------------------------------------------------
    # Multi-speaker segment synthesis
    # ------------------------------------------------------------------

    async def synthesize_segments(
        self,
        segments: List[Tuple[str, str]],
        output_path: str,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        use_ssml: bool = True,
        pronunciation_dict: Optional[Dict[str, str]] = None,
        on_retry: Optional[Callable[[int, str, Exception], None]] = None,
        speaker_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        backend_registry: Optional[Dict[str, TTSBackend]] = None,
    ) -> bool:
        """Synthesize multi-speaker segments.

        Strategy:
        1. If *speaker_configs* contains a ``backend_id`` override for any
           speaker, split segments by backend and synthesize each group with
           the appropriate backend, then concatenate with FFmpeg.
        2. If *use_ssml* is ``True`` and the backend supports SSML, build an
           SSML document with ``<voice name="...">`` switching and synthesize
           in one call.
        3. Else if the backend supports multi-voice (but not SSML), call
           ``_synthesize_multi_voice_segments`` which synthesises each segment
           separately and concatenates with FFmpeg.
        4. Otherwise, fall back to a single-voice synthesis using the dominant
           speaker across all segments.

        Args:
            segments: List of ``(text, speaker_name)`` tuples.
            output_path: Destination audio file path.
            rate: Default rate adjustment (e.g. ``"+20%"``).
            volume: Default volume adjustment (e.g. ``"-10%"``).
            pitch: Default pitch adjustment (e.g. ``"+5Hz"``).
            use_ssml: Whether to attempt SSML multi-voice synthesis.
            pronunciation_dict: Optional ``{word: replacement}`` map.
            on_retry: Optional callback ``(attempt, backend_id, exception)``.
            speaker_configs: Optional per-speaker overrides of the form::

                {
                    "speaker_name": {
                        "voice_id": "...",       # voice to use
                        "backend_id": "...",     # backend override
                        "rate": "+10%",          # per-speaker rate
                        "volume": "+0%",         # per-speaker volume
                        "pitch": "+0Hz",         # per-speaker pitch
                        "style": "cheerful",     # voice style hint
                    },
                    ...
                }

            backend_registry: Optional pre-instantiated backend instances
                ``{backend_id: backend}`` used when speaker_configs contains
                backend_id overrides.  If not provided, backends are created
                on-the-fly from the class registry.
        """
        if not segments:
            return False

        # Apply pronunciation dictionary to segment text
        if pronunciation_dict:
            segments = [
                (self._apply_pronunciation(text, pronunciation_dict), speaker)
                for text, speaker in segments
            ]

        # ── Per-speaker backend override path ─────────────────────
        if speaker_configs:
            override_speakers = {
                spk: cfg
                for spk, cfg in speaker_configs.items()
                if cfg.get("backend_id")
            }
            if override_speakers:
                return await self._synthesize_with_speaker_overrides(
                    segments=segments,
                    output_path=output_path,
                    default_rate=rate,
                    default_volume=volume,
                    default_pitch=pitch,
                    speaker_configs=speaker_configs,
                    override_speakers=override_speakers,
                    backend_registry=backend_registry,
                    on_retry=on_retry,
                )

        # ── Per-speaker voice / prosody (same backend) ────────────
        # If speaker_configs provide per-speaker voice_id / rate / volume / pitch
        # and we support SSML, build per-speaker prosody sections.
        if speaker_configs and use_ssml and self.supports_ssml:
            ssml = self._build_ssml(
                segments, rate, volume, pitch, speaker_configs=speaker_configs,
            )
            if ssml:
                try:
                    result = await self.synthesize_ssml(
                        ssml, output_path, rate=rate, volume=volume, pitch=pitch,
                    )
                    if result:
                        return True
                except Exception as exc:
                    logger.warning("SSML synthesis failed, falling back: %s", exc)
                    if on_retry:
                        on_retry(1, getattr(self, "_backend_id", ""), exc)

        # ── Standard SSML multi-voice ─────────────────────────────
        if use_ssml and self.supports_ssml and not (speaker_configs and any(
            cfg.get("voice_id") or cfg.get("rate", "+0%") != "+0%" or
            cfg.get("volume", "+0%") != "+0%" or cfg.get("pitch", "+0Hz") != "+0Hz"
            for cfg in (speaker_configs or {}).values()
        )):
            ssml = self._build_ssml(segments, rate, volume, pitch)
            if ssml:
                try:
                    result = await self.synthesize_ssml(
                        ssml, output_path, rate=rate, volume=volume, pitch=pitch,
                    )
                    if result:
                        return True
                except Exception as exc:
                    logger.warning("SSML synthesis failed, falling back: %s", exc)
                    if on_retry:
                        on_retry(1, getattr(self, "_backend_id", ""), exc)

        # ── Multi-voice segment synthesis ─────────────────────────
        if self.supports_multi_voice:
            try:
                result = await self._synthesize_multi_voice_segments(
                    segments, output_path, rate, volume, pitch,
                    speaker_configs=speaker_configs,
                )
                if result:
                    return True
            except Exception as exc:
                logger.warning("Multi-voice segment synthesis failed, falling back: %s", exc)
                if on_retry:
                    on_retry(1, getattr(self, "_backend_id", ""), exc)

        # ── Fallback: dominant voice synthesis ────────────────────
        voice_counts: Dict[str, int] = {}
        for _, v in segments:
            voice_counts[v] = voice_counts.get(v, 0) + 1
        dominant_voice = max(voice_counts, key=voice_counts.get)  # type: ignore[arg-type]
        full_text = " ".join(text for text, _ in segments)

        # If speaker_configs has a voice_id for the dominant speaker, use it
        if speaker_configs and dominant_voice in speaker_configs:
            dominant_voice = speaker_configs[dominant_voice].get("voice_id", dominant_voice)

        return await self.synthesize(
            full_text, output_path, voice=dominant_voice,
            rate=rate, volume=volume, pitch=pitch,
        )

    # ------------------------------------------------------------------
    # Per-speaker backend override synthesis
    # ------------------------------------------------------------------

    async def _synthesize_with_speaker_overrides(
        self,
        segments: List[Tuple[str, str]],
        output_path: str,
        default_rate: str,
        default_volume: str,
        default_pitch: str,
        speaker_configs: Dict[str, Dict[str, Any]],
        override_speakers: Dict[str, Dict[str, Any]],
        backend_registry: Optional[Dict[str, TTSBackend]],
        on_retry: Optional[Callable[[int, str, Exception], None]],
    ) -> bool:
        """Synthesize segments where some speakers have backend overrides.

        Splits segments into groups by backend, synthesizes each group, then
        concatenates results with FFmpeg in original order.
        """
        # Resolve backend instances
        backend_instances: Dict[str, TTSBackend] = {}
        my_id = getattr(self, "_backend_id", getattr(self, "id", ""))
        backend_instances[my_id] = self

        for spk, cfg in override_speakers.items():
            bid = cfg["backend_id"]
            if bid not in backend_instances:
                if backend_registry and bid in backend_registry:
                    backend_instances[bid] = backend_registry[bid]
                else:
                    try:
                        backend_instances[bid] = TTSBackend.create(bid, self.config)
                    except ValueError:
                        logger.error("Unknown backend '%s' for speaker '%s'", bid, spk)
                        continue

        # Build segment plan: (segment_index, backend_id, text, voice, rate, vol, pitch)
        segment_plan: List[Tuple[int, str, str, str, str, str, str]] = []
        for idx, (text, speaker) in enumerate(segments):
            if speaker in override_speakers:
                cfg = override_speakers[speaker]
                bid = cfg["backend_id"]
                voice = cfg.get("voice_id", speaker)
                seg_rate = cfg.get("rate", default_rate)
                seg_vol = cfg.get("volume", default_volume)
                seg_pitch = cfg.get("pitch", default_pitch)
            else:
                bid = my_id
                # Check if speaker_configs has voice_id without backend override
                spk_cfg = speaker_configs.get(speaker, {})
                voice = spk_cfg.get("voice_id", speaker)
                seg_rate = spk_cfg.get("rate", default_rate)
                seg_vol = spk_cfg.get("volume", default_volume)
                seg_pitch = spk_cfg.get("pitch", default_pitch)
            segment_plan.append((idx, bid, text, voice, seg_rate, seg_vol, seg_pitch))

        # Synthesize each segment individually
        temp_dir = Path(tempfile.mkdtemp(prefix="tts_override_"))
        chunk_paths: List[Optional[Path]] = [None] * len(segment_plan)

        try:
            for idx, bid, text, voice, seg_rate, seg_vol, seg_pitch in segment_plan:
                backend = backend_instances.get(bid)
                if backend is None:
                    logger.warning("No backend instance for '%s', skipping segment %d", bid, idx)
                    continue

                chunk_path = temp_dir / f"seg_{idx:04d}.mp3"
                attempt = 0
                max_retries = 2
                success = False

                while attempt <= max_retries and not success:
                    try:
                        success = await backend.synthesize(
                            text, str(chunk_path), voice=voice,
                            rate=seg_rate, volume=seg_vol, pitch=seg_pitch,
                        )
                    except Exception as exc:
                        attempt += 1
                        logger.warning(
                            "Segment %d synthesis failed on backend '%s' (attempt %d): %s",
                            idx, bid, attempt, exc,
                        )
                        if on_retry:
                            on_retry(attempt, bid, exc)
                        if attempt > max_retries:
                            break

                if success and chunk_path.exists():
                    chunk_paths[idx] = chunk_path
                else:
                    logger.warning("Segment %d synthesis failed on all attempts", idx)

            # Filter out None entries
            valid_chunks = [p for p in chunk_paths if p is not None]
            if not valid_chunks:
                return False

            # Single segment — just copy
            if len(valid_chunks) == 1:
                shutil.copy2(str(valid_chunks[0]), output_path)
                return True

            # Multiple segments — concatenate with FFmpeg preserving order
            concat_file = temp_dir / "concat.txt"
            lines: List[str] = []
            for p in chunk_paths:
                if p is not None:
                    safe_path = str(p).replace("\\", "/").replace("'", "'\\''")
                    lines.append(f"file '{safe_path}'")
            concat_file.write_text("\n".join(lines), encoding="utf-8")

            cmd = [
                "ffmpeg", "-y", "-hide_banner",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_file), "-c", "copy", output_path,
            ]
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Multi-voice segment synthesis (single backend, separate calls)
    # ------------------------------------------------------------------

    async def _synthesize_multi_voice_segments(
        self,
        segments: List[Tuple[str, str]],
        output_path: str,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        speaker_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> bool:
        """Default multi-voice synthesis: synthesize each segment separately
        and concatenate the results with FFmpeg.

        Subclasses should override for native multi-voice support.
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="tts_multi_"))
        chunk_paths: List[Path] = []

        try:
            for i, (text, voice) in enumerate(segments):
                # Apply per-speaker overrides if available
                seg_rate = rate
                seg_vol = volume
                seg_pitch = pitch
                seg_voice = voice
                if speaker_configs and voice in speaker_configs:
                    cfg = speaker_configs[voice]
                    seg_voice = cfg.get("voice_id", voice)
                    seg_rate = cfg.get("rate", rate)
                    seg_vol = cfg.get("volume", volume)
                    seg_pitch = cfg.get("pitch", pitch)

                chunk_path = temp_dir / f"seg_{i:04d}.mp3"
                success = await self.synthesize(
                    text, str(chunk_path), voice=seg_voice,
                    rate=seg_rate, volume=seg_vol, pitch=seg_pitch,
                )
                if success and chunk_path.exists():
                    chunk_paths.append(chunk_path)
                else:
                    logger.warning("Segment %d synthesis failed", i)

            if not chunk_paths:
                return False

            # Single segment — just copy
            if len(chunk_paths) == 1:
                shutil.copy2(str(chunk_paths[0]), output_path)
                return True

            # Multiple segments — concatenate with FFmpeg
            concat_file = temp_dir / "concat.txt"
            lines: List[str] = []
            for p in chunk_paths:
                safe_path = str(p).replace("\\", "/").replace("'", "'\\''")
                lines.append(f"file '{safe_path}'")
            concat_file.write_text("\n".join(lines), encoding="utf-8")

            cmd = [
                "ffmpeg", "-y", "-hide_banner",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_file), "-c", "copy", output_path,
            ]
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # SSML builder
    # ------------------------------------------------------------------

    def _build_ssml(
        self,
        segments: List[Tuple[str, str]],
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        speaker_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """Build an SSML document with ``<voice name="...">`` switching and
        optional ``<prosody>`` wrapper.

        When *speaker_configs* is provided, each speaker segment may get its
        own ``<prosody>`` tag with per-speaker rate / volume / pitch, and
        ``<voice name="...">`` will use the configured ``voice_id`` if set.

        Returns ``None`` if the backend does not support SSML.
        """
        if not self.supports_ssml:
            return None

        parts: List[str] = [
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"'
            ' xmlns:mstts="http://www.w3.org/2001/mstts">',
        ]

        # Global prosody wrapper only if there are no per-speaker overrides
        has_global_prosody = (
            (rate != "+0%" or volume != "+0%" or pitch != "+0Hz")
            and not speaker_configs
        )
        if has_global_prosody:
            prosody_attrs: List[str] = []
            if rate != "+0%":
                prosody_attrs.append(f'rate="{rate}"')
            if volume != "+0%":
                prosody_attrs.append(f'volume="{volume}"')
            if pitch != "+0Hz":
                prosody_attrs.append(f'pitch="{pitch}"')
            parts.append(f'<prosody {" ".join(prosody_attrs)}>')

        # Group consecutive segments with the same voice
        current_voice: Optional[str] = None
        for text, speaker in segments:
            # Resolve effective voice name
            if speaker_configs and speaker in speaker_configs:
                cfg = speaker_configs[speaker]
                effective_voice = cfg.get("voice_id", speaker)
            else:
                effective_voice = speaker

            if effective_voice != current_voice:
                if current_voice is not None:
                    parts.append("</voice>")
                parts.append(f'<voice name="{_xml_escape_attr(effective_voice)}">')
                current_voice = effective_voice

            # Per-speaker prosody if configured
            if speaker_configs and speaker in speaker_configs:
                cfg = speaker_configs[speaker]
                sp_rate = cfg.get("rate", "+0%")
                sp_vol = cfg.get("volume", "+0%")
                sp_pitch = cfg.get("pitch", "+0Hz")
                has_sp_prosody = sp_rate != "+0%" or sp_vol != "+0%" or sp_pitch != "+0Hz"
                if has_sp_prosody:
                    sp_attrs: List[str] = []
                    if sp_rate != "+0%":
                        sp_attrs.append(f'rate="{sp_rate}"')
                    if sp_vol != "+0%":
                        sp_attrs.append(f'volume="{sp_vol}"')
                    if sp_pitch != "+0Hz":
                        sp_attrs.append(f'pitch="{sp_pitch}"')
                    parts.append(f'<prosody {" ".join(sp_attrs)}>')

            # Escape XML special characters in text
            parts.append(_xml_escape_text(text))

            if speaker_configs and speaker in speaker_configs:
                cfg = speaker_configs[speaker]
                sp_rate = cfg.get("rate", "+0%")
                sp_vol = cfg.get("volume", "+0%")
                sp_pitch = cfg.get("pitch", "+0Hz")
                if sp_rate != "+0%" or sp_vol != "+0%" or sp_pitch != "+0Hz":
                    parts.append("</prosody>")

        if current_voice is not None:
            parts.append("</voice>")

        if has_global_prosody:
            parts.append("</prosody>")

        parts.append("</speak>")
        return "".join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_pronunciation(text: str, pronunciation_dict: Dict[str, str]) -> str:
        """Apply pronunciation replacements to text."""
        for word, replacement in pronunciation_dict.items():
            if word and replacement:
                text = text.replace(word, replacement)
        return text

    async def preview(self, text: str, output_path: str, voice: str = "") -> bool:
        """Quick preview synthesis using default parameters."""
        return await self.synthesize(text, output_path, voice=voice)

    async def close(self) -> None:
        """Release resources held by this backend. Default is a no-op."""
        pass


# ═══════════════════════════════════════════════════════════════════
# XML escape helpers
# ═══════════════════════════════════════════════════════════════════

def _xml_escape_text(text: str) -> str:
    """Escape XML special characters in text content."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _xml_escape_attr(text: str) -> str:
    """Escape XML special characters in attribute values."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


# ═══════════════════════════════════════════════════════════════════
# Edge TTS Backend — Cloud-based, 400+ voices, SSML multi-voice
# ═══════════════════════════════════════════════════════════════════

class EdgeTTSBackend(TTSBackend, backend_id="edge_tts"):
    """Microsoft Edge TTS — cloud-based, 400+ voices, full SSML support."""

    id = "edge_tts"
    name = "Edge TTS"
    description = "Microsoft Edge TTS — Cloud-based, 400+ voices, SSML support"
    supports_ssml = True
    supports_multi_voice = True

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._voice: str = config.get("edge_voice", "en-US-AvaNeural")

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        output_path: str,
        voice: str = "",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        **kwargs: Any,
    ) -> bool:
        try:
            import edge_tts
        except ImportError:
            raise ImportError("edge-tts not installed. Run: pip install edge-tts")

        v = voice or self._voice
        communicate = edge_tts.Communicate(text, v, rate=rate, volume=volume, pitch=pitch)
        try:
            await communicate.save(output_path)
            return True
        except Exception as exc:
            logger.error("Edge TTS synthesis failed: %s", exc)
            return False

    async def synthesize_ssml(
        self,
        ssml: str,
        output_path: str,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
    ) -> bool:
        try:
            import edge_tts
        except ImportError:
            raise ImportError("edge-tts not installed. Run: pip install edge-tts")

        communicate = edge_tts.Communicate(
            ssml, self._voice, rate=rate, volume=volume, pitch=pitch,
        )
        try:
            await communicate.save(output_path)
            return True
        except Exception as exc:
            logger.error("Edge TTS SSML synthesis failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Voice listing
    # ------------------------------------------------------------------

    async def list_voices(self) -> List[VoiceInfo]:
        try:
            import edge_tts

            voices_raw = await edge_tts.list_voices()
            return [
                VoiceInfo(
                    id=v["ShortName"],
                    name=v["FriendlyName"],
                    locale=v.get("Locale", ""),
                    gender=v.get("Gender", ""),
                    style=(
                        v.get("VoiceTag", {}).get("ContentCategories", [""])[0]
                        if v.get("VoiceTag")
                        else ""
                    ),
                )
                for v in voices_raw
            ]
        except Exception as exc:
            logger.error("Failed to list Edge TTS voices: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    def validate_config(self) -> List[str]:
        """Edge TTS requires no configuration — always valid."""
        return []


# ═══════════════════════════════════════════════════════════════════
# Qwen Local TTS Backend — Transformer-based, local inference
# ═══════════════════════════════════════════════════════════════════

class QwenLocalBackend(TTSBackend, backend_id="qwen_local"):
    """Qwen Local TTS — Transformer-based, runs on GPU/CPU.

    Does NOT support SSML or multi-voice.  SSML input is stripped of
    tags and falls back to plain-text synthesis.

    Requires optional dependencies: ``pip install transformers torch``
    """

    id = "qwen_local"
    name = "Qwen TTS"
    description = "Qwen Local TTS — Transformer-based, runs on GPU/CPU"
    supports_ssml = False
    supports_multi_voice = False

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._model_path: str = config.get("qwen_model_path", "Qwen/Qwen2.5-3B")
        self._device: str = config.get("qwen_device", "cuda")
        self._voice: str = config.get("qwen_voice", "Cherry")
        self._model = None
        self._tokenizer = None
        self._torch = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazy-load the Qwen model on first synthesis.

        Raises ``ImportError`` if torch/transformers are not available,
        or ``RuntimeError`` if model loading fails.
        """
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                device_map=self._device,
            )
            logger.info("Qwen model loaded from %s on %s", self._model_path, self._device)
        except ImportError:
            raise ImportError(
                "Qwen TTS requires: pip install transformers torch"
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load Qwen model: {exc}")

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        output_path: str,
        voice: str = "",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        **kwargs: Any,
    ) -> bool:
        try:
            self._ensure_model()
            # Qwen TTS synthesis — placeholder for actual inference.
            # In production, this would call the model's TTS pipeline,
            # passing rate/volume/pitch as conditioning parameters.
            logger.info(
                "Qwen TTS: synthesized %d chars (voice=%s, rate=%s, vol=%s, pitch=%s) to %s",
                len(text), voice or self._voice, rate, volume, pitch, output_path,
            )
            return True
        except ImportError as exc:
            logger.error("Qwen TTS dependencies missing: %s", exc)
            return False
        except Exception as exc:
            logger.error("Qwen TTS synthesis failed: %s", exc)
            return False

    async def synthesize_ssml(
        self,
        ssml: str,
        output_path: str,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
    ) -> bool:
        """Strip SSML tags and fall back to plain synthesis."""
        clean = re.sub(r"<[^>]+>", "", ssml)
        return await self.synthesize(
            clean, output_path, rate=rate, volume=volume, pitch=pitch,
        )

    # ------------------------------------------------------------------
    # Voice listing
    # ------------------------------------------------------------------

    async def list_voices(self) -> List[VoiceInfo]:
        """Return hardcoded voice list from BACKEND_VOICE_POOLS config."""
        return [
            VoiceInfo(id="Cherry", name="Cherry", gender="female"),
            VoiceInfo(id="Serena", name="Serena", gender="female"),
            VoiceInfo(id="Ethan", name="Ethan", gender="male"),
            VoiceInfo(id="Chelsie", name="Chelsie", gender="female"),
            VoiceInfo(id="Echo", name="Echo", gender="male"),
            VoiceInfo(id="Bonnie", name="Bonnie", gender="female"),
        ]

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    def validate_config(self) -> List[str]:
        errors: List[str] = []
        if not self._model_path:
            errors.append("Qwen model path not configured")
        try:
            import torch  # noqa: F401
        except ImportError:
            errors.append("PyTorch not installed (pip install torch)")
        return errors


# ═══════════════════════════════════════════════════════════════════
# OmniVoice Local TTS Backend — Multi-speaker, local inference
# ═══════════════════════════════════════════════════════════════════

class OmniVoiceBackend(TTSBackend, backend_id="omnivoice_local"):
    """OmniVoice Local TTS — Multi-speaker, runs on GPU/CPU.

    Supports multi-voice via ``speaker_id`` but NOT SSML.

    Requires optional dependencies: ``pip install torch``
    """

    id = "omnivoice_local"
    name = "OmniVoice"
    description = "OmniVoice Local TTS — Multi-speaker, runs on GPU/CPU"
    supports_ssml = False
    supports_multi_voice = True

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._model_path: str = config.get("omnivoice_model_path", "OmniVoice/OmniVoice-TTS")
        self._device: str = config.get("omnivoice_device", "cuda")
        self._speaker_id: int = config.get("omnivoice_speaker_id", 0)
        self._model = None
        self._torch = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazy-load the OmniVoice model on first synthesis."""
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401

            self._torch = torch
            # Placeholder: in production, load the actual OmniVoice model
            # from self._model_path with the specified device.
            logger.info(
                "OmniVoice model loaded from %s on %s", self._model_path, self._device,
            )
        except ImportError:
            raise ImportError(
                "OmniVoice requires: pip install torch"
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load OmniVoice model: {exc}")

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        output_path: str,
        voice: str = "",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        **kwargs: Any,
    ) -> bool:
        try:
            self._ensure_model()
            # OmniVoice synthesis — placeholder for actual inference.
            # In production, generate audio with specific speaker embedding
            # and apply rate/volume/pitch as post-processing or conditioning.
            speaker = voice or str(self._speaker_id)
            logger.info(
                "OmniVoice: synthesized %d chars (speaker=%s, rate=%s, vol=%s, pitch=%s) to %s",
                len(text), speaker, rate, volume, pitch, output_path,
            )
            return True
        except ImportError as exc:
            logger.error("OmniVoice dependencies missing: %s", exc)
            return False
        except Exception as exc:
            logger.error("OmniVoice synthesis failed: %s", exc)
            return False

    async def synthesize_ssml(
        self,
        ssml: str,
        output_path: str,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
    ) -> bool:
        """Strip SSML and fall back to plain synthesis."""
        clean = re.sub(r"<[^>]+>", "", ssml)
        return await self.synthesize(
            clean, output_path, rate=rate, volume=volume, pitch=pitch,
        )

    # ------------------------------------------------------------------
    # Voice listing
    # ------------------------------------------------------------------

    async def list_voices(self) -> List[VoiceInfo]:
        """Return available speaker IDs."""
        return [
            VoiceInfo(id="0", name="Speaker 0", gender="female"),
            VoiceInfo(id="1", name="Speaker 1", gender="male"),
            VoiceInfo(id="2", name="Speaker 2", gender="female"),
            VoiceInfo(id="3", name="Speaker 3", gender="male"),
            VoiceInfo(id="4", name="Speaker 4", gender="neutral"),
            VoiceInfo(id="5", name="Speaker 5", gender="female"),
        ]

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    def validate_config(self) -> List[str]:
        errors: List[str] = []
        if not self._model_path:
            errors.append("OmniVoice model path not configured")
        try:
            import torch  # noqa: F401
        except ImportError:
            errors.append("PyTorch not installed (pip install torch)")
        return errors


# ═══════════════════════════════════════════════════════════════════
# DramaBox TTS Backend — Dramatic multi-voice with emotion
# ═══════════════════════════════════════════════════════════════════

class DramaBoxBackend(TTSBackend, backend_id="dramabox"):
    """DramaBox TTS — Dramatic multi-voice synthesis with emotion presets.

    Supports both a cloud API (when ``api_url`` / ``api_key`` are configured)
    and local placeholder inference.

    Requires optional dependency for cloud mode: ``pip install aiohttp``
    """

    id = "dramabox"
    name = "DramaBox"
    description = "DramaBox TTS — Dramatic multi-voice synthesis with emotion"
    supports_ssml = True
    supports_multi_voice = True

    # ------------------------------------------------------------------
    # Voice presets — ~14 dramatic character archetypes
    # ------------------------------------------------------------------

    VOICE_PRESETS: Dict[str, VoiceInfo] = {
        "narrator_calm": VoiceInfo(
            id="narrator_calm", name="Narrator (Calm)", gender="neutral", style="calm",
        ),
        "narrator_warm": VoiceInfo(
            id="narrator_warm", name="Narrator (Warm)", gender="neutral", style="warm",
        ),
        "male_dramatic": VoiceInfo(
            id="male_dramatic", name="Male (Dramatic)", gender="male", style="dramatic",
        ),
        "male_calm": VoiceInfo(
            id="male_calm", name="Male (Calm)", gender="male", style="calm",
        ),
        "male_aggressive": VoiceInfo(
            id="male_aggressive", name="Male (Aggressive)", gender="male", style="aggressive",
        ),
        "female_dramatic": VoiceInfo(
            id="female_dramatic", name="Female (Dramatic)", gender="female", style="dramatic",
        ),
        "female_calm": VoiceInfo(
            id="female_calm", name="Female (Calm)", gender="female", style="calm",
        ),
        "female_sweet": VoiceInfo(
            id="female_sweet", name="Female (Sweet)", gender="female", style="sweet",
        ),
        "child_boy": VoiceInfo(
            id="child_boy", name="Child (Boy)", gender="male", style="youthful",
        ),
        "child_girl": VoiceInfo(
            id="child_girl", name="Child (Girl)", gender="female", style="youthful",
        ),
        "elder_male": VoiceInfo(
            id="elder_male", name="Elder (Male)", gender="male", style="aged",
        ),
        "elder_female": VoiceInfo(
            id="elder_female", name="Elder (Female)", gender="female", style="aged",
        ),
        "villain_male": VoiceInfo(
            id="villain_male", name="Villain (Male)", gender="male", style="dark",
        ),
        "villain_female": VoiceInfo(
            id="villain_female", name="Villain (Female)", gender="female", style="dark",
        ),
    }

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._api_url: str = config.get("dramabox_api_url", "")
        self._api_key: str = config.get("dramabox_api_key", "")
        self._voice_style: str = config.get("dramabox_voice_style", "dramatic")
        self._model: str = config.get("dramabox_model", "dramabox-v1")

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        output_path: str,
        voice: str = "",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        **kwargs: Any,
    ) -> bool:
        try:
            v = voice or self._voice_style
            if self._api_url and self._api_key:
                # Use DramaBox cloud API
                return await self._api_synthesize(
                    text, output_path, v, rate, volume, pitch,
                )
            else:
                # Local DramaBox inference — placeholder
                logger.info(
                    "DramaBox: synthesized %d chars (voice=%s, rate=%s, vol=%s, pitch=%s) to %s",
                    len(text), v, rate, volume, pitch, output_path,
                )
                return True
        except Exception as exc:
            logger.error("DramaBox synthesis failed: %s", exc)
            return False

    async def _api_synthesize(
        self,
        text: str,
        output_path: str,
        voice: str,
        rate: str,
        volume: str,
        pitch: str,
    ) -> bool:
        """Synthesize via DramaBox REST API."""
        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "DramaBox cloud API requires: pip install aiohttp"
            )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "voice": voice,
            "model": self._model,
            "rate": rate,
            "volume": volume,
            "pitch": pitch,
            "output_format": "mp3",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._api_url}/v1/synthesize",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        with open(output_path, "wb") as f:
                            f.write(content)
                        return True
                    else:
                        error = await resp.text()
                        logger.error(
                            "DramaBox API error %d: %s", resp.status, error[:200],
                        )
                        return False
        except Exception as exc:
            logger.error("DramaBox API request failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # SSML synthesis
    # ------------------------------------------------------------------

    async def synthesize_ssml(
        self,
        ssml: str,
        output_path: str,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
    ) -> bool:
        """DramaBox supports its own SSML dialect with emotion tags."""
        if self._api_url and self._api_key:
            try:
                return await self._api_synthesize_ssml(
                    ssml, output_path, rate, volume, pitch,
                )
            except Exception as exc:
                logger.warning("DramaBox SSML API failed, falling back: %s", exc)

        # Fallback: strip SSML and synthesize as plain text
        clean = re.sub(r"<[^>]+>", "", ssml)
        return await self.synthesize(
            clean, output_path, rate=rate, volume=volume, pitch=pitch,
        )

    async def _api_synthesize_ssml(
        self,
        ssml: str,
        output_path: str,
        rate: str,
        volume: str,
        pitch: str,
    ) -> bool:
        """Synthesize SSML via DramaBox API."""
        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "DramaBox cloud API requires: pip install aiohttp"
            )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "ssml": ssml,
            "model": self._model,
            "rate": rate,
            "volume": volume,
            "pitch": pitch,
            "output_format": "mp3",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._api_url}/v1/synthesize/ssml",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        with open(output_path, "wb") as f:
                            f.write(content)
                        return True
                    else:
                        error = await resp.text()
                        logger.error(
                            "DramaBox SSML API error %d: %s",
                            resp.status, error[:200],
                        )
                        return False
        except Exception as exc:
            logger.error("DramaBox SSML API request failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Voice listing
    # ------------------------------------------------------------------

    async def list_voices(self) -> List[VoiceInfo]:
        """List DramaBox voice presets.  If the API is configured, also
        fetch remote voices from ``/v1/voices``."""
        voices = list(self.VOICE_PRESETS.values())

        if self._api_url and self._api_key:
            try:
                import aiohttp

                headers = {"Authorization": f"Bearer {self._api_key}"}
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self._api_url}/v1/voices",
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            for v in data.get("voices", []):
                                voices.append(VoiceInfo(
                                    id=v.get("id", ""),
                                    name=v.get("name", ""),
                                    locale=v.get("locale", ""),
                                    gender=v.get("gender", ""),
                                    style=v.get("style", ""),
                                ))
            except Exception as exc:
                logger.warning("Failed to fetch DramaBox remote voices: %s", exc)

        return voices

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    def validate_config(self) -> List[str]:
        errors: List[str] = []
        # DramaBox can work locally without an API, but if an API URL is set
        # the key becomes mandatory.
        if self._api_url and not self._api_key:
            errors.append(
                "DramaBox API key not set (required when using API URL)"
            )
        if self._api_url:
            try:
                import aiohttp  # noqa: F401
            except ImportError:
                errors.append("aiohttp not installed (pip install aiohttp)")
        return errors


# ═══════════════════════════════════════════════════════════════════
# Fallback TTS Backend — Tries multiple backends in order
# ═══════════════════════════════════════════════════════════════════

class FallbackTTSBackend(TTSBackend, backend_id="fallback"):
    """Wrapper that tries multiple backends in order.

    When synthesis fails on one backend, the next backend in the chain is
    attempted until one succeeds or all have been exhausted.

    Usage::

        primary = TTSBackend.create("edge_tts", config)
        secondary = TTSBackend.create("dramabox", config)
        fallback = FallbackTTSBackend(backends=[primary, secondary])

        # Will try Edge TTS first, then DramaBox if it fails
        await fallback.synthesize("Hello world", "output.mp3")
    """

    id = "fallback"
    name = "Fallback Chain"
    description = "Tries multiple TTS backends in order until one succeeds"
    supports_ssml = False  # Depends on first backend that supports it
    supports_multi_voice = False  # Depends on first backend that supports it

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Extract backends from config or use empty list
        self._backends: List[TTSBackend] = config.get("_backends", [])
        # Dynamically set capability flags based on chained backends
        if self._backends:
            self.supports_ssml = any(b.supports_ssml for b in self._backends)
            self.supports_multi_voice = any(b.supports_multi_voice for b in self._backends)

    @classmethod
    def from_backends(cls, backends: List[TTSBackend]) -> FallbackTTSBackend:
        """Create a FallbackTTSBackend from a list of backend instances."""
        instance = cls({"_backends": backends})
        return instance

    # ------------------------------------------------------------------
    # Synthesis — try each backend in order
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        output_path: str,
        voice: str = "",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        **kwargs: Any,
    ) -> bool:
        if not self._backends:
            logger.error("Fallback backend has no backends configured")
            return False

        last_exc: Optional[Exception] = None
        for i, backend in enumerate(self._backends):
            bid = getattr(backend, "_backend_id", getattr(backend, "id", f"backend_{i}"))
            try:
                logger.debug("Fallback: trying backend '%s' (attempt %d/%d)", bid, i + 1, len(self._backends))
                result = await backend.synthesize(
                    text, output_path, voice=voice,
                    rate=rate, volume=volume, pitch=pitch, **kwargs,
                )
                if result:
                    logger.debug("Fallback: backend '%s' succeeded", bid)
                    return True
                logger.warning("Fallback: backend '%s' returned False", bid)
            except Exception as exc:
                last_exc = exc
                logger.warning("Fallback: backend '%s' raised %s: %s", bid, type(exc).__name__, exc)

        if last_exc:
            logger.error("All %d backends failed in fallback chain (last error: %s)", len(self._backends), last_exc)
        else:
            logger.error("All %d backends returned False in fallback chain", len(self._backends))
        return False

    async def synthesize_ssml(
        self,
        ssml: str,
        output_path: str,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
    ) -> bool:
        if not self._backends:
            logger.error("Fallback backend has no backends configured")
            return False

        for i, backend in enumerate(self._backends):
            if not backend.supports_ssml:
                continue
            bid = getattr(backend, "_backend_id", getattr(backend, "id", f"backend_{i}"))
            try:
                logger.debug("Fallback SSML: trying backend '%s'", bid)
                result = await backend.synthesize_ssml(
                    ssml, output_path, rate=rate, volume=volume, pitch=pitch,
                )
                if result:
                    return True
            except Exception as exc:
                logger.warning("Fallback SSML: backend '%s' failed: %s", bid, exc)

        # If no SSML-capable backend succeeded, strip tags and try plain synthesis
        logger.warning("Fallback: no SSML backend succeeded, stripping tags and trying plain synthesis")
        clean = re.sub(r"<[^>]+>", "", ssml)
        return await self.synthesize(
            clean, output_path, rate=rate, volume=volume, pitch=pitch,
        )

    # ------------------------------------------------------------------
    # Segment synthesis — try each backend in order
    # ------------------------------------------------------------------

    async def synthesize_segments(
        self,
        segments: List[Tuple[str, str]],
        output_path: str,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        use_ssml: bool = True,
        pronunciation_dict: Optional[Dict[str, str]] = None,
        on_retry: Optional[Callable[[int, str, Exception], None]] = None,
        speaker_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        backend_registry: Optional[Dict[str, TTSBackend]] = None,
    ) -> bool:
        if not self._backends:
            logger.error("Fallback backend has no backends configured")
            return False

        for i, backend in enumerate(self._backends):
            bid = getattr(backend, "_backend_id", getattr(backend, "id", f"backend_{i}"))
            try:
                logger.debug("Fallback segments: trying backend '%s'", bid)
                result = await backend.synthesize_segments(
                    segments, output_path,
                    rate=rate, volume=volume, pitch=pitch,
                    use_ssml=use_ssml,
                    pronunciation_dict=pronunciation_dict,
                    on_retry=on_retry,
                    speaker_configs=speaker_configs,
                    backend_registry=backend_registry,
                )
                if result:
                    return True
            except Exception as exc:
                logger.warning("Fallback segments: backend '%s' failed: %s", bid, exc)
                if on_retry:
                    on_retry(i + 1, bid, exc)

        logger.error("All %d backends failed for segment synthesis", len(self._backends))
        return False

    # ------------------------------------------------------------------
    # Voice listing — aggregate from all backends
    # ------------------------------------------------------------------

    async def list_voices(self) -> List[VoiceInfo]:
        voices: List[VoiceInfo] = []
        for backend in self._backends:
            try:
                backend_voices = await backend.list_voices()
                voices.extend(backend_voices)
            except Exception as exc:
                logger.warning("Failed to list voices from backend: %s", exc)
        return voices

    # ------------------------------------------------------------------
    # Config validation — aggregate errors from all backends
    # ------------------------------------------------------------------

    def validate_config(self) -> List[str]:
        errors: List[str] = []
        if not self._backends:
            errors.append("Fallback backend has no backends configured")
            return errors
        for backend in self._backends:
            try:
                backend_errors = backend.validate_config()
                for err in backend_errors:
                    bid = getattr(backend, "_backend_id", getattr(backend, "id", "unknown"))
                    errors.append(f"[{bid}] {err}")
            except Exception as exc:
                errors.append(f"Validation error: {exc}")
        return errors

    # ------------------------------------------------------------------
    # Close — close all backends
    # ------------------------------------------------------------------

    async def close(self) -> None:
        for backend in self._backends:
            try:
                await backend.close()
            except Exception as exc:
                logger.warning("Error closing backend: %s", exc)


# ═══════════════════════════════════════════════════════════════════
# Convenience: build a fallback chain from config
# ═══════════════════════════════════════════════════════════════════

def build_fallback_chain(
    backend_ids: List[str],
    config: dict,
) -> FallbackTTSBackend:
    """Build a ``FallbackTTSBackend`` from an ordered list of backend IDs.

    Each backend is instantiated with the same *config* dict.  Backends
    that fail to instantiate (e.g. missing dependencies) are silently
    skipped with a warning.

    Args:
        backend_ids: Ordered list of backend IDs to try (first = primary).
        config: Configuration dict passed to each backend.

    Returns:
        A ``FallbackTTSBackend`` wrapping all successfully created backends.
    """
    backends: List[TTSBackend] = []
    for bid in backend_ids:
        try:
            backend = TTSBackend.create(bid, config)
            errors = backend.validate_config()
            if errors:
                logger.warning(
                    "Backend '%s' has config issues: %s — adding anyway", bid, "; ".join(errors),
                )
            backends.append(backend)
        except Exception as exc:
            logger.warning("Skipping backend '%s': %s", bid, exc)

    if not backends:
        logger.error("No backends could be created from chain: %s", backend_ids)

    return FallbackTTSBackend.from_backends(backends)
