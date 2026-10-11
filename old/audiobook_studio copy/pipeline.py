"""
Pure-async audiobook generation pipeline + QThread engine wrapper.

Features: pre-flight checks, disk space estimation, pause/resume,
configurable quality presets, SSML synthesis, pronunciation support,
partial chapter regeneration, generation history logging, per-chapter
normalization, crossfade between chapters, room tone overlay,
fallback backend chain, and per-speaker voice overrides.

The AudiobookEngine class bridges async callbacks → Qt signals.
"""

from __future__ import annotations

import asyncio
import logging
import random
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QThread, Signal

from config import EngineConfig, GenerationEntry, QualityPreset
from text_processor import TextProcessor, Chapter, DialogueChunk
from tts_backends import TTSBackend

# FallbackTTSBackend may not exist in tts_backends yet; provide graceful fallback
try:
    from tts_backends import FallbackTTSBackend
except ImportError:
    FallbackTTSBackend = None  # type: ignore[assignment,misc]

from audio_postprocessor import AudioPostProcessor, ChapterMarker, ProcessingResult

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  Pipeline Exceptions
# ══════════════════════════════════════════════════════════════════

class PipelineError(Exception): pass
class ConfigurationError(PipelineError): pass
class PipelineCancelled(PipelineError): pass
class PipelinePaused(PipelineError): pass
class SynthesisError(PipelineError): pass
class InsufficientSpaceError(PipelineError): pass


# ══════════════════════════════════════════════════════════════════
#  Pipeline Result
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PipelineResult:
    output_files: Tuple[Path, ...]
    total_duration_ms: float
    total_chunks: int
    failed_chunks: int
    elapsed_seconds: float
    chapter_count: int

    @property
    def success(self) -> bool: return len(self.output_files) > 0

    @property
    def duration_str(self) -> str: return AudioPostProcessor.format_duration(self.total_duration_ms)

    @property
    def elapsed_str(self) -> str:
        m, s = divmod(int(self.elapsed_seconds), 60)
        return f"{m}m {s}s"

    @property
    def summary_html(self) -> str:
        parts = [f"<b>Audiobook produced in {self.elapsed_str}</b><br>"]
        if self.chapter_count > 1: parts.append(f"Chapters: {self.chapter_count} &bull; ")
        parts.append(f"Chunks: {self.total_chunks}")
        if self.failed_chunks: parts.append(f", <font color='orange'>{self.failed_chunks} failed</font>")
        parts.append(f"<br>Duration: {self.duration_str}<br>")
        for fp in self.output_files:
            try:
                mb = fp.stat().st_size / (1024 * 1024)
                ext = fp.suffix.upper().lstrip(".")
                parts.append(f"{fp.name} ({ext}, {mb:.1f} MB)<br>")
            except OSError: pass
        return "".join(parts)


# ══════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════

@contextmanager
def temp_dir(prefix: str = "audiobook_"):
    d = Path(tempfile.mkdtemp(prefix=prefix))
    try: yield d
    finally: shutil.rmtree(d, ignore_errors=True)


async def retry_async(
    coro_fn: Callable[..., asyncio.coroutines], *args,
    max_attempts: int = 3, base_delay: float = 4.0,
    on_retry: Optional[Callable[[int, str], None]] = None, **kwargs,
) -> bool:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = await coro_fn(*args, **kwargs)
            if result is False: raise RuntimeError("Backend returned False")
            return True
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1.0)
                if on_retry: on_retry(attempt, str(exc)[:80])
                await asyncio.sleep(delay)
    raise SynthesisError(str(last_exc))


# ══════════════════════════════════════════════════════════════════
#  Crossfade & Room Tone Helpers
# ══════════════════════════════════════════════════════════════════

def _apply_crossfade(
    chapter_paths: List[Path],
    crossfade_ms: int,
    output_dir: Path,
    sample_rate: int = 44100,
) -> List[Path]:
    """Apply crossfade between consecutive chapter audio files.

    Returns a new list of chapter paths (possibly with intermediate
    crossfaded files).  If *crossfade_ms* <= 0 the original list is
    returned unchanged.
    """
    if crossfade_ms <= 0 or len(chapter_paths) < 2:
        return chapter_paths

    crossfade_sec = crossfade_ms / 1000.0
    result: List[Path] = []

    for i, path in enumerate(chapter_paths):
        if i == 0:
            result.append(path)
            continue

        prev = result[-1]
        merged = output_dir / f"xfade_{i:03d}.mp3"

        # FFmpeg crossfade using acrossfade filter
        # acrossfade requires two inputs; fade out prev, fade in current
        cmd = [
            "ffmpeg", "-y",
            "-i", str(prev),
            "-i", str(path),
            "-filter_complex",
            f"[0:a][1:a]acrossfade=d={crossfade_sec}:c1=tri:c2=tri[aout]",
            "-map", "[aout]",
            "-ar", str(sample_rate),
            "-b:a", "192k",
            str(merged),
        ]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300, check=False,
            )
            if proc.returncode == 0 and merged.exists():
                result[-1] = merged
                result.append(path)
            else:
                logger.warning("Crossfade failed between chapters %d-%d, falling back to concat", i, i + 1)
                result.append(path)
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("Crossfade error: %s, falling back to concat", exc)
            result.append(path)

    return result


def _apply_room_tone(
    audio_path: Path,
    room_tone: str,
    room_tone_file: str,
    output_path: Path,
    sample_rate: int = 44100,
) -> Path:
    """Overlay a room-tone / ambient noise bed onto the final audio.

    Uses the *ROOM_TONE_PRESETS* volume level from config to mix a subtle
    background.  Returns the path to the processed file (may be *audio_path*
    unchanged if room_tone is ``"none"``).

    If a custom *room_tone_file* is provided it takes priority; otherwise
    FFmpeg's ``anoisesrc`` filter generates pink noise at the appropriate
    volume.
    """
    if room_tone == "none":
        return audio_path

    from config import ROOM_TONE_PRESETS

    preset = ROOM_TONE_PRESETS.get(room_tone, ROOM_TONE_PRESETS.get("subtle", {}))
    volume_db = preset.get("volume_db", -35)

    # Get duration of the source audio
    duration_cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        proc = subprocess.run(
            duration_cmd, capture_output=True, text=True, timeout=30, check=False,
        )
        duration = float(proc.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired, OSError):
        return audio_path

    if room_tone_file and Path(room_tone_file).exists():
        # Use custom room tone file
        tone_input = ["-i", str(room_tone_file)]
        filter_complex = (
            f"[0:a][1:a]amix=inputs=2:duration=first"
            f":dropout_transition=2:weights=1 0.3[vout]"
        )
        map_spec = ["-map", "[vout]"]
    else:
        # Generate pink noise using FFmpeg
        tone_input = [
            "-f", "lavfi",
            "-i", f"anoisesrc=d={duration}:c=pink:r={sample_rate}:a=0.01",
        ]
        filter_complex = (
            f"[1:a]volume={volume_db}dB[bg];"
            f"[0:a][bg]amix=inputs=2:duration=first"
            f":dropout_transition=2:weights=1 0.3[vout]"
        )
        map_spec = ["-map", "[vout]"]

    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        *tone_input,
        "-filter_complex", filter_complex,
        *map_spec,
        "-ar", str(sample_rate),
        "-b:a", "192k",
        str(output_path),
    ]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, check=False,
        )
        if proc.returncode == 0 and output_path.exists():
            return output_path
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Room tone overlay failed: %s", exc)

    return audio_path


# ══════════════════════════════════════════════════════════════════
#  AudiobookPipeline — Pure async, no Qt dependency
# ══════════════════════════════════════════════════════════════════

class AudiobookPipeline:
    def __init__(
        self, config: EngineConfig, backend: TTSBackend,
        on_progress: Optional[Callable[[int, str], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_chapter: Optional[Callable[[int, int], None]] = None,
        on_generation: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.cfg = config
        self.backend = backend
        self._progress = on_progress or (lambda p, m: None)
        self._log = on_log or (lambda m: None)
        self._chapter_cb = on_chapter or (lambda c, t: None)
        self._generation_cb = on_generation or (lambda d: None)
        self._cancel = asyncio.Event()
        self._pause = asyncio.Event()
        self._pause.set()  # Not paused initially
        self._start = 0.0
        self._completed = 0
        self._total = 0
        self._failed = 0

    def cancel(self) -> None: self._cancel.set()

    def pause(self) -> None:
        self._pause.clear()
        self._log('<font color="#f9e2af">Paused</font>')

    def resume(self) -> None:
        self._pause.set()
        self._log('<font color="#a6e3a1">Resumed</font>')

    @property
    def is_paused(self) -> bool: return not self._pause.is_set()

    async def _wait_if_paused(self) -> None:
        """Wait until unpaused. Raises PipelineCancelled if cancelled while paused."""
        while not self._pause.is_set():
            if self._cancel.is_set():
                raise PipelineCancelled("Cancelled while paused")
            await asyncio.sleep(0.25)

    def _check_cancel(self) -> None:
        if self._cancel.is_set(): raise PipelineCancelled("Cancelled by user")

    def _eta(self) -> str:
        if self._completed < 2: return "calculating..."
        elapsed = time.monotonic() - self._start
        rate = self._completed / elapsed
        remaining = (self._total - self._completed) / max(rate, 0.001)
        m, s = divmod(int(remaining), 60)
        return f"{m}m {s}s" if m else f"{s}s"

    def _resolve_voice(self, speaker: str) -> str:
        """Map detected speaker to configured voice, falling back to defaults."""
        if speaker in self.cfg.voice_map:
            return self.cfg.voice_map[speaker]
        if "Narrator" in self.cfg.voice_map and speaker != "Dialogue":
            return self.cfg.voice_map["Narrator"]
        if "Dialogue" in self.cfg.voice_map:
            return self.cfg.voice_map["Dialogue"]
        return self.cfg.voice

    def _resolve_speaker_params(self, speaker: str) -> Dict[str, str]:
        """Resolve per-speaker overrides (voice, rate, volume, pitch).

        Returns a dict with keys: voice, rate, volume, pitch.
        Falls back to global config values when no override is set.
        """
        overrides = self.cfg.speaker_voice_configs.get(speaker, {})
        voice = overrides.get("voice_id") or self._resolve_voice(speaker)
        rate = overrides.get("rate") or self.cfg.rate
        volume = overrides.get("volume") or self.cfg.volume
        pitch = overrides.get("pitch") or self.cfg.pitch
        return {"voice": voice, "rate": rate, "volume": volume, "pitch": pitch}

    # ── Pre-flight Checks ───────────────────────────────────────

    def _preflight_check(self, source: str, is_raw_text: bool) -> List[str]:
        """Run pre-flight validation checks. Returns list of error messages."""
        errors = []

        if not is_raw_text:
            if not source or not Path(source).exists():
                errors.append("Source file does not exist")
            elif not is_raw_text and Path(source).stat().st_size == 0:
                errors.append("Source file is empty")

        ffmpeg_ok, ffmpeg_msg = AudioPostProcessor.check_ffmpeg()
        if not ffmpeg_ok:
            errors.append(ffmpeg_msg)

        backend_errors = self.backend.validate_config()
        errors.extend(backend_errors)

        if not is_raw_text and source and Path(source).exists():
            try:
                text = TextProcessor._collect_text(source, False)
                words = len(text.split())
                est_mb = TextProcessor.estimate_disk_mb(words, self.cfg.bitrate, self.cfg.output_format)
                required_mb = est_mb * 3
                out_dir = Path(self.cfg.cover_art).parent if self.cfg.cover_art else Path(".")
                try:
                    free_mb = shutil.disk_usage(out_dir).free / (1024 * 1024)
                    if free_mb < required_mb:
                        errors.append(
                            f"Insufficient disk space: {free_mb:.0f} MB free, "
                            f"~{required_mb:.0f} MB required"
                        )
                except OSError:
                    pass
            except Exception:
                pass

        return errors

    # ── Main Entry Point ───────────────────────────────────────

    async def run(self, source: str, output_path: str, is_raw_text: bool) -> PipelineResult:
        self._start = time.monotonic()
        self._completed = 0
        self._failed = 0

        self._progress(1, "Pre-flight checks...")
        preflight_errors = self._preflight_check(source, is_raw_text)
        if preflight_errors:
            raise ConfigurationError("\n".join(preflight_errors))

        self._check_cancel()
        await self._wait_if_paused()

        preset_name = self.cfg.quality_preset.replace("_", " ").title()
        self._log(
            f"Quality: {preset_name} | {self.cfg.target_lufs} LUFS | "
            f"{self.cfg.sample_rate} Hz | {self.cfg.bitrate}"
        )

        if self.cfg.chapter_aware:
            result = await self._run_chapter_mode(source, output_path, is_raw_text)
        else:
            result = await self._run_flat_mode(source, output_path, is_raw_text)

        # Emit generation history entry on success
        if result.success:
            self._emit_generation_entry(source, output_path, result)

        return result

    # ── Flat Mode ──────────────────────────────────────────────

    async def _run_flat_mode(self, source: str, output_path: str, is_raw_text: bool) -> PipelineResult:
        self._progress(3, "Chunking text...")
        chunks = list(TextProcessor.chunk_generator(
            source, self.cfg.chunk_size, is_raw_text, self.cfg.multi_speaker,
            pronunciation_dict=self.cfg.pronunciation_dict if self.cfg.pronunciation_dict else None,
        ))
        self._total = len(chunks)
        self._log(f"<b>{self._total} chunks queued</b>")

        with temp_dir() as td:
            self._progress(5, "Synthesizing...")
            chunk_map = await self._synthesize_all(chunks, td)
            self._check_cancel()
            self._progress(85, "Post-processing...")

            ordered = [chunk_map[i] for i in sorted(chunk_map)]
            preset = QualityPreset(
                name="pipeline", description="",
                target_lufs=self.cfg.target_lufs, true_peak=self.cfg.true_peak,
                sample_rate=self.cfg.sample_rate, bitrate=self.cfg.bitrate,
                codec="libmp3lame", mono=True,
                silence_threshold_db=self.cfg.silence_threshold_db,
                inter_chunk_silence=self.cfg.inter_chunk_silence,
                fade_duration=self.cfg.fade_duration,
            )
            processor = AudioPostProcessor(preset=preset)
            pp = processor.process(
                chunk_paths=ordered, output_path=output_path,
                output_format=self.cfg.output_format, normalize=self.cfg.normalize,
                trim_silence=self.cfg.trim_silence, title=self.cfg.title,
                author=self.cfg.author, cover_art=self.cfg.cover_art,
            )
        return self._build_result(pp, 0)

    # ── Chapter Mode (updated with crossfade, room tone, per-chapter norm) ──

    async def _run_chapter_mode(self, source: str, output_path: str, is_raw_text: bool) -> PipelineResult:
        self._progress(2, "Detecting chapters...")
        chapters = TextProcessor.detect_chapters(source, is_raw_text)
        n_ch = len(chapters)
        self._log(f"<b>Detected {n_ch} chapter(s)</b>")

        chapter_audios: List[Path] = []
        chapter_titles: List[str] = []
        total_chunks = 0

        with temp_dir(prefix="audiobook_ch_") as td:
            for ci, chapter in enumerate(chapters):
                self._check_cancel()
                await self._wait_if_paused()
                self._chapter_cb(ci + 1, n_ch)
                self._log(f"Chapter {ci + 1}/{n_ch}: <b>{chapter.title}</b>")

                ch_chunks = list(TextProcessor.chunk_generator(
                    chapter.text, self.cfg.chunk_size, is_raw_text=True,
                    multi_speaker=self.cfg.multi_speaker,
                    pronunciation_dict=self.cfg.pronunciation_dict if self.cfg.pronunciation_dict else None,
                ))
                if not ch_chunks:
                    self._log(f'<font color="orange">Empty chapter: {chapter.title}</font>')
                    continue

                ch_dir = td / f"ch_{ci:03d}"
                ch_dir.mkdir()
                base_pct = 5 + int((ci / n_ch) * 75)

                chunk_map = await self._synthesize_all(ch_chunks, ch_dir, base_pct=base_pct)
                self._check_cancel()

                ordered = [chunk_map[i] for i in sorted(chunk_map)]
                ch_audio = ch_dir / "chapter.mp3"

                preset = QualityPreset(
                    name="pipeline", description="",
                    target_lufs=self.cfg.target_lufs, true_peak=self.cfg.true_peak,
                    sample_rate=self.cfg.sample_rate, bitrate=self.cfg.bitrate,
                    codec="libmp3lame", mono=True,
                    silence_threshold_db=self.cfg.silence_threshold_db,
                    inter_chunk_silence=self.cfg.inter_chunk_silence,
                    fade_duration=self.cfg.fade_duration,
                )
                processor = AudioPostProcessor(preset=preset)
                if not processor.concatenate(ordered, ch_audio):
                    self._log(f'<font color="red">Concat failed: ch {ci + 1}</font>')
                    continue

                # ── Per-chapter normalization ───────────────────
                if self.cfg.per_chapter_normalization:
                    normalized = ch_dir / "chapter_norm.mp3"
                    if processor.normalize_loudness(ch_audio, normalized):
                        ch_audio = normalized
                        self._log(f"Chapter {ci + 1}: normalized to {self.cfg.target_lufs} LUFS")
                    else:
                        self._log(
                            f'<font color="orange">Chapter {ci + 1}: per-chapter normalization failed</font>'
                        )

                chapter_audios.append(ch_audio)
                chapter_titles.append(chapter.title)
                total_chunks += len(ch_chunks)

            self._check_cancel()

            # ── Crossfade between chapters ─────────────────────
            if self.cfg.crossfade_duration_ms > 0 and len(chapter_audios) > 1:
                self._progress(80, "Applying crossfade...")
                chapter_audios = _apply_crossfade(
                    chapter_audios,
                    self.cfg.crossfade_duration_ms,
                    td,
                    self.cfg.sample_rate,
                )
                self._log(
                    f"Crossfade applied: {self.cfg.crossfade_duration_ms} ms between chapters"
                )

            # ── Room tone overlay ──────────────────────────────
            if self.cfg.room_tone != "none" and chapter_audios:
                self._progress(81, "Applying room tone...")
                # First concatenate all chapters for room tone overlay
                pre_room = td / "pre_room_tone.mp3"
                if len(chapter_audios) == 1:
                    pre_room = chapter_audios[0]
                else:
                    processor.concatenate(chapter_audios, pre_room)

                room_out = td / "with_room_tone.mp3"
                result_path = _apply_room_tone(
                    pre_room,
                    self.cfg.room_tone,
                    self.cfg.room_tone_file,
                    room_out,
                    self.cfg.sample_rate,
                )
                if result_path != pre_room:
                    # Replace chapter_audios with the single room-toned file
                    chapter_audios = [result_path]
                    self._log(f"Room tone applied: {self.cfg.room_tone}")

            # ── Build chapter markers & final processing ───────
            self._progress(82, "Post-processing...")
            markers = AudioPostProcessor.build_chapter_markers(chapter_audios, chapter_titles)

            pp = processor.process(
                chunk_paths=chapter_audios, output_path=output_path,
                output_format=self.cfg.output_format, normalize=self.cfg.normalize,
                trim_silence=self.cfg.trim_silence, chapters=markers,
                title=self.cfg.title, author=self.cfg.author, cover_art=self.cfg.cover_art,
            )
        self._total = total_chunks
        return self._build_result(pp, n_ch)

    # ── Partial Regeneration (re-generate a single chapter) ────

    async def run_partial(
        self, source: str, output_path: str, chapter_index: int, is_raw_text: bool,
    ) -> PipelineResult:
        """Re-generate only a specific chapter.

        Steps:
        1. Detect chapters from source
        2. Get only the target chapter by index
        3. Chunk the chapter text
        4. Synthesize only that chapter's chunks
        5. Post-process only that chapter
        6. Return result

        The *output_path* is the file path for the single chapter output.
        """
        self._start = time.monotonic()
        self._completed = 0
        self._failed = 0

        self._progress(1, f"Partial regeneration: chapter {chapter_index + 1}...")

        # Pre-flight
        preflight_errors = self._preflight_check(source, is_raw_text)
        if preflight_errors:
            raise ConfigurationError("\n".join(preflight_errors))

        self._check_cancel()
        await self._wait_if_paused()

        # Detect chapters
        self._progress(2, "Detecting chapters...")
        chapters = TextProcessor.detect_chapters(source, is_raw_text)

        if chapter_index < 0 or chapter_index >= len(chapters):
            raise ConfigurationError(
                f"Invalid chapter index {chapter_index}. "
                f"Found {len(chapters)} chapter(s) (0–{len(chapters) - 1})."
            )

        chapter = chapters[chapter_index]
        self._log(f"Regenerating chapter {chapter_index + 1}: <b>{chapter.title}</b>")

        # Chunk the target chapter
        ch_chunks = list(TextProcessor.chunk_generator(
            chapter.text, self.cfg.chunk_size, is_raw_text=True,
            multi_speaker=self.cfg.multi_speaker,
            pronunciation_dict=self.cfg.pronunciation_dict if self.cfg.pronunciation_dict else None,
        ))

        if not ch_chunks:
            raise ConfigurationError(f"Chapter '{chapter.title}' is empty — nothing to generate.")

        self._total = len(ch_chunks)
        self._log(f"<b>{self._total} chunks queued for chapter {chapter_index + 1}</b>")

        with temp_dir(prefix="audiobook_partial_") as td:
            # Synthesize
            self._progress(5, "Synthesizing...")
            chunk_map = await self._synthesize_all(ch_chunks, td)
            self._check_cancel()

            self._progress(80, "Post-processing chapter...")

            ordered = [chunk_map[i] for i in sorted(chunk_map)]

            preset = QualityPreset(
                name="pipeline", description="",
                target_lufs=self.cfg.target_lufs, true_peak=self.cfg.true_peak,
                sample_rate=self.cfg.sample_rate, bitrate=self.cfg.bitrate,
                codec="libmp3lame", mono=True,
                silence_threshold_db=self.cfg.silence_threshold_db,
                inter_chunk_silence=self.cfg.inter_chunk_silence,
                fade_duration=self.cfg.fade_duration,
            )
            processor = AudioPostProcessor(preset=preset)

            # Concatenate chunks → chapter audio
            ch_audio = td / "chapter.mp3"
            if not processor.concatenate(ordered, ch_audio):
                raise PipelineError(f"Concatenation failed for chapter {chapter_index + 1}")

            # Per-chapter normalization
            if self.cfg.per_chapter_normalization:
                normalized = td / "chapter_norm.mp3"
                if processor.normalize_loudness(ch_audio, normalized):
                    ch_audio = normalized

            # Room tone
            if self.cfg.room_tone != "none":
                room_out = td / "with_room_tone.mp3"
                ch_audio = _apply_room_tone(
                    ch_audio, self.cfg.room_tone, self.cfg.room_tone_file,
                    room_out, self.cfg.sample_rate,
                )

            # Final processing (normalize, trim, encode)
            pp = processor.process(
                chunk_paths=[ch_audio], output_path=output_path,
                output_format=self.cfg.output_format, normalize=self.cfg.normalize,
                trim_silence=self.cfg.trim_silence,
                title=f"{self.cfg.title} — {chapter.title}",
                author=self.cfg.author, cover_art=self.cfg.cover_art,
            )

        result = self._build_result(pp, 1)

        # Emit generation history for partial run
        if result.success:
            self._emit_generation_entry(source, output_path, result, is_partial=True)

        return result

    # ── Synthesis ──────────────────────────────────────────────

    async def _synthesize_all(
        self, chunks: List[DialogueChunk], output_dir: Path, base_pct: int = 5,
    ) -> Dict[int, Path]:
        sem = asyncio.Semaphore(self.cfg.concurrency)
        results: Dict[int, Path] = {}
        lock = asyncio.Lock()
        self._completed = 0
        self._total = len(chunks)

        async def _worker(idx: int, chunk: DialogueChunk) -> None:
            if self._cancel.is_set(): return
            async with sem:
                await self._wait_if_paused()
                if self._cancel.is_set(): return
                out = output_dir / f"chunk_{idx:06d}.mp3"

                # Build segments with per-speaker overrides
                segments: List[Tuple[str, str]] = []
                for text, speaker in chunk.segments:
                    segments.append((text, self._resolve_voice(speaker)))

                # Resolve per-speaker parameters from the dominant speaker
                spk_params = self._resolve_speaker_params(chunk.dominant_speaker)

                def _on_retry(attempt: int, msg: str) -> None:
                    self._log(f'<font color="orange">Retry {attempt}/3: {msg}</font>')

                try:
                    await retry_async(
                        self.backend.synthesize_segments, segments, str(out),
                        rate=spk_params["rate"],
                        volume=spk_params["volume"],
                        pitch=spk_params["pitch"],
                        use_ssml=self.cfg.use_ssml,
                        pronunciation_dict=self.cfg.pronunciation_dict if self.cfg.pronunciation_dict else None,
                        on_retry=_on_retry,
                    )
                    async with lock:
                        results[idx] = out
                        self._completed += 1
                        pct = base_pct + int((self._completed / max(1, self._total)) * 30)
                        self._progress(pct, f"Generated {self._completed}/{self._total}  |  ETA {self._eta()}")
                except SynthesisError as exc:
                    async with lock: self._failed += 1
                    self._log(f'<font color="red">Chunk {idx}: {exc}</font>')

        tasks = [asyncio.create_task(_worker(i, chunk)) for i, chunk in enumerate(chunks)]
        await asyncio.gather(*tasks, return_exceptions=True)
        return results

    # ── Result & History Helpers ───────────────────────────────

    def _build_result(self, pp: ProcessingResult, chapter_count: int) -> PipelineResult:
        elapsed = time.monotonic() - self._start
        return PipelineResult(
            output_files=tuple(pp.files), total_duration_ms=pp.total_duration_ms,
            total_chunks=self._completed, failed_chunks=self._failed,
            elapsed_seconds=elapsed, chapter_count=chapter_count,
        )

    def _emit_generation_entry(
        self,
        source: str,
        output_path: str,
        result: PipelineResult,
        is_partial: bool = False,
    ) -> None:
        """Create and emit a GenerationEntry dict for history logging."""
        try:
            entry = GenerationEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                source_file=source if not is_partial else f"{source} [partial]",
                output_path=output_path,
                backend=self.cfg.backend_id,
                voice=self.cfg.voice,
                quality_preset=self.cfg.quality_preset,
                total_chunks=result.total_chunks,
                failed_chunks=result.failed_chunks,
                duration_ms=result.total_duration_ms,
                elapsed_seconds=result.elapsed_seconds,
                chapter_count=result.chapter_count,
            )
            self._generation_cb(entry.to_dict())
        except Exception as exc:
            logger.warning("Failed to emit generation entry: %s", exc)


# ══════════════════════════════════════════════════════════════════
#  Backend Factory Helper
# ══════════════════════════════════════════════════════════════════

def create_backend(config: EngineConfig) -> TTSBackend:
    """Create the primary TTS backend, optionally wrapped in a fallback chain.

    When *config.enable_fallback* is True and *FallbackTTSBackend* is
    available, the primary backend is wrapped so that if synthesis fails
    the next backend in *fallback_backends* is tried automatically.
    """
    primary = TTSBackend.create(config.backend_id, {
        "voice": config.voice,
        "rate": config.rate,
        "volume": config.volume,
        "pitch": config.pitch,
        "use_ssml": config.use_ssml,
        "qwen_model_path": config.qwen_model_path,
        "qwen_device": config.qwen_device,
        "qwen_voice": config.qwen_voice,
        "omnivoice_model_path": config.omnivoice_model_path,
        "omnivoice_device": config.omnivoice_device,
        "omnivoice_speaker_id": config.omnivoice_speaker_id,
        "dramabox_api_url": config.dramabox_api_url,
        "dramabox_api_key": config.dramabox_api_key,
        "dramabox_voice_style": config.dramabox_voice_style,
        "dramabox_model": config.dramabox_model,
    })

    if config.enable_fallback and FallbackTTSBackend is not None and config.fallback_backends:
        fallbacks: List[TTSBackend] = []
        for bid in config.fallback_backends:
            if bid == config.backend_id:
                continue  # Skip primary in fallback list
            try:
                fb = TTSBackend.create(bid, {
                    "voice": config.voice,
                    "rate": config.rate,
                    "volume": config.volume,
                    "pitch": config.pitch,
                    "use_ssml": config.use_ssml,
                })
                fallbacks.append(fb)
            except Exception as exc:
                logger.warning("Could not create fallback backend '%s': %s", bid, exc)

        if fallbacks:
            primary = FallbackTTSBackend(primary, fallbacks)

    return primary


# ══════════════════════════════════════════════════════════════════
#  AudiobookEngine — QThread wrapper (bridges pipeline → Qt)
# ══════════════════════════════════════════════════════════════════

class AudiobookEngine(QThread):
    """QThread that runs the audiobook pipeline and emits Qt signals."""

    progress = Signal(int, str)
    log = Signal(str)
    chapter_progress = Signal(int, int)
    finished = Signal(bool, str)
    generation_complete = Signal(dict)  # GenerationEntry as dict

    def __init__(
        self,
        source: str,
        output_path: str,
        config: EngineConfig,
        backend: TTSBackend,
        is_raw_text: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.source = source
        self.output_path = output_path
        self.is_raw_text = is_raw_text

        self._backend = backend
        self._pipeline = AudiobookPipeline(
            config=config,
            backend=backend,
            on_progress=self._on_progress,
            on_log=self._log,
            on_chapter=self._on_chapter,
            on_generation=self._on_generation,
        )

    def _on_progress(self, pct: int, msg: str) -> None:
        self.progress.emit(pct, msg)

    def _log(self, msg: str) -> None:
        self.log.emit(msg)

    def _on_chapter(self, current: int, total: int) -> None:
        self.chapter_progress.emit(current, total)

    def _on_generation(self, entry_dict: Dict[str, Any]) -> None:
        self.generation_complete.emit(entry_dict)

    def cancel(self) -> None:
        self._pipeline.cancel()
        self.log.emit('<font color="#f38ba8">Cancelling...</font>')

    def pause(self) -> None:
        self._pipeline.pause()

    def resume(self) -> None:
        self._pipeline.resume()

    @property
    def is_paused(self) -> bool:
        return self._pipeline.is_paused

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result: PipelineResult = loop.run_until_complete(
                self._pipeline.run(self.source, self.output_path, self.is_raw_text)
            )
            self.progress.emit(100, "Complete!")
            self.finished.emit(True, result.summary_html)

        except PipelineCancelled:
            self.finished.emit(False, "Cancelled by user.")

        except Exception as exc:
            self.log.emit(f'<font color="red">{exc}</font>')
            self.finished.emit(False, str(exc))

        finally:
            try:
                loop.run_until_complete(self._backend.close())
            except Exception:
                pass
            loop.close()


# ══════════════════════════════════════════════════════════════════
#  PartialRegenEngine — QThread for single-chapter regeneration
# ══════════════════════════════════════════════════════════════════

class PartialRegenEngine(QThread):
    """QThread that re-generates a single chapter and emits Qt signals."""

    progress = Signal(int, str)
    log = Signal(str)
    chapter_progress = Signal(int, int)
    finished = Signal(bool, str)
    generation_complete = Signal(dict)

    def __init__(
        self,
        source: str,
        output_path: str,
        chapter_index: int,
        config: EngineConfig,
        backend: TTSBackend,
        is_raw_text: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.source = source
        self.output_path = output_path
        self.chapter_index = chapter_index
        self.is_raw_text = is_raw_text

        self._backend = backend
        self._pipeline = AudiobookPipeline(
            config=config,
            backend=backend,
            on_progress=self._on_progress,
            on_log=self._log,
            on_chapter=self._on_chapter,
            on_generation=self._on_generation,
        )

    def _on_progress(self, pct: int, msg: str) -> None:
        self.progress.emit(pct, msg)

    def _log(self, msg: str) -> None:
        self.log.emit(msg)

    def _on_chapter(self, current: int, total: int) -> None:
        self.chapter_progress.emit(current, total)

    def _on_generation(self, entry_dict: Dict[str, Any]) -> None:
        self.generation_complete.emit(entry_dict)

    def cancel(self) -> None:
        self._pipeline.cancel()
        self.log.emit('<font color="#f38ba8">Cancelling...</font>')

    def pause(self) -> None:
        self._pipeline.pause()

    def resume(self) -> None:
        self._pipeline.resume()

    @property
    def is_paused(self) -> bool:
        return self._pipeline.is_paused

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result: PipelineResult = loop.run_until_complete(
                self._pipeline.run_partial(
                    self.source, self.output_path, self.chapter_index, self.is_raw_text,
                )
            )
            self.progress.emit(100, "Chapter regenerated!")
            self.finished.emit(True, result.summary_html)

        except PipelineCancelled:
            self.finished.emit(False, "Cancelled by user.")

        except Exception as exc:
            self.log.emit(f'<font color="red">{exc}</font>')
            self.finished.emit(False, str(exc))

        finally:
            try:
                loop.run_until_complete(self._backend.close())
            except Exception:
                pass
            loop.close()
