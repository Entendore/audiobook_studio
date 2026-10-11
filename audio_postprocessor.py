"""
FFmpeg-based audio post-processing for broadcast-quality audiobooks.

Provides loudness normalization (EBU R128), silence trimming, chunk
concatenation, chapter metadata embedding, crossfade transitions, room tone
/ ambient background beds, per-chapter normalization, batch re-export,
final encoding to MP3 or M4B, noise reduction (FFT denoiser), voice-optimized
equalization (warm/bright/neutral presets), dynamic range compression,
speed/pitch control, silence gap analysis, progress callbacks, and
detailed FFmpeg diagnostics.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import QualityPreset
from typing import Callable, Any

__all__ = [
    "AudioPostProcessor",
    "ChapterMarker",
    "ProcessingResult",
    "ProcessingDiagnostics",
    "SilenceGap",
]

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LUFS: float = -16.0
DEFAULT_TP: float = -1.0

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChapterMarker:
    """Immutable reference for a single audiobook chapter."""

    title: str
    start_ms: int
    end_ms: int


@dataclass
class ProcessingResult:
    """Accumulates outputs and diagnostics from the processing pipeline."""

    files: List[Path] = field(default_factory=list)
    total_duration_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)



@dataclass
class ProcessingDiagnostics:
    """Detailed FFmpeg diagnostics from a processing run."""
    commands_run: int = 0
    commands_succeeded: int = 0
    commands_failed: int = 0
    stderr_output: List[str] = field(default_factory=list)
    total_process_time_s: float = 0.0

    @property
    def success_rate(self) -> float:
        """Fraction of commands that succeeded (0.0–1.0)."""
        return (self.commands_succeeded / self.commands_run
                if self.commands_run > 0 else 0.0)


@dataclass(frozen=True)
class SilenceGap:
    """A detected silence gap in an audio file."""
    start_ms: float
    end_ms: float
    duration_ms: float
    loudness_db: float

    def add_file(self, path: Path, duration_ms: float = 0.0) -> None:
        """Register an output file and optionally accumulate duration."""
        self.files.append(path)
        if duration_ms > 0:
            self.total_duration_ms += duration_ms


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------


def _bash_safe(p: Path) -> str:
    """Escape a path for use inside an FFmpeg concat demuxer list file."""
    s = str(p)
    # Escape single backslashes first, then single-quotes, then wrap in quotes
    s = s.replace("\\", "\\\\")
    s = s.replace("'", "'\\''")
    return f"'{s}'"


# ---------------------------------------------------------------------------
# AudioPostProcessor
# ---------------------------------------------------------------------------


class AudioPostProcessor:
    """Orchestrates FFmpeg commands for audiobook post-production."""

    _DEFAULT_CODEC = "libmp3lame"
    _DEFAULT_MONO = True

    def __init__(
        self,
        preset: Optional[QualityPreset] = None,
        target_lufs: float = DEFAULT_LUFS,
        true_peak: float = DEFAULT_TP,
        inter_chunk_silence: float = 0.6,
        silence_threshold_db: float = -40.0,
        fade_duration: float = 0.15,
        sample_rate: int = 44100,
        bitrate: str = "64k",
        codec: str = _DEFAULT_CODEC,
        mono: bool = _DEFAULT_MONO,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        if preset is not None:
            self.target_lufs = preset.target_lufs
            self.true_peak = preset.true_peak
            self.inter_chunk_silence = preset.inter_chunk_silence
            self.silence_threshold_db = preset.silence_threshold_db
            self.fade_duration = preset.fade_duration
            self.sample_rate = preset.sample_rate
            self.bitrate = preset.bitrate
            self.codec = preset.codec if hasattr(preset, "codec") else self._DEFAULT_CODEC
            self.mono = preset.mono if hasattr(preset, "mono") else self._DEFAULT_MONO
        else:
            self.target_lufs = target_lufs
            self.true_peak = true_peak
            self.inter_chunk_silence = inter_chunk_silence
            self.silence_threshold_db = silence_threshold_db
            self.fade_duration = fade_duration
            self.sample_rate = sample_rate
            self.bitrate = bitrate
            self.codec = codec
            self.mono = mono
        self.progress_callback: Optional[Callable[[float, str], None]] = progress_callback
        self._diagnostics: ProcessingDiagnostics = ProcessingDiagnostics()
        self._temp_dirs: List[str] = []

    # ------------------------------------------------------------------
    # Shared pipeline (DRY)
    # ------------------------------------------------------------------


    def _run_pipeline(
        self,
        chunk_paths: List[Path],
        output_path: Path,
        output_format: str = "mp3",
        normalize: bool = True,
        trim_silence: bool = True,
        chapters: Optional[List[ChapterMarker]] = None,
        title: str = "",
        author: str = "",
        cover_art: Optional[Path] = None,
        crossfade_ms: int = 0,
        room_tone: str = "none",
        room_tone_file: str = "",
        per_chapter_normalize: bool = False,
        denoise: bool = False,
        voice_eq: bool = False,
        compress: bool = False,
        speed_factor: float = 1.0,
    ) -> ProcessingResult:
        """Shared pipeline implementation for both process() and batch_reexport().

        Pipeline stages:
        per-chapter norm → concat → crossfade → denoise → room tone →
        trim → compress → normalize → EQ → speed → encode → cover
        """
        result = ProcessingResult()
        self.reset_diagnostics()
        workdir = tempfile.mkdtemp(prefix="audiobook_pp_")
        self._temp_dirs.append(workdir)

        try:
            self._report_progress(0.0, "Starting pipeline")

            # ── Per-chapter normalization (before concat) ────────────
            if per_chapter_normalize and normalize:
                normalized_chunks: List[Path] = []
                for idx, chunk in enumerate(chunk_paths):
                    self._report_progress(
                        idx / len(chunk_paths) * 0.15,
                        f"Per-chapter normalization {idx + 1}/{len(chunk_paths)}"
                    )
                    out = Path(workdir) / f"norm_chunk_{idx:04d}.mp3"
                    if self.normalize_chapter(chunk, out, target_lufs=self.target_lufs):
                        normalized_chunks.append(out)
                    else:
                        result.warnings.append(
                            f"Per-chapter normalization failed for chunk {idx}; using original"
                        )
                        normalized_chunks.append(chunk)
                chunk_paths = normalized_chunks

            # 1. Concatenate raw chunks
            self._report_progress(0.15, "Concatenating chunks")
            concat_out = Path(workdir) / "concatenated.mp3"
            if not self.concatenate(chunk_paths, concat_out):
                result.warnings.append("Concatenation failed; using first chunk as fallback")
                if chunk_paths:
                    concat_out = chunk_paths[0]
                else:
                    return result

            current = concat_out

            # 2. Crossfade between chapters (if requested)
            if crossfade_ms > 0 and len(chunk_paths) > 1:
                self._report_progress(0.25, "Applying crossfade")
                crossfaded = Path(workdir) / "crossfaded.mp3"
                if self.apply_crossfade(chunk_paths, crossfaded, crossfade_ms=crossfade_ms):
                    current = crossfaded
                else:
                    result.warnings.append("Crossfade failed; using concatenated audio")

            # 3. Noise reduction (if requested)
            if denoise:
                self._report_progress(0.35, "Applying noise reduction")
                denoised = Path(workdir) / "denoised.mp3"
                if self.apply_denoise(current, denoised):
                    current = denoised
                else:
                    result.warnings.append("Noise reduction failed; using original audio")

            # 4. Room tone / ambient bed
            if room_tone != "none" or room_tone_file:
                self._report_progress(0.40, "Applying room tone")
                room_toned = Path(workdir) / "room_toned.mp3"
                if self.apply_room_tone(
                    current, room_toned,
                    room_tone=room_tone,
                    room_tone_file=room_tone_file,
                ):
                    current = room_toned
                else:
                    result.warnings.append("Room tone application failed; using dry audio")

            # 5. Trim silence from concatenated file
            if trim_silence:
                self._report_progress(0.50, "Trimming silence")
                trimmed = Path(workdir) / "trimmed.mp3"
                if self.trim_silence_file(current, trimmed):
                    current = trimmed
                else:
                    result.warnings.append("Silence trimming failed; using untrimmed audio")

            # 6. Dynamic range compression (before normalization)
            if compress:
                self._report_progress(0.55, "Applying dynamic range compression")
                compressed = Path(workdir) / "compressed.mp3"
                if self.apply_compression(current, compressed):
                    current = compressed
                else:
                    result.warnings.append("Compression failed; using uncompressed audio")

            # 7. Normalize loudness (two-pass EBU R128) — whole-book pass
            if normalize and not per_chapter_normalize:
                self._report_progress(0.65, "Normalizing loudness")
                normalized = Path(workdir) / "normalized.mp3"
                if self.normalize_loudness(current, normalized):
                    current = normalized
                else:
                    result.warnings.append("Loudness normalization failed; using unnormalized audio")

            # 8. Voice EQ (after normalization for best results)
            if voice_eq:
                self._report_progress(0.75, "Applying voice EQ")
                eqed = Path(workdir) / "eqed.mp3"
                if self.apply_voice_eq(current, eqed):
                    current = eqed
                else:
                    result.warnings.append("Voice EQ failed; using un-EQ\'d audio")

            # 9. Speed adjustment
            if speed_factor != 1.0:
                self._report_progress(0.82, f"Adjusting speed ({speed_factor:.2f}x)")
                sped = Path(workdir) / "sped.mp3"
                if self.apply_speed(current, sped, factor=speed_factor):
                    current = sped
                else:
                    result.warnings.append("Speed adjustment failed; using original speed")

            # 10. Final encode / container
            self._report_progress(0.88, "Encoding final output")
            if output_format.lower() == "m4b":
                m4b_path = output_path.with_suffix(".m4b")
                if self.create_m4b(current, chapters or [], m4b_path, title, author, cover_art):
                    dur = self.get_duration_ms(m4b_path)
                    result.add_file(m4b_path, dur)
                else:
                    result.warnings.append("M4B creation failed")
            else:
                final = output_path.with_suffix(".mp3")
                cmd = [
                    "ffmpeg", "-y", "-i", str(current),
                    "-ar", str(self.sample_rate),
                    "-b:a", self.bitrate,
                    "-ac", "1" if self.mono else "2",
                    "-codec:a", self.codec,
                    str(final),
                ]
                if self._run_ffmpeg(cmd):
                    if cover_art:
                        tagged = final.with_name(final.stem + "_tagged.mp3")
                        if self._embed_cover_mp3(final, cover_art, tagged, title, author):
                            final = tagged
                        else:
                            result.warnings.append("Cover art embedding failed")
                    dur = self.get_duration_ms(final)
                    result.add_file(final, dur)
                else:
                    result.warnings.append("Final encoding failed")

            self._report_progress(1.0, "Complete")

        except Exception as exc:  # noqa: BLE001
            result.warnings.append(f"Pipeline error: {exc}")
            logger.exception("Pipeline error")

        return result

    # ------------------------------------------------------------------
    # Main pipeline (public)
    # ------------------------------------------------------------------

    def process(
        self,
        chunk_paths: List[Path],
        output_path: Path,
        output_format: str = "mp3",
        normalize: bool = True,
        trim_silence: bool = True,
        chapters: Optional[List[ChapterMarker]] = None,
        title: str = "",
        author: str = "",
        cover_art: Optional[Path] = None,
        crossfade_ms: int = 0,
        room_tone: str = "none",
        room_tone_file: str = "",
        per_chapter_normalize: bool = False,
        denoise: bool = False,
        voice_eq: bool = False,
        compress: bool = False,
        speed_factor: float = 1.0,
    ) -> ProcessingResult:
        """Run the full post-processing pipeline and return a *ProcessingResult*.

        Pipeline stages:
        per-chapter norm → concat → crossfade → denoise → room tone →
        trim → compress → normalize → EQ → speed → encode → cover

        Args:
            chunk_paths: Ordered list of per-chapter audio file paths.
            output_path: Destination path for the final file.
            output_format: ``"mp3"`` or ``"m4b"``.
            normalize: Apply EBU R128 loudness normalization.
            trim_silence: Trim leading/trailing silence.
            chapters: Pre-built chapter markers (for M4B).
            title: Audiobook title (metadata).
            author: Audiobook author (metadata).
            cover_art: Path to cover art image.
            crossfade_ms: Crossfade duration in ms between consecutive
                chapters. 0 disables crossfading.
            room_tone: Room tone style – ``"none"``, ``"subtle"``,
                ``"moderate"``, or ``"heavy"``.
            room_tone_file: Path to a custom room-tone audio file.
            per_chapter_normalize: Normalize each chapter individually.
            denoise: Apply FFT-based noise reduction (``afftdn``).
            voice_eq: Apply voice-optimized equalization.
            compress: Apply dynamic range compression.
            speed_factor: Playback speed multiplier (0.5–2.0).
        """
        return self._run_pipeline(
            chunk_paths=chunk_paths,
            output_path=output_path,
            output_format=output_format,
            normalize=normalize,
            trim_silence=trim_silence,
            chapters=chapters,
            title=title,
            author=author,
            cover_art=cover_art,
            crossfade_ms=crossfade_ms,
            room_tone=room_tone,
            room_tone_file=room_tone_file,
            per_chapter_normalize=per_chapter_normalize,
            denoise=denoise,
            voice_eq=voice_eq,
            compress=compress,
            speed_factor=speed_factor,
        )

    # ------------------------------------------------------------------
    # Loudness measurement
    # ------------------------------------------------------------------

    def measure_loudness(self, input_path: Path) -> Optional[Dict[str, float]]:
        """Run FFmpeg *loudnorm* in measurement mode and return parsed stats."""
        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-af", f"loudnorm=I={self.target_lufs}:TP={self.true_peak}:print_format=json",
            "-f", "null", "-",
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, check=False
            )
            # loudnorm prints JSON to stderr when print_format=json is used
            stderr = proc.stderr
            # Find the last JSON object in stderr (loudnorm may print twice)
            matches = re.findall(r"\{[^{}]*\}", stderr, re.DOTALL)
            if not matches:
                return None
            data = json.loads(matches[-1])
            return {
                "input_i": float(data.get("input_i", 0)),
                "input_tp": float(data.get("input_tp", 0)),
                "input_lra": float(data.get("input_lra", 0)),
                "input_thresh": float(data.get("input_thresh", 0)),
                "target_offset": float(data.get("target_offset", 0)),
            }
        except (json.JSONDecodeError, subprocess.TimeoutExpired, OSError):
            return None

    # ------------------------------------------------------------------
    # Two-pass EBU R128 normalization (DRY helper)
    # ------------------------------------------------------------------


    def _two_pass_normalize(
        self,
        input_path: Path,
        output_path: Path,
        target_lufs: float,
    ) -> bool:
        """Two-pass EBU R128 loudness normalization (shared implementation).

        Pass 1: measure loudness stats via loudnorm measurement mode.
        Pass 2: apply linear normalization with measured values.
        """
        # --- Pass 1: measure ---
        measure_cmd = [
            "ffmpeg", "-i", str(input_path),
            "-af", f"loudnorm=I={target_lufs}:TP={self.true_peak}:print_format=json",
            "-f", "null", "-",
        ]
        try:
            proc = subprocess.run(
                measure_cmd,
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
            matches = re.findall(r"\{[^{}]*\}", proc.stderr, re.DOTALL)
            if not matches:
                return False
            data = json.loads(matches[-1])
            stats = {
                "input_i": float(data.get("input_i", 0)),
                "input_tp": float(data.get("input_tp", 0)),
                "input_lra": float(data.get("input_lra", 0)),
                "input_thresh": float(data.get("input_thresh", 0)),
                "target_offset": float(data.get("target_offset", 0)),
            }
        except (json.JSONDecodeError, subprocess.TimeoutExpired, OSError):
            return False

        # --- Pass 2: apply ---
        af = (
            f"loudnorm=I={target_lufs}:TP={self.true_peak}"
            f":measured_I={stats['input_i']}"
            f":measured_TP={stats['input_tp']}"
            f":measured_LRA={stats['input_lra']}"
            f":measured_thresh={stats['input_thresh']}"
            f":offset={stats['target_offset']}"
            ":linear=true:print_format=summary"
        )

        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", af,
            "-ar", str(self.sample_rate),
            "-b:a", self.bitrate,
            "-ac", "1" if self.mono else "2",
            "-codec:a", self.codec,
            str(output_path),
        ]
        return self._run_ffmpeg(cmd)

    # ------------------------------------------------------------------
    # Normalization public API
    # ------------------------------------------------------------------

    def normalize_loudness(self, input_path: Path, output_path: Path) -> bool:
        """Two-pass EBU R128 loudness normalization.

        Delegates to :meth:`_two_pass_normalize` with the instance target LUFS.
        """
        return self._two_pass_normalize(input_path, output_path, self.target_lufs)

    # ------------------------------------------------------------------
    # Per-chapter normalization
    # ------------------------------------------------------------------

    def normalize_chapter(
        self,
        input_path: Path,
        output_path: Path,
        target_lufs: float = -16.0,
    ) -> bool:
        """Per-chapter loudness normalization with a custom target LUFS.

        Delegates to :meth:`_two_pass_normalize`.
        """
        return self._two_pass_normalize(input_path, output_path, target_lufs)

    # ------------------------------------------------------------------
    # Crossfade
    # ------------------------------------------------------------------

    def apply_crossfade(
        self,
        input_paths: List[Path],
        output_path: Path,
        crossfade_ms: int = 500,
    ) -> bool:
        """Apply crossfade transitions between consecutive audio files.

        Uses FFmpeg's ``acrossfade`` filter.  If only one file is supplied
        the file is simply copied to *output_path*.

        The crossfade is applied iteratively: file1 is crossfaded with
        file2, the result is crossfaded with file3, and so on.
        """
        if not input_paths:
            return False

        if len(input_paths) == 1:
            try:
                shutil.copy2(str(input_paths[0]), str(output_path))
                return True
            except OSError:
                return False

        crossfade_s = crossfade_ms / 1000.0
        workdir = tempfile.mkdtemp(prefix="audiobook_xf_")

        try:
            # Start with the first file
            prev_path = input_paths[0]

            for idx in range(1, len(input_paths)):
                next_path = input_paths[idx]
                is_last = idx == len(input_paths) - 1
                result_path = output_path if is_last else Path(workdir) / f"xf_{idx:04d}.mp3"

                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(prev_path),
                    "-i", str(next_path),
                    "-filter_complex",
                    f"[0:a][1:a]acrossfade=d={crossfade_s}:c1=tri:c2=tri[aout]",
                    "-map", "[aout]",
                    "-ar", str(self.sample_rate),
                    "-ac", "1" if self.mono else "2",
                    str(result_path),
                ]
                if not self._run_ffmpeg(cmd):
                    return False

                prev_path = result_path

            return True
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Room tone / ambient background
    # ------------------------------------------------------------------

    def apply_room_tone(
        self,
        input_path: Path,
        output_path: Path,
        room_tone: str = "none",
        room_tone_file: str = "",
        volume_db: float = -35,
    ) -> bool:
        """Mix a subtle ambient background with the main audio.

        Args:
            input_path: Path to the main audiobook audio.
            output_path: Destination path for the mixed result.
            room_tone: Style identifier – ``"none"``, ``"subtle"``,
                ``"moderate"``, or ``"heavy"``.  When ``"none"`` the input
                file is simply copied.
            room_tone_file: Path to a custom room-tone / ambient audio file.
                When non-empty this takes precedence over *room_tone* style.
            volume_db: Volume of the room-tone track in dB (relative to
                full scale).  The built-in styles override this with their
                own defaults; the value is only used for custom files.
        """
        if room_tone == "none" and not room_tone_file:
            # No room tone requested – just copy
            try:
                shutil.copy2(str(input_path), str(output_path))
                return True
            except OSError:
                return False

        # Determine the room-tone source file
        workdir = tempfile.mkdtemp(prefix="audiobook_rt_")
        input_dur = self.get_duration_ms(input_path) / 1000.0

        if room_tone_file:
            tone_src = Path(room_tone_file)
            if not tone_src.exists():
                return False
        else:
            # Generate room tone based on style
            tone_src = Path(workdir) / "room_tone_generated.wav"
            if not self.generate_room_tone(tone_src, duration_s=input_dur, style=room_tone):
                return False

        # Map style to default volume so the caller doesn't have to guess
        if not room_tone_file:
            style_volumes = {
                "subtle": -40,
                "moderate": -30,
                "heavy": -22,
            }
            volume_db = style_volumes.get(room_tone, volume_db)

        # Use amix: [0:a] is main audio, [1:a] is room tone with volume
        # amix normalizes inputs by default; we use duration=longest and
        # boost main to keep it dominant.
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-i", str(tone_src),
            "-filter_complex",
            (
                f"[1:a]volume={volume_db}dB,aloop=loop=-1:size=2e+09[rt];"
                f"[0:a][rt]amix=inputs=2:duration=first:dropout_transition=3:weights=1 0.15[aout]"
            ),
            "-map", "[aout]",
            "-ar", str(self.sample_rate),
            "-ac", "1" if self.mono else "2",
            "-b:a", self.bitrate,
            str(output_path),
        ]
        return self._run_ffmpeg(cmd)

    def generate_room_tone(
        self,
        output_path: Path,
        duration_s: float,
        style: str = "subtle",
    ) -> bool:
        """Generate ambient room tone using FFmpeg's ``anoisesrc`` filter.

        Args:
            output_path: Destination for the generated audio file.
            duration_s: Length of the generated tone in seconds.
            style: One of ``"subtle"`` (very low brown noise at -40 dB),
                ``"moderate"`` (low pink noise at -30 dB), or
                ``"heavy"`` (noticeable brown noise at -22 dB).

        Returns:
            ``True`` on success.
        """
        style_params = {
            "subtle": {"color": "brown", "volume": -40},
            "moderate": {"color": "pink", "volume": -30},
            "heavy": {"color": "brown", "volume": -22},
        }

        params = style_params.get(style, style_params["subtle"])
        color = params["color"]
        vol = params["volume"]

        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"anoisesrc=d={duration_s}:c={color}:a={vol}:r={self.sample_rate}",
            "-ar", str(self.sample_rate),
            "-ac", "1" if self.mono else "2",
            str(output_path),
        ]
        return self._run_ffmpeg(cmd)

    # ------------------------------------------------------------------
    # Silence trimming
    # ------------------------------------------------------------------

    def trim_silence_file(self, input_path: Path, output_path: Path) -> bool:
        """Trim leading & trailing silence using the *silenceremove* filter."""
        threshold = f"{self.silence_threshold_db}dB"
        fade = str(self.fade_duration)

        # Trim leading silence, reverse, trim trailing silence (now leading), reverse back
        af = (
            f"silenceremove=start_periods=1:start_silence={fade}"
            f":start_threshold={threshold},"
            f"areverse,"
            f"silenceremove=start_periods=1:start_silence={fade}"
            f":start_threshold={threshold},"
            f"areverse"
        )

        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", af,
            "-ar", str(self.sample_rate),
            "-ac", "1" if self.mono else "2",
            str(output_path),
        ]
        return self._run_ffmpeg(cmd)

    # ------------------------------------------------------------------
    # Concatenation
    # ------------------------------------------------------------------

    def concatenate(self, chunk_paths: List[Path], output_path: Path) -> bool:
        """Concatenate audio chunks with inter-chunk silence.

        1. Generate a short silence MP3.
        2. Build an FFmpeg concat demuxer list.
        3. Run the concat.
        """
        if not chunk_paths:
            return False

        workdir = output_path.parent
        silence_path = workdir / "silence.mp3"

        # Generate silence gap
        silence_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r={self.sample_rate}:cl=mono",
            "-t", str(self.inter_chunk_silence),
            "-b:a", self.bitrate,
            str(silence_path),
        ]
        if not self._run_ffmpeg(silence_cmd):
            return False

        # Build concat list (chunk → silence → chunk → silence → … → chunk)
        list_path = workdir / "concat_list.txt"
        with open(list_path, "w", encoding="utf-8") as fh:
            for idx, chunk in enumerate(chunk_paths):
                fh.write(f"file {_bash_safe(chunk)}\n")
                if idx < len(chunk_paths) - 1:
                    fh.write(f"file {_bash_safe(silence_path)}\n")

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(output_path),
        ]
        ok = self._run_ffmpeg(cmd)

        # Cleanup temp files
        for tmp in (silence_path, list_path):
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

        return ok

    # ------------------------------------------------------------------
    # M4B creation
    # ------------------------------------------------------------------

    def create_m4b(
        self,
        mp3_input: Path,
        chapters: List[ChapterMarker],
        output_path: Path,
        title: str = "",
        author: str = "",
        cover_art: Optional[Path] = None,
    ) -> bool:
        """Create an M4B (MP4 audiobook) with chapter markers and optional cover."""
        workdir = output_path.parent
        metadata_path = workdir / "FFmetadata1.txt"

        # Write FFmetadata chapter file
        with open(metadata_path, "w", encoding="utf-8") as fh:
            fh.write(";FFmetadata1\n")
            if title:
                fh.write(f"title={title}\n")
            if author:
                fh.write(f"artist={author}\n")
            fh.write("\n")
            for ch in chapters:
                start_ts = self._ms_to_ffmpeg_ts(ch.start_ms)
                end_ts = self._ms_to_ffmpeg_ts(ch.end_ms)
                fh.write("[CHAPTER]\n")
                fh.write("TIMEBASE=1/1000\n")
                fh.write(f"START={ch.start_ms}\n")
                fh.write(f"END={ch.end_ms}\n")
                fh.write(f"title={ch.title}\n\n")

        # Convert MP3 → AAC inside M4B container
        cmd: list[str] = [
            "ffmpeg", "-y", "-i", str(mp3_input),
            "-i", str(metadata_path),
            "-map_metadata", "1",
            "-c:a", "aac",
            "-b:a", self.bitrate,
            "-ac", "1" if self.mono else "2",
            "-ar", str(self.sample_rate),
            "-f", "mp4",
        ]

        # Embed cover art if provided
        if cover_art and cover_art.exists():
            cmd += ["-i", str(cover_art), "-map", "0:a", "-map", "2:v",
                     "-c:v", "mjpeg", "-disposition:v", "attached_pic"]
        else:
            cmd += ["-map", "0:a"]

        cmd.append(str(output_path))
        ok = self._run_ffmpeg(cmd)

        try:
            metadata_path.unlink(missing_ok=True)
        except OSError:
            pass

        return ok

    # ------------------------------------------------------------------
    # Cover art embedding (MP3)
    # ------------------------------------------------------------------

    def _embed_cover_mp3(
        self,
        mp3: Path,
        cover: Path,
        output: Path,
        title: str = "",
        author: str = "",
    ) -> bool:
        """Embed cover art and ID3 tags into an MP3 file."""
        if not cover.exists():
            return False

        cmd = [
            "ffmpeg", "-y",
            "-i", str(mp3),
            "-i", str(cover),
            "-map", "0:a",
            "-map", "1:v",
            "-c:a", "copy",
            "-c:v", "mjpeg",
            "-disposition:v", "attached_pic",
        ]
        if title:
            cmd += ["-metadata", f"title={title}"]
        if author:
            cmd += ["-metadata", f"artist={author}"]
        cmd.append(str(output))
        return self._run_ffmpeg(cmd)

    # ------------------------------------------------------------------
    # Batch re-export
    # ------------------------------------------------------------------

    def batch_reexport(
        self,
        chunk_paths: List[Path],
        output_path: Path,
        output_format: str = "mp3",
        normalize: bool = True,
        trim_silence: bool = True,
        chapters: Optional[List[ChapterMarker]] = None,
        title: str = "",
        author: str = "",
        cover_art: Optional[Path] = None,
        crossfade_ms: int = 0,
        room_tone: str = "none",
        room_tone_file: str = "",
        denoise: bool = False,
        voice_eq: bool = False,
        compress: bool = False,
        speed_factor: float = 1.0,
    ) -> ProcessingResult:
        """Full re-export pipeline from existing chunk files.

        Allows changing post-processing settings without re-synthesizing.
        Delegates to :meth:`_run_pipeline`.
        """
        return self._run_pipeline(
            chunk_paths=chunk_paths,
            output_path=output_path,
            output_format=output_format,
            normalize=normalize,
            trim_silence=trim_silence,
            chapters=chapters,
            title=title,
            author=author,
            cover_art=cover_art,
            crossfade_ms=crossfade_ms,
            room_tone=room_tone,
            room_tone_file=room_tone_file,
            denoise=denoise,
            voice_eq=voice_eq,
            compress=compress,
            speed_factor=speed_factor,
        )


    # ------------------------------------------------------------------
    # Noise reduction (FFT denoiser)
    # ------------------------------------------------------------------

    def apply_denoise(
        self,
        input_path: Path,
        output_path: Path,
        noise_reduction_db: float = 20.0,
    ) -> bool:
        """Apply FFT-based noise reduction using FFmpeg ``afftdn`` filter.

        This is effective for removing steady background noise (hum,
        hiss, fan noise) from recordings.

        Args:
            input_path: Path to input audio.
            output_path: Destination for denoised audio.
            noise_reduction_db: Amount of noise reduction in dB.
                Higher values remove more noise but may cause artifacts.
                Typical range: 12–30 dB.
        """
        nr = max(1.0, min(noise_reduction_db, 40.0))
        af = f"afftdn=nr={nr}:nf=-25"

        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", af,
            "-ar", str(self.sample_rate),
            "-ac", "1" if self.mono else "2",
            "-b:a", self.bitrate,
            str(output_path),
        ]
        return self._run_ffmpeg(cmd)

    # ------------------------------------------------------------------
    # Voice-optimized EQ
    # ------------------------------------------------------------------

    def apply_voice_eq(
        self,
        input_path: Path,
        output_path: Path,
        preset: str = "warm",
    ) -> bool:
        """Apply a voice-optimized equalizer chain.

        Presets:
        - ``"warm"``: Boost low-mids, gentle high shelf roll-off.
          Good for male narrators, adds richness.
        - ``"bright"``: Boost presence (2–5 kHz), gentle bass roll-off.
          Good for female narrators, adds clarity.
        - ``"neutral"``: Gentle presence boost, subtle overall.
          Good for multi-voice content.

        Args:
            input_path: Path to input audio.
            output_path: Destination for EQ\'d audio.
            preset: EQ preset name.
        """
        presets = {
            "warm": (
                "equalizer=f=200:t=h:w=200:g=2,"
                "equalizer=f=3000:t=h:w=500:g=-1,"
                "equalizer=f=100:t=l:w=50:g=3,"
                "equalizer=f=8000:t=h:w=3000:g=-3"
            ),
            "bright": (
                "equalizer=f=100:t=l:w=50:g=-2,"
                "equalizer=f=2500:t=h:w=1000:g=2,"
                "equalizer=f=4000:t=h:w=1000:g=1.5,"
                "equalizer=f=8000:t=h:w=3000:g=1"
            ),
            "neutral": (
                "equalizer=f=3000:t=h:w=1000:g=1,"
                "equalizer=f=100:t=l:w=50:g=1"
            ),
        }

        af = presets.get(preset, presets["neutral"])

        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", af,
            "-ar", str(self.sample_rate),
            "-ac", "1" if self.mono else "2",
            "-b:a", self.bitrate,
            str(output_path),
        ]
        return self._run_ffmpeg(cmd)

    # ------------------------------------------------------------------
    # Dynamic range compression
    # ------------------------------------------------------------------

    def apply_compression(
        self,
        input_path: Path,
        output_path: Path,
        threshold_db: float = -20.0,
        ratio: float = 3.0,
        attack_ms: float = 5.0,
        release_ms: float = 50.0,
    ) -> bool:
        """Apply dynamic range compression for consistent loudness.

        Uses FFmpeg ``acompressor`` followed by ``alimiter`` to prevent
        clipping. This smooths out volume variations between quiet
        and loud passages, making the audiobook more comfortable to
        listen to in noisy environments.

        Args:
            input_path: Path to input audio.
            output_path: Destination for compressed audio.
            threshold_db: Level above which compression starts (dB).
            ratio: Compression ratio.  Higher = more compression.
            attack_ms: How quickly compression engages (ms).
            release_ms: How quickly compression releases (ms).
        """
        af = (
            f"acompressor=threshold={threshold_db}dB:ratio={ratio}:"
            f"attack={attack_ms}:release={release_ms}:"
            f"mk=0.02:makeup=1,"
            f"alimiter=limit=-1.0dB:level=false"
        )

        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", af,
            "-ar", str(self.sample_rate),
            "-ac", "1" if self.mono else "2",
            "-b:a", self.bitrate,
            str(output_path),
        ]
        return self._run_ffmpeg(cmd)

    # ------------------------------------------------------------------
    # Speed / pitch control
    # ------------------------------------------------------------------

    def apply_speed(
        self,
        input_path: Path,
        output_path: Path,
        factor: float = 1.0,
    ) -> bool:
        """Adjust playback speed without changing pitch.

        Uses FFmpeg ``atempo`` filter.  For factors outside 0.5–2.0,
        multiple atempo filters are chained.

        Args:
            input_path: Path to input audio.
            output_path: Destination for speed-adjusted audio.
            factor: Speed multiplier.  1.5 = 50% faster, 0.75 = 25% slower.
        """
        # Clamp to reasonable range
        factor = max(0.5, min(4.0, factor))

        # Build chained atempo filters (each supports 0.5–100.0)
        filters: list[str] = []
        remaining = factor
        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        filters.append(f"atempo={remaining:.4f}")

        af = ",".join(filters)

        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", af,
            "-ar", str(self.sample_rate),
            "-ac", "1" if self.mono else "2",
            "-b:a", self.bitrate,
            str(output_path),
        ]
        return self._run_ffmpeg(cmd)

    # ------------------------------------------------------------------
    # Silence gap analysis
    # ------------------------------------------------------------------

    def analyze_silence_gaps(
        self,
        input_path: Path,
        min_duration_ms: float = 500.0,
        threshold_db: float = -40.0,
    ) -> List[SilenceGap]:
        """Detect silence gaps in an audio file for quality control.

        Uses FFmpeg ``silencedetect`` to find periods of silence longer
        than *min_duration_ms*.

        Args:
            input_path: Path to the audio file to analyze.
            min_duration_ms: Minimum silence duration to report (ms).
            threshold_db: Noise threshold below which audio is silence.

        Returns:
            List of :class:`SilenceGap` objects.
        """
        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-af", f"silencedetect=noise={threshold_db}dB:d={min_duration_ms / 1000.0}",
            "-f", "null", "-",
        ]
        gaps: List[SilenceGap] = []

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            stderr = proc.stderr

            # Parse silence_start and silence_end pairs
            starts = re.findall(r"silence_start\s*:\s*([\d.]+)", stderr)
            ends = re.findall(r"silence_end\s*:\s*([\d.]+)\s*\|\s*silence_duration\s*:\s*([\d.]+)", stderr)

            for i, start_s in enumerate(starts):
                start_ms = float(start_s) * 1000.0
                if i < len(ends):
                    end_ms = float(ends[i][0]) * 1000.0
                    duration_ms = float(ends[i][1]) * 1000.0
                else:
                    # No end detected — use total duration
                    total_s = self.get_duration_ms(input_path) / 1000.0
                    end_ms = total_s * 1000.0
                    duration_ms = end_ms - start_ms

                gaps.append(SilenceGap(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=duration_ms,
                    loudness_db=threshold_db,
                ))
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass

        return gaps

    # ------------------------------------------------------------------
    # Duration helper
    # ------------------------------------------------------------------

    def get_duration_ms(self, filepath: Path) -> float:
        """Return the duration of *filepath* in milliseconds via ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(filepath),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, check=False
            )
            return float(proc.stdout.strip()) * 1000.0
        except (ValueError, subprocess.TimeoutExpired, OSError):
            return 0.0

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_duration(ms: float) -> str:
        """Format milliseconds into ``HH:MM:SS`` string."""
        total_seconds = int(ms / 1000)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def build_chapter_markers(
        chapter_paths: List[Path],
        chapter_titles: Optional[List[str]] = None,
    ) -> List[ChapterMarker]:
        """Build a list of :class:`ChapterMarker` from per-chapter file paths.

        Each file's duration is measured with ffprobe and markers are
        contiguous (end_ms of one == start_ms of the next).
        """
        markers: List[ChapterMarker] = []
        current_ms = 0
        for idx, path in enumerate(chapter_paths):
            title = chapter_titles[idx] if chapter_titles and idx < len(chapter_titles) else f"Chapter {idx + 1}"
            # Use a temporary processor just for ffprobe access
            dur = AudioPostProcessor().get_duration_ms(path)
            markers.append(ChapterMarker(title=title, start_ms=int(current_ms), end_ms=int(current_ms + dur)))
            current_ms += dur
        return markers

    # ------------------------------------------------------------------
    # FFmpeg availability check
    # ------------------------------------------------------------------

    @staticmethod
    def check_ffmpeg() -> Tuple[bool, str]:
        """Verify that *ffmpeg* and *ffprobe* are available on ``$PATH``."""
        missing: list[str] = []
        for tool in ("ffmpeg", "ffprobe"):
            try:
                subprocess.run(
                    [tool, "-version"],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
            except FileNotFoundError:
                missing.append(tool)
            except subprocess.TimeoutExpired:
                missing.append(tool)

        if missing:
            return False, f"Missing tools: {', '.join(missing)}"
        return True, "FFmpeg and FFprobe are available"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_ffmpeg(self, cmd: List[str]) -> bool:
        """Execute an FFmpeg command and return ``True`` on success.

        Now captures stderr for diagnostics and tracks statistics.
        """
        import time as _time
        self._diagnostics.commands_run += 1
        t0 = _time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=1800,
                check=False,
            )
            elapsed = _time.monotonic() - t0
            self._diagnostics.total_process_time_s += elapsed
            stderr_text = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""

            if proc.returncode == 0:
                self._diagnostics.commands_succeeded += 1
                return True
            else:
                self._diagnostics.commands_failed += 1
                self._diagnostics.stderr_output.append(
                    f"CMD: {' '.join(cmd[:6])}... RC={proc.returncode}\n"
                    f"{stderr_text[-500:]}"
                )
                logger.warning("FFmpeg failed (rc=%d): %s", proc.returncode, stderr_text[-300:])
                return False
        except subprocess.TimeoutExpired:
            self._diagnostics.commands_failed += 1
            self._diagnostics.stderr_output.append(f"CMD timed out: {' '.join(cmd[:6])}...")
            logger.warning("FFmpeg command timed out")
            return False
        except (OSError, FileNotFoundError) as e:
            self._diagnostics.commands_failed += 1
            self._diagnostics.stderr_output.append(f"OS error: {e}")
            logger.error("FFmpeg not found or OS error: %s", e)
            return False

    def _report_progress(self, fraction: float, stage: str) -> None:
        """Fire the progress callback if set."""
        if self.progress_callback is not None:
            try:
                self.progress_callback(fraction, stage)
            except Exception:
                pass  # never let callback errors break the pipeline

    def cleanup_temp_dirs(self) -> None:
        """Remove all temporary directories created during processing."""
        import shutil as _shutil
        for d in self._temp_dirs:
            try:
                _shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass
        self._temp_dirs.clear()

    @property
    def diagnostics(self) -> ProcessingDiagnostics:
        """Access diagnostics from the most recent processing run."""
        return self._diagnostics

    def reset_diagnostics(self) -> None:
        """Reset diagnostics for a new processing run."""
        self._diagnostics = ProcessingDiagnostics()

    @staticmethod
    def _ms_to_ffmpeg_ts(ms: int) -> str:
        """Convert milliseconds to an FFmpeg ``HH:MM:SS.mmm`` timestamp."""
        total_seconds = ms / 1000.0
        hours = int(total_seconds // 3600)
        remainder = total_seconds - hours * 3600
        minutes = int(remainder // 60)
        seconds = remainder - minutes * 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
