"""
TTS Backend Base Classes & Device Manager
==========================================

Provides:

* **Device manager** — singleton ``device_manager`` that probes the
  system once at import time and exposes smart GPU/CPU selection,
  VRAM budgeting, concurrency limits, and memory health checks.
* **Abstract base class** — ``TTSBackend`` with auto-registration
  registry, SSML builder, and per-speaker override logic.
* **Chunk orchestration** — ``SynthesisChunk``, ``ChunkPlanner``,
  ``ChunkOrchestrator`` for concurrent multi-segment synthesis.
* **VoiceInfo** — immutable voice descriptor dataclass.

Concurrency strategy (per ``BackendType``):

* **CLOUD**      → unlimited ``asyncio`` tasks (I/O-bound).
* **LOCAL_GPU**  → bounded by ``DeviceManager.gpu_concurrency_limit()``
  via an ``asyncio.Semaphore``.
* **LOCAL_CPU**  → dispatched to a ``ThreadPoolExecutor`` sized by
  ``DeviceManager.cpu_thread_count()``.

Concrete backends live in sibling modules (``edge_tts``, ``qwen_local``,
``omnivoice``) and are re-exported via ``backends.__init__``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Device types
# ═══════════════════════════════════════════════════════════════════

class DeviceType(str, Enum):
    """Hardware device categories, ordered by preference."""
    CUDA = "cuda"
    MPS = "mps"          # Apple Silicon
    CPU = "cpu"


# ═══════════════════════════════════════════════════════════════════
# Backend execution type — drives concurrency strategy
# ═══════════════════════════════════════════════════════════════════

class BackendType(str, Enum):
    """Execution environment for a backend."""
    CLOUD = "cloud"          # I/O-bound, unlimited concurrency
    LOCAL_GPU = "local_gpu"  # GPU-bound, bounded by VRAM
    LOCAL_CPU = "local_cpu"  # CPU-bound, bounded by core count


# ═══════════════════════════════════════════════════════════════════
# DeviceCapabilities — hardware snapshot
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DeviceCapabilities:
    """Immutable snapshot of hardware capabilities detected at import time."""

    device_type: DeviceType
    gpu_name: str = ""
    cuda_version: str = ""
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    cpu_count_physical: int = 0
    cpu_count_logical: int = 0
    ram_total_gb: float = 0.0
    ram_free_gb: float = 0.0
    torch_available: bool = False
    mps_available: bool = False

    @property
    def has_gpu(self) -> bool:
        return self.device_type in (DeviceType.CUDA, DeviceType.MPS)

    @property
    def has_cuda(self) -> bool:
        return self.device_type == DeviceType.CUDA

    @property
    def has_mps(self) -> bool:
        return self.device_type == DeviceType.MPS

    def summary(self) -> str:
        parts = [f"device={self.device_type.value}"]
        if self.gpu_name:
            parts.append(f"gpu={self.gpu_name}")
        if self.vram_total_mb:
            parts.append(f"vram={self.vram_total_mb}MB")
        parts.append(f"cpu_cores={self.cpu_count_physical}")
        parts.append(f"ram={self.ram_total_gb:.1f}GB")
        return ", ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# MemoryReservation — per-backend VRAM claim
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MemoryReservation:
    """Tracks a single backend's VRAM claim."""
    backend_id: str
    claimed_mb: int


# ═══════════════════════════════════════════════════════════════════
# DeviceManager — singleton hardware probe + resource governor
# ═══════════════════════════════════════════════════════════════════

class DeviceManager:
    """Hardware probe + resource governor.

    Call ``device_manager`` (the module-level singleton) from any backend.
    All methods are thread-safe.
    """

    def __init__(self) -> None:
        self._caps = self._probe()
        self._lock = threading.Lock()
        self._reservations: List[MemoryReservation] = []
        self._vram_headroom_pct: float = 0.20  # keep 20 % free
        self._vram_headroom_mb: int = 512       # at least 512 MB free

        logger.info("DeviceManager initialized: %s", self._caps.summary())

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def caps(self) -> DeviceCapabilities:
        return self._caps

    @property
    def device_type(self) -> DeviceType:
        return self._caps.device_type

    @property
    def optimal_device(self) -> str:
        """String suitable for ``torch.device(...)``."""
        return self._caps.device_type.value

    @property
    def should_use_gpu(self) -> bool:
        """True when a GPU was detected *and* has enough free VRAM."""
        if not self._caps.has_gpu:
            return False
        if self._caps.has_cuda:
            return self._caps.vram_free_mb > 1024  # need at least 1 GB
        if self._caps.has_mps:
            return True  # unified memory — always ok
        return False

    # ------------------------------------------------------------------
    # VRAM budgeting
    # ------------------------------------------------------------------

    def available_vram_mb(self) -> int:
        """Free VRAM minus headroom minus existing reservations."""
        if not self._caps.has_cuda:
            return 0
        with self._lock:
            reserved = sum(r.claimed_mb for r in self._reservations)
        headroom = max(
            int(self._caps.vram_total_mb * self._vram_headroom_pct),
            self._vram_headroom_mb,
        )
        return max(0, self._caps.vram_free_mb - headroom - reserved)

    def reserve_vram(self, backend_id: str, mb: int) -> bool:
        """Try to reserve *mb* of VRAM.  Returns ``True`` on success."""
        if not self._caps.has_cuda:
            return False
        with self._lock:
            if self.available_vram_mb() >= mb:
                self._reservations.append(MemoryReservation(backend_id, mb))
                logger.debug(
                    "VRAM reserved: %d MB for '%s' (total reserved now %d MB)",
                    mb, backend_id,
                    sum(r.claimed_mb for r in self._reservations),
                )
                return True
            logger.warning(
                "VRAM reservation failed: need %d MB but only %d MB available",
                mb, self.available_vram_mb(),
            )
            return False

    def release_vram(self, backend_id: str) -> None:
        """Release all VRAM reservations for a backend."""
        with self._lock:
            before = len(self._reservations)
            self._reservations = [
                r for r in self._reservations if r.backend_id != backend_id
            ]
            if len(self._reservations) < before:
                logger.debug("Released VRAM reservations for '%s'", backend_id)

    # ------------------------------------------------------------------
    # Concurrency helpers
    # ------------------------------------------------------------------

    def gpu_concurrency_limit(self) -> int:
        """Max concurrent GPU inference tasks (1 / 2 / 4 / 8 by VRAM)."""
        if not self._caps.has_cuda:
            return 0
        vram = self._caps.vram_total_mb
        if vram < 4096:
            return 1
        if vram < 8192:
            return 2
        if vram < 16384:
            return 4
        return 8

    def cpu_thread_count(self) -> int:
        """Worker threads for CPU-bound inference (physical_cores - 1, max 4)."""
        cores = max(self._caps.cpu_count_physical, 1)
        return max(1, min(cores - 1, 4))

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def check_memory_health(self) -> float:
        """0.0–1.0 health score (1.0 = plenty of headroom).  Below 0.15 → CPU fallback."""
        if not self._caps.has_cuda:
            return 1.0
        try:
            import torch
            if not torch.cuda.is_available():
                return 1.0
            free = torch.cuda.mem_get_info()[0] / (1024 * 1024)
            total = self._caps.vram_total_mb
            if total == 0:
                return 1.0
            ratio = free / total
            logger.debug("GPU memory health: %.1f %% free", ratio * 100)
            return min(1.0, ratio / 0.5)
        except Exception:
            return 1.0

    # ------------------------------------------------------------------
    # Internal: hardware probing (runs once)
    # ------------------------------------------------------------------

    @staticmethod
    def _probe() -> DeviceCapabilities:
        """Detect hardware capabilities.  Expensive — called once."""
        torch_available = False
        mps_available = False
        gpu_name = ""
        cuda_version = ""
        vram_total = 0
        vram_free = 0

        try:
            import torch  # noqa: F401
            torch_available = True
        except ImportError:
            pass

        if torch_available:
            try:
                import torch
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    gpu_name = props.name
                    vram_total = props.total_mem // (1024 * 1024)
                    free, _ = torch.cuda.mem_get_info()
                    vram_free = free // (1024 * 1024)
                    cuda_version = (
                        f"{torch.version.cuda}" if torch.version.cuda else "unknown"
                    )
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    mps_available = True
            except Exception as exc:
                logger.debug("PyTorch device probe failed: %s", exc)

        if torch_available:
            if vram_total > 0:
                device_type = DeviceType.CUDA
            elif mps_available:
                device_type = DeviceType.MPS
            else:
                device_type = DeviceType.CPU
        else:
            device_type = DeviceType.CPU

        cpu_phys, cpu_log = _cpu_counts()
        ram_total, ram_free = _ram_info()

        return DeviceCapabilities(
            device_type=device_type,
            gpu_name=gpu_name,
            cuda_version=cuda_version,
            vram_total_mb=vram_total,
            vram_free_mb=vram_free,
            cpu_count_physical=cpu_phys,
            cpu_count_logical=cpu_log,
            ram_total_gb=ram_total,
            ram_free_gb=ram_free,
            torch_available=torch_available,
            mps_available=mps_available,
        )


# ═══════════════════════════════════════════════════════════════════
# Platform helpers (private)
# ═══════════════════════════════════════════════════════════════════

def _cpu_counts() -> tuple[int, int]:
    """Return ``(physical_cores, logical_cores)``."""
    logical = os.cpu_count() or 1
    try:
        if platform.system() == "Linux":
            phys = _read_sysfs_cpu_cores() or logical
        elif platform.system() == "Windows":
            phys = logical // 2
        elif platform.system() == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True, text=True, timeout=5,
            )
            phys = int(out.stdout.strip()) if out.returncode == 0 else logical
        else:
            phys = logical
    except Exception:
        phys = logical
    return phys, logical


def _read_sysfs_cpu_cores() -> Optional[int]:
    """Read physical core count from Linux sysfs."""
    try:
        cores = set()
        base = "/sys/devices/system/cpu"
        if not os.path.isdir(base):
            return None
        for entry in os.listdir(base):
            m = entry.startswith("cpu") and entry[3:].isdigit()
            if m:
                core_path = os.path.join(base, entry, "topology", "core_id")
                if os.path.isfile(core_path):
                    cores.add(open(core_path).read().strip())
        return len(cores) if cores else None
    except Exception:
        return None


def _ram_info() -> tuple[float, float]:
    """Return ``(total_gb, free_gb)``."""
    try:
        import psutil  # type: ignore[import-untyped]
        mem = psutil.virtual_memory()
        return mem.total / (1024 ** 3), mem.available / (1024 ** 3)
    except ImportError:
        pass

    if platform.system() == "Linux":
        try:
            info: Dict[str, int] = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        info[parts[0].rstrip(":")] = int(parts[1])
            total = info.get("MemTotal", 0) / (1024 ** 2)
            available = info.get("MemAvailable", info.get("MemFree", 0)) / (1024 ** 2)
            return total, available
        except Exception:
            pass

    return 0.0, 0.0


# ═══════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════

device_manager = DeviceManager()


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
    style: str = ""

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
# XML escape helpers
# ═══════════════════════════════════════════════════════════════════

def _xml_escape_text(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _xml_escape_attr(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;").replace("'", "&apos;")
    )


# ═══════════════════════════════════════════════════════════════════
# SynthesisChunk — single unit of TTS work
# ═══════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class SynthesisChunk:
    """A single, self-contained unit of TTS work.

    Attributes:
        index:       Monotonic position in the original segment list.
        text:        The text to synthesize (may be merged from multiple
                     consecutive same-voice segments).
        voice:       Voice identifier to use.
        speaker:     Speaker label (for logging / grouping).
        rate:        Rate adjustment string (e.g. ``"+20%"``).
        volume:      Volume adjustment string.
        pitch:       Pitch adjustment string.
        backend_id:  Backend to route this chunk to (``""`` = use parent).
        char_count:  ``len(text)`` — cached for quick budgeting.
    """
    index: int
    text: str
    voice: str
    speaker: str = ""
    rate: str = "+0%"
    volume: str = "+0%"
    pitch: str = "+0Hz"
    backend_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "char_count", len(self.text))


# ═══════════════════════════════════════════════════════════════════
# ChunkPlanner — groups raw segments into optimal chunks
# ═══════════════════════════════════════════════════════════════════

class ChunkPlanner:
    """Transforms a flat list of ``(text, speaker)`` tuples into a list
    of :class:`SynthesisChunk` instances ready for the orchestrator.

    Grouping rules
    --------------
    * Consecutive segments with the **same effective voice** are merged
      into one chunk (fewer TTS calls = faster).
    * A merged chunk must not exceed ``max_chars`` (default 3000).
      If merging would exceed the limit, a new chunk is started.
    * Per-speaker ``speaker_configs`` overrides for ``voice_id``,
      ``rate``, ``volume``, ``pitch``, and ``backend_id`` are resolved
      at planning time so the orchestrator needs no further lookups.
    """

    def __init__(self, max_chars: int = 3000, max_chunks: int = 500) -> None:
        self.max_chars = max_chars
        self.max_chunks = max_chunks

    def plan(
        self,
        segments: List[Tuple[str, str]],
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        speaker_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[SynthesisChunk]:
        """Build an optimised chunk list from raw segments.

        Returns chunks sorted by ``index``.
        """
        if not segments:
            return []

        chunks: List[SynthesisChunk] = []
        current_text = ""
        current_chars = 0
        chunk_start_idx = 0

        def _effective_voice(speaker: str) -> str:
            if speaker_configs and speaker in speaker_configs:
                return speaker_configs[speaker].get("voice_id", speaker)
            return speaker

        def _effective_backend(speaker: str) -> str:
            if speaker_configs and speaker in speaker_configs:
                return speaker_configs[speaker].get("backend_id", "")
            return ""

        def _flush(prev_voice: str, prev_speaker: str) -> None:
            nonlocal current_text, current_chars, chunk_start_idx
            if not current_text.strip():
                chunk_start_idx += 1
                return
            spk_cfg = (speaker_configs or {}).get(prev_speaker, {})
            chunks.append(SynthesisChunk(
                index=chunk_start_idx,
                text=current_text.strip(),
                voice=prev_voice,
                speaker=prev_speaker,
                rate=spk_cfg.get("rate", rate),
                volume=spk_cfg.get("volume", volume),
                pitch=spk_cfg.get("pitch", pitch),
                backend_id=spk_cfg.get("backend_id", ""),
            ))
            current_text = ""
            current_chars = 0
            chunk_start_idx += 1

        prev_voice = ""
        prev_speaker = ""

        for i, (text, speaker) in enumerate(segments):
            if len(chunks) >= self.max_chunks:
                logger.warning(
                    "ChunkPlanner hit max_chunks=%d cap at segment %d — "
                    "remaining segments dropped",
                    self.max_chunks, i,
                )
                break

            eff_voice = _effective_voice(speaker)
            eff_backend = _effective_backend(speaker)

            same_group = (
                eff_voice == prev_voice
                and eff_backend == _effective_backend(prev_speaker)
                and prev_speaker == speaker
                and current_chars + len(text) <= self.max_chars
            )

            if same_group and current_text:
                current_text += " " + text
                current_chars += len(text)
            else:
                if current_text:
                    _flush(prev_voice, prev_speaker)
                current_text = text
                current_chars = len(text)
                prev_voice = eff_voice
                prev_speaker = speaker

        if current_text:
            _flush(prev_voice, prev_speaker)

        logger.info(
            "ChunkPlanner: %d segments → %d chunks (max_chars=%d)",
            len(segments), len(chunks), self.max_chars,
        )
        return chunks


# ═══════════════════════════════════════════════════════════════════
# ChunkOrchestrator — concurrent execution + concatenation
# ═══════════════════════════════════════════════════════════════════

@dataclass
class _ChunkResult:
    """Internal bookkeeping for a single chunk's synthesis result."""
    chunk: SynthesisChunk
    path: Optional[Path]
    success: bool
    error: Optional[Exception] = None


class ChunkOrchestrator:
    """Executes a list of :class:`SynthesisChunk` concurrently and
    concatenates the resulting audio files.

    Concurrency strategy per backend type:

    * **cloud**  — ``asyncio`` tasks, no semaphore (I/O-bound).
    * **local_gpu** — ``asyncio`` tasks gated by
      ``DeviceManager.gpu_concurrency_limit()`` semaphore.
    * **local_cpu** — offloaded to a ``ThreadPoolExecutor`` sized by
      ``DeviceManager.cpu_thread_count()``.
    """

    def __init__(
        self,
        dm: Optional[DeviceManager] = None,
        gpu_semaphore: Optional[asyncio.Semaphore] = None,
        cpu_executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        self._dm = dm or device_manager
        self._gpu_sem: Optional[asyncio.Semaphore] = gpu_semaphore
        self._cpu_exec: Optional[ThreadPoolExecutor] = cpu_executor
        self._owns_executor = cpu_executor is None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def synthesize_chunks(
        self,
        chunks: List[SynthesisChunk],
        output_path: str,
        backend: TTSBackend,
        backend_registry: Optional[Dict[str, TTSBackend]] = None,
        on_progress: Optional[Callable[[int, int, SynthesisChunk], None]] = None,
        on_retry: Optional[Callable[[int, str, Exception], None]] = None,
    ) -> bool:
        """Synthesize *chunks* concurrently and concatenate into *output_path*."""
        if not chunks:
            return False

        total = len(chunks)
        temp_dir = Path(tempfile.mkdtemp(prefix="tts_orchestrator_"))
        results: List[_ChunkResult] = [None] * total  # type: ignore[list-item]

        try:
            tasks: List[asyncio.Task] = []
            for chunk in chunks:
                task = asyncio.create_task(
                    self._synthesize_one(
                        chunk, temp_dir, backend, backend_registry, on_retry,
                    )
                )
                tasks.append(task)

            completed = 0
            for future in asyncio.as_completed(tasks):
                result: _ChunkResult = await future
                results[result.chunk.index] = result
                completed += 1
                if on_progress:
                    on_progress(completed, total, result.chunk)

            valid: List[Path] = [
                r.path for r in results
                if r is not None and r.success and r.path is not None
            ]

            if not valid:
                logger.error("ChunkOrchestrator: all %d chunks failed", total)
                return False

            if len(valid) == 1:
                shutil.copy2(str(valid[0]), output_path)
                return True

            ordered = [
                r.path for r in results
                if r is not None and r.success and r.path is not None
            ]
            return self._concatenate(ordered, output_path, temp_dir)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if self._owns_executor and self._cpu_exec is not None:
                self._cpu_exec.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Per-chunk synthesis
    # ------------------------------------------------------------------

    async def _synthesize_one(
        self,
        chunk: SynthesisChunk,
        temp_dir: Path,
        default_backend: TTSBackend,
        backend_registry: Optional[Dict[str, TTSBackend]],
        on_retry: Optional[Callable[[int, str, Exception], None]],
    ) -> _ChunkResult:
        """Synthesize a single chunk, routing to the correct backend."""
        chunk_path = temp_dir / f"chunk_{chunk.index:05d}.mp3"

        backend = default_backend
        bid = getattr(default_backend, "_backend_id", getattr(default_backend, "id", ""))
        if chunk.backend_id and chunk.backend_id != bid:
            if backend_registry and chunk.backend_id in backend_registry:
                backend = backend_registry[chunk.backend_id]
            else:
                try:
                    backend = TTSBackend.create(chunk.backend_id, default_backend.config)
                except ValueError:
                    logger.error("Unknown backend '%s' for chunk %d", chunk.backend_id, chunk.index)
                    return _ChunkResult(chunk, None, False)

        max_retries = 2
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                ok = await self._dispatch(backend, chunk, str(chunk_path))
                if ok and chunk_path.exists():
                    return _ChunkResult(chunk, chunk_path, True)
                last_exc = None
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Chunk %d failed (attempt %d/%d): %s",
                    chunk.index, attempt + 1, max_retries + 1, exc,
                )
                if on_retry:
                    on_retry(attempt + 1, getattr(backend, "_backend_id", ""), exc)

        return _ChunkResult(chunk, None, False, last_exc)

    async def _dispatch(
        self,
        backend: TTSBackend,
        chunk: SynthesisChunk,
        output_path: str,
    ) -> bool:
        """Route synthesis to the correct execution strategy."""
        btype = self._get_backend_type(backend)

        if btype == BackendType.CLOUD:
            return await backend.synthesize(
                chunk.text, output_path, voice=chunk.voice,
                rate=chunk.rate, volume=chunk.volume, pitch=chunk.pitch,
            )

        if btype == BackendType.LOCAL_GPU:
            sem = self._ensure_gpu_semaphore()
            async with sem:
                if self._dm.check_memory_health() < 0.15:
                    logger.warning("GPU memory low — CPU fallback for chunk %d", chunk.index)
                    return await self._dispatch_cpu(backend, chunk, output_path)
                return await backend.synthesize(
                    chunk.text, output_path, voice=chunk.voice,
                    rate=chunk.rate, volume=chunk.volume, pitch=chunk.pitch,
                )

        # LOCAL_CPU
        return await self._dispatch_cpu(backend, chunk, output_path)

    async def _dispatch_cpu(
        self,
        backend: TTSBackend,
        chunk: SynthesisChunk,
        output_path: str,
    ) -> bool:
        """Run CPU-bound synthesis in a thread pool."""
        loop = asyncio.get_running_loop()
        executor = self._ensure_cpu_executor()
        return await loop.run_in_executor(
            executor,
            self._sync_synthesize,
            backend, chunk.text, output_path, chunk.voice,
            chunk.rate, chunk.volume, chunk.pitch,
        )

    @staticmethod
    def _sync_synthesize(
        backend: TTSBackend,
        text: str,
        output_path: str,
        voice: str,
        rate: str,
        volume: str,
        pitch: str,
    ) -> bool:
        """Synchronous wrapper for ThreadPoolExecutor."""
        try:
            result = backend.synthesize(
                text, output_path, voice=voice,
                rate=rate, volume=volume, pitch=pitch,
            )
            if asyncio.iscoroutine(result):
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(result)
                finally:
                    loop.close()
            return bool(result)
        except Exception as exc:
            logger.error("CPU synthesis failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_backend_type(self, backend: TTSBackend) -> BackendType:
        btype = getattr(backend, "backend_type", None)
        if isinstance(btype, BackendType):
            return btype
        if isinstance(btype, str) and btype in BackendType.__members__:
            return BackendType(btype)
        if hasattr(backend, "_device"):
            dev = backend._device if hasattr(backend, "_device") else "cpu"
            return BackendType.LOCAL_GPU if dev in ("cuda", "mps") else BackendType.LOCAL_CPU
        return BackendType.CLOUD

    def _ensure_gpu_semaphore(self) -> asyncio.Semaphore:
        if self._gpu_sem is None:
            limit = max(1, self._dm.gpu_concurrency_limit())
            self._gpu_sem = asyncio.Semaphore(limit)
            logger.debug("GPU semaphore created (limit=%d)", limit)
        return self._gpu_sem

    def _ensure_cpu_executor(self) -> ThreadPoolExecutor:
        if self._cpu_exec is None:
            workers = max(1, self._dm.cpu_thread_count())
            self._cpu_exec = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="tts_cpu_",
            )
            logger.debug("CPU ThreadPoolExecutor created (workers=%d)", workers)
        return self._cpu_exec

    @staticmethod
    def _concatenate(paths: List[Path], output_path: str, temp_dir: Path) -> bool:
        """Concatenate audio files with FFmpeg, preserving order."""
        concat_file = temp_dir / "concat.txt"
        lines: List[str] = []
        for p in paths:
            if p is not None:
                safe = str(p).replace("\\", "/").replace("'", "'\\''")
                lines.append(f"file '{safe}'")
        concat_file.write_text("\n".join(lines), encoding="utf-8")

        cmd = [
            "ffmpeg", "-y", "-hide_banner",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file), "-c", "copy", output_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            logger.error("FFmpeg concat failed: %s", result.stderr.decode(errors="replace")[:300])
        return result.returncode == 0


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

    Set ``backend_type`` to a :class:`BackendType` value so the
    :class:`ChunkOrchestrator` can apply the right concurrency strategy.
    """

    _registry: Dict[str, Type[TTSBackend]] = {}

    # --- Class-level descriptors (override in subclass) ---
    id: str = ""
    name: str = ""
    description: str = ""
    supports_ssml: bool = False
    supports_multi_voice: bool = False
    backend_type: BackendType = BackendType.CLOUD

    # ------------------------------------------------------------------
    # Auto-registration
    # ------------------------------------------------------------------

    def __init_subclass__(cls, backend_id: str = "", **kwargs) -> None:
        super().__init_subclass__(**kwargs)
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
        """Instantiate a registered backend by its id (keyword-arg style)."""
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
    # Abstract methods — every backend MUST implement
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
    # Multi-speaker segment synthesis (orchestrated)
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
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> bool:
        """Synthesize multi-speaker segments.

        Strategy (in order of preference):
        1. Per-speaker ``backend_id`` overrides → ChunkOrchestrator.
        2. SSML with per-speaker prosody → single SSML call.
        3. Multi-voice / multi-chunk → ChunkOrchestrator.
        4. Dominant voice fallback → single ``synthesize()``.
        """
        if not segments:
            return False

        if pronunciation_dict:
            segments = [
                (self._apply_pronunciation(text, pronunciation_dict), speaker)
                for text, speaker in segments
            ]

        # ── Per-speaker backend override → orchestrator ────────────
        has_overrides = bool(
            speaker_configs
            and any(cfg.get("backend_id") for cfg in speaker_configs.values())
        )
        if has_overrides:
            return await self._orchestrated_synthesis(
                segments, output_path, rate, volume, pitch,
                speaker_configs, backend_registry, on_retry, on_progress,
            )

        # ── Per-speaker voice/prosody + SSML ──────────────────────
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

        # ── Orchestrated multi-voice / multi-chunk ────────────────
        if self.supports_multi_voice or len(segments) > 1:
            return await self._orchestrated_synthesis(
                segments, output_path, rate, volume, pitch,
                speaker_configs, backend_registry, on_retry, on_progress,
            )

        # ── Single-voice fallback ─────────────────────────────────
        voice_counts: Dict[str, int] = {}
        for _, v in segments:
            voice_counts[v] = voice_counts.get(v, 0) + 1
        dominant_voice = max(voice_counts, key=voice_counts.get)  # type: ignore[arg-type]
        full_text = " ".join(text for text, _ in segments)
        if speaker_configs and dominant_voice in speaker_configs:
            dominant_voice = speaker_configs[dominant_voice].get("voice_id", dominant_voice)
        return await self.synthesize(
            full_text, output_path, voice=dominant_voice,
            rate=rate, volume=volume, pitch=pitch,
        )

    # ------------------------------------------------------------------
    # Orchestrated path
    # ------------------------------------------------------------------

    async def _orchestrated_synthesis(
        self,
        segments: List[Tuple[str, str]],
        output_path: str,
        rate: str,
        volume: str,
        pitch: str,
        speaker_configs: Optional[Dict[str, Dict[str, Any]]],
        backend_registry: Optional[Dict[str, TTSBackend]],
        on_retry: Optional[Callable[[int, str, Exception], None]],
        on_progress: Optional[Callable[[int, int], None]],
    ) -> bool:
        btype = self._get_own_backend_type()
        max_chars = 3000 if btype == BackendType.CLOUD else 1500

        planner = ChunkPlanner(max_chars=max_chars)
        chunks = planner.plan(
            segments, rate=rate, volume=volume, pitch=pitch,
            speaker_configs=speaker_configs,
        )
        if not chunks:
            return False

        if len(chunks) == 1:
            c = chunks[0]
            return await self.synthesize(
                c.text, output_path, voice=c.voice,
                rate=c.rate, volume=c.volume, pitch=c.pitch,
            )

        orchestrator = ChunkOrchestrator()

        def _progress(completed: int, total: int, _chunk: SynthesisChunk) -> None:
            if on_progress:
                on_progress(completed, total)

        return await orchestrator.synthesize_chunks(
            chunks, output_path, self,
            backend_registry=backend_registry,
            on_progress=_progress,
            on_retry=on_retry,
        )

    def _get_own_backend_type(self) -> BackendType:
        bt = getattr(self, "backend_type", None)
        if isinstance(bt, BackendType):
            return bt
        return BackendType.CLOUD

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
        """Build an SSML document with ``<voice>`` switching."""
        if not self.supports_ssml:
            return None

        parts: List[str] = [
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"'
            ' xmlns:mstts="http://www.w3.org/2001/mstts">',
        ]

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

        current_voice: Optional[str] = None
        for text, speaker in segments:
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
        for word, replacement in pronunciation_dict.items():
            if word and replacement:
                text = text.replace(word, replacement)
        return text

    async def preview(self, text: str, output_path: str, voice: str = "") -> bool:
        return await self.synthesize(text, output_path, voice=voice)

    async def close(self) -> None:
        """Release resources.  Override in subclasses that hold GPU models."""
        pass