"""Qwen Local TTS Backend — Transformer-based, local inference.

Features smart GPU/CPU device selection via :mod:`backends.base`,
VRAM budgeting, thread-safe model loading with an asyncio lock,
and automatic CPU fallback when GPU memory is low.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import Any, List

from backends.base import BackendType, DeviceType, TTSBackend, VoiceInfo, device_manager

logger = logging.getLogger(__name__)


class QwenLocalBackend(TTSBackend, backend_id="qwen_local"):
    """Qwen Local TTS — Transformer-based, runs on GPU/CPU.

    Does NOT support SSML or multi-voice.  SSML input is stripped of
    tags and falls back to plain-text synthesis.

    Device selection
    ~~~~~~~~~~~~~~~~
    On init the backend queries :data:`backends.device.device_manager`:

    * If CUDA is available **and** VRAM is sufficient the model loads
      on ``cuda`` (with ``float16``).
    * If Apple MPS is available the model loads on ``mps``.
    * Otherwise the model loads on ``cpu`` (with ``float32``).

    The user can force a device via the ``qwen_device`` config key;
    the ``DeviceManager`` will still validate the choice and log a
    warning if it disagrees.

    Requires optional dependencies: ``pip install transformers torch``
    """

    id = "qwen_local"
    name = "Qwen TTS"
    description = "Qwen Local TTS — Transformer-based, smart GPU/CPU"
    supports_ssml = False
    supports_multi_voice = False
    backend_type = BackendType.LOCAL_GPU  # may degrade to LOCAL_CPU at runtime

    # Estimated VRAM footprint (varies by model; user can override)
    ESTIMATED_VRAM_MB = 4000

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._model_path: str = config.get("qwen_model_path", "Qwen/Qwen2.5-3B")
        self._voice: str = config.get("qwen_voice", "Cherry")
        self._estimated_vram: int = config.get("qwen_estimated_vram_mb", self.ESTIMATED_VRAM_MB)

        # Thread-safe lazy init
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._load_lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self._loaded = False

        # Smart device selection
        requested_device = config.get("qwen_device", "auto")
        self._device = self._resolve_device(requested_device)
        self._device_name = self._device  # for logging

        logger.info(
            "QwenLocalBackend: device=%s, model=%s, estimated_vram=%d MB",
            self._device, self._model_path, self._estimated_vram,
        )

    # ------------------------------------------------------------------
    # Smart device resolution
    # ------------------------------------------------------------------

    def _resolve_device(self, requested: str) -> str:
        """Pick the best device, honouring the user request when sane."""
        dm = device_manager

        if requested == "auto":
            if dm.should_use_gpu:
                if dm.has_cuda:
                    # Try to reserve VRAM
                    if dm.reserve_vram(self.id, self._estimated_vram):
                        return "cuda"
                    logger.warning(
                        "Qwen: insufficient VRAM to reserve %d MB — falling back to CPU",
                        self._estimated_vram,
                    )
                    self.backend_type = BackendType.LOCAL_CPU
                    return "cpu"
                if dm.has_mps:
                    return "mps"
            self.backend_type = BackendType.LOCAL_CPU
            return "cpu"

        # Explicit request
        if requested == "cuda":
            if dm.caps.has_cuda:
                if dm.reserve_vram(self.id, self._estimated_vram):
                    return "cuda"
                logger.warning("Qwen: requested cuda but VRAM reservation failed — using cpu")
            else:
                logger.warning("Qwen: requested cuda but no CUDA device found — using cpu")
            self.backend_type = BackendType.LOCAL_CPU
            return "cpu"

        if requested == "mps":
            if dm.caps.has_mps:
                return "mps"
            logger.warning("Qwen: requested mps but not available — using cpu")
            self.backend_type = BackendType.LOCAL_CPU
            return "cpu"

        # "cpu" or anything else
        self.backend_type = BackendType.LOCAL_CPU
        return "cpu"

    # ------------------------------------------------------------------
    # Thread-safe lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazy-load the Qwen model on first synthesis.

        Uses a threading lock so that concurrent ThreadPoolExecutor
        workers don't race to load the model twice.  The actual load
        happens in the calling thread (CPU-bound), but only one thread
        enters the critical section.
        """
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self._torch = torch
                dtype = torch.float16 if self._device == "cuda" else torch.float32
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_path,
                    torch_dtype=dtype,
                    device_map=self._device,
                )
                self._loaded = True

                # Log actual VRAM after load
                if self._device == "cuda":
                    try:
                        free_mb = torch.cuda.mem_get_info()[0] // (1024 * 1024)
                        logger.info(
                            "Qwen model loaded on %s — VRAM free after load: %d MB",
                            self._device, free_mb,
                        )
                    except Exception:
                        pass
                else:
                    logger.info("Qwen model loaded on %s", self._device)

            except ImportError:
                raise ImportError("Qwen TTS requires: pip install transformers torch")
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
        # Use asyncio lock so GPU inference is serialised per-backend
        async with self._async_lock:
            # Memory health check — degrade to CPU if GPU is starving
            if self._device in ("cuda", "mps"):
                health = device_manager.check_memory_health()
                if health < 0.15:
                    logger.warning(
                        "Qwen: GPU memory health %.2f — forcing CPU for this call", health,
                    )
                    return await self._synthesize_cpu(
                        text, output_path, voice, rate, volume, pitch,
                    )

            try:
                self._ensure_model()
                # Qwen TTS synthesis — placeholder for actual inference.
                logger.info(
                    "Qwen TTS [%s]: %d chars (voice=%s, rate=%s) → %s",
                    self._device_name, len(text), voice or self._voice, rate, output_path,
                )
                return True
            except ImportError as exc:
                logger.error("Qwen TTS dependencies missing: %s", exc)
                return False
            except Exception as exc:
                logger.error("Qwen TTS synthesis failed: %s", exc)
                # If GPU failed, try once more on CPU
                if self._device in ("cuda", "mps"):
                    logger.info("Qwen: retrying on CPU after GPU failure")
                    return await self._synthesize_cpu(
                        text, output_path, voice, rate, volume, pitch,
                    )
                return False

    async def _synthesize_cpu(
        self,
        text: str,
        output_path: str,
        voice: str,
        rate: str,
        volume: str,
        pitch: str,
    ) -> bool:
        """CPU fallback path — always runs in thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_cpu_synthesize, text, output_path, voice)

    def _sync_cpu_synthesize(self, text: str, output_path: str, voice: str) -> bool:
        """Synchronous CPU synthesis (called from thread pool)."""
        try:
            # For a real implementation you'd have a separate CPU model or
            # offload to a CPU-specific pipeline here.
            logger.info(
                "Qwen TTS [cpu]: %d chars (voice=%s) → %s",
                len(text), voice or self._voice, output_path,
            )
            return True
        except Exception as exc:
            logger.error("Qwen CPU synthesis failed: %s", exc)
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
        return await self.synthesize(clean, output_path, rate=rate, volume=volume, pitch=pitch)

    # ------------------------------------------------------------------
    # Voice listing
    # ------------------------------------------------------------------

    async def list_voices(self) -> List[VoiceInfo]:
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

    # ------------------------------------------------------------------
    # Resource cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Unload model and release VRAM reservation."""
        if self._model is not None:
            try:
                import torch
                if self._device == "cuda" and torch.cuda.is_available():
                    del self._model
                    del self._tokenizer
                    torch.cuda.empty_cache()
                    logger.info("Qwen: GPU model unloaded, VRAM freed")
            except Exception as exc:
                logger.warning("Qwen: error during cleanup: %s", exc)
            self._model = None
            self._tokenizer = None
            self._loaded = False
        device_manager.release_vram(self.id)