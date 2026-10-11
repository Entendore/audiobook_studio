"""
``backends`` — TTS Backend Plugin Package
==========================================

Importing this package auto-registers all concrete backend subclasses
via the ``TTSBackend.__init_subclass__`` mechanism.

**Public API** (re-exported here for convenience)::

    from backends import (
        TTSBackend, VoiceInfo,
        BackendType, DeviceType, DeviceManager, device_manager,
        SynthesisChunk, ChunkPlanner, ChunkOrchestrator,
        EdgeTTSBackend, QwenLocalBackend, OmniVoiceBackend,
    )

**Adding a new backend:**

1. Create a new ``.py`` file inside this package (e.g. ``backends/my_backend.py``).
2. Subclass ``backends.base.TTSBackend``, set ``backend_id="..."`` and
   ``backend_type`` (``CLOUD`` / ``LOCAL_GPU`` / ``LOCAL_CPU``).
3. Import it below so it gets registered on package load.
"""

# --- Base: types, device manager, ABC, orchestration ---
from backends.base import (                           # noqa: F401
    BackendType,
    DeviceType,
    DeviceCapabilities,
    DeviceManager,
    VoiceInfo,
    TTSBackend,
    SynthesisChunk,
    ChunkPlanner,
    ChunkOrchestrator,
    device_manager,
)

# --- Concrete backends (trigger auto-registration) ---
from backends.edge_tts import EdgeTTSBackend          # noqa: F401
from backends.qwen_local import QwenLocalBackend      # noqa: F401
from backends.omnivoice import OmniVoiceBackend       # noqa: F401

__all__ = [
    # Device
    "DeviceType",
    "DeviceCapabilities",
    "DeviceManager",
    "device_manager",
    # Base
    "BackendType",
    "VoiceInfo",
    "TTSBackend",
    "SynthesisChunk",
    "ChunkPlanner",
    "ChunkOrchestrator",
    # Concrete
    "EdgeTTSBackend",
    "QwenLocalBackend",
    "OmniVoiceBackend",
]