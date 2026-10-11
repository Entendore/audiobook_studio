"""
TTS Backend Plugin System
=========================

**This module is now a thin backward-compatible shim.**

All implementation has been moved to the ``backends`` package::

    backends/
        __init__.py      — package init, re-exports, auto-registration
        base.py          — TTSBackend ABC, DeviceManager, VoiceInfo,
                           SynthesisChunk, ChunkPlanner, ChunkOrchestrator
        edge_tts.py      — EdgeTTSBackend (cloud)
        qwen_local.py    — QwenLocalBackend (local GPU/CPU, smart device)
        omnivoice.py     — OmniVoiceBackend (local GPU/CPU, smart device)

Existing imports like ``from tts_backends import TTSBackend, VoiceInfo``
continue to work unchanged.
"""

from backends import (                                  # noqa: F401
    # Device
    DeviceType,
    DeviceCapabilities,
    DeviceManager,
    device_manager,
    # Base
    BackendType,
    VoiceInfo,
    TTSBackend,
    SynthesisChunk,
    ChunkPlanner,
    ChunkOrchestrator,
    # Concrete
    EdgeTTSBackend,
    QwenLocalBackend,
    OmniVoiceBackend,
)

__all__ = [
    "DeviceType",
    "DeviceCapabilities",
    "DeviceManager",
    "device_manager",
    "BackendType",
    "VoiceInfo",
    "TTSBackend",
    "SynthesisChunk",
    "ChunkPlanner",
    "ChunkOrchestrator",
    "EdgeTTSBackend",
    "QwenLocalBackend",
    "OmniVoiceBackend",
]