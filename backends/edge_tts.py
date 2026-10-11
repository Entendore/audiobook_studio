"""Edge TTS Backend — Cloud-based, 400+ voices, SSML multi-voice."""

from __future__ import annotations

import logging
from typing import Any, List

from backends.base import BackendType, TTSBackend, VoiceInfo

logger = logging.getLogger(__name__)


class EdgeTTSBackend(TTSBackend, backend_id="edge_tts"):
    """Microsoft Edge TTS — cloud-based, 400+ voices, full SSML support."""

    id = "edge_tts"
    name = "Edge TTS"
    description = "Microsoft Edge TTS — Cloud-based, 400+ voices, SSML support"
    supports_ssml = True
    supports_multi_voice = True
    backend_type = BackendType.CLOUD

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