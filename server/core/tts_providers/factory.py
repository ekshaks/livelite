from dataclasses import dataclass
from typing import Any, Literal, Optional

from .gemini import GeminiTTSProvider
from .kokoro_fastapi import KokoroFastApiTTSProvider
from .kokoro_onnx import KokoroOnnxTTSProvider


@dataclass
class TTSConfig:
    provider: Literal["kokoro_fastapi", "kokoro_onnx", "gemini"] = "kokoro_fastapi"
    output: Literal["local", "webrtc"] = "local"
    voice: Optional[str] = None
    model: Optional[str] = None


def create_tts_provider(config: TTSConfig, audio_track: Optional[Any] = None):
    if config.provider == "kokoro_fastapi":
        mode = "webrtc" if config.output == "webrtc" else "local"
        return KokoroFastApiTTSProvider(mode=mode, audio_track=audio_track)
    if config.provider == "kokoro_onnx":
        return KokoroOnnxTTSProvider(voice=config.voice or "af_sarah")
    if config.provider == "gemini":
        return GeminiTTSProvider(
            model=config.model or "gemini-2.5-flash-preview-tts",
            voice=config.voice or "Kore",
        )
    raise ValueError(f"Unknown TTS provider: {config.provider}")

