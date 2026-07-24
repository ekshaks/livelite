import asyncio
import time
import warnings
from pathlib import Path

from ..logging_utils import monitor_time


PROJECT_ROOT = Path(__file__).resolve().parents[3]
KOKORO_MODEL_PATH = PROJECT_ROOT / "ext" / "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = PROJECT_ROOT / "ext" / "voices-v1.0.bin"

_kokoro = None


def warn_if_kokoro_assets_missing():
    if not KOKORO_MODEL_PATH.exists():
        warnings.warn(f"Kokoro model file not found at {KOKORO_MODEL_PATH}", RuntimeWarning)
    if not KOKORO_VOICES_PATH.exists():
        warnings.warn(f"Kokoro voices file not found at {KOKORO_VOICES_PATH}", RuntimeWarning)


def get_kokoro():
    global _kokoro
    if _kokoro is None:
        warn_if_kokoro_assets_missing()
        from kokoro_onnx import Kokoro

        _kokoro = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))
    return _kokoro


async def _create_kokoro_audio(text, voice="af_sarah", speed=1.0, lang="en-us"):
    loop = asyncio.get_event_loop()
    started_at = time.perf_counter()
    audio = await loop.run_in_executor(
        None,
        lambda: get_kokoro().create(text, voice=voice, speed=speed, lang=lang),
    )
    monitor_time(
        "tts",
        "synthesize",
        time.perf_counter() - started_at,
        provider="kokoro_onnx",
        voice=voice,
    )
    return audio


async def _play_kokoro_audio(text, interrupt_event, voice="af_sarah", speed=1.0, lang="en-us"):
    import sounddevice as sd

    samples, sr = await _create_kokoro_audio(text, voice=voice, speed=speed, lang=lang)
    chunk_size = int(sr * 0.5)
    loop = asyncio.get_event_loop()
    for start in range(0, len(samples), chunk_size):
        if interrupt_event.is_set():
            sd.stop()
            break
        await loop.run_in_executor(None, lambda chunk=samples[start : start + chunk_size]: sd.play(chunk, sr, blocking=True))


class KokoroOnnxTTSProvider:
    def __init__(self, voice: str = "af_sarah", speed: float = 1.0, lang: str = "en-us"):
        self.voice = voice
        self.speed = speed
        self.lang = lang

    async def speak(self, text: str, interrupt_event: asyncio.Event) -> None:
        if interrupt_event.is_set():
            return
        await _play_kokoro_audio(text, interrupt_event, voice=self.voice, speed=self.speed, lang=self.lang)
