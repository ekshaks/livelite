"""In-process Kokoro TTS backed by the ONNX runtime.

Two output modes:

* ``local``   — plays the synthesized audio on the server speaker via
  ``sounddevice``. Same as before; used on desktop.
* ``webrtc``  — writes 20 ms PCM blocks into an outbound WebRTC audio track
  so a browser client hears the audio. Preferred on headless servers.
"""

import asyncio
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..logging_utils import monitor_log, monitor_time


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


def warm_up(voice: str = "af_sarah", lang: str = "en-us") -> None:
    """Force Kokoro to load its ONNX weights and synthesize one tiny clip.

    The first ``Kokoro.create`` call on a fresh process pays the ONNX
    session-warmup cost (graph optimization, weight allocation, and the
    eSpeak G2P initialization). On a 1-vCPU box that is 1–3 s and, when
    it happens on the first user utterance, shows up as an audible gap
    before the assistant starts speaking. Calling this once before the
    server accepts connections moves the cost off the hot path.
    """
    import time as _time
    started_at = _time.perf_counter()
    kokoro = get_kokoro()
    # A one-word phrase is enough to trip the full pipeline (G2P + inference)
    # without producing meaningful audio.
    kokoro.create("hi", voice=voice, speed=1.0, lang=lang)
    print(f"Kokoro warmed up in {_time.perf_counter() - started_at:.2f} s")


async def _create_kokoro_audio(text, voice="af_sarah", speed=1.0, lang="en-us"):
    """Synthesize ``text`` to a (samples, sample_rate) pair off the event loop."""
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


def _to_pcm16(samples):
    """Convert kokoro-onnx float32 samples in [-1, 1] to int16 PCM."""
    if samples.dtype == np.int16:
        return samples
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16)


async def _play_kokoro_local(text, interrupt_event, voice, speed, lang):
    """Play a synthesized clip on the server's default output device."""
    import sounddevice as sd

    samples, sr = await _create_kokoro_audio(text, voice=voice, speed=speed, lang=lang)
    chunk_size = int(sr * 0.5)
    loop = asyncio.get_event_loop()
    for start in range(0, len(samples), chunk_size):
        if interrupt_event.is_set():
            sd.stop()
            break
        chunk = samples[start : start + chunk_size]
        await loop.run_in_executor(None, lambda chunk=chunk: sd.play(chunk, sr, blocking=True))


async def _stream_kokoro_to_track(text, interrupt_event, audio_track, voice, speed, lang):
    """Synthesize ``text`` and push int16 PCM into the outbound WebRTC track.

    ``kokoro_onnx.Kokoro.create`` is not streaming, so first-audio latency is
    the whole-clip synthesis time. Keep each utterance short (single sentence
    or two) upstream to bound perceived latency.
    """
    samples, sr = await _create_kokoro_audio(text, voice=voice, speed=speed, lang=lang)
    if interrupt_event.is_set() or len(samples) == 0:
        return
    pcm = _to_pcm16(samples)
    frame_samples = max(1, int(sr * 0.02))  # 20 ms frames — the aiortc default
    first_write_at = None
    started_at = time.perf_counter()
    for start in range(0, len(pcm), frame_samples):
        if interrupt_event.is_set():
            monitor_log("tts provider=kokoro_onnx event=interrupted")
            break
        block = pcm[start : start + frame_samples]
        await audio_track.write_pcm(block, sample_rate=sr)
        if first_write_at is None:
            first_write_at = time.perf_counter()
            monitor_log("tts provider=kokoro_onnx event=first_pcm_to_webrtc_track")
    if first_write_at is not None:
        monitor_time(
            "tts",
            "first_audio_track_write",
            first_write_at - started_at,
            provider="kokoro_onnx",
            voice=voice,
        )


class KokoroOnnxTTSProvider:
    """In-process Kokoro TTS provider (ONNX runtime).

    Parameters
    ----------
    voice, speed, lang:
        Voice pack, speech rate, and language tag passed to ``Kokoro.create``.
    output:
        ``"local"`` writes to the server speaker via ``sounddevice``.
        ``"webrtc"`` writes 20 ms PCM blocks into ``audio_track``.
    audio_track:
        Outbound aiortc audio track — required when ``output="webrtc"``.
    """

    def __init__(
        self,
        voice: str = "af_sarah",
        speed: float = 1.0,
        lang: str = "en-us",
        output: str = "local",
        audio_track: Optional[Any] = None,
    ):
        if output not in {"local", "webrtc"}:
            raise ValueError(f"Unknown KokoroOnnxTTSProvider output: {output}")
        if output == "webrtc" and audio_track is None:
            raise ValueError("audio_track is required for KokoroOnnxTTSProvider(output='webrtc')")
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self.output = output
        self.audio_track = audio_track

    async def speak(self, text: str, interrupt_event: asyncio.Event) -> None:
        if interrupt_event.is_set():
            return
        if self.output == "local":
            await _play_kokoro_local(text, interrupt_event, self.voice, self.speed, self.lang)
            return
        await _stream_kokoro_to_track(
            text, interrupt_event, self.audio_track, self.voice, self.speed, self.lang
        )

    def clear_output(self) -> None:
        """Best-effort clear of any queued audio in the outbound track."""
        if self.output == "webrtc" and self.audio_track is not None and hasattr(self.audio_track, "clear"):
            self.audio_track.clear()
