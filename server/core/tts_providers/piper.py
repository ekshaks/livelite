"""In-process Piper TTS provider.

Piper (``OHF-Voice/piper1-gpl``, PyPI ``piper-tts``) is a small,
GPL-3.0 ONNX TTS. On 1-vCPU CPU boxes its real-time factor is ~0.1–0.3
for the *medium* voices, and unlike ``kokoro_onnx`` it streams PCM as
it synthesizes — so first-audio latency is a small fraction of the
utterance length instead of the whole clip's synthesis time. That
directly kills the TTS gaps we saw with ``kokoro_onnx`` on the AWS
1-vCPU box.

Two output modes:

* ``local``  — plays synthesized PCM on the server speaker via
  ``sounddevice``. Same shape as the other providers.
* ``webrtc`` — writes 20 ms int16 PCM frames into an outbound WebRTC
  audio track (``pc.assistant_audio_track``) for browser playback.

Model / config file paths are configurable via env:

* ``PIPER_MODEL_PATH``   — path to ``<voice>.onnx`` (default:
  ``ext/piper/en_US-lessac-medium.onnx`` under repo root).
* ``PIPER_CONFIG_PATH``  — path to ``<voice>.onnx.json`` (default:
  ``<PIPER_MODEL_PATH>.json``).
"""

import asyncio
import os
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..logging_utils import monitor_log, monitor_time


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _model_path() -> Path:
    return Path(
        os.environ.get(
            "PIPER_MODEL_PATH",
            str(PROJECT_ROOT / "ext" / "piper" / "en_US-lessac-medium.onnx"),
        )
    )


def _config_path() -> Path:
    override = os.environ.get("PIPER_CONFIG_PATH")
    if override:
        return Path(override)
    return _model_path().with_suffix(_model_path().suffix + ".json")


_voice = None


def warn_if_piper_assets_missing() -> None:
    model = _model_path()
    config = _config_path()
    if not model.exists():
        warnings.warn(f"Piper model file not found at {model}", RuntimeWarning)
    if not config.exists():
        warnings.warn(f"Piper config file not found at {config}", RuntimeWarning)


def get_voice():
    """Load the Piper voice on first use and cache it process-wide."""
    global _voice
    if _voice is None:
        warn_if_piper_assets_missing()
        from piper import PiperVoice

        _voice = PiperVoice.load(str(_model_path()), str(_config_path()))
    return _voice


def _synthesize_pcm_bytes(text: str) -> tuple[list[bytes], int]:
    """Synthesize ``text`` into a list of raw int16 PCM byte chunks.

    Piper 1.x (``piper1-gpl``) exposes ``PiperVoice.synthesize(text)`` as
    a generator of ``AudioChunk`` dataclasses. Each chunk carries a
    ``sample_rate`` and raw little-endian int16 PCM bytes via
    ``audio_int16_bytes``. We drain the generator in a worker thread so
    the event loop stays responsive; each yielded chunk is a natural
    streaming boundary (typically one sentence).
    """
    voice = get_voice()
    chunks: list[bytes] = []
    sample_rate: Optional[int] = None
    for chunk in voice.synthesize(text):
        if sample_rate is None:
            sample_rate = chunk.sample_rate
        chunks.append(chunk.audio_int16_bytes)
    if sample_rate is None:
        # ``synthesize`` yielded nothing (empty text) — fall back to the
        # voice's configured sample rate so downstream framing math works.
        sample_rate = voice.config.sample_rate
    return chunks, sample_rate


async def _stream_piper_pcm(text, interrupt_event, on_audio_block):
    """Delegate int16 PCM blocks from Piper to ``on_audio_block``."""
    if interrupt_event.is_set():
        return
    started_at = time.perf_counter()
    loop = asyncio.get_event_loop()
    chunks, sample_rate = await loop.run_in_executor(None, _synthesize_pcm_bytes, text)
    monitor_time(
        "tts",
        "synthesize",
        time.perf_counter() - started_at,
        provider="piper",
    )
    frame_samples = max(1, int(sample_rate * 0.02))  # 20 ms frames

    first = True
    for chunk_bytes in chunks:
        if interrupt_event.is_set():
            monitor_log("tts provider=piper event=interrupted")
            return
        block = np.frombuffer(chunk_bytes, dtype=np.int16)
        for start in range(0, len(block), frame_samples):
            if interrupt_event.is_set():
                monitor_log("tts provider=piper event=interrupted")
                return
            frame = block[start : start + frame_samples]
            await on_audio_block(frame, sample_rate)
            if first:
                first = False
                monitor_log("tts provider=piper event=first_pcm")


def warm_up() -> None:
    """Force-load the Piper voice and pay the graph-init cost.

    First ``PiperVoice.load`` + first synthesize on a fresh process is
    ~200–800 ms on a 1-vCPU box. Doing it before the server accepts
    connections keeps that off the first user utterance.
    """
    started_at = time.perf_counter()
    voice = get_voice()
    for _ in voice.synthesize("hi"):
        pass
    print(f"Piper warmed up in {time.perf_counter() - started_at:.2f} s")


async def _play_piper_local(text, interrupt_event):
    """Play a Piper stream on the server's default output device."""
    import sounddevice as sd

    voice = get_voice()
    stream = sd.OutputStream(
        samplerate=voice.config.sample_rate,
        channels=1,
        dtype="int16",
    )
    stream.start()

    loop = asyncio.get_event_loop()

    async def write_local(block, sample_rate):
        await loop.run_in_executor(None, stream.write, block)

    try:
        await _stream_piper_pcm(text, interrupt_event, write_local)
    finally:
        stream.stop()
        stream.close()


async def _stream_piper_to_track(text, interrupt_event, audio_track):
    """Write Piper PCM into an outbound WebRTC audio track."""
    first_track_write = True

    async def write_track(block, sample_rate):
        nonlocal first_track_write
        if first_track_write:
            first_track_write = False
            monitor_log("tts provider=piper event=first_pcm_to_webrtc_track")
        await audio_track.write_pcm(block, sample_rate=sample_rate)

    await _stream_piper_pcm(text, interrupt_event, write_track)


class PiperTTSProvider:
    """In-process Piper TTS provider.

    Parameters
    ----------
    output:
        ``"local"`` writes to the server speaker via ``sounddevice``.
        ``"webrtc"`` writes 20 ms PCM blocks into ``audio_track``.
    audio_track:
        Outbound aiortc audio track — required when ``output="webrtc"``.
    """

    def __init__(
        self,
        output: str = "local",
        audio_track: Optional[Any] = None,
    ):
        if output not in {"local", "webrtc"}:
            raise ValueError(f"Unknown PiperTTSProvider output: {output}")
        if output == "webrtc" and audio_track is None:
            raise ValueError("audio_track is required for PiperTTSProvider(output='webrtc')")
        self.output = output
        self.audio_track = audio_track

    async def speak(self, text: str, interrupt_event: asyncio.Event) -> None:
        if interrupt_event.is_set():
            return
        if self.output == "local":
            await _play_piper_local(text, interrupt_event)
            return
        await _stream_piper_to_track(text, interrupt_event, self.audio_track)

    def clear_output(self) -> None:
        """Best-effort clear of any queued audio in the outbound track."""
        if (
            self.output == "webrtc"
            and self.audio_track is not None
            and hasattr(self.audio_track, "clear")
        ):
            self.audio_track.clear()
