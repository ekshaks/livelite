import asyncio
import time
from typing import Any, Optional

from ..logging_utils import monitor_log, monitor_time

KOKORO_PCM_SAMPLE_RATE = 24000


def _log_tts_metrics(timings, chunk_count):
    if timings["first_buffer_received"] is not None:
        monitor_time(
            "tts",
            "first_audio",
            timings["first_buffer_received"] - timings["start"],
            provider="kokoro",
        )
    monitor_time(
        "tts",
        "stream_complete",
        timings["end"] - timings["start"],
        provider="kokoro",
        chunks=chunk_count,
        first_audio_received=timings["first_buffer_received"] is not None,
    )


async def _tts_kokoro_stream_chunks(text, interrupt_event, on_audio_block):
    """Stream Kokoro-FastAPI PCM blocks and delegate output."""
    from openai import AsyncOpenAI
    import numpy as np

    timings = {"start": time.perf_counter(), "first_buffer_received": None, "end": None}
    samplerate = KOKORO_PCM_SAMPLE_RATE
    blocksize = 1024
    chunk_count = 0

    try:
        monitor_log("tts provider=kokoro event=request_start")
        client = AsyncOpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")
        async with client.audio.speech.with_streaming_response.create(
            model="kokoro",
            voice="af_sky+af_bella",
            input=text,
            response_format="pcm",
        ) as response:
            first_chunk = True
            async for chunk in response.iter_bytes(chunk_size=blocksize):
                if first_chunk:
                    timings["first_buffer_received"] = time.perf_counter()
                    first_chunk = False
                    monitor_log("[interrupt] kokoro event=first_chunk")
                if interrupt_event.is_set():
                    print("[interrupt] kokoro break on interrupt")
                    break
                audio_block = np.frombuffer(chunk, dtype=np.int16)
                await on_audio_block(audio_block, samplerate)
                chunk_count += 1
    except Exception as exc:
        print(f"Error during TTS streaming: {exc}")
    finally:
        timings["end"] = time.perf_counter()
        _log_tts_metrics(timings, chunk_count)


async def tts_kokoro_stream_async(text, interrupt_event):
    """Stream Kokoro-FastAPI TTS to the local speaker."""
    import sounddevice as sd

    stream = sd.OutputStream(
        samplerate=KOKORO_PCM_SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    stream.start()

    async def write_local(audio_block, samplerate):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, stream.write, audio_block)

    try:
        await _tts_kokoro_stream_chunks(text, interrupt_event, write_local)
    finally:
        stream.stop()
        stream.close()


async def tts_kokoro_to_track_async(text, interrupt_event, audio_track):
    """Stream Kokoro-FastAPI PCM into an outbound WebRTC audio track."""

    first_track_write = True

    async def write_track(audio_block, samplerate):
        nonlocal first_track_write
        if first_track_write:
            first_track_write = False
            monitor_log("tts provider=kokoro event=first_pcm_to_webrtc_track")
        await audio_track.write_pcm(audio_block, sample_rate=samplerate)

    await _tts_kokoro_stream_chunks(text, interrupt_event, write_track)


async def tts_kokoro_sequence_async(texts, speech_signals=None):
    interrupt_event = asyncio.Event()

    def on_signal(event):
        if str(event) == "SPEECH_START":
            interrupt_event.set()
        if str(event) == "SPEECH_END":
            interrupt_event.clear()
        print(f"INTERRUPT Signal received: {event}, interrupt: {interrupt_event}")

    if speech_signals:
        speech_signals.subscribe(on_signal)

    for text in texts:
        if interrupt_event.is_set():
            break
        await tts_kokoro_stream_async(text, interrupt_event)
        await asyncio.sleep(0.001)
    interrupt_event.clear()


def tts_kokoro_stream(text):
    return asyncio.run(tts_kokoro_stream_async(text, asyncio.Event()))


class KokoroFastApiTTSProvider:
    def __init__(self, mode: str = "local", audio_track: Optional[Any] = None):
        self.mode = mode
        self.audio_track = audio_track

    async def speak(self, text: str, interrupt_event: asyncio.Event) -> None:
        if self.mode == "local":
            await tts_kokoro_stream_async(text, interrupt_event)
            return
        if self.mode == "webrtc":
            if self.audio_track is None:
                raise ValueError("audio_track is required for KokoroFastApiTTSProvider(mode='webrtc')")
            await tts_kokoro_to_track_async(text, interrupt_event, self.audio_track)
            return
        raise ValueError(f"Unknown Kokoro TTS mode: {self.mode}")

    def clear_output(self) -> None:
        if self.mode == "webrtc" and self.audio_track is not None and hasattr(self.audio_track, "clear"):
            self.audio_track.clear()
