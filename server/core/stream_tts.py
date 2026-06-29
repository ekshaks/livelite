import asyncio
import threading
import time
from typing import Any, Optional, Protocol

from reactivex.disposable import CompositeDisposable, Disposable


class TTSProvider(Protocol):
    async def speak(self, text: str, interrupt_event: asyncio.Event) -> None:
        ...


class PlaybackState:
    """Thread-safe state exposed by TTS while local playback is active."""

    def __init__(self):
        self._playing = threading.Event()
        self._last_stopped_at = 0.0

    def set_playing(self, playing: bool):
        if playing:
            self._playing.set()
        else:
            self._last_stopped_at = time.monotonic()
            self._playing.clear()

    def is_playing(self) -> bool:
        return self._playing.is_set()

    def is_playing_or_recent(self, cooldown_seconds: float) -> bool:
        if self.is_playing():
            return True
        return (time.monotonic() - self._last_stopped_at) < cooldown_seconds


def _as_observable(stream_or_observable: Any):
    return getattr(stream_or_observable, "observable", stream_or_observable)


def _clear_queue(queue: asyncio.Queue):
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return


def tts_sink(
    provider: TTSProvider,
    interrupts: Optional[Any] = None,
    name: str = "tts",
    clear_queue_on_interrupt: bool = True,
    state: Optional[PlaybackState] = None,
):
    """Reusable TTS sink for `.to(...)`.

    Providers implement only `speak(text, interrupt_event)`.
    This function owns RxPY subscription, queueing, interrupts, and disposal.
    """

    def attach(text_observable):
        queue = asyncio.Queue()
        interrupt_event = asyncio.Event()
        disposable = CompositeDisposable()

        def on_signal(event):
            print(f"[interrupt] {name} received {event}")
            if str(event) == "SPEECH_START":
                interrupt_event.set()
                if clear_queue_on_interrupt:
                    _clear_queue(queue)
                clear_output = getattr(provider, "clear_output", None)
                if clear_output is not None:
                    clear_output()
            elif str(event) == "SPEECH_END":
                interrupt_event.clear()
            print(f"[interrupt] {name} interrupt_event set={interrupt_event.is_set()}")

        async def worker():
            while True:
                text = await queue.get()
                if text is None:
                    break
                if text and text.strip():
                    try:
                        if state is not None:
                            state.set_playing(True)
                        print(f"[interrupt] {name} speak start")
                        await provider.speak(text, interrupt_event)
                        print(f"[interrupt] {name} speak end interrupted={interrupt_event.is_set()}")
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        print(f"{name} provider error: {exc}")
                    finally:
                        if state is not None:
                            state.set_playing(False)

        task = asyncio.create_task(worker(), name=name)

        if interrupts is not None:
            interrupt_observable = _as_observable(interrupts)
            disposable.add(interrupt_observable.subscribe(on_signal))

        disposable.add(
            text_observable.subscribe(
                on_next=lambda text: queue.put_nowait(text),
                on_error=lambda error: print(f"{name} text stream error: {error}"),
                on_completed=lambda: queue.put_nowait(None),
            )
        )
        disposable.add(Disposable(lambda: task.cancel()))
        return disposable

    return attach


class KokoroTTSProvider:
    def __init__(self, mode: str = "local", audio_track: Optional[Any] = None):
        self.mode = mode
        self.audio_track = audio_track

    async def speak(self, text: str, interrupt_event: asyncio.Event) -> None:
        if self.mode == "local":
            from .tts import tts_kokoro_stream_async

            await tts_kokoro_stream_async(text, interrupt_event)
            return

        if self.mode == "webrtc":
            from .tts import tts_kokoro_to_track_async

            if self.audio_track is None:
                raise ValueError("audio_track is required for KokoroTTSProvider(mode='webrtc')")
            await tts_kokoro_to_track_async(text, interrupt_event, self.audio_track)
            return

        raise ValueError(f"Unknown Kokoro TTS mode: {self.mode}")

    def clear_output(self) -> None:
        if self.mode == "webrtc" and self.audio_track is not None and hasattr(self.audio_track, "clear"):
            self.audio_track.clear()


def kokoro_tts_sink(interrupts: Optional[Any] = None, name: str = "kokoro_tts"):
    return tts_sink(KokoroTTSProvider(), interrupts=interrupts, name=name)
