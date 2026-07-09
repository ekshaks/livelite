import asyncio
import json
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import reactivex
from reactivex import operators as ops
from reactivex.disposable import CompositeDisposable

from .events import TranscriptEvent
from .logging_utils import monitor_log


class Sub:
    """Disposable returned by `.to(...)`."""

    def __init__(self, disposable: Any, name: Optional[str] = None):
        self.disposable = disposable
        self.name = name
        self.disposed = False

    def dispose(self):
        if self.disposed:
            return
        self.disposed = True
        if self.disposable is not None:
            self.disposable.dispose()


class SubGroup:
    """Session-level owner for all active stream subscriptions."""

    def __init__(self):
        self._subs = []

    def add(self, sub: Sub) -> Sub:
        if sub not in self._subs:
            self._subs.append(sub)
        return sub

    def dispose(self):
        for sub in list(self._subs):
            sub.dispose()
        self._subs.clear()


class Stream:
    """Small wrapper that adds `|` and `.to(...)` on top of an RxPY observable."""

    def __init__(self, observable, name: Optional[str] = None):
        self.observable = observable
        self.name = name

    @classmethod
    def source(cls, observable, name: Optional[str] = None):
        return cls(observable, name=name)

    def __or__(self, stage):
        return stage(self)

    def to(self, sink, name: Optional[str] = None, subs: Optional[SubGroup] = None) -> Sub:
        sink_disposable = sink(self.observable)
        disposable = CompositeDisposable()
        if sink_disposable is not None:
            disposable.add(sink_disposable)
        sub = Sub(disposable, name=name or self.name)
        if subs is not None:
            subs.add(sub)
        return sub

    def pipe(self, *pipe_ops, name: Optional[str] = None):
        return Stream(
            self.observable.pipe(*pipe_ops),
            name=name,
        )

    def latest(self, name: Optional[str] = None, subs: Optional[SubGroup] = None):
        # Side input helper. It samples the newest value when another stage asks.
        latest_value = LatestValue(name=name or self.name)
        upstream_sub = self.to(latest_value.sink(), name=f"{name or self.name}.latest", subs=subs)
        latest_value._sub = upstream_sub
        return latest_value


class MultiOutput:
    """Named outputs for stages like VAD that produce data plus side signals."""

    def __init__(self, **streams: Stream):
        self._streams = streams

    def __getattr__(self, name: str):
        try:
            return self._streams[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass
class LatestValue:
    value: Any = None
    name: Optional[str] = None
    _sub: Optional[Sub] = None

    def get(self):
        return self.value

    def sink(self):
        def attach(observable):
            return observable.subscribe(lambda item: setattr(self, "value", item))
        return attach

    def dispose(self):
        if self._sub is not None:
            self._sub.dispose()


def turn_detector(name: str = "turn_detector", **kwargs):
    def apply(stream: Stream):
        from .turndet import turn_detector_vad

        events = turn_detector_vad(stream.observable, **kwargs).pipe(ops.share())

        return MultiOutput(
            segments=Stream(
                events.pipe(
                    ops.filter(lambda event: event.segment is not None),
                    ops.map(lambda event: event.segment),
                ),
                name=f"{name}.segments",
            ),
            signals=Stream(
                events.pipe(
                    ops.filter(lambda event: event.signal is not None),
                    ops.map(lambda event: event.signal),
                ),
                name=f"{name}.signals",
            ),
        )

    return apply


def _dump_stt_audio(segment, directory: str, rate: int = 16000) -> Path:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"stt-{time.time_ns()}.wav"
    samples = np.asarray(segment)
    if samples.dtype != np.int16:
        samples = np.clip(samples, -1.0, 1.0)
        samples = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.reshape(-1).tobytes())
    print(f"[stt-debug] wrote {path} samples={samples.size}")
    return path


def whisper_stt(
    name: str = "whisper_stt",
    model_size: str = "tiny",
    mode: str = "mlx",
    debug_audio_dir: Optional[str] = None,
    **kwargs,
):
    """Create an STT stage. Use `model_size="medium"` for better letter recognition."""

    if mode == "mlx":
        from .stt_mlx import MlxPinnedWhisper

        mlx_stt = MlxPinnedWhisper(model_size=model_size, **kwargs)

        async def transcribe(segment):
            if debug_audio_dir:
                _dump_stt_audio(segment, debug_audio_dir)
            text = await mlx_stt.transcribe(segment)
            return TranscriptEvent(text=text or "", is_final=True)

        return async_map_stage(transcribe, name=name, on_dispose=mlx_stt.shutdown)

    from .stt import WhisperSTT

    stt = WhisperSTT(mode=mode, model_size=model_size, **kwargs)

    def apply(stream: Stream):
        def transcribe(segment):
            if debug_audio_dir:
                _dump_stt_audio(segment, debug_audio_dir)
            return TranscriptEvent(
                text=stt(segment) or "",
                is_final=True,
            )

        return stream.pipe(
            ops.map(transcribe),
            name=name,
        )

    return apply


def drop_while(predicate: Callable[[], bool], name: str = "drop_while"):
    """Drop stream items while a zero-argument predicate is true."""

    def apply(stream: Stream):
        return stream.pipe(ops.filter(lambda item: not predicate()), name=name)

    return apply


def filter_items(predicate: Callable[[Any], bool], name: str = "filter_items"):
    """Keep stream items that satisfy an item-level predicate."""

    def apply(stream: Stream):
        return stream.pipe(ops.filter(predicate), name=name)

    return apply


def map_items(mapper: Callable[[Any], Any], name: str = "map_items"):
    """Map each stream item with a synchronous item-level function."""

    def apply(stream: Stream):
        return stream.pipe(ops.map(mapper), name=name)

    return apply


def map_filter_items(
    map_fn: Callable[[Any], Any],
    filter_fn: Callable[[Any], bool],
    name: str = "map_filter_items",
):
    """Map each item, then keep mapped values that satisfy ``filter_fn``."""

    def apply(stream: Stream):
        return stream.pipe(
            ops.map(map_fn),
            ops.filter(filter_fn),
            name=name,
        )

    return apply


def final_transcript_text(name: str = "final_transcript_text"):
    """Select final, non-empty text from a TranscriptEvent stream."""

    def apply(stream: Stream):
        return stream.pipe(
            ops.filter(lambda event: event.is_final),
            ops.map(lambda event: event.text),
            ops.filter(lambda text: bool(text and text.strip())),
            name=name,
        )

    return apply


def async_map_stage(func, name: str = "async_stage", concurrency: str = "serial", on_dispose: Optional[Callable[[], None]] = None):
    """Create an RxPY stage that awaits one coroutine call per input item.

    A normal Rx ``map`` is synchronous: mapping an item with an ``async``
    function would emit a coroutine object rather than await it. This adapter
    converts each coroutine call into an inner Observable with
    ``defer``/``from_future`` and lets RxPY control how those inner operations
    are flattened:

    - ``serial`` processes inputs in order, one at a time.
    - ``parallel`` allows operations to overlap.
    - ``latest`` cancels stale work when a newer item arrives.
    - ``drop`` ignores new items while one operation is active.

    The current asyncio loop is captured when the stage is built. Inputs may
    arrive from another thread (for example, the VAD timer), so coroutine work
    is scheduled onto that loop safely. The output is shared so multiple sinks
    reuse one async execution rather than invoking ``func`` once per sink.
    Disposing the final subscription cancels active futures; ``on_dispose``
    optionally releases provider resources such as a model executor.
    """

    supported = {"serial", "parallel", "latest", "drop"}
    if concurrency not in supported:
        raise ValueError(f"Unknown concurrency policy: {concurrency}")

    def apply(stream: Stream):
        loop = asyncio.get_running_loop()

        def async_observable(item):
            def future_factory(_scheduler=None):
                coroutine = func(item)
                try:
                    if asyncio.get_running_loop() is loop:
                        future = loop.create_task(coroutine)
                    else:
                        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
                except RuntimeError:
                    future = asyncio.run_coroutine_threadsafe(coroutine, loop)
                return reactivex.from_future(future)

            return reactivex.defer(future_factory)

        inner_streams = stream.observable.pipe(ops.map(async_observable))
        if concurrency == "serial":
            output = inner_streams.pipe(ops.merge(max_concurrent=1))
        elif concurrency == "parallel":
            output = inner_streams.pipe(ops.flat_map())
        elif concurrency == "latest":
            output = inner_streams.pipe(ops.switch_latest())
        else:
            output = inner_streams.pipe(ops.exclusive())

        if on_dispose is not None:
            output = output.pipe(ops.finally_action(on_dispose))

        return Stream(output.pipe(ops.share()), name=name)

    return apply


def print_sink(prefix: str = ""):
    def attach(observable):
        return observable.subscribe(lambda item: print(f"{prefix}{item}"))
    return attach


def client_message_sink(data_channels: Dict[str, Any], loop, channel: str = "server_text"):
    def attach(observable):
        def on_next(message):
            # Data channel sends must hop back to the WebRTC event loop.
            message_type = type(message).__name__
            monitor_log(f"sending {message_type} to client")
            try:
                data_channel = data_channels.get(channel)
                if data_channel and data_channel.readyState == "open":
                    if not hasattr(message, "to_client_dict"):
                        raise TypeError(
                            f"{message_type} must implement to_client_dict()"
                        )
                    data = json.dumps(message.to_client_dict())
                    loop.call_soon_threadsafe(lambda: data_channel.send(data))
            except Exception as exc:
                print(f"Error sending client message: {exc}")

        def on_error(error):
            print(f"client_message_sink error: {error}")

        return observable.subscribe(on_next=on_next, on_error=on_error)

    return attach
