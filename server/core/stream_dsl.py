import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from reactivex import operators as ops
from reactivex.disposable import CompositeDisposable, Disposable


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

    def __init__(self, observable, name: Optional[str] = None, on_subscribe: Optional[Callable[[], Any]] = None):
        self.observable = observable
        self.name = name
        self._on_subscribe = on_subscribe

    @classmethod
    def source(cls, observable, name: Optional[str] = None):
        return cls(observable, name=name)

    def __or__(self, stage):
        return stage(self)

    def to(self, sink, name: Optional[str] = None, subs: Optional[SubGroup] = None) -> Sub:
        # `.to(...)` is the materialization boundary: upstream wiring starts here.
        upstream_disposable = self._on_subscribe() if self._on_subscribe else None
        sink_disposable = sink(self.observable)
        disposable = CompositeDisposable()
        if upstream_disposable is not None:
            disposable.add(upstream_disposable)
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
            on_subscribe=self._on_subscribe,
        )

    def latest(self, name: Optional[str] = None):
        # Side input helper. It samples the newest value when another stage asks.
        latest_value = LatestValue(name=name or self.name)
        upstream_sub = self.to(latest_value.sink(), name=f"{name or self.name}.latest")
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

        turn_input, segments, signals = turn_detector_vad(**kwargs)
        upstream_sub = None

        def ensure_connected():
            # Connect audio to VAD once, even if both segments and signals are consumed.
            nonlocal upstream_sub
            if upstream_sub is None:
                upstream_sub = stream.observable.subscribe(turn_input)
            return upstream_sub

        return MultiOutput(
            segments=Stream(segments, name=f"{name}.segments", on_subscribe=ensure_connected),
            signals=Stream(signals, name=f"{name}.signals", on_subscribe=ensure_connected),
        )

    return apply


def whisper_stt(name: str = "whisper_stt", model_size: str = "tiny", mode: str = "mlx", **kwargs):
    from .stt import WhisperSTT

    stt = WhisperSTT(mode=mode, model_size=model_size, **kwargs)

    def apply(stream: Stream):
        return stream.pipe(ops.map(lambda segment: stt(segment)), name=name)

    return apply


def async_map_stage(func, name: str = "async_stage", concurrency: str = "serial"):
    """Serial async transform stage. Other concurrency policies are future work."""

    if concurrency != "serial":
        raise NotImplementedError("Only serial async stages are implemented in the MVP")

    def apply(stream: Stream):
        from reactivex.subject import Subject

        subject = Subject()
        task = None
        source_sub = None

        async def worker(queue: asyncio.Queue):
            # One worker keeps async transforms ordered for the MVP.
            while True:
                item = await queue.get()
                if item is None:
                    subject.on_completed()
                    break
                try:
                    result = await func(item)
                    subject.on_next(result)
                except Exception as exc:
                    subject.on_error(exc)

        def ensure_connected():
            nonlocal task, source_sub
            if source_sub is not None:
                return CompositeDisposable(source_sub, Disposable(lambda: task.cancel() if task else None))

            loop = asyncio.get_running_loop()
            queue = asyncio.Queue()
            task = loop.create_task(worker(queue))

            source_sub = stream.observable.subscribe(
                on_next=lambda item: queue.put_nowait(item),
                on_error=subject.on_error,
                on_completed=lambda: queue.put_nowait(None),
            )
            return CompositeDisposable(source_sub, Disposable(lambda: task.cancel()))

        return Stream(subject, name=name, on_subscribe=ensure_connected)

    return apply


def print_sink(prefix: str = ""):
    def attach(observable):
        return observable.subscribe(lambda item: print(f"{prefix}{item}"))
    return attach


def client_text_sink(data_channels: Dict[str, Any], loop, role: str, channel: str = "server_text"):
    def attach(observable):
        def on_next(text):
            # Data channel sends must hop back to the WebRTC event loop.
            print(f"sending to client: {text}")
            try:
                data_channel = data_channels.get(channel)
                if data_channel and data_channel.readyState == "open":
                    data = json.dumps({"role": role, "content": text})
                    loop.call_soon_threadsafe(lambda: data_channel.send(data))
            except Exception as exc:
                print(f"Error sending text to client: {exc}")

        def on_error(error):
            print(f"client_text_sink error: {error}")

        return observable.subscribe(on_next=on_next, on_error=on_error)

    return attach
