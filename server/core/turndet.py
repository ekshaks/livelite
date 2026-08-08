import asyncio
import os

import numpy as np
import reactivex
from reactivex.disposable import CompositeDisposable, Disposable
from reactivex.scheduler.eventloop import AsyncIOScheduler
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from .events import SpeechEvent

# Singleton instances
_VAD_MODEL = None
_VAD_UTILS = None


def _load_silero_torch():
    """Load Silero VAD via torch.hub. Same shape it has always returned."""
    import torch  # imported lazily so the onnx backend can skip the torch dep

    return torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)


def _load_silero_onnx():
    """Load Silero VAD via the silero-vad PyPI package with the ONNX runtime.

    Returns (model, utils) where utils[0] is get_speech_timestamps so the
    call sites do not need to know which backend they are talking to.
    """
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad(onnx=True)
    # Match the (model, utils_tuple) shape torch.hub returns; only utils[0]
    # is consumed by _build_is_speech.
    return model, (get_speech_timestamps,)


def get_vad_model() -> Tuple[Any, Any]:
    """Return (model, utils) for the Silero VAD backend.

    Backend selection: env ``SILERO_BACKEND`` = ``torch`` (default) | ``onnx``.
    ONNX drops the ~1.5 GB torch install and shaves ~200-400 MB RAM, which
    matters on a 2 GB AWS server. Desktop keeps the torch backend.
    """
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is None or _VAD_UTILS is None:
        backend = os.environ.get("SILERO_BACKEND", "torch").lower()
        print(f"Loading Silero VAD model (backend={backend})...")
        if backend == "onnx":
            _VAD_MODEL, _VAD_UTILS = _load_silero_onnx()
        elif backend == "torch":
            _VAD_MODEL, _VAD_UTILS = _load_silero_torch()
        else:
            raise ValueError(f"Unknown SILERO_BACKEND: {backend!r} (expected 'torch' or 'onnx')")
    return _VAD_MODEL, _VAD_UTILS



@dataclass(frozen=True)
class VadEmission:
    segment: Optional[np.ndarray] = None
    signal: Optional[SpeechEvent] = None


def _build_is_speech(
    threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
    rate: int,
) -> Callable[[np.ndarray], bool]:
    vad_model, utils = get_vad_model()
    get_speech_timestamps = utils[0]

    def is_speech(samples: np.ndarray) -> bool:
        samples_fp32 = samples.astype(np.float32) / 32768.0
        speech_ts = get_speech_timestamps(
            samples_fp32,
            vad_model,
            sampling_rate=rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        return len(speech_ts) > 0

    return is_speech


def turn_detector_vad(
    audio_observable,
    silence_timeout: float = 1.0,
    poll_interval: float = 0.1,
    min_speech_duration_ms: int = 100, min_silence_duration_ms: int = 2000,
    speech_pad_ms: int = 200, threshold: float = 0.4, RATE: int = 16000,
    is_speech_fn: Optional[Callable[[np.ndarray], bool]] = None,
):
    """Return a cold VAD event observable with subscription-owned resources."""

    is_speech = is_speech_fn or _build_is_speech(
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        rate=RATE,
    )

    def subscribe(observer, scheduler=None):
        # The silence-poll timer runs on the asyncio event loop, so every
        # callback below (audio on_next, timer tick, dispose) is serialized on
        # one loop and no locking is needed. Audio sources must also emit on
        # this loop.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError(
                "turn_detector_vad must be subscribed from a running asyncio "
                "event loop; its poll timer is scheduled on that loop."
            ) from None

        buffer = []
        last_speech_time = [0.0]
        disposed = [False]

        def process_chunk(chunk: np.ndarray):
            if disposed[0]:
                return
            if is_speech(chunk):
                if not buffer:
                    print("[interrupt] VAD SPEECH_START")
                    observer.on_next(VadEmission(signal=SpeechEvent.SPEECH_START))
                buffer.append(chunk)
                last_speech_time[0] = time.monotonic()

        def emit_segment_if_ready(force: bool = False):
            if disposed[0] or not buffer:
                return
            if not force and (time.monotonic() - last_speech_time[0]) < silence_timeout:
                return
            print("[interrupt] VAD SPEECH_END")
            observer.on_next(VadEmission(signal=SpeechEvent.SPEECH_END))
            segment = np.concatenate(buffer, axis=0)
            buffer.clear()
            observer.on_next(VadEmission(segment=segment))

        def on_completed():
            emit_segment_if_ready(force=True)
            observer.on_completed()

        source_sub = audio_observable.subscribe(
            on_next=process_chunk,
            on_error=observer.on_error,
            on_completed=on_completed,
        )
        timer_sub = reactivex.interval(
            poll_interval, scheduler=AsyncIOScheduler(loop)
        ).subscribe(lambda _: emit_segment_if_ready())

        def dispose():
            disposed[0] = True

        return CompositeDisposable(source_sub, timer_sub, Disposable(dispose))

    return reactivex.create(subscribe)
