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


def _find_silero_onnx_model():
    """Locate the Silero VAD ONNX model file without importing torch.

    The ``silero-vad`` PyPI package bundles ``data/silero_vad.onnx``, but its
    ``__init__`` imports torch — so we find the file via ``find_spec`` (which
    does not execute the package) instead of importing it. Install with
    ``pip install --no-deps silero-vad onnxruntime`` to skip torch entirely,
    or point ``SILERO_ONNX_MODEL_PATH`` at a standalone model file.
    """
    import importlib.util
    from pathlib import Path

    env_path = os.environ.get("SILERO_ONNX_MODEL_PATH")
    if env_path:
        path = Path(env_path)
        if not path.exists():
            raise FileNotFoundError(f"SILERO_ONNX_MODEL_PATH does not exist: {path}")
        return path
    spec = importlib.util.find_spec("silero_vad")
    if spec is not None and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            candidate = Path(location) / "data" / "silero_vad.onnx"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        "Silero VAD ONNX model not found. Either `pip install --no-deps "
        "silero-vad onnxruntime` or set SILERO_ONNX_MODEL_PATH to the "
        "silero_vad.onnx file."
    )


class _NumpyOnnxVad:
    """Silero VAD on onnxruntime with numpy I/O — no torch dependency.

    Mirrors the reference OnnxWrapper: 512-sample windows at 16 kHz (256 at
    8 kHz) with a 64-sample rolling context and a (2, 1, 128) recurrent state.
    """

    def __init__(self, model_path):
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"], sess_options=opts
        )

    def speech_probs(self, audio_fp32, rate):
        """Return the per-window speech probability list for one audio clip."""
        window = 512 if rate == 16000 else 256
        context_size = 64 if rate == 16000 else 32
        audio = np.asarray(audio_fp32, dtype=np.float32).reshape(-1)
        if len(audio) % window:
            audio = np.pad(audio, (0, window - len(audio) % window))
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, context_size), dtype=np.float32)
        sr = np.array(rate, dtype=np.int64)
        probs = []
        for start in range(0, len(audio), window):
            chunk = audio[start : start + window][None, :]
            x = np.concatenate([context, chunk], axis=1)
            out, state = self.session.run(None, {"input": x, "state": state, "sr": sr})
            context = x[:, -context_size:]
            probs.append(float(out[0, 0]))
        return probs


def _onnx_get_speech_timestamps(
    audio,
    model,
    sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=100,
    speech_pad_ms=30,
    **_,
):
    """Numpy segmenter over :class:`_NumpyOnnxVad` probabilities.

    Same call signature as silero's ``get_speech_timestamps`` (as used by
    ``_build_is_speech``); returns a list of ``{"start", "end"}`` sample
    ranges. ``speech_pad_ms`` is accepted for compatibility but padding is
    irrelevant to the boolean is-speech decision this pipeline makes.
    """
    window = 512 if sampling_rate == 16000 else 256
    probs = model.speech_probs(audio, sampling_rate)
    min_speech_frames = max(1, int(min_speech_duration_ms * sampling_rate / 1000 / window))
    min_silence_frames = max(1, int(min_silence_duration_ms * sampling_rate / 1000 / window))
    neg_threshold = max(threshold - 0.15, 0.01)

    segments = []
    current_start = None
    silence_run = 0
    for index, prob in enumerate(probs):
        if prob >= threshold:
            if current_start is None:
                current_start = index
            silence_run = 0
        elif current_start is not None and prob < neg_threshold:
            silence_run += 1
            if silence_run >= min_silence_frames:
                end = index - silence_run + 1
                if end - current_start >= min_speech_frames:
                    segments.append({"start": current_start * window, "end": end * window})
                current_start = None
                silence_run = 0
    if current_start is not None and len(probs) - current_start >= min_speech_frames:
        segments.append({"start": current_start * window, "end": len(probs) * window})
    return segments


def _load_silero_onnx():
    """Load Silero VAD on onnxruntime + numpy, with no torch dependency.

    Returns (model, utils) where utils[0] is a get_speech_timestamps-shaped
    callable so call sites do not need to know which backend they are on.
    """
    model = _NumpyOnnxVad(_find_silero_onnx_model())
    return model, (_onnx_get_speech_timestamps,)


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
