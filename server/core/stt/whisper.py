"""Local Whisper STT providers and their Stream DSL adapter."""

import asyncio
import time

import numpy as np

from ..events import TranscriptEvent
from ..logging_utils import monitor_time


# Cache Whisper models keyed by (mode, model_size, model_id, frozen kwargs).
# Loading a faster-whisper base int8 model takes ~1-3 s on a 1-vCPU box and
# was previously repeated on every new browser connect because the pipeline
# built a fresh WhisperSTT per session — the visible "hang on URL connect".
_WHISPER_MODEL_CACHE: dict = {}


def _cache_key(mode, model_size, model_id, kwargs):
    return (mode, model_size, model_id, tuple(sorted(kwargs.items())))


def get_faster_whisper_model(model_name: str = "base", compute_type: str = "int8", **kwargs):
    """Load a faster-whisper CTranslate2 model.

    Extra keyword arguments are forwarded to ``faster_whisper.WhisperModel`` so
    callers can tune the CPU-bound settings that matter on small servers
    (``cpu_threads``, ``num_workers``, ``device``, ``device_index``, ...).
    """
    from faster_whisper import WhisperModel

    print(f"Loading faster Whisper model... model={model_name} compute_type={compute_type} kwargs={kwargs}")
    return WhisperModel(model_name, compute_type=compute_type, **kwargs)


def get_whisper_model(mode="faster_whisper", model_size: str = "base", model_id=None, **kwargs):
    """Return a process-wide singleton Whisper model for the given config.

    Second and later callers with the same (mode, model_size, model_id, kwargs)
    share the same underlying model handle, so multiple WebRTC sessions do
    not each reload the weights.
    """
    key = _cache_key(mode, model_size, model_id, kwargs)
    cached = _WHISPER_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    if mode == "faster_whisper":
        model = get_faster_whisper_model(model_size, **kwargs)
    elif mode == "mlx":
        from .mlx import get_mlx_whisper_model

        model = get_mlx_whisper_model(model_size=model_size, model_id=model_id)
    else:
        raise ValueError(f"Unknown Whisper mode: {mode}")
    _WHISPER_MODEL_CACHE[key] = model
    return model


def warm_up(mode: str = "faster_whisper", model_size: str = "base", **kwargs) -> None:
    """Preload the Whisper model so the first session doesn't pay for it."""
    started = time.perf_counter()
    get_whisper_model(mode, model_size, **kwargs)
    print(f"Whisper ({mode}/{model_size}) warmed up in {time.perf_counter() - started:.2f} s")


def infer_faster_whisper(audio_data, model, language="en"):
    segments, _ = model.transcribe(audio_data, language=language)
    return " ".join(segment.text for segment in segments)


def infer_whisper(mode, audio_data, model, language="en"):
    if mode == "faster_whisper":
        return infer_faster_whisper(audio_data, model, language=language)
    if mode == "mlx":
        from .mlx import infer_mlx

        return infer_mlx(audio_data, model)
    raise ValueError(f"Unknown Whisper mode: {mode}")


class WhisperSTT:
    """Run a local Whisper model against one completed audio segment."""

    def __init__(self, mode="faster_whisper", model_size: str = "base", language: str = "en", **kwargs):
        self.mode = mode
        self.model_size = model_size
        self.kwargs = kwargs
        self.language = language
        self._model = None
        _ = self.model

    @property
    def model(self):
        if self._model is None:
            self._model = get_whisper_model(self.mode, self.model_size, **self.kwargs)
        return self._model

    def __call__(self, samples: np.ndarray) -> str:
        if len(samples) == 0:
            return ""
        start_time = time.perf_counter()
        audio_fp32 = samples.astype(np.float32) / 32768.0
        result = infer_whisper(self.mode, audio_fp32, self.model, language=self.language)
        monitor_time(
            "stt",
            "transcribe",
            time.perf_counter() - start_time,
            provider=self.mode,
            model=self.model_size,
        )
        return result


def whisper_stt(
    name: str = "whisper_stt",
    model_size: str = "tiny",
    mode: str = "mlx",
    debug_audio_dir=None,
    timeout_s: float = 20.0,
    on_status=None,
    **kwargs,
):
    """Create a final-transcript stage backed by a local Whisper provider."""

    if mode == "mlx":
        from .mlx import MlxPinnedWhisper
        from ..stream_dsl import _dump_stt_audio, async_map_stage

        mlx_stt = MlxPinnedWhisper(model_size=model_size, **kwargs)

        async def transcribe(segment):
            if debug_audio_dir:
                _dump_stt_audio(segment, debug_audio_dir)
            if on_status and mlx_stt.is_loading():
                on_status("loading", {"model_size": model_size})
            try:
                text = await asyncio.wait_for(mlx_stt.transcribe(segment), timeout=timeout_s)
            except asyncio.TimeoutError:
                if on_status:
                    on_status("error", {"model_size": model_size, "reason": f"STT timed out after {timeout_s:g}s"})
                return TranscriptEvent(text="", is_final=True)
            except Exception as exc:
                if on_status:
                    on_status("error", {"model_size": model_size, "reason": str(exc)})
                return TranscriptEvent(text="", is_final=True)
            if on_status:
                on_status("ready", {"model_size": model_size})
            return TranscriptEvent(text=text or "", is_final=True)

        return async_map_stage(transcribe, name=name, on_dispose=mlx_stt.shutdown)

    from ..stream_dsl import _dump_stt_audio, async_map_stage

    stt = WhisperSTT(mode=mode, model_size=model_size, **kwargs)

    async def transcribe(segment):
        # Whisper inference is blocking; run it off the event loop so audio
        # receive and WebRTC pacing keep running during transcription.
        if debug_audio_dir:
            _dump_stt_audio(segment, debug_audio_dir)
        text = await asyncio.to_thread(stt, segment)
        return TranscriptEvent(text=text or "", is_final=True)

    return async_map_stage(transcribe, name=name)
