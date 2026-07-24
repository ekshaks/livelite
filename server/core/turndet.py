import numpy as np
import reactivex
from reactivex import operators as ops
from reactivex.disposable import CompositeDisposable, Disposable
import time
import torch
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from .events import SpeechEvent

# Singleton instances
_VAD_MODEL = None
_VAD_UTILS = None


def get_vad_model() -> Tuple[Any, Any]:
    """Get or create singleton instances of VAD model and utils."""
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is None or _VAD_UTILS is None:
        print("Loading Silero VAD model...")
        _VAD_MODEL, _VAD_UTILS = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
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
        buffer = []
        last_speech_time = [0.0]
        lock = threading.RLock()
        disposed = threading.Event()

        def process_chunk(chunk: np.ndarray):
            if disposed.is_set():
                return
            with lock:
                if is_speech(chunk):
                    if not buffer:
                        print("[interrupt] VAD SPEECH_START")
                        observer.on_next(VadEmission(signal=SpeechEvent.SPEECH_START))
                    buffer.append(chunk)
                    last_speech_time[0] = time.monotonic()

        def emit_segment_if_ready(force: bool = False):
            if disposed.is_set():
                return
            with lock:
                if not buffer:
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
        timer_sub = reactivex.interval(poll_interval).subscribe(
            lambda _: emit_segment_if_ready()
        )

        def dispose():
            disposed.set()

        return CompositeDisposable(source_sub, timer_sub, Disposable(dispose))

    return reactivex.create(subscribe)


def test():
    """Example usage with proper resource cleanup."""
    from mic import AudioGenerator
    from .stt.whisper import WhisperSTT
    audio_gen = AudioGenerator()
    from reactivex.subject import Subject

    turn_input = Subject()
    events = turn_detector_vad(turn_input)
    turn_output = events.pipe(
        ops.filter(lambda event: event.segment is not None),
        ops.map(lambda event: event.segment),
    )
    stt = WhisperSTT()
    
    def print_transcription(segment):
        text = stt(segment)
        if text.strip():
            print(f"Transcription: {text}")
    
    try:
        audio_stream = audio_gen()
        audio_stream.subscribe(turn_input)
        turn_output.subscribe(print_transcription)
        
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        audio_gen.close()

if __name__ == "__main__":
    test()
