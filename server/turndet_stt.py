import numpy as np
import json

from .utils import rx_ops as ops, rx_Subject as Subject, rx_Observable, rx_interval
from faster_whisper import WhisperModel
import time
import torch
from typing import Tuple, Callable, Any, Optional
from functools import lru_cache



# Singleton instances
_WHISPER_MODEL = None
_VAD_MODEL = None
_VAD_UTILS = None

def get_whisper_model(model_name: str = "tiny", compute_type: str = "int8") -> WhisperModel:
    """Get or create a singleton instance of Whisper model."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        print("Loading Whisper model...")
        _WHISPER_MODEL = WhisperModel(model_name, compute_type=compute_type)
    return _WHISPER_MODEL

def get_vad_model() -> Tuple[Any, Any]:
    """Get or create singleton instances of VAD model and utils."""
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is None or _VAD_UTILS is None:
        print("Loading Silero VAD model...")
        _VAD_MODEL, _VAD_UTILS = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
    return _VAD_MODEL, _VAD_UTILS



def turn_detector_vad(silence_timeout: float = 1.0, poll_interval: float = 0.1, 
    min_speech_duration_ms: int = 100, min_silence_duration_ms: int = 2000,
    speech_pad_ms: int = 200, threshold: float = 0.4, RATE: int = 16000
) -> Tuple[Subject, rx_Observable]:
    """Voice Activity Detection using Silero VAD with singleton model loading."""

    vad_model, utils = get_vad_model()
    get_speech_timestamps = utils[0]
    
    input_subject = Subject()
    output_subject = Subject()
    
    def is_speech(samples: np.ndarray) -> bool:
        samples_fp32 = samples.astype(np.float32) / 32768.0
        speech_ts = get_speech_timestamps(
            samples_fp32,
            vad_model,
            sampling_rate=RATE,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms
        )
        return len(speech_ts) > 0
    
    buffer = []
    last_speech_time = [0]
    
    def process_chunk(chunk: np.ndarray):
        #print('processing chunk..', chunk.shape)
        if is_speech(chunk):
            buffer.append(chunk)
            last_speech_time[0] = time.time()
    
    def check_silence(_):
        #print('checking silence..')
        if buffer and (time.time() - last_speech_time[0]) >= silence_timeout:
            print('silence detected')
            try:
                segment = np.concatenate(buffer, axis=0)
                #print("check_silence: segment", segment.shape)
                output_subject.on_next(segment)
                buffer.clear()
            except ValueError as e:
                print(f"Error in segment concatenation: {e}")
    
    # input_subject.pipe(
    #     ops.map(lambda chunk: np.asarray(chunk, dtype=np.int16).flatten()),
    #     ops.filter(lambda x: len(x) > 0)
    # )
    input_subject.subscribe(
        on_next=process_chunk,
        on_error=lambda e: print(f"VAD processing error: {e}")
    )
    
    rx_interval(poll_interval).subscribe(check_silence)
    
    return input_subject, output_subject

class SpeechToText:
    """Handles speech-to-text conversion with singleton model management."""
    def __init__(self, model_name: str = "tiny", compute_type: str = "int8", language: str = 'en'):
        self.model_name = model_name
        self.compute_type = compute_type
        self.language = language
        self._model = None
    
    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            self._model = get_whisper_model(self.model_name, self.compute_type)
        return self._model
    
    def __call__(self, samples: np.ndarray) -> str:
        if len(samples) == 0:
            return ""
        
        audio_fp32 = samples.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(audio_fp32, language=self.language)
        return " ".join(segment.text for segment in segments)


def run_stt(audio_input):
    turn_input, turn_output = turn_detector_vad()
    stt = SpeechToText()
    
    # def print_transcription(segment):
    #     text = stt(segment)
    #     if text.strip():
    #         print(f"Transcription: {text}")
    #turn_output.subscribe(print_transcription) #turn_output -> print_transcription
    

    audio_input.subscribe(turn_input) #audio_input -> turn_input .... turn_output
    
    text_output = turn_output.pipe(
        ops.map(lambda segment: stt(segment))
    )

    return text_output




def test():
    """Example usage with proper resource cleanup."""
    from mic import AudioGenerator
    audio_gen = AudioGenerator()
    turn_input, turn_output = turn_detector_vad()
    stt = SpeechToText()
    
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