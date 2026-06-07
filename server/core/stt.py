from pathlib import Path

import numpy as np
import json
import time

def get_faster_whisper_model(model_name: str = "base", compute_type: str = "int8") :
    from faster_whisper import WhisperModel
    print("Loading faster Whisper model...")
    return WhisperModel(model_name, compute_type=compute_type)

def get_whisper_model(mode = 'faster_whisper', model_size: str = "base", model_id=None, **kwargs) :
    if mode == 'faster_whisper':
        _WHISPER_MODEL = get_faster_whisper_model(model_size, **kwargs)
    elif mode == 'mlx':
        from .stt_mlx import get_mlx_whisper_model

        _WHISPER_MODEL = get_mlx_whisper_model(model_size=model_size, model_id=model_id)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return _WHISPER_MODEL


def infer_faster_whisper(audio_data, model):
    segments, _ = model.transcribe(audio_data, language='en')
    return " ".join(segment.text for segment in segments)

def infer_whisper(mode, audio_data, model):
    if mode == 'faster_whisper':
        return infer_faster_whisper(audio_data, model)
    elif mode == 'mlx':
        from .stt_mlx import infer_mlx

        return infer_mlx(audio_data, model)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    

class WhisperSTT:
    """Handles speech-to-text conversion with singleton model management."""
    def __init__(self, mode='faster_whisper', model_size: str = "base", language: str = 'en', **kwargs):
        self.mode = mode
        self.model_size = model_size
        self.kwargs = kwargs
        self.language = language
        self._model = None
        _ = self.model
    
    @property
    def model(self) :
        if self._model is None:
            self._model = get_whisper_model(self.mode, self.model_size, **self.kwargs)
        return self._model
    
    def __call__(self, samples: np.ndarray) -> str:
        if len(samples) == 0:
            return ""
        start_time = time.perf_counter()

        audio_fp32 = samples.astype(np.float32) / 32768.0
        res = infer_whisper(self.mode, audio_fp32, self.model)
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"STT time: {elapsed_time} seconds")
        return res



def run_stt_rx(audio_input, model_size='tiny'):
    from .utils import rx_ops as ops, rx_Subject as Subject, rx_Observable, rx_interval
    from .turndet import turn_detector_vad

    turn_input, turn_output, speech_signals = turn_detector_vad()
    stt = WhisperSTT(mode='mlx', model_size=model_size)
    
    # def print_transcription(segment):
    #     text = stt(segment)
    #     if text.strip():
    #         print(f"Transcription: {text}")
    #turn_output.subscribe(print_transcription) #turn_output -> print_transcription
    

    audio_input.subscribe(turn_input) #audio_input -> turn_input .... turn_output
    
    text_output = turn_output.pipe(
        ops.map(lambda segment: stt(segment))
    )

    return text_output, speech_signals


def test_stt():
    from audio_utils import load_wav_to_array
    from .stt_mlx import load_mlx_model, infer_mlx

    audio_fname = Path(__file__).parent.parent.parent / "data" / "kjfk_1m.wav"
    assert audio_fname.exists(), f"Audio file not found: {audio_fname}"
    model = load_mlx_model("mlx-community/whisper-small-mlx")
    audio_data = load_wav_to_array(audio_fname)
    res = infer_mlx(audio_data, model)
    print(res.text)

if __name__ == "__main__":
    test_stt()
