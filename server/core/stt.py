from pathlib import Path

import numpy as np
import json
import time

def get_faster_whisper_model(model_name: str = "base", compute_type: str = "int8") :
    from faster_whisper import WhisperModel
    print("Loading faster Whisper model...")
    return WhisperModel(model_name, compute_type=compute_type)

def load_mlx_model(model_id):
    import mlx.core as mx
    print('Loading mlx model')
    dtype = mx.float16
    from mlx_whisper import load_models as mlx_load_models
    model = mlx_load_models.load_model(path_or_hf_repo=model_id, dtype=dtype)
    return model

def get_whisper_model(mode = 'faster_whisper', model_size: str = "base", model_id=None, **kwargs) :
    if mode == 'faster_whisper':
        _WHISPER_MODEL = get_faster_whisper_model(model_size, **kwargs)
    elif mode == 'mlx':
        model_id = model_id or f"mlx-community/whisper-{model_size}-mlx"
        _WHISPER_MODEL = load_mlx_model(model_id)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return _WHISPER_MODEL


def infer_faster_whisper(audio_data, model):
    segments, _ = model.transcribe(audio_data, language='en')
    return " ".join(segment.text for segment in segments)

def mlx_feats(audio_data,n_mels: int = 80):
    import mlx.core as mx
    from mlx_whisper import audio as mlx_audio, decoding as mlx_decoding, load_models as mlx_load_models
    array = mx.array(audio_data)
    data = mlx_audio.pad_or_trim(array) 
    mels = mlx_audio.log_mel_spectrogram(data, n_mels)
    mx.eval(mels)
    return mels

def infer_mlx(audio_data, model):
    import mlx.core as mx
    from mlx_whisper import audio as mlx_audio, decoding as mlx_decoding, load_models as mlx_load_models

    tokens = mx.array(
            [
                50364,
                1396,
                264,
                665,
                5133,
                23109,
                25462,
                264,
                6582,
                293,
                750,
                632,
                42841,
                292,
                370,
                938,
                294,
                4054,
                293,
                12653,
                356,
                50620,
                50620,
                23563,
                322,
                3312,
                13,
                50680,
            ],
            mx.int32,
        )[None]
    mels = mlx_feats(audio_data, model.dims.n_mels)[None].astype(mx.float16)
    #logits = mlx_model_forward(model, mels, tokens)
    options = mlx_decoding.DecodingOptions(language='en')
    result = mlx_decoding.decode(model, mels, options=options)
    return result[0].text

def infer_whisper(mode, audio_data, model):
    if mode == 'faster_whisper':
        return infer_faster_whisper(audio_data, model)
    elif mode == 'mlx':
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



def run_stt(audio_input, model_size='tiny'):
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
    audio_fname = Path(__file__).parent.parent.parent / "data" / "kjfk_1m.wav"
    assert audio_fname.exists(), f"Audio file not found: {audio_fname}"
    model = load_mlx_model("mlx-community/whisper-small-mlx")
    audio_data = load_wav_to_array(audio_fname)
    res = infer_mlx(audio_data, model)
    print(res.text)

if __name__ == "__main__":
    test_stt()

