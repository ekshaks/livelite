import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import threading
import time


DEBUG_MLX_STT = os.getenv("DEBUG_MLX_STT") == "1"


def _log(message):
    if DEBUG_MLX_STT:
        print(message)


def load_mlx_model(model_id):
    import mlx.core as mx
    from mlx_whisper import load_models as mlx_load_models

    _log(f"[mlx-stt] Loading mlx model {model_id} on thread={threading.get_ident()}")
    start = time.perf_counter()
    dtype = mx.float16
    model = mlx_load_models.load_model(path_or_hf_repo=model_id, dtype=dtype)
    _log(f"[mlx-stt] Loaded mlx model in {time.perf_counter() - start:.2f}s")
    return model


def get_mlx_whisper_model(model_size: str = "base", model_id=None):
    model_id = model_id or f"mlx-community/whisper-{model_size}-mlx"
    return load_mlx_model(model_id)


def mlx_feats(audio_data, n_mels: int = 80):
    import mlx.core as mx
    from mlx_whisper import audio as mlx_audio

    array = mx.array(audio_data)
    data = mlx_audio.pad_or_trim(array)
    mels = mlx_audio.log_mel_spectrogram(data, n_mels)
    mx.eval(mels)
    return mels


def infer_mlx(audio_data, model):
    import mlx.core as mx
    from mlx_whisper import decoding as mlx_decoding

    _log(f"[mlx-stt] infer_mlx start audio_shape={getattr(audio_data, 'shape', None)} thread={threading.get_ident()}")
    start = time.perf_counter()
    mels = mlx_feats(audio_data, model.dims.n_mels)[None].astype(mx.float16)
    _log(f"[mlx-stt] mels ready shape={mels.shape} elapsed={time.perf_counter() - start:.2f}s")
    options = mlx_decoding.DecodingOptions(language="en")
    result = mlx_decoding.decode(model, mels, options=options)
    _log(f"[mlx-stt] decode done elapsed={time.perf_counter() - start:.2f}s")
    return result[0].text


class MlxPinnedWhisper:
    """Run Whisper MLX model load and inference on one dedicated thread."""

    def __init__(self, model_size: str = "tiny", **kwargs):
        self.model_size = model_size
        self.kwargs = kwargs
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-whisper")
        _log(f"[mlx-stt] creating pinned worker from thread={threading.get_ident()}")
        self._stt_future = self.executor.submit(self._load)

    def _load(self):
        from .stt import WhisperSTT

        _log(f"[mlx-stt] worker load start thread={threading.get_ident()}")
        stt = WhisperSTT(mode="mlx", model_size=self.model_size, **self.kwargs)
        _log(f"[mlx-stt] worker load done thread={threading.get_ident()}")
        return stt

    def _infer(self, segment):
        _log(f"[mlx-stt] worker infer requested segment_shape={getattr(segment, 'shape', None)} thread={threading.get_ident()}")
        stt = self._stt_future.result()
        _log(f"[mlx-stt] worker infer model ready thread={threading.get_ident()}")
        result = stt(segment)
        _log(f"[mlx-stt] worker infer done text_len={len(result or '')}")
        return result

    async def transcribe(self, segment):
        _log(f"[mlx-stt] enqueue transcribe segment_shape={getattr(segment, 'shape', None)}")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._infer, segment)

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
