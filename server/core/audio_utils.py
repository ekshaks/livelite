import numpy as np
import av
from pydub import AudioSegment


def load_wav_to_array(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load wav file and return 1-D float32 mono array resampled to 16kHz."""
    import librosa
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32")
    # If stereo, take mean
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    # Resample if needed
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return data.astype(np.float32)


def convert_and_resample_frame(frame: av.audio.frame.AudioFrame, target_sample_rate: int = 16000, target_channels: int = 1, debug: bool = False) -> np.ndarray:
    # ISC: R1 R2 T1 T2 I_SAFE I_AUTH I_LIVE I_FRESH I_ATOMIC
    """Convert stereo 48kHz frame to mono 16kHz and return as NumPy array in (1, N) format."""
    # Convert PyAV AudioFrame to pydub AudioSegment
    samples = frame.to_ndarray()
    if debug: print(f"Input samples shape: {samples.shape}")
    
    # Convert samples to bytes (int16, stereo)
    audio_bytes = samples.tobytes()
    
    # Create AudioSegment
    audio_segment = AudioSegment(
        data=audio_bytes,
        sample_width=2,  # int16
        frame_rate=frame.rate,
        channels=2 if frame.layout.name == 'stereo' else 1
    )
    
    # Convert to mono if needed
    if frame.layout.name == 'stereo' and target_channels == 1:
        audio_segment = audio_segment.set_channels(1)
        if debug: print(f"After mono conversion, channels: {audio_segment.channels}")
    
    # Convert back to mono PCM16, then use the shared resampler.
    samples = np.frombuffer(audio_segment.raw_data, dtype=np.int16)
    samples = resample_pcm16_mono(samples, frame.rate, target_sample_rate)
    if debug and frame.rate != target_sample_rate: print(f"After resampling, frame rate: {target_sample_rate}")
    if debug: print(f"After processing, samples shape: {samples.shape}")
    
    # Reshape to (1, N) format
    #samples = samples.reshape(1, -1)
    #print(f"Output samples shape: {samples.shape}")
    
    return samples


def resample_pcm16_mono(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    # ISC: R1 R2 T1 T2 I_SAFE I_AUTH I_LIVE I_FRESH I_ATOMIC
    """Resample flat mono signed-16-bit PCM without changing its representation."""
    if samples.dtype != np.int16 or samples.ndim != 1:
        raise TypeError("samples must be a flat int16 array")
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be positive")
    if src_rate == dst_rate:
        return samples
    segment = AudioSegment(
        data=samples.tobytes(),
        sample_width=2,
        frame_rate=src_rate,
        channels=1,
    ).set_frame_rate(dst_rate)
    return np.frombuffer(segment.raw_data, dtype=np.int16)
