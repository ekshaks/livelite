import numpy as np
import av
from pydub import AudioSegment
from io import BytesIO
import librosa



def show_frame_properties(frame: av.audio.frame.AudioFrame):
    for attr in dir(frame):
        if not attr.startswith("_"):
            try:
                value = getattr(frame, attr)
                print(f"{attr}: {value}")
            except Exception as e:
                print(f"{attr}: <error: {e}>")

def estimate_pitch(y, sr):
    # Convert to float in [-1, 1]
    y = y.astype(np.float32) / 32768.0
    # librosa's YIN pitch detection



def is_active_speaker(chunk, sr, rms_thresh=0.05, centroid_thresh=2000, pitch_threshold=165, debug=False):
    """
    Quick heuristic for near vs far:
    - RMS loudness must be above rms_thresh
    - Spectral centroid must be above centroid_thresh (Hz)
    """
    # Normalize to float32 [-1, 1]
    y = chunk.astype(np.float32) / 32768.0  

    rms = np.mean(librosa.feature.rms(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    pitches = librosa.yin(y, fmin=50, fmax=300, sr=sr)
    f0 = np.nanmean(pitches)
    gender = "male" if f0 < pitch_threshold else "female"

    active = (rms > rms_thresh) #and (centroid > centroid_thresh)

    # Decision

    if active and debug:
        print(f"RMS: {rms}, Centroid: {centroid}, Rolloff: {rolloff}, ZCR: {zcr}, Gender: {gender}, F0: {f0}")
    return active

def convert_and_resample_frame(frame: av.audio.frame.AudioFrame, target_sample_rate: int = 16000, target_channels: int = 1, debug: bool = False) -> np.ndarray:
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
    
    # Resample to target rate
    if frame.rate != target_sample_rate:
        audio_segment = audio_segment.set_frame_rate(target_sample_rate)
        if debug: print(f"After resampling, frame rate: {audio_segment.frame_rate}")
    
    # Convert back to numpy array
    samples = np.frombuffer(audio_segment.raw_data, dtype=np.int16)
    if debug: print(f"After processing, samples shape: {samples.shape}")
    
    # Reshape to (1, N) format
    #samples = samples.reshape(1, -1)
    #print(f"Output samples shape: {samples.shape}")
    
    return samples

def generate_fake_audio_frame(duration_ms: int = 20, sample_rate: int = 48000, channels: int = 2) -> av.audio.frame.AudioFrame:
    """Generate a fake audio frame with sine wave data."""
    # Calculate number of samples for given duration
    num_samples = int(sample_rate * duration_ms / 1000)
    
    # Generate time array
    t = np.linspace(0, duration_ms/1000, num_samples, False)
    
    # Generate stereo sine wave (440Hz for left, 880Hz for right)
    left_channel = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    right_channel = (np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)
    
    # Combine channels
    samples = np.stack([left_channel, right_channel], axis=0) if channels == 2 else left_channel
    
    # Create PyAV AudioFrame
    frame = av.audio.frame.AudioFrame(
        format='s16',
        layout='stereo' if channels == 2 else 'mono',
        samples=num_samples
    )
    frame.rate = sample_rate
    
    # Copy samples to frame
    frame.planes[0].update(samples.tobytes())
    
    return frame

def test_convert_and_resample():
    # Generate fake frame (20ms, 48kHz, stereo, int16)
    input_frame = generate_fake_audio_frame(duration_ms=20, sample_rate=48000, channels=2)
    
    # Convert and resample
    output_samples = convert_and_resample_frame(input_frame, target_sample_rate=16000, target_channels=1)
    
    # Verify output
    print(f"Input frame: {input_frame.rate}Hz, {input_frame.layout.name}, samples: {input_frame.samples}")
    print(f"Output array shape: {output_samples.shape}, dtype: {output_samples.dtype}")
    
    # Verify properties
    expected_samples = int(20 * 16000 / 1000)  # 20ms at 16kHz = 320 samples
    assert output_samples.shape == (1, expected_samples), f"Expected shape (1, {expected_samples}), got {output_samples.shape}"
    assert output_samples.dtype == np.int16, "Output dtype should be int16"
    print("All tests passed!")

if __name__ == "__main__":
    test_convert_and_resample()