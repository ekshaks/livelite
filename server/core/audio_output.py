"""Transport-neutral destination for generated assistant audio."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """Interleaved signed-16-bit PCM samples and their playback metadata.

    ``samples`` is a flat array. For multichannel audio, samples are interleaved
    frame by frame and the array length must be divisible by ``channels``.
    """

    samples: npt.NDArray[np.int16]
    sample_rate: int
    channels: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.samples, np.ndarray):
            raise TypeError("AudioChunk samples must be a numpy array")
        if self.samples.dtype != np.dtype(np.int16):
            raise TypeError("AudioChunk samples must use signed 16-bit PCM")
        if self.samples.ndim != 1:
            raise ValueError("AudioChunk samples must be a flat interleaved array")
        if not isinstance(self.sample_rate, int) or isinstance(self.sample_rate, bool):
            raise TypeError("AudioChunk sample_rate must be an integer")
        if self.sample_rate <= 0:
            raise ValueError("AudioChunk sample_rate must be positive")
        if not isinstance(self.channels, int) or isinstance(self.channels, bool):
            raise TypeError("AudioChunk channels must be an integer")
        if self.channels <= 0:
            raise ValueError("AudioChunk channels must be positive")
        if self.samples.size % self.channels:
            raise ValueError("AudioChunk samples must contain complete channel frames")


@runtime_checkable
class AudioOutput(Protocol):
    """Accept typed audio chunks produced by a TTS provider.

    Implementations own any resampling, buffering, pacing, and transport
    framing required by their destination.
    """

    async def write(self, chunk: AudioChunk) -> None:
        """Queue an audio chunk for emission."""

    def clear(self) -> None:
        """Drop audio that has been queued but not yet emitted."""
