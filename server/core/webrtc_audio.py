import asyncio
import fractions
import time
from io import BytesIO
from typing import Optional

import av
import numpy as np
from aiortc.mediastreams import AudioStreamTrack, MediaStreamError
from pydub import AudioSegment

from .audio_output import AudioChunk


class AssistantAudioTrack(AudioStreamTrack):
    """Outbound WebRTC audio track for assistant speech.

    The track emits silence until audio is queued. Later `client_audio_sink()`
    can feed this with PCM chunks from a TTS provider.
    """

    kind = "audio"

    def __init__(self, sample_rate: int = 48000, frame_duration_ms: int = 20):
        super().__init__()
        self.sample_rate = sample_rate
        self.samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
        self._queue = asyncio.Queue()
        self._buffer = np.array([], dtype=np.int16)
        self._start: Optional[float] = None
        self._timestamp = 0
        self._logged_first_pcm = False

    async def write(self, chunk: AudioChunk) -> None:
        """Queue mono PCM16 audio for playback on the browser track."""
        if chunk.channels != 1:
            raise ValueError("WebRTC assistant audio currently requires mono chunks")
        pcm = chunk.samples
        if chunk.sample_rate != self.sample_rate:
            pcm = self._resample_mono_int16(pcm, chunk.sample_rate, self.sample_rate)
        if not self._logged_first_pcm:
            self._logged_first_pcm = True
            print(f"AssistantAudioTrack queued PCM: {len(pcm)} samples @ {self.sample_rate}Hz")
        await self._queue.put(pcm)

    async def write_pcm(self, samples, sample_rate: int) -> None:
        """Compatibility adapter for callers using the previous PCM method."""
        await self.write(
            AudioChunk(np.asarray(samples, dtype=np.int16).reshape(-1), sample_rate)
        )

    def clear(self):
        """Drop queued assistant audio that has not yet been emitted."""
        dropped_chunks = 0
        while True:
            try:
                self._queue.get_nowait()
                dropped_chunks += 1
            except asyncio.QueueEmpty:
                break
        buffered_samples = len(self._buffer)
        self._buffer = np.array([], dtype=np.int16)
        print(f"[interrupt] AssistantAudioTrack cleared chunks={dropped_chunks} buffered_samples={buffered_samples}")

    async def recv(self):
        if self.readyState != "live":
            raise MediaStreamError

        await self._pace()
        frame_samples = self._next_samples()

        frame = av.AudioFrame(format="s16", layout="mono", samples=self.samples_per_frame)
        frame.planes[0].update(frame_samples.tobytes())
        frame.sample_rate = self.sample_rate
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        self._timestamp += self.samples_per_frame
        return frame

    async def _pace(self):
        if self._start is None:
            self._start = time.time()
            return
        wait = self._start + (self._timestamp / self.sample_rate) - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

    def _next_samples(self):
        while len(self._buffer) < self.samples_per_frame:
            try:
                chunk = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._buffer = np.concatenate([self._buffer, chunk])

        if len(self._buffer) >= self.samples_per_frame:
            out = self._buffer[: self.samples_per_frame]
            self._buffer = self._buffer[self.samples_per_frame :]
            return out

        out = np.zeros(self.samples_per_frame, dtype=np.int16)
        if len(self._buffer) > 0:
            out[: len(self._buffer)] = self._buffer
            self._buffer = np.array([], dtype=np.int16)
        return out

    @staticmethod
    def _resample_mono_int16(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        segment = AudioSegment(
            data=samples.tobytes(),
            sample_width=2,
            frame_rate=src_rate,
            channels=1,
        )
        segment = segment.set_frame_rate(dst_rate)
        return np.frombuffer(segment.raw_data, dtype=np.int16)
