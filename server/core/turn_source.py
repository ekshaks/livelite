"""Transport-neutral formation of committed voice turns."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

import numpy as np

from .events import SpeechEvent


@dataclass(frozen=True)
class VoiceTurn:
    turn_id: str
    pcm16: bytes


@dataclass(frozen=True)
class TurnSignal:
    event: SpeechEvent
    turn_id: str | None


class TurnStreams:
    """Ordered turn data and speech signals emitted by a transport adapter."""

    def __init__(self) -> None:
        self.events: asyncio.Queue[VoiceTurn | TurnSignal] = asyncio.Queue()

    async def emit_signal(self, event: SpeechEvent, turn_id: str | None) -> None:
        signal = TurnSignal(event, turn_id)
        await self.events.put(signal)

    async def emit_turn(self, turn_id: str, pcm16: bytes) -> None:
        turn = VoiceTurn(turn_id, pcm16)
        await self.events.put(turn)


class PTTTurnSource:
    """One explicitly started and committed PCM16 capture at a time."""

    def __init__(self, streams: TurnStreams, *, max_samples: int = 16_000 * 60) -> None:
        self.streams = streams
        self.max_samples = max_samples
        self.turn_id: str | None = None
        self._chunks: list[bytes] = []
        self._samples = 0

    async def start(self, turn_id: str) -> None:
        if self.turn_id is not None:
            raise ValueError("capture turn already active")
        self.turn_id = turn_id
        self._chunks = []
        self._samples = 0
        await self.streams.emit_signal(SpeechEvent.SPEECH_START, turn_id)

    def write(self, payload: bytes) -> None:
        if self.turn_id is None:
            raise ValueError("audio outside capture turn")
        _validate_pcm16(payload)
        samples = len(payload) // 2
        if self._samples + samples > self.max_samples:
            raise ValueError("turn exceeds 60 seconds")
        self._chunks.append(payload)
        self._samples += samples

    async def commit(self, turn_id: str) -> None:
        if turn_id != self.turn_id:
            raise ValueError("commit outside capture turn")
        pcm16 = b"".join(self._chunks)
        self._reset()
        if not pcm16:
            raise ValueError("empty pcm16")
        await self.streams.emit_signal(SpeechEvent.SPEECH_END, turn_id)
        await self.streams.emit_turn(turn_id, pcm16)

    def cancel(self, turn_id: str | None = None) -> None:
        if turn_id is not None and turn_id != self.turn_id:
            return
        self._reset()

    def _reset(self) -> None:
        self.turn_id = None
        self._chunks = []
        self._samples = 0


class VADTurnSource:
    """Continuous PCM16 source that emits a turn after VAD silence."""

    def __init__(self, streams: TurnStreams, is_speech, *, silence_timeout: float = 1.0, max_samples: int = 16_000 * 60) -> None:
        self.streams = streams
        self.is_speech = is_speech
        self.silence_timeout = silence_timeout
        self.max_samples = max_samples
        self.turn_id: str | None = None
        self._chunks: list[bytes] = []
        self._samples = 0
        self._last_speech = 0.0
        self._timer: asyncio.Task | None = None
        self._analysis = b""
        self._analysis_samples = 1_600

    async def write(self, payload: bytes) -> None:
        _validate_pcm16(payload)
        self._analysis += payload
        window_bytes = self._analysis_samples * 2
        while len(self._analysis) >= window_bytes:
            window, self._analysis = self._analysis[:window_bytes], self._analysis[window_bytes:]
            samples = np.frombuffer(window, dtype="<i2")
            if self.is_speech(samples):
                if self.turn_id is None:
                    self.turn_id = uuid.uuid4().hex
                    await self.streams.emit_signal(SpeechEvent.SPEECH_START, self.turn_id)
                    self._timer = asyncio.create_task(self._flush_after_silence(), name="voice-vad-flush")
                if self._samples + samples.size > self.max_samples:
                    await self._flush()
                    return
                self._chunks.append(window)
                self._samples += samples.size
                self._last_speech = time.monotonic()

    async def _flush_after_silence(self) -> None:
        try:
            while self.turn_id is not None:
                await asyncio.sleep(self.silence_timeout)
                if self.turn_id is not None and time.monotonic() - self._last_speech >= self.silence_timeout:
                    await self._flush()
        except asyncio.CancelledError:
            pass

    async def _flush(self) -> None:
        turn_id, pcm16 = self.turn_id, b"".join(self._chunks)
        self.turn_id = None
        self._chunks = []
        self._samples = 0
        self._analysis = b""
        if turn_id and pcm16:
            await self.streams.emit_signal(SpeechEvent.SPEECH_END, turn_id)
            await self.streams.emit_turn(turn_id, pcm16)

    def cancel(self, _turn_id: str | None = None) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self.turn_id = None
        self._chunks = []
        self._samples = 0


def _validate_pcm16(payload: bytes) -> None:
    if not payload or len(payload) % 2:
        raise ValueError("invalid pcm16")
