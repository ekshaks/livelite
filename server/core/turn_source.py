"""Transport-neutral formation of committed voice turns."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from .events import SpeechEvent


@dataclass(frozen=True)
class TurnContext:
    """Identity and cancellation state carried through one voice turn."""

    turn_id: str
    generation: int = 0
    cancelled: asyncio.Event = field(default_factory=asyncio.Event, compare=False)


@dataclass(frozen=True)
class VoiceTurn:
    context: TurnContext
    pcm16: bytes

    @property
    def turn_id(self) -> str:
        return self.context.turn_id


@dataclass(frozen=True)
class AudioTurn:
    """A VAD-completed PCM segment paired with its originating turn."""

    context: TurnContext
    samples: object


@dataclass(frozen=True)
class TurnSignal:
    event: SpeechEvent
    context: TurnContext | str | None

    @property
    def turn_id(self) -> str | None:
        if isinstance(self.context, TurnContext):
            return self.context.turn_id
        return self.context


class TurnStreams:
    """Ordered turn data and speech signals emitted by a transport adapter."""

    def __init__(self) -> None:
        self.events: asyncio.Queue[VoiceTurn | TurnSignal] = asyncio.Queue()

    async def emit_signal(self, event: SpeechEvent, context: TurnContext | None) -> None:
        signal = TurnSignal(event, context)
        await self.events.put(signal)

    async def emit_turn(self, context: TurnContext, pcm16: bytes) -> None:
        turn = VoiceTurn(context, pcm16)
        await self.events.put(turn)


class PTTTurnSource:
    """One explicitly started and committed PCM16 capture at a time."""

    def __init__(self, streams: TurnStreams, *, max_samples: int = 16_000 * 60) -> None:
        self.streams = streams
        self.max_samples = max_samples
        self.context: TurnContext | None = None
        self._chunks: list[bytes] = []
        self._samples = 0

    @property
    def turn_id(self) -> str | None:
        return self.context.turn_id if self.context is not None else None

    async def start(self, turn_id: str) -> None:
        if self.context is not None:
            raise ValueError("capture turn already active")
        self.context = TurnContext(turn_id)
        self._chunks = []
        self._samples = 0
        await self.streams.emit_signal(SpeechEvent.SPEECH_START, self.context)

    def write(self, payload: bytes) -> None:
        if self.context is None:
            raise ValueError("audio outside capture turn")
        _validate_pcm16(payload)
        samples = len(payload) // 2
        if self._samples + samples > self.max_samples:
            raise ValueError("turn exceeds 60 seconds")
        self._chunks.append(payload)
        self._samples += samples

    async def commit(self, turn_id: str) -> None:
        if self.context is None or turn_id != self.context.turn_id:
            raise ValueError("commit outside capture turn")
        context = self.context
        pcm16 = b"".join(self._chunks)
        self._reset()
        if not pcm16:
            raise ValueError("empty pcm16")
        await self.streams.emit_signal(SpeechEvent.SPEECH_END, context)
        await self.streams.emit_turn(context, pcm16)

    def cancel(self, turn_id: str | None = None) -> None:
        if turn_id is not None and (self.context is None or turn_id != self.context.turn_id):
            return
        if self.context is not None:
            self.context.cancelled.set()
        self._reset()

    def _reset(self) -> None:
        self.context = None
        self._chunks = []
        self._samples = 0


class VADTurnSource:
    """Continuous PCM16 source that emits a turn after VAD silence."""

    def __init__(self, streams: TurnStreams, is_speech, *, silence_timeout: float = 1.0, max_samples: int = 16_000 * 60) -> None:
        self.streams = streams
        self.is_speech = is_speech
        self.silence_timeout = silence_timeout
        self.max_samples = max_samples
        self.context: TurnContext | None = None
        self._chunks: list[bytes] = []
        self._samples = 0
        self._last_speech = 0.0
        self._timer: asyncio.Task | None = None
        self._analysis = b""
        self._analysis_samples = 1_600

    @property
    def turn_id(self) -> str | None:
        return self.context.turn_id if self.context is not None else None

    async def write(self, payload: bytes) -> None:
        _validate_pcm16(payload)
        self._analysis += payload
        window_bytes = self._analysis_samples * 2
        while len(self._analysis) >= window_bytes:
            window, self._analysis = self._analysis[:window_bytes], self._analysis[window_bytes:]
            samples = np.frombuffer(window, dtype="<i2")
            if self.is_speech(samples):
                if self.context is None:
                    self.context = TurnContext(uuid.uuid4().hex)
                    await self.streams.emit_signal(SpeechEvent.SPEECH_START, self.context)
                    self._timer = asyncio.create_task(self._flush_after_silence(), name="voice-vad-flush")
                if self._samples + samples.size > self.max_samples:
                    await self._flush()
                    return
                self._chunks.append(window)
                self._samples += samples.size
                self._last_speech = time.monotonic()

    async def _flush_after_silence(self) -> None:
        try:
            while self.context is not None:
                await asyncio.sleep(self.silence_timeout)
                if self.context is not None and time.monotonic() - self._last_speech >= self.silence_timeout:
                    await self._flush()
        except asyncio.CancelledError:
            pass

    async def _flush(self) -> None:
        context, pcm16 = self.context, b"".join(self._chunks)
        self.context = None
        self._chunks = []
        self._samples = 0
        self._analysis = b""
        if context and pcm16:
            await self.streams.emit_signal(SpeechEvent.SPEECH_END, context)
            await self.streams.emit_turn(context, pcm16)

    def cancel(self, _turn_id: str | None = None) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self.context is not None:
            self.context.cancelled.set()
        self.context = None
        self._chunks = []
        self._samples = 0


def _validate_pcm16(payload: bytes) -> None:
    if not payload or len(payload) % 2:
        raise ValueError("invalid pcm16")
