"""Loopback WebSocket protocol adapters for the shared voice turn engine."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from dataclasses import replace

from openai import AsyncOpenAI
from .events import SpeechEvent
from .multimodal_pipeline import run_voice_turn
from .stt.pinned import PinnedWhisper
from .token_signing import decode_json, encode_json, sign, signature_matches
from .turn_source import PTTTurnSource, TurnContext, TurnSignal, TurnStreams, VADTurnSource, VoiceTurn
from .turndet import _build_is_speech
from .tts_providers.kokoro_fastapi import _kokoro_fastapi_base_url
from .websocket_audio import WebSocketPCMOutput

PROTOCOL = "mulive.voice.v1"
PCM16_SAMPLE_RATE = 16_000
logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class VoicePrincipal:
    subject: str
    display_name: str


class VoiceTokenStore:
    """Short-lived single-use tokens; process-local replay prevention is intentional for MVP."""

    def __init__(self, secret: str, ttl_seconds: int = 300):
        if len(secret.encode()) < 32:
            raise ValueError("MULIVE_AUTH_COOKIE_SECRET must be at least 32 bytes")
        self.secret = secret.encode()
        self.ttl_seconds = ttl_seconds
        self.used: dict[str, int] = {}

    def issue(self, principal: VoicePrincipal) -> str:
        payload = {"sub": principal.subject, "name": principal.display_name, "aud": PROTOCOL, "exp": int(time.time()) + self.ttl_seconds, "jti": secrets.token_urlsafe(16)}
        encoded = encode_json(payload)
        return f"{encoded}.{sign(self.secret, encoded)}"

    def consume(self, token: str | None) -> VoicePrincipal | None:
        try:
            now = int(time.time())
            self.used = {jti: exp for jti, exp in self.used.items() if exp >= now}
            encoded, signature = (token or "").split(".", 1)
            if not signature_matches(self.secret, encoded, signature):
                return None
            payload = decode_json(encoded)
            if payload["aud"] != PROTOCOL or int(payload["exp"]) < now or payload["jti"] in self.used:
                return None
            self.used[payload["jti"]] = int(payload["exp"])
            return VoicePrincipal(payload["sub"], payload["name"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None


def _default_assistant_identity(_text: str) -> str:
    """Return the transport-neutral identity used when an app supplies none."""
    return "assistant"


class VoicePipeline:
    """Provider configuration for the transport-neutral voice turn engine."""

    def __init__(self, *, companion_for_text=_default_assistant_identity):
        self.stt = PinnedWhisper(mode=os.getenv("MULIVE_VOICE_STT_MODE", "faster_whisper"), model_size=os.getenv("MULIVE_VOICE_STT_MODEL", "tiny"))
        self.stt_timeout_seconds = float(os.getenv("MULIVE_VOICE_STT_TIMEOUT_SECONDS", "30"))
        self.stt_load_timeout_seconds = float(os.getenv("MULIVE_VOICE_STT_LOAD_TIMEOUT_SECONDS", "180"))
        if self.stt_timeout_seconds <= 0 or self.stt_load_timeout_seconds <= 0:
            raise ValueError("voice STT timeouts must be positive")
        self.llm_model = os.getenv("MULIVE_VOICE_LLM_MODEL", "openai/gpt-oss-20b")
        self.tts_client = AsyncOpenAI(base_url=_kokoro_fastapi_base_url(), api_key="not-needed")
        self.companion_for_text = companion_for_text

    async def wait_ready(self) -> None:
        await asyncio.wait_for(self.stt.wait_ready(), timeout=self.stt_load_timeout_seconds)

    async def run(self, pcm16, context, is_current, emit, audio_output) -> None:
        await run_voice_turn(pcm16, context=context, transcribe=self.stt.transcribe, stt_timeout_seconds=self.stt_timeout_seconds, llm_model=self.llm_model, is_current=is_current, emit=emit, audio_output=audio_output, companion_for_text=getattr(self, "companion_for_text", _default_assistant_identity), response_id_factory=lambda: secrets.token_urlsafe(12), tts_client=getattr(self, "tts_client", None))

    def close(self) -> None:
        self.stt.shutdown()

    async def aclose(self) -> None:
        self.close()
        await self.tts_client.close()


class VoiceSession:
    """Mode-selected turn source plus shared engine and serialized output."""

    def __init__(self, principal: VoicePrincipal, pipeline: VoicePipeline, send_json, send_bytes):
        self.principal, self.pipeline = principal, pipeline
        self._send_json, self._send_bytes = send_json, send_bytes
        self._send_lock = asyncio.Lock()
        self.audio_output = WebSocketPCMOutput(send_bytes, self._send_lock)
        self.ready = False
        self.mode: str | None = None
        self.streams: TurnStreams | None = None
        self.source = None
        self._events_task: asyncio.Task | None = None
        self._response_task: asyncio.Task | None = None
        self._generation = 0
        self._response_context: TurnContext | None = None
        self._response_turn_id: str | None = None
        self.sequence = 0

    @property
    def turn_id(self) -> str | None:
        if self.source is not None and self.source.turn_id is not None:
            return self.source.turn_id
        return self._response_turn_id

    @property
    def task(self):
        return self._response_task

    async def handle_json(self, event: dict) -> None:
        kind = event.get("type")
        if not self.ready:
            if kind != "session.hello" or (event.get("protocol") or event.get("text")) != PROTOCOL:
                raise ValueError("hello required")
            mode = event.get("mode", "ptt")
            if mode not in {"ptt", "vad"}:
                raise ValueError("unknown voice mode")
            self.ready, self.mode, self.streams = True, mode, TurnStreams()
            self.source = PTTTurnSource(self.streams) if mode == "ptt" else VADTurnSource(self.streams, _voice_is_speech())
            self._events_task = asyncio.create_task(self._consume_source_events(), name="voice-turn-events")
            await self.emit({"type": "session.ready", "protocol": PROTOCOL, "mode": mode, "subject": self.principal.subject})
            return
        if self.mode == "ptt":
            if kind == "turn.start":
                turn_id = event.get("turn_id")
                if not isinstance(turn_id, str) or not turn_id:
                    raise ValueError("turn_id required")
                await self.source.start(turn_id)
            elif kind == "turn.commit":
                await self.source.commit(event.get("turn_id"))
            elif kind == "turn.cancel":
                self.source.cancel(event.get("turn_id"))
                await self._cancel_response(emit_cancelled=False)
        elif kind in {"turn.start", "turn.commit", "turn.cancel"}:
            raise ValueError("PTT events are invalid in vad mode")

    async def handle_pcm16(self, payload: bytes) -> None:
        if not self.ready or self.source is None:
            raise ValueError("audio before hello")
        result = self.source.write(payload)
        if asyncio.iscoroutine(result):
            await result

    async def _consume_source_events(self) -> None:
        assert self.streams is not None
        try:
            while True:
                event = await self.streams.events.get()
                if isinstance(event, TurnSignal):
                    if event.event == SpeechEvent.SPEECH_START:
                        await self._cancel_response(emit_cancelled=True)
                elif isinstance(event, VoiceTurn):
                    await self._start_response(event)
        except asyncio.CancelledError:
            pass

    async def _start_response(self, turn: VoiceTurn) -> None:
        await self._cancel_response(emit_cancelled=False)
        self._generation += 1
        generation = self._generation
        context = replace(turn.context, generation=generation)
        self._response_context = context
        self._response_turn_id = turn.turn_id
        await self.emit({"type": "turn.committed", "turn_id": turn.turn_id})
        self._response_task = asyncio.create_task(self._run(turn, context), name=f"voice:{turn.turn_id}")

    async def _run(self, turn: VoiceTurn, context: TurnContext) -> None:
        try:
            await self.pipeline.run(turn.pcm16, context, lambda: self._response_context is context and not context.cancelled.is_set(), self.emit, self.audio_output)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if self._response_context is context:
                logger.exception("voice pipeline failed turn_id=%s", turn.turn_id)
                await self.emit({"type": "error", "turn_id": turn.turn_id, "text": f"pipeline failed: {type(exc).__name__}"})
        finally:
            if self._response_context is context:
                self._response_task = None
                self._response_turn_id = None
                self._response_context = None

    async def _cancel_response(self, *, emit_cancelled: bool) -> None:
        task, turn_id = self._response_task, self._response_turn_id
        if task is None:
            return
        self._generation += 1
        if self._response_context is not None:
            self._response_context.cancelled.set()
            self._response_context = None
        task.cancel()
        self.audio_output.clear()
        self._response_task = None
        self._response_turn_id = None
        if emit_cancelled and turn_id is not None:
            await self.emit({"type": "response.cancelled", "turn_id": turn_id})

    async def cancel(self) -> None:
        if self.source is not None:
            self.source.cancel()
        await self._cancel_response(emit_cancelled=False)
        if self._events_task is not None:
            self._events_task.cancel()
            await self._events_task
            self._events_task = None
        await self.audio_output.close()

    async def emit(self, event: dict) -> None:
        self.sequence += 1
        event["sequence"] = self.sequence
        async with self._send_lock:
            await self._send_json(event)


def _voice_is_speech():
    return _build_is_speech(threshold=0.4, min_speech_duration_ms=100, min_silence_duration_ms=2000, speech_pad_ms=200, rate=PCM16_SAMPLE_RATE)
