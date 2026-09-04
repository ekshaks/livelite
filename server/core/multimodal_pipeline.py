import asyncio
import uuid
from dataclasses import replace

from .events import ClientTranscriptMessage, SpeechEvent
from .turn_source import TurnContext
from .voice_engine import run_voice_turn


class WebRTCVoiceTurnRunner:
    """Bridge completed Rx transcripts into the shared voice-turn engine.

    VAD and STT deliberately stay in the existing Rx pipeline.  This runner
    owns only the post-transcript turn lifetime, including VAD barge-in.
    """

    def __init__(self, *, session, answer, speak):
        self.session = session
        self.answer = answer
        self.speak = speak
        self._generation = 0
        self._input_context: TurnContext | None = None
        self._active_context: TurnContext | None = None
        self._task: asyncio.Task | None = None

    def on_vad_event(self, event) -> None:
        if event.signal == SpeechEvent.SPEECH_START:
            self._cancel_active()
            if self._input_context is not None:
                self._input_context.cancelled.set()
            self._input_context = event.context

    def on_signal(self, event) -> None:
        """Compatibility hook for callers that only retain the signal value."""
        if event == SpeechEvent.SPEECH_START:
            self._cancel_active()

    def start(self, text: str, context: TurnContext | None = None) -> None:
        if context is not None and context.cancelled.is_set():
            return
        self._cancel_active()
        self._generation += 1
        generation = self._generation
        context = replace(context, generation=generation) if context is not None else TurnContext(uuid.uuid4().hex, generation)
        self._input_context = None
        self._active_context = context
        self._task = asyncio.create_task(
            self._run(text, context), name=f"webrtc-voice:{context.turn_id}"
        )

    def cancel(self) -> asyncio.Task | None:
        task = self._cancel_active()
        if self._input_context is not None:
            self._input_context.cancelled.set()
            self._input_context = None
        return task

    def _cancel_active(self) -> asyncio.Task | None:
        self._generation += 1
        if self._active_context is not None:
            self._active_context.cancelled.set()
            self._active_context = None
        task = self._task
        if self._task is not None:
            self._task.cancel()
            self._task = None
        clear = getattr(self.session.audio_output, "clear", None)
        if clear is not None:
            clear()
        return task

    async def aclose(self) -> None:
        task = self.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    async def _run(self, text: str, context: TurnContext) -> None:
        def is_current() -> bool:
            return self._active_context is context and not context.cancelled.is_set()

        async def emit(event: dict) -> None:
            if not is_current():
                return
            if event["type"] == "transcript.final":
                self.session.send_to_client(ClientTranscriptMessage(role="user", content=event["text"]))
            elif event["type"] == "response.text":
                self.session.send_to_client(ClientTranscriptMessage(role="assistant", content=event["text"]))
            else:
                self.session.send_to_client(event)

        try:
            await run_voice_turn(
                b"",
                context=context,
                transcribe_turn=None,
                stt_timeout_seconds=0,
                llm_model="",
                is_current=is_current,
                emit=emit,
                audio_output=self.session.audio_output,
                response_id_factory=lambda: uuid.uuid4().hex,
                get_transcript=lambda: _completed(text),
                answer=self.answer,
                speak=self.speak,
            )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if is_current():
                self.session.send_to_client(
                    {
                        "type": "error",
                        "turn_id": context.turn_id,
                        "text": f"pipeline failed: {type(exc).__name__}",
                    }
                )
        finally:
            if self._active_context is context:
                self._active_context = None
                self._task = None


async def _completed(value):
    return value
