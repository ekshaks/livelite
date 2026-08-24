import asyncio
import time
import uuid
from dataclasses import replace

import numpy as np

from .audio_output import AudioChunk
from .events import ClientTranscriptMessage, SpeechEvent
from .llm_utils import call_groq_chat, call_llm, create_agent, split_spoken_written
from .pipeline_helpers import make_tts_provider
from .stream_dsl import (
    Stream,
    SubGroup,
    final_transcripts,
    turn_detector,
    stt,
)
from .turn_source import TurnContext
from .tts_providers.kokoro_fastapi import _tts_kokoro_stream_chunks


async def run_voice_turn(
    pcm16: bytes,
    *,
    context: TurnContext,
    transcribe,
    stt_timeout_seconds: float,
    llm_model: str,
    is_current,
    emit,
    audio_output,
    companion_for_text,
    response_id_factory,
    tts_client=None,
    get_transcript=None,
    answer=None,
    speak=None,
    split_response=split_spoken_written,
) -> None:
    """Run the shared local voice turn, rejecting output from stale generations."""
    cancelled = context.cancelled
    turn_id = context.turn_id
    try:
        text = await (get_transcript() if get_transcript is not None else asyncio.wait_for(transcribe(np.frombuffer(pcm16, dtype=np.int16)), timeout=stt_timeout_seconds))
    except TimeoutError:
        if is_current():
            await emit({"type": "error", "turn_id": turn_id, "text": "Speech recognition timed out. Try again."})
        return
    if cancelled.is_set() or not is_current():
        return
    companion = companion_for_text(text)
    await emit({"type": "transcript.final", "turn_id": turn_id, "text": text, "companion": companion})
    reply = await (answer(text) if answer is not None else call_groq_chat(
        llm_model,
        system_prompt=f"You are {companion.title()}, a concise Mulive voice companion. Answer the user directly. Distinguish hypotheses from verified facts.",
        user_prompt=text,
    ))
    if cancelled.is_set() or not is_current():
        return
    response_id = response_id_factory()
    await emit({"type": "response.started", "turn_id": turn_id, "response_id": response_id, "sample_rate": 24_000})
    parts = split_response(reply)
    spoken = parts["spoken"]
    written = parts["written"]
    await emit({"type": "response.text", "turn_id": turn_id, "response_id": response_id, "text": written})

    async def write_audio(block, sample_rate) -> None:
        if cancelled.is_set() or not is_current():
            return
        await audio_output.write(AudioChunk(block.astype(np.int16, copy=False), sample_rate))

    if speak is not None:
        await speak(spoken, cancelled, audio_output)
    else:
        await _tts_kokoro_stream_chunks(spoken, cancelled, write_audio, client=tts_client)
    if not cancelled.is_set() and is_current():
        wait_until_drained = getattr(audio_output, "wait_until_drained", None)
        if wait_until_drained is not None:
            await wait_until_drained()
        if is_current():
            await emit({"type": "response.finished", "turn_id": turn_id, "response_id": response_id})


class WebRTCVoiceTurnRunner:
    """Bridge completed Rx transcripts into the shared voice-turn engine.

    VAD and STT deliberately stay in the existing Rx pipeline.  This runner
    owns only the post-transcript turn lifetime, including VAD barge-in.
    """

    def __init__(self, *, session, answer, speak, companion_for_text=lambda _text: "assistant"):
        self.session = session
        self.answer = answer
        self.speak = speak
        self.companion_for_text = companion_for_text
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
                transcribe=None,
                stt_timeout_seconds=0,
                llm_model="",
                is_current=is_current,
                emit=emit,
                audio_output=self.session.audio_output,
                companion_for_text=self.companion_for_text,
                response_id_factory=lambda: uuid.uuid4().hex,
                get_transcript=lambda: _completed(text),
                answer=self.answer,
                speak=self.speak,
            )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if is_current():
                print(f"WebRTC voice turn failed: {type(exc).__name__}: {exc}")
        finally:
            if self._active_context is context:
                self._active_context = None
                self._task = None


async def _completed(value):
    return value


async def run_multimodal_session(
    session,
    *,
    mode="av",
    stt_provider="mlx",
    stt_model_size="tiny",
    stt_model=None,
    stt_language="en",
    stt_kwargs=None,
    llm_model,
    prompts_path,
    prompt_id="visual_solver",
    agent_name="Agent",
    tts_mode=None,
    tts_provider="kokoro_fastapi",
):
    await session.wait_until_ready()

    audio_input = session.audio_input
    video_input = session.video_input
    subs = SubGroup()
    latest_frame = None

    audio = Stream.source(audio_input, name="audio")
    video = Stream.source(video_input, name="video")
    latest_frame = video.latest(name="latest_frame", subs=subs)

    turn = audio | turn_detector()
    transcripts = turn.turns | stt(
        provider=stt_provider,
        model=stt_model,
        model_size=stt_model_size,
        language=stt_language,
        **(stt_kwargs or {}),
    )
    completed_transcripts = transcripts | final_transcripts()

    agent = create_agent(llm_model, prompts_path=prompts_path, prompt_id=prompt_id, name=agent_name)

    async def answer(text):
        return await call_llm(agent, text, latest_frame.get(), mode)

    tts = None
    if tts_mode is not None:
        output_mode = "local" if tts_mode == "local" else "webrtc" if tts_mode == "browser" else None
        if output_mode is None:
            raise ValueError(f"Unknown TTS mode: {tts_mode}")
        tts = make_tts_provider(tts_provider, output_mode, session.audio_output if output_mode == "webrtc" else None)

    async def speak(text, cancelled, _audio_output):
        if tts is not None and text.strip():
            await tts.speak(text, cancelled)

    turns = WebRTCVoiceTurnRunner(session=session, answer=answer, speak=speak)
    # Rx still owns VAD + STT.  The runner takes over only once a final text
    # value exists, which gives WebRTC the same LLM/TTS/cancellation core as WS.
    turn.events.to(lambda observable: observable.subscribe(turns.on_vad_event), name="webrtc_barge_in", subs=subs)
    completed_transcripts.to(
        lambda observable: observable.subscribe(lambda event: turns.start(event.text, event.context)),
        name="webrtc_voice_turns",
        subs=subs,
    )

    try:
        await session.closed.wait()
    finally:
        if latest_frame is not None:
            latest_frame.dispose()
        subs.dispose()
        await turns.aclose()
        close = getattr(tts, "aclose", None)
        if close is not None:
            await close()
