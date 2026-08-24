import asyncio
import time

import numpy as np

from .audio_output import AudioChunk
from .llm_utils import call_groq_chat, call_llm, create_agent, split_spoken_written
from .pipeline_helpers import add_tts, add_text_sinks
from .stream_dsl import (
    Stream,
    SubGroup,
    async_map_stage,
    final_transcript_text,
    filter_items,
    map_items,
    turn_detector,
    stt,
)
from .tts_providers.kokoro_fastapi import _tts_kokoro_stream_chunks


async def run_voice_turn(
    pcm16: bytes,
    *,
    transcribe,
    stt_timeout_seconds: float,
    llm_model: str,
    cancelled: asyncio.Event,
    is_current,
    emit,
    audio_output,
    turn_id: str,
    companion_for_text,
    response_id_factory,
    tts_client=None,
    get_transcript=None,
    answer=None,
    speak=None,
) -> None:
    """Run the shared local voice turn, rejecting output from stale generations."""
    started = time.perf_counter()
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
    await emit({"type": "response.text", "turn_id": turn_id, "response_id": response_id, "text": reply})

    async def write_audio(block, sample_rate) -> None:
        if cancelled.is_set() or not is_current():
            return
        await audio_output.write(AudioChunk(block.astype(np.int16, copy=False), sample_rate))

    if speak is not None:
        await speak(reply, cancelled, audio_output)
    else:
        await _tts_kokoro_stream_chunks(reply, cancelled, write_audio, client=tts_client)
    if not cancelled.is_set() and is_current():
        wait_until_drained = getattr(audio_output, "wait_until_drained", None)
        if wait_until_drained is not None:
            await wait_until_drained()
        if is_current():
            await emit({"type": "response.finished", "turn_id": turn_id, "response_id": response_id})


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
    llm_stage_name="multimodal_llm",
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
    transcripts = turn.segments | stt(
        provider=stt_provider,
        model=stt_model,
        model_size=stt_model_size,
        language=stt_language,
        **(stt_kwargs or {}),
    )
    user_text = transcripts | final_transcript_text()

    agent = create_agent(llm_model, prompts_path=prompts_path, prompt_id=prompt_id, name=agent_name)

    async def answer(text):
        return await call_llm(agent, text, latest_frame.get(), mode)

    assistant_response = user_text | async_map_stage(answer, name=llm_stage_name)
    assistant_response = assistant_response | filter_items(
        lambda text: bool(text and text.strip()),
        name="non_empty_assistant_response",
    )
    assistant_parts = assistant_response | map_items(split_spoken_written, name="split_spoken_written")
    assistant_spoken = assistant_parts | map_items(lambda part: part["spoken"], name="assistant_spoken")
    assistant_spoken = assistant_spoken | filter_items(lambda text: bool(text and text.strip()), name="non_empty_spoken")
    assistant_written = assistant_parts | map_items(lambda part: part["written"], name="assistant_written")
    assistant_written = assistant_written | filter_items(lambda text: bool(text and text.strip()), name="non_empty_written")

    add_text_sinks(user_text, session, role="user", subs=subs)
    add_text_sinks(assistant_written, session, role="assistant", subs=subs)
    add_tts(
        assistant_spoken,
        session.audio_output,
        turn.signals,
        subs=subs,
        mode=tts_mode,
        provider=tts_provider,
    )

    try:
        await session.closed.wait()
    finally:
        if latest_frame is not None:
            latest_frame.dispose()
        subs.dispose()
