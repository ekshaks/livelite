from .llm_utils import call_llm, create_agent, split_spoken_written
from .pipeline_helpers import add_kokoro_tts, add_text_sinks
from .stream_dsl import (
    Stream,
    SubGroup,
    async_map_stage,
    final_transcript_text,
    filter_items,
    map_items,
    turn_detector,
    whisper_stt,
)


async def run_multimodal_session(
    session,
    *,
    mode="av",
    stt_provider="mlx",
    stt_model_size="tiny",
    llm_model,
    prompts_path,
    prompt_id="visual_solver",
    agent_name="Agent",
    llm_stage_name="multimodal_llm",
    tts_mode=None,
):
    await session.wait_until_ready()

    pc = session.pc
    data_channels = session.data_channels
    audio_input = session.audio_input
    video_input = session.video_input
    main_loop = session.main_loop
    subs = SubGroup()
    latest_frame = None

    audio = Stream.source(audio_input, name="audio")
    video = Stream.source(video_input, name="video")
    latest_frame = video.latest(name="latest_frame", subs=subs)

    turn = audio | turn_detector()
    transcripts = turn.segments | whisper_stt(mode=stt_provider, model_size=stt_model_size)
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

    add_text_sinks(user_text, data_channels, main_loop, role="user", subs=subs)
    add_text_sinks(assistant_written, data_channels, main_loop, role="assistant", subs=subs)
    add_kokoro_tts(assistant_spoken, pc, turn.signals, subs=subs, mode=tts_mode)

    try:
        await session.closed.wait()
    finally:
        if latest_frame is not None:
            latest_frame.dispose()
        subs.dispose()
