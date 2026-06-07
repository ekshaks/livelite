import asyncio
from pathlib import Path

from server.core.stream_dsl import (
    Stream,
    SubGroup,
    client_text_sink,
    print_sink,
    turn_detector,
    whisper_stt,
)


async def create_pipeline(pc, data_channels, audio_input, video_input, main_loop):
    """
    DSL version of the basic pipeline.

    audio -> turn detector -> STT -> text sinks
    """
    subs = SubGroup()

    audio = Stream.source(audio_input, name="audio")
    turn = audio | turn_detector()
    user_text = turn.segments | whisper_stt(mode="faster_whisper")

    user_text.to(print_sink(prefix="User: "), name="print_user_text", subs=subs)
    user_text.to(
        client_text_sink(data_channels, main_loop, role="user"),
        name="client_user_text",
        subs=subs,
    )

    try:
        while pc.connectionState not in {"closed", "failed"}:
            await asyncio.sleep(0.25)
    finally:
        subs.dispose()


if __name__ == "__main__":
    from server.server_asyncio import Server

    server_dir = Path(__file__).parent.parent / "server"
    config = {
        "debug": False,
        "ssl_keyfile": server_dir / "certs/key.pem",
        "ssl_certfile": server_dir / "certs/cert.pem",
    }

    server = Server(create_pipeline=create_pipeline, config=config)
    server.run()
