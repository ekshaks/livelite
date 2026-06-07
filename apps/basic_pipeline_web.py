import argparse
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
from server.core.stream_tts import KokoroTTSProvider, tts_sink


async def create_pipeline(pc, data_channels, audio_input, video_input, main_loop, tts_mode=None):
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

    if tts_mode == "local":
        user_text.to(
            tts_sink(KokoroTTSProvider(mode="local"), interrupts=turn.signals, name="kokoro_local_tts"),
            name="kokoro_local_tts",
            subs=subs,
        )
    elif tts_mode == "browser":
        audio_track = getattr(pc, "assistant_audio_track", None)
        if audio_track is None:
            raise RuntimeError("Browser TTS requested, but pc.assistant_audio_track is not available")
        user_text.to(
            tts_sink(
                KokoroTTSProvider(mode="webrtc", audio_track=audio_track),
                interrupts=turn.signals,
                name="kokoro_browser_tts",
            ),
            name="kokoro_browser_tts",
            subs=subs,
        )

    try:
        while pc.connectionState not in {"closed", "failed"}:
            await asyncio.sleep(0.25)
    finally:
        subs.dispose()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the basic DSL WebRTC pipeline.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tts-local", action="store_true", help="Play Kokoro TTS on the server speaker.")
    group.add_argument("--tts-browser", action="store_true", help="Stream Kokoro TTS to the browser audio track.")
    return parser.parse_args()


if __name__ == "__main__":
    from server.server_asyncio import Server

    args = parse_args()
    tts_mode = "local" if args.tts_local else "browser" if args.tts_browser else None

    server_dir = Path(__file__).parent.parent / "server"
    config = {
        "debug": False,
        "ssl_keyfile": server_dir / "certs/key.pem",
        "ssl_certfile": server_dir / "certs/cert.pem",
    }

    server = Server(
        create_pipeline=lambda *pipeline_args: create_pipeline(*pipeline_args, tts_mode=tts_mode),
        config=config,
    )
    server.run()
