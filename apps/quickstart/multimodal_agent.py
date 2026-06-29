import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from server.core.multimodal_pipeline import run_multimodal_session
from server.core.server_config import web_config


PROMPTS_FILE = Path(__file__).parent / "prompts.yml"


async def run_session(
    session,
    mode="av",
    tts_mode=None,
    stt_model_size="tiny",
    llm_model="groq:meta-llama/llama-4-scout-17b-16e-instruct",
):
    await run_multimodal_session(
        session,
        mode=mode,
        stt_provider="mlx",
        stt_model_size=stt_model_size,
        llm_model=llm_model,
        prompts_path=PROMPTS_FILE,
        prompt_id="visual_solver",
        agent_name="Math Helper",
        llm_stage_name="multimodal_llm",
        tts_mode=tts_mode,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run the multimodal DSL WebRTC pipeline.")
    parser.add_argument("--mode", choices=["a", "av"], default="av")
    parser.add_argument("--stt-model-size", default="tiny")
    parser.add_argument("--llm-model", default="groq:meta-llama/llama-4-scout-17b-16e-instruct")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tts-local", action="store_true", help="Play Kokoro TTS on the server speaker.")
    group.add_argument("--tts-browser", action="store_true", help="Stream Kokoro TTS to the browser audio track.")
    tls_group = parser.add_mutually_exclusive_group()
    tls_group.add_argument("--https", action="store_true", default=True, help="Serve HTTPS with local certs. Default.")
    tls_group.add_argument("--http", action="store_true", help="Serve plain HTTP.")
    return parser.parse_args()


if __name__ == "__main__":
    from server.server_asyncio import Server

    args = parse_args()
    tts_mode = "local" if args.tts_local else "browser" if args.tts_browser else None

    config = web_config(
        use_https=not args.http,
        debug=True,
        rms_thresh=0.025,
        input_video_sample_interval=100,
        filter_gender=None,
    )

    server = Server(
        run_session=lambda session: run_session(
            session,
            mode=args.mode,
            tts_mode=tts_mode,
            stt_model_size=args.stt_model_size,
            llm_model=args.llm_model,
        ),
        config=config,
    )
    server.run()
