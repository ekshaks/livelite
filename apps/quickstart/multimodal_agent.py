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
    tts_provider="kokoro_fastapi",
    stt_model_size="tiny",
    stt_provider="mlx",
    stt_model=None,
    stt_language="en",
    stt_kwargs=None,
    llm_model="groq:meta-llama/llama-4-scout-17b-16e-instruct",
):
    await run_multimodal_session(
        session,
        mode=mode,
        stt_provider=stt_provider,
        stt_model_size=stt_model_size,
        stt_model=stt_model,
        stt_language=stt_language,
        stt_kwargs=stt_kwargs,
        llm_model=llm_model,
        prompts_path=PROMPTS_FILE,
        prompt_id="visual_solver",
        agent_name="Math Helper",
        llm_stage_name="multimodal_llm",
        tts_mode=tts_mode,
        tts_provider=tts_provider,
    )


def _coerce(value: str):
    """Turn a CLI ``--stt-kwarg KEY=VALUE`` string into an int/float/bool/str."""
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_kv_list(pairs):
    """Parse a list of ``key=value`` strings into a dict, coercing values."""
    result = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise ValueError(f"--stt-kwarg expects key=value, got: {pair!r}")
        key, _, value = pair.partition("=")
        result[key.strip()] = _coerce(value)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run the multimodal DSL WebRTC pipeline.")
    parser.add_argument("--mode", choices=["a", "av"], default="av")
    parser.add_argument("--stt-model-size", default="tiny")
    parser.add_argument("--stt-provider", default="mlx",
                        help="STT provider: mlx | faster_whisper | deepgram")
    parser.add_argument("--stt-model")
    parser.add_argument("--stt-language", default="en")
    parser.add_argument(
        "--stt-kwarg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra keyword argument forwarded to the STT backend. Repeatable. "
            "Examples: cpu_threads=1 compute_type=int8 num_workers=1 device=cpu."
        ),
    )
    parser.add_argument("--llm-model", default="groq:meta-llama/llama-4-scout-17b-16e-instruct")
    parser.add_argument(
        "--max-concurrent-sessions",
        type=int,
        default=2,
        help="Reject WebRTC offers beyond this many active sessions (0 = unlimited).",
    )
    parser.add_argument(
        "--tts-provider",
        choices=["kokoro_fastapi", "kokoro_onnx"],
        default="kokoro_fastapi",
        help=(
            "kokoro_fastapi (default) talks to an external Kokoro-FastAPI HTTP "
            "server; kokoro_onnx runs the ONNX model in-process (no external "
            "server, fits a 2 GB box)."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tts-local", action="store_true", help="Play Kokoro TTS on the server speaker.")
    group.add_argument("--tts-browser", action="store_true", help="Stream Kokoro TTS to the browser audio track.")
    tls_group = parser.add_mutually_exclusive_group()
    tls_group.add_argument("--https", action="store_true", default=True, help="Serve HTTPS with local certs. Default.")
    tls_group.add_argument("--http", action="store_true", help="Serve plain HTTP.")
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help=(
            "Disable debug logging and per-chunk librosa audio metrics. "
            "Recommended on small servers: enables the cheap numpy-only "
            "active-speaker check."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    from server.server_asyncio import Server

    args = parse_args()
    tts_mode = "local" if args.tts_local else "browser" if args.tts_browser else None

    config = web_config(
        use_https=not args.http,
        debug=not args.no_debug,
        rms_thresh=0.025,
        input_video_sample_interval=100,
        filter_gender=None,
        max_concurrent_sessions=args.max_concurrent_sessions or None,
    )

    stt_kwargs = _parse_kv_list(args.stt_kwarg)

    server = Server(
        run_session=lambda session: run_session(
            session,
            mode=args.mode,
            tts_mode=tts_mode,
            tts_provider=args.tts_provider,
            stt_model_size=args.stt_model_size,
            stt_provider=args.stt_provider,
            stt_model=args.stt_model,
            stt_language=args.stt_language,
            stt_kwargs=stt_kwargs,
            llm_model=args.llm_model,
        ),
        config=config,
    )
    server.run()
