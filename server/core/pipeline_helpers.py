from .logging_utils import log_text_block
from .events import ClientTranscriptMessage
from .stream_dsl import client_message_sink, map_items
from .tts_providers import KokoroFastApiTTSProvider, tts_sink
from .tts_providers.kokoro_onnx import KokoroOnnxTTSProvider


def _text_log_sink(title, max_chars=1600):
    def attach(observable):
        return observable.subscribe(lambda text: log_text_block(title, text, max_chars=max_chars))
    return attach


def add_text_sinks(stream, session, role, subs, log_title=None, max_log_chars=1600):
    title = log_title or ("ASSISTANT WRITTEN RESPONSE" if role == "assistant" else "USER MESSAGE")
    if role != "assistant":
        stream.to(_text_log_sink(title, max_chars=max_log_chars), name=f"log_{role}_text", subs=subs)
    client_messages = stream | map_items(
        lambda text: ClientTranscriptMessage(role=role, content=str(text)),
        name=f"{role}_client_message",
    )
    client_messages.to(
        client_message_sink(session),
        name=f"client_{role}_message",
        subs=subs,
    )


def _make_kokoro_provider(provider_name, output_mode, audio_track):
    """Instantiate a Kokoro TTS provider for the requested output mode.

    Two providers are supported today:

    * ``kokoro_fastapi`` (default) — talks to an external Kokoro-FastAPI HTTP
      server (a second process). Streams PCM chunks; good on desktop.
    * ``kokoro_onnx`` — runs the ONNX model in-process. Preferred on small
      cloud servers because it avoids running a second PyTorch process.
    """
    if provider_name == "kokoro_fastapi":
        if output_mode == "local":
            return KokoroFastApiTTSProvider(mode="local")
        return KokoroFastApiTTSProvider(mode="webrtc", audio_track=audio_track)
    if provider_name == "kokoro_onnx":
        if output_mode == "local":
            return KokoroOnnxTTSProvider(output="local")
        return KokoroOnnxTTSProvider(output="webrtc", audio_track=audio_track)
    raise ValueError(f"Unknown Kokoro TTS provider: {provider_name}")


def add_kokoro_tts(
    stream,
    pc,
    turn_signals,
    subs,
    mode,
    provider="kokoro_fastapi",
    name_prefix="kokoro",
):
    """Attach a Kokoro TTS sink to ``stream``.

    ``mode`` picks the output surface (``local`` = server speaker via
    sounddevice, ``browser`` = outbound WebRTC audio track). ``provider``
    picks the synthesis backend (``kokoro_fastapi`` or ``kokoro_onnx``).
    """
    if mode is None:
        return

    if mode == "local":
        audio_track = None
        output_mode = "local"
    elif mode == "browser":
        audio_track = getattr(pc, "assistant_audio_track", None)
        if audio_track is None:
            raise RuntimeError("Browser TTS requested, but pc.assistant_audio_track is not available")
        output_mode = "webrtc"
    else:
        raise ValueError(f"Unknown TTS mode: {mode}")

    tts_provider = _make_kokoro_provider(provider, output_mode, audio_track)

    name = f"{name_prefix}_{provider}_{mode}_tts"
    stream.to(
        tts_sink(tts_provider, interrupts=turn_signals, name=name),
        name=name,
        subs=subs,
    )
