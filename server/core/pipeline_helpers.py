from .logging_utils import log_text_block
from .events import ClientTranscriptMessage
from .stream_dsl import client_message_sink, map_items
from .stream_tts import KokoroTTSProvider, tts_sink


def _text_log_sink(title, max_chars=1600):
    def attach(observable):
        return observable.subscribe(lambda text: log_text_block(title, text, max_chars=max_chars))
    return attach


def add_text_sinks(stream, data_channels, loop, role, subs, log_title=None, max_log_chars=1600):
    title = log_title or ("ASSISTANT WRITTEN RESPONSE" if role == "assistant" else "USER MESSAGE")
    if role != "assistant":
        stream.to(_text_log_sink(title, max_chars=max_log_chars), name=f"log_{role}_text", subs=subs)
    client_messages = stream | map_items(
        lambda text: ClientTranscriptMessage(role=role, content=str(text)),
        name=f"{role}_client_message",
    )
    client_messages.to(
        client_message_sink(data_channels, loop),
        name=f"client_{role}_message",
        subs=subs,
    )


def add_kokoro_tts(stream, pc, turn_signals, subs, mode, name_prefix="kokoro"):
    if mode is None:
        return

    if mode == "local":
        provider = KokoroTTSProvider(mode="local")
    elif mode == "browser":
        audio_track = getattr(pc, "assistant_audio_track", None)
        if audio_track is None:
            raise RuntimeError("Browser TTS requested, but pc.assistant_audio_track is not available")
        provider = KokoroTTSProvider(mode="webrtc", audio_track=audio_track)
    else:
        raise ValueError(f"Unknown TTS mode: {mode}")

    name = f"{name_prefix}_{mode}_tts"
    stream.to(
        tts_sink(provider, interrupts=turn_signals, name=name),
        name=name,
        subs=subs,
    )
