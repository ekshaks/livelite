import re

from .audio_output import AudioOutput
from .logging_utils import log_text_block, monitor_log
from .events import ClientTranscriptMessage
from .stream_dsl import client_message_sink, expand_items, map_items
from .tts_providers import KokoroFastApiTTSProvider, tts_sink
from .tts_providers.kokoro_onnx import KokoroOnnxTTSProvider
from .tts_providers.piper import PiperTTSProvider

# Split after sentence/clause punctuation (plus any closing quotes/brackets)
# followed by whitespace. Same boundary the spell app uses for its per-phrase
# TTS requests; hoisted here so every pipeline can share it.
SPOKEN_PHRASE_BOUNDARY = re.compile(r'(?<=[,.!?;:])(?:["\')\]]+)?\s+')


def split_spoken_phrases(text):
    """Split spoken text into punctuation-delimited phrases for TTS.

    Non-streaming TTS providers (kokoro_onnx) synthesize a whole clip per
    request, so first-audio latency equals the synthesis time of the first
    phrase — short phrases keep that bounded. Returns a tuple of non-empty
    phrases.
    """
    return tuple(
        phrase.strip()
        for phrase in SPOKEN_PHRASE_BOUNDARY.split(text or "")
        if phrase.strip()
    )


def _text_log_sink(title, max_chars=1600):
    def attach(observable):
        return observable.subscribe(lambda text: log_text_block(title, text, max_chars=max_chars))
    return attach


def add_text_sinks(
    stream,
    session,
    role,
    subs,
    log_title=None,
    max_log_chars=1600,
    log=True,
):
    """Send transcript text to the client, with optional server-side logging.

    Set ``log=False`` for authentication or other sensitive user input. Client
    delivery is unaffected.
    """
    title = log_title or ("ASSISTANT WRITTEN RESPONSE" if role == "assistant" else "USER MESSAGE")
    if role != "assistant" and log:
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


# Providers that synthesize the whole clip before yielding audio: split the
# input into short phrases so first-audio latency stays bounded.
_NON_STREAMING_TTS_PROVIDERS = frozenset({"kokoro_onnx"})


_TTS_PROVIDER_ALIASES = {"kokoro": "kokoro_fastapi"}


def _resolve_tts_provider_name(provider_name):
    """Canonicalize a TTS provider identifier (translating legacy aliases).

    Historically the muapps configs used ``provider: kokoro`` to mean
    "Kokoro-FastAPI HTTP server". Newer providers (``kokoro_onnx``,
    ``piper``) introduced explicit names, so the bare ``"kokoro"`` value is
    treated as an alias to keep old configs working without a silent switch
    in behavior.
    """
    return _TTS_PROVIDER_ALIASES.get(provider_name, provider_name)


def _make_tts_provider(provider_name, output_mode, audio_output):
    """Instantiate a TTS provider for the requested output mode.

    Supported providers:

    * ``kokoro_fastapi`` — external Kokoro-FastAPI HTTP server. Streams PCM
      chunks; good on desktop where a second PyTorch process is fine.
      The legacy alias ``"kokoro"`` maps here.
    * ``kokoro_onnx``   — in-process Kokoro (ONNX runtime). Whole-clip
      synth; combine with :func:`split_spoken_phrases` to bound TTFB.
    * ``piper``         — in-process Piper (rhasspy) TTS. Truly streaming
      PCM output, small memory footprint, preferred on small CPU boxes.
    """
    provider_name = _resolve_tts_provider_name(provider_name)
    if provider_name == "kokoro_fastapi":
        if output_mode == "local":
            return KokoroFastApiTTSProvider(mode="local")
        return KokoroFastApiTTSProvider(mode="webrtc", audio_track=audio_output)
    if provider_name == "kokoro_onnx":
        if output_mode == "local":
            return KokoroOnnxTTSProvider(output="local")
        return KokoroOnnxTTSProvider(output="webrtc", audio_track=audio_output)
    if provider_name == "piper":
        if output_mode == "local":
            return PiperTTSProvider(output="local")
        return PiperTTSProvider(output="webrtc", audio_track=audio_output)
    raise ValueError(f"Unknown TTS provider: {provider_name}")


def add_kokoro_tts(
    stream,
    audio_output,
    turn_signals,
    subs,
    mode,
    provider="kokoro_fastapi",
    name_prefix="kokoro",
):
    """Attach a TTS sink to ``stream``.

    Backwards-compatible alias for :func:`add_tts`. Kept as ``add_kokoro_tts``
    so existing muapp entrypoints (spell, chess) don't need to be updated
    beyond passing the new ``provider`` argument.
    """
    return add_tts(
        stream,
        audio_output,
        turn_signals,
        subs=subs,
        mode=mode,
        provider=provider,
        name_prefix=name_prefix,
    )


def add_tts(
    stream,
    audio_output,
    turn_signals,
    subs,
    mode,
    provider="kokoro_fastapi",
    name_prefix="tts",
):
    """Attach a TTS sink to ``stream``.

    ``mode`` picks the output surface (``local`` = server speaker via
    sounddevice, ``browser`` = the session's audio output). ``provider``
    picks the synthesis backend (``kokoro_fastapi``, ``kokoro_onnx``, or
    ``piper``).
    """
    if mode is None:
        return

    if mode == "local":
        output = None
        output_mode = "local"
    elif mode == "browser":
        output = audio_output
        if output is None:
            raise RuntimeError("Browser TTS requested, but no audio output is available")
        if not isinstance(output, AudioOutput):
            raise TypeError("Browser TTS requires an AudioOutput implementation")
        output_mode = "webrtc"
    else:
        raise ValueError(f"Unknown TTS mode: {mode}")

    canonical_provider = _resolve_tts_provider_name(provider)
    tts_provider = _make_tts_provider(canonical_provider, output_mode, output)
    monitor_log(
        f"tts sink attached provider_input={provider} "
        f"provider_effective={canonical_provider} mode={mode} name_prefix={name_prefix}"
    )

    if canonical_provider in _NON_STREAMING_TTS_PROVIDERS:
        # Whole-clip providers pay all their synthesis time before any audio
        # comes out. Split into short phrases so first-audio latency stays
        # bounded per request; streaming providers (piper, kokoro_fastapi)
        # already emit PCM incrementally and don't need this.
        stream = stream | expand_items(split_spoken_phrases, name=f"{name_prefix}_phrase_split")

    name = f"{name_prefix}_{canonical_provider}_{mode}_tts"
    stream.to(
        tts_sink(tts_provider, interrupts=turn_signals, name=name),
        name=name,
        subs=subs,
    )
    return tts_provider
