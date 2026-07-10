import argparse
import asyncio
import threading

import sounddevice as sd
from reactivex.subject import Subject

from server.core.stream_dsl import (
    Stream,
    SubGroup,
    drop_while,
    final_transcript_text,
    print_sink,
    turn_detector,
    whisper_stt,
)
from server.core.tts_providers import KokoroFastApiTTSProvider, PlaybackState, tts_sink


async def run_mic_pipeline(args):
    audio_input = Subject()
    subs = SubGroup()
    playback_state = PlaybackState() if args.tts else None
    mic_muted = threading.Event()
    stop_event = asyncio.Event()

    audio = Stream.source(audio_input, name="mic_audio")
    turn = audio | turn_detector()
    segments = turn.segments
    if playback_state is not None and args.allow_interruptions:
        segments = segments | drop_while(
            lambda: playback_state.is_playing_or_recent(args.echo_suppress_seconds),
            name="drop_tts_feedback",
        )
    transcripts = segments | whisper_stt(mode="mlx", model_size=args.model_size)
    user_text = transcripts | final_transcript_text()

    user_text.to(print_sink(prefix="User: "), name="print_user_text", subs=subs)

    if args.tts:
        user_text.to(
            tts_sink(
                KokoroFastApiTTSProvider(),
                interrupts=turn.signals,
                name="kokoro_tts",
                state=playback_state,
            ),
            name="kokoro_tts",
            subs=subs,
        )
        if args.allow_interruptions:
            print(f"TTS enabled. VAD stays active; STT segments are dropped during playback and for {args.echo_suppress_seconds}s after.")
        else:
            print("TTS enabled. Mic packets are muted during playback.")


    # Pickup audio from microphone and feed it to the pipeline (via audio_input Subject)
    
    def callback(indata, frames, time_info, status):
        if status:
            print(f"mic status: {status}")
        if mic_muted.is_set():
            return
        if playback_state is not None and not args.allow_interruptions and playback_state.is_playing():
            return
        samples = indata[:, 0].copy()
        audio_input.on_next(samples)

    async def keyboard_controls():
        while not stop_event.is_set():
            command = (await asyncio.to_thread(input, "")).strip().lower()
            if command == "m":
                if mic_muted.is_set():
                    mic_muted.clear()
                    print("Mic unmuted.")
                else:
                    mic_muted.set()
                    print("Mic muted.")
            elif command == "q":
                stop_event.set()

    controls_task = asyncio.create_task(keyboard_controls())

    print("Listening. Speak, then pause for transcription.")
    print("Controls: m + Enter toggles mic mute, q + Enter stops, Ctrl-C also stops.")
    try:
        with sd.InputStream(
            samplerate=args.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=args.block_size,
            callback=callback,
        ):
            while not stop_event.is_set():
                await asyncio.sleep(0.25)
    finally:
        stop_event.set()
        controls_task.cancel()
        audio_input.on_completed()
        subs.dispose()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the basic DSL pipeline from local microphone input.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--block-size", type=int, default=8000)
    parser.add_argument("--model-size", default="tiny")
    parser.add_argument("--tts", action="store_true", help="Speak transcribed text with Kokoro TTS.")
    parser.add_argument("--echo-suppress-seconds", type=float, default=2.0)
    parser.add_argument("--allow-interruptions", action="store_true", help="Keep VAD active during TTS so speech can interrupt playback.")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(run_mic_pipeline(parse_args()))
    except KeyboardInterrupt:
        print("\nStopped.")
