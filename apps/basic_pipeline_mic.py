import argparse
import asyncio

import sounddevice as sd
from reactivex.subject import Subject

from server.core.stream_dsl import Stream, SubGroup, print_sink, turn_detector, whisper_stt
from server.core.stream_tts import KokoroTTSProvider, PlaybackState, tts_sink


async def run_mic_pipeline(args):
    audio_input = Subject()
    subs = SubGroup()
    playback_state = PlaybackState() if args.tts else None

    audio = Stream.source(audio_input, name="mic_audio")
    turn = audio | turn_detector()
    user_text = turn.segments | whisper_stt(mode="faster_whisper", model_size=args.model_size)

    user_text.to(print_sink(prefix="User: "), name="print_user_text", subs=subs)

    if args.tts:
        user_text.to(
            tts_sink(
                KokoroTTSProvider(),
                interrupts=turn.signals,
                name="kokoro_tts",
                state=playback_state,
            ),
            name="kokoro_tts",
            subs=subs,
        )
        print("TTS enabled. Mic packets are discarded during playback.")


    # Pickup audio from microphone and feed it to the pipeline (via audio_input Subject)
    
    def callback(indata, frames, time_info, status):
        if status:
            print(f"mic status: {status}")
        if playback_state is not None and playback_state.is_playing():
            return
        samples = indata[:, 0].copy()
        audio_input.on_next(samples)

    print("Listening. Speak, then pause for transcription. Press Ctrl-C to stop.")
    try:
        with sd.InputStream(
            samplerate=args.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=args.block_size,
            callback=callback,
        ):
            while True:
                await asyncio.sleep(0.25)
    finally:
        audio_input.on_completed()
        subs.dispose()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the basic DSL pipeline from local microphone input.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--block-size", type=int, default=8000)
    parser.add_argument("--model-size", default="tiny")
    parser.add_argument("--tts", action="store_true", help="Speak transcribed text with Kokoro TTS.")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(run_mic_pipeline(parse_args()))
    except KeyboardInterrupt:
        print("\nStopped.")
