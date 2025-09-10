import pyaudio
from google import genai
from google.genai import types

def tts_gemini(text, model="gemini-2.5-flash-preview-tts", voice="Kore"):
    # PCM format info (matches your original wave_file parameters)
    channels = 1
    rate = 24000
    sample_width = 2  # bytes per sample (16-bit audio)

    client = genai.Client()

    response = client.models.generate_content(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice,
                    )
                )
            ),
        )
    )

    # Get PCM bytes from Gemini
    pcm_data = response.candidates[0].content.parts[0].inline_data.data
    return pcm_data

def play_pcm_data(pcm_data):

    # Play directly through speakers
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(sample_width),
                    channels=channels,
                    rate=rate,
                    output=True)

    stream.write(pcm_data)

    stream.stop_stream()
    stream.close()
    p.terminate()

def gemini_tts_play(text, model="gemini-2.5-flash-preview-tts", voice="Kore"):
    pcm_data = tts_gemini(text, model, voice)
    play_pcm_data(pcm_data)


import asyncio
import pyaudio

# Audio settings
CHANNELS = 1
RATE = 24000
SAMPLE_WIDTH = 2  # 16-bit PCM

# Create client
client = genai.Client()

async def tts_gemini_stream(text, model="gemini-2.5-flash-preview-tts", voice="Kore"):
    """Stream Gemini TTS output and play it in real-time."""

    # Open PyAudio stream
    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(SAMPLE_WIDTH),
        channels=CHANNELS,
        rate=RATE,
        output=True
    )

    loop = asyncio.get_event_loop()

    # Use async streaming API
    stream_gen = await client.aio.models.generate_content_stream(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice,
                    )
                )
            ),
        ),
    )

    async for event in stream_gen:
        # Some events are metadata, some are audio
        if (hasattr(event, "candidates") 
            and event.candidates
            and event.candidates[0].content.parts
            and event.candidates[0].content.parts[0].inline_data):

            pcm_chunk = event.candidates[0].content.parts[0].inline_data.data
            # Feed PCM chunk to PyAudio in executor so it won’t block asyncio
            await loop.run_in_executor(None, stream.write, pcm_chunk)

    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()




def kokoro_tts_play(text):
    import soundfile as sf
    import sounddevice as sd
    from kokoro_onnx import Kokoro

    kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    samples, sample_rate = kokoro.create(
       text, voice="af_sarah", speed=1.0, lang="en-us"
    )
    print(samples.shape)
    print(sample_rate)
    sd.play(samples, sample_rate)
    sd.wait()
    #sf.write("output.wav", samples, sample_rate)
    
# async def kokoro_tts_play_async(text, voice="af_sarah", speed=1.0, lang="en-us"):
#     from kokoro_onnx import Kokoro
#     import numpy as np
#     import sounddevice as sd
#     kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

#     loop = asyncio.get_running_loop()

#     # Run blocking TTS in a thread
#     samples, sample_rate = await loop.run_in_executor(
#         None, kokoro.create, text, voice, speed, lang
#     )
#     samples = np.clip(samples, -1.0, 1.0)  # avoid clipping

#     if samples.ndim == 1:  # mono → stereo
#         samples = np.stack([samples, samples], axis=1)

#     # Play audio in a thread
#     await loop.run_in_executor(None, sd.play, samples, sample_rate)
#     await loop.run_in_executor(None, sd.wait)

from kokoro_onnx import Kokoro
import sounddevice as sd

kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")



# --- sync helper (blocking)
async def _tts_and_play_kokoro(text, voice="af_sarah", speed=1.0, lang="en-us"):
    # Generate audio
    samples, sr = kokoro.create(text, voice=voice, speed=speed, lang=lang)
    sd.play(samples, sr)
    sd.wait()


    return samples, sr


# --- async safe wrapper
async def tts_and_play_kokoro(text, voice="af_sarah", speed=1.0, lang="en-us"):
    # loop = asyncio.get_event_loop()
    # return await loop.run_in_executor(
    #     None, _tts_and_play_kokoro, text, voice, speed, lang
    # )
    await _tts_and_play_kokoro(text, voice, speed, lang)


if __name__ == "__main__":
    asyncio.run(tts_and_play_kokoro("Hello there! This is real-time streaming playback from Kokoro."))
    #kokoro_tts_play("Hello there! This is real-time streaming playback from Kokoro.")

    #pcm_data = tts_gemini("Hello, how are you? I will solve all the math problems you can think of.", model="gemini-2.0-flash-live-001")
    #play_pcm_data(pcm_data)
