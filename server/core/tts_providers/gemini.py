import asyncio
from typing import Optional


CHANNELS = 1
RATE = 24000
SAMPLE_WIDTH = 2


def tts_gemini(text, model="gemini-2.5-flash-preview-tts", voice="Kore"):
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        ),
    )
    return response.candidates[0].content.parts[0].inline_data.data


def play_pcm_data(pcm_data, channels=CHANNELS, rate=RATE, sample_width=SAMPLE_WIDTH):
    import pyaudio

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(sample_width), channels=channels, rate=rate, output=True)
    stream.write(pcm_data)
    stream.stop_stream()
    stream.close()
    p.terminate()


def gemini_tts_play(text, model="gemini-2.5-flash-preview-tts", voice="Kore"):
    play_pcm_data(tts_gemini(text, model, voice))


async def tts_gemini_stream(text, model="gemini-2.5-flash-preview-tts", voice="Kore"):
    from google import genai
    from google.genai import types
    import pyaudio

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(SAMPLE_WIDTH), channels=CHANNELS, rate=RATE, output=True)
    loop = asyncio.get_event_loop()
    client = genai.Client()
    stream_gen = await client.aio.models.generate_content_stream(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        ),
    )
    try:
        async for event in stream_gen:
            if (
                hasattr(event, "candidates")
                and event.candidates
                and event.candidates[0].content.parts
                and event.candidates[0].content.parts[0].inline_data
            ):
                await loop.run_in_executor(None, stream.write, event.candidates[0].content.parts[0].inline_data.data)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


class GeminiTTSProvider:
    def __init__(self, model: str = "gemini-2.5-flash-preview-tts", voice: str = "Kore"):
        self.model = model
        self.voice = voice

    async def speak(self, text: str, interrupt_event: asyncio.Event) -> None:
        if interrupt_event.is_set():
            return
        await tts_gemini_stream(text, model=self.model, voice=self.voice)

