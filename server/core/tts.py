import pyaudio
from google import genai
from google.genai import types
import numpy as np
import sounddevice as sd
import asyncio
import pyaudio
import time


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

from kokoro_onnx import Kokoro
import sounddevice as sd
import threading

# Initialize the global variable
cancel_requested = False
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")



# --- sync helper (blocking)
async def _tts_and_play_kokoro(text, voice="af_sarah", speed=1.0, lang="en-us"):
    # Generate audio
    samples, sr = kokoro.create(text, voice=voice, speed=speed, lang=lang)
    #sd.play(samples, sr)
    #sd.wait()

    chunk_size = int(sr * 0.5)  # 0.5 second chunks

    def _play():
        global cancel_requested
        for start in range(0, len(samples), chunk_size):
            if cancel_requested:
                print("Playback canceled!")
                break
            end = start + chunk_size
            chunk = samples[start:end]
            sd.play(chunk, sr, blocking=True)

    threading.Thread(target=_play, daemon=True).start()


    return samples, sr

def cancel_tts():
    global cancel_requested
    cancel_requested = True
    sd.stop()   # immediately stop playback

# --- async safe wrapper
def tts_and_play_kokoro(text, voice="af_sarah", speed=1.0, lang="en-us"):
    """Run TTS in a separate task without blocking."""
    try:
        # Try to get the running loop first
        loop = asyncio.get_running_loop()
        # If we get here, we're in an async context
        return loop.create_task(_tts_and_play_kokoro(text, voice, speed, lang))
    except RuntimeError:
        # No running loop, create a new one and run it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_tts_and_play_kokoro(text, voice, speed, lang))
        finally:
            loop.close()


async def play_tts_sequence(texts):
    for i, text in enumerate(texts, 1):
        print(f"Playing message {i}/{len(texts)}")
        await _tts_and_play_kokoro(f"{i}. {text}")
        
        # Add a small pause between sentences (in seconds)
        await asyncio.sleep(0.2)
    
    print("Finished playing all messages")


def _print_tts_metrics(timings, chunk_count):
    """Print TTS performance metrics.
    
    Args:
        timings (dict): Dictionary containing timing metrics with keys:
            - start: Timestamp when TTS started
            - first_buffer_received: Timestamp when first audio chunk was received
            - end: Timestamp when TTS completed
        chunk_count (int): Number of audio chunks processed
    """
    if timings['first_buffer_received'] is None:
        print("Error: No audio data was received")
        return
        
    time_to_first_buffer = (timings['first_buffer_received'] - timings['start']) * 1000  # in ms
    total_time = (timings['end'] - timings['start']) * 1000  # in ms
    streaming_duration = (timings['end'] - timings['first_buffer_received']) * 1000  # in ms
    
    print("\n--- TTS Performance Metrics ---")
    print(f"Time to first buffer: {time_to_first_buffer:.2f} ms")
    print(f"Total processing time: {total_time:.2f} ms")
    print(f"Streaming duration: {streaming_duration:.2f} ms")
    print(f"Number of chunks: {chunk_count}")
    print("---  ---\n")


async def tts_kokoro_stream_async(text, interrupt_event):
    """Non-blocking version of tts_kokoro_stream using asyncio."""
    from openai import AsyncOpenAI
    import sounddevice as sd
    import numpy as np
    
    # Initialize timing metrics
    timings = {
        'start': time.perf_counter(),
        'first_buffer_received': None,
        'end': None
    }
    
    samplerate = 22050  
    blocksize = 1024
    chunk_count = 0

    # Initialize audio stream
    stream = sd.OutputStream(samplerate=samplerate, channels=1, dtype='int16')
    stream.start()
    
    try:
        client = AsyncOpenAI(
            base_url="http://localhost:8880/v1", 
            api_key="not-needed"
        )

        async with client.audio.speech.with_streaming_response.create(
            model="kokoro",
            voice="af_sky+af_bella",
            input=text,
            response_format="pcm",
        ) as response:
            first_chunk = True
            
            async for chunk in response.iter_bytes(chunk_size=blocksize):
                if first_chunk:
                    timings['first_buffer_received'] = time.perf_counter()
                    first_chunk = False
                
                if interrupt_event.is_set():
                    break
                    
                audio_block = np.frombuffer(chunk, dtype=np.int16)
                #stream.write(audio_block)
                # Run the blocking write in a thread
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, stream.write, audio_block)
                chunk_count += 1
                
    except Exception as e:
        print(f"Error during TTS streaming: {e}")
    finally:
        # Ensure resources are cleaned up
        stream.stop()
        stream.close()
        
        # Record end time and print metrics
        timings['end'] = time.perf_counter()
        _print_tts_metrics(timings, chunk_count)

async def tts_kokoro_sequence_async(texts, speech_signals=None):
    interrupt_event = asyncio.Event()

    def on_signal(event):
        nonlocal interrupt_event
        if (str(event) == "SPEECH_START"): interrupt_event.set()
        if (str(event) == "SPEECH_END"): interrupt_event.clear() #do we need this?
        print(f"INTERRUPT Signal received: {event}, interrupt: {interrupt_event}")

    if speech_signals:
        speech_signals.subscribe(on_signal)

    async def list_to_async(lst):
        for item in lst:
            await asyncio.sleep(0)  # yield control
            yield item

    async for text in list_to_async(texts):
        if interrupt_event.is_set():
            break
        await tts_kokoro_stream_async(text, interrupt_event)
        await asyncio.sleep(0.001)
    interrupt_event.clear()

def tts_kokoro_stream(text):
    """Synchronous wrapper for the async version.
    
    This is maintained for backward compatibility and simple scripts.
    """
    return asyncio.run(tts_kokoro_stream_async(text))

if __name__ == "__main__":
    import time
    import asyncio
    texts = [
        "Hello there! This is the first sentence.",
        "This is the second sentence in the sequence.",
        "And this is the final sentence in our sequence."
    ]
    #asyncio.run(play_tts_sequence(texts))
    #tts_and_play_kokoro(texts[0])
    #time.sleep(2)    
    #cancel_tts()

    asyncio.run(tts_kokoro_sequence_async(texts))

    #pcm_data = tts_gemini("Hello, how are you? I will solve all the math problems you can think of.", model="gemini-2.0-flash-live-001")
    #play_pcm_data(pcm_data)
