from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.groq import Groq

from server.core.utils import rx_to_async_iter, send_text_to_client
from server.core.stt import run_stt_rx

from server.core.llm_utils import call_vlm_agent
from server.core.tts import gemini_tts_play, tts_and_play_kokoro, tts_kokoro_sequence_async
from server.core.utils import timeit

import yaml
from pathlib import Path

'''
A Math solving Agent pipeline:
- takes in audio frames from browser (via webrtc)
- runs VAD and STT to transcribe -> (user:text)
- samples video frames -> (last_frame)
- sends user:text and last_frame to Gemini Agent, get response (assistant:text)
- sends user:text and assistant:text to client for display


'''

PROMPTS_FILE = Path(__file__).parent / "prompts.yml"

def create_agent(model_id, id='math_helper'):
    with open(PROMPTS_FILE) as f:
        prompts = yaml.safe_load(f)
        instructions = prompts[id]['instructions']
        description = prompts[id]['description']

    if 'google' in model_id:
        model = Gemini(id=model_id.split(':')[-1])
    elif 'groq' in model_id:
        model = Groq(id=model_id.split(':')[-1])
    else:
        raise ValueError(f"Unknown model: {model_id}")
    
    return Agent(
        name = "Math Helper",
        description=description,
        instructions=[instructions],

        model=model,
        add_history_to_messages=True,
        markdown=True,
        #debug_mode=True,
    )

# Sofia bought 3 notebooks for $2.50 each and 2 pens for $1.20 each.
# How much did she spend in total?


@timeit(name="Call_LLM")
async def call_llm(agent, text, last_frame, mode='a'):
    if mode == 'av':
        response = await call_vlm_agent(agent, text, last_frame)
        ass_text = response.content
    else:
        response = await agent.arun(text)
        ass_text = response.content
    return ass_text

async def create_pipeline(pc, data_channels, audio_input, video_input, main_loop, mode='a'):
    '''
    Create a pipeline that takes in audio and video frames.
    Input audio -> vad-stt -> user:text -> send_to_client
    user:text -> agent -> assistant:text -> send_to_client
    todo: agent -> tts -> Output audio
    '''

    text_output, speech_signals = run_stt_rx(audio_input)
    speech_signals.subscribe(lambda event: print(f"**** Speech event: {event}"))

    #text_output.subscribe(lambda text: print(f"Transcription: {text}"))

    last_frame = None

    def update_last(x):
        nonlocal last_frame
        last_frame = x
        print(f"Received video frame : {x.width}x{x.height}, update last frame")
        #print('updated last frame')

    video_input.subscribe(update_last)  # keeps latest frame
    
    model_id = 'google:gemini-2.5-flash-lite'
    #model_id1 = 'groq:openai/gpt-oss-120b' #'llama-3.3-70b-versatile'
    model_id2 = 'groq:meta-llama/llama-4-scout-17b-16e-instruct' #visual
    model_id3 = 'groq:meta-llama/llama-4-maverick-17b-128e-instruct'
    agent = create_agent(model_id, id='visual_solver')
    agent.model = Groq(id=model_id2.split(':')[-1])
    
    async for text in rx_to_async_iter(text_output):
        print(f"User: {text}")
        send_text_to_client(text, data_channels, main_loop, channel="server_text", role="user")
        ass_text = ""
        #print('call vlm agent', text, last_frame)
        if not text.strip():
            continue
        
        ass_text = await call_llm(agent, text, last_frame, mode)


        if ass_text.strip():
            print(f"Assistant: {ass_text}")
            send_text_to_client(ass_text, data_channels, main_loop, channel="server_text", role="assistant")
            #tts_and_play_kokoro(ass_text)
            await tts_kokoro_sequence_async([ass_text], speech_signals=speech_signals)


def test_vlm_call():
    import asyncio
    import numpy as np
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    #text = "what do you see in this image?"
    text = "Sofia bought 3 notebooks for $2.50 each and 2 pens for $1.20 each. How much did she spend in total?"
    asyncio.run(vlm_agent2(text, arr))

if __name__ == "__main__":
    from server.server_fastapi_webrtc import Server
    
    #test_vlm_call()
    #quit()
    # Create and run the server
    config = dict(
        debug=True,
        rms_thresh=0.025, #0.02,
        input_video_sample_interval=100,
        filter_gender=None #'male'
    )
    server_dir = Path(__file__).parent.parent / "server"
    assert server_dir.exists(), f"Server directory not found: {server_dir}"
    config.update({
        "ssl_keyfile": server_dir / "certs/key.pem",
        "ssl_certfile": server_dir / "certs/cert.pem"
    })
    mode = 'av'
    #mode = 'a'
    server = Server(create_pipeline=lambda *args: create_pipeline(*args, mode=mode),
                    config=config)
    server.run()



    
