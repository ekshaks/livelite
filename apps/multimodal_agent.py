from agno.agent import Agent
from agno.models.google import Gemini
from server.utils import rx_to_async_iter, send_text_to_client
from server.turndet_stt import run_stt
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

def create_agent():
    with open(PROMPTS_FILE) as f:
        prompts = yaml.safe_load(f)
        instructions = prompts['math_helper']['instructions']
        description = prompts['math_helper']['description']

    return Agent(
        name = "Math Helper",
        description=description,
        instructions=[instructions],

        model=Gemini(id="gemini-2.5-flash"),
        add_history_to_messages=True,
        markdown=True,
        #debug_mode=True,
    )

# Sofia bought 3 notebooks for $2.50 each and 2 pens for $1.20 each.
# How much did she spend in total?

async def vlm_agent(text, last_frame):
    from PIL import Image
    import google.generativeai as genai

    with open(PROMPTS_FILE) as f:
        system_prompt = yaml.safe_load(f)['vlm_math']

    model = genai.GenerativeModel('gemini-2.0-flash')

    # Create the parts of the prompt
    prompt_parts = [
        system_prompt,
        text,
    ]
    if last_frame is not None:
        pil_image = Image.fromarray(last_frame.to_ndarray(format='rgb24'))
        prompt_parts.append(pil_image)
    print('call vlm agent', prompt_parts)
    
    # Send the request to the API
    response = model.generate_content(prompt_parts)
    print(response.text)
    return response

async def create_pipeline(pc, data_channels, audio_input, video_input, main_loop, mode='visual'):
    '''
    Create a pipeline that takes in audio and video frames.
    Input audio -> vad-stt -> user:text -> send_to_client
    user:text -> agent -> assistant:text -> send_to_client
    todo: agent -> tts -> Output audio
    '''

    text_output = run_stt(audio_input)
    #text_output.subscribe(lambda text: print(f"Transcription: {text}"))
    text_gen = rx_to_async_iter(text_output)

    last_frame = None

    def update_last(x):
        nonlocal last_frame
        last_frame = x
        print('updated last frame')

    video_input.subscribe(update_last)  # keeps latest frame

    agent = create_agent()
    async for text in text_gen:
        print(f"User: {text}")
        send_text_to_client(text, data_channels, main_loop, channel="server_text", role="user")
        ass_text = ""
        #print('call vlm agent', text, last_frame)
        if not text.strip():
            continue
        
        if mode == 'visual':
            response = await vlm_agent(text, last_frame)
            ass_text = response.text
        else:
            response = await agent.arun(text)
            ass_text = response.content
        if ass_text.strip():
            print(f"Assistant: {ass_text}")
            send_text_to_client(ass_text, data_channels, main_loop, channel="server_text", role="assistant")



def test_vlm_call():
    import asyncio
    import numpy as np
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    #text = "what do you see in this image?"
    text = "Sofia bought 3 notebooks for $2.50 each and 2 pens for $1.20 each. How much did she spend in total?"
    asyncio.run(vlm_agent(text, arr))

if __name__ == "__main__":
    from server.server_asyncio import Server
    
    #test_vlm_call()
    # Create and run the server
    config = dict(
        debug=False,
        rms_thresh=0.02,
        input_video_sample_interval=100,
    )
    server = Server(create_pipeline=create_pipeline, config=config)
    server.run()



    
