from agno.agent import Agent
from agno.models.google import Gemini
from server.utils import rx_to_async_iter, send_text_to_client
from server.turndet_stt import run_stt

'''
A basic pipeline that
- takes in audio frames from browser (via webrtc)
- runs VAD and STT to transcribe (user:text)
- sends user:text to Gemini Agent, get response (assistant:text)
- sends user:text and assistant:text to client for display


'''

def create_agent():
    return Agent(
        name = "Memory Agent",
        description="You are a helpful agent that can solve math questions. Your input is obtained via STT so may have errors. Respond in a simple and concise manner.",
        #instructions=["Analyze the images carefully and give precise answers."],

        model=Gemini(id="gemini-2.0-flash"),
        add_history_to_messages=True,
        markdown=True,
        #debug_mode=True,
    )


from rx.subject import Subject

async def create_pipeline(pc, data_channels, audio_input, video_input, main_loop):
    '''
    Create a pipeline that takes in audio and video frames.
    Input audio -> vad-stt -> user:text -> send_to_client
    user:text -> agent -> assistant:text -> send_to_client
    todo: agent -> tts -> Output audio
    '''

    text_output = run_stt(audio_input)
    #text_output.subscribe(lambda text: print(f"Transcription: {text}"))
    text_gen = rx_to_async_iter(text_output)

    agent = create_agent()
    async for text in text_gen:
        print(f"User: {text}")
        send_text_to_client(text, data_channels, main_loop, channel="server_text", role="user")
        response = await agent.arun(text)
        print(f"Assistant: {response.content}")
        send_text_to_client(response.content, data_channels, main_loop, channel="server_text", role="assistant")



#

if __name__ == "__main__":
    from server.server_asyncio import Server
    
    # Create and run the server
    server = Server(create_pipeline=create_pipeline)
    server.run()



    
