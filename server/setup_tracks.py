from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
import asyncio

from .core.audio_utils import convert_and_resample_frame
from .core.audio_utils import is_active_speaker
from .core.utils import rx_Subject as Subject # for input audio/video subjects
import numpy as np

async def setup_audio_track(pc, track: MediaStreamTrack, speech_turn_input, stop_event, config):
    """Handle incoming audio track and process it through the pipeline."""
    rms_thresh = config.get("rms_thresh", 0.02)
    debug = config.get("debug", False)
    audio_buffer_size = config.get("input_audio_buffer_size", 8000)

    buffer = np.array([], dtype=np.int16)
    bsize = audio_buffer_size #audio buffer size, for sending to pipeline
    sr = 16000
    
    while not stop_event.is_set():
        try:
            frame = await track.recv()
            chunk = convert_and_resample_frame(frame, target_sample_rate=sr)
            buffer = np.concatenate([buffer, chunk])
            
            # Process complete chunks
            while len(buffer) >= bsize:
                chunk_out = buffer[:bsize]
                active = is_active_speaker(chunk_out, sr, rms_thresh=rms_thresh, debug=debug, 
                            filter_gender=config.get("filter_gender", None))
                if active:
                    speech_turn_input.on_next(chunk_out)
                buffer = buffer[bsize:]
                
        except MediaStreamError:
            print("Audio track ended")
            stop_event.set()
            break
        except Exception as e:
            import traceback
            print(f"Error processing audio: {e}")
            print(f"StackTrace: {traceback.format_exc()}")

            stop_event.set()
            break
    
    # Process any remaining audio
    if len(buffer) > 0 and not stop_event.is_set():
        speech_turn_input.on_next(buffer)
    
    print("Audio processing stopped")

async def setup_video_track(pc, track: MediaStreamTrack, video_obs_input, stop_event, config):
    
    """Handle incoming video track. Sample at interval, send to observer"""
    input_video_sample_interval = config.get("input_video_sample_interval", 500)
    
    frame_count = 0
    
    while not stop_event.is_set():
        try:
            frame = await track.recv()
            frame_count += 1
            
            # Log frame info occasionally
            if frame_count % input_video_sample_interval == 0:
                print('sending frame')
                video_obs_input.on_next(frame)
                
        except MediaStreamError:
            print("Video track ended")
            stop_event.set()
            break
        except Exception as e:
            print(f"Error processing video: {e}")
            stop_event.set()
            break
    
    print("Video processing stopped")

def pc_pipeline_setup(create_pipeline, config):
    
    pc = RTCPeerConnection()
    
    stop_event = asyncio.Event()
    data_channels = {}

    def on_datachannel(channel):
        print(f"Data channel received: {channel.label}")
        data_channels[channel.label] = channel

        @channel.on("message")
        def on_message(message):
            print("Message from client:", message)
            response = json.dumps({"role": "assistant", "content": "Hello from Assistant!"})
            data_channels[channel.label].send(response)
    
    pc.on("datachannel", on_datachannel)
    
    main_loop = asyncio.get_running_loop()
    audio_input, video_input = Subject(), Subject()
    # audio/video input -> main pipeline
    asyncio.create_task(create_pipeline(pc, data_channels, audio_input, video_input, main_loop))
    
    def on_track(track: MediaStreamTrack):
        print(f"Track received: {track.kind}")
        # audio/video track --- (initial filters)--> audio/video input
        if track.kind == "audio":
            asyncio.create_task(
                setup_audio_track(pc, track, audio_input, stop_event, config)
            )
        elif track.kind == "video":
            asyncio.create_task(setup_video_track(pc, track, video_input, stop_event, config))
    
    pc.on("track", on_track)
    
    async def on_connectionstatechange():
        print(f"Connection state changed to: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            stop_event.set()
            self.pcs.discard(pc)
            await pc.close()
    
    pc.on("connectionstatechange", on_connectionstatechange)
    return pc
