import asyncio
import json
import numpy as np
from typing import Callable, Dict, Set, Any, Awaitable
from pathlib import Path
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc import MediaStreamError
from .audio_utils import convert_and_resample_frame
from .audio_utils import is_active_speaker
from .utils import rx_Subject as Subject # for input audio/video subjects

DEFAULT_CLIENT_HTML_PATH = Path(__file__).parent.parent / "client/client.html"

class Server:
    def __init__(self, create_pipeline: Callable, client_html_path: Path = DEFAULT_CLIENT_HTML_PATH, config: Dict = {}):
        """Initialize the WebRTC server with a pipeline creation function.
        
        Args:
            create_pipeline: A function that creates and returns a media processing pipeline.
        """

        self.create_pipeline = create_pipeline
        self.pcs: Set[RTCPeerConnection] = set()
        self.app = web.Application()
        self._setup_routes(client_html_path)
        self.app.on_shutdown.append(self.on_shutdown)

        self.config = config
        self.rms_thresh = config.get("rms_thresh", 0.02)
        self.debug = config.get("debug", False)
        self.input_audio_buffer_size = config.get("input_audio_buffer_size", 8000)
        self.input_video_sample_interval = config.get("input_video_sample_interval", 500)
    
    def _setup_routes(self, client_html_path):
        """Set up the web application routes."""
        print('setting up routes..')
        self.app.router.add_post("/offer", self.offer_handler)
        self.app.router.add_get("/", lambda request: web.FileResponse(client_html_path))
    
    async def handle_audio_track(self, pc, track: MediaStreamTrack, speech_turn_input, stop_event, 
                                audio_buffer_size, rms_thresh):
        """Handle incoming audio track and process it through the pipeline."""

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
                    active = is_active_speaker(chunk_out, sr, rms_thresh=rms_thresh, debug=self.debug)
                    if active:
                        speech_turn_input.on_next(chunk_out)
                    buffer = buffer[bsize:]
                    
            except MediaStreamError:
                print("Audio track ended")
                stop_event.set()
                break
            except Exception as e:
                print(f"Error processing audio: {e}")
                stop_event.set()
                break
        
        # Process any remaining audio
        if len(buffer) > 0 and not stop_event.is_set():
            speech_turn_input.on_next(buffer)
        
        print("Audio processing stopped")

    async def handle_video_track(self, pc, track: MediaStreamTrack, video_obs_input, stop_event, interval):
        """Handle incoming video track. Sample at interval, send to observer"""
        frame_count = 0
        
        while not stop_event.is_set():
            try:
                frame = await track.recv()
                frame_count += 1
                
                # Log frame info occasionally
                if frame_count % interval == 0:
                    print(f"Received video frame {frame_count}: {frame.width}x{frame.height}")
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

    async def offer_handler(self, request):
        """Handle WebRTC offer and set up media processing pipeline."""
        params = await request.json()
        pc = RTCPeerConnection()
        self.pcs.add(pc)
        
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
        asyncio.create_task(self.create_pipeline(pc, data_channels, audio_input, video_input, main_loop))
        
        def on_track(track: MediaStreamTrack):
            print(f"Track received: {track.kind}")
            if track.kind == "audio":
                asyncio.create_task(
                    self.handle_audio_track(pc, track, audio_input, stop_event, self.input_audio_buffer_size, self.rms_thresh)
                )
            elif track.kind == "video":
                asyncio.create_task(self.handle_video_track(pc, track, video_input, stop_event, self.input_video_sample_interval))
        
        pc.on("track", on_track)
        
        async def on_connectionstatechange():
            print(f"Connection state changed to: {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                stop_event.set()
                self.pcs.discard(pc)
                await pc.close()
        
        pc.on("connectionstatechange", on_connectionstatechange)
        
        try:
            offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            return web.json_response({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            })
        except Exception as e:
            print(f"Error in offer handler: {e}")
            stop_event.set()
            self.pcs.discard(pc)
            await pc.close()
            raise web.HTTPInternalServerError(text=str(e))

    async def on_shutdown(self):
        """Handle application shutdown."""
        print("Shutting down...")
        # Close all peer connections
        for pc in self.pcs:
            await pc.close()
        self.pcs.clear()
    
    def run(self, host="localhost", port=9000):
        """Run the web server."""
        web.run_app(self.app, host=host, port=port)


