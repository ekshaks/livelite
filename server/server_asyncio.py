from typing import Callable, Dict, Set, Any, Awaitable
from pathlib import Path
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc import MediaStreamError

from .core.utils import rx_Subject as Subject # for input audio/video subjects
from .setup_tracks import pc_pipeline_setup

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

    
    def _setup_routes(self, client_html_path):
        """Set up the web application routes."""
        print('setting up routes..')
        self.app.router.add_post("/offer", self.offer_handler)
        
        client_dir = client_html_path.parent  # base directory containing index.html, js/, assets/, etc.

        async def serve_client_file(request):
            requested_path = request.match_info.get('path', '')
            if requested_path == '':
                # root → serve index.html
                return web.FileResponse(client_html_path)

            # Resolve full path and prevent directory traversal
            full_path = (client_dir / requested_path).resolve()
            if not full_path.is_relative_to(client_dir) or not full_path.exists():
                raise web.HTTPNotFound()
            return web.FileResponse(full_path)

        # catch-all route
        self.app.router.add_get("/{path:.*}", serve_client_file)
        
   
    async def offer_handler(self, request):
        """Handle WebRTC offer and set up media processing pipeline."""
        params = await request.json()
        pc = pc_pipeline_setup(self.create_pipeline, self.config)
        self.pcs.add(pc)
 
        
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
        web.run_app(self.app, host=host, port=port)



