from typing import Callable, Dict, Set, Any, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from aiortc import RTCPeerConnection, RTCSessionDescription
import uvicorn

from .core.utils import rx_Subject as Subject
from .setup_tracks import pc_session_setup

DEFAULT_CLIENT_HTML_PATH = Path(__file__).parent.parent / "client/client.html"

class Server:
    def __init__(self, run_session: Callable, client_html_path: Path = DEFAULT_CLIENT_HTML_PATH, config: Dict = None):
        """Initialize the WebRTC server with a session runner.
        
        Args:
            run_session: Async function that owns one SessionContext lifecycle.
            client_html_path: Path to the client HTML file.
            config: Configuration dictionary.
        """
        if config is None:
            config = {}
            
        self.run_session = run_session
        self.pcs: Set[RTCPeerConnection] = set()
        self.app = FastAPI()
        self.client_dir = client_html_path.parent
        self.config = config
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Set up the FastAPI routes."""
        print('Setting up routes...')
        
        # API routes
        self.app.post("/offer")(self.offer_handler)
        
        # Static files and catch-all route for SPA
        @self.app.get("/{full_path:path}")
        async def catch_all(full_path: str):
            # Handle root path
            if not full_path:
                return FileResponse(self.client_dir / "client.html")
                
            # Resolve the requested path
            file_path = (self.client_dir / full_path).resolve()
            
            # Security check: prevent directory traversal
            if not file_path.is_relative_to(self.client_dir) or not file_path.exists():
                if full_path == "client.html":
                    return FileResponse(self.client_dir / "client.html")
                raise HTTPException(status_code=404, detail="File not found")
                
            # Serve the file if it exists
            if file_path.is_file():
                return FileResponse(file_path)
                
            # For SPA routing, serve index.html for any non-file paths
            return FileResponse(self.client_dir / "client.html")
    
    async def offer_handler(self, request: Request):
        """Handle WebRTC offer and set up media processing pipeline."""
        pc = None
        try:
            params = await request.json()
            pc = pc_session_setup(self.run_session, self.config)
            self.pcs.add(pc)
            
            offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            return {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            }
            
        except Exception as e:
            print(f"Error in offer handler: {e}")
            if pc is not None:
                self.pcs.discard(pc)
                await pc.close()
            raise HTTPException(status_code=500, detail=str(e))
    
    async def on_shutdown(self):
        """Handle application shutdown."""
        print("Shutting down...")
        # Close all peer connections
        for pc in self.pcs:
            await pc.close()
        self.pcs.clear()
    
    def run(self, host: str = "0.0.0.0", port: int = 9000):
        """Run the FastAPI server."""
        # Add shutdown event handler
        @self.app.on_event("shutdown")
        async def shutdown_event():
            await self.on_shutdown()
            
        # Start the server
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            ssl_keyfile=self.config.get("ssl_keyfile"),
            ssl_certfile=self.config.get("ssl_certfile")
        )

# For running directly with python -m
if __name__ == "__main__":
    # Example usage
    async def run_session(session):
        """Example session runner."""
        await session.wait_until_ready()
        print("Session ready")
    
    server = Server(run_session)
    server.run()
