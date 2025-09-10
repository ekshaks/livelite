from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pathlib
from fastapi.websockets import WebSocketDisconnect


app = FastAPI()

# Serve frontend files
frontend_dir = pathlib.Path(__file__).parent.parent / "client"
assert frontend_dir.exists(), f"Frontend directory not found: {frontend_dir}"

app.mount("/client", StaticFiles(directory=frontend_dir), name="client")

@app.get("/", response_class=FileResponse)
async def index():
    return FileResponse(frontend_dir / "client_full.html")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            await ws.send_text(f"Echo: {msg}")
    except WebSocketDisconnect:
        print("WebSocket disconnected")  # cleanup only
    except Exception:
        print("WebSocket error")
        await ws.close()
        # no need: await ws.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_fastapi_ws:app", host="127.0.0.1", port=8200, reload=True, workers=1)
