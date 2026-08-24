import json
import logging

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pathlib
from fastapi.websockets import WebSocketDisconnect

from muapps.chief.companion import classify_companion
from server.core.voice_protocol import VoicePipeline, VoicePrincipal, VoiceSession


app = FastAPI()
_pipeline: VoicePipeline | None = None
logger = logging.getLogger("uvicorn.error")

# Serve frontend files
frontend_dir = pathlib.Path(__file__).parent.parent / "client"
assert frontend_dir.exists(), f"Frontend directory not found: {frontend_dir}"

app.mount("/client", StaticFiles(directory=frontend_dir), name="client")

@app.get("/", response_class=FileResponse)
async def index():
    return FileResponse(frontend_dir / "websocket/client.html")


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


def _voice_pipeline() -> VoicePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = VoicePipeline(companion_for_text=classify_companion)
    return _pipeline


@app.on_event("startup")
async def warm_voice_pipeline() -> None:
    pipeline = _voice_pipeline()
    if isinstance(pipeline, VoicePipeline):
        logger.info("voice pipeline warming stt_loading=%s", pipeline.stt.is_loading())
        await pipeline.wait_ready()
        logger.info(
            "voice pipeline ready stt_mode=%s stt_model=%s",
            pipeline.stt.mode,
            pipeline.stt.model_size,
        )


@app.on_event("shutdown")
async def close_voice_pipeline() -> None:
    if _pipeline is not None and isinstance(_pipeline, VoicePipeline):
        await _pipeline.aclose()


@app.websocket("/voice")
async def voice_endpoint(ws: WebSocket):
    # ISC R1/I_AUTH: this server is bound to 127.0.0.1; remote use needs authentication.
    await ws.accept()
    logger.info("voice socket accepted client=%s", ws.client)
    session = VoiceSession(VoicePrincipal("local", "Local"), _voice_pipeline(), ws.send_json, ws.send_bytes)
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                logger.info("voice socket disconnected client=%s", ws.client)
                break
            if message.get("text") is not None:
                event = json.loads(message["text"])
                log = logger.debug if event.get("type") == "turn.start" else logger.info
                log("voice event received type=%s turn_id=%s", event.get("type"), event.get("turn_id"))
                await session.handle_json(event)
            elif message.get("bytes") is not None:
                logger.debug("voice PCM received bytes=%d turn_id=%s", len(message["bytes"]), session.turn_id)
                await session.handle_pcm16(message["bytes"])
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("voice protocol error: %s", exc)
        await ws.send_json({"type": "error", "text": "invalid voice protocol event"})
    except WebSocketDisconnect:
        logger.info("voice socket disconnected client=%s", ws.client)
    except Exception:
        logger.exception("voice socket failed")
        raise
    finally:
        logger.info("voice session cleanup turn_id=%s", session.turn_id)
        await session.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_fastapi_ws:app", host="127.0.0.1", port=8200, reload=True, workers=1)
