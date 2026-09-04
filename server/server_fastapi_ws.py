import asyncio
import json
import logging

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pathlib
from fastapi.websockets import WebSocketDisconnect

from server.core.ws_voice_protocols import STT_LLM_TTS_Flow, VoiceHandler, VoicePrincipal, VoiceSession

logger = logging.getLogger("uvicorn.error")
_pipeline: STT_LLM_TTS_Flow | None = None  # compatibility hook for embedders/tests

def create_app(
    config: dict | None = None,
    voice_handler: VoiceHandler | None = None,
    run_session=None,
) -> FastAPI:
    """Create the generic voice WebSocket application.

    The transport server uses the generic STT_LLM_TTS_Flow defaults unless an
    application supplies a complete VoiceHandler.
    """
    app = FastAPI()
    pipeline = voice_handler
    frontend_dir = pathlib.Path(__file__).parent.parent / "client"
    assert frontend_dir.exists(), f"Frontend directory not found: {frontend_dir}"
    app.mount("/client", StaticFiles(directory=frontend_dir), name="client")

    def voice_pipeline() -> STT_LLM_TTS_Flow:
        nonlocal pipeline
        global _pipeline
        if pipeline is not None:
            return pipeline
        if _pipeline is not None:
            return _pipeline
        if pipeline is None:
            pipeline = STT_LLM_TTS_Flow(config=config)
            _pipeline = pipeline
        return pipeline

    async def default_run_session(session):
        created = session.pipeline is None
        handler = session.pipeline or voice_handler or STT_LLM_TTS_Flow(
            config=config,
        )
        if session.pipeline is None:
            session.set_handler(handler)
        try:
            await session.wait_until_ready()
        finally:
            if created and hasattr(handler, "aclose"):
                await handler.aclose()

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
            logger.info("echo WebSocket disconnected")
        except Exception:
            logger.exception("echo WebSocket error")
            await ws.close()

    @app.on_event("startup")
    async def warm_voice_pipeline() -> None:
        if run_session is not None:
            prepare = getattr(run_session, "prepare", None)
            if prepare is not None:
                await prepare()
                logger.info("voice session runner prepared")
            return
        selected = voice_pipeline()
        if not hasattr(selected, "wait_ready"):
            return
        if hasattr(selected, "stt"):
            logger.info("voice pipeline warming stt_loading=%s", selected.stt.is_loading())
        await selected.wait_ready()
        logger.info("voice pipeline ready")

    @app.on_event("shutdown")
    async def close_voice_pipeline() -> None:
        if run_session is not None:
            close = getattr(run_session, "close", None)
            if close is not None:
                await close()
            return
        if run_session is None and pipeline is not None and hasattr(pipeline, "aclose"):
            await pipeline.aclose()

    @app.websocket("/voice")
    async def voice_endpoint(ws: WebSocket):
        # ISC R1/I_AUTH: this server is bound to 127.0.0.1; remote use needs authentication.
        await ws.accept()
        logger.info("voice socket accepted client=%s", ws.client)
        session = VoiceSession(
            VoicePrincipal("local", "Local"),
            None if run_session is not None else voice_pipeline(),
            ws.send_json,
            ws.send_bytes,
        )
        session_runner = asyncio.create_task(
            (run_session or default_run_session)(session),
            name="voice-session-runner",
        )
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
            session_runner.cancel()
            await asyncio.gather(session_runner, return_exceptions=True)

    # Test and embedding hooks, matching the aio WebRTC server's injected runner.
    app.state.get_voice_pipeline = voice_pipeline
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_fastapi_ws:app", host="127.0.0.1", port=8200, reload=True, workers=1)
