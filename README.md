# mulive

Real-time voice/video agent framework over WebRTC. Browser streams audio and video to a Python backend, which runs VAD → STT → LLM/VLM → TTS and returns transcript + responses to the browser via a data channel.

## Quickstart

```bash
pip install -r requirements.txt

# Run the multimodal agent (primary app)
python -m apps.multimodal_agent

# Or with auto-reload on file changes
watchfiles --filter python "python -m apps.multimodal_agent"
```

Open `https://localhost:9000` (HTTPS required for camera/mic access). Click **Start Streaming** and speak. The agent responds in text and via local speakers.

> Note: the server uses a self-signed cert in `server/certs/`. You'll need to accept the browser warning on first load.

## Folder structure

```
apps/           Agent pipelines (runnable apps)
client/         Browser-side WebRTC client (HTML + JS)
server/         WebRTC server + core processing modules
  core/         VAD, STT, TTS, LLM utils, audio/video utils
docs/           Architecture reference and task backlog
data/           Sample audio files for testing
models/         Local model files
Kokoro-FastAPI/ Local Kokoro TTS server
```

## Apps

| App | Status | Description |
|-----|--------|-------------|
| `apps/multimodal_agent.py` | ✅ Runnable | Audio + video → VLM agent → TTS with interruption |
| `apps/basic_pipeline.py` | ⚠️ Needs refactor | Audio-only → LLM → text back to browser |
| `apps/spell/app.py` | ✅ Runnable | Camera spelling game with VLM handwriting transcription |
| `apps/v2v.py` | 🚧 Scaffold | Voice-to-voice variant of spelling tutor |

## Core pipeline

```
Browser mic/cam
    ↓  WebRTC
server/setup_tracks.py       ← per-session subjects
    ↓
core/audio_utils.py          ← resample to 16kHz mono
core/turndet.py              ← Silero VAD → speech turns
core/stt.py                  ← Whisper (faster-whisper or mlx)
    ↓ user text
apps/<pipeline>.py           ← LLM / VLM agent
    ↓ assistant text
client (data channel)        ← text rendered in browser
core/tts.py                  ← Kokoro TTS → local speakers
```

## TTS backends

- **Kokoro** (default) — local ONNX model via OpenAI-compatible API (`localhost:8880`). Start with `Kokoro-FastAPI/`.
- **Gemini TTS** — cloud, via `google-genai`.

## Docs

- [Architecture reference](docs/architecture.md) — component design, data flow, implementation notes
- [Open problems](docs/private/open_problems.md) — known runtime and product gaps
- [Test plan](docs/private/test_plan.md) — application verification strategy
