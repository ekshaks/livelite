# mulive

Real-time voice/video agent framework over WebRTC. Browser streams audio and video to a Python backend, which runs VAD → STT → LLM/VLM → TTS and returns transcript + responses to the browser via a data channel.

## Quickstart

```bash
pip install -r requirements.txt

# Run the multimodal agent quickstart
python -m apps.quickstart.multimodal_agent

# Or with auto-reload on file changes
watchfiles --filter python "python -m apps.quickstart.multimodal_agent"
```

Open `https://localhost:9000` (HTTPS required for camera/mic access). Click **Start Streaming** and speak. The agent responds in text and via local speakers.

> Note: the server uses a self-signed cert in `server/certs/`. You'll need to accept the browser warning on first load.

## Folder structure

```
apps/           Runnable app entrypoints and examples
client/         Browser-side WebRTC client shell
server/         WebRTC server + core processing modules
  core/         Streams, VAD, STT, TTS, LLM/VLM, session helpers
tests/          Unit tests for streams, controllers, and app logic
```

## Apps

| App | Status | Description |
|-----|--------|-------------|
| `apps/quickstart/multimodal_agent.py` | ✅ Runnable | Minimal audio/video agent: browser mic/camera → STT/VLM → text + TTS response |
| `apps/quickstart/mic.py` | ✅ Runnable | Audio-only quickstart for mic → STT/agent response |
| `apps/quickstart/web.py` | ✅ Runnable | WebRTC quickstart entrypoint using the shared server/session stack |

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

## Building interactive games

Mulive apps can be written as stateful controllers instead of free-form agents. Voice transcripts, browser UI events, and model/tool results are converted into typed events, then sent through one controller mailbox. The controller owns game state and emits structured outputs for transcript text, TTS, client UI feedback, and async side effects such as OCR or VLM calls.

Use deterministic controller logic for scoring and state transitions. Use LLM/VLM calls only for flexible perception or language tasks, then convert results back into typed events.

## TTS backends

- **Kokoro** (default) — local ONNX model via OpenAI-compatible API (`localhost:8880`). Start with `Kokoro-FastAPI/`.
- **Gemini TTS** — cloud, via `google-genai`.


## Misc Install
- Kokoro: change pytorch cpu version in pyproject.toml to the one the matches pre-installed (ALbertModel error)
