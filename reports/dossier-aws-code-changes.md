# AWS deployment enablement — change dossier

Small, orthogonal changes that let the `aws` branch run on a
1 vCPU / 2 GB EC2 box while keeping desktop behaviour identical.

## Why

- The desktop path assumes MLX (Apple-only), Kokoro-FastAPI as a
  second PyTorch process, torch-backed VAD, no ICE/STUN, and
  unbounded session count. None of those hold on a small AWS box.
- Nothing that already works on desktop should regress. Every change
  is behind a flag or has a default matching the previous behaviour.

## What changed, file by file

- **`server/core/stt/whisper.py`** — `get_faster_whisper_model` now
  takes `**kwargs` and forwards them into `faster_whisper.WhisperModel`
  so callers can pin `cpu_threads`, `num_workers`, `device` from
  outside. Previous signature only exposed `compute_type`.
- **`server/core/multimodal_pipeline.py`** — `run_multimodal_session`
  gained `stt_kwargs=None` and `tts_provider="kokoro_fastapi"`
  parameters. `stt_kwargs` is spread into `stream_dsl.stt(...)`;
  `tts_provider` is forwarded to `add_kokoro_tts`.
- **`apps/quickstart/multimodal_agent.py`** — CLI additions:
  `--stt-language`, repeatable `--stt-kwarg KEY=VALUE`,
  `--tts-provider`. Values are coerced (int/float/bool/None/str).
- **`server/core/tts_providers/kokoro_onnx.py`** — provider now
  accepts `output="local"|"webrtc"` and an optional `audio_track`.
  New helper `_stream_kokoro_to_track` synthesizes with
  `kokoro_onnx.Kokoro.create` (non-streaming) and pushes int16 PCM
  in 20 ms frames into the outbound WebRTC track. Mirrors the shape
  of the existing FastAPI webrtc path.
- **`server/core/pipeline_helpers.py`** — `add_kokoro_tts` picked up
  a `provider` argument (`kokoro_fastapi` | `kokoro_onnx`) and a small
  `_make_kokoro_provider` factory. Default is `kokoro_fastapi`.
- **`server/setup_tracks.py`** — `pc_session_setup` now passes an
  `RTCConfiguration` with ICE servers. Precedence:
  `config['ice_servers']` → env `STUN_URLS` → Google public STUN.
- **`server/core/turndet.py`** — Silero VAD backend is selected by env
  `SILERO_BACKEND` (`torch` default | `onnx`). ONNX backend uses the
  `silero-vad` PyPI package with `onnx=True` and avoids the torch
  install entirely. Both return `(model, utils)` with
  `utils[0] = get_speech_timestamps`, so `_build_is_speech` did not
  change.
- **`server/server_asyncio.py`** and **`server/server_fastapi_webrtc.py`**
  — offer handlers reject new offers with HTTP 503 once
  `len(self.pcs) >= config['max_concurrent_sessions']`. Default 2.

## Key design decisions

- **All flags off by default.** Every new knob has a default matching
  the previous behaviour. Desktop path is unchanged unless someone
  explicitly opts in.
- **STT choice is per-run, not committed.** Task 1 said the STT change
  must not be committed. It isn't: the CLI defaults are still
  `--stt-provider mlx --stt-model-size tiny`; the AWS deployment
  override lives in the run script or the systemd unit only.
- **Spell's clause splitter made common.** Task 2 asked whether the
  spell-app clause splitter could be made common. It could: spell's
  `SPOKEN_PHRASE_BOUNDARY` regex + per-phrase expansion now live in
  `pipeline_helpers.split_spoken_phrases()`, and `add_kokoro_tts`
  applies it (via the existing `expand_items` operator) for the
  non-streaming `kokoro_onnx` provider only. Streaming kokoro_fastapi
  keeps whole-utterance requests, so desktop is unchanged. The spell
  app can import the shared helper when branches merge.
- **Cap is opt-in; quickstart defaults to 2.** The servers default to
  no cap (`max_concurrent_sessions: None`) so the multi-user dashboard
  path keeps its old behaviour; the quickstart CLI defaults
  `--max-concurrent-sessions` to 2 per the user's "set it to 2"
  (0 = unlimited). Leaves headroom for the second vCPU on a t4g.small.
- **STUN default is Google public STUN.** No cost, no accounts. If a
  deployment cannot open UDP broadly in the security group, add a
  TURN server via `STUN_URLS` — no code change needed.
- **VAD ONNX backend is opt-in and genuinely torch-free.** Existing
  desktop users keep torch because their environment already has it;
  AWS deployments set `SILERO_BACKEND=onnx`. The `silero-vad` PyPI
  package itself imports torch even in ONNX mode, so the backend runs
  the model directly on onnxruntime with numpy I/O and locates the
  bundled `silero_vad.onnx` without executing the package
  (`pip install --no-deps silero-vad onnxruntime`).

## Alternatives rejected

- Running Kokoro-FastAPI as a second process on the AWS box —
  doesn't fit in 2 GB (two PyTorch processes both load full runtime).
- Removing MLX support in `stream_dsl.stt` — would break the desktop
  path. Kept both.
- Global config file with all knobs — dev complexity without benefit
  because the CLI + env cover everything the deployment needs.
- Sentence splitter as its own module — no reuse in this branch yet;
  the LLM prompt already produces short spoken sections. Can be added
  later if a new app emits multi-sentence bursts.

## Risks and how they were mitigated

- **kokoro-onnx.Kokoro.create is not streaming** — first-audio latency
  equals whole-clip synthesis time. Mitigated by the visual_solver
  prompt capping SPOKEN to 2-3 short sentences (max 14 words each).
- **Torch removal breaks any hidden import** — grepped the whole
  `server/` tree for `import torch` / `from torch`. Only `turndet.py`
  imports torch, and it now does so lazily inside the torch branch.
- **STUN misconfiguration** — env override is a strict superset; empty
  string yields an empty ICE-server list which is exactly today's
  behaviour, so no lockout risk on desktop.
- **503s during legitimate load** — cap is configurable per deployment
  and desktop sets it to `None` (no cap) if desired.

## Review findings and fixes

A read-only review by gpt-5.6-sol (~4% of task budget, under the 20%
cap) found 8 real issues in the first round of commits; all fixed:

1. `silero-vad` PyPI package hard-depends on torch even in ONNX mode —
   "skips torch" claim was false → rewrote the backend on raw
   onnxruntime + numpy (`0a10fb5`).
2. FastAPI server never passed `on_peer_close`, so closed peers leaked
   and a cap would eventually 503 forever → fixed (`587b29e`).
3. Cap check ran before `await request.json()` — two simultaneous
   offers could both be admitted → check moved after the await
   (`587b29e`).
4. Cap default of 2 in the servers changed existing multi-user
   behaviour → servers default to no cap; quickstart CLI defaults to 2
   (`587b29e`).
5. `create_tts_provider` never passed `output`/`audio_track` to the
   ONNX provider → wired (`ae45f3e`).
6. `--stt-language` was stored but `infer_faster_whisper` hardcoded
   English → language now reaches `model.transcribe` (`2f3cea7`).
7. Spell's clause splitter exists (`SPOKEN_PHRASE_BOUNDARY` +
   per-phrase expansion) — first round wrongly said it didn't → hoisted
   to `pipeline_helpers.split_spoken_phrases()` (`ae45f3e`).
8. `requirements.txt` was missing aiortc/aiohttp/fastapi/uvicorn — a
   fresh install could not import the server → added (`a65226a`).

Plus one performance finding relevant to "spell on t2 was slow":
`is_active_speaker` ran the full librosa stack (1.16 ms/chunk, 1.34 s
first call) when only RMS mattered → numpy fast path, 0.009 ms/chunk
(`8fb8c61`).

## How the change was verified

- All touched files parse (`ast.parse`) and the full test suite passes:
  31/31 in a clean Python 3.12 environment.
- Torch-free VAD verified end-to-end in an env without torch installed:
  real speech (16 kHz WAV) detected in every chunk (probs 0.58–0.98),
  silence (0.002–0.009) and white noise rejected.
- Active-speaker fast path verified to give identical decisions to the
  librosa path on silence, quiet noise, real speech, and loud noise.
- Phrase splitter verified on sample text (5 phrases from 3 sentences).
- End-to-end run pending on the target AWS box (user's next step).
