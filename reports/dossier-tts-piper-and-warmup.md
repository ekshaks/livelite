# iter-19 — TTS provider wiring, Piper, run_apps warm-up scan

## Scope

Four items pulled out of iter-18's follow-up review:

1. Forward `tts.provider` from every muapp into `add_kokoro_tts(..., provider=…)` so `MULIVE_PROFILE=aws` (which sets `tts.provider: kokoro_onnx`) actually takes effect. Without this, spell/chess silently defaulted to `kokoro_fastapi` and tried to open an HTTP stream to `http://localhost:8880/v1`, printing the ambiguous `Error during TTS streaming: Connection error.` and playing silence.
2. Env-driven `base_url` in `KokoroFastApiTTSProvider` and typed exception logging so those errors don't collapse `httpx.ConnectError`, 5xx, and JSON errors into one line.
3. New `PiperTTSProvider` — rhasspy Piper is a small, MIT-licensed, truly-streaming ONNX TTS that is a materially better fit than `kokoro_onnx` on a 1-vCPU box (first-audio latency ~200–400 ms vs 2–3 s/phrase because Piper yields PCM as it synthesizes).
4. Warm-up scan in `server/apps/run_apps.py` so the muapps dashboard preloads Whisper / Silero VAD / Kokoro-ONNX / Piper the same way the quickstart CLI does — no more first-connect / first-speech stalls on the AWS box.

## Files touched

| File | Kind | Summary |
| --- | --- | --- |
| `server/core/tts_providers/piper.py` | new | `PiperTTSProvider` (local + webrtc), `get_voice()`, `warm_up()`; env-overridable `PIPER_MODEL_PATH` / `PIPER_CONFIG_PATH`. |
| `server/core/tts_providers/__init__.py` | mod | Export `PiperTTSProvider`. |
| `server/core/tts_providers/factory.py` | mod | `create_tts_provider` learns `piper`; `TTSConfig` literal extended. |
| `server/core/tts_providers/kokoro_fastapi.py` | mod | `_kokoro_fastapi_base_url()` reads `KOKORO_FASTAPI_URL`; both hardcoded URLs replaced. Streaming error prints exception class + message + base_url. |
| `server/core/pipeline_helpers.py` | mod | New `add_tts()` canonical helper; `add_kokoro_tts()` kept as a back-compat alias. `_make_tts_provider` handles piper. Phrase splitting is gated on `_NON_STREAMING_TTS_PROVIDERS` (only `kokoro_onnx`). |
| `apps/quickstart/multimodal_agent.py` | mod | `--tts-provider` gains `piper` choice; `warm_up()` call added for piper. |
| `server/apps/run_apps.py` | mod | `_collect_warm_up_targets` + `_warm_up_from_registry` scan enabled apps' merged configs; guarded by `server.warm_up: true`; sets `SILERO_BACKEND` from `silero_backend` in the merged infra. |
| `requirements.txt` | mod | Pin `piper-tts`. |
| `tests/test_run_apps_warmup.py` | new | 7 end-to-end tests over a temporary muapps tree. |
| `muapps/spell/app.py`, `muapps/chess_app/app.py` | mod (muapps repo) | Forward `tts.get("provider", "kokoro_fastapi")` through `run_session`/`attach_outputs`/`attach_game_outputs` into `add_kokoro_tts`. |
| `muapps/tutor/app.py` | mod (muapps repo, untracked in that repo) | Rename `resolve_tts_mode` → `resolve_tts` returning `(mode, provider)`; accept `kokoro/kokoro_fastapi/kokoro_onnx/piper`. |
| `reports/dossier-tts-piper-and-warmup.md`, `reports/evolution.html` | new/mod | This dossier + iter-19 card. |

## Verification

- `uv run --with-requirements requirements.txt --with pytest python -m pytest tests/` → **50 passed** in ~1.4 s (7 new + 43 pre-existing).
- Import smoke test:
  ```
  from server.core.pipeline_helpers import add_kokoro_tts, add_tts
  from server.core.tts_providers.factory import TTSConfig, create_tts_provider
  from server.core.tts_providers import PiperTTSProvider
  ```
  all resolve without pulling piper's voice or Kokoro's ONNX weights.
- `KOKORO_FASTAPI_URL=http://foo:1234/v1` → `_kokoro_fastapi_base_url()` returns the override; verified inline.

## Sink-name check (the smoking gun the review called out)

The tag that appears in interrupt / speak_start logs is built from `f"{name_prefix}_{provider}_{mode}_tts"`. With `MULIVE_PROFILE=aws` and `tts.provider: kokoro_onnx` after iter-19 the sink name is now `kokoro_kokoro_onnx_browser_tts`, and with `tts.provider: piper` it becomes `kokoro_piper_browser_tts`. Anything containing `kokoro_fastapi` in the logs on AWS is now a real config bug rather than the silent default.

## Design impact

- `add_kokoro_tts` is unchanged from a caller's POV (backward-compat alias). New sites should prefer `add_tts`.
- `piper-tts` is a mandatory dependency now. Piper's assets are NOT bundled — the deployment runbook needs a `piper-download-voice en_US-lessac-medium` step (or a manual `ext/piper/` drop) before Piper works. When the file is missing, `PiperTTSProvider` still imports; only the first synth raises. `warm_up()` emits a `RuntimeWarning` at process start when assets are missing.
- Warm-up is opt-in via `server.warm_up: true`. Desktop (default profile) does not preload anything.

## Known follow-ups

- Piper voice download is manual today. Adding a `_ensure_piper_asset()` fetcher (like the Silero ONNX auto-fetch) is a small next iteration.
- `add_kokoro_tts` is worth renaming to `add_tts` in the muapp call sites eventually — kept as-is here to keep the diff local.
- Spell app has its own `SPOKEN_PHRASE_BOUNDARY` splitter that was hoisted to `pipeline_helpers.split_spoken_phrases` in iter-15. When the spell branch merges, drop the duplicate.
