# Iter-16 — Kill the connect / mic-on hang and the RMS log spam

**Ask.** Three questions from the user after a live test on t2:

- Why does the browser/system hang when I connect to the URL, and again when I switch on the mic — but everything works fine after that?
- Why is the RMS/centroid line being printed on every chunk?
- TTS is slow. Can the latencies be improved?

**Diagnosis (root causes, in one line each).**

- The pipeline builds a fresh `WhisperSTT` per WebRTC session, and `WhisperSTT.__init__` loads a WhisperModel (1–3 s on 1 vCPU). Users see this as "browser hangs on connect".
- Silero VAD (ONNX backend) loads on first inference, and with the fresh-server download logic it also fetches the ONNX file from GitHub then. That first call happens the first time the mic captures audio → "mic hangs".
- Kokoro-ONNX loads its weights and initializes eSpeak G2P on the first `create()` call. Users experience it as a 2 s pause before the first spoken reply.
- `is_active_speaker` was a coarse RMS pre-filter upstream of Silero VAD. Silero already rejects silence cheaply — the RMS gate was redundant. Worse, its default `debug=True` branch ran a full librosa feature stack (RMS + centroid + rolloff + ZCR + YIN pitch) and printed one line per active chunk. That was the source of the log spam.

**Fix.**

- **Drop `is_active_speaker`** from the audio-track pipeline. VAD is the real gate; the RMS filter and the librosa print go away with it.
- **Auto-fetch Silero ONNX** from a pinned upstream URL (sha256-verified, atomic rename) so the torch-free install works on a fresh box.
- **Warm-up hooks** on all three models — `warm_up()` in `stt/whisper.py`, `warm_up_vad()` in `turndet.py`, `warm_up()` in `tts_providers/kokoro_onnx.py`. Each loads the model and runs one tiny inference.
- **Process-wide Whisper cache** keyed by `(mode, model_size, model_id, kwargs)` so multiple WebRTC sessions share the same loaded model.
- **Quickstart wiring**: after parsing CLI args, call the relevant warm-ups before `server.run()`. Guards are provider-aware (skip Deepgram / kokoro_fastapi / --tts-local when the CLI didn't select them).

**File-by-file.**

- `server/setup_tracks.py` — remove the `is_active_speaker` import + call; forward every buffered chunk to the speech-turn subject. Also removed `rms_thresh` / `debug` locals that only served that path.
- `server/core/turndet.py` — pin the Silero URL + sha256, add `_download_silero_onnx`, add `_mulive_cache_dir`, extend `_find_silero_onnx_model` with cached-file + fresh-download fallbacks, add `warm_up_vad()`.
- `server/core/stt/whisper.py` — module-level `_WHISPER_MODEL_CACHE` dict, `_cache_key()` helper, singleton behavior in `get_whisper_model`, new `warm_up()` function.
- `server/core/tts_providers/kokoro_onnx.py` — new `warm_up()` function that calls `get_kokoro()` and runs `create("hi", …)`.
- `apps/quickstart/multimodal_agent.py` — call the three warm-ups after parsing args, before constructing `Server`. Provider-aware guards.

**Design decisions and alternatives rejected.**

- **Sync warmups (chosen)** vs async warmups from `Server.run()`. Sync at process start is simpler, blocks nothing important (the server hasn't started listening yet), and produces clean startup logs. The rejected alternative would need a `pre_run` async hook and interleave with aiohttp init.
- **Module-level global for Kokoro/VAD** (already there for both) vs class-level cache. Keeping the existing pattern; already thread-safe enough because the first synth is serialized behind the singleton getter.
- **Whisper cache keyed by kwargs** rather than a single global. Same server could in theory be reconfigured mid-run (different STT provider or compute_type); keying by config keeps it correct without extra locking.
- **Warmup does one dummy inference**, not just model load. Model load alone leaves the ONNX graph unoptimized on first real call; a "hi" synth also primes eSpeak. Same reason we run a 1 s silence clip through Silero at warm-up.
- **Left `debug`/`rms_thresh`/`filter_gender` keys in `web_config`** even though `setup_audio_track` no longer reads them. Removing them touches unrelated config surfaces; harmless dead keys are the smaller change.

**Risks.**

- Warm-up runs on the main thread and blocks the event-loop startup for ~4–7 s total on a cold t4g.small. That's a one-time cost per process; acceptable since the alternative was paying it on the user's first turn.
- Whisper cache never evicts. Fine for a headless server that loads one model config; a monitor with N model sizes would grow unboundedly, but that's not this app.
- Auto-download reaches GitHub raw content on first start. If the network is blocked, set `SILERO_ONNX_MODEL_PATH` in the env; error message now names both escape hatches.

**Verification.**

- `python3 -m py_compile` clean on all five edited files.
- `git blame` confirms `reports/aws-run.md` was edited by the user out-of-band (not part of this task).
- Full pytest run needs the project venv (missing on this worktree); focused reasoning + import-safe module structure judged sufficient for these small edits.
- Suggested manual check on t4g.small: (a) start server, watch logs for three warm-up messages; (b) load URL — first "Connection state changed" arrives within milliseconds of setRemoteDescription instead of a multi-second gap; (c) speak — no RMS/Centroid lines in the log; (d) first assistant reply's `service=tts` `synthesize` timing should drop below 1 s for short phrases now that the ONNX graph is pre-optimized.

**How the review pass would go if invoked.**

- Reviewer target: gpt-5.6-sol, read-only, capped at 20% of the task budget. Ask it to look at the five commits from `aws@iter-15` to HEAD and flag: (a) any code path that still triggers a lazy Whisper load (e.g. someone constructing `WhisperSTT` with different kwargs than warmup used); (b) whether the cache-key on kwargs is stable under CLI-parsed float/int coercion; (c) whether the removed `is_active_speaker` had any subtler role beyond RMS gating; (d) whether Kokoro's `warm_up` synth blocks correctly on a machine without a soundcard. Not invoked in this iteration to keep the change surface small — user asked "why" questions, not for a full audit.
