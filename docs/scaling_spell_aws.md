# Scaling spell to hundreds of clients on one AWS server

Plan for running the spell app (WebRTC voice + camera OCR spelling game) with the server on an
AWS machine and ~100–500 concurrent browser clients. Two variants are analyzed: **A) STT/TTS via
APIs** and **B) everything local on CPU**. Numbers below come from current public benchmarks and
pricing (Feb 2026); re-verify before committing.

## Baseline: what one session costs today

Per session the server does: WebRTC audio (+ video) decode → Silero VAD poll → STT on each
speech segment → controller (game logic) → LLM word/letter matching (Groq) → OCR via Gemini VLM
(on demand) → Kokoro TTS. Spell is turn-based: a user speaks ~1–3 s bursts, maybe 5–10% of wall
time. That duty cycle is what makes hundreds of clients feasible at all — the per-session
*continuous* cost is only WebRTC decode + VAD (a few % of a core); the heavy stages (STT, LLM,
TTS, OCR) are short bursts that can be pooled.

## Changes needed in BOTH versions (do these first)

1. **Replace mlx STT.** `stt.provider: mlx` in spell/config.yml is Apple-Silicon-only. On AWS it
   must become `faster-whisper` (CPU int8) or `deepgram`. Mandatory, day one.
2. **One concurrency world.** Apply the audited AsyncIOScheduler fix from the overview
   (+ its 3 companions: `to_thread` around sync faster-whisper, mic bridge, turndet loop
   fallback), and unify the three copy-pasted browser senders into
   `SessionContext.send_to_client()`. Today's hand-rolled locks and thread-guessing in
   `async_map_stage`/`ControllerFlow.submit` are latent corruption bugs that *will* fire at
   300 sessions.
3. **Delete duplicates before scaling them.** One controller framework (keep async), one spell
   game, one server (`server_fastapi_webrtc.py`). Extract `standard_pipeline()` + a generic
   effect runner so per-app wiring is small. You don't want to load-test 700 duplicated lines.
4. **Multi-process workers + sticky sessions.** aiortc is GIL-bound: DTLS/SRTP crypto and packet
   handling run on the one event loop; realistic ceiling is ~20–50 audio(+sampled video)
   sessions per process. So: run N worker processes (uvicorn workers won't do — the
   RTCPeerConnection must live where the offer was answered). Simplest scheme: a front router
   (nginx/small FastAPI) that assigns each new session to worker i and returns that worker's
   offer endpoint; media then flows client↔worker directly. 300 clients ≈ 8–16 workers on one
   32-vCPU box.
5. **Network plumbing.** Elastic IP, security group opening UDP media ports, coturn (TURN) for
   clients behind strict NATs, TLS certs (already `https: true`). Note: AWS NLB/ALB can front
   the signaling HTTP, but WebRTC *media* goes straight to the instance IP — plan ports per
   worker.
6. **Error semantics + supervision.** Today one controller exception closes the Rx flow and
   kills the session. At 300 sessions errors are constant background noise: catch per-event,
   log, keep the session; restart workers on crash without dropping the other 30 sessions
   (session draining, client auto-reconnect with state resume).
7. **Backpressure and shedding.** Bound every mailbox/queue; drop video frames (raise
   `input_video_sample_interval`, only grab a frame on OCR request) and skip stale STT jobs
   (if a segment waited > ~2 s, the turn is dead). Admission control: cap sessions per worker,
   return "server full" cleanly.
8. **Shared state out of process memory.** `users.yml`, wordlists, per-user progress must move
   to a small store (Postgres/DynamoDB/S3) once there are many workers; sessions themselves
   stay in-memory + sticky.
9. **Observability + load harness.** `monitor_time` → Prometheus (per-stage latency histograms,
   queue depths, sessions per worker). Build a synthetic client (aiortc script replaying
   recorded utterances + a handwriting image) and run 100/300/500-client soak tests. Without
   this every number in this doc is a guess.

## Version A — STT/TTS/LLM via APIs

Server does only WebRTC + VAD + game logic; all ML is network calls.

- **STT — Deepgram** (already a supported backend). Two modes:
  - Keep the current design (VAD produces finished segments → one pre-recorded request per
    turn): $0.0043/min of *actual speech*, but default limit is 50 concurrent pre-recorded
    requests — fine, since segments are short and turn-based.
  - Or a persistent streaming socket per session: better latency and interim results, but
    $0.46/hr per session ⇒ ~$140/hr at 300 sessions, and the 225-stream default cap. Verdict:
    **stay segment-based**; it's ~10× cheaper and fits the existing pipeline.
- **TTS** — Kokoro-FastAPI is already supported as a network provider, but as an API: Amazon
  Polly Neural ($16/1M chars, in-region, no egress, effectively unlimited concurrency) or
  Deepgram Aura-2 ($30/1M chars, ~150 ms TTFB). Add an **audio cache** anyway (see B) — spell's
  utterances are highly repetitive, cache cuts API spend ~10×.
- **LLM/VLM** — keep Groq (paid tier; free tier's 30 RPM dies at ~15 users) and Gemini for OCR
  (Tier-1 = 1,000 RPM but only 10k requests/day — hundreds of daily sessions need Tier 2 /
  Vertex). Build one shared rate-limiter + 429 backoff + per-session budget; today every
  session calls the SDKs independently and a burst will stampede the key.
- **Sizing/cost (≈300 clients):** c7i.4xlarge (16 vCPU, $0.71/hr) runs WebRTC+VAD+logic.
  APIs: STT ≈ 300 sess × ~6 min speech/hr = 30 audio-hr/hr ≈ $8/hr; TTS ≈ 600k chars/hr ≈
  $10/hr (Polly, uncached; ~$1 cached); LLM/VLM a few $/hr. **Total ≈ $15–25/hr, scales with
  usage.**
- **Hardest parts of A:** (1) rate limits/429/quotas across three vendors, with graceful
  degradation when one is down; (2) tail latency — p99 of API calls stacks up in a live voice
  loop, need timeouts + "say something" fillers; (3) cost control (a stuck streaming socket
  or retry loop burns money); (4) secrets management and audio egress/privacy.

## Version B — everything local on CPU

All models run on the box; cost is fixed; nothing leaves the machine.

- **STT — faster-whisper int8** (already in `stt/whisper.py`, needs the `to_thread` fix).
  `small` ≈6–8× real-time on 4 threads, `large-v3-turbo` ≈2–4×. Replace the single pinned
  executor with a **shared STT worker pool** (e.g. 4 processes × 4 threads, one queue): 300
  sessions × ~7.5% speech duty ≈ 22 concurrent audio-s/s ⇒ ~13 cores with `small`. Spell's
  vocabulary is letters + known words, so `small` (or even `base` + wordlist biasing) is enough.
- **TTS — Kokoro ONNX + aggressive caching.** Raw Kokoro CPU RTF ≈0.5–0.7 (≈2–3 core-s per
  audio-s). Uncached at 300 sessions (~20% assistant-speaking duty ≈ 60 audio-s/s) that's
  **120–180 cores — the killer**. But spell's speech is templates + wordlist words: cache WAVs
  keyed by (text, voice, speed) on disk/S3, **pre-synthesize the whole wordlist and all prompt
  templates at deploy time**. Expected cache hit ≥95%, leaving live TTS ≈ a handful of cores.
  This one change is what makes local CPU viable. (Fallback: swap in a faster CPU TTS —
  Supertonic ~4× RT — if uncachable text grows.)
- **LLM — mostly remove it.** `groq:openai/gpt-oss-20b` has no local CPU equivalent at this
  speed. Spell uses the text LLM to map transcripts to letter/word events — largely doable
  deterministically (fuzzy match against the known target word + phonetic alphabet table,
  extending `input_parser.py`). Keep a small local model (llama.cpp server, Qwen3-4B Q4 on
  AMX: ~30–45 tok/s/stream, memory-bandwidth-bound so budget ~4–8 concurrent generations) only
  for the rare ambiguous turn, with strict max_tokens and a JSON grammar.
- **VLM/OCR — the genuinely hard one.** Gemini reads kids' handwriting; local CPU options are
  much weaker: TrOCR-handwritten / PaddleOCR (fast but weak on handwriting), or Qwen2.5-VL-3B
  via llama.cpp (better but ~5–15 s per image on CPU — acceptable only because OCR is an async
  effect, not in the voice loop). Expect a real accuracy drop; mitigate by constraining
  matching to the expected word (the code already does closest-match scoring). If quality is
  unacceptable, this is the one call to keep remote (making it a hybrid).
- **Process architecture:** WebRTC workers (asyncio) + separate model-server processes
  (STT pool, TTS pool + cache, llama.cpp, OCR) connected by local HTTP/unix-socket queues —
  never in the event-loop process. The kokoro-fastapi provider and Deepgram client interfaces
  mean core/ already has the right seams; add a `whisper-server` client analogous to them.
- **Sizing/cost (≈300 clients):** WebRTC+VAD ≈ 10–15 cores, STT ≈ 13, TTS (cached) ≈ 4–8,
  LLM ≈ 4–8, OCR ≈ 4–8, headroom ⇒ **~48–64 vCPU: one c7i.12xlarge/16xlarge or
  2 × c7i.8xlarge ≈ $2.2–2.9/hr fixed**, dropping ~40–70% with savings plans. Cheaper than A
  at sustained high usage; more expensive when idle.
- **Hardest parts of B:** (1) TTS CPU cost — solved only if the cache/pre-synthesis works;
  (2) handwriting OCR quality without a big VLM; (3) building/operating the model-server pool
  (queues, warm loading ~4–6 GB of models, health checks, per-stage autoscaling); (4) keeping
  p95 voice-loop latency under ~1.5 s when STT/TTS queue under burst load; (5) losing quality
  headroom — every model is now the small version, so game logic must tolerate worse
  transcripts.

## The hardest parts overall, ranked

1. **aiortc's per-process ceiling** → multi-worker sticky-session architecture, the one change
   neither version avoids (Python GIL, crypto on the loop, ~110 Mb/s data-channel cap).
2. **The three-thread-world concurrency debt** (RxPY timer thread + asyncio + worker threads
   with hand-rolled locks) — must be collapsed before load, or debugging at 300 sessions is
   hopeless.
3. **TTS throughput** — per-session streaming cost (A) or CPU burn (B); the fix in both cases
   is the same: cache + pre-synthesize, because spell's speech is repetitive.
4. **Handwriting OCR** — no cheap local substitute for Gemini (B); rate/day limits (A).
5. **Error/failure semantics at scale** — one exception currently kills a session; supervision,
   reconnect, and draining are new subsystems.
6. **NAT/TURN/port topology on AWS** — media doesn't go through the load balancer.
7. **Load testing + observability** — synthetic WebRTC clients are a prerequisite for trusting
   any capacity number here.
8. **LLM rate limits and per-key stampedes** (A) / replacing the 20B parser with deterministic
   matching + a 4B fallback (B).

## Recommendation

Do the common items 1–9 first, then ship **A** (fastest path, ~$15–25/hr at 300 users), while
building the two pieces of **B** that pay off regardless: the TTS cache/pre-synthesis and the
faster-whisper worker pool. That lands you on the natural hybrid — local STT + cached local
TTS + API VLM for OCR — which is both the cheapest and the least risky end state.
