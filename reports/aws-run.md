# Running mulive on AWS — CLI and systemd

**Scope.** Runbook for the `aws` branch on a 1 vCPU / 2 GB / 4 GB-swap
server. Certs and nginx are already configured; espeak-ng is installed.

---

## 1. What ships in this branch

Five small, orthogonal changes were added, all behind flags so the
desktop stack is bit-for-bit unaffected:

- `--stt-provider` / `--stt-model-size` / `--stt-language` and
  repeatable `--stt-kwarg KEY=VALUE` on the quickstart CLI. Extra kwargs
  flow all the way into `faster_whisper.WhisperModel`.
- `--tts-provider {kokoro_fastapi, kokoro_onnx}` on the quickstart CLI.
  `kokoro_onnx` runs in-process (no second PyTorch server) and can now
  push PCM into the outbound WebRTC audio track.
- STUN configured on the peer connection (env `STUN_URLS`,
  comma-separated; defaults to Google public STUN).
- `SILERO_BACKEND=onnx` env flag makes the VAD load via `silero-vad`
  onnx (no torch).
- `max_concurrent_sessions` (default 2) at the offer handler.

Default provider is still `mlx` and default TTS is still
`kokoro_fastapi`, so the committed defaults match desktop.

---

## 2. One-time setup on the server

Assuming Ubuntu on ARM (t4g.small / c6g.medium). Adjust for x86.

```bash
# Repo
git clone <repo> mulive && cd mulive && git checkout aws

# System deps (espeak-ng already installed per user)
sudo apt update
sudo apt install -y python3-venv build-essential ffmpeg

# Python venv
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel

# Runtime deps — trim torch, add silero-vad + onnxruntime
pip install -r requirements.txt
pip uninstall -y torch                 # not needed with SILERO_BACKEND=onnx
pip install silero-vad onnxruntime

# 4 GB swap safety net (only a safety net; models must stay resident)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
sudo sysctl vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.d/99-mulive.conf

# Pre-download the models so cold start is 15-30 s, not multiple minutes
mkdir -p ext
# faster-whisper base int8 lives in the HF cache after first run;
# for kokoro-onnx put the int8 model + voices file here:
#   ext/kokoro-v1.0.onnx  (or the int8 variant)
#   ext/voices-v1.0.bin
```

---

## 3. Running from the command line (recommended first)

The user already ran `muapps/spell` on a t2 and it was slow — do a
clean CPU-only run of the quickstart agent first to isolate the bottleneck.

```bash
source .venv/bin/activate

# Environment overrides (all optional; effective defaults shown)
export SILERO_BACKEND=onnx          # skip torch entirely
export OMP_NUM_THREADS=1            # respect the 1-vCPU box
export STUN_URLS="stun:stun.l.google.com:19302"  # default

# Start the agent
python apps/quickstart/multimodal_agent.py \
  --http \
  --stt-provider faster_whisper \
  --stt-model-size base \
  --stt-kwarg compute_type=int8 \
  --stt-kwarg cpu_threads=1 \
  --stt-kwarg num_workers=1 \
  --tts-provider kokoro_onnx \
  --tts-browser \
  --llm-model groq:meta-llama/llama-4-scout-17b-16e-instruct
```

Notes:

- `--http` because nginx is doing TLS termination in front. If you
  point the client straight at port 9000, use the default `--https`
  and let nginx forward transparent TLS.
- `--tts-browser` selects the WebRTC-track output path in the pipeline;
  combined with `--tts-provider kokoro_onnx` it uses the in-process
  ONNX runtime.
- Cloud STT alternative: `--stt-provider deepgram` requires the
  Deepgram API key already read by `server/core/stt/deepgram.py`. It
  removes the local Whisper CPU cost entirely.
- Watch the console: the pipeline prints per-stage timings via
  `monitor_time`. First-audio latency for `kokoro_onnx` is the whole-
  clip synthesis time (the ONNX Kokoro is non-streaming).

Persistent CLI run with detached logs:

```bash
mkdir -p ./logs
nohup python apps/quickstart/multimodal_agent.py --http \
  --stt-provider faster_whisper --stt-model-size base \
  --stt-kwarg compute_type=int8 --stt-kwarg cpu_threads=1 \
  --tts-provider kokoro_onnx --tts-browser \
  > ./logs/mulive.log 2>&1 < /dev/null &
```

Follow logs: `tail -f ./logs/mulive.log`.

---

## 4. Running under systemd (optional)

**Answer to "systemd is a hassle to see logs or easy?"**  It's easy
once you know two flags. `journalctl -u mulive -f` streams the log
just like `tail -f`; `journalctl -u mulive --since "10 min ago"` shows
recent history. The advantage over `nohup` is auto-restart, resource
limits, and boot-time start. The downside is the extra unit file.

Unit file at `/etc/systemd/system/mulive.service`:

```ini
[Unit]
Description=mulive quickstart agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/mulive
Environment=PATH=/home/ubuntu/mulive/.venv/bin:/usr/bin:/bin
Environment=SILERO_BACKEND=onnx
Environment=OMP_NUM_THREADS=1
Environment=STUN_URLS=stun:stun.l.google.com:19302
ExecStart=/home/ubuntu/mulive/.venv/bin/python \
  apps/quickstart/multimodal_agent.py --http \
  --stt-provider faster_whisper --stt-model-size base \
  --stt-kwarg compute_type=int8 --stt-kwarg cpu_threads=1 \
  --stt-kwarg num_workers=1 \
  --tts-provider kokoro_onnx --tts-browser
Restart=on-failure
RestartSec=3
# Memory safety: hard-cap so OOM kicks the process instead of the box
MemoryMax=1800M

[Install]
WantedBy=multi-user.target
```

Commands:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mulive.service   # start + boot-time enable
sudo systemctl restart mulive                # reload after edits
sudo systemctl status mulive                 # health snapshot
journalctl -u mulive -f                      # live tail (like tail -f)
journalctl -u mulive --since "10 min ago"    # recent slice
journalctl -u mulive -p err                  # error entries only
```

Log rotation is handled by journald automatically; no extra config
needed unless you want a persistent on-disk copy.

---

## 5. nginx notes (already configured)

Make sure nginx passes WebSocket-style upgrades for aiortc offer/answer
and forwards the client IP so aiortc can log it. Typical block:

```nginx
location / {
    proxy_pass http://127.0.0.1:9000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
}
```

Also make sure UDP is open in the AWS security group (aiortc gathers
ephemeral UDP ports for RTP). If your security group cannot open a
range, front the deployment with a TURN server on a fixed UDP port and
add it to `STUN_URLS`, e.g.
`STUN_URLS="stun:stun.l.google.com:19302,turn:turn.example.com:3478?transport=udp"`.

---

## 6. Sanity checks after starting

```bash
# TTS + STT model loaded?
grep -E "Loading|kokoro|whisper" ./logs/mulive.log   # or journalctl

# Memory footprint
ps -o pid,rss,cmd -C python

# Any swap traffic? (should be zero after warm-up)
vmstat 5

# Session cap effective?
curl -s -o /dev/null -w "%{http_code}\n" \
  -X POST http://127.0.0.1:9000/offer -H 'content-type: application/json' -d '{}'
# Should be 500 (bad sdp) on the first call, 503 once max_concurrent_sessions
# active peers are up.
```

---

## 7. What to do if it is still slow

- Confirm `SILERO_BACKEND=onnx`, `OMP_NUM_THREADS=1`, and that
  `torch` is NOT importable in the venv.
- Confirm `compute_type=int8` reached ctranslate2 (grep the startup log
  for `compute_type=int8`).
- If a t2 credit balance is drained, latency is not the app — it is the
  vCPU being throttled to 20% baseline. Check `CloudWatch` metric
  `CPUCreditBalance` or move to `c6g.medium` (dedicated core).
- Fall back to `--stt-provider deepgram` to isolate the whisper CPU
  cost. Kokoro-onnx synthesis time is measured in the log as
  `tts.synthesize`; expect ~1 s per short sentence on a shared t3-class
  vCPU, ~0.5 s on c6g.medium.
