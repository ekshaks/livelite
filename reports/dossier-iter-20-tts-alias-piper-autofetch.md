# iter-20 — Kokoro-FastAPI connection error fix, Piper auto-fetch, add_tts rename

Branch: `aws` (mulive-aws repo) — commits `2730327`, `e626123`, `fc19584`,
`9c7f46b`, `c5bffe3`. muapps repo: `a350865` on `main`.

## Reported symptom

```
Error during Kokoro-FastAPI TTS streaming:
openai.APIConnectionError: Connection error. (base_url=http://localhost:8880/v1)
```

The user was running under `MULIVE_PROFILE=aws` — the profile whose
`apps.aws.yml` sets `tts.provider: piper` — yet Kokoro-FastAPI (the
legacy HTTP TTS backend) was still being loaded and, unable to reach
its localhost port, failing loudly.

## Root cause (two bugs, one symptom)

### 1. Per-app config shadowed the profile

`muapps/chess_app/config.yml` and `muapps/tutor/config.yml` still
carried full `tts:` and `stt:` blocks with `provider: kokoro`. The
deep-merge order is:

```
defaults (apps.yml)
  → apps.<profile>.yml           (apps.aws.yml)
  → per-app config.yml           (chess_app/config.yml)
  → per-app config.<profile>.yml (chess_app/config.aws.yml)
```

Per-app `config.yml` sits *above* the profile file, so its
`provider: kokoro` silently overrode `apps.aws.yml`'s
`provider: piper`.

### 2. Standalone launches skipped the merge entirely

`python -m muapps.spell.app` / chess / tutor called
`load_app_config(config.yml)` directly — a raw single-file load that
never touched `apps.yml`'s `defaults:` or any `apps.<profile>.yml`.
So even after the per-app files were fixed, standalone launches would
still miss the profile overrides.

### 3. Legacy `provider: kokoro` was not centrally aliased

`_make_tts_provider` accepted only `kokoro_fastapi`, `kokoro_onnx`,
`piper` — the bare `"kokoro"` value raised `ValueError` from
spell/chess, but tutor's own `resolve_tts` silently translated it to
`kokoro_fastapi`, so the "kokoro" value produced three different
outcomes depending on which app was booting.

## Fixes

### 1. Central `"kokoro" → "kokoro_fastapi"` alias

`server/core/pipeline_helpers.py` now defines
`_resolve_tts_provider_name()` and applies it both in `add_tts` and
inside `_make_tts_provider`. Legacy configs keep working; new configs
get a stable, explicit name.

### 2. Effective-provider log line

`add_tts` now emits, at sink attach time:

```
tts sink attached provider_input=<from config>
                  provider_effective=<canonical>
                  mode=<local|browser>
                  name_prefix=<app-prefix>
```

The sink observable's name uses the canonical provider too, so a
config value of `"kokoro"` now names the sink
`kokoro_kokoro_fastapi_browser_tts` — matching the actual backend
that runs. Any future mis-route is a single grep away.

### 3. Piper voice auto-fetch

`server/core/tts_providers/piper.py` now walks four resolution
layers, mirroring the Silero VAD ONNX fetcher:

1. `PIPER_MODEL_PATH` env — must exist, no download attempted.
2. `ext/piper/<voice>.onnx` under the repo root — deployment-friendly
   pre-placement.
3. Cached copy at `$XDG_CACHE_HOME/mulive/piper/<voice>.onnx`.
4. Fresh download of the pinned `en_US-lessac-medium` voice from the
   `rhasspy/piper-voices` Hugging Face repo into the cache dir,
   written atomically via `<path>.part → rename` so a Ctrl-C never
   leaves a partial file in place.

Air-gapped deployments keep working via `PIPER_MODEL_PATH` or by
pre-placing the voice files.

### 4. Standalone launches are now profile-aware

`server/core/app_config.py` gained
`load_bundle_config_with_profile(config_path)` which:

* Walks up from `config_path` to find the enclosing `apps.yml`
  (checks up to two parent levels; `MULIVE_APPS_CATALOG` overrides).
* Runs the exact same `load_infra_layers` + `load_bundle_layers` that
  `server.apps.loader.load_app_bundle` uses.
* Falls back to the plain single-file load for anything outside the
  muapps tree, and for schema-1 legacy catalogs.

`muapps/spell/app.py`, `muapps/chess_app/app.py`, and
`muapps/tutor/app.py` all swapped their `__main__` blocks from
`load_app_config` to the new helper.

### 5. Config trim + AWS overrides

`muapps/chess_app/config.yml` is now down to genuine per-app deltas
(`stt.model_size: small`, `models.text`). `muapps/chess_app/config.aws.yml`
re-asserts `stt.model_size: base` under the AWS profile. Same shape for
tutor (tutor lives on disk only per repo policy — same as prior iters).

### 6. `add_kokoro_tts` → `add_tts` rename

Call sites: `apps/quickstart/web.py`, `server/core/multimodal_pipeline.py`,
`muapps/spell/app.py`, `muapps/chess_app/app.py`.
`add_kokoro_tts` stays in `pipeline_helpers.py` as a thin shim for
out-of-tree consumers.

## Verification

* `uv run --with-requirements requirements.txt --with pytest python -m pytest tests/`
  → **64 passed** (was 60 before iter-20 — 14 new cases across two
  new test files).
* Merged effective config, both profiles, all three apps:

```
TUTOR LOCAL: tts.provider=kokoro     stt=mlx/tiny
CHESS LOCAL: tts.provider=kokoro     stt=mlx/small
SPELL LOCAL: tts.provider=kokoro     stt=mlx/turbo

TUTOR AWS  : tts.provider=piper      stt=faster_whisper/base +kwargs
CHESS AWS  : tts.provider=piper      stt=faster_whisper/base +kwargs
SPELL AWS  : tts.provider=piper      stt=faster_whisper/base +kwargs
```

* `python -m py_compile` on every modified `.py` file: clean.

## Commits

### aws repo — branch `aws`

| SHA | Title |
| --- | --- |
| `2730327` | TTS: add legacy 'kokoro' provider alias + effective-provider log |
| `e626123` | TTS: auto-fetch default Piper voice from Hugging Face on first use |
| `fc19584` | apps: profile-aware config loading for standalone entrypoints |
| `9c7f46b` | pipeline_helpers/quickstart/multimodal_pipeline: rename add_kokoro_tts to add_tts |
| `c5bffe3` | tests: coverage for TTS alias, Piper auto-fetch, and standalone profile merge |

### muapps repo — branch `main`

| SHA | Title |
| --- | --- |
| `a350865` | apps: trim per-app configs + use profile-aware standalone loader |

## Not touched (intentional)

* `reports/aws-run.md` — pending user edit preserved as-is.
* `tutor/` in the muapps repo is untracked (repo policy from prior
  iters). The trimmed `tutor/config.yml`, new `tutor/config.aws.yml`,
  and updated `tutor/app.py` all live on disk only.
* `muapps/apps.aws.yml` was pre-edited by the user
  (`provider: kokoro_onnx` commented out, `provider: piper` active).
  Left as-is.
* `muapps/taskfile.yml` — user added a manual voice-download task.
  It's now redundant with the auto-fetch, but not removed.

## Try it

```
MULIVE_PROFILE=aws OMP_NUM_THREADS=1 python -m server.apps.run_apps --http

# In logs, look for:
#   Piper warmed up in X.XX s
#   tts sink attached provider_input=piper provider_effective=piper mode=browser ...
#   tts provider=piper event=first_pcm_to_webrtc_track
```

Standalone launches work identically:

```
MULIVE_PROFILE=aws python -m muapps.spell.app --config muapps/spell/config.yml --http
```
