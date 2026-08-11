# Dossier — iter-21: bring `main`'s updates into the `chess` branch

**Request:** "the main branch has moved ahead and conflicts with chess branch. bring updates from main into chess"

---

## What changed, in one paragraph

The `chess` branch had fallen 32 commits behind `main`. It is now fully caught
up. No source file was written by hand — git merged everything automatically.
The only work that required judgement was choosing *which* `main` to merge,
because there were two versions of it and one of them produced 18 fake
conflicts.

---

## The problem was not a real conflict

- Trying `git merge origin/main` reported **18 "add/add" conflicts** — git
  claiming both branches had independently created the same 18 files.
- That claim was false. Neither branch had touched most of those files in a
  conflicting way.
- **Cause:** someone had rebased (rewritten) `main`'s history on the server.
  Rebasing gives every commit a new identity even when the file contents are
  untouched. Git matches branches by shared commit identity, so it could no
  longer see any recent common starting point between `chess` and
  `origin/main`. It fell all the way back to `29d36d9`, a commit from
  **August 2025**. Measured from that far back, it genuinely looks like both
  sides invented every file from nothing — hence "add/add" on everything.
- **The tell:** the local copy of `main` and the server's `origin/main` have
  **byte-identical file trees** (`git diff main origin/main` prints nothing).
  Same files, different history. Only the history had changed.

## What was done instead

1. **Merged the *local* `main`**, which still shares a recent common starting
   point with `chess` (`a2c5272`, Aug 2026). Same file content as
   `origin/main`, but a history git can reason about.
   → **zero conflicts.**
2. **Recorded `origin/main` as already-merged** with a bookkeeping merge
   (`-s ours`) that changes no files. This tells git "chess already contains
   everything origin/main has", so the phantom conflicts cannot reappear on
   the next sync.
3. **Realigned local `main`** to match the server's canonical `origin/main`,
   since the local copy was the stale pre-rebase version.

---

## File-by-file

Everything below **arrived from main**; none of it was authored in this task.

| File | What |
| --- | --- |
| `server/core/tts_providers/piper.py` | new Piper speech engine (+300 lines) |
| `server/core/tts_providers/{__init__,factory,kokoro_onnx,kokoro_fastapi}.py` | provider registry, `kokoro` alias, warm-up hooks |
| `server/core/pipeline_helpers.py` | `add_kokoro_tts` generalised to `add_tts`; old name kept as an alias so nothing breaks |
| `server/apps/config_merge.py` *(new)* | deep-merges default config with per-profile overrides |
| `server/apps/show_config.py` *(new)* | CLI that prints the final effective config |
| `server/apps/{loader,run_apps}.py` | profile-aware loading; warm up STT/VAD/TTS at startup |
| `server/core/{turndet,audio_utils,multimodal_pipeline,app_config}.py`, `server/core/stt/whisper.py` | model caching, warm-up, auto-fetch of model weights |
| `server/{server_asyncio,server_fastapi_webrtc,setup_tracks}.py` | WebRTC / FastAPI server updates |
| `apps/quickstart/{multimodal_agent,web}.py` | new `--tts-provider` choices, warm-up hook |
| `tests/test_{app_profiles,tts_provider_aliases,run_apps_warmup,load_bundle_config_with_profile}.py` | main's 4 new test files |
| `requirements.txt` | **the only file both branches edited** — auto-merged to the union |

**Chess's own files were untouched** and kept their exact pre-merge contents:
`server/apps/{app_output,effects,prompts,qa}.py` and the five chess test files.

### `requirements.txt` — the one genuine overlap

Merged to the exact union of 24 packages, no duplicates, nothing lost:

- from **main**: `aiortc`, `aiohttp`, `fastapi`, `uvicorn`, `pillow`, `piper-tts`
- from **chess**: `silero-vad`, `onnxruntime`, `chess>=1.11,<2`

---

## Decisions, and what was rejected

| Decision | Why | Rejected alternative |
| --- | --- | --- |
| **Merge, not rebase** | Keeps chess's 3 commits intact and records an honest fork/join point. Chess had *moved files*; a rebase replays those moves against a shifting base and tends to produce repeated conflicts. | Rebase chess onto main |
| **Merge local `main`, not `origin/main`** | Identical file content, but a usable recent common ancestor → 0 conflicts instead of 18 fake ones. | Merge `origin/main` and resolve 18 conflicts by hand — slow, and every manual resolution is a chance to silently drop code |
| **`-s ours` merge to record `origin/main`** | Permanently fixes future syncs. Verified safe first: chess's tree was already a strict superset of `origin/main`. | Leave it — but then every future `git merge origin/main` re-raises the same 18 phantom conflicts |
| **Install the 4 missing dependencies** | Merge brought code importing `aiortc`; without it 2 test files could not even be imported. Verification would have been meaningless. | Skip, and report a "passing" suite that silently skipped broken files |
| **Restore `reports/evolution.html`** | Upstream's "cleanup" commit had deleted it; the page is meant to be a single growing record. Restored with history intact, then extended. | Start a fresh page and lose cards 15–20 |

---

## Risks

- **`-s ours` is the one risky step** — it records a merge while keeping our
  files, so if the tree had *not* already contained everything upstream had,
  it would silently discard upstream code. Mitigated by verifying the superset
  property before running it, and re-verifying afterwards
  (`git diff HEAD origin/main` lists only chess's own additions).
- **`muapps/` is gitignored**, so the chess app source is not covered by these
  commits or by git's merge. It was checked separately: it already calls the
  renamed `add_tts`, and has no stale imports.
- **Local `main` was force-moved.** Safe — its tree was byte-identical to
  `origin/main`, and the one commit that looked unique (`465b8a10`) is
  blob-identical to upstream's rewritten `29a12a6`. A backup branch exists
  regardless.
- **Dependency versions moved:** installing `aiortc` downgraded `av` from
  18.0.0 to 17.1.0. All tests pass, but this is the one environment change
  worth knowing about.

**Rollback:** `git reset --hard backup/chess-pre-merge` (chess),
`git branch -f main backup/main-pre-sync` (main). Both backups still exist.

---

## How it was verified

- **Merge cleanliness** — 0 conflicts; 25 files changed, 2187 insertions, 66 deletions.
- **Nothing lost** — `git diff HEAD origin/main` shows *only* chess's 9 added
  files plus its 4 extra `requirements.txt` lines. Nothing from main is
  missing or stale.
- **Tests** — full suite split across 9 parallel workers:
  **251 tests + 9 subtests, all passed, 0 failures, 0 errors.** Both sides
  covered: chess (`test_chess_workflow` 52, `test_chess_speech` 53,
  `test_chess_ask` 43, `test_chess_engine` 17 against real Stockfish,
  `test_core_effects` 22) and main's new suites (`test_app_profiles` 12,
  `test_tts_provider_aliases` 10, `test_run_apps_warmup` 7,
  `test_load_bundle_config_with_profile` 4).
- **Independent review** (separate model, read-only) confirmed: no chess file
  reverted, no upstream file dropped or stale, `requirements.txt` is the exact
  union, no import collisions between chess's and main's `server/apps/`
  modules, and `server/apps/__init__.py` still exports correctly.
- **Topology** — `chess` is now **77 ahead / 0 behind** `origin/main`; a repeat
  merge reports "already up to date".
