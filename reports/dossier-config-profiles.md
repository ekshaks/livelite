# Dossier — profile-aware config loader (iter-18)

## What changed

- New module **`server/apps/config_merge.py`** with pure merge helpers:
  `active_profiles()`, `deep_merge()`, `merge_all()`, `load_infra_layers()`,
  `load_bundle_layers()`.
- **`server/apps/loader.py`** now understands catalog schema 2, layering
  infra defaults + per-profile catalog overrides + per-bundle base config +
  per-bundle profile overrides. Schema 1 keeps its original behavior.
- New CLI **`server/apps/show_config.py`** — prints the merged effective
  config as YAML for either the whole infra or one app, with optional
  source-file listing.
- New tests **`tests/test_app_profiles.py`** — 12 end-to-end cases covering
  the merge rules, `MULIVE_PROFILE` parsing, both schema paths, and the CLI.
- Proof of concept in the (git-ignored) `muapps/` folder:
  `apps.yml` now schema 2 with a `defaults:` block, new `apps.aws.yml`
  carrying the AWS overrides, and `muapps/spell/config.yml` trimmed to
  spell-specific keys only.

## Why

Every deployment (desktop, AWS, staging, Pi) needed the same STT / TTS /
model choices copy-pasted into every app's `config.yml`. Fields drifted,
adding a new environment meant editing N files, and the AWS runbook
carried CLI flags that overlapped with the per-app config. This change
centralizes shared knobs in one `defaults:` block and lets each
environment express its differences in a single small override file.

## File-by-file

| File | Kind | Notes |
| --- | --- | --- |
| `server/apps/config_merge.py` | new | Pure merge; no side effects; ~120 lines. |
| `server/apps/loader.py` | modified | Adds schema-2 branch + optional `infra`/`profiles` kwargs on `load_app_bundle`. Schema-1 path is byte-for-byte identical. |
| `server/apps/show_config.py` | new | CLI entry `python -m server.apps.show_config`. |
| `tests/test_app_profiles.py` | new | 12 end-to-end tests. |
| `muapps/apps.yml` | modified (git-ignored) | schema bumped to 2, `defaults:` block added. |
| `muapps/apps.aws.yml` | new (git-ignored) | AWS infra overrides. |
| `muapps/spell/config.yml` | trimmed (git-ignored) | Only spell-specific keys remain. |

## Design decisions

- **Deep merge = dicts recurse; lists and scalars replace.** Matches the
  common expectation for config layering and keeps `stt.kwargs` fully
  overridable per profile.
- **Profiles compose via comma separation** (`MULIVE_PROFILE=aws,gpu`).
  Order is left-to-right; duplicates dropped; empty falls back to
  `local`. No exponential explosion of per-environment files.
- **Schema 1 stays supported** so any downstream tool that still loads a
  legacy `apps.yml` keeps working. The schema is decided per file, not
  globally.
- **Merged infra is promoted to catalog top-level** in schema 2 so
  `run_apps.py` continues to read `catalog['server']` without changes.
- **`load_app_bundle`'s new kwargs default to empty**, so any external
  caller still gets the exact behavior it had before iter-18.

## Rejected alternatives

- Adding STT/TTS/max_concurrent_sessions CLI flags to `run_apps`. Would
  have duplicated what already lives in per-app YAML and split the source
  of truth in two.
- Two separate catalog files (`apps.local.yml` / `apps.aws.yml`) with
  full duplication. Was in the earlier design; dropped in favor of the
  defaults + override layering, which scales to any number of
  environments without duplicating the shared parts.
- Env-var overrides (`MULIVE__stt__model_size=tiny`). Deferred; not
  needed for the PoC and easy to add later on top of the same merger.

## Risks

- If a `defaults:` or `apps.<profile>.yml` file accidentally sets
  `schema`, `users`, or `apps`, the loader silently overwrites them
  after the merge for schema/apps and keeps `users` from the base
  catalog only. This is safe but silent; a future warning would help.
- `web_config()` now receives extra keys (`max_concurrent_sessions`,
  `warm_up`) that it does not yet consume. They are inert until a
  follow-up wires them into the server; no functional regression today.

## Verification

- `python -m pytest tests/` → **43 passed, 1 warning** (12 new + 31 pre-existing).
- Real muapps load, both profiles, showed the expected values:
  - Local: spell `stt=mlx/turbo`, `tts=kokoro`, `server.https=true`.
  - AWS: spell `stt=faster_whisper/base + int8 kwargs`, `tts=kokoro_onnx`,
    `server.https=false`, `max_concurrent_sessions=2`, `warm_up=true`.
- `python -m server.apps.show_config --app spell --sources` prints the
  merged config plus the ordered list of source files that fed the merge.
- `ruff` clean on the new files (one style nit intentionally left alone
  to stay consistent with the codebase's existing `ValueError` pattern).

## Commits (iter-18)

1. `5679a16` apps: add profile-aware deep-merge helper
2. `ec7f008` apps: teach loader to layer defaults + profile overrides (schema 2)
3. `deee854` apps: add show_config CLI to render the merged effective config
4. `9ec38d7` tests: end-to-end coverage for profile merge + show_config CLI
