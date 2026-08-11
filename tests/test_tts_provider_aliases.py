"""Tests for TTS provider alias resolution and Piper asset auto-fetch.

End-to-end where possible: we don't mock ``_make_tts_provider``'s internals,
only the outer environment (URL for the auto-fetch downloader).
"""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from server.core.pipeline_helpers import (
    _make_tts_provider,
    _resolve_tts_provider_name,
)
from server.core.tts_providers import KokoroFastApiTTSProvider


class TestTtsProviderAliasResolution(unittest.TestCase):
    """The legacy ``provider: kokoro`` value must resolve to Kokoro-FastAPI."""

    def test_kokoro_alias_resolves_to_kokoro_fastapi(self):
        self.assertEqual(_resolve_tts_provider_name("kokoro"), "kokoro_fastapi")

    def test_canonical_names_pass_through_unchanged(self):
        for name in ("kokoro_fastapi", "kokoro_onnx", "piper"):
            self.assertEqual(_resolve_tts_provider_name(name), name)

    def test_unknown_names_pass_through_unchanged(self):
        # Unknown provider values still reach ``_make_tts_provider`` which
        # is where the ValueError is raised — keeps the error path close
        # to the misuse rather than swallowed here.
        self.assertEqual(_resolve_tts_provider_name("bogus"), "bogus")

    def test_make_tts_provider_accepts_legacy_alias(self):
        provider = _make_tts_provider("kokoro", "local", audio_track=None)
        self.assertIsInstance(provider, KokoroFastApiTTSProvider)

    def test_make_tts_provider_rejects_unknown_provider(self):
        with self.assertRaises(ValueError) as ctx:
            _make_tts_provider("bogus", "local", audio_track=None)
        self.assertIn("Unknown TTS provider", str(ctx.exception))


class TestPiperAssetAutoFetch(unittest.TestCase):
    """``_ensure_piper_asset`` should hit env override, cache, and download."""

    def test_env_override_returns_existing_file(self):
        from server.core.tts_providers import piper

        with TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "custom.onnx"
            model_path.write_bytes(b"stub model bytes")
            config_path = model_path.with_suffix(".onnx.json")
            config_path.write_text('{"sample_rate": 22050}')

            env = {
                "PIPER_MODEL_PATH": str(model_path),
                "PIPER_CONFIG_PATH": str(config_path),
            }
            with patch.dict(os.environ, env, clear=False):
                self.assertEqual(piper._ensure_piper_asset("model"), model_path)
                self.assertEqual(piper._ensure_piper_asset("config"), config_path)

    def test_env_override_raises_when_missing(self):
        from server.core.tts_providers import piper

        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.onnx"
            with patch.dict(os.environ, {"PIPER_MODEL_PATH": str(missing)}):
                with self.assertRaises(FileNotFoundError):
                    piper._ensure_piper_asset("model")

    def test_config_derived_from_model_override_directory(self):
        from server.core.tts_providers import piper

        with TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "voice.onnx"
            model_path.write_bytes(b"stub")
            config_path = model_path.with_suffix(".onnx.json")
            config_path.write_text('{"sample_rate": 22050}')
            env = {"PIPER_MODEL_PATH": str(model_path)}
            # Explicitly unset PIPER_CONFIG_PATH so the model-derived
            # fallback path is exercised.
            with patch.dict(os.environ, env, clear=False):
                os.environ.pop("PIPER_CONFIG_PATH", None)
                self.assertEqual(piper._ensure_piper_asset("config"), config_path)

    def test_cached_asset_is_used_when_present(self):
        from server.core.tts_providers import piper

        with TemporaryDirectory() as tmp:
            fake_home = Path(tmp)
            env = {"XDG_CACHE_HOME": str(fake_home)}
            with patch.dict(os.environ, env, clear=False):
                for k in ("PIPER_MODEL_PATH", "PIPER_CONFIG_PATH"):
                    os.environ.pop(k, None)
                cache_dir = fake_home / "mulive" / "piper"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cached_model = cache_dir / f"{piper.DEFAULT_PIPER_VOICE}.onnx"
                cached_model.write_bytes(b"cached-onnx")

                # Point the ext/piper bundled path at a nonexistent
                # directory to force falling through to the cache.
                with patch.object(
                    piper, "PROJECT_ROOT", Path(tmp) / "no-such-repo"
                ):
                    self.assertEqual(piper._ensure_piper_asset("model"), cached_model)

    def test_missing_asset_triggers_download(self):
        from server.core.tts_providers import piper

        downloads = []

        def _fake_download(url: str, dest: Path) -> Path:
            downloads.append((url, dest))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"downloaded-bytes")
            return dest

        with TemporaryDirectory() as tmp:
            env = {"XDG_CACHE_HOME": str(tmp)}
            with patch.dict(os.environ, env, clear=False):
                for k in ("PIPER_MODEL_PATH", "PIPER_CONFIG_PATH"):
                    os.environ.pop(k, None)
                with patch.object(piper, "PROJECT_ROOT", Path(tmp) / "no-such-repo"):
                    with patch.object(piper, "_download_piper_asset", _fake_download):
                        result = piper._ensure_piper_asset("model")

                self.assertEqual(len(downloads), 1)
                url, dest = downloads[0]
                self.assertTrue(url.endswith(f"{piper.DEFAULT_PIPER_VOICE}.onnx"))
                self.assertEqual(result, dest)
                self.assertEqual(dest.read_bytes(), b"downloaded-bytes")


if __name__ == "__main__":
    unittest.main()
