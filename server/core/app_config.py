from pathlib import Path

import yaml

from .server_config import web_config

UI_API_VERSION = 2


def _app_asset_url(value):
    if not value:
        return None
    if not isinstance(value, str):
        raise ValueError("App UI asset paths must be strings")
    if not value.startswith("/app-assets/") or ".." in value.split("/"):
        raise ValueError(f"App UI asset path must be under /app-assets/: {value}")
    return value


def load_app_config(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_bundle_config_with_profile(config_path):
    """Load a per-app config with the same defaults + profile stacking as ``run_apps``.

    Standalone entrypoints (``python -m muapps.<app>.app``) historically
    called :func:`load_app_config` directly on their ``config.yml``, so
    they saw only the per-app file — missing the shared ``defaults:``
    block in ``muapps/apps.yml`` and any ``MULIVE_PROFILE`` overrides.
    That divergence produced silent misroutes on AWS (e.g. TTS falling
    back to ``kokoro_fastapi`` because the per-app file redeclared it).

    This helper reproduces the same merge that
    ``server.apps.loader.load_app_bundle`` performs:

    1. Walk up from ``config_path`` to find the enclosing ``apps.yml``.
    2. If found (and schema 2), collect the infra layers (``defaults:``
       plus every ``apps.<profile>.yml`` for the active profiles).
    3. Merge those infra layers under the bundle's ``config.yml`` and
       every ``config.<profile>.yml`` next to it.
    4. If no ``apps.yml`` is found, fall back to the plain single-file
       load so callers outside the muapps tree keep working.
    """
    from server.apps.config_merge import (
        active_profiles,
        load_bundle_layers,
        load_infra_layers,
    )

    config_path = Path(config_path).resolve()
    bundle_dir = config_path.parent
    catalog_path = _find_apps_catalog(bundle_dir)
    if catalog_path is None:
        return load_app_config(config_path)

    raw_catalog = load_app_config(catalog_path)
    if raw_catalog.get("schema") != 2:
        return load_app_config(config_path)

    profiles = active_profiles()
    infra, _ = load_infra_layers(catalog_path, raw_catalog, profiles)
    merged, _ = load_bundle_layers(bundle_dir, config_path, infra, profiles)
    return merged


def _find_apps_catalog(start_dir):
    """Find the nearest enclosing ``apps.yml`` above ``start_dir`` (inclusive).

    We only walk up two levels — the catalog is expected to sit at
    ``muapps/apps.yml`` next to the bundle directory, so
    ``bundle_dir.parent / 'apps.yml'`` is the natural first hit. A second
    parent hop covers unusual layouts (e.g. a bundle nested one level
    deep). Beyond that we don't guess — callers can point at an explicit
    catalog via the env var below.
    """
    import os

    override = os.environ.get("MULIVE_APPS_CATALOG")
    if override:
        candidate = Path(override).resolve()
        return candidate if candidate.is_file() else None

    for parent in (start_dir, start_dir.parent, start_dir.parent.parent):
        candidate = parent / "apps.yml"
        if candidate.is_file():
            return candidate.resolve()
    return None


def app_section(config, name, default=None):
    value = config.get(name)
    if value is None:
        return default or {}
    return value


def build_web_config(config, use_https=None):
    server = app_section(config, "server")
    app = app_section(config, "app")
    if use_https is None:
        use_https = server.get("https", True)
    overrides = {key: value for key, value in server.items() if key != "https"}
    overrides["client_config"] = {
        "app_name": app.get("display_name") or app.get("name") or "Mulive",
        "ui_api_version": UI_API_VERSION,
        "ui_module": _app_asset_url(app.get("ui_module")),
        "ui_stylesheet": _app_asset_url(app.get("ui_stylesheet")),
    }
    return web_config(use_https=use_https, **overrides)


def resolve_app_path(base_dir, path):
    path = Path(path)
    if path.is_absolute():
        return path
    return base_dir / path
