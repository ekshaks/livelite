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
