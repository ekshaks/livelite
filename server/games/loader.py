import importlib
import re
import sys
from pathlib import Path
from typing import Any

from server.core.app_config import app_section, load_app_config

from .registry import GAME_ID_RE, GameDefinition, GameRegistry


def load_game_catalog(path: Path) -> tuple[GameRegistry, dict[str, Any]]:
    catalog_path = Path(path).resolve()
    catalog = load_app_config(catalog_path)
    if catalog.get("schema") != 1:
        raise ValueError("Game catalog schema must be 1")

    entries = catalog.get("games")
    if not isinstance(entries, list):
        raise ValueError("Game catalog games must be a list")

    registry = GameRegistry()
    for index, entry in enumerate(entries):
        enabled = not isinstance(entry, dict) or bool(entry.get("enabled", True))
        try:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError("Each catalog game must have a path")
            bundle_dir = _resolve_inside(
                catalog_path.parent,
                entry["path"],
                directory=True,
            )
            registry.register(
                load_game_bundle(
                    bundle_dir,
                    enabled=enabled,
                )
            )
        except Exception as exc:
            if not enabled:
                continue
            game_id, title, bundle_path = _unavailable_identity(
                catalog_path.parent,
                entry,
                index,
            )
            registry.record_unavailable(
                game_id=game_id,
                title=title,
                bundle_path=bundle_path,
                reason=f"{type(exc).__name__}: {exc}",
            )

    return registry, catalog


def load_game_bundle(bundle_dir: Path, *, enabled: bool = True) -> GameDefinition:
    bundle_dir = Path(bundle_dir).resolve()
    manifest = load_app_config(bundle_dir / "game.yml")
    if manifest.get("schema") != 1:
        raise ValueError(f"{bundle_dir}: game manifest schema must be 1")

    game_id = manifest.get("id")
    if not isinstance(game_id, str) or not GAME_ID_RE.fullmatch(game_id):
        raise ValueError(f"{bundle_dir}: invalid game id")
    title = _required_text(manifest, "title", bundle_dir)
    description = _required_text(manifest, "description", bundle_dir)

    capabilities = manifest.get("capabilities")
    if (
        not isinstance(capabilities, list)
        or not capabilities
        or any(not isinstance(item, str) or not item for item in capabilities)
    ):
        raise ValueError(f"{bundle_dir}: capabilities must be non-empty strings")

    backend = manifest.get("backend")
    if not isinstance(backend, dict):
        raise ValueError(f"{bundle_dir}: backend must be an object")
    entrypoint = _required_text(backend, "entrypoint", bundle_dir)
    config_path = _resolve_inside(
        bundle_dir,
        _required_text(backend, "config", bundle_dir),
    )
    config = load_app_config(config_path)
    session_runner_factory = _load_entrypoint(bundle_dir, entrypoint)

    ui = manifest.get("ui") or {}
    if not isinstance(ui, dict):
        raise ValueError(f"{bundle_dir}: ui must be an object")
    module_path = _optional_bundle_web_file(bundle_dir, ui.get("module"))
    stylesheet_path = _optional_bundle_web_file(bundle_dir, ui.get("stylesheet"))
    assets_dir = bundle_dir / "web" if module_path or stylesheet_path else None

    return GameDefinition(
        id=game_id,
        title=title,
        description=description,
        capabilities=tuple(dict.fromkeys(capabilities)),
        bundle_dir=bundle_dir,
        assets_dir=assets_dir,
        ui_module=_asset_url(game_id, assets_dir, module_path),
        ui_stylesheet=_asset_url(game_id, assets_dir, stylesheet_path),
        transport_config=dict(app_section(config, "server")),
        config=config,
        session_runner_factory=session_runner_factory,
        enabled=enabled,
    )


def _load_entrypoint(bundle_dir: Path, entrypoint: str):
    try:
        module_name, attribute = entrypoint.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"{bundle_dir}: invalid backend entrypoint") from exc
    if not module_name or not attribute or "." in module_name:
        raise ValueError(f"{bundle_dir}: entrypoint must be bundle-relative")

    parent = str(bundle_dir.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = importlib.import_module(f"{bundle_dir.name}.{module_name}")
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise ValueError(f"{bundle_dir}: backend entrypoint is not callable")
    return factory


def _required_text(mapping: dict, key: str, owner: Path) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner}: {key} must be a non-empty string")
    return value.strip()


def _resolve_inside(base: Path, relative: str, *, directory: bool = False) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"Bundle paths must be relative: {relative}")
    resolved = (base / candidate).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError(f"Bundle path escapes its owner: {relative}")
    exists = resolved.is_dir() if directory else resolved.is_file()
    if not exists:
        raise ValueError(f"Bundle path does not exist: {resolved}")
    return resolved


def _optional_bundle_web_file(bundle_dir: Path, relative: Any) -> Path | None:
    if relative is None:
        return None
    if not isinstance(relative, str) or not relative.startswith("web/"):
        raise ValueError(f"{bundle_dir}: UI assets must be under web/")
    return _resolve_inside(bundle_dir, relative)


def _asset_url(
    game_id: str,
    assets_dir: Path | None,
    path: Path | None,
) -> str | None:
    if path is None:
        return None
    relative = path.relative_to(assets_dir).as_posix()
    return f"/game-assets/{game_id}/{relative}"


def _unavailable_identity(
    catalog_dir: Path,
    entry: Any,
    index: int,
) -> tuple[str, str, str]:
    raw_path = entry.get("path") if isinstance(entry, dict) else None
    raw_id = entry.get("id") if isinstance(entry, dict) else None
    raw_title = entry.get("title") if isinstance(entry, dict) else None

    manifest = {}
    if isinstance(raw_path, str):
        candidate = Path(raw_path)
        resolved = (catalog_dir / candidate).resolve()
        if (
            not candidate.is_absolute()
            and resolved.is_relative_to(catalog_dir.resolve())
            and (resolved / "game.yml").is_file()
        ):
            try:
                manifest = load_app_config(resolved / "game.yml")
            except Exception:
                manifest = {}

    candidate_id = raw_id or manifest.get("id")
    if not isinstance(candidate_id, str) or not GAME_ID_RE.fullmatch(candidate_id):
        basename = Path(raw_path).name if isinstance(raw_path, str) else ""
        candidate_id = re.sub(r"[^a-z0-9-]+", "-", basename.lower()).strip("-")
        if not candidate_id or not candidate_id[0].isalpha():
            candidate_id = f"game-{index + 1}"

    candidate_title = raw_title or manifest.get("title")
    if not isinstance(candidate_title, str) or not candidate_title.strip():
        candidate_title = candidate_id.replace("-", " ").title()

    return candidate_id, candidate_title.strip(), str(raw_path or "")
