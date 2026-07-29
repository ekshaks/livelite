import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from server.core.app_config import UI_API_VERSION


GAME_ID_RE = re.compile(r"^[a-z][a-z0-9-]*$")


@dataclass(frozen=True)
class GameDefinition:
    id: str
    title: str
    description: str
    capabilities: tuple[str, ...]
    bundle_dir: Path
    assets_dir: Path | None
    ui_module: str | None
    ui_stylesheet: str | None
    transport_config: dict[str, Any]
    config: dict[str, Any]
    session_runner_factory: Callable[[dict[str, Any]], Callable]
    enabled: bool = True

    def create_session_runner(self):
        """Create the per-connection session entry point."""
        return self.session_runner_factory(deepcopy(self.config))

    def public_metadata(self, *, include_ui: bool = False) -> dict[str, Any]:
        metadata = {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "capabilities": list(self.capabilities),
            "available": True,
        }
        if include_ui:
            metadata.update(
                {
                    "app_name": self.title,
                    "ui_api_version": UI_API_VERSION,
                    "ui_module": self.ui_module,
                    "ui_stylesheet": self.ui_stylesheet,
                }
            )
        return metadata


@dataclass(frozen=True)
class UnavailableGame:
    id: str
    title: str
    bundle_path: str
    reason: str

    def public_metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": "This game is currently unavailable.",
            "capabilities": [],
            "available": False,
        }


class GameRegistry:
    def __init__(self):
        self._games: dict[str, GameDefinition] = {}
        self._entries: list[GameDefinition | UnavailableGame] = []

    def register(self, game: GameDefinition) -> None:
        if not GAME_ID_RE.fullmatch(game.id):
            raise ValueError(f"Invalid game id: {game.id!r}")
        if game.id in self._games:
            raise ValueError(f"Duplicate game id: {game.id}")
        self._games[game.id] = game
        self._entries.append(game)

    def record_unavailable(
        self,
        *,
        game_id: str,
        title: str,
        bundle_path: str,
        reason: str,
    ) -> None:
        self._entries.append(
            UnavailableGame(
                id=game_id,
                title=title,
                bundle_path=bundle_path,
                reason=reason,
            )
        )

    def get(self, game_id: str) -> GameDefinition | None:
        game = self._games.get(game_id)
        return game if game is not None and game.enabled else None

    def require(self, game_id: str) -> GameDefinition:
        game = self.get(game_id)
        if game is None:
            raise KeyError(game_id)
        return game

    def enabled_games(self) -> tuple[GameDefinition, ...]:
        return tuple(game for game in self._games.values() if game.enabled)

    def unavailable_games(self) -> tuple[UnavailableGame, ...]:
        return tuple(
            entry for entry in self._entries if isinstance(entry, UnavailableGame)
        )

    def public_games(self) -> list[dict[str, Any]]:
        return [
            entry.public_metadata()
            for entry in self._entries
            if isinstance(entry, UnavailableGame) or entry.enabled
        ]
