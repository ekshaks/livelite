from pathlib import Path

from aiohttp import web


class GameRoutes:
    """Attach dashboard and game-specific HTTP routes to an aiohttp app."""

    def __init__(
        self,
        *,
        registry,
        user_directory,
        client_html_path: Path,
        dashboard_html_path: Path,
        accept_offer,
        base_transport_config: dict,
    ):
        self.registry = registry
        if user_directory is None:
            raise ValueError("Game dashboard requires a user directory")
        self.user_directory = user_directory
        self.client_html_path = Path(client_html_path)
        self.dashboard_html_path = Path(dashboard_html_path)
        self.accept_offer = accept_offer
        self.base_transport_config = dict(base_transport_config)

    def register(self, app: web.Application) -> None:
        app.router.add_get("/", self.dashboard)
        app.router.add_get("/api/games", self.list_games)
        app.router.add_get("/api/users", self.list_users)
        app.router.add_get("/api/games/{game_id}", self.game_config)
        app.router.add_get("/games/{game_id}", self.game_page)
        app.router.add_post("/games/{game_id}/offer", self.game_offer)

        for game in self.registry.enabled_games():
            if game.assets_dir is None:
                continue
            app.router.add_static(
                f"/game-assets/{game.id}/",
                path=game.assets_dir,
                name=f"game_assets_{game.id}",
                follow_symlinks=False,
            )

    async def dashboard(self, request):
        return web.FileResponse(self.dashboard_html_path)

    async def list_games(self, request):
        return web.json_response(self.registry.public_games())

    async def list_users(self, request):
        return web.json_response(self.user_directory.to_client_dict())

    async def game_config(self, request):
        game = self._requested_game(request)
        return web.json_response(game.public_metadata(include_ui=True))

    async def game_page(self, request):
        self._requested_game(request)
        return web.FileResponse(self.client_html_path)

    async def game_offer(self, request):
        game = self._requested_game(request)
        try:
            profile = self.user_directory.resolve(request.query.get("user_id"))
        except KeyError:
            raise web.HTTPBadRequest(text="Unknown user profile")

        session_runner = game.create_session_runner()

        async def run_user_session(session):
            session.user_id = profile.user_id
            await session_runner(session)

        config = {**self.base_transport_config, **game.transport_config}
        return await self.accept_offer(
            request,
            run_user_session,
            config,
        )

    def _requested_game(self, request):
        game_id = request.match_info.get("game_id", "")
        game = self.registry.get(game_id)
        if game is None:
            raise web.HTTPNotFound(text=f"Unknown or disabled game: {game_id}")
        return game
