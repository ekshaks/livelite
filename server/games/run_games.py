import argparse
from pathlib import Path

from server.core.app_config import app_section
from server.core.server_config import web_config
from server.server_asyncio import Server

from .loader import load_game_catalog


DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "muapps" / "games.yml"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Mulive game dashboard.")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)

    tls_group = parser.add_mutually_exclusive_group()
    tls_group.add_argument("--https", action="store_true")
    tls_group.add_argument("--http", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    registry, catalog = load_game_catalog(Path(args.catalog))
    for unavailable in registry.unavailable_games():
        print(
            "Game unavailable: "
            f"{unavailable.title} ({unavailable.bundle_path}): "
            f"{unavailable.reason}"
        )
    server_config = dict(app_section(catalog, "server"))

    use_https = (
        True
        if args.https
        else False
        if args.http
        else server_config.pop("https", True)
    )
    host = args.host or server_config.pop("host", "0.0.0.0")
    port = args.port or int(server_config.pop("port", 9000))

    server = Server(
        game_registry=registry,
        config=web_config(use_https=use_https, **server_config),
    )
    server.run(host=host, port=port)


if __name__ == "__main__":
    main()
