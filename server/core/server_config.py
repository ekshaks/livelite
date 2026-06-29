from pathlib import Path


def ssl_config():
    server_dir = Path(__file__).parent.parent.resolve()
    ssl_keyfile = server_dir / "certs/key.pem"
    ssl_certfile = server_dir / "certs/cert.pem"
    assert ssl_keyfile.exists(), f"SSL key file not found: {ssl_keyfile}"
    assert ssl_certfile.exists(), f"SSL cert file not found: {ssl_certfile}"
    return {
        "ssl_keyfile": str(ssl_keyfile),
        "ssl_certfile": str(ssl_certfile),
    }


def web_config(use_https: bool = True, **overrides):
    config = {
        "debug": False,
    }
    if use_https:
        config.update(ssl_config())
    config.update(overrides)
    return config
