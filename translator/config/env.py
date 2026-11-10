"""Loading of environment variables from a .env file."""

from __future__ import annotations

import os
from pathlib import Path

from translator.config.settings import ENV_PATH


def load_env(path: Path | None = None) -> None:
    """Parse KEY=VALUE lines from a .env file and set them in os.environ.

    Idempotent: calling it multiple times with the same file is safe because
    os.environ is updated in-place (same values overwrite themselves).
    """
    env_path = path or ENV_PATH
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def ensure_env_loaded() -> None:
    """Load the default .env file if the API key is not yet present in the environment."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        load_env()


def get_env(name: str, default: str = "") -> str:
    ensure_env_loaded()
    return os.environ.get(name, default)
