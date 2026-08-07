"""Load the numeric DRA-Guard experiment configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "dra_guard.yaml"


@lru_cache(maxsize=1)
def load_dra_guard_config() -> dict[str, Any]:
    """Load and cache configs/dra_guard.yaml."""
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Missing DRA-Guard configuration: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"DRA-Guard configuration must be a mapping: {CONFIG_PATH}")
    return config


def get_config_section(*keys: str) -> dict[str, Any]:
    """Return a nested configuration mapping and fail clearly on invalid keys."""
    value: Any = load_dra_guard_config()
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            path = ".".join(keys)
            raise KeyError(f"Missing DRA-Guard configuration section: {path}")
        value = value[key]
    if not isinstance(value, dict):
        path = ".".join(keys)
        raise TypeError(f"DRA-Guard configuration section is not a mapping: {path}")
    return value
