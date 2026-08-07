"""Resolve OpenVLA checkpoints from local model folders or Hugging Face."""

import os
from pathlib import Path
from typing import Optional

from experiments.robot.settings import (
    DEFAULT_MODEL_ROOT,
    SUITE_MODEL_DIRS,
    SUITE_MODEL_REPOS,
)


def is_local_checkpoint(path: Path) -> bool:
    """Return whether a directory has the minimum Hugging Face checkpoint metadata."""
    return path.is_dir() and (path / "config.json").is_file()


def resolve_openvla_checkpoint(suite: str, model_root: Optional[str] = None) -> str:


    if suite not in SUITE_MODEL_DIRS:
        raise ValueError(f"Unsupported LIBERO suite: {suite}")

    configured_root = model_root or os.environ.get("MODEL_ROOT")
    root = Path(configured_root).expanduser() if configured_root else DEFAULT_MODEL_ROOT
    local_path = root / SUITE_MODEL_DIRS[suite]
    if configured_root is not None or is_local_checkpoint(local_path):
        return str(local_path)
    return SUITE_MODEL_REPOS[suite]
