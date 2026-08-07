"""Download the OpenVLA checkpoints required by the public experiments."""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from experiments.robot.settings import (
    DEFAULT_MODEL_ROOT,
    SUITES,
    SUITE_MODEL_DIRS,
    SUITE_MODEL_REPOS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_root",
        type=Path,
        default=Path(os.environ.get("MODEL_ROOT", DEFAULT_MODEL_ROOT)),
        help="Directory in which suite-specific OpenVLA checkpoints are stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_root = args.model_root.expanduser().resolve()
    model_root.mkdir(parents=True, exist_ok=True)

    for suite in SUITES:
        repo_id = SUITE_MODEL_REPOS[suite]
        local_dir = model_root / SUITE_MODEL_DIRS[suite]
        print(f"Downloading {repo_id} to {local_dir}")
        snapshot_download(repo_id=repo_id, local_dir=local_dir)


if __name__ == "__main__":
    main()
