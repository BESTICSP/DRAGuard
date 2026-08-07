"""Shared configuration for the Insight experiments."""

from pathlib import Path

from experiments.robot.settings import (
    ATTACK_PATCH_POSITION_BY_SUITE,
    SUITE_MAX_STEPS as DEFAULT_MAX_STEPS_BY_SUITE,
    SUITES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "analysis" / "insight"
TOKEN_ROOT = OUTPUT_ROOT / "tokens"
ROLLOUT_ROOT = PROJECT_ROOT / "rollouts" / "analysis" / "insight"


DEFAULT_COLLECTION_STEPS = 30


ATTACK_PATCH_GLOBS = {
    "upa": "simulation/untargeted/UPA-*/patch.pt",
    "uada": "simulation/untargeted/UADA-*/patch.pt",
    "tma": "simulation/targeted/T-*/patch.pt",
}

DEFAULT_TASK_IDS = list(range(10))
DEFAULT_ANALYSIS_FRAMES = [6, 7, 8, 9, 10, 11, 12]
DEFAULT_SOURCE_TASK = 0
DEFAULT_TARGET_TASKS = list(range(10))
DEFAULT_SEED = 7
DEFAULT_LAYERS = (18, 18)

IMAGE_SIZE = 224
TOKEN_GRID_SIZE = 16
NUM_PATCH_TOKENS = TOKEN_GRID_SIZE * TOKEN_GRID_SIZE


def token_file(suite: str, attack: str, task_id: int, token_root: Path = TOKEN_ROOT) -> Path:
    """Return the canonical path for a task-level representation file."""
    return token_root / suite / attack.lower() / f"task_{task_id:02d}.pt"
