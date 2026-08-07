"""Shared constants for the public DRA-Guard robot experiments."""

from pathlib import Path

from experiments.robot.config_loader import get_config_section


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_CONFIG = get_config_section("runtime")

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

IMAGE_SIZE = int(RUNTIME_CONFIG["image_size"])
ENVIRONMENT_RESOLUTION = int(RUNTIME_CONFIG["environment_resolution"])
ENVIRONMENT_SEED = int(RUNTIME_CONFIG["environment_seed"])
CENTER_CROP_SCALE = float(RUNTIME_CONFIG["center_crop_scale"])
ROLLOUT_FPS = int(RUNTIME_CONFIG["rollout_fps"])
DINO_PATCH_SIZE = int(RUNTIME_CONFIG["dino_patch_size"])
DINO_NUM_PREFIX_TOKENS = int(RUNTIME_CONFIG["dino_num_prefix_tokens"])
DINO_TOKEN_GRID_SIZE = IMAGE_SIZE // DINO_PATCH_SIZE
DINO_NUM_PATCH_TOKENS = DINO_TOKEN_GRID_SIZE * DINO_TOKEN_GRID_SIZE
SUITE_MAX_STEPS = {
    suite: int(steps)
    for suite, steps in RUNTIME_CONFIG["suite_max_steps"].items()
}
NUM_STEPS_WAIT = int(RUNTIME_CONFIG["warmup_steps"])
DINO_TARGET_LAYERS = [int(layer) for layer in RUNTIME_CONFIG["dino_target_layers"]]


def _parse_attack_patch_positions(value):
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(not isinstance(item, (list, tuple)) for item in value)
    ):
        value = [value]

    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Attack patch positions must contain at least one [x, y] pair.")

    positions = []
    for position in value:
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            raise ValueError(f"Invalid attack patch position: {position!r}")
        positions.append(tuple(int(coordinate) for coordinate in position))
    return tuple(positions)


ATTACK_PATCH_POSITIONS_BY_SUITE = {
    suite: _parse_attack_patch_positions(positions)
    for suite, positions in RUNTIME_CONFIG["attack_patch_positions"].items()
}
ATTACK_PATCH_POSITION_BY_SUITE = {
    suite: positions[0]
    for suite, positions in ATTACK_PATCH_POSITIONS_BY_SUITE.items()
}

DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models" / "openvla"

SUITE_MODEL_DIRS = {
    "libero_spatial": "openvla-7b-finetuned-libero-spatial",
    "libero_object": "openvla-7b-finetuned-libero-object",
    "libero_goal": "openvla-7b-finetuned-libero-goal",
    "libero_10": "openvla-7b-finetuned-libero-10",
}

SUITE_MODEL_REPOS = {
    suite: f"openvla/{directory}"
    for suite, directory in SUITE_MODEL_DIRS.items()
}
