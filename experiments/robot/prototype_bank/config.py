
from pathlib import Path

from experiments.robot.config_loader import get_config_section
from experiments.robot.settings import (
    DINO_NUM_PATCH_TOKENS as NUM_PATCH_TOKENS,
    DINO_NUM_PREFIX_TOKENS as NUM_PREFIX_TOKENS,
    DINO_TARGET_LAYERS as TARGET_LAYERS,
    NUM_STEPS_WAIT,
    SUITE_MAX_STEPS as MAX_STEPS,
    SUITES,
)

PROTOTYPE_CONFIG = get_config_section("prototype_bank")
COLLECTION_CONFIG = PROTOTYPE_CONFIG["collection"]
CLUSTERING_CONFIG = PROTOTYPE_CONFIG["clustering"]


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOKENS_DIR = PROJECT_ROOT / "data/prototype_bank/tokens"
PROTOTYPE_BANK_DIR = PROJECT_ROOT / "data/prototype_bank/prototypes"
LOG_DIR = PROJECT_ROOT / "outputs/prototype_bank"

for _d in (TOKENS_DIR, PROTOTYPE_BANK_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SUITE_TO_IDX = {s: i for i, s in enumerate(SUITES)}


TOWER = "dinov2"
EMBED_DIM = int(PROTOTYPE_CONFIG["embedding_dim"])


LAYER_COMBOS = {
    "L18-22": list(TARGET_LAYERS),
}


NUM_SUCCESS_PER_TASK = int(COLLECTION_CONFIG["successes_per_task"])
NUM_FAIL_PER_TASK = int(COLLECTION_CONFIG["failures_per_task"])
FRAMES_PER_ROLLOUT = int(COLLECTION_CONFIG["frames_per_rollout"])
MINIMUM_POLICY_STEPS = int(COLLECTION_CONFIG["minimum_policy_steps"])
COLLECT_SEED = int(COLLECTION_CONFIG["seed"])


K_PROTOTYPES = int(CLUSTERING_CONFIG["num_prototypes"])
KMEANS_BATCH = int(CLUSTERING_CONFIG["batch_size"])
KMEANS_ITER = int(CLUSTERING_CONFIG["max_iterations"])
KMEANS_SEED = int(CLUSTERING_CONFIG["seed"])
KMEANS_N_INIT = int(CLUSTERING_CONFIG["num_initializations"])
L2_NORMALIZE_BEFORE_KMEANS = True
