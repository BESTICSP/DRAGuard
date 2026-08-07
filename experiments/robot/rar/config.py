
from pathlib import Path

from experiments.robot.config_loader import get_config_section
from experiments.robot.settings import (
    DINO_NUM_PREFIX_TOKENS as NUM_PREFIX_TOKENS,
    DINO_PATCH_SIZE,
    DINO_TARGET_LAYERS as TARGET_LAYERS,
    IMAGE_SIZE as IMG_SIZE,
)

RAR_CONFIG = get_config_section("rar")
FRAME_COLLECTION_CONFIG = RAR_CONFIG["frame_collection"]
TRAINING_CONFIG = RAR_CONFIG["training"]


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLEAN_FRAMES_DIR = PROJECT_ROOT / "data/rar/clean_frames"


CKPT_DIR = PROJECT_ROOT / "models/rar"
LOG_DIR = PROJECT_ROOT / "outputs/rar"

for _d in (CKPT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


MAE_MODEL_NAME = "facebook/vit-mae-base"


MAE_PATCH_SIZE = int(RAR_CONFIG["mae_patch_size"])
DEFAULT_MASK_SIZE = tuple(int(v) for v in RAR_CONFIG["default_mask_size"])
MASK_MARGIN_PATCHES = int(RAR_CONFIG["mask_margin_patches"])
MAE_GRID_MASK_SIZES = [
    tuple(int(v) for v in size) for size in RAR_CONFIG["mask_sizes"]
]
MAE_GRID_MASK_WEIGHTS = [float(v) for v in RAR_CONFIG["mask_weights"]]
IMAGENET_MEAN = [float(v) for v in RAR_CONFIG["imagenet_mean"]]
IMAGENET_STD = [float(v) for v in RAR_CONFIG["imagenet_std"]]
MASK_TIE_BREAK_NOISE = float(RAR_CONFIG["mask_tie_break_noise"])


FRAMES_PER_TASK = int(FRAME_COLLECTION_CONFIG["frames_per_task"])
FRAME_COLLECTION_SEED = int(FRAME_COLLECTION_CONFIG["seed"])


BATCH_SIZE = int(TRAINING_CONFIG["batch_size"])
NUM_EPOCHS = int(TRAINING_CONFIG["epochs"])
LR = float(TRAINING_CONFIG["learning_rate"])
WEIGHT_DECAY = float(TRAINING_CONFIG["weight_decay"])
GRAD_CLIP = float(TRAINING_CONFIG["gradient_clip"])
LAMBDA_REC = float(TRAINING_CONFIG["reconstruction_weight"])
LAMBDA_ALIGN = float(TRAINING_CONFIG["alignment_weight"])
WARMUP_STEPS = int(TRAINING_CONFIG["warmup_steps"])
TRAINING_SEED = int(TRAINING_CONFIG["seed"])
LOG_INTERVAL = int(TRAINING_CONFIG["log_interval"])
VALIDATION_FRACTION = float(TRAINING_CONFIG["validation_fraction"])
DATALOADER_WORKERS = int(TRAINING_CONFIG["dataloader_workers"])
SCHEDULER_MIN_LR_RATIO = float(TRAINING_CONFIG["scheduler_min_lr_ratio"])
VISUALIZATION_SAMPLES = int(TRAINING_CONFIG["visualization_samples"])
