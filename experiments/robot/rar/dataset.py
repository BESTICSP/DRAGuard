


from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.robot.rar.config import (
    DEFAULT_MASK_SIZE,
    DINO_PATCH_SIZE,
    IMG_SIZE,
    MAE_GRID_MASK_SIZES,
    MAE_GRID_MASK_WEIGHTS,
    MAE_PATCH_SIZE,
    MASK_MARGIN_PATCHES,
    TRAINING_SEED,
)


MAE_GRID = IMG_SIZE // MAE_PATCH_SIZE
MAE_NUM_PATCHES = MAE_GRID * MAE_GRID

DINO_GRID = IMG_SIZE // DINO_PATCH_SIZE
DINO_NUM_TOKENS = DINO_GRID * DINO_GRID


def sample_rect_mask(
    mae_grid_mask_size: Tuple[int, int] = DEFAULT_MASK_SIZE,
    img_size: int = IMG_SIZE,
    margin_patches: int = MASK_MARGIN_PATCHES,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:


    if rng is None:
        rng = np.random.default_rng()

    mh, mw = mae_grid_mask_size


    max_row = MAE_GRID - mh - margin_patches
    max_col = MAE_GRID - mw - margin_patches
    row0 = int(rng.integers(margin_patches, max(margin_patches + 1, max_row + 1)))
    col0 = int(rng.integers(margin_patches, max(margin_patches + 1, max_col + 1)))
    row1 = row0 + mh
    col1 = col0 + mw


    mae_grid_mask = torch.zeros(MAE_GRID, MAE_GRID, dtype=torch.bool)
    mae_grid_mask[row0:row1, col0:col1] = True
    mae_patch_mask = mae_grid_mask.reshape(-1)


    px1 = col0 * MAE_PATCH_SIZE
    py1 = row0 * MAE_PATCH_SIZE
    px2 = col1 * MAE_PATCH_SIZE
    py2 = row1 * MAE_PATCH_SIZE

    mask_pixel = torch.zeros(1, img_size, img_size, dtype=torch.float32)
    mask_pixel[0, py1:py2, px1:px2] = 1.0


    dino_col0 = px1 // DINO_PATCH_SIZE
    dino_row0 = py1 // DINO_PATCH_SIZE
    dino_col1 = min((px2 + DINO_PATCH_SIZE - 1) // DINO_PATCH_SIZE, DINO_GRID)
    dino_row1 = min((py2 + DINO_PATCH_SIZE - 1) // DINO_PATCH_SIZE, DINO_GRID)

    dino_grid_mask = torch.zeros(DINO_GRID, DINO_GRID, dtype=torch.bool)
    dino_grid_mask[dino_row0:dino_row1, dino_col0:dino_col1] = True
    dino_token_mask = dino_grid_mask.reshape(-1)

    return mask_pixel, mae_patch_mask, dino_token_mask


def make_collate_fn(
    mask_sizes: List[Tuple[int, int]] = MAE_GRID_MASK_SIZES,
    mask_weights: List[float] = MAE_GRID_MASK_WEIGHTS,
):


    rng = np.random.default_rng()
    weights = np.array(mask_weights, dtype=np.float64)
    weights = weights / weights.sum()

    def collate_fn(batch_items):

        size_idx = int(rng.choice(len(mask_sizes), p=weights))
        chosen_size = mask_sizes[size_idx]


        new_batch = []
        for item in batch_items:
            mask_pixel, mae_patch_mask, dino_token_mask = sample_rect_mask(
                mae_grid_mask_size=chosen_size, rng=rng
            )
            new_batch.append({
                "image": item["image"],
                "mask_pixel": mask_pixel,
                "mae_patch_mask": mae_patch_mask,
                "dino_token_mask": dino_token_mask,
            })


        return {
            "image": torch.stack([b["image"] for b in new_batch]),
            "mask_pixel": torch.stack([b["mask_pixel"] for b in new_batch]),
            "mae_patch_mask": torch.stack([b["mae_patch_mask"] for b in new_batch]),
            "dino_token_mask": torch.stack([b["dino_token_mask"] for b in new_batch]),
        }

    return collate_fn


class CleanFrameDataset(Dataset):


    def __init__(
        self,
        image_dir: str,
        img_size: int = IMG_SIZE,
        mask_sizes: List[Tuple[int, int]] = MAE_GRID_MASK_SIZES,
        max_frames: Optional[int] = None,
        seed: int = TRAINING_SEED,
    ):
        self.img_size = img_size
        self.mask_sizes = mask_sizes
        self.rng = np.random.default_rng(seed)


        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        image_dir = Path(image_dir)
        self.image_paths = sorted([
            p for ext in exts
            for p in image_dir.rglob(f"*{ext}")
            if p.is_file()
        ])
        if max_frames is not None:
            self.image_paths = self.image_paths[:max_frames]

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        import cv2


        img_path = str(self.image_paths[idx])
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to read {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)


        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


        size_idx = int(self.rng.integers(0, len(self.mask_sizes)))
        mask_pixel, mae_patch_mask, dino_token_mask = sample_rect_mask(
            mae_grid_mask_size=self.mask_sizes[size_idx], rng=self.rng
        )

        return {
            "image": img_t,
            "mask_pixel": mask_pixel,
            "mae_patch_mask": mae_patch_mask,
            "dino_token_mask": dino_token_mask,
        }
