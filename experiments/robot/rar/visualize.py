


from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from experiments.robot.rar.config import (
    DINO_PATCH_SIZE,
    IMG_SIZE,
    LOG_DIR,
    TARGET_LAYERS,
    VISUALIZATION_SAMPLES,
)

DINO_GRID_SIZE = IMG_SIZE // DINO_PATCH_SIZE


def save_sample_comparison(
    image: torch.Tensor,
    mask_pixel: torch.Tensor,
    img_masked: torch.Tensor,
    reconstructed: torch.Tensor,
    anomaly_before: Optional[np.ndarray] = None,
    anomaly_after: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    epoch: int = 0,
    sample_idx: int = 0,
):


    img_np = image.cpu().permute(1, 2, 0).numpy().clip(0, 1)
    mask_np = mask_pixel.cpu().squeeze(0).numpy()
    masked_np = img_masked.cpu().permute(1, 2, 0).numpy().clip(0, 1)
    rec_np = reconstructed.cpu().permute(1, 2, 0).numpy().clip(0, 1)

    has_anomaly = anomaly_before is not None and anomaly_after is not None
    n_cols = 6 if has_anomaly else 4

    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5))

    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Mask", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(masked_np)
    axes[2].set_title("Masked Input", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(rec_np)
    axes[3].set_title("Reconstructed", fontsize=10)
    axes[3].axis("off")

    if has_anomaly:
        vmin = min(anomaly_before.min(), anomaly_after.min())
        vmax = max(anomaly_before.max(), anomaly_after.max())

        im4 = axes[4].imshow(anomaly_before, cmap="hot", vmin=vmin, vmax=vmax,
                             interpolation="nearest")
        axes[4].set_title(f"Anomaly Before\nmax={anomaly_before.max():.3f}", fontsize=9)
        axes[4].axis("off")

        im5 = axes[5].imshow(anomaly_after, cmap="hot", vmin=vmin, vmax=vmax,
                             interpolation="nearest")
        axes[5].set_title(f"Anomaly After\nmax={anomaly_after.max():.3f}", fontsize=9)
        axes[5].axis("off")

        fig.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04)

    plt.suptitle(f"Epoch {epoch}, Sample {sample_idx}", fontsize=11)
    plt.tight_layout()

    if save_path is None:
        vis_dir = LOG_DIR / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(vis_dir / f"epoch{epoch:03d}_sample{sample_idx:02d}.png")

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_batch_comparison(
    model,
    batch: Dict[str, torch.Tensor],
    epoch: int,
    device: str,
    n_samples: int = VISUALIZATION_SAMPLES,
) -> List[str]:


    from experiments.robot.rar.model import compute_differentiable_anomaly_score

    model.mae.eval()
    saved_paths = []

    with torch.no_grad():
        image = batch["image"][:n_samples].to(device)
        mask_pixel = batch["mask_pixel"][:n_samples].to(device)
        mae_patch_mask = batch["mae_patch_mask"][:n_samples].to(device)
        dino_token_mask = batch["dino_token_mask"][:n_samples].to(device)

        img_masked = image * (1.0 - mask_pixel)


        mae_out = model.mae(img_masked, mae_patch_mask, mask_pixel)
        reconstructed = mae_out["reconstructed"]


        tokens_before = model.dinov2(img_masked)
        tokens_after = model.dinov2(reconstructed)

        for i in range(min(n_samples, image.shape[0])):

            scores_before = []
            scores_after = []
            for L in model.target_layers:
                if L in tokens_before and L in model.banks:
                    sb = compute_differentiable_anomaly_score(
                        tokens_before[L][i:i+1], model.banks[L]
                    )
                    sa = compute_differentiable_anomaly_score(
                        tokens_after[L][i:i+1], model.banks[L]
                    )
                    scores_before.append(sb[0])
                    scores_after.append(sa[0])

            if scores_before:
                amap_before = (
                    torch.stack(scores_before)
                    .mean(0)
                    .cpu()
                    .numpy()
                    .reshape(DINO_GRID_SIZE, DINO_GRID_SIZE)
                )
                amap_after = (
                    torch.stack(scores_after)
                    .mean(0)
                    .cpu()
                    .numpy()
                    .reshape(DINO_GRID_SIZE, DINO_GRID_SIZE)
                )
            else:
                amap_before = np.zeros((DINO_GRID_SIZE, DINO_GRID_SIZE))
                amap_after = np.zeros((DINO_GRID_SIZE, DINO_GRID_SIZE))

            path = save_sample_comparison(
                image=image[i],
                mask_pixel=mask_pixel[i],
                img_masked=img_masked[i],
                reconstructed=reconstructed[i],
                anomaly_before=amap_before,
                anomaly_after=amap_after,
                epoch=epoch,
                sample_idx=i,
            )
            saved_paths.append(path)

    model.mae.train()
    return saved_paths


def plot_training_curves(history: List[Dict], save_path: Optional[str] = None):


    if not history:
        return

    epochs = [h["epoch"] for h in history]

    train_rec = [h["train"]["loss_rec"] for h in history]
    train_alignment_loss = [h["train"]["loss_align"] for h in history]
    val_rec = [h["val"]["loss_rec"] for h in history]
    val_alignment_loss = [h["val"]["loss_align"] for h in history]

    val_anomaly_before = [h["val"].get("anomaly_before", 0) for h in history]
    val_anomaly_after = [h["val"].get("anomaly_after", 0) for h in history]
    val_anomaly_reduction = [h["val"].get("anomaly_reduction", 0) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))


    ax = axes[0, 0]
    ax.plot(epochs, train_rec, "b-o", markersize=3, label="train")
    ax.plot(epochs, val_rec, "r-o", markersize=3, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L_rec (L1 pixel)")
    ax.set_title("Pixel Reconstruction Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)


    ax = axes[0, 1]
    ax.plot(epochs, train_alignment_loss, "b-o", markersize=3, label="train")
    ax.plot(epochs, val_alignment_loss, "r-o", markersize=3, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L_align")
    ax.set_title("Representation Alignment Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)


    ax = axes[1, 0]
    ax.plot(epochs, val_anomaly_before, "r-s", markersize=4, label="before (masked)")
    ax.plot(epochs, val_anomaly_after, "g-s", markersize=4, label="after (reconstructed)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Anomaly Score (mask region)")
    ax.set_title("Anomaly Score: Before vs After")
    ax.legend()
    ax.grid(True, alpha=0.3)


    ax = axes[1, 1]
    ax.plot(epochs, val_anomaly_reduction, "k-D", markersize=4)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(epochs, 0, val_anomaly_reduction,
                    where=[r > 0 for r in val_anomaly_reduction],
                    alpha=0.2, color="green", label="effective")
    ax.fill_between(epochs, 0, val_anomaly_reduction,
                    where=[r <= 0 for r in val_anomaly_reduction],
                    alpha=0.2, color="red", label="ineffective")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Anomaly Reduction")
    ax.set_title("Anomaly Reduction (before - after)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Representation-Aligned Reconstruction Training", fontsize=13)
    plt.tight_layout()

    if save_path is None:
        save_path = str(LOG_DIR / "training_curves.png")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
