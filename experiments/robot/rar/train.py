


from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.robot.rar.config import (
    BATCH_SIZE,
    CLEAN_FRAMES_DIR,
    CKPT_DIR,
    DATALOADER_WORKERS,
    GRAD_CLIP,
    IMG_SIZE,
    LAMBDA_ALIGN,
    LAMBDA_REC,
    LOG_DIR,
    LOG_INTERVAL,
    LR,
    MAE_MODEL_NAME,
    NUM_EPOCHS,
    SCHEDULER_MIN_LR_RATIO,
    TARGET_LAYERS,
    TRAINING_SEED,
    VALIDATION_FRACTION,
    VISUALIZATION_SAMPLES,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from experiments.robot.rar.dataset import (
    CleanFrameDataset,
    make_collate_fn,
    MAE_GRID_MASK_SIZES,
    MAE_GRID_MASK_WEIGHTS,
)
from experiments.robot.rar.model import (
    RARModel,
    compute_alignment_loss,
    compute_differentiable_anomaly_score,
)
from experiments.robot.rar.visualize import (
    save_batch_comparison,
    plot_training_curves,
)
from experiments.robot.settings import SUITES as ALL_SUITES


@torch.no_grad()
def compute_anomaly_comparison(
    model: RARModel,
    image: torch.Tensor,
    mask_pixel: torch.Tensor,
    mae_patch_mask: torch.Tensor,
    dino_token_mask: torch.Tensor,
) -> dict:


    device = image.device
    B = image.shape[0]
    mask_flat = dino_token_mask.float()


    img_masked = image * (1.0 - mask_pixel)
    tokens_before = model.dinov2(img_masked)
    scores_before_layers = []
    for L in model.target_layers:
        if L in tokens_before and L in model.banks:
            s = compute_differentiable_anomaly_score(
                tokens_before[L], model.banks[L]
            )
            masked_s = (s * mask_flat).sum(dim=1) / mask_flat.sum(dim=1).clamp(min=1)
            scores_before_layers.append(masked_s.mean())
    anomaly_before = torch.stack(scores_before_layers).mean().item() if scores_before_layers else 0.0


    mae_out = model.mae(img_masked, mae_patch_mask, mask_pixel)
    reconstructed = mae_out["reconstructed"]
    tokens_after = model.dinov2(reconstructed)
    scores_after_layers = []
    for L in model.target_layers:
        if L in tokens_after and L in model.banks:
            s = compute_differentiable_anomaly_score(
                tokens_after[L], model.banks[L]
            )
            masked_s = (s * mask_flat).sum(dim=1) / mask_flat.sum(dim=1).clamp(min=1)
            scores_after_layers.append(masked_s.mean())
    anomaly_after = torch.stack(scores_after_layers).mean().item() if scores_after_layers else 0.0

    return {
        "anomaly_before": anomaly_before,
        "anomaly_after": anomaly_after,
        "anomaly_reduction": anomaly_before - anomaly_after,
    }


def verify_gradients(model: RARModel, step: int):


    decoder_grad_info = []
    for name, p in model.mae.mae.decoder.named_parameters():
        if p.requires_grad and p.grad is not None:
            decoder_grad_info.append((name, p.grad.abs().mean().item()))
            break


    dinov2_has_grad = False
    for name, p in model.dinov2.named_parameters():
        if p.grad is not None:
            dinov2_has_grad = True
            break

    if decoder_grad_info:
        name, grad_mean = decoder_grad_info[0]
        print(f"  [GRAD CHECK step={step}] decoder '{name}': grad_mean={grad_mean:.2e}")
    else:
        print(f"  [GRAD CHECK step={step}] WARNING: decoder has NO gradients!")

    if dinov2_has_grad:
        print(f"  [GRAD CHECK step={step}] WARNING: DINOv2 has gradients (should be None)!")
    else:
        print(f"  [GRAD CHECK step={step}] DINOv2 grad=None (correct)")


def train_one_epoch(
    model: RARModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: str,
    grad_clip: float = GRAD_CLIP,
    log_interval: int = LOG_INTERVAL,
    warmup_steps: int = WARMUP_STEPS,
    global_step: int = 0,
    lambda_rec: float = LAMBDA_REC,
    lambda_align: float = LAMBDA_ALIGN,
) -> tuple:

    model.mae.train()

    total_loss = 0.0
    total_rec = 0.0
    total_alignment_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        image = batch["image"].to(device)
        mask_pixel = batch["mask_pixel"].to(device)
        mae_patch_mask = batch["mae_patch_mask"].to(device)
        dino_token_mask = batch["dino_token_mask"].to(device)


        outputs = model(image, mask_pixel, mae_patch_mask, dino_token_mask)


        if global_step < warmup_steps:
            loss = outputs["loss_rec"]
        else:
            loss = lambda_rec * outputs["loss_rec"] + lambda_align * outputs["loss_align"]


        optimizer.zero_grad()
        loss.backward()


        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), grad_clip)

        optimizer.step()


        if global_step == 0 or global_step == warmup_steps:
            verify_gradients(model, global_step)

        total_loss += loss.item()
        total_rec += outputs["loss_rec"].item()
        total_alignment_loss += outputs["loss_align"].item()
        n_batches += 1
        global_step += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_l = total_loss / n_batches
            avg_r = total_rec / n_batches
            avg_align = total_alignment_loss / n_batches
            phase = "WARMUP" if global_step <= warmup_steps else "TRAIN"
            print(
                f"  [{phase} Epoch {epoch}] batch {batch_idx+1}/{len(dataloader)} | "
                f"loss={avg_l:.5f} rec={avg_r:.5f} align={avg_align:.5f}"
            )

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "loss_rec": total_rec / max(n_batches, 1),
        "loss_align": total_alignment_loss / max(n_batches, 1),
        "n_batches": n_batches,
    }
    return metrics, global_step


@torch.no_grad()
def validate(
    model: RARModel,
    dataloader: DataLoader,
    device: str,
) -> dict:

    model.mae.eval()

    total_loss = 0.0
    total_rec = 0.0
    total_alignment_loss = 0.0
    total_anomaly_before = 0.0
    total_anomaly_after = 0.0
    n_batches = 0

    for batch in dataloader:
        image = batch["image"].to(device)
        mask_pixel = batch["mask_pixel"].to(device)
        mae_patch_mask = batch["mae_patch_mask"].to(device)
        dino_token_mask = batch["dino_token_mask"].to(device)


        img_masked = image * (1.0 - mask_pixel)
        mae_out = model.mae(img_masked, mae_patch_mask, mask_pixel)
        reconstructed = mae_out["reconstructed"]


        pixel_diff = (reconstructed - image).abs()
        masked_diff = pixel_diff * mask_pixel
        n_masked_pixels = mask_pixel.sum().clamp(min=1.0)
        loss_rec = masked_diff.sum() / (n_masked_pixels * 3.0)


        tokens_by_layer = model.dinov2(reconstructed)
        loss_align = compute_alignment_loss(
            tokens_by_layer, model.banks, dino_token_mask
        )

        loss_total = model.lambda_rec * loss_rec + model.lambda_align * loss_align


        anomaly_info = compute_anomaly_comparison(
            model, image, mask_pixel, mae_patch_mask, dino_token_mask
        )

        total_loss += loss_total.item()
        total_rec += loss_rec.item()
        total_alignment_loss += loss_align.item()
        total_anomaly_before += anomaly_info["anomaly_before"]
        total_anomaly_after += anomaly_info["anomaly_after"]
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "loss_rec": total_rec / n,
        "loss_align": total_alignment_loss / n,
        "anomaly_before": total_anomaly_before / n,
        "anomaly_after": total_anomaly_after / n,
        "anomaly_reduction": (total_anomaly_before - total_anomaly_after) / n,
    }


def train_one_suite(
    args,
    suite: str,
    mae_state_dict: dict,
    optimizer_state_dict: dict,
    device: str,
    global_step: int,
    history: list,
) -> tuple:


    from experiments.robot.rar.model import RARModel

    image_dir = Path(args.image_dir) / suite
    if not image_dir.exists():

        image_dir = Path(args.image_dir)

    print(f"\n{'#'*70}")
    print(f"  Suite: {suite}")
    print(f"  Image dir: {image_dir}")
    print(f"{'#'*70}")


    dataset = CleanFrameDataset(
        image_dir=str(image_dir),
        img_size=IMG_SIZE,
        mask_sizes=MAE_GRID_MASK_SIZES,
        max_frames=args.max_frames,
        seed=args.seed,
    )
    print(f"  Dataset: {len(dataset)} frames")

    n_val = max(1, int(len(dataset) * VALIDATION_FRACTION))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_collate = make_collate_fn(mask_sizes=MAE_GRID_MASK_SIZES, mask_weights=MAE_GRID_MASK_WEIGHTS)
    val_collate = make_collate_fn(mask_sizes=MAE_GRID_MASK_SIZES, mask_weights=MAE_GRID_MASK_WEIGHTS)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=DATALOADER_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_collate,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=DATALOADER_WORKERS,
        pin_memory=True,
        collate_fn=val_collate,
    )
    print(f"  Train: {n_train} frames ({len(train_loader)} batches/epoch), Val: {n_val}")


    model = RARModel(
        mae_model_name=args.mae_model,
        featurizer=None,
        suite=suite,
        target_layers=TARGET_LAYERS,
        lambda_rec=args.lambda_rec,
        lambda_align=args.lambda_align,
        device=device,
        model_root=args.model_root,
    )
    model = model.to(device)


    if mae_state_dict is not None:
        missing, unexpected = model.mae.mae.decoder.load_state_dict(mae_state_dict, strict=False)
        print(f"  Restored decoder weights (missing={len(missing)}, unexpected={len(unexpected)})")


    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    if optimizer_state_dict is not None:
        try:
            optimizer.load_state_dict(optimizer_state_dict)
            print(f"  Restored optimizer state")
        except Exception:
            print(f"  Could not restore optimizer state, using fresh")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * SCHEDULER_MIN_LR_RATIO,
    )


    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics, global_step = train_one_epoch(
            model, train_loader, optimizer, epoch, device,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            warmup_steps=args.warmup_steps,
            global_step=global_step,
            lambda_rec=args.lambda_rec,
            lambda_align=args.lambda_align,
        )

        val_metrics = validate(model, val_loader, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(
            f"\n  [{suite}] Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | lr={current_lr:.2e}\n"
            f"    Train: loss={train_metrics['loss']:.5f} "
            f"rec={train_metrics['loss_rec']:.5f} "
            f"align={train_metrics['loss_align']:.5f}\n"
            f"    Val:   loss={val_metrics['loss']:.5f} "
            f"rec={val_metrics['loss_rec']:.5f} "
            f"align={val_metrics['loss_align']:.5f}\n"
            f"    Anomaly: before={val_metrics['anomaly_before']:.4f} "
            f"after={val_metrics['anomaly_after']:.4f} "
            f"reduction={val_metrics['anomaly_reduction']:.4f}"
        )

        if val_metrics['anomaly_reduction'] > 0:
            print(f"    ✓ Anomaly reduced by {val_metrics['anomaly_reduction']:.4f}")
        else:
            print(f"    ✗ Anomaly NOT reduced yet")

        history.append({
            "suite": suite,
            "epoch": epoch,
            "lr": current_lr,
            "train": train_metrics,
            "val": val_metrics,
            "time_s": elapsed,
        })


        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt_path = CKPT_DIR / "best_decoder.pt"
            torch.save({
                "suite": suite,
                "epoch": epoch,
                "model_state_dict": {
                    k: v for k, v in model.mae.state_dict().items()
                    if "decoder" in k
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "anomaly_before": val_metrics["anomaly_before"],
                "anomaly_after": val_metrics["anomaly_after"],
                "anomaly_reduction": val_metrics["anomaly_reduction"],
                "config": {
                    "mae_model": args.mae_model,
                    "suite": suite,
                    "lambda_rec": args.lambda_rec,
                    "lambda_align": args.lambda_align,
                    "lr": args.lr,
                    "target_layers": TARGET_LAYERS,
                },
            }, ckpt_path)
            print(f"    -> Saved best model (val_loss={best_val_loss:.5f})")


        vis_batch = next(iter(val_loader))
        vis_paths = save_batch_comparison(
            model,
            vis_batch,
            epoch=epoch,
            device=device,
            n_samples=VISUALIZATION_SAMPLES,
        )
        if vis_paths:
            print(f"    -> Vis: {Path(vis_paths[0]).parent}")

        plot_training_curves(history)


    decoder_state = {
        k: v.cpu() for k, v in model.mae.mae.decoder.state_dict().items()
    }
    opt_state = optimizer.state_dict()


    model.close()
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Suite {suite} done. Released VLA DINOv2.")

    return decoder_state, opt_state, global_step, history, best_val_loss


def main():
    args = parse_args()

    device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"


    if args.suite == "all":
        suites = ALL_SUITES
    else:
        suites = [args.suite]

    print("=" * 70)
    print("  Representation-Aligned Reconstruction (RAR) Training")
    print("  Prototype bank: shared L18-22")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Image dir: {args.image_dir}")
    print(f"Suites: {suites}")
    print(f"Epochs per suite: {args.epochs}, Batch size: {args.batch_size}")
    print(f"LR: {args.lr}, lambda_rec: {args.lambda_rec}, lambda_align: {args.lambda_align}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Target layers: {TARGET_LAYERS}")
    print(f"Output: {CKPT_DIR}")
    print("=" * 70)


    mae_state_dict = None
    optimizer_state_dict = None
    global_step = 0
    history = []

    for suite in suites:
        mae_state_dict, optimizer_state_dict, global_step, history, best_val = train_one_suite(
            args=args,
            suite=suite,
            mae_state_dict=mae_state_dict,
            optimizer_state_dict=optimizer_state_dict,
            device=device,
            global_step=global_step,
            history=history,
        )


    history_path = LOG_DIR / "train_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"All suites complete.")
    print(f"  Suites trained: {suites}")
    print(f"  Total steps: {global_step}")
    print(f"  Best RAR checkpoint: {CKPT_DIR / 'best_decoder.pt'}")
    print(f"  History: {history_path}")
    print(f"  Training curves: {LOG_DIR / 'training_curves.png'}")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Representation-Aligned Reconstruction decoder")

    p.add_argument("--image_dir", type=str, default=str(CLEAN_FRAMES_DIR),
                   help="Root dir of clean frames. For --suite all, expects subdirs per suite.")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--mae_model", type=str, default=MAE_MODEL_NAME)
    p.add_argument("--suite", type=str, default="libero_10",
                   choices=ALL_SUITES + ["all"],
                   help="Which suite, or 'all' for serial 4-suite training")

    p.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                   help="Epochs PER SUITE (not total)")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--grad_clip", type=float, default=GRAD_CLIP)
    p.add_argument("--lambda_rec", type=float, default=LAMBDA_REC)
    p.add_argument("--lambda_align", type=float, default=LAMBDA_ALIGN,
                   help="Weight of the deep-representation alignment loss.")
    p.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS,
                   help="Use only L_rec for the first N steps of the first suite.")

    p.add_argument("--cuda_id", type=int, default=0)
    p.add_argument(
        "--model_root",
        type=str,
        default=None,
        help="Optional local OpenVLA root. MODEL_ROOT is used when omitted.",
    )
    p.add_argument("--seed", type=int, default=TRAINING_SEED)
    p.add_argument("--log_interval", type=int, default=LOG_INTERVAL)

    return p.parse_args()


if __name__ == "__main__":
    main()
