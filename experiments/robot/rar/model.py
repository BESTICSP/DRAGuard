


from __future__ import annotations

import gc
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.robot.rar.config import (
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LAMBDA_ALIGN,
    LAMBDA_REC,
    NUM_PREFIX_TOKENS,
    TARGET_LAYERS,
)
from experiments.robot.model_paths import resolve_openvla_checkpoint
from experiments.robot.rar.mae_wrapper import MAEReconstructor


from experiments.robot.prototype_bank.anomaly_score import load_shared_bank


from experiments.robot.robot_utils import get_model, get_image_resize_size
from experiments.robot.openvla_utils import get_processor


def load_vla_vision_only(
    suite: str = "libero_10",
    device: str = "cuda:0",
    model_root: Optional[str] = None,
) -> nn.Module:


    import os
    model_path = resolve_openvla_checkpoint(suite, model_root)
    model_basename = os.path.basename(model_path)
    unnorm_key = (
        model_basename.split("finetuned-")[-1].replace("-", "_")
        if "finetuned-" in model_basename
        else suite
    )

    cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=model_path,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        task_suite_name=suite,
        unnorm_key=unnorm_key,
    )

    print(f"[load_vla_vision_only] Loading VLA from {model_path}...")
    vla_model = get_model(cfg, DEVICE=device)


    featurizer = vla_model.vision_backbone.featurizer


    featurizer_ref = featurizer


    del vla_model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[load_vla_vision_only] VLA released, keeping only DINOv2 featurizer "
          f"({sum(p.numel() for p in featurizer_ref.parameters()) / 1e6:.1f}M params)")

    return featurizer_ref


class DINOv2FeatureExtractor(nn.Module):


    def __init__(self, featurizer: nn.Module, target_layers: List[int] = TARGET_LAYERS):
        super().__init__()
        self.target_layers = list(target_layers)
        self.captured: Dict[int, Optional[torch.Tensor]] = {L: None for L in self.target_layers}
        self._handles = []


        self.featurizer = featurizer


        for p in self.featurizer.parameters():
            p.requires_grad = False


        for L in self.target_layers:
            handle = self.featurizer.blocks[L].register_forward_hook(self._make_hook(L))
            self._handles.append(handle)


        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def _make_hook(self, L: int):
        def _hook(module, input, output):


            self.captured[L] = output[:, NUM_PREFIX_TOKENS:, :].to(torch.float32)
        return _hook

    def forward(self, pixel_values: torch.Tensor) -> Dict[int, torch.Tensor]:


        for L in self.target_layers:
            self.captured[L] = None


        x = (pixel_values - self.mean) / self.std


        featurizer_dtype = next(self.featurizer.parameters()).dtype
        x = x.to(dtype=featurizer_dtype)


        _ = self.featurizer(x)

        tokens = {}
        for L in self.target_layers:
            if self.captured[L] is not None:
                tokens[L] = self.captured[L]
        return tokens

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def compute_differentiable_anomaly_score(
    z: torch.Tensor,
    bank: Dict,
) -> torch.Tensor:


    centroids = bank["centroids"]
    if centroids.device != z.device:
        centroids = centroids.to(z.device)

    z_norm = F.normalize(z.to(centroids.dtype), dim=-1, eps=1e-8)

    sim = z_norm @ centroids.T

    max_sim, _ = sim.max(dim=-1)

    return 1.0 - max_sim


def compute_alignment_loss(
    tokens_by_layer: Dict[int, torch.Tensor],
    banks: Dict[int, Dict],
    dino_token_mask: torch.Tensor,
) -> torch.Tensor:


    B = dino_token_mask.shape[0]
    mask_flat = dino_token_mask.float()

    layer_losses = []
    for L in tokens_by_layer:
        if L not in banks:
            continue
        tokens_L = tokens_by_layer[L]


        scores = compute_differentiable_anomaly_score(
            tokens_L, banks[L]
        )


        masked_scores = scores * mask_flat
        n_masked = mask_flat.sum(dim=1).clamp(min=1.0)
        per_sample_loss = masked_scores.sum(dim=1) / n_masked
        layer_losses.append(per_sample_loss.mean())

    if not layer_losses:
        return torch.tensor(0.0, device=dino_token_mask.device, requires_grad=True)


    return torch.stack(layer_losses).mean()


class RARModel(nn.Module):


    def __init__(
        self,
        mae_model_name: str = "facebook/vit-mae-base",
        featurizer: Optional[nn.Module] = None,
        suite: str = "libero_10",
        target_layers: List[int] = TARGET_LAYERS,
        lambda_rec: float = LAMBDA_REC,
        lambda_align: float = LAMBDA_ALIGN,
        device: str = "cuda:0",
        model_root: Optional[str] = None,
    ):


        super().__init__()

        self.lambda_rec = lambda_rec
        self.lambda_align = lambda_align
        self.target_layers = target_layers
        self.device = device


        self.mae = MAEReconstructor(model_name=mae_model_name, freeze_encoder=True)


        if featurizer is None:
            featurizer = load_vla_vision_only(
                suite=suite, device=device, model_root=model_root
            )
        self.dinov2 = DINOv2FeatureExtractor(featurizer, target_layers=target_layers)
        self.dinov2 = self.dinov2.to(device)


        self.banks: Dict[int, Dict] = {}
        for L in target_layers:
            self.banks[L] = load_shared_bank(L, device=device)
        print(f"[RAR] Loaded shared normal prototype banks: {list(self.banks.keys())}")

    def forward(
        self,
        image: torch.Tensor,
        mask_pixel: torch.Tensor,
        mae_patch_mask: torch.Tensor,
        dino_token_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:


        img_masked = image * (1.0 - mask_pixel)


        mae_out = self.mae(img_masked, mae_patch_mask, mask_pixel)
        reconstructed = mae_out["reconstructed"]


        pixel_diff = (reconstructed - image).abs()
        masked_diff = pixel_diff * mask_pixel
        n_masked_pixels = mask_pixel.sum().clamp(min=1.0)
        loss_rec = masked_diff.sum() / (n_masked_pixels * 3.0)


        tokens_by_layer = self.dinov2(reconstructed)


        loss_align = compute_alignment_loss(
            tokens_by_layer, self.banks, dino_token_mask
        )


        loss_total = self.lambda_rec * loss_rec + self.lambda_align * loss_align

        return {
            "loss": loss_total,
            "loss_rec": loss_rec,
            "loss_align": loss_align,
            "reconstructed": reconstructed.detach(),
            "img_masked": img_masked.detach(),
        }

    def get_trainable_params(self) -> List[nn.Parameter]:

        return [p for p in self.mae.parameters() if p.requires_grad]

    def count_params(self) -> Dict[str, int]:

        mae_info = self.mae.count_params()
        dinov2_total = sum(p.numel() for p in self.dinov2.parameters())
        return {
            "mae_trainable": mae_info["trainable"],
            "mae_frozen": mae_info["frozen"],
            "dinov2_frozen": dinov2_total,
            "total_trainable": mae_info["trainable"],
        }

    def close(self):

        self.dinov2.close()
