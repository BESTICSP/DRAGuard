


from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTMAEForPreTraining

from experiments.robot.rar.config import (
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAE_PATCH_SIZE,
    MASK_TIE_BREAK_NOISE,
)


MAE_GRID_SIZE = IMG_SIZE // MAE_PATCH_SIZE
MAE_NUM_PATCHES = MAE_GRID_SIZE * MAE_GRID_SIZE


IMAGENET_MEAN_TENSOR = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
IMAGENET_STD_TENSOR = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


class MAEReconstructor(nn.Module):


    def __init__(
        self,
        model_name: str = "facebook/vit-mae-base",
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.mae = ViTMAEForPreTraining.from_pretrained(model_name)
        self.config = self.mae.config

        if freeze_encoder:
            for p in self.mae.vit.parameters():
                p.requires_grad = False
            for p in self.mae.decoder.parameters():
                p.requires_grad = True

        self.patch_size = self.config.patch_size
        self.grid_size = IMG_SIZE // self.patch_size
        self.num_patches = self.grid_size * self.grid_size


        self.register_buffer("img_mean", IMAGENET_MEAN_TENSOR.clone())
        self.register_buffer("img_std", IMAGENET_STD_TENSOR.clone())

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """[0,1] RGB → ImageNet normalized."""
        return (x - self.img_mean) / self.img_std

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:

        return x * self.img_std + self.img_mean

    def _build_masking_tensors(
        self, mae_patch_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:


        B, N = mae_patch_mask.shape
        device = mae_patch_mask.device


        n_masked_per_sample = mae_patch_mask.long().sum(dim=1)
        len_mask = int(n_masked_per_sample.max().item())
        len_mask = max(len_mask, 1)
        len_keep = N - len_mask
        if len_keep < 1:
            len_keep = 1
            len_mask = N - 1


        noise = torch.zeros(B, N, device=device)
        noise[mae_patch_mask] = 1.0
        noise = noise + torch.rand(B, N, device=device) * MASK_TIE_BREAK_NOISE

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        mask = torch.ones(B, N, device=device)
        mask.scatter_(1, ids_keep, 0.0)

        return ids_keep, ids_restore, mask, len_keep, len_mask

    def forward(
        self,
        pixel_values: torch.Tensor,
        mae_patch_mask: torch.Tensor,
        mask_pixel: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:


        B = pixel_values.shape[0]
        device = pixel_values.device


        x_norm = self._normalize(pixel_values)


        embeddings = self.mae.vit.embeddings
        patch_embeddings = embeddings.patch_embeddings(x_norm)
        patch_embeddings = patch_embeddings + embeddings.position_embeddings[:, 1:, :]


        ids_keep, ids_restore, mask, len_keep, len_mask = self._build_masking_tensors(
            mae_patch_mask.to(device)
        )


        ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, patch_embeddings.shape[-1])
        visible_patches = torch.gather(patch_embeddings, dim=1, index=ids_keep_expanded)


        cls_token = embeddings.cls_token + embeddings.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        visible_with_cls = torch.cat([cls_tokens, visible_patches], dim=1)


        with torch.no_grad():
            encoder_output = visible_with_cls
            for block in self.mae.vit.encoder.layer:
                encoder_output = block(encoder_output)[0]
            encoder_output = self.mae.vit.layernorm(encoder_output)
        encoder_output = encoder_output.detach()


        decoder = self.mae.decoder

        decoder_input = decoder.decoder_embed(encoder_output)
        decoder_cls = decoder_input[:, :1, :]
        decoder_visible = decoder_input[:, 1:, :]

        mask_tokens = decoder.mask_token.expand(B, len_mask, -1)
        full_tokens = torch.cat([decoder_visible, mask_tokens], dim=1)


        ids_restore_expanded = ids_restore.unsqueeze(-1).expand(-1, -1, full_tokens.shape[-1])
        full_tokens = torch.gather(full_tokens, dim=1, index=ids_restore_expanded)


        full_with_cls = torch.cat([decoder_cls, full_tokens], dim=1)
        full_with_cls = full_with_cls + decoder.decoder_pos_embed


        decoder_output = full_with_cls
        for block in decoder.decoder_layers:
            decoder_output = block(decoder_output)[0]
        decoder_output = decoder.decoder_norm(decoder_output)


        logits = decoder.decoder_pred(decoder_output)
        logits = logits[:, 1:, :]


        rec_norm = self._patches_to_image(logits)


        rec_rgb = self._denormalize(rec_norm)
        rec_rgb = rec_rgb.clamp(0.0, 1.0)


        reconstructed = pixel_values * (1.0 - mask_pixel) + rec_rgb * mask_pixel

        return {
            "reconstructed": reconstructed,
            "rec_full": rec_rgb,
            "mask": mask,
        }

    def _patches_to_image(self, patches: torch.Tensor) -> torch.Tensor:
        """[B, 196, patch_size**2 * 3] → [B, 3, 224, 224]"""
        B = patches.shape[0]
        p = self.patch_size
        h = w = self.grid_size

        x = patches.reshape(B, h, w, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, 3, h * p, w * p)
        return x

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def count_params(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}
