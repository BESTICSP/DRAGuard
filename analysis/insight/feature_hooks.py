"""Visual feature collection and replacement hooks for Insight experiments."""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch

from analysis.insight.config import NUM_PATCH_TOKENS


def _extract_tensor(output) -> torch.Tensor:
    """Extract the token tensor from a transformer block output."""
    tensor = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(tensor) or tensor.ndim != 3:
        raise TypeError("A visual block must return a [B, N, C] tensor.")
    return tensor


class LayerFeatureCollector:
    """Collect patch tokens from every intermediate DINOv2 and SigLIP layer."""

    def __init__(self, model) -> None:
        self.features: Dict[str, Dict[int, torch.Tensor]] = {
            "dino": {},
            "siglip": {},
        }
        self._handles: List = []

        dino_blocks = model.vision_backbone.featurizer.blocks
        self._register_blocks("dino", dino_blocks)

        fused = getattr(model.vision_backbone, "fused_featurizer", None)
        if fused is None or not hasattr(fused, "blocks"):
            raise AttributeError(
                "The OpenVLA checkpoint does not expose a SigLIP fused_featurizer."
            )
        self._register_blocks("siglip", fused.blocks)

    def _register_blocks(self, name: str, blocks) -> None:
        for layer, block in enumerate(blocks):
            self._handles.append(
                block.register_forward_hook(self._create_hook(name, layer))
            )

    def _create_hook(self, name: str, layer: int):
        def hook_fn(_module, _inputs, output):
            tokens = _extract_tensor(output)
            self.features[name][layer] = (
                tokens[:, -NUM_PATCH_TOKENS:, :].detach().to("cpu", torch.float32)
            )
        return hook_fn

    def clear(self) -> None:
        self.features["dino"].clear()
        self.features["siglip"].clear()

    def patch_tokens(self, name: str, indices: Iterable[int]) -> torch.Tensor:
        """Return a [n_layers, n_patch_tokens, C] tensor."""
        selected = list(indices)
        layer_map = self.features[name]
        if not layer_map:
            raise RuntimeError(f"No {name} features were captured.")
        return torch.stack([
            layer_map[layer][0, selected, :]
            for layer in sorted(layer_map)
        ])

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class RepresentationReplacer:
    """Replace target tokens with pre-collected attack representations."""

    def __init__(
        self,
        model,
        target_indices: Iterable[int],
        dino_layer: int,
        siglip_layer: int,
    ) -> None:
        self.target_indices = list(target_indices)
        self.embeddings = {
            "dino": None,
            "siglip": None,
        }
        self._handles: List = []

        dino_blocks = model.vision_backbone.featurizer.blocks
        self._handles.append(
            dino_blocks[dino_layer].register_forward_hook(self._create_hook("dino"))
        )

        fused = getattr(model.vision_backbone, "fused_featurizer", None)
        if fused is None or not hasattr(fused, "blocks"):
            raise AttributeError(
                "The OpenVLA checkpoint does not expose a SigLIP fused_featurizer."
            )
        self._handles.append(
            fused.blocks[siglip_layer].register_forward_hook(
                self._create_hook("siglip")
            )
        )

    def set_embeddings(
        self,
        dino: torch.Tensor,
        siglip: torch.Tensor,
    ) -> None:
        self._validate_embedding("dino", dino)
        self._validate_embedding("siglip", siglip)
        self.embeddings["dino"] = dino
        self.embeddings["siglip"] = siglip

    def _validate_embedding(self, name: str, embedding: torch.Tensor) -> None:
        if embedding.ndim != 2:
            raise ValueError(
                f"The {name} embedding must have shape [N, C], but received "
                f"{embedding.shape}."
            )
        if embedding.shape[0] != len(self.target_indices):
            raise ValueError(
                f"The {name} source has {embedding.shape[0]} tokens, but the "
                f"target contains {len(self.target_indices)} positions."
            )

    def _create_hook(self, name: str):
        def hook_fn(_module, _inputs, output):
            embedding = self.embeddings[name]
            if embedding is None:
                return output

            is_tuple = isinstance(output, tuple)
            tokens = _extract_tensor(output).clone()
            source = embedding.to(tokens.device, tokens.dtype)
            offset = tokens.shape[1] - NUM_PATCH_TOKENS

            for source_index, target_index in enumerate(self.target_indices):
                tokens[:, offset + target_index, :] = source[source_index]

            return (tokens,) + output[1:] if is_tuple else tokens
        return hook_fn

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
