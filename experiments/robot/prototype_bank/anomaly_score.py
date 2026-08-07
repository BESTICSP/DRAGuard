


from typing import Dict, Iterable, Optional, Union

import torch
import torch.nn.functional as F

from experiments.robot.prototype_bank.config import (
    LAYER_COMBOS,
    PROTOTYPE_BANK_DIR,
    TARGET_LAYERS,
)


def load_shared_bank(
    layer: int,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict:

    path = PROTOTYPE_BANK_DIR / f"dinov2_L{layer}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run build_prototypes_shared.py first."
        )

    bank = torch.load(path, map_location="cpu", weights_only=False)
    if device is not None:
        bank["centroids"] = bank["centroids"].to(device)
        if "occupancy" in bank:
            bank["occupancy"] = bank["occupancy"].to(device)
    return bank


def load_shared_banks(
    layers: Optional[Iterable[int]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[int, Dict]:

    selected_layers = list(layers) if layers else TARGET_LAYERS
    return {
        layer: load_shared_bank(layer, device=device)
        for layer in selected_layers
    }


@torch.no_grad()
def score_single_layer(
    z: torch.Tensor,
    bank: Dict,
    return_argmax: bool = False,
):

    centroids = bank["centroids"]
    if centroids.device != z.device:
        centroids = centroids.to(z.device)

    normalized_tokens = F.normalize(
        z.to(centroids.dtype), dim=-1, eps=1e-8
    )
    similarities = normalized_tokens @ centroids.T
    max_similarity, nearest_index = similarities.max(dim=-1)
    anomaly_scores = 1.0 - max_similarity

    if return_argmax:
        return anomaly_scores, nearest_index
    return anomaly_scores


@torch.no_grad()
def score_fused(
    tokens_by_layer: Dict[int, torch.Tensor],
    banks: Dict[int, Dict],
    combo: Union[str, Iterable[int]] = "L18-22",
):

    if isinstance(combo, str):
        if combo not in LAYER_COMBOS:
            raise KeyError(
                f"combo {combo!r} not in LAYER_COMBOS; "
                f"valid={list(LAYER_COMBOS)}"
            )
        selected_layers = LAYER_COMBOS[combo]
    else:
        selected_layers = list(combo)

    layer_scores = []
    for layer in selected_layers:
        if layer not in banks:
            raise KeyError(f"bank for layer {layer} not loaded")
        if layer not in tokens_by_layer:
            raise KeyError(f"candidate tokens for layer {layer} not supplied")
        layer_scores.append(
            score_single_layer(tokens_by_layer[layer], banks[layer])
        )
    return torch.stack(layer_scores, dim=0).mean(dim=0)


@torch.no_grad()
def score_all_combos(
    tokens_by_layer: Dict[int, torch.Tensor],
    banks: Dict[int, Dict],
) -> Dict[str, torch.Tensor]:

    results = {}
    for name, layers in LAYER_COMBOS.items():
        if not all(
            layer in tokens_by_layer and layer in banks
            for layer in layers
        ):
            continue
        results[name] = score_fused(tokens_by_layer, banks, combo=name)
    return results
