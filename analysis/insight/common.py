"""Shared path, image, model, and rollout utilities for Insight experiments."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional, Sequence, Tuple

import imageio
import numpy as np
import torch

from analysis.insight.config import (
    ATTACK_PATCH_GLOBS,
    ATTACK_PATCH_POSITION_BY_SUITE,
    DATA_ROOT,
    IMAGE_SIZE,
    NUM_PATCH_TOKENS,
    TOKEN_GRID_SIZE,
)
from experiments.robot.libero.libero_utils import quat2axisangle
from experiments.robot.model_paths import resolve_openvla_checkpoint
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_model


def parse_position(values: Optional[Sequence[int]], suite: str) -> Tuple[int, int]:
    """Use a command-line position or fall back to the suite default."""
    if values is None:
        return ATTACK_PATCH_POSITION_BY_SUITE[suite]
    return int(values[0]), int(values[1])


def resolve_patch(attack: str, patch_path: Optional[str] = None) -> Path:
    """Resolve an explicit patch or the first public patch of an attack type."""
    if patch_path:
        path = Path(patch_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Patch file does not exist: {path}")
        return path

    attack = attack.lower()
    if attack not in ATTACK_PATCH_GLOBS:
        raise ValueError(f"Unsupported attack type: {attack}")
    patch_root = DATA_ROOT / "adversarial_patches"
    candidates = sorted(patch_root.glob(ATTACK_PATCH_GLOBS[attack]))
    if not candidates:
        raise FileNotFoundError(
            f"No {attack.upper()} patch matched: "
            f"{patch_root / ATTACK_PATCH_GLOBS[attack]}"
        )
    return candidates[0]


def load_patch(path: Path) -> torch.Tensor:
    """Load a [3, H, W] patch and convert it to uint8."""
    value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        for key in ("patch", "adv_patch", "tensor"):
            if key in value:
                value = value[key]
                break
    if not torch.is_tensor(value):
        raise TypeError(f"Could not read a patch tensor from {path}.")
    value = value.detach().cpu()
    if value.ndim == 4:
        value = value[0]
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError(
            f"Expected a [3, H, W] patch, but received {tuple(value.shape)}."
        )
    if value.dtype != torch.uint8:
        if float(value.max()) <= 1.0:
            value = value * 255.0
        value = value.clamp(0, 255).to(torch.uint8)
    return value


def apply_patch(
    image: np.ndarray,
    patch: torch.Tensor,
    position: Tuple[int, int],
) -> np.ndarray:
    """Paste a patch into an RGB uint8 image."""
    output = image.copy()
    x0, y0 = position
    patch_h, patch_w = int(patch.shape[1]), int(patch.shape[2])
    x1 = min(x0 + patch_w, output.shape[1])
    y1 = min(y0 + patch_h, output.shape[0])
    if x0 < 0 or y0 < 0 or x0 >= x1 or y0 >= y1:
        raise ValueError(
            f"Patch position is outside the image: position={position}, "
            f"image={output.shape}"
        )
    patch_rgb = patch.permute(1, 2, 0).numpy()
    output[y0:y1, x0:x1] = patch_rgb[:y1 - y0, :x1 - x0]
    return output


def build_token_indices(
    position: Tuple[int, int],
    patch_hw: Tuple[int, int],
    target_count: Optional[int] = None,
) -> list[int]:
    """Map a pixel patch box to the 16x16 token grid."""
    x0, y0 = position
    patch_h, patch_w = patch_hw
    cell = IMAGE_SIZE / TOKEN_GRID_SIZE
    gx0 = max(int(x0 / cell), 0)
    gy0 = max(int(y0 / cell), 0)
    gx1 = min(int(np.ceil((x0 + patch_w) / cell)), TOKEN_GRID_SIZE)
    gy1 = min(int(np.ceil((y0 + patch_h) / cell)), TOKEN_GRID_SIZE)

    mask = np.zeros((TOKEN_GRID_SIZE, TOKEN_GRID_SIZE), dtype=bool)
    mask[gy0:gy1, gx0:gx1] = True

    if target_count is not None and int(mask.sum()) > target_count:
        raise ValueError(
            f"The target region covers {int(mask.sum())} tokens, which exceeds "
            f"the {target_count} source tokens. The original experiment does not "
            "define a shrinking map."
        )


    while target_count is not None and int(mask.sum()) < target_count:
        changed = False
        for direction in ("right", "left", "down", "up"):
            ys, xs = np.where(mask)
            if direction == "right" and int(xs.max()) + 1 < TOKEN_GRID_SIZE:
                mask[int(ys.min()):int(ys.max()) + 1, int(xs.max()) + 1] = True
                changed = True
            elif direction == "left" and int(xs.min()) - 1 >= 0:
                mask[int(ys.min()):int(ys.max()) + 1, int(xs.min()) - 1] = True
                changed = True
            elif direction == "down" and int(ys.max()) + 1 < TOKEN_GRID_SIZE:
                mask[int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1] = True
                changed = True
            elif direction == "up" and int(ys.min()) - 1 >= 0:
                mask[int(ys.min()) - 1, int(xs.min()):int(xs.max()) + 1] = True
                changed = True
            if target_count is None or int(mask.sum()) >= target_count:
                break
        if not changed:
            break

    indices = np.flatnonzero(mask.reshape(-1)).tolist()
    if target_count is not None:
        if len(indices) != target_count:
            raise ValueError(
                f"The target region produced {len(indices)} tokens and cannot "
                f"match the {target_count} source tokens with the original mapping."
            )
    if not indices or max(indices) >= NUM_PATCH_TOKENS:
        raise ValueError("The generated token mask is invalid.")
    return indices


def build_policy_observation(image: np.ndarray, env_observation: dict) -> dict:
    """Build an OpenVLA policy observation."""
    state = np.concatenate((
        env_observation["robot0_eef_pos"],
        quat2axisangle(env_observation["robot0_eef_quat"]),
        env_observation["robot0_gripper_qpos"],
    ))
    return {"full_image": image, "state": state}


def load_policy(suite: str, cuda_id: int, model_root: Optional[str] = None):
    """Load the OpenVLA model, processor, and runtime config for a suite."""
    device = f"cuda:{cuda_id}"
    checkpoint = resolve_openvla_checkpoint(suite, model_root)
    cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=checkpoint,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        task_suite_name=suite,
        unnorm_key=suite,
    )
    model = get_model(cfg, DEVICE=device)
    if cfg.unnorm_key not in model.norm_stats:
        alternative = f"{cfg.unnorm_key}_no_noops"
        if alternative in model.norm_stats:
            cfg.unnorm_key = alternative
    processor = get_processor(cfg)
    return model, processor, cfg, device


def save_video(frames: Iterable[np.ndarray], path: Path, fps: int = 20) -> None:
    """Save an experiment video under the git-ignored rollout directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), list(frames), fps=fps)
