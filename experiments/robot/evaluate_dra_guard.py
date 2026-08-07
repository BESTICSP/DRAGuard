#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

PROJECT_ROOT = Path(__file__).resolve().parents[2]


LOGGER = logging.getLogger(__name__)


def _time_now(device: str) -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.config_loader import get_config_section
from experiments.robot.model_paths import resolve_openvla_checkpoint
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    normalize_gripper_action,
    invert_gripper_action,
)
from experiments.robot.rar.config import MAE_PATCH_SIZE
from experiments.robot.rar.mae_wrapper import MAEReconstructor
from experiments.robot.prototype_bank.config import (
    NUM_PREFIX_TOKENS,
    PROTOTYPE_BANK_DIR,
)
from experiments.robot.settings import (
    ATTACK_PATCH_POSITIONS_BY_SUITE,
    DINO_TOKEN_GRID_SIZE,
    DINO_TARGET_LAYERS as DRL_LAYERS,
    ENVIRONMENT_SEED,
    ENVIRONMENT_RESOLUTION,
    IMAGE_SIZE,
    NUM_STEPS_WAIT,
    SUITE_MAX_STEPS,
)
from experiments.robot.prototype_bank.anomaly_score import (
    load_shared_bank,
    score_single_layer,
)

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def _resize_map(arr_2d: np.ndarray, out_hw: int) -> np.ndarray:
    t = torch.from_numpy(arr_2d).to(torch.float32).unsqueeze(0).unsqueeze(0)
    t2 = F.interpolate(t, size=(out_hw, out_hw), mode="bilinear", align_corners=False)
    return t2.squeeze(0).squeeze(0).cpu().numpy()


def _connected_components_8(binary: np.ndarray, score_map: np.ndarray) -> List[Dict[str, Any]]:
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    regions: List[Dict[str, Any]] = []
    cur = 0

    for y in range(h):
        for x in range(w):
            if binary[y, x] == 0 or labels[y, x] != 0:
                continue
            cur += 1
            q = [(y, x)]
            labels[y, x] = cur
            minx = maxx = x
            miny = maxy = y
            area = 0
            score_sum = 0.0

            while q:
                cy, cx = q.pop()
                area += 1
                score_sum += float(score_map[cy, cx])
                if cx < minx:
                    minx = cx
                if cx > maxx:
                    maxx = cx
                if cy < miny:
                    miny = cy
                if cy > maxy:
                    maxy = cy

                for dy in (-1, 0, 1):
                    ny = cy + dy
                    if ny < 0 or ny >= h:
                        continue
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        nx = cx + dx
                        if nx < 0 or nx >= w:
                            continue
                        if binary[ny, nx] == 0 or labels[ny, nx] != 0:
                            continue
                        labels[ny, nx] = cur
                        q.append((ny, nx))

            regions.append({
                "bbox_cells": (minx, miny, maxx + 1, maxy + 1),
                "area_cells": int(area),
                "score_mean": float(score_sum / max(area, 1)),
            })

    return regions


DEFENSE_CONFIG = get_config_section("defense")
PDV_CONFIG = DEFENSE_CONFIG["pdv"]
DRL_CONFIG = DEFENSE_CONFIG["drl"]
EVALUATION_CONFIG = DEFENSE_CONFIG["evaluation"]

IMG_SIZE = IMAGE_SIZE
GRID_SIZE = DINO_TOKEN_GRID_SIZE
PDV_WINDOW_SIZES = tuple(int(v) for v in PDV_CONFIG["window_sizes"])
PDV_STRIDE = int(PDV_CONFIG["stride"])
PDV_THRESHOLD_STD_MULTIPLIER = float(PDV_CONFIG["threshold_std_multiplier"])
PDV_RING_WIDTH = int(PDV_CONFIG["ring_width"])
PDV_RING_WEIGHT = float(PDV_CONFIG["ring_weight"])
DRL_ANOMALY_THRESHOLD = float(DRL_CONFIG["anomaly_threshold"])
DRL_REFINEMENT_GRID_SIZE = int(DRL_CONFIG["refinement_grid_size"])
DRL_MINIMUM_COMPONENT_CELLS = int(DRL_CONFIG["minimum_component_cells"])
EVALUATION_NUM_TASKS = int(EVALUATION_CONFIG["num_tasks"])
EVALUATION_NUM_EPISODES = int(EVALUATION_CONFIG["episodes_per_condition"])
DIAGNOSTIC_FRAMES = int(EVALUATION_CONFIG["diagnostic_frames"])

AVAILABLE_CONDITIONS = [
    "clean",
    "attack_no_defense",
    "attack_with_dra_guard",
]
DEFAULT_CONDITIONS = ["attack_with_dra_guard"]


class PhysicalDetailVerifier:


    def __init__(
        self,
        win_sizes: Tuple[int, ...] = PDV_WINDOW_SIZES,
        stride: int = PDV_STRIDE,
        alpha: float = PDV_THRESHOLD_STD_MULTIPLIER,
        ring_width: int = PDV_RING_WIDTH,
        lambda_param: float = PDV_RING_WEIGHT,
    ) -> None:
        self.win_sizes = win_sizes
        self.stride = stride
        self.alpha = alpha
        self.ring_width = ring_width
        self.lambda_param = lambda_param
        self.H = 0
        self.W = 0
        self._integral_map = None

    def _precompute(self, rgb: np.ndarray) -> None:
        self.H, self.W = rgb.shape[:2]
        if cv2 is not None:
            tenen = np.zeros((self.H, self.W), dtype=np.float32)
            for c in range(3):
                gx_c = cv2.Sobel(rgb[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
                gy_c = cv2.Sobel(rgb[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
                tenen += gx_c * gx_c + gy_c * gy_c
            self._integral_map = cv2.integral(tenen)
            return

        x = torch.from_numpy(rgb).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
        w_x = kx.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        w_y = ky.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        gx = F.conv2d(x, w_x, padding=1, groups=3)
        gy = F.conv2d(x, w_y, padding=1, groups=3)
        tenen = (gx * gx + gy * gy).sum(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        integ = np.cumsum(np.cumsum(tenen, axis=0), axis=1)
        self._integral_map = np.pad(
            integ, ((1, 0), (1, 0)), mode="constant", constant_values=0.0
        )

    def _rect_mean(self, y0: int, x0: int, y1: int, x1: int) -> float:
        area = max((y1 - y0) * (x1 - x0), 1)
        s = (
            self._integral_map[y1, x1]
            - self._integral_map[y0, x1]
            - self._integral_map[y1, x0]
            + self._integral_map[y0, x0]
        )
        return float(s / area)

    def _score(self, y0: int, x0: int, y1: int, x1: int) -> float:
        rw = self.ring_width
        ry0 = max(y0 - rw, 0)
        rx0 = max(x0 - rw, 0)
        ry1 = min(y1 + rw, self.H)
        rx1 = min(x1 + rw, self.W)
        f_in = self._rect_mean(y0, x0, y1, x1)
        if (ry1 - ry0) <= (y1 - y0) or (rx1 - rx0) <= (x1 - x0):
            f_ring = f_in
        else:
            f_ring = self._rect_mean(ry0, rx0, ry1, rx1)
        return f_in - self.lambda_param * f_ring

    def detect(self, img_uint8: np.ndarray) -> Tuple[bool, List[Tuple[int, int, int, int]], Dict[str, Any]]:
        H, W = img_uint8.shape[:2]
        self._precompute(img_uint8)

        all_scores: List[float] = []
        all_windows: List[Tuple[int, int, int, int, float]] = []
        for ws in self.win_sizes:
            for y0 in range(0, H - ws + 1, self.stride):
                for x0 in range(0, W - ws + 1, self.stride):
                    y1, x1 = y0 + ws, x0 + ws
                    s = self._score(y0, x0, y1, x1)
                    all_scores.append(s)
                    all_windows.append((x0, y0, x1, y1, s))

        if not all_windows:
            return False, [], {"n_windows": 0, "bboxes": [], "scores": []}

        arr = np.asarray(all_scores, dtype=np.float32)
        mu = float(arr.mean())
        sigma = float(arr.std()) + 1e-10
        thresh = mu + self.alpha * sigma

        candidates = sorted(
            [w for w in all_windows if w[4] > thresh],
            key=lambda x: x[4],
            reverse=True,
        )

        bboxes = [(b[0], b[1], b[2], b[3]) for b in candidates]
        scores = [float(b[4]) for b in candidates]
        info = {
            "n_windows": len(all_windows),
            "bboxes": bboxes,
            "top_score": scores[0] if scores else 0.0,
            "scores": scores,
            "thresh": thresh,
        }
        return len(bboxes) > 0, bboxes, info


class DinoV2FeatureHooks:
    def __init__(self, vla: Any, target_layers: List[int]) -> None:
        self.target_layers = list(target_layers)
        self.captured = {layer: None for layer in self.target_layers}
        self._handles = []
        featurizer = vla.vision_backbone.featurizer
        for layer in self.target_layers:
            self._handles.append(
                featurizer.blocks[layer].register_forward_hook(
                    self._create_hook(layer)
                )
            )

    def _create_hook(self, layer: int):
        def _hook(_m, _i, out):
            self.captured[layer] = (
                out[:, NUM_PREFIX_TOKENS:, :].detach().to(torch.float32)
            )
        return _hook

    def clear(self) -> None:
        for layer in self.target_layers:
            self.captured[layer] = None

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


class DeepRepresentationLocalizer:


    def __init__(
        self,
        banks: Dict[int, Dict],
        device: str,
        threshold: float = DRL_ANOMALY_THRESHOLD,
        refine_grid: int = DRL_REFINEMENT_GRID_SIZE,
        min_cells: int = DRL_MINIMUM_COMPONENT_CELLS,
    ) -> None:
        self.banks = banks
        self.device = device
        self.threshold = threshold
        self.refine_grid = refine_grid
        self.min_cells = min_cells
        self.mean_map: Optional[np.ndarray] = None
        self.frame_count = 0

    def detect(self, tokens: Dict[int, torch.Tensor]) -> Dict[str, Any]:

        all_layer_scores = []
        for layer in DRL_LAYERS:
            if layer not in tokens or layer not in self.banks:
                continue
            layer_scores = score_single_layer(
                tokens[layer].to(self.device), self.banks[layer]
            )
            all_layer_scores.append(layer_scores)

        if not all_layer_scores:
            anomaly_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        else:
            fused_scores = torch.stack(all_layer_scores, dim=0).mean(dim=0)
            anomaly_map = fused_scores.detach().cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)


        if cv2 is not None:
            fine_map = cv2.resize(anomaly_map, (self.refine_grid, self.refine_grid), interpolation=cv2.INTER_LINEAR)
        else:
            fine_map = _resize_map(anomaly_map, self.refine_grid)
        self.frame_count += 1
        if self.mean_map is None:
            self.mean_map = fine_map.copy()
        else:
            self.mean_map = ((self.frame_count - 1) * self.mean_map + fine_map) / self.frame_count

        mean_mask = self.mean_map > self.threshold

        scale = IMG_SIZE / self.refine_grid
        regions = []
        if cv2 is not None:
            binary = mean_mask.astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for lab in range(1, num_labels):
                tx, ty, tw, th, area = stats[lab]
                if area < self.min_cells:
                    continue
                x0 = int(tx * scale)
                y0 = int(ty * scale)
                x1 = int(min((tx + tw) * scale, IMG_SIZE))
                y1 = int(min((ty + th) * scale, IMG_SIZE))
                component_scores = self.mean_map[labels == lab]
                regions.append({
                    "bbox": (x0, y0, x1, y1),
                    "area_cells": int(area),
                    "score_mean": float(component_scores.mean()),
                })
        else:
            binary = mean_mask.astype(np.uint8)
            cc = _connected_components_8(binary, self.mean_map)
            for r in cc:
                tx0, ty0, tx1, ty1 = r["bbox_cells"]
                area = int(r["area_cells"])
                if area < self.min_cells:
                    continue
                x0 = int(tx0 * scale)
                y0 = int(ty0 * scale)
                x1 = int(min(tx1 * scale, IMG_SIZE))
                y1 = int(min(ty1 * scale, IMG_SIZE))
                regions.append({
                    "bbox": (x0, y0, x1, y1),
                    "area_cells": area,
                    "score_mean": float(r["score_mean"]),
                })
        regions.sort(key=lambda r: r["score_mean"], reverse=True)

        return {
            "regions": regions,
            "mean_max": float(self.mean_map.max()),
            "confirmed": len(regions) > 0,
        }

    def reset(self) -> None:
        self.mean_map = None
        self.frame_count = 0


def pdv_boxes_to_mask(
    bboxes: List[Tuple[int, int, int, int]], img_size: int = IMG_SIZE
) -> np.ndarray:

    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    for x0, y0, x1, y1 in bboxes:
        x0i, y0i = max(int(x0), 0), max(int(y0), 0)
        x1i, y1i = min(int(x1), img_size), min(int(y1), img_size)
        mask[y0i:y1i, x0i:x1i] = 255
    return mask


def drl_regions_to_mask(
    regions: List[Dict[str, Any]], img_size: int = IMG_SIZE
) -> np.ndarray:
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        x0i, y0i = max(int(x0), 0), max(int(y0), 0)
        x1i, y1i = min(int(x1), img_size), min(int(y1), img_size)
        mask[y0i:y1i, x0i:x1i] = 255
    return mask


def intersect_drl_pdv(
    pdv_bboxes: List[Tuple[int, int, int, int]],
    drl_regions: List[Dict[str, Any]],
) -> Tuple[bool, List[Dict[str, Any]]]:


    if not pdv_bboxes or not drl_regions:
        return False, []

    pdv_mask = pdv_boxes_to_mask(pdv_bboxes, img_size=IMG_SIZE)
    drl_mask = drl_regions_to_mask(drl_regions, img_size=IMG_SIZE)
    if cv2 is not None:
        intersection = cv2.bitwise_and(pdv_mask, drl_mask)
    else:
        intersection = ((pdv_mask > 0) & (drl_mask > 0)).astype(np.uint8) * 255

    if not np.any(intersection > 0):
        return False, []

    if cv2 is not None:
        contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fused_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 0 and h > 0:
                fused_regions.append({
                    "bbox": (int(x), int(y), int(x + w), int(y + h)),
                    "area_pixels": int(cv2.countNonZero(intersection[y:y + h, x:x + w])),
                })
    else:
        components = _connected_components_8(intersection > 0, intersection.astype(np.float32) / 255.0)
        fused_regions = [
            {"bbox": component["bbox_cells"], "area_pixels": component["area_cells"]}
            for component in components
        ]

    return bool(fused_regions), fused_regions


def compute_enclosing_box(
    regions: List[Dict[str, Any]],
) -> Tuple[int, int, int, int]:

    x0 = min(r["bbox"][0] for r in regions)
    y0 = min(r["bbox"][1] for r in regions)
    x1 = max(r["bbox"][2] for r in regions)
    y1 = max(r["bbox"][3] for r in regions)

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(IMG_SIZE, x1)
    y1 = min(IMG_SIZE, y1)
    return x0, y0, x1, y1


def load_rar_decoder(rar_checkpoint: str, device: str) -> MAEReconstructor:
    mae = MAEReconstructor(model_name="facebook/vit-mae-base", freeze_encoder=True)

    ckpt_path = Path(rar_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"RAR checkpoint not found: {rar_checkpoint}")

    ckpt = torch.load(rar_checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise KeyError("RAR checkpoint must contain key 'model_state_dict'.")
    state_dict = ckpt["model_state_dict"]

    expected_decoder_keys = set(mae.mae.decoder.state_dict().keys())
    decoder_state = {}
    for k, v in state_dict.items():
        if k.startswith("mae.decoder."):
            clean_key = k[len("mae.decoder."):]
        elif k.startswith("decoder."):
            clean_key = k[len("decoder."):]
        elif k in expected_decoder_keys:
            clean_key = k
        else:
            continue
        decoder_state[clean_key] = v

    if not decoder_state:
        raise RuntimeError("No decoder weights found in checkpoint.")

    mae.mae.decoder.load_state_dict(decoder_state, strict=True)
    mae.eval()
    mae = mae.to(device)
    return mae


def build_patch_masks_from_box(
    detected_box: Tuple[int, int, int, int],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:


    x0, y0, x1, y1 = detected_box
    patch_grid = MAE_PATCH_SIZE
    mae_grid_size = IMG_SIZE // patch_grid


    tx1 = x0 // patch_grid
    ty1 = y0 // patch_grid
    tx2 = math.ceil(x1 / patch_grid)
    ty2 = math.ceil(y1 / patch_grid)

    tx1 = max(0, min(tx1, mae_grid_size))
    ty1 = max(0, min(ty1, mae_grid_size))
    tx2 = max(0, min(tx2, mae_grid_size))
    ty2 = max(0, min(ty2, mae_grid_size))

    grid_mask = torch.zeros(mae_grid_size, mae_grid_size, dtype=torch.bool)
    grid_mask[ty1:ty2, tx1:tx2] = True
    mae_patch_mask = grid_mask.flatten().unsqueeze(0).to(device)


    mask_pixel = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE, device=device)
    mask_pixel[0, 0, y0:y1, x0:x1] = 1.0

    return mae_patch_mask, mask_pixel


@torch.no_grad()
def reconstruct_with_rar(
    image_u8: np.ndarray,
    rar_decoder: MAEReconstructor,
    mae_patch_mask: torch.Tensor,
    mask_pixel: torch.Tensor,
    device: str,
) -> np.ndarray:

    pixel_values = torch.from_numpy(image_u8).float().div(255.0)
    pixel_values = pixel_values.permute(2, 0, 1).unsqueeze(0).to(device)

    out = rar_decoder(pixel_values, mae_patch_mask, mask_pixel)
    reconstructed = out["reconstructed"]

    result = reconstructed[0].permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()
    return result


def load_adversarial_patch(patch_path: str) -> torch.Tensor:

    pt = torch.load(patch_path, map_location="cpu")
    if pt.dtype == torch.float32 or pt.dtype == torch.float16:
        patch_np = np.asarray(torchvision.transforms.ToPILImage()(pt))
        patch_u8 = torch.from_numpy(patch_np.copy()).permute(2, 0, 1)
    else:
        patch_u8 = pt
    return patch_u8


def apply_adversarial_patch(
    image: np.ndarray,
    patch_u8: torch.Tensor,
    position: tuple,
) -> np.ndarray:

    img_t = torch.from_numpy(image).permute(2, 0, 1)
    _, H, W = img_t.shape
    _, ph, pw = patch_u8.shape
    x, y = position

    canvas = torch.ones(3, H, W, dtype=torch.short) * -100
    x2, y2 = min(x + pw, W), min(y + ph, H)
    canvas[:, y:y2, x:x2] = patch_u8[:, :y2 - y, :x2 - x].short()
    result = torch.where(canvas < 0, img_t.short(), canvas)
    return result.permute(1, 2, 0).numpy().astype(np.uint8)


def build_policy_observation(
    image: np.ndarray,
    env_observation: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:

    robot_state = np.concatenate((
        env_observation["robot0_eef_pos"],
        quat2axisangle(env_observation["robot0_eef_quat"]),
        env_observation["robot0_gripper_qpos"],
    ))
    return {"full_image": image, "state": robot_state}


def run_episode(
    env,
    initial_state,
    model,
    processor,
    cfg,
    resize_size: int,
    task_description: str,
    patch_u8: torch.Tensor,
    attack_patch_position: tuple,
    condition: str,
    rar_decoder: Optional[MAEReconstructor],
    max_steps: int,
    device: str,
    dino_hooks: DinoV2FeatureHooks,
    pdv: PhysicalDetailVerifier,
    drl: DeepRepresentationLocalizer,
    episode_idx: int = 0,
    output_dir: Optional[Path] = None,
) -> dict:


    env.reset()
    obs = env.set_init_state(initial_state)

    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))

    _, ph, pw = patch_u8.shape
    defense_events = []
    video_frames = []

    drl.reset()


    recovery_active = False
    recovery_box = None
    recovered_region = None

    for step in range(max_steps):
        img = get_libero_image(obs, resize_size)


        if condition == "clean":
            img_input = img
        else:
            img_input = apply_adversarial_patch(
                img, patch_u8, attack_patch_position
            )


        if condition == "attack_with_dra_guard" and recovery_active:

            x0, y0, x1, y1 = recovery_box
            img_defended = img_input.copy()
            img_defended[y0:y1, x0:x1] = recovered_region

            video_frames.append(img_defended.copy())


            dino_hooks.clear()
            observation = build_policy_observation(img_defended, obs)
            action_raw = get_action(cfg, model, observation, task_description, processor=processor, DEVICE=device)

        elif condition == "attack_with_dra_guard" and not recovery_active:

            dino_hooks.clear()
            observation = build_policy_observation(img_input, obs)
            action_raw = get_action(cfg, model, observation, task_description, processor=processor, DEVICE=device)


            _, pdv_bboxes, _ = pdv.detect(img_input)


            tokens = {
                layer: dino_hooks.captured[layer][0].to(device)
                for layer in DRL_LAYERS
                if dino_hooks.captured[layer] is not None
            }
            drl_info = drl.detect(tokens)
            drl_regions = drl_info["regions"]


            detected, detected_regions = intersect_drl_pdv(pdv_bboxes, drl_regions)


            if step < DIAGNOSTIC_FRAMES:
                drl_max = drl_info.get("mean_max", 0.0)
                LOGGER.info(
                    f"      [det] step={step} PDV_boxes={len(pdv_bboxes)} "
                    f"DRL_regions={len(drl_regions)} DRL_mean_max={drl_max:.3f} "
                    f"fused={'HIT' if detected else 'miss'}"
                )

            if detected and rar_decoder is not None and detected_regions:

                det_box = compute_enclosing_box(detected_regions)
                x0, y0, x1, y1 = det_box


                mae_patch_mask, mask_pixel = build_patch_masks_from_box(det_box, device)
                img_reconstructed = reconstruct_with_rar(
                    img_input, rar_decoder, mae_patch_mask, mask_pixel, device
                )


                recovered_region = img_reconstructed[y0:y1, x0:x1].copy()
                recovery_box = det_box
                recovery_active = True
                LOGGER.info(f"      Recovery activated at step={step}, box={det_box}")


                img_defended = img_input.copy()
                img_defended[y0:y1, x0:x1] = recovered_region

                video_frames.append(img_defended.copy())


                dino_hooks.clear()
                observation_defended = build_policy_observation(img_defended, obs)
                action_raw = get_action(
                    cfg, model, observation_defended, task_description,
                    processor=processor, DEVICE=device,
                )

                defense_events.append({
                    "step": step,
                    "detected_box": det_box,
                    "n_regions": len(detected_regions),
                    "event": "recovery_activated",
                })
            else:
                video_frames.append(img_input.copy())

        else:

            video_frames.append(img_input.copy())
            dino_hooks.clear()
            observation = build_policy_observation(img_input, obs)
            action_raw = get_action(cfg, model, observation, task_description, processor=processor, DEVICE=device)


        action = normalize_gripper_action(action_raw, binarize=True)
        if cfg.model_family == "openvla":
            action = invert_gripper_action(action)

        obs, _, done, _ = env.step(
            action.tolist() if isinstance(action, np.ndarray) else action
        )

        if done:
            if output_dir and video_frames:
                _save_video(video_frames, episode_idx, True, condition, task_description, output_dir)
            return {"success": True, "steps": step + 1, "defense_events": defense_events,
                    "recovery_active": recovery_active, "recovery_box": recovery_box}

    if output_dir and video_frames:
        _save_video(video_frames, episode_idx, False, condition, task_description, output_dir)
    return {"success": False, "steps": max_steps, "defense_events": defense_events,
            "recovery_active": recovery_active, "recovery_box": recovery_box}


def _save_video(
    video_frames: list,
    episode_idx: int,
    success: bool,
    condition: str,
    task_description: str,
    output_dir: Path,
) -> None:

    try:
        save_rollout_video(
            rollout_images=video_frames,
            idx=episode_idx,
            success=success,
            task_description=task_description,
            exp_name=condition,
            rollout_root=output_dir,
        )
    except Exception as e:
        LOGGER.warning(f"Video save failed: {e}")


def _resolve_suite_path(pattern_or_path: str, suite_name: str) -> str:
    if "{suite}" in pattern_or_path:
        return pattern_or_path.format(suite=suite_name)
    return pattern_or_path


def _resolve_rar_checkpoint(rar_checkpoint_arg: str, suite_name: str) -> str:
    resolved = Path(_resolve_suite_path(rar_checkpoint_arg, suite_name))
    if resolved.is_dir():
        resolved = resolved / "best_decoder.pt"
    if resolved.name != "best_decoder.pt":
        raise ValueError(
            "DRA-Guard expects the released RAR checkpoint to be named "
            f"best_decoder.pt, got: {resolved}"
        )
    return str(resolved)


def _collect_patch_paths(args: argparse.Namespace) -> List[Path]:
    if args.patch_path:
        p = Path(args.patch_path)
        if not p.exists():
            raise FileNotFoundError(f"--patch_path not found: {p}")
        if not p.is_file():
            raise FileNotFoundError(f"--patch_path is not a file: {p}")
        return [p]
    patch_dir = Path(args.patch_dir)
    candidates = sorted(patch_dir.glob(args.patch_glob))
    valid = [p for p in candidates if p.exists() and p.is_file()]
    skipped = len(candidates) - len(valid)
    if skipped:
        LOGGER.warning(
            f"Skipped {skipped} invalid patch paths under {patch_dir}"
        )
    if not valid:
        raise FileNotFoundError(f"No patch found under {patch_dir} with glob={args.patch_glob}")
    return valid


def _patch_label_from_path(patch_path: Path, patch_dir: Path) -> str:
    try:
        rel = patch_path.resolve().relative_to(patch_dir.resolve())
        parts = list(rel.parts)
        if parts and parts[-1].lower().endswith(".pt"):
            parts = parts[:-1]
        if parts:
            return "__".join(parts)
    except Exception:
        pass
    return patch_path.parent.name


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = f"cuda:{args.cuda_id}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = Path(args.rollout_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    suites = [s.strip() for s in args.task_suite_name.split(",") if s.strip()]
    conditions = args.conditions
    patch_paths = _collect_patch_paths(args)
    patch_dir = Path(args.patch_dir)

    LOGGER.info("=" * 70)
    LOGGER.info("  DRA-Guard: DRL-PDV Detection with RAR")
    LOGGER.info("=" * 70)
    LOGGER.info(f"  Device: {device}")
    LOGGER.info(f"  Suites: {suites}")
    LOGGER.info(f"  Patches: {len(patch_paths)} (first={patch_paths[0]})")
    LOGGER.info(f"  RAR checkpoint: {args.rar_checkpoint}")
    LOGGER.info(f"  Episodes/condition/task: {args.num_episodes}")
    LOGGER.info(f"  Conditions: {conditions}")
    LOGGER.info("=" * 70)


    LOGGER.info("\n[1] Loading shared prototype banks...")
    banks = {}
    for layer in DRL_LAYERS:
        bank = load_shared_bank(layer, device=device)
        if bank is not None:
            banks[layer] = bank
    LOGGER.info(f"  Banks loaded: {sorted(banks.keys())}")
    missing_banks = sorted(set(DRL_LAYERS) - set(banks))
    if missing_banks:
        raise FileNotFoundError(
            f"Missing shared prototype banks for layers {missing_banks} "
            f"under {PROTOTYPE_BANK_DIR}."
        )


    pdv = PhysicalDetailVerifier(
        win_sizes=PDV_WINDOW_SIZES,
        stride=PDV_STRIDE,
        alpha=PDV_THRESHOLD_STD_MULTIPLIER,
        ring_width=PDV_RING_WIDTH,
        lambda_param=PDV_RING_WEIGHT,
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    all_results = {}
    patch_position_rng = random.Random(ENVIRONMENT_SEED)

    LOGGER.info("\n[2] Running evaluation...")
    total_t0 = _time_now(device)
    suite_times = {}

    for suite_idx, suite_name in enumerate(suites):
        suite_t0 = _time_now(device)
        LOGGER.info(f"\n{'=' * 70}")
        LOGGER.info(f"  Suite [{suite_idx+1}/{len(suites)}]: {suite_name}")
        LOGGER.info(f"{'=' * 70}")

        if suite_name not in benchmark_dict:
            LOGGER.warning(f"Suite {suite_name} is not available; skipping")
            continue

        attack_patch_positions = ATTACK_PATCH_POSITIONS_BY_SUITE[suite_name]
        LOGGER.info(f"  Attack patch positions: {attack_patch_positions}")


        if args.vla_checkpoint:
            model_path = _resolve_suite_path(args.vla_checkpoint, suite_name)
        else:
            model_path = resolve_openvla_checkpoint(suite_name, args.model_root)

        cfg = argparse.Namespace(
            model_family="openvla",
            pretrained_checkpoint=model_path,
            load_in_8bit=False,
            load_in_4bit=False,
            center_crop=True,
            task_suite_name=suite_name,
            unnorm_key=suite_name,
        )

        LOGGER.info(f"  Loading VLA: {model_path}")
        model = get_model(cfg, DEVICE=device)
        if cfg.unnorm_key not in model.norm_stats:
            alt = f"{cfg.unnorm_key}_no_noops"
            if alt in model.norm_stats:
                cfg.unnorm_key = alt
        processor = get_processor(cfg)
        resize_size = get_image_resize_size(cfg)
        LOGGER.info(f"  VLA ready. resize={resize_size}")


        rar_checkpoint = _resolve_rar_checkpoint(args.rar_checkpoint, suite_name)
        LOGGER.info(f"  Loading RAR decoder: {rar_checkpoint}")
        rar_decoder = load_rar_decoder(rar_checkpoint, device)


        dino_hooks = DinoV2FeatureHooks(model, DRL_LAYERS)


        drl = DeepRepresentationLocalizer(
            banks=banks,
            device=device,
            threshold=args.drl_threshold,
            refine_grid=DRL_REFINEMENT_GRID_SIZE,
            min_cells=DRL_MINIMUM_COMPONENT_CELLS,
        )


        task_suite = benchmark_dict[suite_name]()
        suite_results = {}

        n_tasks = min(args.n_tasks, task_suite.n_tasks)
        for patch_idx, patch_path in enumerate(patch_paths):
            patch_label = _patch_label_from_path(patch_path, patch_dir)
            LOGGER.info(f"\n  Patch [{patch_idx+1}/{len(patch_paths)}]: {patch_label}")
            LOGGER.info(f"    path={patch_path}")
            patch_u8 = load_adversarial_patch(str(patch_path))
            _, ph, pw = patch_u8.shape
            LOGGER.info(f"    size={pw}x{ph}")

            patch_results = {}
            for task_id in range(n_tasks):
                task_t0 = _time_now(device)
                task = task_suite.get_task(task_id)
                task_description = task.language
                initial_states = task_suite.get_task_init_states(task_id)
                env, env_task_desc = get_libero_env(
                    task, "openvla", resolution=ENVIRONMENT_RESOLUTION
                )

                LOGGER.info(
                    f"\n    Task [{task_id+1}/{n_tasks}]: "
                    f"{task_description[:60]}"
                )
                task_result = {}

                for condition in conditions:
                    successes = 0
                    total_defense_events = 0
                    episodes_data = []
                    episode_patch_positions = []

                    for ep in range(args.num_episodes):
                        state_idx = ep % len(initial_states)
                        attack_patch_position = patch_position_rng.choice(
                            attack_patch_positions
                        )
                        episode_patch_positions.append(attack_patch_position)

                        episode_rar_decoder = (
                            rar_decoder if condition == "attack_with_dra_guard" else None
                        )

                        t0 = time.time()
                        result = run_episode(
                            env=env,
                            initial_state=initial_states[state_idx],
                            model=model,
                            processor=processor,
                            cfg=cfg,
                            resize_size=resize_size,
                            task_description=env_task_desc,
                            patch_u8=patch_u8,
                            attack_patch_position=attack_patch_position,
                            condition=condition,
                            rar_decoder=episode_rar_decoder,
                            max_steps=SUITE_MAX_STEPS[suite_name],
                            device=device,
                            dino_hooks=dino_hooks,
                            pdv=pdv,
                            drl=drl,
                            episode_idx=ep,
                            output_dir=video_dir / suite_name / patch_label / f"task{task_id}",
                        )
                        elapsed = time.time() - t0

                        episodes_data.append(result)
                        if result["success"]:
                            successes += 1
                        total_defense_events += len(result.get("defense_events", []))

                        status = "OK" if result["success"] else "FAIL"
                        n_def = len(result.get("defense_events", []))
                        LOGGER.info(
                            f"      [{condition}] ep {ep+1:02d}: {status} "
                            f"position={attack_patch_position} steps={result['steps']} "
                            f"def_events={n_def} time={elapsed:.1f}s"
                        )

                    rate = successes / args.num_episodes
                    task_result[condition] = {
                        "successes": successes,
                        "total": args.num_episodes,
                        "rate": rate,
                        "total_defense_events": total_defense_events,
                        "attack_patch_positions": [
                            list(position) for position in episode_patch_positions
                        ],
                    }
                    LOGGER.info(
                        f"      -> {condition}: {successes}/{args.num_episodes} "
                        f"= {rate:.1%}"
                    )

                patch_results[f"task_{task_id:02d}"] = task_result
                env.close()
                task_elapsed = _time_now(device) - task_t0
                suite_elapsed = _time_now(device) - suite_t0
                LOGGER.info(
                    f"    Task [{task_id+1}/{n_tasks}] done in "
                    f"{task_elapsed:.1f}s (suite elapsed: {suite_elapsed:.0f}s)"
                )

            suite_results[patch_label] = patch_results

        all_results[suite_name] = suite_results


        dino_hooks.close()
        del model, processor, rar_decoder
        pdv._integral_map = None
        drl.reset()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        suite_elapsed = _time_now(device) - suite_t0
        suite_times[suite_name] = suite_elapsed
        LOGGER.info(
            f"\n  Suite {suite_name} done in {suite_elapsed:.1f}s "
            f"({suite_elapsed/60:.1f} min)"
        )

    total_elapsed = _time_now(device) - total_t0


    LOGGER.info("\n[3] Saving results...")
    output_json = output_dir / "dra_guard_results.json"
    save_data = {
        "config": {
            "suites": suites,
            "conditions": conditions,
            "patch_paths": [str(p) for p in patch_paths],
            "patch_dir": str(args.patch_dir),
            "patch_glob": str(args.patch_glob),
            "rollout_dir": str(args.rollout_dir),
            "attack_patch_positions_by_suite": ATTACK_PATCH_POSITIONS_BY_SUITE,
            "rar_checkpoint_arg": args.rar_checkpoint,
            "num_episodes": args.num_episodes,
            "drl_threshold": args.drl_threshold,
        },
        "results": all_results,
    }
    with open(output_json, "w") as f:
        json.dump(save_data, f, indent=2)
    LOGGER.info(f"  Saved: {output_json}")


    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("  SUMMARY")
    LOGGER.info("=" * 70)
    LOGGER.info(
        f"  {'Suite':<16} {'Patch':<28} {'Condition':<22} {'Success Rate'}"
    )
    LOGGER.info(f"  {'-'*16} {'-'*28} {'-'*22} {'-'*12}")
    for suite_name, suite_res in all_results.items():
        for patch_label, patch_res in suite_res.items():
            for cond in conditions:
                total_s = 0
                total_n = 0
                for task_res in patch_res.values():
                    if cond in task_res:
                        total_s += task_res[cond]["successes"]
                        total_n += task_res[cond]["total"]
                rate = total_s / total_n if total_n > 0 else 0
                LOGGER.info(
                    f"  {suite_name:<16} {patch_label:<28} {cond:<22} "
                    f"{total_s}/{total_n} = {rate:.1%}"
                )

    LOGGER.info(f"\n  {'--- Timing ---'}")
    for sn, st in suite_times.items():
        LOGGER.info(f"  {sn:<16} {st:.1f}s ({st/60:.1f} min)")
    LOGGER.info(
        f"  {'TOTAL':<16} {total_elapsed:.1f}s "
        f"({total_elapsed/60:.1f} min)"
    )
    LOGGER.info("=" * 70)
    LOGGER.info("Done.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate DRA-Guard using DRL-PDV intersection and RAR recovery"
    )

    p.add_argument("--task_suite_name", type=str, default="libero_10",
                   help="Comma-separated suite names")
    p.add_argument("--n_tasks", type=int, default=EVALUATION_NUM_TASKS)
    p.add_argument("--model_root", type=str, default=None,
                   help="Optional local OpenVLA root. Without it, use a valid models/openvla checkpoint or Hugging Face.")
    p.add_argument("--patch_path", type=str, default=None,
                   help="Path to adversarial patch .pt file. If omitted, choose the first match under --patch_dir.")
    p.add_argument("--patch_dir", type=str,
                   default=str(PROJECT_ROOT / "data" / "adversarial_patches"),
                   help="Directory to search for patch files when --patch_path is omitted.")
    p.add_argument("--patch_glob", type=str, default="**/patch.pt",
                   help="Glob pattern (relative to --patch_dir) used when --patch_path is omitted.")
    p.add_argument("--rar_checkpoint", type=str,
                   default=str(PROJECT_ROOT / "models" / "rar" / "best_decoder.pt"),
                   help="Path to models/rar/best_decoder.pt.")
    p.add_argument("--vla_checkpoint", type=str, default=None,
                   help="Optional VLA path/template. If omitted, use per-suite defaults.")
    p.add_argument("--num_episodes", type=int, default=EVALUATION_NUM_EPISODES,
                   help="Episodes per condition per task")
    p.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        choices=AVAILABLE_CONDITIONS,
        default=DEFAULT_CONDITIONS,
        help=(
            "Evaluation conditions. Default: attack_with_dra_guard. "
            "Other choices are clean and attack_no_defense."
        ),
    )
    p.add_argument("--cuda_id", type=int, default=0)
    p.add_argument("--drl_threshold", type=float, default=DRL_ANOMALY_THRESHOLD,
                   help="DRL prototype-anomaly threshold")
    p.add_argument("--output_dir", type=str,
                   default=str(PROJECT_ROOT / "outputs" / "evaluation"),
                   help="Output directory for JSON results")
    p.add_argument("--rollout_dir", type=str,
                   default=str(PROJECT_ROOT / "rollouts"),
                   help="Output directory for rollout videos")

    return p.parse_args()


if __name__ == "__main__":
    main()
