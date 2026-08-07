"""Collect DINOv2 and SigLIP patch tokens for Insight experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from libero.libero import benchmark

from analysis.insight.common import (
    apply_patch,
    build_policy_observation,
    build_token_indices,
    load_patch,
    load_policy,
    parse_position,
    resolve_patch,
)
from analysis.insight.config import (
    DEFAULT_ANALYSIS_FRAMES,
    DEFAULT_COLLECTION_STEPS,
    DEFAULT_SEED,
    DEFAULT_TASK_IDS,
    SUITES,
    TOKEN_ROOT,
    token_file,
)
from analysis.insight.feature_hooks import LayerFeatureCollector
from experiments.robot.libero.libero_utils import (
    get_libero_env,
    get_libero_image,
)
from experiments.robot.robot_utils import (
    get_action,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect attack-region tokens from every DINOv2 and SigLIP layer."
        ),
    )
    parser.add_argument(
        "--suite",
        choices=SUITES,
        default="libero_spatial",
        help="LIBERO suite used to collect source representations.",
    )
    parser.add_argument(
        "--attack",
        choices=("upa", "tma", "uada"),
        default="upa",
        help="Adversarial patch family.",
    )
    parser.add_argument(
        "--patch_path",
        type=str,
        default=None,
        help="Optional explicit patch.pt path.",
    )
    parser.add_argument(
        "--patch_position",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        help="Optional pixel position. Each suite has a default when omitted.",
    )
    parser.add_argument(
        "--task_ids",
        type=int,
        nargs="+",
        default=DEFAULT_TASK_IDS,
        help="Task IDs to collect.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=DEFAULT_ANALYSIS_FRAMES,
        help="Rollout frame indices saved in each representation file.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help=f"Collection horizon. Default: {DEFAULT_COLLECTION_STEPS}.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--cuda_id",
        type=int,
        default=0,
        help="CUDA device index used by this single process.",
    )
    parser.add_argument(
        "--model_root",
        type=str,
        default=None,
        help="Optional root containing the four suite-specific OpenVLA folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=TOKEN_ROOT,
        help="Root directory for collected representation files.",
    )
    return parser.parse_args()


def collect_task(
    *,
    task_suite,
    task_id: int,
    model,
    processor,
    cfg,
    device: str,
    collector: LayerFeatureCollector,
    patch: torch.Tensor,
    position: tuple[int, int],
    token_indices: list[int],
    requested_frames: list[int],
    max_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Run one patched rollout and return layer-wise tokens from both towers."""
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

    dino_frames = []
    siglip_frames = []
    captured_frames = []
    requested = set(requested_frames)

    try:
        env.reset()
        obs = env.set_init_state(initial_states[0])

        for step in range(max_steps):
            clean_image = get_libero_image(obs, 224)
            attacked_image = apply_patch(clean_image, patch, position)

            collector.clear()
            policy_observation = build_policy_observation(attacked_image, obs)
            action = get_action(
                cfg,
                model,
                policy_observation,
                task_description,
                processor=processor,
                DEVICE=device,
            )

            if step in requested:
                dino_frames.append(collector.patch_tokens("dino", token_indices))
                siglip_frames.append(collector.patch_tokens("siglip", token_indices))
                captured_frames.append(step)

            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)
            obs, _, done, _ = env.step(action.tolist())

            if done or len(captured_frames) == len(requested):
                break
    finally:
        env.close()

    if not captured_frames:
        raise RuntimeError(
            f"Task {task_id} did not reach any requested frame: {requested_frames}."
        )


    dino = torch.stack(dino_frames, dim=1)
    siglip = torch.stack(siglip_frames, dim=1)
    return dino, siglip, captured_frames


def main() -> None:
    args = parse_args()
    set_seed_everywhere(args.seed)

    patch_path = resolve_patch(args.attack, args.patch_path)
    patch = load_patch(patch_path)
    position = parse_position(args.patch_position, args.suite)
    patch_hw = (int(patch.shape[1]), int(patch.shape[2]))


    token_indices = build_token_indices(position, patch_hw)
    model, processor, cfg, device = load_policy(
        args.suite,
        args.cuda_id,
        args.model_root,
    )
    collector = LayerFeatureCollector(model)

    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    max_steps = args.max_steps or DEFAULT_COLLECTION_STEPS
    requested_frames = sorted(set(args.frames))

    try:
        for task_id in args.task_ids:
            if not 0 <= task_id < task_suite.n_tasks:
                raise ValueError(
                    f"task_id={task_id} is outside the task range of {args.suite}."
                )


            set_seed_everywhere(args.seed)
            dino, siglip, captured_frames = collect_task(
                task_suite=task_suite,
                task_id=task_id,
                model=model,
                processor=processor,
                cfg=cfg,
                device=device,
                collector=collector,
                patch=patch,
                position=position,
                token_indices=token_indices,
                requested_frames=requested_frames,
                max_steps=max_steps,
            )
            if captured_frames != requested_frames:
                raise RuntimeError(
                    f"Task {task_id} captured frames {captured_frames}, which do not "
                    f"fully cover the requested frames {requested_frames}."
                )

            output_path = token_file(args.suite, args.attack, task_id, args.output_dir)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "meta": {
                        "suite": args.suite,
                        "attack": args.attack,
                        "task_id": task_id,
                        "frames": captured_frames,
                        "patch_position": position,
                        "patch_hw": patch_hw,
                        "token_indices": token_indices,
                        "seed": args.seed,
                        "backbones": ["dinov2", "siglip"],
                        "dino_layers": list(range(dino.shape[0])),
                        "siglip_layers": list(range(siglip.shape[0])),
                    },
                    "dino": dino,
                    "siglip": siglip,
                },
                output_path,
            )
            print(f"Saved task {task_id} representations to {output_path}")
    finally:
        collector.close()


if __name__ == "__main__":
    main()
