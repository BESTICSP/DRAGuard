"""Transplant attack representations into clean Insight evaluation rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from libero.libero import benchmark

from analysis.insight.common import (
    build_policy_observation,
    build_token_indices,
    load_policy,
    parse_position,
    save_video,
)
from analysis.insight.config import (
    DEFAULT_LAYERS,
    DEFAULT_MAX_STEPS_BY_SUITE,
    DEFAULT_SEED,
    DEFAULT_SOURCE_TASK,
    DEFAULT_TARGET_TASKS,
    ROLLOUT_ROOT,
    SUITES,
    TOKEN_ROOT,
    token_file,
)
from analysis.insight.feature_hooks import RepresentationReplacer
from experiments.robot.settings import NUM_STEPS_WAIT
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
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
            "Transplant source-task DINOv2 and SigLIP attack representations "
            "into clean target rollouts."
        ),
    )
    parser.add_argument(
        "--attack",
        choices=("upa", "tma", "uada"),
        default="upa",
        help="Patch family associated with the collected source representations.",
    )
    parser.add_argument(
        "--source_suite",
        choices=SUITES,
        default="libero_spatial",
        help="Suite from which source representations were collected.",
    )
    parser.add_argument(
        "--target_suite",
        choices=SUITES,
        default=None,
        help="Evaluation suite. Defaults to source_suite.",
    )
    parser.add_argument(
        "--source_task",
        type=int,
        default=DEFAULT_SOURCE_TASK,
        help="Task providing the transplanted source representations.",
    )
    parser.add_argument(
        "--target_tasks",
        type=int,
        nargs="+",
        default=DEFAULT_TARGET_TASKS,
        help="Tasks evaluated with transplanted representations.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs=2,
        default=DEFAULT_LAYERS,
        metavar=("DINO_LAYER", "SIGLIP_LAYER"),
        help=(
            "Optional visual-layer pair. The default uses DINOv2 L18 and "
            "SigLIP L18."
        ),
    )
    parser.add_argument(
        "--source_mode",
        choices=("mean", "frame"),
        default="mean",
        help="Use the mean across collected frames or one source frame.",
    )
    parser.add_argument(
        "--source_frame",
        type=int,
        default=10,
        help="Source frame used when source_mode=frame.",
    )
    parser.add_argument(
        "--target_position",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        help="Optional target pixel position. The target suite default is used otherwise.",
    )
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help=(
            "Optional rollout horizon override. Defaults by target suite: "
            "spatial=220, object=280, goal=300, libero_10=520."
        ),
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
        "--token_dir",
        type=Path,
        default=TOKEN_ROOT,
        help="Root directory containing collected representation files.",
    )
    parser.add_argument(
        "--rollout_dir",
        type=Path,
        default=ROLLOUT_ROOT,
        help="Root directory for evaluation videos.",
    )
    return parser.parse_args()


def select_layer(
    tensor: torch.Tensor,
    available_layers: list[int],
    layer: int,
) -> torch.Tensor:
    """Select [frame, patch_token, channel] by the stored layer index."""
    if layer not in available_layers:
        raise ValueError(
            f"Layer {layer} is unavailable. Collected layers: {available_layers}."
        )
    return tensor[available_layers.index(layer)]


def select_source_embedding(
    layer_frames: torch.Tensor,
    captured_frames: list[int],
    mode: str,
    source_frame: int,
) -> torch.Tensor:
    """Select the seven-frame mean or one captured source frame."""
    if mode == "mean":
        return layer_frames.mean(dim=0)
    if source_frame not in captured_frames:
        raise ValueError(
            f"Frame {source_frame} was not collected. Available frames: "
            f"{captured_frames}."
        )
    return layer_frames[captured_frames.index(source_frame)]


def run_trial(
    *,
    task_suite,
    task_id: int,
    trial: int,
    model,
    processor,
    cfg,
    device: str,
    seed: int,
    max_steps: int,
    video_path: Path,
) -> bool:
    """Run one clean-image rollout with two-tower representation transplant."""
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
    frames = []
    success = False

    try:
        set_seed_everywhere(seed + trial)
        env.reset()
        obs = env.set_init_state(initial_states[trial % len(initial_states)])
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

        for _ in range(max_steps):
            clean_image = get_libero_image(obs, 224)
            frames.append(clean_image)
            policy_observation = build_policy_observation(clean_image, obs)
            action = get_action(
                cfg,
                model,
                policy_observation,
                task_description,
                processor=processor,
                DEVICE=device,
            )
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)
            obs, reward, done, _ = env.step(action.tolist())
            if done:
                success = bool(reward > 0)
                break
    finally:
        env.close()

    suffix = "success" if success else "failure"
    save_video(frames, video_path.with_name(f"{video_path.stem}_{suffix}.mp4"))
    return success


def main() -> None:
    args = parse_args()
    target_suite_name = args.target_suite or args.source_suite
    dino_layer, siglip_layer = args.layers
    set_seed_everywhere(args.seed)

    source_path = token_file(
        args.source_suite,
        args.attack,
        args.source_task,
        args.token_dir,
    )
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Missing source representations: {source_path}. Run "
            "analysis.insight.collect_representations first."
        )

    source = torch.load(source_path, map_location="cpu")
    meta = source["meta"]
    captured_frames = list(meta["frames"])
    dino_frames = select_layer(source["dino"], list(meta["dino_layers"]), dino_layer)
    siglip_frames = select_layer(source["siglip"], list(meta["siglip_layers"]), siglip_layer)
    dino_embedding = select_source_embedding(
        dino_frames,
        captured_frames,
        args.source_mode,
        args.source_frame,
    )
    siglip_embedding = select_source_embedding(
        siglip_frames,
        captured_frames,
        args.source_mode,
        args.source_frame,
    )

    target_position = parse_position(args.target_position, target_suite_name)
    target_indices = build_token_indices(
        target_position,
        tuple(meta["patch_hw"]),
        target_count=dino_embedding.shape[0],
    )

    model, processor, cfg, device = load_policy(
        target_suite_name,
        args.cuda_id,
        args.model_root,
    )
    replacer = RepresentationReplacer(
        model,
        target_indices,
        dino_layer=dino_layer,
        siglip_layer=siglip_layer,
    )
    replacer.set_embeddings(dino_embedding, siglip_embedding)

    task_suite = benchmark.get_benchmark_dict()[target_suite_name]()
    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else DEFAULT_MAX_STEPS_BY_SUITE[target_suite_name]
    )
    layer_tag = f"dino{dino_layer}_siglip{siglip_layer}"
    experiment_dir = (
        args.rollout_dir
        / f"{args.source_suite}_to_{target_suite_name}"
        / args.attack
        / layer_tag
    )

    successes = 0
    total = 0
    try:
        for task_id in args.target_tasks:
            if not 0 <= task_id < task_suite.n_tasks:
                raise ValueError(
                    f"task_id={task_id} is outside the task range of "
                    f"{target_suite_name}."
                )
            for trial in range(args.trials):
                video_path = experiment_dir / f"task_{task_id:02d}_trial_{trial:02d}.mp4"
                success = run_trial(
                    task_suite=task_suite,
                    task_id=task_id,
                    trial=trial,
                    model=model,
                    processor=processor,
                    cfg=cfg,
                    device=device,
                    seed=args.seed,
                    max_steps=max_steps,
                    video_path=video_path,
                )
                successes += int(success)
                total += 1
                print(f"task={task_id}, trial={trial}, success={success}")
    finally:
        replacer.close()

    print(f"Completed: {successes}/{total} successful rollouts.")


if __name__ == "__main__":
    main()
