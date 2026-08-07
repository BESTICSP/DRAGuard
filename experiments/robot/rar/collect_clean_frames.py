


from __future__ import annotations

import argparse
import os
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.model_paths import resolve_openvla_checkpoint
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


from experiments.robot.rar.config import (
    CLEAN_FRAMES_DIR,
    FRAMES_PER_TASK,
    FRAME_COLLECTION_SEED,
)
from experiments.robot.settings import (
    ENVIRONMENT_RESOLUTION,
    NUM_STEPS_WAIT,
    SUITES,
    SUITE_MAX_STEPS,
)

OUTPUT_DIR = CLEAN_FRAMES_DIR


def build_rollout_config(
    suite: str, cuda_id: int, model_root: str | None = None
) -> SimpleNamespace:
    model_path = resolve_openvla_checkpoint(suite, model_root)
    model_basename = os.path.basename(model_path)
    unnorm_key = (
        model_basename.split("finetuned-")[-1].replace("-", "_")
        if "finetuned-" in model_basename
        else suite
    )
    return SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=model_path,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        task_suite_name=suite,
        unnorm_key=unnorm_key,
        cuda_id=cuda_id,
    )


def collect_task_frames(
    cfg: SimpleNamespace,
    model,
    processor,
    resize_size: int,
    env,
    task_description: str,
    initial_states: list,
    max_steps: int,
    frames_per_task: int,
    device: str,
) -> list:


    all_frames = []
    n_init_states = len(initial_states)
    attempt = 0

    while len(all_frames) < frames_per_task:
        episode_idx = attempt % n_init_states
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx])


        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))


        for step in range(max_steps):
            img = get_libero_image(obs, resize_size)
            all_frames.append(img)


            observation = {
                "full_image": img,
                "state": np.concatenate((
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )),
            }
            action = get_action(
                cfg, model, observation, task_description,
                processor=processor, DEVICE=device,
            )
            action = normalize_gripper_action(action, binarize=True)
            if cfg.model_family == "openvla":
                action = invert_gripper_action(action)

            obs, _, done, _ = env.step(
                action.tolist() if isinstance(action, np.ndarray) else action.tolist()
            )
            if done:
                break

        attempt += 1
        print(f"      attempt={attempt}, collected={len(all_frames)} frames so far")


    total = len(all_frames)
    indices = np.linspace(0, total - 1, frames_per_task).round().astype(int)
    indices = np.unique(indices)

    if len(indices) < frames_per_task:
        extra = np.linspace(0, total - 1, frames_per_task - len(indices) + 10).round().astype(int)
        indices = np.unique(np.concatenate([indices, extra]))[:frames_per_task]

    sampled = [all_frames[i] for i in indices[:frames_per_task]]
    return sampled


def collect_suite_frames(
    suite: str,
    cuda_id: int,
    frames_per_task: int,
    model_root: str | None = None,
):

    print(f"\n{'='*60}")
    print(f"  Suite: {suite}")
    print(f"  Frames per task: {frames_per_task}")
    print(f"{'='*60}")

    set_seed_everywhere(FRAME_COLLECTION_SEED)
    cfg = build_rollout_config(suite, cuda_id, model_root)
    device = f"cuda:{cuda_id}"

    print(f"  Loading model: {cfg.pretrained_checkpoint}")
    model = get_model(cfg, DEVICE=device)
    if cfg.unnorm_key not in model.norm_stats:
        alt = f"{cfg.unnorm_key}_no_noops"
        if alt in model.norm_stats:
            cfg.unnorm_key = alt
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)
    print(f"  Model loaded. resize={resize_size}, unnorm_key={cfg.unnorm_key}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite]()
    num_tasks = task_suite.n_tasks
    max_steps = SUITE_MAX_STEPS[suite]

    suite_dir = OUTPUT_DIR / suite
    suite_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(
            task, cfg.model_family, resolution=ENVIRONMENT_RESOLUTION
        )

        print(f"\n  Task {task_id}/{num_tasks}: {task_description[:60]}")

        t0 = time.time()
        frames = collect_task_frames(
            cfg=cfg,
            model=model,
            processor=processor,
            resize_size=resize_size,
            env=env,
            task_description=task_description,
            initial_states=initial_states,
            max_steps=max_steps,
            frames_per_task=frames_per_task,
            device=device,
        )
        elapsed = time.time() - t0


        task_dir = suite_dir / f"task{task_id:02d}"
        task_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(frames):
            out_path = task_dir / f"frame_{i:03d}.png"

            cv2.imwrite(str(out_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        total_saved += len(frames)
        print(f"    Saved {len(frames)} frames to {task_dir} ({elapsed:.1f}s)")

        env.close()


    del model, processor
    torch.cuda.empty_cache()

    print(f"\n  Suite {suite} done. Total frames: {total_saved}")
    return total_saved


def parse_args():
    p = argparse.ArgumentParser(
        description="Collect clean rollout frames as PNG for decoder training."
    )
    p.add_argument("--task_suite_name", type=str, default="libero_10",
                   choices=SUITES + ["all"],
                   help="Which suite (or 'all' for all 4)")
    p.add_argument("--frames_per_task", type=int, default=FRAMES_PER_TASK,
                   help="Number of frames to collect per task (uniform sampled)")
    p.add_argument("--cuda_id", type=int, default=0)
    p.add_argument(
        "--model_root",
        type=str,
        default=None,
        help="Optional local OpenVLA root. MODEL_ROOT is used when omitted.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    targets = SUITES if args.task_suite_name == "all" else [args.task_suite_name]

    print("=" * 60)
    print("  Clean Frame Collection for Decoder Training")
    print(f"  Suites: {targets}")
    print(f"  Frames/task: {args.frames_per_task}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    grand_total = 0
    t_start = time.time()

    for suite in targets:
        n = collect_suite_frames(
            suite, args.cuda_id, args.frames_per_task, args.model_root
        )
        grand_total += n

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  All done! Total frames: {grand_total}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
