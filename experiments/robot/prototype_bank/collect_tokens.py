


import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
import torch
import tqdm
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.model_paths import resolve_openvla_checkpoint
from experiments.robot.settings import ENVIRONMENT_RESOLUTION
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.prototype_bank.config import (
    COLLECT_SEED,
    EMBED_DIM,
    FRAMES_PER_ROLLOUT,
    LOG_DIR,
    MAX_STEPS,
    MINIMUM_POLICY_STEPS,
    NUM_FAIL_PER_TASK,
    NUM_PATCH_TOKENS,
    NUM_PREFIX_TOKENS,
    NUM_STEPS_WAIT,
    NUM_SUCCESS_PER_TASK,
    SUITES,
    TARGET_LAYERS,
    TOKENS_DIR,
)


class DinoV2FeatureHooks:


    def __init__(self, vla, target_layers):
        self.target_layers = list(target_layers)
        self.captured = {layer: None for layer in self.target_layers}
        self._handles = []
        featurizer = vla.vision_backbone.featurizer
        for layer in self.target_layers:
            block = featurizer.blocks[layer]
            self._handles.append(
                block.register_forward_hook(self._create_hook(layer))
            )

    def _create_hook(self, layer):
        def _hook(module, _inp, out):


            patches = out[:, NUM_PREFIX_TOKENS:, :].detach().to(torch.float32).cpu().numpy().astype(np.float16)
            self.captured[layer] = patches
        return _hook

    def clear(self):
        for layer in self.target_layers:
            self.captured[layer] = None

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def build_rollout_config(suite, cuda_id, model_root=None):

    return SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=resolve_openvla_checkpoint(suite, model_root),
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        task_suite_name=suite,
        unnorm_key=suite,
        cuda_id=cuda_id,
    )


def collect_suite_tokens(suite, cuda_id, log_file, model_root=None):
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"=== suite={suite}  device=cuda:{cuda_id} ===")
    log(f"target: {NUM_SUCCESS_PER_TASK} success + {NUM_FAIL_PER_TASK} fail per task; "
        f"{FRAMES_PER_ROLLOUT} frames/rollout via uniform linspace")
    log(f"target layers: {TARGET_LAYERS}")

    set_seed_everywhere(COLLECT_SEED)
    cfg = build_rollout_config(suite, cuda_id, model_root)
    DEVICE = f"cuda:{cuda_id}"

    log(f"loading model {cfg.pretrained_checkpoint}")
    model = get_model(cfg, DEVICE=DEVICE)
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    assert cfg.unnorm_key in model.norm_stats, f"unnorm_key {cfg.unnorm_key} missing"
    processor   = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)

    hooks = DinoV2FeatureHooks(model, TARGET_LAYERS)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict[suite]()
    num_tasks      = task_suite.n_tasks
    max_steps      = MAX_STEPS[suite]


    per_layer = {layer: [] for layer in TARGET_LAYERS}
    n_frames        = 0
    rollout_summary = []

    for task_id in tqdm.tqdm(range(num_tasks), desc=f"{suite} tasks"):
        task            = task_suite.get_task(task_id)
        initial_states  = task_suite.get_task_init_states(task_id)
        n_init_states   = len(initial_states)
        env, task_desc = get_libero_env(
            task, cfg.model_family, resolution=ENVIRONMENT_RESOLUTION
        )

        n_success_kept = 0
        n_fail_kept    = 0
        attempt        = 0
        t_task         = time.time()


        while n_success_kept < NUM_SUCCESS_PER_TASK or n_fail_kept < NUM_FAIL_PER_TASK:
            episode_idx = attempt % n_init_states
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t           = 0
            done        = False

            per_layer_ep = {layer: [] for layer in TARGET_LAYERS}

            while t < max_steps + NUM_STEPS_WAIT:
                try:
                    if t < NUM_STEPS_WAIT:
                        obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    img         = get_libero_image(obs, resize_size)
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                             obs["robot0_gripper_qpos"])),
                    }
                    hooks.clear()
                    action = get_action(cfg, model, observation, task_desc,
                                        processor=processor, DEVICE=DEVICE)

                    for layer in TARGET_LAYERS:
                        per_layer_ep[layer].append(hooks.captured[layer][0])

                    action = normalize_gripper_action(action, binarize=True)
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)
                    obs, _, done, _ = env.step(action.tolist())
                    if done:
                        break
                    t += 1
                except Exception as e:
                    log(f"  task={task_id} attempt={attempt} EXCEPTION at t={t}: {e}")
                    break

            attempt        += 1
            n_policy_steps  = len(per_layer_ep[TARGET_LAYERS[0]])
            outcome         = "SUCC" if done else "FAIL"


            if done and n_success_kept >= NUM_SUCCESS_PER_TASK:
                log(f"  task={task_id} attempt={attempt-1} SUCC but success quota filled -> drop")
                continue
            if (not done) and n_fail_kept >= NUM_FAIL_PER_TASK:
                log(f"  task={task_id} attempt={attempt-1} FAIL but fail quota filled -> drop")
                continue
            if n_policy_steps < MINIMUM_POLICY_STEPS:

                log(f"  task={task_id} attempt={attempt-1} {outcome} but only {n_policy_steps} "
                    f"policy steps -> drop")
                continue


            target_n = min(FRAMES_PER_ROLLOUT, n_policy_steps)
            pick     = np.linspace(0, n_policy_steps - 1, target_n).round().astype(int)
            pick     = np.unique(pick)

            for layer in TARGET_LAYERS:
                for i in pick:
                    per_layer[layer].append(per_layer_ep[layer][i])

            if done:
                n_success_kept += 1
            else:
                n_fail_kept    += 1
            n_frames += len(pick)
            rollout_summary.append({
                "task_id": int(task_id),
                "attempt": int(attempt - 1),
                "episode_idx": int(episode_idx),
                "outcome": outcome,
                "n_policy_steps": int(n_policy_steps),
                "kept_frames": int(len(pick)),
            })
            log(f"  task={task_id} attempt={attempt-1} {outcome} "
                f"steps={n_policy_steps:4d}  kept_frames={len(pick):2d}  "
                f"(succ={n_success_kept}/{NUM_SUCCESS_PER_TASK}  "
                f"fail={n_fail_kept}/{NUM_FAIL_PER_TASK}  "
                f"running total frames={n_frames})")

        log(f"  task={task_id} DONE in {time.time()-t_task:.1f}s "
            f"({attempt} attempts, succ={n_success_kept}, fail={n_fail_kept})")

    hooks.close()


    out_path  = TOKENS_DIR / f"{suite}_dinov2.npz"
    out_arrays = {}
    for layer in TARGET_LAYERS:
        if not per_layer[layer]:
            log(f"  WARNING: layer {layer} has 0 frames!")
            layer_tokens = np.zeros(
                (0, NUM_PATCH_TOKENS, EMBED_DIM), dtype=np.float16
            )
        else:
            layer_tokens = np.stack(per_layer[layer], axis=0)
        out_arrays[f"arr_L{layer}"] = layer_tokens
        log(
            f"  layer L{layer}: stacked shape {layer_tokens.shape}, "
            f"dtype={layer_tokens.dtype}"
        )
    np.savez_compressed(out_path, **out_arrays)
    size_mb = out_path.stat().st_size / 1024 / 1024
    log(f"saved {out_path}  ({size_mb:.1f} MB, {n_frames} frames)")


    meta_path = TOKENS_DIR / f"{suite}_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "suite": suite,
            "config": {
                "num_success_per_task": NUM_SUCCESS_PER_TASK,
                "num_fail_per_task":    NUM_FAIL_PER_TASK,
                "frames_per_rollout":   FRAMES_PER_ROLLOUT,
                "target_layers":        TARGET_LAYERS,
                "max_steps":            max_steps,
                "collect_seed":         COLLECT_SEED,
            },
            "n_tasks": int(num_tasks),
            "n_frames_total": int(n_frames),
            "rollouts": rollout_summary,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    log(f"saved {meta_path}")


    del model, processor
    torch.cuda.empty_cache()
    return n_frames, rollout_summary


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Collect clean DinoV2 tokens from OpenVLA LIBERO rollouts "
            f"({NUM_SUCCESS_PER_TASK} success + {NUM_FAIL_PER_TASK} fail per task)."
        )
    )
    p.add_argument("--task_suite_name", type=str, required=True,
                   choices=SUITES + ["all"],
                   help="Which suite to collect (or 'all' for serial 4-suite run).")
    p.add_argument("--cuda_id", type=int, default=0)
    p.add_argument(
        "--model_root",
        type=str,
        default=None,
        help="Optional local OpenVLA root. MODEL_ROOT is used when omitted.",
    )
    return p.parse_args()


def main():
    args    = parse_args()
    targets = SUITES if args.task_suite_name == "all" else [args.task_suite_name]
    for suite in targets:
        log_path = LOG_DIR / f"collect_{suite}.log"
        with open(log_path, "a") as log_file:
            log_file.write(f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            n_frames, _ = collect_suite_tokens(
                suite, args.cuda_id, log_file, args.model_root
            )
            log_file.write(f"DONE suite={suite} n_frames={n_frames}\n")


if __name__ == "__main__":
    main()
