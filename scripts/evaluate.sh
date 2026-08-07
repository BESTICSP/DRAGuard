#!/bin/bash
set -e

root=$(cd "$(dirname "$0")/.." && pwd)
suite=${1:-libero_10}
cuda_id=${2:-0}
patch_dir=${PATCH_DIR:-"$root/data/adversarial_patches"}
rar_checkpoint=${RAR_CHECKPOINT:-"$root/models/rar/best_decoder.pt"}
model_root=${MODEL_ROOT:-"$root/models/openvla"}
read -r -a conditions <<< "${CONDITIONS:-attack_with_dra_guard}"

evaluation_args=()
[[ -z "${NUM_EPISODES:-}" ]] || evaluation_args+=(--num_episodes "$NUM_EPISODES")
[[ -z "${N_TASKS:-}" ]] || evaluation_args+=(--n_tasks "$N_TASKS")
[[ -z "${DRL_THRESHOLD:-}" ]] || evaluation_args+=(--drl_threshold "$DRL_THRESHOLD")
cd "$root"

python -m experiments.robot.evaluate_dra_guard \
    --task_suite_name "$suite" \
    --cuda_id "$cuda_id" \
    --model_root "$model_root" \
    --conditions "${conditions[@]}" \
    --patch_dir "$patch_dir" \
    --rar_checkpoint "$rar_checkpoint" \
    --output_dir "$root/outputs/evaluation" \
    --rollout_dir "$root/rollouts" \
    "${evaluation_args[@]}"
