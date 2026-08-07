#!/bin/bash
set -e

root=$(cd "$(dirname "$0")/.." && pwd)
source_suite=${1:-libero_spatial}
target_suite=${2:-$source_suite}
attack=${3:-upa}
cuda_id=${4:-0}
dino_layer=${5:-18}
siglip_layer=${6:-18}

model_root=${MODEL_ROOT:-"$root/models/openvla"}
token_dir=${TOKEN_DIR:-"$root/outputs/analysis/insight/tokens"}
rollout_dir=${ROLLOUT_DIR:-"$root/rollouts/analysis/insight"}
trials=${TRIALS:-2}
source_mode=${SOURCE_MODE:-mean}
source_frame=${SOURCE_FRAME:-10}

patch_args=()
[[ -z "${PATCH_PATH:-}" ]] || patch_args=(--patch_path "$PATCH_PATH")

collection_step_args=()
[[ -z "${COLLECTION_STEPS:-}" ]] || collection_step_args=(--max_steps "$COLLECTION_STEPS")

evaluation_step_args=()
[[ -z "${MAX_STEPS:-}" ]] || evaluation_step_args=(--max_steps "$MAX_STEPS")

cd "$root"

echo "Collecting Insight representations"
python -m analysis.insight.collect_representations \
    --suite "$source_suite" \
    --attack "$attack" \
    --cuda_id "$cuda_id" \
    --model_root "$model_root" \
    --output_dir "$token_dir" \
    "${patch_args[@]}" \
    "${collection_step_args[@]}"

echo "Evaluating Insight representation transplant"
python -m analysis.insight.evaluate_transplant \
    --source_suite "$source_suite" \
    --target_suite "$target_suite" \
    --attack "$attack" \
    --layers "$dino_layer" "$siglip_layer" \
    --source_mode "$source_mode" \
    --source_frame "$source_frame" \
    --trials "$trials" \
    --cuda_id "$cuda_id" \
    --model_root "$model_root" \
    --token_dir "$token_dir" \
    --rollout_dir "$rollout_dir" \
    "${evaluation_step_args[@]}"
