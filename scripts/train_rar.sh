#!/bin/bash
set -e

root=$(cd "$(dirname "$0")/.." && pwd)
suite=${1:-libero_10}
cuda_id=${2:-0}
model_root=${MODEL_ROOT:-"$root/models/openvla"}

frame_args=()
[[ -z "${FRAMES_PER_TASK:-}" ]] || frame_args=(--frames_per_task "$FRAMES_PER_TASK")

training_args=()
[[ -z "${EPOCHS:-}" ]] || training_args+=(--epochs "$EPOCHS")
[[ -z "${BATCH_SIZE:-}" ]] || training_args+=(--batch_size "$BATCH_SIZE")
[[ -z "${LAMBDA_ALIGN:-}" ]] || training_args+=(--lambda_align "$LAMBDA_ALIGN")
cd "$root"

echo "Collecting RAR training frames: $suite"
python -m experiments.robot.rar.collect_clean_frames \
    --task_suite_name "$suite" \
    --cuda_id "$cuda_id" \
    --model_root "$model_root" \
    "${frame_args[@]}"

echo "Training RAR decoder: $suite"
python -m experiments.robot.rar.train \
    --suite "$suite" \
    --cuda_id "$cuda_id" \
    --model_root "$model_root" \
    "${training_args[@]}"
