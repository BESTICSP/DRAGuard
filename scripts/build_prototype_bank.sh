#!/bin/bash
set -e

root=$(cd "$(dirname "$0")/.." && pwd)
cuda_id=${1:-0}
model_root=${MODEL_ROOT:-"$root/models/openvla"}
cd "$root"

echo "Collecting clean tokens for all suites"
python -m experiments.robot.prototype_bank.collect_tokens \
    --task_suite_name all \
    --cuda_id "$cuda_id" \
    --model_root "$model_root"

echo "Building the shared L18-L22 prototype bank"
python -m experiments.robot.prototype_bank.build_prototypes_shared
