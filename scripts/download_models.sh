#!/bin/bash
set -e

root=$(cd "$(dirname "$0")/.." && pwd)
model_root=${MODEL_ROOT:-"$root/models/openvla"}
cd "$root"

python -m experiments.robot.download_models \
    --model_root "$model_root"
