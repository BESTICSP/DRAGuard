# DRA-Guard: Deep Representation Alignment for VLA Adversarial Patch Defense

## Abstract

Vision-Language-Action (VLA) models achieve end-to-end coupling of perception, language, and control, yet this integration exposes them to adversarial threats at the visual level. However, existing defense methods either lack mechanisms to model the continuous action streams of VLA or incur prohibitive computational overhead through adversarial training. Through a comprehensive analysis, we reveal that local visual perturbations evolve into anomalous semantic representations as they propagate through the Vision Transformer (ViT), that is, representations that can independently reproduce attack behavior even without visible patches in the input. Leveraging this insight, we propose Deep Representation Alignment for VLA Adversarial Patch Defense (DRA-Guard), a test-time defense framework that detects anomalous regions through deep representation alignment, verifies them through the consistency of physical and visual, and reconstructs patch-free outputs to restore benign behavior. Extensive experiments across diverse robotic manipulation suites under multiple adversarial patch configurations demonstrate that DRA-Guard maintains nominal performance and significantly mitigates adversarial patch attacks.

## Setup

The reference environment uses Linux, Python 3.10, an NVIDIA GPU, and CUDA 12.1 or later.

```bash
conda create -n dra-guard python=3.10 -y
conda activate dra-guard
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## Download Models

This downloads the OpenVLA checkpoints for all four LIBERO suites.

```bash
bash scripts/download_models.sh
```

## Build the Prototype Bank

Build the shared prototype bank from clean rollouts:

```bash
bash scripts/build_prototype_bank.sh 0
```

The argument specifies the CUDA device. The generated prototype bank is saved to `data/prototype_bank/prototypes/`.

## Train RAR

```bash
bash scripts/train_rar.sh libero_10 0
```

The arguments are the LIBERO suite and CUDA device. The best checkpoint is saved to `models/rar/best_decoder.pt`.

## Evaluate DRA-Guard

```bash
bash scripts/evaluate.sh libero_10 0
```

To run multiple conditions:

```bash
CONDITIONS="clean attack_no_defense attack_with_dra_guard" \
    bash scripts/evaluate.sh libero_10 0
```

Results are saved to `outputs/evaluation/`, and videos are saved to `rollouts/`.

## Insight Experiment

```bash
bash scripts/run_analysis.sh \
    libero_spatial libero_spatial upa 0
```

Outputs are saved to `outputs/analysis/insight/` and `rollouts/analysis/insight/`.

## Configuration

Experiment defaults are defined in `configs/dra_guard.yaml`.
