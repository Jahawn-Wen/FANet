#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/path/to/University-Release/train}"
GPU_IDS="${GPU_IDS:-0}"
RUN_NAME="${FANET_RUN_NAME:-fanet_best_reproduce}"

python train.py \
  --name="$RUN_NAME" \
  --experiment_name="$RUN_NAME" \
  --data_dir="$DATA_DIR" \
  --views=3 \
  --droprate=0.5 \
  --extra_Google \
  --share \
  --stride=1 \
  --h=256 \
  --w=256 \
  --lr=0.005 \
  --gpu_ids="$GPU_IDS" \
  --norm=spade \
  --iaa \
  --focal \
  --multi_weather \
  --btnk 0 1 1 0 0 0 0 \
  --conv_norm=none \
  --reptile \
  --adain=a \
  --seed=1
