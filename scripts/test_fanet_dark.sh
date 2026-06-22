#!/usr/bin/env bash
set -euo pipefail

TEST_DIR="${TEST_DIR:-/path/to/University-Release/test}"
GPU_IDS="${GPU_IDS:-0}"

python test_iaa_all.py \
  --name best_ckpt \
  --test_dir "$TEST_DIR" \
  --batchsize 128 \
  --gpu_ids "$GPU_IDS" \
  --iaa \
  --weather dark \
  --modes d2s
