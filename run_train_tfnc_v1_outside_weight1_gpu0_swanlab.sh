#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# TFNC-v1 ablation: keep every outside frame in the denominator with weight 1.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RUN_NAME="${RUN_NAME:-tfnc_v1_xpool_8f_expand15_outside_weight1_gpu0}"
export EXP_NAME="${EXP_NAME:-$RUN_NAME}"
export SAVE_PREFIX="${SAVE_PREFIX:-/data/znh/outputs/$RUN_NAME/}"
export VIDEO_ROOT_FOLDER="${VIDEO_ROOT_FOLDER:-/home/zhangnuohua/nfs_data/CLIP/downloaded_video_224_test}"

export TRAINING_METHOD="${TRAINING_METHOD:-tfnc}"
export TFNC_UNIFORM_OUTSIDE_WEIGHT="${TFNC_UNIFORM_OUTSIDE_WEIGHT:-true}"
export EPOCHS="${EPOCHS:-30}"

export NUM_WORKERS="${NUM_WORKERS:-6}"
export VIDEO_READER_THREADS="${VIDEO_READER_THREADS:-1}"
export DECODE_CPU_THREAD_LIMIT="${DECODE_CPU_THREAD_LIMIT:-1}"
export DATALOADER_IN_ORDER="${DATALOADER_IN_ORDER:-false}"
export DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"

bash run_train_tfnc_swanlab.sh
