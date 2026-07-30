#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RUN_NAME="${RUN_NAME:-tfnc_xpool_8f_expand175_gpu0}"
export EXP_NAME="${EXP_NAME:-$RUN_NAME}"
export SAVE_PREFIX="${SAVE_PREFIX:-/home/zhangnuohua/nfs_data/SurgAlign_results/$RUN_NAME/}"
export VIDEO_ROOT_FOLDER="${VIDEO_ROOT_FOLDER:-/data/znh/downloaded_video_224_test}"

export TRAINING_METHOD="${TRAINING_METHOD:-tfnc}"
export TRAIN_WINDOW_EXPAND_RATIO=1.75
export EPOCHS=20

export NUM_WORKERS="${NUM_WORKERS:-8}"
export VIDEO_READER_THREADS="${VIDEO_READER_THREADS:-1}"
export DECODE_CPU_THREAD_LIMIT="${DECODE_CPU_THREAD_LIMIT:-1}"
export DATALOADER_IN_ORDER="${DATALOADER_IN_ORDER:-false}"
export DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"

bash run_train_tfnc_swanlab.sh
