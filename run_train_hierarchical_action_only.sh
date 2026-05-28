#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Minimal run: action level only, no augmentation, no eval.
# Use this to verify the pipeline works before enabling full hierarchical mode.
#
# Same as the full version but:
#   --keystep_every 1000  (effectively disabled)
#   --abstract_every 1000 (effectively disabled)
#   --no_aug              (no SimCLR augmentation)
#   --eval_every 0        (no in-training eval)
#   --multi_text 0        (no multi-text candidates)
#
# The loss degrades to standard CLIP InfoNCE — same as train.py.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-vllm}"

# ==============================================================================
# GPU & experiment
# ==============================================================================
NPROC="${NPROC:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_NAME="${RUN_NAME:-hierarchical_action_only_test}"

# ==============================================================================
# Data paths
# ==============================================================================
MAIN_CSV_PATH="${MAIN_CSV_PATH:-/mnt/mydisk/CLIP/surglavi_level_csv/all_video.csv}"
VIDEO_ROOT_FOLDER="${VIDEO_ROOT_FOLDER:-/mnt/mydisk/CLIP/downloaded_video_224_test}"
ANNOTATIONS_ROOT="${ANNOTATIONS_ROOT:-/mnt/mydisk/CLIP/surglavi_level_csv}"

# ==============================================================================
# Model
# ==============================================================================
EMBED_DIM="${EMBED_DIM:-512}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
MAX_LENGTH="${MAX_LENGTH:-256}"
NUM_FRAMES="${NUM_FRAMES:-4}"
TEXT_MODEL_NAME="${TEXT_MODEL_NAME:-marcobombieri/surgicberta}"
VISION_PRETRAINED_WEIGHTS="${VISION_PRETRAINED_WEIGHTS:-/mnt/mydisk/CLIP/lemonfm.pth}"

# ==============================================================================
# Training (small scale for testing)
# ==============================================================================
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE_ACTION="${BATCH_SIZE_ACTION:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.02}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/$RUN_NAME}"

mkdir -p "$OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

torchrun \
    --standalone \
    --nproc_per_node="$NPROC" \
    train_hierarchical.py \
    --epochs "$EPOCHS" \
    --embed_dim "$EMBED_DIM" \
    --image_size "$IMAGE_SIZE" \
    --max_length "$MAX_LENGTH" \
    --num_frames "$NUM_FRAMES" \
    --text_model_name "$TEXT_MODEL_NAME" \
    --vision_pretrained_weights "$VISION_PRETRAINED_WEIGHTS" \
    --main_csv_path "$MAIN_CSV_PATH" \
    --video_root_folder "$VIDEO_ROOT_FOLDER" \
    --annotations_root "$ANNOTATIONS_ROOT" \
    --batch_size_action "$BATCH_SIZE_ACTION" \
    --num_workers "$NUM_WORKERS" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --action_every 1 \
    --keystep_every 1000 \
    --abstract_every 1000 \
    --no_aug \
    --eval_every 0 \
    --multi_text 0 \
    --output_dir "$OUTPUT_DIR"
