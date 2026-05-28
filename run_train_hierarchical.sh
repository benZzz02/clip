#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-vllm}"

# ==============================================================================
# GPU & experiment
# ==============================================================================
NPROC="${NPROC:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
RUN_NAME="${RUN_NAME:-hierarchical_peskavlp_full}"
EXP_NAME="${EXP_NAME:-$RUN_NAME}"

# ==============================================================================
# Data paths (same as run_train_frozen_vis_swanlab.sh)
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
# Training
# ==============================================================================
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE_ACTION="${BATCH_SIZE_ACTION:-32}"
BATCH_SIZE_KEYSTEP="${BATCH_SIZE_KEYSTEP:-16}"
BATCH_SIZE_ABSTRACT="${BATCH_SIZE_ABSTRACT:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.02}"

# ==============================================================================
# Hierarchical schedule (PeskaVLP defaults)
# ==============================================================================
ACTION_EVERY="${ACTION_EVERY:-1}"
KEYSTEP_EVERY="${KEYSTEP_EVERY:-3}"
ABSTRACT_EVERY="${ABSTRACT_EVERY:-5}"

# ==============================================================================
# Loss & augmentation
# ==============================================================================
TEMPERATURE="${TEMPERATURE:-0.1}"
MULTI_TEXT="${MULTI_TEXT:-4}"
AUG_SCALE_MIN="${AUG_SCALE_MIN:-0.2}"

# ==============================================================================
# Evaluation
# ==============================================================================
EVAL_EVERY="${EVAL_EVERY:-0}"
EVAL_DATASETS="${EVAL_DATASETS:-cholec80_phase}"

# ==============================================================================
# Logging
# ==============================================================================
OUTPUT_DIR="${OUTPUT_DIR:-outputs/$RUN_NAME}"
RESUME="${RESUME:-}"

mkdir -p "$OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"

cmd=(
    torchrun
    --standalone
    --nproc_per_node="$NPROC"
    train_hierarchical.py
    --epochs "$EPOCHS"
    --embed_dim "$EMBED_DIM"
    --image_size "$IMAGE_SIZE"
    --max_length "$MAX_LENGTH"
    --num_frames "$NUM_FRAMES"
    --text_model_name "$TEXT_MODEL_NAME"
    --vision_pretrained_weights "$VISION_PRETRAINED_WEIGHTS"
    --main_csv_path "$MAIN_CSV_PATH"
    --video_root_folder "$VIDEO_ROOT_FOLDER"
    --annotations_root "$ANNOTATIONS_ROOT"
    --batch_size_action "$BATCH_SIZE_ACTION"
    --batch_size_keystep "$BATCH_SIZE_KEYSTEP"
    --batch_size_abstract "$BATCH_SIZE_ABSTRACT"
    --accum_steps "$ACCUM_STEPS"
    --num_workers "$NUM_WORKERS"
    --learning_rate "$LEARNING_RATE"
    --weight_decay "$WEIGHT_DECAY"
    --action_every "$ACTION_EVERY"
    --keystep_every "$KEYSTEP_EVERY"
    --abstract_every "$ABSTRACT_EVERY"
    --temperature "$TEMPERATURE"
    --multi_text "$MULTI_TEXT"
    --aug_scale_min "$AUG_SCALE_MIN"
    --eval_every "$EVAL_EVERY"
    --eval_datasets "$EVAL_DATASETS"
    --output_dir "$OUTPUT_DIR"
)

if [[ -n "$RESUME" ]]; then
    cmd+=(--resume "$RESUME")
fi

echo "Launching: ${cmd[*]}"
"${cmd[@]}"
