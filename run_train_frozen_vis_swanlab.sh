#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-vllm}"

# Coarse-only multi-query latent xpool preset:
# stay close to the previous best xpool launcher, and only let coarse
# samples use extra latent text queries.
RUN_NAME="${RUN_NAME:-latent_xpool_coarse_only_114_sw07_8f_run1}"
SELECTION_POOLING="${SELECTION_POOLING:-latent_xpool}"
LEVEL_TEXT_QUERY_COUNTS="${LEVEL_TEXT_QUERY_COUNTS:-1,1,4}"
MAX_TEXT_QUERIES="${MAX_TEXT_QUERIES:-4}"
SELECTION_LOSS_WEIGHT="${SELECTION_LOSS_WEIGHT:-0.7}"
TRAIN_WINDOW_EXPAND_RATIO="${TRAIN_WINDOW_EXPAND_RATIO:-1.5}"

NPROC="${NPROC:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
EXP_NAME="${EXP_NAME:-$RUN_NAME}"

PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-128}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_FRAMES="${NUM_FRAMES:-8}"

EPOCHS="${EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.02}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"

EMBED_DIM="${EMBED_DIM:-256}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
MAX_LENGTH="${MAX_LENGTH:-256}"

FFMPEG_TIMEOUT="${FFMPEG_TIMEOUT:-10}"
MAX_RETRY="${MAX_RETRY:-5}"
VIDEO_READER_THREADS="${VIDEO_READER_THREADS:-6}"
VIDEO_READER_CACHE_SIZE="${VIDEO_READER_CACHE_SIZE:-64}"
ASSUME_RESIZED_VIDEO="${ASSUME_RESIZED_VIDEO:-true}"
USE_SWANLAB="${USE_SWANLAB:-true}"

TEXT_MODEL_NAME="${TEXT_MODEL_NAME:-marcobombieri/surgicberta}"
VISION_PRETRAINED_WEIGHTS="${VISION_PRETRAINED_WEIGHTS:-/mnt/mydisk/CLIP/lemonfm.pth}"
VIDEO_ROOT_FOLDER="${VIDEO_ROOT_FOLDER:-/mnt/mydisk/CLIP/downloaded_video_224_test}"
MAIN_CSV_PATH="${MAIN_CSV_PATH:-/mnt/mydisk/CLIP/surglavi_level_csv/all_video.csv}"

ANNOTATIONS_ROOT="${ANNOTATIONS_ROOT:-/mnt/mydisk/CLIP/surglavi_level_csv}"
ANNOTATION_LEVELS="${ANNOTATION_LEVELS:-coarse,mid,fine}"
LEVEL_MIX="${LEVEL_MIX:-concat}"
LEVEL_BATCH_SIZES="${LEVEL_BATCH_SIZES:-fine:80,mid:32,coarse:16}"

SAMPLES_CACHE_DIR="${SAMPLES_CACHE_DIR:-/mnt/mydisk/CLIP/.cache/pretrain_samples}"
USE_SAMPLES_CACHE="${USE_SAMPLES_CACHE:-true}"
REBUILD_SAMPLES_CACHE="${REBUILD_SAMPLES_CACHE:-false}"
SAMPLES_CACHE_VERSION="${SAMPLES_CACHE_VERSION:-v1}"

LOCAL_TEMPERATURE="${LOCAL_TEMPERATURE:-0.15}"
LEVEL_FRAME_TEMPERATURES="${LEVEL_FRAME_TEMPERATURES:-0.35,0.8,1.6}"

RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
SAVE_PREFIX="${SAVE_PREFIX:-outputs/$RUN_NAME/}"

if [[ "$SAVE_PREFIX" == */ ]]; then
    mkdir -p "$SAVE_PREFIX"
else
    mkdir -p "$(dirname "$SAVE_PREFIX")"
fi

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export SWANLAB_EXPERIMENT_NAME="$EXP_NAME"
export SAVE_PREFIX

cmd=(
    torchrun
    --standalone
    --nproc_per_node="$NPROC"
    train_frozen_vis.py
    --epochs "$EPOCHS"
    --learning_rate "$LEARNING_RATE"
    --weight_decay "$WEIGHT_DECAY"
    --adam_beta1 "$ADAM_BETA1"
    --adam_beta2 "$ADAM_BETA2"
    --per_gpu_batch_size "$PER_GPU_BATCH_SIZE"
    --accum_steps "$ACCUM_STEPS"
    --num_workers "$NUM_WORKERS"
    --embed_dim "$EMBED_DIM"
    --image_size "$IMAGE_SIZE"
    --max_length "$MAX_LENGTH"
    --num_frames "$NUM_FRAMES"
    --text_model_name "$TEXT_MODEL_NAME"
    --vision_pretrained_weights "$VISION_PRETRAINED_WEIGHTS"
    --video_root_folder "$VIDEO_ROOT_FOLDER"
    --ffmpeg_timeout "$FFMPEG_TIMEOUT"
    --max_retry "$MAX_RETRY"
    --video_reader_threads "$VIDEO_READER_THREADS"
    --video_reader_cache_size "$VIDEO_READER_CACHE_SIZE"
    --assume_resized_video "$ASSUME_RESIZED_VIDEO"
    --main_csv_path "$MAIN_CSV_PATH"
    --annotations_root "$ANNOTATIONS_ROOT"
    --annotation_levels "$ANNOTATION_LEVELS"
    --level_mix "$LEVEL_MIX"
    --level_batch_sizes "$LEVEL_BATCH_SIZES"
    --samples_cache_dir "$SAMPLES_CACHE_DIR"
    --use_samples_cache "$USE_SAMPLES_CACHE"
    --rebuild_samples_cache "$REBUILD_SAMPLES_CACHE"
    --samples_cache_version "$SAMPLES_CACHE_VERSION"
    --use_swanlab "$USE_SWANLAB"
    --local_temperature "$LOCAL_TEMPERATURE"
    --selection_pooling "$SELECTION_POOLING"
    --level_frame_temperatures "$LEVEL_FRAME_TEMPERATURES"
    --max_text_queries "$MAX_TEXT_QUERIES"
    --level_text_query_counts "$LEVEL_TEXT_QUERY_COUNTS"
    --train_window_expand_ratio "$TRAIN_WINDOW_EXPAND_RATIO"
    --selection_loss_weight "$SELECTION_LOSS_WEIGHT"
    --hierarchical_consistency_weight 0.1
)

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    cmd+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

"${cmd[@]}"
