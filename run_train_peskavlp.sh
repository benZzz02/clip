#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh


NPROC="${NPROC:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
RUN_NAME="${RUN_NAME:-same_video_triplet_peskavlp_8f_run1}"
EXP_NAME="${EXP_NAME:-$RUN_NAME}"

PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-128}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-6}"
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
VIDEO_READER_THREADS="${VIDEO_READER_THREADS:-1}"
VIDEO_READER_CACHE_SIZE="${VIDEO_READER_CACHE_SIZE:-1}"
ASSUME_RESIZED_VIDEO="${ASSUME_RESIZED_VIDEO:-true}"
USE_SWANLAB="${USE_SWANLAB:-true}"

TEXT_MODEL_NAME="${TEXT_MODEL_NAME:-marcobombieri/surgicberta}"
VISION_BACKBONE="${VISION_BACKBONE:-convnext_lemonfm}"
VISION_PRETRAINED_WEIGHTS="${VISION_PRETRAINED_WEIGHTS:-lemonfm.pth}"
VIDEO_ROOT_FOLDER="${VIDEO_ROOT_FOLDER:-/data/surglavi_video/downloaded_video_224_test}"
MAIN_CSV_PATH="${MAIN_CSV_PATH:-surglavi_level_csv/all_video.csv}"

ANNOTATIONS_ROOT="${ANNOTATIONS_ROOT:-surglavi_level_csv}"
ANNOTATION_LEVELS="${ANNOTATION_LEVELS:-coarse,mid,fine}"
LEVEL_MIX="${LEVEL_MIX:-concat}"
LEVEL_BATCH_SIZES="${LEVEL_BATCH_SIZES:-fine:70,mid:36,coarse:22}"

SAMPLES_CACHE_DIR="${SAMPLES_CACHE_DIR:-.cache/pretrain_samples}"
USE_SAMPLES_CACHE="${USE_SAMPLES_CACHE:-true}"
REBUILD_SAMPLES_CACHE="${REBUILD_SAMPLES_CACHE:-true}"
SAMPLES_CACHE_VERSION="${SAMPLES_CACHE_VERSION:-v1}"

PESKAVLP_TEMPERATURE="${PESKAVLP_TEMPERATURE:-0.1}"
PESKAVLP_ALPHA_WEIGHT="${PESKAVLP_ALPHA_WEIGHT:-0.75}"
PESKAVLP_DTW_BETA="${PESKAVLP_DTW_BETA:-0}"
PESKAVLP_DTW_RATIO="${PESKAVLP_DTW_RATIO:-0.5}"
PESKAVLP_DTW_SCALE_FACTOR="${PESKAVLP_DTW_SCALE_FACTOR:-0.01}"
PESKAVLP_MAX_CANDIDATES="${PESKAVLP_MAX_CANDIDATES:-8}"

ENCODER_LORA_RANK="${ENCODER_LORA_RANK:-8}"
ENCODER_LORA_ALPHA="${ENCODER_LORA_ALPHA:-16}"
ENCODER_LORA_DROPOUT="${ENCODER_LORA_DROPOUT:-0.05}"
ENCODER_LORA_TARGETS="${ENCODER_LORA_TARGETS:-visual,text}"
TRAIN_ENCODER_BASE_LAYERS="${TRAIN_ENCODER_BASE_LAYERS:-false}"

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
export SWANLAB_EXPERIMENT_NAME="$EXP_NAME"
export SAVE_PREFIX

cmd=(
    torchrun
    --standalone
    --nproc_per_node="$NPROC"
    train_frozen_vis.py
    --training_method peskavlp
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
    --vision_backbone "$VISION_BACKBONE"
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
    --peskavlp_temperature "$PESKAVLP_TEMPERATURE"
    --peskavlp_alpha_weight "$PESKAVLP_ALPHA_WEIGHT"
    --peskavlp_dtw_beta "$PESKAVLP_DTW_BETA"
    --peskavlp_dtw_ratio "$PESKAVLP_DTW_RATIO"
    --peskavlp_dtw_scale_factor "$PESKAVLP_DTW_SCALE_FACTOR"
    --peskavlp_max_candidates "$PESKAVLP_MAX_CANDIDATES"
    --encoder_lora_rank "$ENCODER_LORA_RANK"
    --encoder_lora_alpha "$ENCODER_LORA_ALPHA"
    --encoder_lora_dropout "$ENCODER_LORA_DROPOUT"
    --encoder_lora_targets "$ENCODER_LORA_TARGETS"
    --train_encoder_base_layers "$TRAIN_ENCODER_BASE_LAYERS"
)

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    cmd+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

"${cmd[@]}"
