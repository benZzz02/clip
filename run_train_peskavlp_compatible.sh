#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh

NPROC="${NPROC:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,3}"
RUN_NAME="${RUN_NAME:-same_video_triplet_peskavlp_compatible_8f_run1}"
EXP_NAME="${EXP_NAME:-$RUN_NAME}"

PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-32}"
ACTION_BATCH_SIZE="${ACTION_BATCH_SIZE:-$PER_GPU_BATCH_SIZE}"
MID_BATCH_SIZE="${MID_BATCH_SIZE:-$PER_GPU_BATCH_SIZE}"
COARSE_BATCH_SIZE="${COARSE_BATCH_SIZE:-$PER_GPU_BATCH_SIZE}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
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
VIDEO_ROOT_FOLDER="${VIDEO_ROOT_FOLDER:-downloaded_video_224_test}"
MAIN_CSV_PATH="${MAIN_CSV_PATH:-surglavi_level_csv/all_video.csv}"

ANNOTATIONS_ROOT="${ANNOTATIONS_ROOT:-surglavi_level_csv}"
ANNOTATION_LEVELS="${ANNOTATION_LEVELS:-coarse,mid,fine}"
LEVEL_MIX="${LEVEL_MIX:-concat}"

SAMPLES_CACHE_DIR="${SAMPLES_CACHE_DIR:-.cache/pretrain_samples}"
USE_SAMPLES_CACHE="${USE_SAMPLES_CACHE:-true}"
REBUILD_SAMPLES_CACHE="${REBUILD_SAMPLES_CACHE:-true}"
SAMPLES_CACHE_VERSION="${SAMPLES_CACHE_VERSION:-v1}"

MAX_CANDIDATES="${MAX_CANDIDATES:-8}"
MID_INTERVAL="${MID_INTERVAL:-3}"
COARSE_INTERVAL="${COARSE_INTERVAL:-5}"
ACTION_VIEW_FLIP_PROB="${ACTION_VIEW_FLIP_PROB:-0.5}"
ACTION_VIEW_NOISE_PROB="${ACTION_VIEW_NOISE_PROB:-0.25}"
ACTION_VIEW_NOISE_STD="${ACTION_VIEW_NOISE_STD:-0.01}"

LOCAL_TEMPERATURE="${LOCAL_TEMPERATURE:-0.07}"
SELECTION_POOLING="${SELECTION_POOLING:-similarity}"
LEVEL_FRAME_TEMPERATURES="${LEVEL_FRAME_TEMPERATURES:-0.6,0.9,1.2}"

PESKAVLP_TEMPERATURE="${PESKAVLP_TEMPERATURE:-0.1}"
PESKAVLP_ALPHA_WEIGHT="${PESKAVLP_ALPHA_WEIGHT:-0.75}"
PESKAVLP_DTW_BETA="${PESKAVLP_DTW_BETA:-0}"
PESKAVLP_DTW_RATIO="${PESKAVLP_DTW_RATIO:-0.5}"
PESKAVLP_DTW_SCALE_FACTOR="${PESKAVLP_DTW_SCALE_FACTOR:-0.01}"

ENCODER_LORA_RANK="${ENCODER_LORA_RANK:-8}"
ENCODER_LORA_ALPHA="${ENCODER_LORA_ALPHA:-16}"
ENCODER_LORA_DROPOUT="${ENCODER_LORA_DROPOUT:-0.05}"
ENCODER_LORA_TARGETS="${ENCODER_LORA_TARGETS:-visual,text}"

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
    train_peskavlp_compatible.py
    --epochs "$EPOCHS"
    --learning_rate "$LEARNING_RATE"
    --weight_decay "$WEIGHT_DECAY"
    --adam_beta1 "$ADAM_BETA1"
    --adam_beta2 "$ADAM_BETA2"
    --per_gpu_batch_size "$PER_GPU_BATCH_SIZE"
    --action_batch_size "$ACTION_BATCH_SIZE"
    --mid_batch_size "$MID_BATCH_SIZE"
    --coarse_batch_size "$COARSE_BATCH_SIZE"
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
    --samples_cache_dir "$SAMPLES_CACHE_DIR"
    --use_samples_cache "$USE_SAMPLES_CACHE"
    --rebuild_samples_cache "$REBUILD_SAMPLES_CACHE"
    --samples_cache_version "$SAMPLES_CACHE_VERSION"
    --max_candidates "$MAX_CANDIDATES"
    --mid_interval "$MID_INTERVAL"
    --coarse_interval "$COARSE_INTERVAL"
    --action_view_flip_prob "$ACTION_VIEW_FLIP_PROB"
    --action_view_noise_prob "$ACTION_VIEW_NOISE_PROB"
    --action_view_noise_std "$ACTION_VIEW_NOISE_STD"
    --local_temperature "$LOCAL_TEMPERATURE"
    --selection_pooling "$SELECTION_POOLING"
    --level_frame_temperatures "$LEVEL_FRAME_TEMPERATURES"
    --use_swanlab "$USE_SWANLAB"
    --peskavlp_temperature "$PESKAVLP_TEMPERATURE"
    --peskavlp_alpha_weight "$PESKAVLP_ALPHA_WEIGHT"
    --peskavlp_dtw_beta "$PESKAVLP_DTW_BETA"
    --peskavlp_dtw_ratio "$PESKAVLP_DTW_RATIO"
    --peskavlp_dtw_scale_factor "$PESKAVLP_DTW_SCALE_FACTOR"
    --encoder_lora_rank "$ENCODER_LORA_RANK"
    --encoder_lora_alpha "$ENCODER_LORA_ALPHA"
    --encoder_lora_dropout "$ENCODER_LORA_DROPOUT"
    --encoder_lora_targets "$ENCODER_LORA_TARGETS"
)

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    cmd+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

"${cmd[@]}"
