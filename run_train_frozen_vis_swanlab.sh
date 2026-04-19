#!/usr/bin/env bash
set -euo pipefail

set +u
source ~/miniconda3/etc/profile.d/conda.sh
if [[ "${CONDA_DEFAULT_ENV:-}" != "vllm" ]]; then
  conda activate vllm
fi
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pick_first_existing() {
  local kind="$1"
  shift
  local candidate
  for candidate in "$@"; do
    [[ -z "$candidate" ]] && continue
    if [[ "$kind" == "file" && -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
    if [[ "$kind" == "dir" && -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

# Experiment
EXP_NAME="${EXP_NAME:-same_video_triplet_8f}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a CUDA_DEVICE_LIST <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT="${#CUDA_DEVICE_LIST[@]}"
NPROC="${NPROC:-$GPU_COUNT}"

if [[ "$NPROC" -ne "$GPU_COUNT" ]]; then
  echo "NPROC ($NPROC) must match the number of GPUs in CUDA_VISIBLE_DEVICES ($GPU_COUNT)." >&2
  exit 1
fi

# Optimization
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-128}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-6}"
EPOCHS="${EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.02}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"

# Model
EMBED_DIM="${EMBED_DIM:-256}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
MAX_LENGTH="${MAX_LENGTH:-256}"
NUM_FRAMES="${NUM_FRAMES:-8}"
TEXT_MODEL_NAME="${TEXT_MODEL_NAME:-marcobombieri/surgicberta}"
VISION_PRETRAINED_WEIGHTS="$(pick_first_existing file \
  "${VISION_PRETRAINED_WEIGHTS:-}" \
  "$REPO_ROOT/lemonfm.pth" \
  "/data/nfs_data/CLIP/lemonfm.pth" \
  "/mnt/mydisk/CLIP/lemonfm.pth")"

# Data
VIDEO_ROOT_FOLDER="$(pick_first_existing dir \
  "${VIDEO_ROOT_FOLDER:-}" \
  "$REPO_ROOT/downloaded_video_224_test" \
  "/data/nfs_data/CLIP/downloaded_video_224_test" \
  "/mnt/mydisk/CLIP/downloaded_video_224_test")"
MAIN_CSV_PATH="$(pick_first_existing file \
  "${MAIN_CSV_PATH:-}" \
  "$REPO_ROOT/surglavi_level_csv/all_video.csv" \
  "/data/nfs_data/CLIP/surglavi_level_csv/all_video.csv" \
  "/mnt/mydisk/CLIP/surglavi_level_csv/all_video.csv")"
ANNOTATIONS_ROOT="$(pick_first_existing dir \
  "${ANNOTATIONS_ROOT:-}" \
  "$REPO_ROOT/surglavi_level_csv" \
  "/data/nfs_data/CLIP/surglavi_level_csv" \
  "/mnt/mydisk/CLIP/surglavi_level_csv")"
ANNOTATION_LEVELS="${ANNOTATION_LEVELS:-coarse,mid,fine}"
LEVEL_MIX="${LEVEL_MIX:-concat}"
LEVEL_BATCH_SIZES="${LEVEL_BATCH_SIZES:-fine:80,mid:32,coarse:16}"

# Video loading
FFMPEG_TIMEOUT="${FFMPEG_TIMEOUT:-10}"
MAX_RETRY="${MAX_RETRY:-5}"
ASSUME_RESIZED_VIDEO="${ASSUME_RESIZED_VIDEO:-true}"

# Sample cache
SAMPLES_CACHE_DIR="${SAMPLES_CACHE_DIR:-$REPO_ROOT/.cache/pretrain_samples}"
USE_SAMPLES_CACHE="${USE_SAMPLES_CACHE:-true}"
REBUILD_SAMPLES_CACHE="${REBUILD_SAMPLES_CACHE:-false}"
SAMPLES_CACHE_VERSION="${SAMPLES_CACHE_VERSION:-v1}"

# Train-time window denoising
LOCAL_TEMPERATURE="${LOCAL_TEMPERATURE:-0.15}"
LEVEL_FRAME_TEMPERATURES="${LEVEL_FRAME_TEMPERATURES:-0.35,0.8,1.6}"
TRAIN_WINDOW_EXPAND_RATIO="${TRAIN_WINDOW_EXPAND_RATIO:-1.5}"
SELECTION_LOSS_WEIGHT="${SELECTION_LOSS_WEIGHT:-0.7}"

# Logging and outputs
USE_SWANLAB="${USE_SWANLAB:-true}"
SAVE_ROOT="${SAVE_ROOT:-/data/nfs_data/CLIP/outputs}"
SAVE_PREFIX="${SAVE_PREFIX:-${SAVE_ROOT}/${EXP_NAME}/}"
TB_LOGDIR="${TB_LOGDIR:-runs/${EXP_NAME}}"
SWANLAB_LOGDIR="${SWANLAB_LOGDIR:-swanlog/${EXP_NAME}}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

mkdir -p "$SAVE_PREFIX" "$TB_LOGDIR" "$SWANLAB_LOGDIR"

export CUDA_VISIBLE_DEVICES
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
export SWANLAB_EXPERIMENT_NAME="$EXP_NAME"
export SAVE_PREFIX
export TB_LOGDIR
export SWANLAB_LOGDIR

echo "EXP_NAME=$EXP_NAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NPROC=$NPROC"
echo "SAVE_PREFIX=$SAVE_PREFIX"
echo "TRAIN_WINDOW_EXPAND_RATIO=$TRAIN_WINDOW_EXPAND_RATIO"

if [[ -z "$VISION_PRETRAINED_WEIGHTS" ]]; then
  echo "Could not locate lemonfm.pth. Set VISION_PRETRAINED_WEIGHTS explicitly." >&2
  exit 1
fi

if [[ -z "$VIDEO_ROOT_FOLDER" || -z "$MAIN_CSV_PATH" || -z "$ANNOTATIONS_ROOT" ]]; then
  echo "Could not resolve one or more data paths. Set VIDEO_ROOT_FOLDER / MAIN_CSV_PATH / ANNOTATIONS_ROOT explicitly." >&2
  exit 1
fi

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  torchrun --standalone --nproc_per_node="$NPROC" train_frozen_vis.py \
      --epochs "$EPOCHS" \
      --learning_rate "$LEARNING_RATE" \
      --weight_decay "$WEIGHT_DECAY" \
      --adam_beta1 "$ADAM_BETA1" \
      --adam_beta2 "$ADAM_BETA2" \
      --per_gpu_batch_size "$PER_GPU_BATCH_SIZE" \
      --accum_steps "$ACCUM_STEPS" \
      --num_workers "$NUM_WORKERS" \
      --embed_dim "$EMBED_DIM" \
      --image_size "$IMAGE_SIZE" \
      --max_length "$MAX_LENGTH" \
      --num_frames "$NUM_FRAMES" \
      --text_model_name "$TEXT_MODEL_NAME" \
      --vision_pretrained_weights "$VISION_PRETRAINED_WEIGHTS" \
      --video_root_folder "$VIDEO_ROOT_FOLDER" \
      --ffmpeg_timeout "$FFMPEG_TIMEOUT" \
      --max_retry "$MAX_RETRY" \
      --assume_resized_video "$ASSUME_RESIZED_VIDEO" \
      --main_csv_path "$MAIN_CSV_PATH" \
      --annotations_root "$ANNOTATIONS_ROOT" \
      --annotation_levels "$ANNOTATION_LEVELS" \
      --level_mix "$LEVEL_MIX" \
      --level_batch_sizes "$LEVEL_BATCH_SIZES" \
      --samples_cache_dir "$SAMPLES_CACHE_DIR" \
      --use_samples_cache "$USE_SAMPLES_CACHE" \
      --rebuild_samples_cache "$REBUILD_SAMPLES_CACHE" \
      --samples_cache_version "$SAMPLES_CACHE_VERSION" \
      --use_swanlab "$USE_SWANLAB" \
      --local_temperature "$LOCAL_TEMPERATURE" \
      --level_frame_temperatures "$LEVEL_FRAME_TEMPERATURES" \
      --train_window_expand_ratio "$TRAIN_WINDOW_EXPAND_RATIO" \
      --selection_loss_weight "$SELECTION_LOSS_WEIGHT" \
      --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"
else
  torchrun --standalone --nproc_per_node="$NPROC" train_frozen_vis.py \
      --epochs "$EPOCHS" \
      --learning_rate "$LEARNING_RATE" \
      --weight_decay "$WEIGHT_DECAY" \
      --adam_beta1 "$ADAM_BETA1" \
      --adam_beta2 "$ADAM_BETA2" \
      --per_gpu_batch_size "$PER_GPU_BATCH_SIZE" \
      --accum_steps "$ACCUM_STEPS" \
      --num_workers "$NUM_WORKERS" \
      --embed_dim "$EMBED_DIM" \
      --image_size "$IMAGE_SIZE" \
      --max_length "$MAX_LENGTH" \
      --num_frames "$NUM_FRAMES" \
      --text_model_name "$TEXT_MODEL_NAME" \
      --vision_pretrained_weights "$VISION_PRETRAINED_WEIGHTS" \
      --video_root_folder "$VIDEO_ROOT_FOLDER" \
      --ffmpeg_timeout "$FFMPEG_TIMEOUT" \
      --max_retry "$MAX_RETRY" \
      --assume_resized_video "$ASSUME_RESIZED_VIDEO" \
      --main_csv_path "$MAIN_CSV_PATH" \
      --annotations_root "$ANNOTATIONS_ROOT" \
      --annotation_levels "$ANNOTATION_LEVELS" \
      --level_mix "$LEVEL_MIX" \
      --level_batch_sizes "$LEVEL_BATCH_SIZES" \
      --samples_cache_dir "$SAMPLES_CACHE_DIR" \
      --use_samples_cache "$USE_SAMPLES_CACHE" \
      --rebuild_samples_cache "$REBUILD_SAMPLES_CACHE" \
      --samples_cache_version "$SAMPLES_CACHE_VERSION" \
      --use_swanlab "$USE_SWANLAB" \
      --local_temperature "$LOCAL_TEMPERATURE" \
      --level_frame_temperatures "$LEVEL_FRAME_TEMPERATURES" \
      --train_window_expand_ratio "$TRAIN_WINDOW_EXPAND_RATIO" \
      --selection_loss_weight "$SELECTION_LOSS_WEIGHT"
fi
