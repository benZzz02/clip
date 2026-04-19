#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV:-vllm}" ]]; then
  conda activate "${CONDA_ENV:-vllm}"
fi
set -u

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

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a CUDA_DEVICE_LIST <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT="${#CUDA_DEVICE_LIST[@]}"
NPROC="${NPROC:-$GPU_COUNT}"

if [[ "$NPROC" -ne "$GPU_COUNT" ]]; then
  echo "NPROC ($NPROC) must match the number of GPUs in CUDA_VISIBLE_DEVICES ($GPU_COUNT)." >&2
  exit 1
fi

EXP_NAME="${EXP_NAME:-same_video_triplet_8f_run1}"

PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-128}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
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
ASSUME_RESIZED_VIDEO="${ASSUME_RESIZED_VIDEO:-true}"
USE_SWANLAB="${USE_SWANLAB:-true}"

TEXT_MODEL_NAME="${TEXT_MODEL_NAME:-marcobombieri/surgicberta}"
VISION_PRETRAINED_WEIGHTS="$(pick_first_existing file \
  "${VISION_PRETRAINED_WEIGHTS:-}" \
  "$SCRIPT_DIR/lemonfm.pth" \
  "/data/nfs_data/CLIP/lemonfm.pth" \
  "/mnt/mydisk/CLIP/lemonfm.pth")"
VIDEO_ROOT_FOLDER="$(pick_first_existing dir \
  "${VIDEO_ROOT_FOLDER:-}" \
  "$SCRIPT_DIR/downloaded_video_224_test" \
  "/data/nfs_data/CLIP/downloaded_video_224_test" \
  "/mnt/mydisk/CLIP/downloaded_video_224_test")"
MAIN_CSV_PATH="$(pick_first_existing file \
  "${MAIN_CSV_PATH:-}" \
  "$SCRIPT_DIR/surglavi_level_csv/all_video.csv" \
  "/data/nfs_data/CLIP/surglavi_level_csv/all_video.csv" \
  "/mnt/mydisk/CLIP/surglavi_level_csv/all_video.csv")"
ANNOTATIONS_ROOT="$(pick_first_existing dir \
  "${ANNOTATIONS_ROOT:-}" \
  "$SCRIPT_DIR/surglavi_level_csv" \
  "/data/nfs_data/CLIP/surglavi_level_csv" \
  "/mnt/mydisk/CLIP/surglavi_level_csv")"

ANNOTATION_LEVELS="${ANNOTATION_LEVELS:-coarse,mid,fine}"
LEVEL_MIX="${LEVEL_MIX:-concat}"
LEVEL_BATCH_SIZES="${LEVEL_BATCH_SIZES:-fine:80,mid:32,coarse:16}"

SAMPLES_CACHE_DIR="${SAMPLES_CACHE_DIR:-$SCRIPT_DIR/.cache/pretrain_samples}"
USE_SAMPLES_CACHE="${USE_SAMPLES_CACHE:-true}"
REBUILD_SAMPLES_CACHE="${REBUILD_SAMPLES_CACHE:-false}"
SAMPLES_CACHE_VERSION="${SAMPLES_CACHE_VERSION:-v1}"

LOCAL_TEMPERATURE="${LOCAL_TEMPERATURE:-0.15}"
LEVEL_FRAME_TEMPERATURES="${LEVEL_FRAME_TEMPERATURES:-0.35,0.8,1.6}"
TRAIN_WINDOW_EXPAND_RATIO="${TRAIN_WINDOW_EXPAND_RATIO:-1.5}"
SELECTION_LOSS_WEIGHT="${SELECTION_LOSS_WEIGHT:-0.7}"

RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
SAVE_ROOT="${SAVE_ROOT:-/data/nfs_data/CLIP/outputs}"
SAVE_PREFIX="${SAVE_PREFIX:-${SAVE_ROOT}/${EXP_NAME}/}"
TB_LOGDIR="${TB_LOGDIR:-runs/${EXP_NAME}}"
SWANLAB_LOGDIR="${SWANLAB_LOGDIR:-swanlog/${EXP_NAME}}"

if [[ -z "$VISION_PRETRAINED_WEIGHTS" ]]; then
  echo "Could not locate lemonfm.pth. Set VISION_PRETRAINED_WEIGHTS explicitly." >&2
  exit 1
fi

if [[ -z "$VIDEO_ROOT_FOLDER" || -z "$MAIN_CSV_PATH" || -z "$ANNOTATIONS_ROOT" ]]; then
  echo "Could not resolve one or more data paths. Set VIDEO_ROOT_FOLDER / MAIN_CSV_PATH / ANNOTATIONS_ROOT explicitly." >&2
  exit 1
fi

mkdir -p "$SAVE_PREFIX" "$TB_LOGDIR" "$SWANLAB_LOGDIR"

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
export SWANLAB_EXPERIMENT_NAME="$EXP_NAME"
export SAVE_PREFIX
export TB_LOGDIR
export SWANLAB_LOGDIR

echo "EXP_NAME=$EXP_NAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NPROC=$NPROC"
echo "VISION_PRETRAINED_WEIGHTS=$VISION_PRETRAINED_WEIGHTS"
echo "VIDEO_ROOT_FOLDER=$VIDEO_ROOT_FOLDER"
echo "MAIN_CSV_PATH=$MAIN_CSV_PATH"
echo "ANNOTATIONS_ROOT=$ANNOTATIONS_ROOT"
echo "SAVE_PREFIX=$SAVE_PREFIX"

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
  --level_frame_temperatures "$LEVEL_FRAME_TEMPERATURES"
  --train_window_expand_ratio "$TRAIN_WINDOW_EXPAND_RATIO"
  --selection_loss_weight "$SELECTION_LOSS_WEIGHT"
)

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  cmd+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

"${cmd[@]}"
