#!/usr/bin/env bash
set -euo pipefail

NPROC="${NPROC:-3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
RUN_NAME="${RUN_NAME:-surglavi_8frame_run1}"
EXP_NAME="${EXP_NAME:-$RUN_NAME}"

PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-100}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_FRAMES="${NUM_FRAMES:-16}"

EPOCHS="${EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.02}"

FFMPEG_TIMEOUT="${FFMPEG_TIMEOUT:-10}"
MAX_RETRY="${MAX_RETRY:-5}"
VIDEO_READER_THREADS="${VIDEO_READER_THREADS:-1}"
VIDEO_READER_CACHE_SIZE="${VIDEO_READER_CACHE_SIZE:-1}"
ASSUME_RESIZED_VIDEO="${ASSUME_RESIZED_VIDEO:-1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-/data/surglavi_checkpoint/surglavi_8frame_run1/surglavi_epoch_45.pt}"
SAVE_DIR="${SAVE_DIR:-/data/surglavi_checkpoint/$RUN_NAME}"

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
export SWANLAB_EXPERIMENT_NAME="$EXP_NAME"

cmd=(
    torchrun
    --standalone
    --nproc_per_node="$NPROC"
    train_surglavi_ddp.py
    --epochs "$EPOCHS"
    --learning_rate "$LEARNING_RATE"
    --weight_decay "$WEIGHT_DECAY"
    --adam_beta1 0.9
    --adam_beta2 0.999
    --per_gpu_batch_size "$PER_GPU_BATCH_SIZE"
    --accum_steps "$ACCUM_STEPS"
    --num_workers "$NUM_WORKERS"
    --image_size 224
    --max_length 256
    --num_frames "$NUM_FRAMES"
    --tokenizer_name "bert-base-uncased"
    --surgclip_model_name "SurgCLIP-B"
    --video_root_folder "downloaded_video_224_test"
    --assume_resized_video "$ASSUME_RESIZED_VIDEO"
    --main_csv_path "surglavi_level_csv/all_video.csv"
    --annotations_root "surglavi_level_csv"
    --annotation_levels "coarse,mid,fine"
    --level_mix "concat"
    --sample_mode "center"
    --normalization "surgclip"
    --ffmpeg_timeout "$FFMPEG_TIMEOUT"
    --max_retry "$MAX_RETRY"
    --video_reader_threads "$VIDEO_READER_THREADS"
    --video_reader_cache_size "$VIDEO_READER_CACHE_SIZE"
    --save_dir "$SAVE_DIR"
    --save_every 5
    --save_name "final.pt"
)

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    cmd+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

mkdir -p "$SAVE_DIR"
"${cmd[@]}"
