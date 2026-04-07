#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

NPROC=2
EXP_NAME="htg_8f_v1"

PER_GPU_BATCH_SIZE=128
ACCUM_STEPS=1
NUM_WORKERS=8
NUM_FRAMES=8

EPOCHS=50
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.02
ADAM_BETA1=0.9
ADAM_BETA2=0.999

EMBED_DIM=256
IMAGE_SIZE=224
MAX_LENGTH=256

FFMPEG_TIMEOUT=10
MAX_RETRY=5
ASSUME_RESIZED_VIDEO=true
USE_SWANLAB=true

TEXT_MODEL_NAME="marcobombieri/surgicberta"
VISION_PRETRAINED_WEIGHTS="/mnt/mydisk/CLIP/lemonfm.pth"
VIDEO_ROOT_FOLDER="/mnt/mydisk/CLIP/downloaded_video_224_test"
MAIN_CSV_PATH="/mnt/mydisk/CLIP/surglavi_level_csv/all_video.csv"

ANNOTATIONS_ROOT="/mnt/mydisk/CLIP/surglavi_level_csv"
ANNOTATION_LEVELS="coarse,mid,fine"
LEVEL_MIX="concat"
LEVEL_BATCH_SIZES="fine:80,mid:32,coarse:16"

SAMPLES_CACHE_DIR="/mnt/mydisk/CLIP/.cache/pretrain_samples"
USE_SAMPLES_CACHE=true
REBUILD_SAMPLES_CACHE=false
SAMPLES_CACHE_VERSION="v1"

LOCAL_TEMPERATURE=0.15
LEVEL_FRAME_TEMPERATURES="0.35,0.8,1.6"
TRAIN_WINDOW_EXPAND_RATIO=1.5
SELECTION_LOSS_WEIGHT=0.7

ENABLE_HTG=true
FINE_ANNOTATIONS_DIR="/mnt/mydisk/CLIP/surglavi_level_csv/fine"
HTG_LOSS_WEIGHT=0.1

export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export SWANLAB_EXPERIMENT_NAME="$EXP_NAME"
export SAVE_PREFIX="outputs/htg_8f_run1/"

mkdir -p "$SAVE_PREFIX"

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
    --enable_htg "$ENABLE_HTG" \
    --fine_annotations_dir "$FINE_ANNOTATIONS_DIR" \
    --htg_loss_weight "$HTG_LOSS_WEIGHT"