#!/usr/bin/env bash
set -euo pipefail

set +u
source ~/miniconda3/etc/profile.d/conda.sh
if [[ "${CONDA_DEFAULT_ENV:-}" != "vllm" ]]; then
    conda activate vllm
fi
set -u

NPROC=3
EXP_NAME="surglavi_lora_3gpu_bs100"

PER_GPU_BATCH_SIZE=32
ACCUM_STEPS=1
NUM_WORKERS=8
NUM_FRAMES=8

EPOCHS=50
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.02
ADAM_BETA1=0.9
ADAM_BETA2=0.999

IMAGE_SIZE=224
MAX_LENGTH=256
SEED=42

TOKENIZER_NAME="bert-base-uncased"
SURGCLIP_MODEL_NAME="SurgCLIP-B"

VIDEO_ROOT_FOLDER="/data/nfs_data/CLIP/downloaded_video_224_test"
ASSUME_RESIZED_VIDEO=1
MAIN_CSV_PATH="/data/nfs_data/CLIP/surglavi_level_csv/all_video.csv"
ANNOTATIONS_ROOT="/data/nfs_data/CLIP/surglavi_level_csv"
ANNOTATION_LEVELS="coarse,mid,fine"
LEVEL_MIX="concat"
SAMPLE_MODE="center"
LEVEL_BATCH_SIZES="fine:16,mid:10,coarse:6"
SAMPLES_CACHE_DIR="/data/nfs_data/CLIP/.cache/pretrain_samples"
USE_SAMPLES_CACHE=true
REBUILD_SAMPLES_CACHE=true
SAMPLES_CACHE_VERSION="nfs_v1"

FINETUNE_MODE="lora"
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGETS="text_encoder.encoder.layer.,vision_encoder.model.blocks."
GRADIENT_CHECKPOINTING=1
LOCAL_TEMPERATURE=0.15
LEVEL_FRAME_TEMPERATURES="0.35,0.8,1.6"
TRAIN_WINDOW_EXPAND_RATIO=1.5
SELECTION_LOSS_WEIGHT=0.5

SAVE_DIR="/data/surglavi_checkpoint/${EXP_NAME}"
SAVE_EVERY=5
SAVE_NAME="final.pt"
TB_LOGDIR="runs/${EXP_NAME}"
RESUME_FROM_CHECKPOINT=""

export CUDA_VISIBLE_DEVICES=0,1,2
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export SWANLAB_EXPERIMENT_NAME="$EXP_NAME"
export TB_LOGDIR
export PYTHONPATH="$(pwd)"

mkdir -p "$SAVE_DIR" "$TB_LOGDIR"

torchrun --standalone --nproc_per_node="$NPROC" train_surglavi_ddp.py \
    --epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --adam_beta1 "$ADAM_BETA1" \
    --adam_beta2 "$ADAM_BETA2" \
    --per_gpu_batch_size "$PER_GPU_BATCH_SIZE" \
    --accum_steps "$ACCUM_STEPS" \
    --num_workers "$NUM_WORKERS" \
    --image_size "$IMAGE_SIZE" \
    --max_length "$MAX_LENGTH" \
    --num_frames "$NUM_FRAMES" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --surgclip_model_name "$SURGCLIP_MODEL_NAME" \
    --video_root_folder "$VIDEO_ROOT_FOLDER" \
    --assume_resized_video "$ASSUME_RESIZED_VIDEO" \
    --main_csv_path "$MAIN_CSV_PATH" \
    --annotations_root "$ANNOTATIONS_ROOT" \
    --annotation_levels "$ANNOTATION_LEVELS" \
    --level_mix "$LEVEL_MIX" \
    --level_batch_sizes "$LEVEL_BATCH_SIZES" \
    --sample_mode "$SAMPLE_MODE" \
    --samples_cache_dir "$SAMPLES_CACHE_DIR" \
    --use_samples_cache "$USE_SAMPLES_CACHE" \
    --rebuild_samples_cache "$REBUILD_SAMPLES_CACHE" \
    --samples_cache_version "$SAMPLES_CACHE_VERSION" \
    --save_dir "$SAVE_DIR" \
    --save_every "$SAVE_EVERY" \
    --save_name "$SAVE_NAME" \
    --seed "$SEED" \
    --finetune_mode "$FINETUNE_MODE" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --lora_targets "$LORA_TARGETS" \
    --local_temperature "$LOCAL_TEMPERATURE" \
    --level_frame_temperatures "$LEVEL_FRAME_TEMPERATURES" \
    --train_window_expand_ratio "$TRAIN_WINDOW_EXPAND_RATIO" \
    --selection_loss_weight "$SELECTION_LOSS_WEIGHT" \
    ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"}
