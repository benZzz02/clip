#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-vllm}"
FALLBACK_CONDA_ENV="${FALLBACK_CONDA_ENV:-py310}"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ]]; then
    if conda info --envs | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
        conda activate "$CONDA_ENV_NAME"
    elif [[ -n "${FALLBACK_CONDA_ENV:-}" ]] && conda info --envs | awk '{print $1}' | grep -qx "$FALLBACK_CONDA_ENV"; then
        conda activate "$FALLBACK_CONDA_ENV"
    fi
fi
set -u

EXP_NAME="${EXP_NAME:-peskavlp_text_bioclinicalbert_vis_imagenet}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
IFS=',' read -r -a CUDA_DEVICE_LIST <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT="${#CUDA_DEVICE_LIST[@]}"
NPROC="${NPROC:-3}"

if [[ "$NPROC" -ne "$GPU_COUNT" ]]; then
    echo "NPROC ($NPROC) must match the number of GPUs in CUDA_VISIBLE_DEVICES ($GPU_COUNT)." >&2
    exit 1
fi

EPOCHS="${EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.02}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-100}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_FRAMES="${NUM_FRAMES:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
MAX_LENGTH="${MAX_LENGTH:-256}"
SEED="${SEED:-42}"

TOKENIZER_NAME="${TOKENIZER_NAME:-emilyalsentzer/Bio_ClinicalBERT}"
MODEL_FAMILY="${MODEL_FAMILY:-peskavlp}"
PESKAVLP_VISION_BACKBONE="${PESKAVLP_VISION_BACKBONE:-resnet_50}"
PESKAVLP_VISION_PRETRAINED="${PESKAVLP_VISION_PRETRAINED:-imagenet}"
PESKAVLP_EMBED_DIM="${PESKAVLP_EMBED_DIM:-768}"
FINETUNE_MODE="${FINETUNE_MODE:-lora}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGETS="${LORA_TARGETS:-backbone_text.model.encoder.layer.,backbone_img.global_embedder}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

VIDEO_ROOT_FOLDER="${VIDEO_ROOT_FOLDER:-/data/nfs_data/CLIP/downloaded_video_224_test}"
ASSUME_RESIZED_VIDEO="${ASSUME_RESIZED_VIDEO:-1}"
MAIN_CSV_PATH="${MAIN_CSV_PATH:-/data/nfs_data/CLIP/surglavi_level_csv/all_video.csv}"
ANNOTATIONS_ROOT="${ANNOTATIONS_ROOT:-/data/nfs_data/CLIP/surglavi_level_csv}"
ANNOTATION_LEVELS="${ANNOTATION_LEVELS:-coarse,mid,fine}"
LEVEL_MIX="${LEVEL_MIX:-concat}"
SAMPLE_MODE="${SAMPLE_MODE:-center}"
LEVEL_BATCH_SIZES="${LEVEL_BATCH_SIZES:-fine:50,mid:31,coarse:19}"
SAMPLES_CACHE_DIR="${SAMPLES_CACHE_DIR:-/data/nfs_data/CLIP/.cache/pretrain_samples}"
USE_SAMPLES_CACHE="${USE_SAMPLES_CACHE:-true}"
REBUILD_SAMPLES_CACHE="${REBUILD_SAMPLES_CACHE:-true}"
SAMPLES_CACHE_VERSION="${SAMPLES_CACHE_VERSION:-nfs_v1}"

LOCAL_TEMPERATURE="${LOCAL_TEMPERATURE:-0.15}"
LEVEL_FRAME_TEMPERATURES="${LEVEL_FRAME_TEMPERATURES:-0.35,0.8,1.6}"
TRAIN_WINDOW_EXPAND_RATIO="${TRAIN_WINDOW_EXPAND_RATIO:-1.5}"
SELECTION_LOSS_WEIGHT="${SELECTION_LOSS_WEIGHT:-0.5}"
ENABLE_HTG="${ENABLE_HTG:-true}"
HTG_LOSS_WEIGHT="${HTG_LOSS_WEIGHT:-0.1}"

SAVE_DIR="${SAVE_DIR:-/data/surglavi_checkpoint/${EXP_NAME}}"
SAVE_EVERY="${SAVE_EVERY:-5}"
SAVE_NAME="${SAVE_NAME:-final.pt}"
TB_LOGDIR="${TB_LOGDIR:-runs/${EXP_NAME}}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

# Keep text backbone pretrained, use ImageNet init for vision backbone,
# and do not load the full PeskaVLP init checkpoint.
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"

export CUDA_VISIBLE_DEVICES
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
export SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-$EXP_NAME}"
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
    --model_family "$MODEL_FAMILY" \
    --peskavlp_vision_backbone "$PESKAVLP_VISION_BACKBONE" \
    --peskavlp_vision_pretrained "$PESKAVLP_VISION_PRETRAINED" \
    --peskavlp_embed_dim "$PESKAVLP_EMBED_DIM" \
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
    --enable_htg "$ENABLE_HTG" \
    --htg_loss_weight "$HTG_LOSS_WEIGHT" \
    ${INIT_CHECKPOINT:+--init_checkpoint "$INIT_CHECKPOINT"} \
    ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"}
