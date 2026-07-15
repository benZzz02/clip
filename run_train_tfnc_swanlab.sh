#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-surgalign}"
if [[ -n "$CONDA_ENV" ]]; then
    CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
    if [[ ! -f "$CONDA_SH" ]]; then
        echo "ERROR: conda init script not found: $CONDA_SH" >&2
        echo "Set CONDA_SH=/path/to/conda.sh or CONDA_ENV= to skip activation." >&2
        exit 2
    fi
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi

NPROC="${NPROC:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_NAME="${RUN_NAME:-tfnc_xpool_8f_expand15}"
EXP_NAME="${EXP_NAME:-$RUN_NAME}"
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-train_frozen_vis.py}"

DECODE_CPU_THREAD_LIMIT="${DECODE_CPU_THREAD_LIMIT:-}"
DECODE_CPU_AFFINITY="${DECODE_CPU_AFFINITY:-}"
DECODE_NICE_LEVEL="${DECODE_NICE_LEVEL:-}"
DECODE_IONICE_CLASS="${DECODE_IONICE_CLASS:-}"
DECODE_IONICE_LEVEL="${DECODE_IONICE_LEVEL:-}"

PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-220}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_FRAMES="${NUM_FRAMES:-8}"

EPOCHS="${EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.02}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"

EMBED_DIM="${EMBED_DIM:-256}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
MAX_LENGTH="${MAX_LENGTH:-256}"

FFMPEG_TIMEOUT="${FFMPEG_TIMEOUT:-10}"
MAX_RETRY="${MAX_RETRY:-5}"
VIDEO_READER_THREADS="${VIDEO_READER_THREADS:-1}"
VIDEO_READER_CACHE_SIZE="${VIDEO_READER_CACHE_SIZE:-4}"
DATALOADER_IN_ORDER="${DATALOADER_IN_ORDER:-true}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
ASSUME_RESIZED_VIDEO="${ASSUME_RESIZED_VIDEO:-true}"
USE_SWANLAB="${USE_SWANLAB:-true}"
CUDA_PREFETCH="${CUDA_PREFETCH:-false}"
DRY_RUN="${DRY_RUN:-false}"

TEXT_MODEL_NAME="${TEXT_MODEL_NAME:-marcobombieri/surgicberta}"
VISION_BACKBONE="${VISION_BACKBONE:-convnext_lemonfm}"
VISION_PRETRAINED_WEIGHTS="${VISION_PRETRAINED_WEIGHTS:-lemonfm.pth}"
VIDEO_ROOT_FOLDER="${VIDEO_ROOT_FOLDER:-/data/znh/downloaded_video_224_test}"
MAIN_CSV_PATH="${MAIN_CSV_PATH:-surglavi_level_csv/all_video.csv}"

ANNOTATIONS_ROOT="${ANNOTATIONS_ROOT:-surglavi_level_csv}"
ANNOTATION_LEVELS="${ANNOTATION_LEVELS:-coarse,mid,fine}"
LEVEL_MIX="${LEVEL_MIX:-concat}"
LEVEL_BATCH_SIZES="${LEVEL_BATCH_SIZES:-fine:140,mid:60,coarse:20}"

SAMPLES_CACHE_DIR="${SAMPLES_CACHE_DIR:-.cache/pretrain_samples}"
USE_SAMPLES_CACHE="${USE_SAMPLES_CACHE:-true}"
REBUILD_SAMPLES_CACHE="${REBUILD_SAMPLES_CACHE:-false}"
SAMPLES_CACHE_VERSION="${SAMPLES_CACHE_VERSION:-v1}"

TRAINING_METHOD="${TRAINING_METHOD:-tfnc}"
LOCAL_TEMPERATURE="${LOCAL_TEMPERATURE:-0.15}"
SELECTION_POOLING="${SELECTION_POOLING:-xpool}"
LEVEL_FRAME_TEMPERATURES="${LEVEL_FRAME_TEMPERATURES:-0.6,0.9,1.2}"
TRAIN_WINDOW_EXPAND_RATIO="${TRAIN_WINDOW_EXPAND_RATIO:-1.5}"
ANCHOR_SAME_VIDEO_TRIPLETS="${ANCHOR_SAME_VIDEO_TRIPLETS:-}"
TFNC_V2_XPOOL_LOSS_WEIGHT="${TFNC_V2_XPOOL_LOSS_WEIGHT:-1.0}"
TFNC_V2_OUTSIDE_DENOMINATOR_WEIGHT="${TFNC_V2_OUTSIDE_DENOMINATOR_WEIGHT:-1.0}"

TRAIN_ENCODER_BASE_LAYERS="${TRAIN_ENCODER_BASE_LAYERS:-true}"
TRAIN_ENCODER_NUM_STAGES="${TRAIN_ENCODER_NUM_STAGES:-8}"
ENCODER_GRADIENT_CHECKPOINTING="${ENCODER_GRADIENT_CHECKPOINTING:-true}"

RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT-}"
SAVE_PREFIX="${SAVE_PREFIX:-/data/znh/outputs/$RUN_NAME/}"

sum_level_batch_sizes() {
    local spec="$1"
    local sum=0
    local item count
    IFS=',' read -r -a items <<< "$spec"
    for item in "${items[@]}"; do
        count="${item##*:}"
        if [[ ! "$count" =~ ^[0-9]+$ ]]; then
            echo "ERROR: Bad LEVEL_BATCH_SIZES item: $item" >&2
            exit 2
        fi
        sum=$((sum + count))
    done
    echo "$sum"
}

for level in fine mid coarse; do
    if [[ "$ANNOTATION_LEVELS" != *"$level"* ]]; then
        echo "ERROR: TFNC mix-level training expects ANNOTATION_LEVELS to include $level, got $ANNOTATION_LEVELS" >&2
        exit 2
    fi
    if [[ "$LEVEL_BATCH_SIZES" != *"$level:"* ]]; then
        echo "ERROR: TFNC mix-level training expects LEVEL_BATCH_SIZES to include $level, got $LEVEL_BATCH_SIZES" >&2
        exit 2
    fi
done

LEVEL_BATCH_TOTAL="$(sum_level_batch_sizes "$LEVEL_BATCH_SIZES")"
if [[ "$LEVEL_BATCH_TOTAL" -ne "$PER_GPU_BATCH_SIZE" ]]; then
    echo "ERROR: sum(LEVEL_BATCH_SIZES)=$LEVEL_BATCH_TOTAL must equal PER_GPU_BATCH_SIZE=$PER_GPU_BATCH_SIZE" >&2
    exit 2
fi

if [[ "$TRAINING_METHOD" != "tfnc" && "$TRAINING_METHOD" != "tfnc_v2" ]]; then
    echo "ERROR: run_train_tfnc_swanlab.sh is intended for TRAINING_METHOD=tfnc or tfnc_v2, got $TRAINING_METHOD" >&2
    exit 2
fi

if [[ ! -f "$TRAIN_ENTRYPOINT" ]]; then
    echo "ERROR: TRAIN_ENTRYPOINT not found: $TRAIN_ENTRYPOINT" >&2
    exit 2
fi

if [[ "$DRY_RUN" != "true" ]]; then
    if [[ "$SAVE_PREFIX" == */ ]]; then
        mkdir -p "$SAVE_PREFIX"
    else
        mkdir -p "$(dirname "$SAVE_PREFIX")"
    fi
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
export REQUIRE_VISUAL_GRAD="${REQUIRE_VISUAL_GRAD:-true}"
export SWANLAB_EXPERIMENT_NAME="$EXP_NAME"
export SAVE_PREFIX
export DECODE_CPU_THREAD_LIMIT
export DECODE_CPU_AFFINITY
export DECODE_NICE_LEVEL
export DECODE_IONICE_CLASS
export DECODE_IONICE_LEVEL
export DATALOADER_IN_ORDER
export DATALOADER_PREFETCH_FACTOR
export TFNC_V2_XPOOL_LOSS_WEIGHT
export TFNC_V2_OUTSIDE_DENOMINATOR_WEIGHT

cmd=(
    torchrun
    --standalone
    --nproc_per_node="$NPROC"
    "$TRAIN_ENTRYPOINT"
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
    --cuda_prefetch "$CUDA_PREFETCH"
    --local_temperature "$LOCAL_TEMPERATURE"
    --selection_pooling "$SELECTION_POOLING"
    --level_frame_temperatures "$LEVEL_FRAME_TEMPERATURES"
    --train_window_expand_ratio "$TRAIN_WINDOW_EXPAND_RATIO"
    --training_method "$TRAINING_METHOD"
    --tfnc_v2_xpool_loss_weight "$TFNC_V2_XPOOL_LOSS_WEIGHT"
    --tfnc_v2_outside_denominator_weight "$TFNC_V2_OUTSIDE_DENOMINATOR_WEIGHT"
    --train_encoder_base_layers "$TRAIN_ENCODER_BASE_LAYERS"
    --train_encoder_num_stages "$TRAIN_ENCODER_NUM_STAGES"
    --encoder_gradient_checkpointing "$ENCODER_GRADIENT_CHECKPOINTING"
)

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    cmd+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
if [[ -n "$ANCHOR_SAME_VIDEO_TRIPLETS" ]]; then
    cmd+=(--anchor_same_video_triplets "$ANCHOR_SAME_VIDEO_TRIPLETS")
fi

echo "TFNC training config:"
echo "  RUN_NAME=$RUN_NAME"
echo "  TRAIN_ENTRYPOINT=$TRAIN_ENTRYPOINT"
echo "  EPOCHS=$EPOCHS"
echo "  LEARNING_RATE=$LEARNING_RATE"
echo "  NUM_FRAMES=$NUM_FRAMES"
echo "  TRAIN_WINDOW_EXPAND_RATIO=$TRAIN_WINDOW_EXPAND_RATIO"
echo "  PER_GPU_BATCH_SIZE=$PER_GPU_BATCH_SIZE"
echo "  NUM_WORKERS=$NUM_WORKERS"
echo "  VIDEO_READER_THREADS=$VIDEO_READER_THREADS"
echo "  VIDEO_READER_CACHE_SIZE=$VIDEO_READER_CACHE_SIZE"
echo "  DATALOADER_IN_ORDER=$DATALOADER_IN_ORDER"
echo "  DATALOADER_PREFETCH_FACTOR=$DATALOADER_PREFETCH_FACTOR"
echo "  LEVEL_BATCH_SIZES=$LEVEL_BATCH_SIZES"
echo "  TRAINING_METHOD=$TRAINING_METHOD"
echo "  SELECTION_POOLING=$SELECTION_POOLING"
echo "  LEVEL_FRAME_TEMPERATURES=$LEVEL_FRAME_TEMPERATURES"
echo "  ANCHOR_SAME_VIDEO_TRIPLETS=${ANCHOR_SAME_VIDEO_TRIPLETS:-<auto>}"
echo "  TFNC_V2_XPOOL_LOSS_WEIGHT=$TFNC_V2_XPOOL_LOSS_WEIGHT"
echo "  TFNC_V2_OUTSIDE_DENOMINATOR_WEIGHT=$TFNC_V2_OUTSIDE_DENOMINATOR_WEIGHT"
echo "  TRAIN_ENCODER_NUM_STAGES=$TRAIN_ENCODER_NUM_STAGES"
echo "  RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-<none>}"
echo "  SAVE_PREFIX=$SAVE_PREFIX"
echo "  DECODE_CPU_THREAD_LIMIT=${DECODE_CPU_THREAD_LIMIT:-<unset>}"
echo "  DECODE_CPU_AFFINITY=${DECODE_CPU_AFFINITY:-<none>}"
echo "  DECODE_NICE_LEVEL=${DECODE_NICE_LEVEL:-<none>}"
echo "  DECODE_IONICE_CLASS=${DECODE_IONICE_CLASS:-<none>}"
echo "  DECODE_IONICE_LEVEL=${DECODE_IONICE_LEVEL:-<none>}"

if [[ "$DRY_RUN" == "true" ]]; then
    printf "Command:"
    printf " %q" "${cmd[@]}"
    printf "\n"
    exit 0
fi

"${cmd[@]}"
