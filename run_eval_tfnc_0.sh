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

RUN_NAME="${RUN_NAME:-tfnc_xpool_8f_expand15}"
CKPT="${CKPT:-/data/znh/outputs/$RUN_NAME/vlp_epoch_50.pt}"
VISION_BACKBONE="${VISION_BACKBONE:-convnext_lemonfm}"
VISION_WEIGHTS="${VISION_WEIGHTS:-lemonfm.pth}"
TEXT_MODEL="${TEXT_MODEL:-marcobombieri/surgicberta}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/znh/eval_outputs/${RUN_NAME}_epoch50_gpu1}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
DRY_RUN="${DRY_RUN:-false}"

EMBED_DIM="${EMBED_DIM:-256}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-6}"
NUM_FRAMES="${NUM_FRAMES:-8}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
TEMPORAL_LAYERS="${TEMPORAL_LAYERS:-2}"
TEMPORAL_HEADS="${TEMPORAL_HEADS:-12}"
TEMPORAL_DROPOUT="${TEMPORAL_DROPOUT:-0.1}"
SELECTION_POOLING="${SELECTION_POOLING:-similarity}"
EVAL_POOLING="${EVAL_POOLING:-global}"
XPOOL_TEXT_CHUNK_SIZE="${XPOOL_TEXT_CHUNK_SIZE:-64}"

DATASETS=(
    cholec80_phase
    bern_bypass70_phase
    grasp_phase
    grasp_instrument
    cholect50_triplet
    heichole_phase
)

if [[ "$DRY_RUN" != "true" ]]; then
    if [[ ! -f "$CKPT" ]]; then
        echo "ERROR: checkpoint not found: $CKPT" >&2
        exit 2
    fi
    mkdir -p "$OUTPUT_DIR"
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "TFNC eval config:"
echo "  RUN_NAME=$RUN_NAME"
echo "  CKPT=$CKPT"
echo "  CUDA_DEVICE=$CUDA_DEVICE"
echo "  NUM_FRAMES=$NUM_FRAMES"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  SELECTION_POOLING=$SELECTION_POOLING"
echo "  EVAL_POOLING=$EVAL_POOLING"
echo "  XPOOL_TEXT_CHUNK_SIZE=$XPOOL_TEXT_CHUNK_SIZE"
echo "  DATASETS=${DATASETS[*]}"

for ds in "${DATASETS[@]}"; do
    cmd=(
        python
        zeroshot_evaluate.py
        --dataset "$ds"
        --ckpt "$CKPT"
        --text_model "$TEXT_MODEL"
        --vision_backbone "$VISION_BACKBONE"
        --vision_weights "$VISION_WEIGHTS"
        --batch_size "$BATCH_SIZE"
        --num_workers "$NUM_WORKERS"
        --embed_dim "$EMBED_DIM"
        --num_frames "$NUM_FRAMES"
        --frame_stride "$FRAME_STRIDE"
        --temporal_layers "$TEMPORAL_LAYERS"
        --temporal_heads "$TEMPORAL_HEADS"
        --temporal_dropout "$TEMPORAL_DROPOUT"
        --selection_pooling "$SELECTION_POOLING"
        --eval_pooling "$EVAL_POOLING"
        --xpool_text_chunk_size "$XPOOL_TEXT_CHUNK_SIZE"
        --output_dir "$OUTPUT_DIR"
    )

    echo "Evaluating dataset: $ds"
    if [[ "$DRY_RUN" == "true" ]]; then
        printf "Command: CUDA_VISIBLE_DEVICES=%q" "$CUDA_DEVICE"
        printf " %q" "${cmd[@]}"
        printf "\n"
    else
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${cmd[@]}"
    fi
done
