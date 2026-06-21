#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh

set -u

CKPT="${CKPT:-outputs/same_video_triplet_xpool_adapter_vitl_timesformer_8f_run1/vlp_epoch_50.pt}"
VISION_WEIGHTS="${VISION_WEIGHTS:-endossl_vitl.pth}"
TEXT_MODEL="${TEXT_MODEL:-marcobombieri/surgicberta}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_vitl_timesformer_epoch_50}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

EMBED_DIM=256
BATCH_SIZE=32
NUM_WORKERS=6
NUM_FRAMES=8
FRAME_STRIDE=1
TEMPORAL_LAYERS=2
TEMPORAL_HEADS=12
TEMPORAL_DROPOUT=0.1

eval_dataset() {
    local ds=$1
    echo "Evaluating dataset: $ds"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python evaluate_timesformer_vitl.py \
        --dataset "$ds" \
        --ckpt "$CKPT" \
        --text_model "$TEXT_MODEL" \
        --vision_weights "$VISION_WEIGHTS" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --embed_dim "$EMBED_DIM" \
        --num_frames "$NUM_FRAMES" \
        --frame_stride "$FRAME_STRIDE" \
        --temporal_layers "$TEMPORAL_LAYERS" \
        --temporal_heads "$TEMPORAL_HEADS" \
        --temporal_dropout "$TEMPORAL_DROPOUT" \
        --output_dir "$OUTPUT_DIR"
}

# Phase datasets
eval_dataset cholec80_phase
eval_dataset autolaparo_phase
eval_dataset heichole_phase
eval_dataset bern_bypass70_phase
eval_dataset stras_bypass70_phase
eval_dataset grasp_phase
eval_dataset grasp_step
eval_dataset sarrarp50_phase

# Tool datasets
eval_dataset cholec80_instrument
eval_dataset heichole_instrument
eval_dataset grasp_instrument

# Triplet
eval_dataset cholect50_triplet

echo "所有数据集评估完成: $OUTPUT_DIR"
