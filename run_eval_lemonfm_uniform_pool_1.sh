#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh


CKPT="${CKPT:-outputs/same_video_triplet_xpool_adapter_lemonfm_uniform_8f_run2/vlp_epoch_50.pt}"
VISION_WEIGHTS="${VISION_WEIGHTS:-lemonfm.pth}"
TEXT_MODEL="${TEXT_MODEL:-marcobombieri/surgicberta}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_lemonfm_uniform_epoch_50_run2}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"

EMBED_DIM="${EMBED_DIM:-256}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_FRAMES="${NUM_FRAMES:-8}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
TEMPORAL_LAYERS="${TEMPORAL_LAYERS:-2}"
TEMPORAL_HEADS="${TEMPORAL_HEADS:-12}"
TEMPORAL_DROPOUT="${TEMPORAL_DROPOUT:-0.1}"

for ds in \
  cholec80_instrument \
  stras_bypass70_phase \
  grasp_step \
  autolaparo_phase \
  sarrarp50_phase \
  heichole_instrument
do
  echo "Evaluating dataset: $ds"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python evaluate_lemonfm_uniform_pool.py \
    --dataset "$ds" \
    --ckpt "$CKPT" \
    --text_model "$TEXT_MODEL" \
    --vision_backbone convnext_lemonfm \
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
done
