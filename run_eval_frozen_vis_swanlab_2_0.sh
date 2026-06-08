#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh

CKPT_DIR="${CKPT_DIR:-outputs/same_video_triplet_xpool_adapter_no_warmup_8f_run1_gsvit}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-10 20 30 40 50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./eval_6.4_every_10_epochs_gpu3}"
SKIP_MISSING="${SKIP_MISSING:-true}"

VISION_BACKBONE="${VISION_BACKBONE:-gsvit_m5}"
VISION_WEIGHTS="${VISION_WEIGHTS:-GSViT.pkl}"
TEXT_MODEL="${TEXT_MODEL:-marcobombieri/surgicberta}"
CUDA_DEVICE="${CUDA_DEVICE:-3}"

EMBED_DIM="${EMBED_DIM:-256}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-6}"
NUM_FRAMES="${NUM_FRAMES:-8}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
TEMPORAL_LAYERS="${TEMPORAL_LAYERS:-2}"
TEMPORAL_HEADS="${TEMPORAL_HEADS:-12}"
TEMPORAL_DROPOUT="${TEMPORAL_DROPOUT:-0.1}"

DATASETS=(
  cholec80_phase
  cholec80_instrument
  bern_bypass70_phase
  stras_bypass70_phase
  grasp_phase
  grasp_step
)

for epoch in $CHECKPOINT_EPOCHS; do
  CKPT="$CKPT_DIR/vlp_epoch_${epoch}.pt"
  OUTPUT_DIR="$OUTPUT_ROOT/epoch_${epoch}"

  if [[ ! -f "$CKPT" ]]; then
    if [[ "$SKIP_MISSING" == "true" ]]; then
      echo "Skipping missing checkpoint: $CKPT"
      continue
    fi
    echo "Checkpoint not found: $CKPT" >&2
    exit 1
  fi

  mkdir -p "$OUTPUT_DIR"
  echo "Evaluating checkpoint: $CKPT"
  echo "Output directory: $OUTPUT_DIR"

  for ds in "${DATASETS[@]}"; do
    echo "Evaluating epoch ${epoch}, dataset: $ds"

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python zeroshot_evaluate.py \
      --dataset "$ds" \
      --ckpt "$CKPT" \
      --text_model "$TEXT_MODEL" \
      --vision_backbone "$VISION_BACKBONE" \
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
done
