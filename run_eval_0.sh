#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

CKPT="/mnt/mydisk/CLIP/vlp_epoch_50.pt"
VISION_WEIGHTS="/mnt/mydisk/CLIP/lemonfm.pth"
TEXT_MODEL="marcobombieri/surgicberta"
OUTPUT_DIR="./eval_outputs11"
CUDA_DEVICE=0

EMBED_DIM=256
BATCH_SIZE=32
NUM_WORKERS=4
NUM_FRAMES=8
FRAME_STRIDE=1
TEMPORAL_LAYERS=2
TEMPORAL_HEADS=8
TEMPORAL_DROPOUT=0.1

for ds in \
  cholec80_phase \
  bern_bypass70_phase \
  grasp_phase \
  grasp_instrument \
  cholect50_triplet \
  heichole_phase
do
  echo "Evaluating dataset: $ds"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python zeroshot_evaluate.py \
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
done
