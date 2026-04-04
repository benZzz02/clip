#!/usr/bin/env bash
set -euo pipefail

set +u
source ~/miniconda3/etc/profile.d/conda.sh
if [[ "${CONDA_DEFAULT_ENV:-}" != "vllm" ]]; then
  conda activate vllm
fi
set -u

CKPT="${CKPT:-/data/nfs_data/CLIP/surglavi_checkpoint/surglavi_lora_3gpu_bs100/surglavi_epoch_50.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_outputs_surglavi}"
CUDA_DEVICE="${CUDA_DEVICE:-2}"

SURGCLIP_MODEL_NAME="${SURGCLIP_MODEL_NAME:-SurgCLIP-B}"
TOKENIZER_NAME="${TOKENIZER_NAME:-bert-base-uncased}"

NUM_FRAMES="${NUM_FRAMES:-8}"
MODEL_NUM_FRAMES="${MODEL_NUM_FRAMES:-8}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

mkdir -p "$OUTPUT_DIR"

for ds in \
  cholect50_triplet \
  sarrarp50_phase \
  heichole_phase \
  heichole_instrument
do
  echo "Evaluating dataset: $ds"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python zeroshot_evaluate_surglavi.py \
    --dataset "$ds" \
    --ckpt "$CKPT" \
    --surgclip_model_name "$SURGCLIP_MODEL_NAME" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --num_frames "$NUM_FRAMES" \
    --model_num_frames "$MODEL_NUM_FRAMES" \
    --frame_stride "$FRAME_STRIDE" \
    --image_size "$IMAGE_SIZE" \
    --output_dir "$OUTPUT_DIR"
done
