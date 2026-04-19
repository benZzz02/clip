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

CKPT="${CKPT:-/data/surglavi_checkpoint/peskavlp_text_bioclinicalbert_vis_imagenet_no_method/final.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_outputs_peskavlp_2}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

SURGCLIP_MODEL_NAME="${SURGCLIP_MODEL_NAME:-SurgCLIP-B}"
TOKENIZER_NAME="${TOKENIZER_NAME:-bert-base-uncased}"
MODEL_FAMILY="${MODEL_FAMILY:-surgclip}"
PESKAVLP_VISION_BACKBONE="${PESKAVLP_VISION_BACKBONE:-resnet_50}"
PESKAVLP_VISION_PRETRAINED="${PESKAVLP_VISION_PRETRAINED:-random}"
PESKAVLP_EMBED_DIM="${PESKAVLP_EMBED_DIM:-768}"
DATA_ROOT="${DATA_ROOT:-/data/nfs_data}"

NUM_FRAMES="${NUM_FRAMES:-8}"
MODEL_NUM_FRAMES="${MODEL_NUM_FRAMES:-8}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
FINETUNE_MODE="${FINETUNE_MODE:-lora}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGETS="${LORA_TARGETS:-text_encoder.encoder.layer.,vision_encoder.model.blocks.,backbone_text.model.encoder.layer.,backbone_img.global_embedder}"

mkdir -p "$OUTPUT_DIR"

for ds in \
  cholec80_phase \
  cholec80_instrument \
  bern_bypass70_phase \
  stras_bypass70_phase
do
  echo "Evaluating dataset: $ds"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" DATA_ROOT="$DATA_ROOT" python zeroshot_evaluate_surglavi.py \
    --dataset "$ds" \
    --ckpt "$CKPT" \
    --model_family "$MODEL_FAMILY" \
    --surgclip_model_name "$SURGCLIP_MODEL_NAME" \
    --peskavlp_vision_backbone "$PESKAVLP_VISION_BACKBONE" \
    --peskavlp_vision_pretrained "$PESKAVLP_VISION_PRETRAINED" \
    --peskavlp_embed_dim "$PESKAVLP_EMBED_DIM" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --num_frames "$NUM_FRAMES" \
    --model_num_frames "$MODEL_NUM_FRAMES" \
    --frame_stride "$FRAME_STRIDE" \
    --image_size "$IMAGE_SIZE" \
    --finetune_mode "$FINETUNE_MODE" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --lora_targets "$LORA_TARGETS" \
    --output_dir "$OUTPUT_DIR"
done
