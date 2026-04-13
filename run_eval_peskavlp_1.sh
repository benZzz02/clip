#!/usr/bin/env bash
set -euo pipefail

export MODEL_FAMILY="${MODEL_FAMILY:-peskavlp}"
export TOKENIZER_NAME="${TOKENIZER_NAME:-emilyalsentzer/Bio_ClinicalBERT}"
export PESKAVLP_VISION_BACKBONE="${PESKAVLP_VISION_BACKBONE:-resnet_50}"
export PESKAVLP_VISION_PRETRAINED="${PESKAVLP_VISION_PRETRAINED:-random}"
export PESKAVLP_EMBED_DIM="${PESKAVLP_EMBED_DIM:-768}"
export FINETUNE_MODE="${FINETUNE_MODE:-lora}"
export LORA_RANK="${LORA_RANK:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export LORA_TARGETS="${LORA_TARGETS:-backbone_text.model.encoder.layer.,backbone_img.global_embedder}"
export OUTPUT_DIR="${OUTPUT_DIR:-./eval_outputs_peskavlp}"

exec bash run_eval_surglavi_1.sh
