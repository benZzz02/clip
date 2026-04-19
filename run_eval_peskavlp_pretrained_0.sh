#!/usr/bin/env bash
set -euo pipefail

export MODEL_FAMILY="${MODEL_FAMILY:-peskavlp}"
export TOKENIZER_NAME="${TOKENIZER_NAME:-emilyalsentzer/Bio_ClinicalBERT}"
export PESKAVLP_VISION_BACKBONE="${PESKAVLP_VISION_BACKBONE:-resnet_50}"
export PESKAVLP_VISION_PRETRAINED="${PESKAVLP_VISION_PRETRAINED:-random}"
export PESKAVLP_EMBED_DIM="${PESKAVLP_EMBED_DIM:-768}"
export FINETUNE_MODE="${FINETUNE_MODE:-full}"
export CKPT="${CKPT:-PeskaVLP.pth}"
export OUTPUT_DIR="${OUTPUT_DIR:-./eval_outputs_peskavlp_pretrained}"

exec bash run_eval_surglavi_0.sh
