#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f ~/miniconda3/etc/profile.d/conda.sh ]]; then
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-vllm}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export CUDA_VISIBLE_DEVICES

FEATURE_MODE="${FEATURE_MODE:-vlp}"
DEFAULT_VLP_CKPT="${DEFAULT_VLP_CKPT:-/mnt/mydisk/CLIP/outputs/same_video_triplet_xpool_adapter_warmup_16f_run1/vlp_epoch_32.pt}"
CKPT="${CKPT:-}"
if [[ "$FEATURE_MODE" == "vlp" || "$FEATURE_MODE" == "surgalign" ]]; then
  CKPT="${CKPT:-$DEFAULT_VLP_CKPT}"
fi
RUN_NAME="${RUN_NAME:-linear_probe}"
OUTPUT_ROOT="${OUTPUT_ROOT:-linear_probe_outputs/$RUN_NAME}"
CACHE_DIR="${CACHE_DIR:-$OUTPUT_ROOT/cache}"

ANNO_ROOT="${ANNO_ROOT:-anno_downstream}"
DATA_ROOT="${DATA_ROOT:-/mnt/mydisk}"
VISION_WEIGHTS="${VISION_WEIGHTS:-lemonfm.pth}"
TEXT_MODEL="${TEXT_MODEL:-marcobombieri/surgicberta}"
EXTERNAL_CONFIG="${EXTERNAL_CONFIG:-}"
EXTERNAL_CACHE_DIR="${EXTERNAL_CACHE_DIR:-}"
SURGCLIP_MODEL_NAME="${SURGCLIP_MODEL_NAME:-SurgCLIP-B}"
PROBE_FEATURE="${PROBE_FEATURE:-projected}"

DATASETS="${DATASETS:-cholec80_phase,cholec80_instrument,grasp_phase,grasp_step,grasp_instrument,heichole_phase,heichole_instrument}"
SHOT_MODE="${SHOT_MODE:-ratio}"
SHOTS="${SHOTS:-0.1,1.0}"
SEEDS="${SEEDS:-0,1,2}"

NUM_FRAMES="${NUM_FRAMES:-8}"
CONTEXT_NUM_FRAMES="${CONTEXT_NUM_FRAMES:-}"
CONTEXT_STRIDE="${CONTEXT_STRIDE:-}"
CONTEXT_POOLING="${CONTEXT_POOLING:-mean}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
EMBED_DIM="${EMBED_DIM:-256}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
PROBE_OPTIMIZER="${PROBE_OPTIMIZER:-adamw}"
MOMENTUM="${MOMENTUM:-0.9}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-128}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-4096}"
NUM_WORKERS="${NUM_WORKERS:-4}"
AMP="${AMP:-true}"

if [[ ( "$FEATURE_MODE" == "vlp" || "$FEATURE_MODE" == "surgalign" ) && -z "$CKPT" ]]; then
  echo "CKPT is required when FEATURE_MODE=vlp" >&2
  exit 1
fi

IFS=',' read -ra DATASET_ARR <<< "$DATASETS"
IFS=',' read -ra SHOT_ARR <<< "$SHOTS"

for dataset in "${DATASET_ARR[@]}"; do
  dataset="$(echo "$dataset" | xargs)"
  [[ -z "$dataset" ]] && continue

  for shot in "${SHOT_ARR[@]}"; do
    shot="$(echo "$shot" | xargs)"
    [[ -z "$shot" ]] && continue

    out_dir="$OUTPUT_ROOT/${SHOT_MODE}/${dataset}/shot_${shot}"
    mkdir -p "$out_dir"

    cmd=(
      "$PYTHON_BIN" linear_probe.py
      --dataset "$dataset"
      --feature_mode "$FEATURE_MODE"
      --vision_weights "$VISION_WEIGHTS"
      --text_model "$TEXT_MODEL"
      --anno_root "$ANNO_ROOT"
      --data_root "$DATA_ROOT"
      --output_dir "$out_dir"
      --cache_dir "$CACHE_DIR"
      --surgclip_model_name "$SURGCLIP_MODEL_NAME"
      --probe_feature "$PROBE_FEATURE"
      --shot_mode "$SHOT_MODE"
      --shot_ratio "$shot"
      --seeds "$SEEDS"
      --num_frames "$NUM_FRAMES"
      --context_pooling "$CONTEXT_POOLING"
      --frame_stride "$FRAME_STRIDE"
      --embed_dim "$EMBED_DIM"
      --epochs "$EPOCHS"
      --lr "$LR"
      --weight_decay "$WEIGHT_DECAY"
      --probe_optimizer "$PROBE_OPTIMIZER"
      --momentum "$MOMENTUM"
      --encode_batch_size "$ENCODE_BATCH_SIZE"
      --probe_batch_size "$PROBE_BATCH_SIZE"
      --num_workers "$NUM_WORKERS"
    )

    if [[ -n "$EXTERNAL_CONFIG" ]]; then
      cmd+=(--external_config "$EXTERNAL_CONFIG")
    fi
    if [[ -n "$EXTERNAL_CACHE_DIR" ]]; then
      cmd+=(--external_cache_dir "$EXTERNAL_CACHE_DIR")
    fi
    if [[ -n "$CONTEXT_NUM_FRAMES" ]]; then
      cmd+=(--context_num_frames "$CONTEXT_NUM_FRAMES")
    fi
    if [[ -n "$CONTEXT_STRIDE" ]]; then
      cmd+=(--context_stride "$CONTEXT_STRIDE")
    fi
    if [[ "$SHOT_MODE" == "cls" ]]; then
      cmd+=(--shots_per_class "$shot")
    fi

    if [[ -n "$CKPT" ]]; then
      cmd+=(--ckpt "$CKPT")
    fi
    if [[ "$AMP" == "true" ]]; then
      cmd+=(--amp)
    fi

    echo "Running linear probe: dataset=$dataset shot=$shot shot_mode=$SHOT_MODE seeds=$SEEDS feature_mode=$FEATURE_MODE"
    "${cmd[@]}"
  done
done
