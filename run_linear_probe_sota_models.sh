#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODELS="${MODELS:-surgvlp,hecvl,surgclip_beta,surgalign}"

SURGVLP_CKPT="${SURGVLP_CKPT:-}"
HECVL_CKPT="${HECVL_CKPT:-}"
PESKAVLP_CKPT="${PESKAVLP_CKPT:-}"
SURGCLIP_BETA_CKPT="${SURGCLIP_BETA_CKPT:-}"
SURGALIGN_CKPT="${SURGALIGN_CKPT:-${CKPT:-}}"

IFS=',' read -ra MODEL_ARR <<< "$MODELS"

for model in "${MODEL_ARR[@]}"; do
  model="$(echo "$model" | xargs)"
  [[ -z "$model" ]] && continue

  export FEATURE_MODE="$model"
  export RUN_NAME="${RUN_NAME_PREFIX:-linear_probe_sota}/$model"

  case "$model" in
    surgvlp)
      export CKPT="$SURGVLP_CKPT"
      ;;
    hecvl)
      export CKPT="$HECVL_CKPT"
      ;;
    peskavlp)
      export CKPT="$PESKAVLP_CKPT"
      ;;
    surgclip_beta|surgclip-beta|surgclip)
      export CKPT="$SURGCLIP_BETA_CKPT"
      ;;
    surgalign|vlp)
      export CKPT="$SURGALIGN_CKPT"
      ;;
    *)
      echo "Unknown SOTA model: $model" >&2
      exit 1
      ;;
  esac

  echo "Running SOTA linear probe for FEATURE_MODE=$FEATURE_MODE"
  bash ./run_linear_probe.sh "$@"
done
