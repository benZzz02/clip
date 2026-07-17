#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# SurgLaVi linear probing protocol:
# - downstream tasks: Cholec80 phase, AutoLaparo phase, GraSP phase/step
# - temporal input: 32-frame window around each labeled frame
# - CLS few-shot: 1, 2, 4, 8, 16 labeled frames per class
# - Video few/full-shot: 10%, 50%, 100% labeled training videos
# - five random subsets/seeds
export DATASETS="${DATASETS:-cholec80_phase,autolaparo_phase,grasp_phase,grasp_step}"
export NUM_FRAMES="${NUM_FRAMES:-32}"
export FRAME_STRIDE="${FRAME_STRIDE:-1}"
export SEEDS="${SEEDS:-0,1,2,3,4}"
export EPOCHS="${EPOCHS:-50}"
export PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-256}"
export PROBE_OPTIMIZER="${PROBE_OPTIMIZER:-sgd}"
export MOMENTUM="${MOMENTUM:-0.9}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
export CACHE_DIR="${CACHE_DIR:-linear_probe_outputs/linear_probe_surglavi_protocol/cache}"

# Encoding batch size is a resource knob rather than part of the protocol.
# 32-frame windows are memory-hungry, so keep this conservative by default.
export ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-16}"

export MODELS="${MODELS:-surgvlp,hecvl,surgclip_beta,surgalign}"

export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-linear_probe_surglavi_protocol/cls}"
export SHOT_MODE=cls
export SHOTS="${CLS_SHOTS:-1,2,4,8,16}"
bash ./run_linear_probe_sota_models.sh "$@"

export RUN_NAME_PREFIX="${RUN_NAME_PREFIX_VIDEO:-linear_probe_surglavi_protocol/video}"
export SHOT_MODE=video
export SHOTS="${VIDEO_SHOTS:-0.1,0.5,1.0}"
bash ./run_linear_probe_sota_models.sh "$@"
