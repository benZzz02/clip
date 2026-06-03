#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export SHOT_MODE="${SHOT_MODE:-cls}"
export SHOTS="${SHOTS:-1,2,4,8,16}"
export SEEDS="${SEEDS:-0,1,2,3,4}"

exec ./run_linear_probe.sh "$@"
