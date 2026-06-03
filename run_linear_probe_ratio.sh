#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export SHOT_MODE="${SHOT_MODE:-ratio}"
export SHOTS="${SHOTS:-0.1,1.0}"
export SEEDS="${SEEDS:-0,1,2}"

exec bash ./run_linear_probe.sh "$@"
