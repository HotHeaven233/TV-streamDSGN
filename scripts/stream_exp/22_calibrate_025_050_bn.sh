#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EPOCH="${1:-20}"
BATCHES="${2:-1000}"

"${SCRIPT_DIR}/21_calibrate_fixed_bn.sh" \
    "${EPOCH}" "${BATCHES}" \
    "0.25,0.25,0.25,0.25,0.25,0.25"

"${SCRIPT_DIR}/21_calibrate_fixed_bn.sh" \
    "${EPOCH}" "${BATCHES}" \
    "0.5,0.5,0.5,0.5,0.5,0.5"

