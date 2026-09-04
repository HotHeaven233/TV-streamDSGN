#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-20}"
N="${2:-100}"
LEVELS_JSON="${3:-outputs/elastic_bev/${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}/contention_calibration_v4/persistent_window_dual_stream/contention_levels.json}"

"${SCRIPT_DIR}/36_profile_all_84_persistent_contention_streamfix.sh" \
  "${EPOCH}" "${N}" \
  20 5 \
  1 1 \
  "${LEVELS_JSON}"

