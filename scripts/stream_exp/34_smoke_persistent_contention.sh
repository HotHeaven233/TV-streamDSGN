#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-20}"
N="${2:-100}"
LEVELS_JSON="${3:-outputs/elastic_bev/${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}/contention_calibration_v3/persistent_window/contention_levels.json}"

# Profile ID 1 = Full.
"${SCRIPT_DIR}/33_profile_all_84_persistent_contention.sh" \
  "${EPOCH}" "${N}" \
  20 5 \
  1 1 \
  "${LEVELS_JSON}"

