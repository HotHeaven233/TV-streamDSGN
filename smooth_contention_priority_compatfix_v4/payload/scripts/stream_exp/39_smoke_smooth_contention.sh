#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; "${SCRIPT_DIR}/40_profile_all_84_smooth_contention.sh" "${1:-20}" "${2:-100}" 20 5 1 1 "${3:-outputs/elastic_bev/${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}/contention_calibration_v6/smooth_microchain_priority/contention_levels.json}"
