#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "${SCRIPT_DIR}/00_env.sh"
PROBE_FRAMES="${1:-80}"; PROBE_WARMUP="${2:-10}"; EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"
CONTENT_WINDOW_MS="${CONTENT_WINDOW_MS:-100}"; CONTENT_SLICE_MS="${CONTENT_SLICE_MS:-0.05}"; CONTENT_START_DELAY_MS="${CONTENT_START_DELAY_MS:-2}"; CONTENT_THREADS="${CONTENT_THREADS:-256}"
STRENGTHS="${STRENGTHS:-0.03125 0.0625 0.09375 0.125 0.1875 0.25 0.375 0.5 0.75 1.0 1.5 2.0}"; TARGETS="${TARGETS:-1.20,1.45,1.80,2.30}"
ROOT="outputs/elastic_bev/${EXP_NAME}/contention_calibration_v6/smooth_microchain_priority"; mkdir -p "${ROOT}/candidates"
python tools/smooth_cuda_contention.py
python tools/profile_fixed_prefix_probe_smooth.py --full_cfg "${FULL_CFG}" --full_ckpt "${FULL_CKPT}" --warmup "${PROBE_WARMUP}" --frames "${PROBE_FRAMES}" --output_json "${ROOT}/probe_L0.json"
SUCCESS=0
for S in ${STRENGTHS}; do TAG="$(echo "${S}"|sed 's/\./p/g')"; echo "[CANDIDATE] strength=${S}"; set +e; python tools/profile_fixed_prefix_probe_smooth.py --full_cfg "${FULL_CFG}" --full_ckpt "${FULL_CKPT}" --warmup "${PROBE_WARMUP}" --frames "${PROBE_FRAMES}" --contention-strength "${S}" --contention-window-ms "${CONTENT_WINDOW_MS}" --contention-slice-ms "${CONTENT_SLICE_MS}" --contention-start-delay-ms "${CONTENT_START_DELAY_MS}" --contention-threads "${CONTENT_THREADS}" --output_json "${ROOT}/candidates/probe_s${TAG}.json" 2>&1|tee "${ROOT}/candidates/probe_s${TAG}.log"; RC=${PIPESTATUS[0]}; set -e; if [ "${RC}" -eq 0 ]; then SUCCESS=$((SUCCESS+1)); else rm -f "${ROOT}/candidates/probe_s${TAG}.json"; fi; done
[ "${SUCCESS}" -ge 4 ] || { echo "[ERROR] only ${SUCCESS} usable candidates"; exit 4; }
python tools/select_contention_levels_smooth.py --baseline "${ROOT}/probe_L0.json" --candidate-dir "${ROOT}/candidates" --window-ms "${CONTENT_WINDOW_MS}" --slice-ms "${CONTENT_SLICE_MS}" --start-delay-ms "${CONTENT_START_DELAY_MS}" --threads "${CONTENT_THREADS}" --targets "${TARGETS}" --output "${ROOT}/contention_levels.json"
echo "[RESULT] ${ROOT}/contention_levels.json"
