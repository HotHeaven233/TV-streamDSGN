#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-20}"
N="${2:-100}"
FRAMES="${3:-80}"
WARMUP="${4:-10}"
START_ID="${5:-1}"
END_ID="${6:-84}"
LEVELS_JSON="${7:-outputs/elastic_bev/${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}/contention_calibration_v4/persistent_window_dual_stream/contention_levels.json}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

RAW_CKPT="outputs/elastic_bev/${EXP_NAME}/ckpt/checkpoint_epoch_${EPOCH}.pth"
BANK="outputs/elastic_bev/${EXP_NAME}/bn_bank/causal_prefix_bank_e${EPOCH}_n${N}.pth"
SAP_ROOT="outputs/elastic_bev/${EXP_NAME}/all84_sap/e${EPOCH}_n${N}_10Hz"
SCHEDULE_TSV="${SAP_ROOT}/schedules.tsv"
QUALITY_CSV="${QUALITY_CSV:-${SAP_ROOT}/all84_sap_summary.csv}"
BASELINE_ROOT="outputs/elastic_bev/${EXP_NAME}/all84_forward_profile/e${EPOCH}_n${N}"

ROOT="outputs/elastic_bev/${EXP_NAME}/all84_contention_profile_v4/e${EPOCH}_n${N}"
TMP_DIR="${ROOT}/_tmp"

mkdir -p "${ROOT}" "${TMP_DIR}"

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"
require_file "${RAW_CKPT}"
require_file "${BANK}"
require_file "${SCHEDULE_TSV}"
require_file "${QUALITY_CSV}"
require_file "${LEVELS_JSON}"
require_file "tools/materialize_causal_prefix_bn_profile.py"
require_file "tools/profile_fixed_forward_components.py"
require_file "tools/build_contention_latency_table_persistent.py"

mapfile -t LEVEL_ROWS < <(python - "${LEVELS_JSON}" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
w = x["workload"]
for level in x["levels"]:
    if level["level"] == "L0":
        continue
    print(
        f'{level["level"]}\t'
        f'{level["strength"]}\t'
        f'{w["duration_ms"]}\t'
        f'{w["start_delay_ms"]}\t'
        f'{w["threads"]}'
    )
PY
)

for ROW in "${LEVEL_ROWS[@]}"; do
  IFS=$'\t' read -r LEVEL STRENGTH DURATION_MS START_DELAY_MS THREADS <<< "${ROW}"

  LEVEL_ROOT="${ROOT}/${LEVEL}"
  mkdir -p "${LEVEL_ROOT}"

  echo
  echo "======================================================================"
  echo "${LEVEL}: strength=${STRENGTH}"
  echo "window=${DURATION_MS}ms; start-delay=${START_DELAY_MS}ms"
  echo "dual non-default CUDA streams"
  echo "profiles ${START_ID}..${END_ID}; frames=${FRAMES}; warmup=${WARMUP}"
  echo "======================================================================"

  tail -n +2 "${SCHEDULE_TSV}" | while IFS=$'\t' read -r ID SCHEDULE TAG RES2 RES3 RES4 FPN STEREO RPN; do
    ID_NUM=$((10#${ID}))
    if [ "${ID_NUM}" -lt "${START_ID}" ] || [ "${ID_NUM}" -gt "${END_ID}" ]; then
      continue
    fi

    ID3="$(printf '%03d' "${ID_NUM}")"
    PROFILE_ROOT="${LEVEL_ROOT}/${ID3}_${TAG}"
    SUMMARY="${PROFILE_ROOT}/forward_profile_summary.json"
    RAW="${PROFILE_ROOT}/forward_profile_raw.csv"

    if [ -f "${SUMMARY}" ]; then
      echo "[SKIP ${LEVEL} ${ID3}/084] ${SCHEDULE}"
      continue
    fi

    mkdir -p "${PROFILE_ROOT}"
    TMP_CKPT="${TMP_DIR}/${LEVEL}_${ID3}_${TAG}.pth"

    python tools/materialize_causal_prefix_bn_profile.py \
      --elastic_ckpt "${RAW_CKPT}" \
      --prefix_bn_bank "${BANK}" \
      --schedule "${SCHEDULE}" \
      --output_ckpt "${TMP_CKPT}"

    set +e
    python tools/profile_fixed_forward_components.py \
      --full_cfg "${FULL_CFG}" \
      --full_ckpt "${FULL_CKPT}" \
      --elastic_ckpt "${TMP_CKPT}" \
      --schedule "${SCHEDULE}" \
      --warmup "${WARMUP}" \
      --frames "${FRAMES}" \
      --workers 0 \
      --contention-strength "${STRENGTH}" \
      --contention-duration-ms "${DURATION_MS}" \
      --contention-start-delay-ms "${START_DELAY_MS}" \
      --contention-threads "${THREADS}" \
      --raw_csv "${RAW}" \
      --summary_json "${SUMMARY}" \
      2>&1 | tee "${PROFILE_ROOT}/console.log"
    RC=${PIPESTATUS[0]}
    set -e

    if [ "${RC}" -ne 0 ]; then
      echo "[ERROR] ${LEVEL} profile ${ID3} failed."
      echo "If it is a genuine window-overrun under heavy contention,"
      echo "increase CONTENT_DURATION_MS and recalibrate before continuing."
      echo "[KEPT] ${TMP_CKPT}"
      exit "${RC}"
    fi

    rm -f "${TMP_CKPT}"
  done
done

python tools/build_contention_latency_table_persistent.py \
  --levels-json "${LEVELS_JSON}" \
  --schedule-tsv "${SCHEDULE_TSV}" \
  --quality-csv "${QUALITY_CSV}" \
  --baseline-root "${BASELINE_ROOT}" \
  --contention-root "${ROOT}" \
  --output-wide-csv "${ROOT}/all_levels_quality_forward_latency.csv" \
  --output-controller-csv "${ROOT}/controller_remaining_latency_table.csv" \
  --output-json "${ROOT}/contention_profile_summary.json" \
  --allow-partial

echo
echo "[RESULT] ${ROOT}/all_levels_quality_forward_latency.csv"
echo "[RESULT] ${ROOT}/controller_remaining_latency_table.csv"
echo "[RESULT] ${ROOT}/contention_profile_summary.json"

