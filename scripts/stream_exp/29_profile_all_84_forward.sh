#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-20}"
N="${2:-100}"
FRAMES="${3:-120}"
WARMUP="${4:-20}"
START_ID="${5:-1}"
END_ID="${6:-84}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

RAW_CKPT="outputs/elastic_bev/${EXP_NAME}/ckpt/checkpoint_epoch_${EPOCH}.pth"
BANK="outputs/elastic_bev/${EXP_NAME}/bn_bank/causal_prefix_bank_e${EPOCH}_n${N}.pth"

SAP_ROOT="outputs/elastic_bev/${EXP_NAME}/all84_sap/e${EPOCH}_n${N}_10Hz"
SCHEDULE_TSV="${SAP_ROOT}/schedules.tsv"
QUALITY_CSV="${QUALITY_CSV:-${SAP_ROOT}/all84_sap_summary.csv}"

ROOT="outputs/elastic_bev/${EXP_NAME}/all84_forward_profile/e${EPOCH}_n${N}"
TMP_DIR="${ROOT}/_tmp"

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"
require_file "${RAW_CKPT}"
require_file "${BANK}"
require_file "${SCHEDULE_TSV}"
require_file "${QUALITY_CSV}"
require_file "tools/materialize_causal_prefix_bn_profile.py"
require_file "tools/profile_fixed_forward_components.py"
require_file "tools/aggregate_84_quality_forward.py"

mkdir -p "${ROOT}" "${TMP_DIR}"

echo
echo "======================================================================"
echo "ALL-84 FORWARD-ONLY COMPONENT PROFILING"
echo "epoch       : ${EPOCH}"
echo "BN bank N   : ${N}"
echo "frames      : ${FRAMES}"
echo "warmup      : ${WARMUP}"
echo "profile IDs : ${START_ID}..${END_ID}"
echo "timing      : CUDA events, one sync after forward_end"
echo "scope       : forward only; NO data/H2D; NO post-processing/NMS"
echo "result root : ${ROOT}"
echo "======================================================================"

tail -n +2 "${SCHEDULE_TSV}" | while IFS=$'\t' read -r ID SCHEDULE TAG RES2 RES3 RES4 FPN STEREO RPN; do
  ID_NUM=$((10#${ID}))
  if [ "${ID_NUM}" -lt "${START_ID}" ] || [ "${ID_NUM}" -gt "${END_ID}" ]; then
    continue
  fi

  ID3="$(printf '%03d' "${ID_NUM}")"
  PROFILE_ROOT="${ROOT}/${ID3}_${TAG}"
  SUMMARY="${PROFILE_ROOT}/forward_profile_summary.json"
  RAW="${PROFILE_ROOT}/forward_profile_raw.csv"

  if [ -f "${SUMMARY}" ]; then
    echo "[SKIP ${ID3}/084] ${SCHEDULE} -> already complete"
    continue
  fi

  mkdir -p "${PROFILE_ROOT}"
  TMP_CKPT="${TMP_DIR}/profile_${ID3}_${TAG}.pth"

  echo
  echo "========================================================================"
  echo "[PROFILE ${ID3}/084] ${SCHEDULE}"
  echo "========================================================================"

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
    --raw_csv "${RAW}" \
    --summary_json "${SUMMARY}" \
    2>&1 | tee "${PROFILE_ROOT}/console.log"
  RC=${PIPESTATUS[0]}
  set -e

  if [ "${RC}" -ne 0 ]; then
    echo "[ERROR] profile ${ID3} failed. Temporary checkpoint kept:"
    echo "        ${TMP_CKPT}"
    exit "${RC}"
  fi

  rm -f "${TMP_CKPT}"
  echo "[OK ${ID3}/084] ${SCHEDULE}"
done

python tools/aggregate_84_quality_forward.py \
  --schedule_tsv "${SCHEDULE_TSV}" \
  --profile_root "${ROOT}" \
  --quality_csv "${QUALITY_CSV}" \
  --output_csv "${ROOT}/all84_quality_forward_latency.csv" \
  --output_json "${ROOT}/all84_quality_forward_latency.json" \
  --allow_partial

echo
echo "[RESULT] ${ROOT}/all84_quality_forward_latency.csv"
echo "[RESULT] ${ROOT}/all84_quality_forward_latency.json"

