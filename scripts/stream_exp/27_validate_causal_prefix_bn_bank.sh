#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-20}"
N="${2:-100}"
HZ="${3:-10}"
EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

BANK="outputs/elastic_bev/${EXP_NAME}/bn_bank/causal_prefix_bank_e${EPOCH}_n${N}.pth"
RAW_CKPT="outputs/elastic_bev/${EXP_NAME}/ckpt/checkpoint_epoch_${EPOCH}.pth"

require_file "${BANK}"
require_file "${RAW_CKPT}"

VERIFY_DIR="outputs/v4_prefix_bank_verify/e${EPOCH}_n${N}"
mkdir -p "${VERIFY_DIR}"

python tools/verify_causal_prefix_bn_bank.py \
  --bank "${BANK}" \
  --expect_batches "${N}" \
  --json_out "${VERIFY_DIR}/bank_structure.json"

# These six profiles have dedicated-calibration reference results from the
# diagnostic experiments, so they are the best validation set for the bank.
PROFILES=(
  "0.75,0.75,0.75,0.75,0.75,0.75"
  "0.5,0.5,0.5,0.5,0.5,0.5"
  "0.25,0.25,0.25,0.25,0.25,0.25"
  "1.0,1.0,0.75,0.5,0.5,0.25"
  "1.0,1.0,1.0,0.75,0.5,0.25"
  "1.0,0.75,0.5,0.5,0.25,0.25"
)

for SCHEDULE in "${PROFILES[@]}"; do
  TAG="$(echo "${SCHEDULE}" | sed 's/1\.0/100/g; s/0\.75/075/g; s/0\.5/050/g; s/0\.25/025/g; s/,/_/g')"
  PROFILE_CKPT="outputs/elastic_bev/${EXP_NAME}/prefix_profiles/e${EPOCH}_n${N}_${TAG}.pth"
  OUT_DIR="${VERIFY_DIR}/${TAG}"
  mkdir -p "${OUT_DIR}"

  echo
  echo "========================================================================"
  echo "VERIFY PROFILE ${SCHEDULE}"
  echo "========================================================================"

  "${SCRIPT_DIR}/24_materialize_prefix_profile.sh" \
    "${EPOCH}" "${N}" "${SCHEDULE}"

  python tools/test_elastic_stream_v4_bn.py \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --elastic_ckpt "${PROFILE_CKPT}" \
    --input_hz "${HZ}" \
    --warmup 20 \
    --mode elastic_fixed \
    --fixed_schedule "${SCHEDULE}" \
    --output_dir "${OUT_DIR}" \
    2>&1 | tee "${OUT_DIR}/console.log"
done

echo
echo "[DONE] bank validation outputs:"
echo "  ${VERIFY_DIR}"

