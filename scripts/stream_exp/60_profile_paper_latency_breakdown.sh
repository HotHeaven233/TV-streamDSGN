#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-20}"
FRAMES="${2:-1000}"
WARMUP="${3:-100}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

ELASTIC_CKPT="outputs/elastic_bev/${EXP_NAME}/ckpt/checkpoint_epoch_${EPOCH}.pth"

OUT_ROOT="outputs/elastic_bev/${EXP_NAME}/paper_latency_breakdown"
OUT_DIR="${OUT_ROOT}/fullwidth_L0_e${EPOCH}_n${FRAMES}"

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"
require_file "${ELASTIC_CKPT}"
require_file "tools/profile_fixed_forward_components.py"
require_file "tools/profile_paper_latency_breakdown.py"

mkdir -p "${OUT_DIR}"

banner "PAPER | Full-width forward latency breakdown"

echo "[CONFIG] Full cfg     : ${FULL_CFG}"
echo "[CONFIG] Full ckpt    : ${FULL_CKPT}"
echo "[CONFIG] Elastic ckpt : ${ELASTIC_CKPT}"
echo "[CONFIG] Schedule     : 1,1,1,1,1,1"
echo "[CONFIG] Warmup       : ${WARMUP}"
echo "[CONFIG] Frames       : ${FRAMES}"
echo "[CONFIG] Contention   : L0 / none"
echo "[CONFIG] Output       : ${OUT_DIR}"

python tools/profile_paper_latency_breakdown.py \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --elastic_ckpt "${ELASTIC_CKPT}" \
    --schedule "1,1,1,1,1,1" \
    --warmup "${WARMUP}" \
    --frames "${FRAMES}" \
    --workers 0 \
    --seed 1024 \
    --output_dir "${OUT_DIR}" \
    2>&1 | tee "${OUT_DIR}/console.log"

echo
echo "======================================================================"
echo "PAPER TABLE"
echo "======================================================================"
cat "${OUT_DIR}/paper_latency_breakdown.csv"

echo
echo "======================================================================"
echo "LATEX TABLE"
echo "======================================================================"
cat "${OUT_DIR}/paper_latency_breakdown.tex"

echo
echo "[RESULT] raw     : ${OUT_DIR}/latency_breakdown_raw.csv"
echo "[RESULT] summary : ${OUT_DIR}/latency_breakdown_summary.json"
echo "[RESULT] table   : ${OUT_DIR}/paper_latency_breakdown.csv"
echo "[RESULT] latex   : ${OUT_DIR}/paper_latency_breakdown.tex"
