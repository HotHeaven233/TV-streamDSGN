#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-20}"
N="${2:-100}"
HZ="${3:-30}"
WARMUP="${4:-30}"
FRAMES="${5:-300}"
LEVELS="${6:-L0,L1,L2,L3,L4}"
GUARD="${7:-0.25}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

RAW_CKPT="outputs/elastic_bev/${EXP_NAME}/ckpt/checkpoint_epoch_${EPOCH}.pth"

BANK="outputs/elastic_bev/${EXP_NAME}/bn_bank/causal_prefix_bank_e${EPOCH}_n${N}.pth"

PROFILE_ROOT="outputs/elastic_bev/${EXP_NAME}/all84_contention_profile_v6/e${EPOCH}_n${N}"

CTRL_CSV="${PROFILE_ROOT}/controller_remaining_latency_table.csv"

LEVELS_JSON="outputs/elastic_bev/${EXP_NAME}/contention_calibration_v6/smooth_microchain_priority/contention_levels.json"

OUT_DIR="outputs/elastic_bev/${EXP_NAME}/post_nms_profile/e${EPOCH}_n${N}_${HZ}Hz"

mkdir -p "${OUT_DIR}"

for f in \
  "${FULL_CFG}" \
  "${FULL_CKPT}" \
  "${RAW_CKPT}" \
  "${BANK}" \
  "${CTRL_CSV}" \
  "${LEVELS_JSON}" \
  "tools/profile_tv_stream3d_post_nms_p99.py" \
  "tools/test_tv_stream3d_online_forward.py" \
  "tools/tv_stream3d_controller.py" \
  "tools/tv_stream3d_causal_fused_runtime.py" \
  "tools/smooth_cuda_contention.py"
do
  if [ ! -f "${f}" ]; then
    echo "[ERROR] missing: ${f}"
    exit 2
  fi
done

echo "======================================================================"
echo "TV-Stream3D POST/NMS PROFILE"
echo "======================================================================"
echo "epoch       : ${EPOCH}"
echo "bank n      : ${N}"
echo "input Hz    : ${HZ}"
echo "warmup      : ${WARMUP}"
echo "frames      : ${FRAMES}"
echo "levels      : ${LEVELS}"
echo "guard       : ${GUARD} ms/boundary"
echo "elastic ckpt: ${RAW_CKPT}"
echo "BN bank     : ${BANK}"
echo "controller  : ${CTRL_CSV}"
echo "levels json : ${LEVELS_JSON}"
echo "output      : ${OUT_DIR}"
echo "======================================================================"

python tools/profile_tv_stream3d_post_nms_p99.py \
  --full_cfg "${FULL_CFG}" \
  --full_ckpt "${FULL_CKPT}" \
  --elastic_ckpt "${RAW_CKPT}" \
  --prefix_bn_bank "${BANK}" \
  --controller_csv "${CTRL_CSV}" \
  --levels_json "${LEVELS_JSON}" \
  --input_hz "${HZ}" \
  --warmup_frames "${WARMUP}" \
  --frames "${FRAMES}" \
  --workers 0 \
  --control_guard_per_boundary_ms "${GUARD}" \
  --levels "${LEVELS}" \
  --output_dir "${OUT_DIR}"

echo
echo "[RESULT] ${OUT_DIR}/post_nms_p99_summary.json"
