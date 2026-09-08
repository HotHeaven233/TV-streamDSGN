#!/usr/bin/env bash
set -euo pipefail

cd /data/jhb/workspace/streamDSGN
source scripts/stream_exp/00_env.sh

TS_CFG="${1:-configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-transtreaming_tat.yaml}"

if [ ! -f "${TS_CFG}" ]; then
    echo "[ERROR] missing cfg: ${TS_CFG}"
    exit 2
fi

if [ ! -f "${FULL_CKPT}" ]; then
    echo "[ERROR] missing FULL_CKPT: ${FULL_CKPT}"
    exit 2
fi

echo "================================================================================"
echo "Transtreaming TAT retraining -- H8 supervision bug fixed"
echo "================================================================================"
echo "CFG  : ${TS_CFG}"
echo "INIT : ${FULL_CKPT}"
echo
echo "Future supervision:"
echo "  H1 -> next"
echo "  H2 -> next2"
echo "  H4 -> next4"
echo "  H8 -> next8"
echo
echo "IMPORTANT:"
echo "  starting from FULL_CKPT, NOT old Transtreaming checkpoint"
echo "================================================================================"

python tools/train.py \
  --cfg_file "${TS_CFG}" \
  --pretrained_model "${FULL_CKPT}" \
  --fix_random_seed \
  --workers 4 \
  --exp_name transtreaming_tat_fix_h8
