#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

require_file "${LIGHT_CFG}"
require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"
require_file "tools/train_light_distill.py"

OUT_DIR="outputs/stream_kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-light_v1.${LIGHT_EXP_NAME}"

# Avoid accidentally overwriting a previous Light experiment.
if compgen -G "${OUT_DIR}/ckpt/checkpoint_epoch_*.pth" > /dev/null; then
    echo "[ERROR] Existing Light checkpoints found:"
    echo "        ${OUT_DIR}/ckpt"
    echo
    echo "Use a new experiment name, for example:"
    echo
    echo "LIGHT_EXP_NAME=light_distill_v1_retrain10_b $0"
    exit 1
fi

banner "Train Light-v1 | Frozen Full-e5 Teacher | 10 epochs"

echo "[TRAIN] Light cfg  : ${LIGHT_CFG}"
echo "[TRAIN] Full cfg   : ${FULL_CFG}"
echo "[TRAIN] Full ckpt  : ${FULL_CKPT}"
echo "[TRAIN] experiment : ${LIGHT_EXP_NAME}"
echo "[TRAIN] epochs     : 10"
echo "[TRAIN] bev_weight : 1.0"

python tools/train_light_distill.py \
    --cfg_file "${LIGHT_CFG}" \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --epochs 10 \
    --bev_weight 1.0 \
    --workers 4 \
    --fix_random_seed \
    --ckpt_save_interval 1 \
    --max_ckpt_save_num 10 \
    --exp_name "${LIGHT_EXP_NAME}"
