#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

require_file "${FULL_CFG}"
require_file "${ORIGINAL_CKPT}"

OUT_DIR="outputs/stream_kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual.${MH_RETRAIN_EXP_NAME}"

# Safety: do not silently overwrite an existing experiment.
if compgen -G "${OUT_DIR}/ckpt/checkpoint_epoch_*.pth" > /dev/null; then
    echo "[ERROR] Existing checkpoints found:"
    echo "        ${OUT_DIR}/ckpt"
    echo
    echo "Use another name, e.g."
    echo "MH_RETRAIN_EXP_NAME=mh3_residual_retrain2 $0"
    exit 1
fi

banner "Train Multi-History Residual Adapter | 5 epochs"

python tools/train.py \
    --cfg_file "${FULL_CFG}" \
    --pretrained_model "${ORIGINAL_CKPT}" \
    --train_mh_adapter_only \
    --fix_random_seed \
    --epochs 5 \
    --exp_name "${MH_RETRAIN_EXP_NAME}" \
    --ckpt_save_interval 1 \
    --max_ckpt_save_num 5
