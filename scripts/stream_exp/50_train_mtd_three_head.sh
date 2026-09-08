#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

H2_CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-mtd_h2.yaml"
H3_CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-mtd_h3.yaml"

require_file "${FULL_CKPT}"
require_file "${H2_CFG}"
require_file "${H3_CFG}"

banner "TRUE MTD | TRAIN H2 = next2 | DENSE HEAD ONLY"

python tools/train.py \
    --cfg_file "${H2_CFG}" \
    --pretrained_model "${FULL_CKPT}" \
    --train_mtd_head_only \
    --fix_random_seed \
    --workers 4 \
    --exp_name mtd_head_only

banner "TRUE MTD | TRAIN H3 = next3 | DENSE HEAD ONLY"

python tools/train.py \
    --cfg_file "${H3_CFG}" \
    --pretrained_model "${FULL_CKPT}" \
    --train_mtd_head_only \
    --fix_random_seed \
    --workers 4 \
    --exp_name mtd_head_only

H2_TAG="$(basename "${H2_CFG}" .yaml)"
H3_TAG="$(basename "${H3_CFG}" .yaml)"

H2_CKPT="outputs/stream_kitti_models/${H2_TAG}.mtd_head_only/ckpt/checkpoint_epoch_5.pth"
H3_CKPT="outputs/stream_kitti_models/${H3_TAG}.mtd_head_only/ckpt/checkpoint_epoch_5.pth"

require_file "${H2_CKPT}"
require_file "${H3_CKPT}"

echo
echo "======================================================================"
echo "MTD TRAINING COMPLETE"
echo "======================================================================"
echo "H1 = ${FULL_CKPT}"
echo "H2 = ${H2_CKPT}"
echo "H3 = ${H3_CKPT}"
echo "======================================================================"
