#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

H2_CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-mtd_h2.yaml"
H3_CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-mtd_h3.yaml"

H2_TAG="$(basename "${H2_CFG}" .yaml)"
H3_TAG="$(basename "${H3_CFG}" .yaml)"

H2_CKPT="outputs/stream_kitti_models/${H2_TAG}.mtd_head_only/ckpt/checkpoint_epoch_5.pth"
H3_CKPT="outputs/stream_kitti_models/${H3_TAG}.mtd_head_only/ckpt/checkpoint_epoch_5.pth"

for f in \
    "${FULL_CFG}" \
    "${FULL_CKPT}" \
    "${H2_CKPT}" \
    "${H3_CKPT}" \
    "tools/transtreaming_three_head_runtime.py" \
    "tools/eval_transtreaming_three_head.py"
do
    if [ ! -f "${f}" ]; then
        echo "[ERROR] missing: ${f}"
        exit 2
    fi
done

python -m py_compile \
    tools/transtreaming_three_head_runtime.py \
    tools/eval_transtreaming_three_head.py

echo
echo "======================================================================"
echo "TRANSTREAMING-STYLE BASELINE READY"
echo "======================================================================"
echo "shared cfg : ${FULL_CFG}"
echo "H1 next    : ${FULL_CKPT}"
echo "H2 next2   : ${H2_CKPT}"
echo "H3 next3   : ${H3_CKPT}"
echo "======================================================================"
