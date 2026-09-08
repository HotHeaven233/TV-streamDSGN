#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

H2_CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-mtd_h2.yaml"
H3_CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-mtd_h3.yaml"

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"

banner "BUILD TRUE MTD H2/H3 TRAINING CONFIGS"

python tools/build_mtd_train_configs.py \
    --base_cfg "${FULL_CFG}" \
    --h2_out "${H2_CFG}" \
    --h3_out "${H3_CFG}" \
    --epochs 5 \
    --lr 2e-4

banner "BUILD TRUE MTD EVALUATORS FROM FROZEN BASELINE EVALUATORS"

python tools/build_mtd_three_head_evaluator.py \
    --base tools/eval_original_streamdsgn_no_load.py \
    --out tools/eval_mtd_three_head_no_load.py

python tools/build_mtd_three_head_evaluator.py \
    --base tools/eval_original_streamdsgn_30hz_random50.py \
    --out tools/eval_mtd_three_head_random50.py

python -m py_compile \
    tools/mtd_three_head_runtime.py \
    tools/build_mtd_three_head_evaluator.py \
    tools/eval_mtd_three_head_no_load.py \
    tools/eval_mtd_three_head_random50.py

echo
echo "[OK] MTD files generated."
echo "H2 CFG = ${H2_CFG}"
echo "H3 CFG = ${H3_CFG}"
