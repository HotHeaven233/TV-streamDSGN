#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source scripts/stream_exp/00_env.sh

banner "STEP 06 | Train Transtreaming-style 3D baseline | 15 epochs"

require_file "${FULL_CKPT}"

CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-transtreaming_tat_v2_15ep.yaml"
EXP="transtreaming_tat_v2_15ep"
TAG="$(basename "${CFG}" .yaml)"
OUT="outputs/stream_kitti_models/${TAG}.${EXP}"

require_file "${CFG}"
require_file pcdet/models/fusion_module/transtreaming_bev_tat.py
require_file pcdet/models/detectors_stream/transtreaming_stream_v2.py

# Do not silently mix a new reproduction with stale Transtreaming checkpoints.
if compgen -G "${OUT}/ckpt/checkpoint_epoch_*.pth" >/dev/null; then
    echo "[ERROR] Existing Transtreaming checkpoints found:"
    echo "        ${OUT}/ckpt"
    echo "Remove them or explicitly archive the previous run before reproducing."
    exit 4
fi

python tools/train.py \
    --cfg_file "${CFG}" \
    --pretrained_model "${FULL_CKPT}" \
    --fix_random_seed \
    --workers 4 \
    --ckpt_save_interval 1 \
    --max_ckpt_save_num 15 \
    --num_epochs_to_eval 15 \
    --exp_name "${EXP}"

# Paper baseline is frozen to the validation-selected checkpoint epoch 13.
BEST="${OUT}/ckpt/checkpoint_epoch_13.pth"
require_file "${BEST}"

# Re-run ordinary offline H1 evaluation as a checkpoint identity sanity check.
python tools/test.py \
    --cfg_file "${CFG}" \
    --ckpt "${BEST}" \
    --batch_size 1 \
    --workers 4 \
    --exp_name "${EXP}" \
    --eval_tag final_epoch13_check

echo "[PASS] Transtreaming paper checkpoint = ${BEST}"
echo "[INFO] Paper selection: validation best epoch = 13"
