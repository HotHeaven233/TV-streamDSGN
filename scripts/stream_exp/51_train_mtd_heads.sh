#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

source "${SCRIPT_DIR}/00_env.sh"

EPOCHS="${1:-5}"
WORKERS="${2:-4}"
LR="${3:-0.0002}"

BUILDER="tools/build_mtd_train_configs.py"
SMOKE="tools/smoke_mtd_head_training.py"

H2_CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl-mtd_h2.yaml"

H3_CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl-mtd_h3.yaml"

H2_EXP="mtd_h2_head_only"
H3_EXP="mtd_h3_head_only"

H2_TAG="$(
    basename "${H2_CFG}" .yaml
)"

H3_TAG="$(
    basename "${H3_CFG}" .yaml
)"

H2_CKPT="outputs/stream_kitti_models/${H2_TAG}.${H2_EXP}/ckpt/checkpoint_epoch_${EPOCHS}.pth"

H3_CKPT="outputs/stream_kitti_models/${H3_TAG}.${H3_EXP}/ckpt/checkpoint_epoch_${EPOCHS}.pth"

for f in \
    "${ORIGINAL_CFG}" \
    "${ORIGINAL_CKPT}" \
    "${BUILDER}" \
    "${SMOKE}"
do
    require_file "${f}"
done

if \
    [ "${ORIGINAL_CFG}" = "${FULL_CFG}" ] || \
    [ "${ORIGINAL_CKPT}" = "${FULL_CKPT}" ]
then
    echo "[ERROR] ORIGINAL_* equals FULL_*"
    exit 20
fi

python "${BUILDER}" \
    --base_cfg "${ORIGINAL_CFG}" \
    --h2_out "${H2_CFG}" \
    --h3_out "${H3_CFG}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}"

python -m py_compile \
    tools/train.py \
    tools/train_utils/train_utils.py \
    pcdet/models/detectors_stream/stream.py \
    pcdet/datasets/kitti_streaming/stereo_kitti_streaming.py \
    "${BUILDER}" \
    "${SMOKE}"

banner "MTD TRAINING SMOKE | h=2"

python "${SMOKE}" \
    --cfg "${H2_CFG}" \
    --pretrained "${ORIGINAL_CKPT}"

banner "MTD TRAINING SMOKE | h=3"

python "${SMOKE}" \
    --cfg "${H3_CFG}" \
    --pretrained "${ORIGINAL_CKPT}"

train_one () {
    local horizon="$1"
    local cfg_file="$2"
    local exp="$3"
    local ckpt="$4"

    if [ -f "${ckpt}" ]; then
        echo "[SKIP] checkpoint exists:"
        echo "       ${ckpt}"
        return
    fi

    banner \
        "TRAIN MTD h=${horizon} | Original frozen"

    python tools/train.py \
        --cfg_file "${cfg_file}" \
        --exp_name "${exp}" \
        --pretrained_model "${ORIGINAL_CKPT}" \
        --train_mtd_head_only \
        --epochs "${EPOCHS}" \
        --batch_size 1 \
        --workers "${WORKERS}" \
        --fix_random_seed \
        --ckpt_save_interval 1 \
        --max_ckpt_save_num 2

    require_file \
        "${ckpt}"
}

train_one \
    2 \
    "${H2_CFG}" \
    "${H2_EXP}" \
    "${H2_CKPT}"

train_one \
    3 \
    "${H3_CFG}" \
    "${H3_EXP}" \
    "${H3_CKPT}"

echo
echo "============================================================"
echo "MTD TRAINING COMPLETE"
echo "h=2: ${H2_CKPT}"
echo "h=3: ${H3_CKPT}"
echo "============================================================"

# TRAIN_MTD_HEADS_EOF
