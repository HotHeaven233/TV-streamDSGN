#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

FREQUENCIES="${1:-35,40,45,50}"
WARMUP="${2:-80}"
MAX_FRAMES="${3:-0}"
TRACE_SEED="${4:-20260903}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

LEVELS_JSON="outputs/elastic_bev/${EXP_NAME}/contention_calibration_v6/smooth_microchain_priority/contention_levels.json"

H2_CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-mtd_h2.yaml"
H3_CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-mtd_h3.yaml"

H2_TAG="$(basename "${H2_CFG}" .yaml)"
H3_TAG="$(basename "${H3_CFG}" .yaml)"

H2_CKPT="outputs/stream_kitti_models/${H2_TAG}.mtd_head_only/ckpt/checkpoint_epoch_5.pth"
H3_CKPT="outputs/stream_kitti_models/${H3_TAG}.mtd_head_only/ckpt/checkpoint_epoch_5.pth"

EVAL="tools/eval_transtreaming_three_head.py"

for f in \
    "${FULL_CFG}" \
    "${FULL_CKPT}" \
    "${H2_CKPT}" \
    "${H3_CKPT}" \
    "${LEVELS_JSON}" \
    "${EVAL}" \
    "tools/transtreaming_three_head_runtime.py"
do
    if [ ! -f "${f}" ]; then
        echo "[ERROR] missing: ${f}"
        exit 2
    fi
done

if [ "${MAX_FRAMES}" -eq 0 ]; then
    RUN_KIND="formal_streaming_no_load"
else
    RUN_KIND="smoke_streaming_no_load_max${MAX_FRAMES}"
fi

IFS=',' read -r -a HZ_ARRAY <<< "${FREQUENCIES}"

for raw_hz in "${HZ_ARRAY[@]}"
do
    HZ="$(echo "${raw_hz}" | xargs)"

    OUT_DIR="outputs/transtreaming_three_head/${RUN_KIND}/${HZ}Hz_forward_only/L0"

    rm -rf "${OUT_DIR}"
    mkdir -p "${OUT_DIR}"

    echo
    echo "===================================================================================================="
    echo "TRANSTREAMING-STYLE | ${HZ} Hz | L0"
    echo "===================================================================================================="

    python "${EVAL}" \
        --cfg "${FULL_CFG}" \
        --ckpt "${FULL_CKPT}" \
        --h2_ckpt "${H2_CKPT}" \
        --h3_ckpt "${H3_CKPT}" \
        --levels_json "${LEVELS_JSON}" \
        --pressure_level L0 \
        --pressure_fraction 0.0 \
        --trace_seed "${TRACE_SEED}" \
        --input_hz "${HZ}" \
        --runtime_warmup_frames "${WARMUP}" \
        --max_frames "${MAX_FRAMES}" \
        --planner_window 5 \
        --workers 0 \
        --seed 1024 \
        --output_dir "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/run.log"
done
