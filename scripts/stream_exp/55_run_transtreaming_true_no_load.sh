#!/usr/bin/env bash
set -euo pipefail

cd /data/jhb/workspace/streamDSGN
source scripts/stream_exp/00_env.sh

FREQUENCIES="${1:-35,40,45,50}"
WARMUP="${2:-80}"
MAX_FRAMES="${3:-0}"
TRACE_SEED="${4:-20260903}"

TS_CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-transtreaming_tat.yaml"

TS_TAG="$(basename "${TS_CFG}" .yaml)"

TS_CKPT="outputs/stream_kitti_models/${TS_TAG}.transtreaming_tat/ckpt/checkpoint_epoch_5.pth"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

LEVELS_JSON="outputs/elastic_bev/${EXP_NAME}/contention_calibration_v6/smooth_microchain_priority/contention_levels.json"

EVAL="tools/eval_transtreaming_true.py"

for f in \
    "${TS_CFG}" \
    "${TS_CKPT}" \
    "${LEVELS_JSON}" \
    "${EVAL}" \
    "tools/transtreaming_adaptive_runtime.py"
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

    OUT_DIR="outputs/transtreaming_true/${RUN_KIND}/${HZ}Hz_forward_only/L0"

    rm -rf "${OUT_DIR}"
    mkdir -p "${OUT_DIR}"

    echo
    echo "===================================================================================================="
    echo "TRUE TRANSTREAMING | ${HZ} Hz | L0"
    echo "===================================================================================================="

    python "${EVAL}" \
        --cfg "${TS_CFG}" \
        --ckpt "${TS_CKPT}" \
        --levels_json "${LEVELS_JSON}" \
        --pressure_level L0 \
        --pressure_fraction 0.0 \
        --trace_seed "${TRACE_SEED}" \
        --input_hz "${HZ}" \
        --runtime_warmup_frames "${WARMUP}" \
        --max_frames "${MAX_FRAMES}" \
        --past_length 3 \
        --future_length 4 \
        --max_future 8 \
        --planner_window 5 \
        --workers 0 \
        --seed 1024 \
        --output_dir "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/run.log"

done
