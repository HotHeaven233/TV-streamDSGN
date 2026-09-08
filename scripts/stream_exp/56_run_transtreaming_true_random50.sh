#!/usr/bin/env bash
set -euo pipefail

cd /data/jhb/workspace/streamDSGN
source scripts/stream_exp/00_env.sh

HZ="${1:-35}"
WARMUP="${2:-80}"
PRESSURE_LEVELS="${3:-L1,L2,L3,L4}"
MAX_FRAMES="${4:-0}"
TRACE_SEED="${5:-20260903}"
PRESSURE_FRACTION="${6:-0.5}"

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
    RUN_KIND="formal_streaming_random50"
else
    RUN_KIND="smoke_streaming_random50_max${MAX_FRAMES}"
fi

OUT_ROOT="outputs/transtreaming_true/${RUN_KIND}/${HZ}Hz_forward_only_seed${TRACE_SEED}"

mkdir -p "${OUT_ROOT}"

IFS=',' read -r -a LEVEL_ARRAY <<< "${PRESSURE_LEVELS}"

for raw_level in "${LEVEL_ARRAY[@]}"
do
    PRESSURE="$(echo "${raw_level}" | xargs)"

    case "${PRESSURE}" in
        L1|L2|L3|L4)
            ;;
        *)
            echo "[ERROR] invalid level: ${PRESSURE}"
            exit 3
            ;;
    esac

    OUT_DIR="${OUT_ROOT}/L0_${PRESSURE}_p${PRESSURE_FRACTION}"

    rm -rf "${OUT_DIR}"
    mkdir -p "${OUT_DIR}"

    echo
    echo "===================================================================================================="
    echo "TRUE TRANSTREAMING | ${HZ} Hz | L0 + ${PRESSURE}"
    echo "===================================================================================================="

    python "${EVAL}" \
        --cfg "${TS_CFG}" \
        --ckpt "${TS_CKPT}" \
        --levels_json "${LEVELS_JSON}" \
        --pressure_level "${PRESSURE}" \
        --pressure_fraction "${PRESSURE_FRACTION}" \
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

    TRACE="${OUT_DIR}/contention_trace.csv"

    if [ "${MAX_FRAMES}" -eq 0 ]; then

        TV_TRACE="outputs/elastic_bev/${EXP_NAME}/formal_streaming_random50/${HZ}Hz_forward_only_seed${TRACE_SEED}/L0_${PRESSURE}_p${PRESSURE_FRACTION}/contention_trace.csv"

        ORIGINAL_TRACE="outputs/original_streamdsgn/formal_streaming_random50/${HZ}Hz_forward_only_seed${TRACE_SEED}/L0_${PRESSURE}_p${PRESSURE_FRACTION}/contention_trace.csv"

        MTD_TRACE="outputs/mtd_three_head/formal_streaming_random50/${HZ}Hz_forward_only_seed${TRACE_SEED}/L0_${PRESSURE}_p${PRESSURE_FRACTION}/contention_trace.csv"

        for REF in \
            "${TV_TRACE}" \
            "${ORIGINAL_TRACE}" \
            "${MTD_TRACE}"
        do
            if [ -f "${REF}" ]; then
                if cmp -s "${REF}" "${TRACE}"; then
                    echo "[TRACE PASS] ${REF}"
                else
                    echo "[ERROR] trace mismatch:"
                    echo "  REF = ${REF}"
                    echo "  TS  = ${TRACE}"
                    exit 10
                fi
            fi
        done
    else
        echo "[TRACE CHECK] smoke run; formal trace comparison skipped."
    fi

    python - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])

s = json.loads(
    (out / "summary.json").read_text()
)

q = s[
    "stream_sap_3d_moderate_R40"
]

f = s[
    "forward_latency"
]

print()
print("=" * 105)
print("TRUE TRANSTREAMING AUDIT")
print("=" * 105)

print(
    "PF histogram       :",
    s["proposal_histogram"],
)

print(
    "dispatch horizons  :",
    s["dispatch_horizon_histogram"],
)

print(
    "executed outputs   :",
    s["executed_future_outputs"],
)

print(
    "skipped outputs    :",
    s["skipped_future_outputs"],
)

print(
    "outputs/job        :",
    s["mean_executed_outputs_per_job"],
)

print(
    "drop rate          :",
    s["drop_rate"],
)

print(
    "deadline miss rate :",
    s["deadline_miss_rate"],
)

print(
    "p50/p90/p99       :",
    f["p50_ms"],
    f["p90_ms"],
    f["p99_ms"],
)

print(
    "Car/Ped/Cyc/Macro : "
    f"{q['Car']:.4f}/"
    f"{q['Pedestrian']:.4f}/"
    f"{q['Cyclist']:.4f}/"
    f"{q['Macro']:.4f}"
)

print("=" * 105)
PY

done
