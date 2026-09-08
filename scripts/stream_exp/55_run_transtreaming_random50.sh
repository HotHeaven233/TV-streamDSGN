#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

HZ="${1:-35}"
WARMUP="${2:-80}"
PRESSURE_LEVELS="${3:-L1,L2,L3,L4}"
MAX_FRAMES="${4:-0}"
TRACE_SEED="${5:-20260903}"
PRESSURE_FRACTION="${6:-0.5}"

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
    RUN_KIND="formal_streaming_random50"
else
    RUN_KIND="smoke_streaming_random50_max${MAX_FRAMES}"
fi

OUT_ROOT="outputs/transtreaming_three_head/${RUN_KIND}/${HZ}Hz_forward_only_seed${TRACE_SEED}"

mkdir -p "${OUT_ROOT}"

IFS=',' read -r -a LEVEL_ARRAY <<< "${PRESSURE_LEVELS}"

for raw in "${LEVEL_ARRAY[@]}"
do
    pressure="$(echo "${raw}" | xargs)"

    case "${pressure}" in
        L1|L2|L3|L4)
            ;;
        *)
            echo "[ERROR] invalid pressure level: ${pressure}"
            exit 3
            ;;
    esac

    OUT_DIR="${OUT_ROOT}/L0_${pressure}_p${PRESSURE_FRACTION}"

    rm -rf "${OUT_DIR}"
    mkdir -p "${OUT_DIR}"

    echo
    echo "===================================================================================================="
    echo "TRANSTREAMING-STYLE | ${HZ} Hz | L0 + ${pressure}"
    echo "===================================================================================================="

    python "${EVAL}" \
        --cfg "${FULL_CFG}" \
        --ckpt "${FULL_CKPT}" \
        --h2_ckpt "${H2_CKPT}" \
        --h3_ckpt "${H3_CKPT}" \
        --levels_json "${LEVELS_JSON}" \
        --pressure_level "${pressure}" \
        --pressure_fraction "${PRESSURE_FRACTION}" \
        --trace_seed "${TRACE_SEED}" \
        --input_hz "${HZ}" \
        --runtime_warmup_frames "${WARMUP}" \
        --max_frames "${MAX_FRAMES}" \
        --planner_window 5 \
        --workers 0 \
        --seed 1024 \
        --output_dir "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/run.log"

    TRACE="${OUT_DIR}/contention_trace.csv"

    if [ "${MAX_FRAMES}" -eq 0 ]; then

        TV_TRACE="outputs/elastic_bev/${EXP_NAME}/formal_streaming_random50/${HZ}Hz_forward_only_seed${TRACE_SEED}/L0_${pressure}_p${PRESSURE_FRACTION}/contention_trace.csv"

        if [ -f "${TV_TRACE}" ]; then
            if cmp -s "${TV_TRACE}" "${TRACE}"; then
                echo "[TRACE CHECK PASS] Transtreaming == TV"
            else
                echo "[ERROR] Transtreaming trace differs from TV"
                exit 10
            fi
        fi

        ORIGINAL_TRACE="outputs/original_streamdsgn/formal_streaming_random50/${HZ}Hz_forward_only_seed${TRACE_SEED}/L0_${pressure}_p${PRESSURE_FRACTION}/contention_trace.csv"

        if [ -f "${ORIGINAL_TRACE}" ]; then
            if cmp -s "${ORIGINAL_TRACE}" "${TRACE}"; then
                echo "[TRACE CHECK PASS] Transtreaming == Original"
            else
                echo "[ERROR] Transtreaming trace differs from Original"
                exit 11
            fi
        fi
    else
        echo "[TRACE CHECK] smoke run; formal cross-method cmp skipped"
    fi

    python - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "summary.json"
s = json.loads(p.read_text())

q = s["stream_sap_3d_moderate_R40"]
f = s["forward_latency"]

print()
print("=" * 100)
print("TRANSTREAMING AUDIT")
print("=" * 100)
print("proposal histogram :", s["proposal_histogram"])
print("head executions    :", s["head_execution_counts"])
print("mean heads/job     :", s["mean_future_heads_per_job"])
print("drop rate          :", s["drop_rate"])
print("deadline miss rate :", s["deadline_miss_rate"])
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
print("=" * 100)
PY

done
