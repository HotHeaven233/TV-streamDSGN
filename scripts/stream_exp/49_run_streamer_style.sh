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
EWMA_ALPHA="${7:-0.5}"
FORECAST_MODE="${8:-kf}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"
LEVELS_JSON="outputs/elastic_bev/${EXP_NAME}/contention_calibration_v6/smooth_microchain_priority/contention_levels.json"
EVAL="tools/eval_streamer_style_streamdsgn.py"

for f in \
    "${ORIGINAL_CFG}" \
    "${ORIGINAL_CKPT}" \
    "${LEVELS_JSON}" \
    "${EVAL}" \
    "tools/streamer_style_runtime.py" \
    "tools/eval_original_streamdsgn_30hz_random50.py" \
    "tools/eval_tv_stream3d_30hz_random50.py"
do
    require_file "${f}"
done

if [ "${ORIGINAL_CFG}" = "${FULL_CFG}" ] || [ "${ORIGINAL_CKPT}" = "${FULL_CKPT}" ]; then
    echo "[ERROR] ORIGINAL_* unexpectedly equals FULL_*"
    exit 20
fi

case "${FORECAST_MODE}" in
    kf|copy) ;;
    *)
        echo "[ERROR] invalid forecast mode: ${FORECAST_MODE}"
        exit 4
        ;;
esac

IFS=',' read -r -a LEVEL_ARRAY <<< "${PRESSURE_LEVELS}"

for raw in "${LEVEL_ARRAY[@]}"
do
    pressure="$(echo "${raw}" | xargs)"

    case "${pressure}" in
        L0|L1|L2|L3|L4) ;;
        *)
            echo "[ERROR] invalid pressure level: ${pressure}"
            exit 3
            ;;
    esac

    if [ "${pressure}" = "L0" ]; then
        if [ "${PRESSURE_FRACTION}" != "0" ] && [ "${PRESSURE_FRACTION}" != "0.0" ]; then
            echo "[ERROR] L0 requires pressure fraction 0.0"
            exit 5
        fi
        ESTIMATOR="static_median"
        BASE_OUT="outputs/streamer_style_streamdsgn/formal_streaming_no_load/${HZ}Hz_forward_only/L0"
    else
        ESTIMATOR="ewma"
        BASE_OUT="outputs/streamer_style_streamdsgn/formal_streaming_random50/${HZ}Hz_forward_only_seed${TRACE_SEED}/L0_${pressure}_p${PRESSURE_FRACTION}"
    fi

    if [ "${FORECAST_MODE}" = "kf" ]; then
        OUT_DIR="${BASE_OUT}"
    else
        OUT_DIR="${BASE_OUT}_${FORECAST_MODE}"
    fi
    mkdir -p "${OUT_DIR}"

    banner "STREAMER-STYLE StreamDSGN | ${HZ} Hz | ${pressure} | ${FORECAST_MODE}"

    python "${EVAL}" \
        --cfg "${ORIGINAL_CFG}" \
        --ckpt "${ORIGINAL_CKPT}" \
        --levels_json "${LEVELS_JSON}" \
        --pressure_level "${pressure}" \
        --pressure_fraction "${PRESSURE_FRACTION}" \
        --trace_seed "${TRACE_SEED}" \
        --input_hz "${HZ}" \
        --runtime_warmup_frames "${WARMUP}" \
        --max_frames "${MAX_FRAMES}" \
        --workers 0 \
        --seed 1024 \
        --runtime_estimator "${ESTIMATOR}" \
        --ewma_alpha "${EWMA_ALPHA}" \
        --forecast_mode "${FORECAST_MODE}" \
        --output_dir "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/run.log"

    # Formal random-contention runs must have exactly the same sensor trace as TV.
    # Smoke runs may intentionally use fewer frames, so unequal trace length only warns.
    if [ "${pressure}" != "L0" ]; then
        TV_TRACE="outputs/elastic_bev/${EXP_NAME}/formal_streaming_random50/${HZ}Hz_forward_only_seed${TRACE_SEED}/L0_${pressure}_p${PRESSURE_FRACTION}/contention_trace.csv"
        OUR_TRACE="${OUT_DIR}/contention_trace.csv"

        if [ -f "${TV_TRACE}" ]; then
            tv_lines="$(wc -l < "${TV_TRACE}")"
            our_lines="$(wc -l < "${OUR_TRACE}")"

            if [ "${tv_lines}" = "${our_lines}" ]; then
                if cmp -s "${TV_TRACE}" "${OUR_TRACE}"; then
                    echo "[TRACE CHECK PASS] Streamer-style and TV traces are byte-identical."
                else
                    echo "[ERROR] equal-length contention traces differ from TV"
                    exit 10
                fi
            elif [ "${MAX_FRAMES}" = "0" ]; then
                echo "[ERROR] formal trace length differs: TV=${tv_lines}, Streamer=${our_lines}"
                exit 11
            else
                echo "[WARN] smoke trace length differs from existing TV trace; comparison skipped."
            fi
        else
            echo "[WARN] TV trace not found; comparison skipped: ${TV_TRACE}"
        fi
    fi
done

# STREAMER_STYLE_RUN_EOF
