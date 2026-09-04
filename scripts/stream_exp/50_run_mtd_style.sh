#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

source "${SCRIPT_DIR}/00_env.sh"

HZ="${1:-35}"

PRESSURE_LEVELS="${
    2:-L1,L2,L3,L4
}"

TRACE_SEED="${
    3:-20260903
}"

PRESSURE_FRACTION="${
    4:-0.5
}"

MATCH_THRESHOLD_M="${
    5:-10.0
}"

EVAL="tools/eval_mtd_style_streamdsgn.py"

# Always use the same L0 calibration point for DAM initialization.
CALIB="outputs/original_streamdsgn/formal_streaming_no_load/35Hz_forward_only/L0/summary.json"

EXP_NAME="${
    ELASTIC_EXP_NAME:-
    elastic_bev_v4_bn_from_k3
}"

for f in \
    "${ORIGINAL_CFG}" \
    "${EVAL}" \
    "tools/mtd_style_runtime.py" \
    "${CALIB}"
do
    require_file "${f}"
done

if [ "${ORIGINAL_CFG}" = "${FULL_CFG}" ]; then
    echo "[ERROR] ORIGINAL_CFG unexpectedly equals FULL_CFG"
    exit 20
fi

IFS=',' read -r -a LEVEL_ARRAY <<< "${PRESSURE_LEVELS}"

for raw in "${LEVEL_ARRAY[@]}"
do
    level="$(
        echo "${raw}" |
        xargs
    )"

    case "${level}" in
        L0|L1|L2|L3|L4)
            ;;
        *)
            echo "[ERROR] invalid level: ${level}"
            exit 3
            ;;
    esac

    if [ "${level}" = "L0" ]; then
        if \
            [ "${PRESSURE_FRACTION}" != "0" ] && \
            [ "${PRESSURE_FRACTION}" != "0.0" ]
        then
            echo "[ERROR] L0 requires pressure fraction 0.0"
            exit 4
        fi

        SRC="outputs/original_streamdsgn/formal_streaming_no_load/${HZ}Hz_forward_only/L0"

        OUT="outputs/mtd_style_streamdsgn/formal_streaming_no_load/${HZ}Hz_forward_only/L0"

    else
        SRC="outputs/original_streamdsgn/formal_streaming_random50/${HZ}Hz_forward_only_seed${TRACE_SEED}/L0_${level}_p${PRESSURE_FRACTION}"

        OUT="outputs/mtd_style_streamdsgn/formal_streaming_random50/${HZ}Hz_forward_only_seed${TRACE_SEED}/L0_${level}_p${PRESSURE_FRACTION}"
    fi

    require_file \
        "${SRC}/summary.json"

    require_file \
        "${SRC}/prediction_events.pkl"

    require_file \
        "${SRC}/frame_timeline.csv"

    mkdir -p \
        "${OUT}"

    banner \
        "MTD-STYLE StreamDSGN | ${HZ} Hz | ${level}"

    echo "source      : ${SRC}"
    echo "calibration : ${CALIB}"
    echo "match thres : ${MATCH_THRESHOLD_M} m"
    echo "output      : ${OUT}"

    python "${EVAL}" \
        --cfg "${ORIGINAL_CFG}" \
        --source_dir "${SRC}" \
        --calibration_summary "${CALIB}" \
        --output_dir "${OUT}" \
        --match_threshold_m "${MATCH_THRESHOLD_M}" \
        --workers 0 \
        --require_full \
        2>&1 | tee \
        "${OUT}/run.log"

    # Random-contention experiments must use
    # exactly the same exogenous TV trace.
    if [ "${level}" != "L0" ]; then

        SRC_TRACE="${SRC}/contention_trace.csv"

        TV_TRACE="outputs/elastic_bev/${EXP_NAME}/formal_streaming_random50/${HZ}Hz_forward_only_seed${TRACE_SEED}/L0_${level}_p${PRESSURE_FRACTION}/contention_trace.csv"

        require_file \
            "${SRC_TRACE}"

        require_file \
            "${TV_TRACE}"

        if cmp -s \
            "${SRC_TRACE}" \
            "${TV_TRACE}"
        then
            echo \
                "[TRACE CHECK PASS] Original/MTD source and TV traces are byte-identical."
        else
            echo \
                "[ERROR] contention trace differs from TV"

            echo \
                "Original: ${SRC_TRACE}"

            echo \
                "TV      : ${TV_TRACE}"

            exit 10
        fi
    fi
done

# MTD_STYLE_RUN_EOF
