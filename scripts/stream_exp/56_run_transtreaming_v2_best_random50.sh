#!/usr/bin/env bash
set -euo pipefail

cd /data/jhb/workspace/streamDSGN
source scripts/stream_exp/00_env.sh

HZ="${1:-35}"
WARMUP="${2:-80}"
LEVELS="${3:-L1,L2,L3,L4}"
MAX_FRAMES="${4:-0}"
TRACE_SEED="${5:-20260903}"
FRACTION="${6:-0.5}"

TS_CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-transtreaming_tat_v2_15ep.yaml"

TS_CKPT="outputs/stream_kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-transtreaming_tat_v2_15ep.transtreaming_tat_v2_15ep/ckpt/checkpoint_epoch_13.pth"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

LEVELS_JSON="outputs/elastic_bev/${EXP_NAME}/contention_calibration_v6/smooth_microchain_priority/contention_levels.json"

EVAL="tools/eval_transtreaming_v2_stream.py"

for f in \
    "$TS_CFG" \
    "$TS_CKPT" \
    "$LEVELS_JSON" \
    "$EVAL"
do
    if [ ! -f "$f" ]; then
        echo "[ERROR] missing: $f"
        exit 2
    fi
done

if [ "$MAX_FRAMES" -eq 0 ]; then
    KIND="formal_streaming_random50"
else
    KIND="smoke_streaming_random50_max${MAX_FRAMES}"
fi

ROOT="outputs/transtreaming_v2_best/${KIND}/${HZ}Hz_forward_only_seed${TRACE_SEED}"

IFS=',' read -ra LEVEL_ARRAY <<< "$LEVELS"

for raw in "${LEVEL_ARRAY[@]}"
do
    LEVEL="$(echo "$raw" | xargs)"

    case "$LEVEL" in
        L1|L2|L3|L4) ;;
        *)
            echo "[ERROR] invalid level: $LEVEL"
            exit 3
            ;;
    esac

    OUT="${ROOT}/L0_${LEVEL}_p${FRACTION}"

    rm -rf "$OUT"
    mkdir -p "$OUT"

    echo
    echo "================================================================================"
    echo "Transtreaming V2 BEST epoch=13 | ${HZ}Hz | L0/${LEVEL} random50"
    echo "================================================================================"

    python "$EVAL" \
        --cfg "$TS_CFG" \
        --ckpt "$TS_CKPT" \
        --levels_json "$LEVELS_JSON" \
        --pressure_level "$LEVEL" \
        --pressure_fraction "$FRACTION" \
        --trace_seed "$TRACE_SEED" \
        --input_hz "$HZ" \
        --runtime_warmup_frames "$WARMUP" \
        --max_frames "$MAX_FRAMES" \
        --past_length 3 \
        --future_length 4 \
        --max_future 8 \
        --workers 0 \
        --seed 1024 \
        --output_dir "$OUT" \
        2>&1 | tee "$OUT/run.log"

    # Exact trace comparison only for formal full runs.
    if [ "$MAX_FRAMES" -eq 0 ]; then

        TRACE="$OUT/contention_trace.csv"

        REFS=(
            "outputs/original_streamdsgn/formal_streaming_random50/${HZ}Hz_forward_only_seed${TRACE_SEED}/L0_${LEVEL}_p${FRACTION}/contention_trace.csv"
            "outputs/elastic_bev/${EXP_NAME}/formal_streaming_random50/${HZ}Hz_forward_only_seed${TRACE_SEED}/L0_${LEVEL}_p${FRACTION}/contention_trace.csv"
            "outputs/mtd_three_head/formal_streaming_random50/${HZ}Hz_forward_only_seed${TRACE_SEED}/L0_${LEVEL}_p${FRACTION}/contention_trace.csv"
        )

        for REF in "${REFS[@]}"
        do
            if [ -f "$REF" ]; then
                if cmp -s "$REF" "$TRACE"; then
                    echo "[TRACE PASS] $REF"
                else
                    echo "[ERROR] contention trace mismatch"
                    echo "REF = $REF"
                    echo "TS  = $TRACE"
                    exit 10
                fi
            fi
        done
    else
        echo "[TRACE] smoke run: exact formal comparison skipped"
    fi
done
