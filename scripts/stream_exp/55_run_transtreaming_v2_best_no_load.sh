#!/usr/bin/env bash
set -euo pipefail

cd /data/jhb/workspace/streamDSGN
source scripts/stream_exp/00_env.sh

FREQS="${1:-35,40,45,50}"
WARMUP="${2:-80}"
MAX_FRAMES="${3:-0}"
TRACE_SEED="${4:-20260903}"

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
    KIND="formal_streaming_no_load"
else
    KIND="smoke_streaming_no_load_max${MAX_FRAMES}"
fi

IFS=',' read -ra FREQ_ARRAY <<< "$FREQS"

for raw in "${FREQ_ARRAY[@]}"
do
    HZ="$(echo "$raw" | xargs)"

    OUT="outputs/transtreaming_v2_best/${KIND}/${HZ}Hz_forward_only/L0"

    rm -rf "$OUT"
    mkdir -p "$OUT"

    echo
    echo "================================================================================"
    echo "Transtreaming V2 BEST epoch=13 | ${HZ}Hz | L0"
    echo "================================================================================"

    python "$EVAL" \
        --cfg "$TS_CFG" \
        --ckpt "$TS_CKPT" \
        --levels_json "$LEVELS_JSON" \
        --pressure_level L0 \
        --pressure_fraction 0 \
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
done
