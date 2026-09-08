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

CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-transtreaming_tat_v2_frozen_shared_5ep.yaml"

TRAIN_ROOT="outputs/stream_kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-transtreaming_tat_v2_frozen_shared_5ep.transtreaming_tat_v2_frozen_shared_5ep"

BEST_FILE="${TRAIN_ROOT}/best_checkpoint.txt"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

LEVELS_JSON="outputs/elastic_bev/${EXP_NAME}/contention_calibration_v6/smooth_microchain_priority/contention_levels.json"

EVAL="tools/eval_transtreaming_v2_stream.py"

for F in \
    "$CFG" \
    "$BEST_FILE" \
    "$LEVELS_JSON" \
    "$EVAL"
do
    if [ ! -f "$F" ]; then
        echo "[ERROR] missing: $F"
        exit 2
    fi
done

BEST_CKPT="$(cat "$BEST_FILE")"

if [ ! -f "$BEST_CKPT" ]; then
    echo "[ERROR] missing best checkpoint:"
    echo "$BEST_CKPT"
    exit 3
fi

BEST_EPOCH="$(
python - "$BEST_CKPT" <<'PY'
import re
import sys
m = re.search(r'checkpoint_epoch_(\d+)\.pth$', sys.argv[1])
if not m:
    raise SystemExit("cannot parse epoch")
print(m.group(1))
PY
)"

if [ "$MAX_FRAMES" -eq 0 ]; then
    KIND="formal_random50"
else
    KIND="smoke_random50_max${MAX_FRAMES}"
fi

ROOT="outputs/transtreaming_frozen5_best/${KIND}/${HZ}Hz_seed${TRACE_SEED}"

IFS=',' read -ra LEVEL_ARRAY <<< "$LEVELS"

for RAW in "${LEVEL_ARRAY[@]}"
do
    LEVEL="$(echo "$RAW" | xargs)"

    case "$LEVEL" in
        L1|L2|L3|L4)
            ;;
        *)
            echo "[ERROR] invalid level: ${LEVEL}"
            exit 4
            ;;
    esac

    OUT="${ROOT}/L0_${LEVEL}_p${FRACTION}"

    rm -rf "$OUT"
    mkdir -p "$OUT"

    echo
    echo "================================================================================"
    echo "Transtreaming frozen-5 BEST epoch=${BEST_EPOCH}"
    echo "${HZ} Hz | L0/${LEVEL} Random50 | fraction=${FRACTION}"
    echo "================================================================================"

    python "$EVAL" \
        --cfg "$CFG" \
        --ckpt "$BEST_CKPT" \
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

done
