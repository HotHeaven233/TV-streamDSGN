#!/usr/bin/env bash
set -euo pipefail

cd /data/jhb/workspace/streamDSGN
source scripts/stream_exp/00_env.sh

if [[ $# -ne 1 ]]; then
    echo "Usage:"
    echo "  $0 /absolute/or/repo/path/to/contention_levels.json" >&2
    exit 2
fi

LEVELS_JSON="$1"

CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev_next-lasp_style_15ep.yaml"

CKPT="outputs/stream_kitti_models/stream_dsgn_r18-token_prev_next-lasp_style_15ep.lasp_style_15ep/ckpt/checkpoint_epoch_2.pth"

ROOT="outputs/stream_exp/lasp_style_final/random50_35hz"

for F in \
    "$CFG" \
    "$CKPT" \
    "$LEVELS_JSON" \
    tools/eval_lasp_stream.py
do
    [[ -f "$F" ]] || {
        echo "[ERROR] missing: $F" >&2
        exit 3
    }
done

sha256sum -c \
    outputs/lasp/final/checkpoint.sha256

for LEVEL in L1 L2 L3 L4
do
    OUT="${ROOT}/${LEVEL}"

    echo
    echo "============================================================"
    echo "LASP RANDOM50 @ 35 Hz | L0 + ${LEVEL}"
    echo "============================================================"

    rm -rf "$OUT"

    python tools/eval_lasp_stream.py \
        --cfg "$CFG" \
        --ckpt "$CKPT" \
        --levels_json "$LEVELS_JSON" \
        --pressure_level "$LEVEL" \
        --pressure_fraction 0.5 \
        --trace_seed 20260903 \
        --input_hz 35 \
        --runtime_warmup_frames 80 \
        --max_frames 0 \
        --workers 0 \
        --seed 1024 \
        --output_dir "$OUT"

    [[ -f "$OUT/summary.json" ]] || {
        echo "[ERROR] missing summary: $OUT/summary.json" >&2
        exit 4
    }
done

echo
echo "[PASS] LASP Random50 L1-L4 complete"
