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

EVAL="tools/eval_mtd_three_head_no_load.py"

H2_CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-mtd_h2.yaml"
H3_CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-mtd_h3.yaml"

H2_TAG="$(basename "${H2_CFG}" .yaml)"
H3_TAG="$(basename "${H3_CFG}" .yaml)"

H2_CKPT="outputs/stream_kitti_models/${H2_TAG}.mtd_head_only/ckpt/checkpoint_epoch_5.pth"
H3_CKPT="outputs/stream_kitti_models/${H3_TAG}.mtd_head_only/ckpt/checkpoint_epoch_5.pth"

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"
require_file "${H2_CKPT}"
require_file "${H3_CKPT}"
require_file "${LEVELS_JSON}"
require_file "${EVAL}"
require_file "tools/mtd_three_head_runtime.py"

echo
echo "======================================================================"
echo "TRUE MTD THREE-HEAD StreamDSGN | NO-LOAD FREQUENCY SWEEP"
echo "======================================================================"
echo "shared cfg   : ${FULL_CFG}"
echo "H1 next      : ${FULL_CKPT}"
echo "H2 next2     : ${H2_CKPT}"
echo "H3 next3     : ${H3_CKPT}"
echo "frequencies  : ${FREQUENCIES}"
echo "contention   : L0 only"
echo "timing       : forward-only"
echo "DAM          : causal previous-runtime routing"
echo "======================================================================"

IFS=',' read -r -a HZ_ARRAY <<< "${FREQUENCIES}"

for raw_hz in "${HZ_ARRAY[@]}"
do
    HZ="$(echo "${raw_hz}" | xargs)"

    if ! [[ "${HZ}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "[ERROR] invalid Hz: ${HZ}"
        exit 3
    fi

    OUT_DIR="outputs/mtd_three_head/formal_streaming_no_load/${HZ}Hz_forward_only/L0"

    rm -rf "${OUT_DIR}"
    mkdir -p "${OUT_DIR}"

    echo
    echo "======================================================================"
    echo "MTD Three-Head | ${HZ} Hz | L0"
    echo "======================================================================"

    python "${EVAL}" \
        --cfg "${FULL_CFG}" \
        --ckpt "${FULL_CKPT}" \
        --h2_ckpt "${H2_CKPT}" \
        --h3_ckpt "${H3_CKPT}" \
        --levels_json "${LEVELS_JSON}" \
        --pressure_level "L0" \
        --pressure_fraction 0.0 \
        --trace_seed "${TRACE_SEED}" \
        --input_hz "${HZ}" \
        --runtime_warmup_frames "${WARMUP}" \
        --max_frames "${MAX_FRAMES}" \
        --workers 0 \
        --seed 1024 \
        --output_dir "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/run.log"

    echo
    echo "----------------------------------------------------------------------"
    echo "MTD RESULT CHECK | ${HZ} Hz"
    echo "----------------------------------------------------------------------"

    python - "${OUT_DIR}" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

out = Path(sys.argv[1])

summary_path = out / "summary.json"
decision_path = out / "mtd_decisions.csv"

if not summary_path.exists():
    raise RuntimeError(
        f"missing summary: {summary_path}"
    )

if not decision_path.exists():
    raise RuntimeError(
        f"missing MTD decisions: {decision_path}"
    )

summary = json.loads(
    summary_path.read_text()
)

print(
    "method:",
    summary.get("method")
)
print(
    "base_detector:",
    summary.get("base_detector")
)

rows = list(
    csv.DictReader(
        decision_path.open()
    )
)

if not rows:
    raise RuntimeError(
        "mtd_decisions.csv is empty"
    )

branch_key = None

for candidate in [
    "branch_step",
    "selected_branch_step",
    "mtd_branch_step",
]:
    if candidate in rows[0]:
        branch_key = candidate
        break

if branch_key is None:
    raise RuntimeError(
        "cannot find branch column in "
        f"{list(rows[0].keys())}"
    )

hist = Counter(
    int(row[branch_key])
    for row in rows
)

print(
    "processed decisions:",
    len(rows)
)
print(
    "branch histogram:",
    dict(sorted(hist.items()))
)

bad = [
    x
    for x in hist
    if x not in (1, 2, 3)
]

if bad:
    raise RuntimeError(
        f"invalid MTD branches: {bad}"
    )

print(
    "[MTD ROUTING CHECK PASS]"
)
PY

done

echo
echo "======================================================================"
echo "MTD THREE-HEAD NO-LOAD SWEEP COMPLETE"
echo "======================================================================"
