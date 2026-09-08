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

EVAL="tools/eval_mtd_three_head_random50.py"

H2_CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-mtd_h2.yaml"
H3_CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-mtd_h3.yaml"

H2_TAG="$(basename "${H2_CFG}" .yaml)"
H3_TAG="$(basename "${H3_CFG}" .yaml)"

H2_CKPT="outputs/stream_kitti_models/${H2_TAG}.mtd_head_only/ckpt/checkpoint_epoch_5.pth"
H3_CKPT="outputs/stream_kitti_models/${H3_TAG}.mtd_head_only/ckpt/checkpoint_epoch_5.pth"

if [ "${MAX_FRAMES}" -eq 0 ]; then
    RUN_KIND="formal_streaming_random50"
else
    RUN_KIND="smoke_streaming_random50_max${MAX_FRAMES}"
fi

OUT_ROOT="outputs/mtd_three_head/${RUN_KIND}/${HZ}Hz_forward_only_seed${TRACE_SEED}"

for f in \
    "${FULL_CFG}" \
    "${FULL_CKPT}" \
    "${H2_CKPT}" \
    "${H3_CKPT}" \
    "${LEVELS_JSON}" \
    "${EVAL}" \
    "tools/mtd_three_head_runtime.py" \
    "tools/eval_tv_stream3d_30hz_random50.py" \
    "tools/test_stream_buffer_timestamp.py" \
    "tools/test_tv_stream3d_online_forward.py" \
    "tools/smooth_cuda_contention.py"
do
    if [ ! -f "${f}" ]; then
        echo "[ERROR] missing: ${f}"
        exit 2
    fi
done

if ! [[ "${HZ}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "[ERROR] invalid Hz: ${HZ}"
    exit 3
fi

mkdir -p "${OUT_ROOT}"

echo
echo "===================================================================================================="
echo "TRUE MTD THREE-HEAD StreamDSGN | RANDOM-50 CONTENTION"
echo "===================================================================================================="
echo "shared cfg        : ${FULL_CFG}"
echo "H1 next           : ${FULL_CKPT}"
echo "H2 next2          : ${H2_CKPT}"
echo "H3 next3          : ${H3_CKPT}"
echo "input Hz          : ${HZ}"
echo "pressure levels   : ${PRESSURE_LEVELS}"
echo "pressure fraction : ${PRESSURE_FRACTION}"
echo "trace seed        : ${TRACE_SEED}"
echo "warmup            : ${WARMUP}"
echo "max frames        : ${MAX_FRAMES}"
echo "timing            : forward-only"
echo "DAM               : causal previous-runtime routing"
echo "===================================================================================================="

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
    echo "MTD THREE-HEAD | ${HZ} Hz | L0 + ${pressure}"
    echo "===================================================================================================="
    echo "output : ${OUT_DIR}"
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
        --workers 0 \
        --seed 1024 \
        --output_dir "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/run.log"

    BASE_TRACE="${OUT_DIR}/contention_trace.csv"

    if [ ! -f "${BASE_TRACE}" ]; then
        echo "[ERROR] evaluator did not produce contention_trace.csv"
        exit 8
    fi

    # ------------------------------------------------------------
    # Cross-method trace identity.
    #
    # IMPORTANT:
    # Exact comparison is meaningful only for the formal full run.
    #
    # build_balanced_trace() samples pressure positions as a
    # function of each scene length n. Therefore a MAX_FRAMES=200
    # smoke trace is NOT necessarily a prefix of a full-run trace,
    # even with exactly the same seed.
    # ------------------------------------------------------------

    if [ "${MAX_FRAMES}" -eq 0 ]; then

        TV_TRACE="outputs/elastic_bev/${EXP_NAME}/formal_streaming_random50/${HZ}Hz_forward_only_seed${TRACE_SEED}/L0_${pressure}_p${PRESSURE_FRACTION}/contention_trace.csv"

        if [ -f "${TV_TRACE}" ]; then
            if cmp -s "${TV_TRACE}" "${BASE_TRACE}"; then
                echo "[TRACE CHECK PASS] MTD and TV traces are byte-identical."
            else
                echo "[ERROR] FORMAL MTD contention trace differs from TV-Stream3D!"
                echo "TV : ${TV_TRACE}"
                echo "MTD: ${BASE_TRACE}"
                exit 10
            fi
        else
            echo "[WARN] formal TV trace not found:"
            echo "       ${TV_TRACE}"
        fi

        ORIGINAL_TRACE="outputs/original_streamdsgn/formal_streaming_random50/${HZ}Hz_forward_only_seed${TRACE_SEED}/L0_${pressure}_p${PRESSURE_FRACTION}/contention_trace.csv"

        if [ -f "${ORIGINAL_TRACE}" ]; then
            if cmp -s "${ORIGINAL_TRACE}" "${BASE_TRACE}"; then
                echo "[TRACE CHECK PASS] MTD and Original traces are byte-identical."
            else
                echo "[ERROR] FORMAL MTD contention trace differs from Original StreamDSGN!"
                echo "Original: ${ORIGINAL_TRACE}"
                echo "MTD     : ${BASE_TRACE}"
                exit 11
            fi
        else
            echo "[WARN] formal Original trace not found:"
            echo "       ${ORIGINAL_TRACE}"
        fi

    else
        echo "[TRACE CHECK] smoke run: formal cross-method cmp intentionally skipped."
        echo "[TRACE CHECK] MAX_FRAMES=${MAX_FRAMES}; trace is generated on the truncated scene set."
    fi

    # ------------------------------------------------------------
    # 3. Basic MTD routing audit.
    # ------------------------------------------------------------
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
        f"missing summary.json: {summary_path}"
    )

if not decision_path.exists():
    raise RuntimeError(
        f"missing mtd_decisions.csv: {decision_path}"
    )

summary = json.loads(
    summary_path.read_text()
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
        "cannot locate MTD branch column; "
        f"columns={list(rows[0].keys())}"
    )

hist = Counter(
    int(r[branch_key])
    for r in rows
)

invalid = [
    x
    for x in hist
    if x not in (1, 2, 3)
]

if invalid:
    raise RuntimeError(
        f"invalid branches: {invalid}"
    )

print()
print("-" * 90)
print("MTD ROUTING AUDIT")
print("-" * 90)
print(
    "method       :",
    summary.get("method")
)
print(
    "base_detector:",
    summary.get("base_detector")
)
print(
    "decisions    :",
    len(rows)
)
print(
    "H1/H2/H3     :",
    hist.get(1, 0),
    hist.get(2, 0),
    hist.get(3, 0),
)
print(
    "fractions    :",
    {
        f"H{k}": hist.get(k, 0) / len(rows)
        for k in (1, 2, 3)
    }
)

if "oracle_branch_step" in rows[0]:
    matches = sum(
        int(r[branch_key]) ==
        int(r["oracle_branch_step"])
        for r in rows
    )

    print(
        "oracle match :",
        matches / len(rows)
    )

print("[MTD ROUTING CHECK PASS]")
print("-" * 90)
PY

done

# ----------------------------------------------------------------------
# Aggregate paper-facing results.
# ----------------------------------------------------------------------

python - "${OUT_ROOT}" "${PRESSURE_LEVELS}" "${PRESSURE_FRACTION}" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])

levels = [
    x.strip()
    for x in sys.argv[2].split(",")
    if x.strip()
]

fraction = sys.argv[3]

rows_out = []

for level in levels:
    out = (
        root
        / f"L0_{level}_p{fraction}"
    )

    summary_path = out / "summary.json"
    decision_path = out / "mtd_decisions.csv"

    s = json.loads(
        summary_path.read_text()
    )

    q = s[
        "stream_sap_3d_moderate_R40"
    ]

    fwd = s[
        "forward_latency"
    ]

    decisions = list(
        csv.DictReader(
            decision_path.open()
        )
    )

    branch_key = None

    for candidate in [
        "branch_step",
        "selected_branch_step",
        "mtd_branch_step",
    ]:
        if decisions and candidate in decisions[0]:
            branch_key = candidate
            break

    if branch_key is None:
        raise RuntimeError(
            f"branch column missing in {decision_path}"
        )

    hist = Counter(
        int(x[branch_key])
        for x in decisions
    )

    n = max(
        len(decisions),
        1,
    )

    rows_out.append({
        "pressure_level":
            level,

        "sensor_frames":
            s["sensor_frames"],

        "processed_frames":
            s["processed_frames"],

        "dropped_frames":
            s["dropped_frames"],

        "drop_rate":
            s["drop_rate"],

        "deadline_miss_count":
            s["deadline_miss_count"],

        "deadline_miss_rate":
            s["deadline_miss_rate"],

        "forward_p50_ms":
            fwd["p50_ms"],

        "forward_p90_ms":
            fwd["p90_ms"],

        "forward_p99_ms":
            fwd["p99_ms"],

        "H1_count":
            hist.get(1, 0),

        "H2_count":
            hist.get(2, 0),

        "H3_count":
            hist.get(3, 0),

        "H1_fraction":
            hist.get(1, 0) / n,

        "H2_fraction":
            hist.get(2, 0) / n,

        "H3_fraction":
            hist.get(3, 0) / n,

        "Car_3d_mod_R40":
            q["Car"],

        "Pedestrian_3d_mod_R40":
            q["Pedestrian"],

        "Cyclist_3d_mod_R40":
            q["Cyclist"],

        "Macro_3d_mod_R40":
            q["Macro"],
    })

out_csv = (
    root
    / "all_mtd_three_head_random50_summary.csv"
)

with out_csv.open(
    "w",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(
            rows_out[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        rows_out
    )

print()
print("=" * 120)
print("MTD THREE-HEAD RANDOM-50 SUMMARY")
print("=" * 120)

for r in rows_out:
    print(
        f"L0/{r['pressure_level']} | "
        f"Macro={r['Macro_3d_mod_R40']:.4f} | "
        f"miss={100*r['deadline_miss_rate']:.3f}% | "
        f"drop={100*r['drop_rate']:.3f}% | "
        f"p99={r['forward_p99_ms']:.4f} ms | "
        f"H1/H2/H3="
        f"{100*r['H1_fraction']:.1f}%/"
        f"{100*r['H2_fraction']:.1f}%/"
        f"{100*r['H3_fraction']:.1f}%"
    )

print(
    "CSV =",
    out_csv
)
print("=" * 120)
PY

echo
echo "===================================================================================================="
echo "MTD THREE-HEAD RANDOM-50 COMPLETE"
echo "===================================================================================================="

# MTD_THREE_HEAD_RANDOM50_EOF
