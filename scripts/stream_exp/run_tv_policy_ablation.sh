#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-20}"
N="${2:-100}"
HZ="${3:-35}"
WARMUP="${4:-80}"

PRESSURE_LEVELS="${5:-L2,L3,L4}"

MAX_FRAMES="${6:-0}"
GUARD="${7:-0.25}"

TRACE_SEED="${8:-20260903}"
PRESSURE_FRACTION="${9:-0.5}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

ELASTIC_CKPT="outputs/elastic_bev/${EXP_NAME}/ckpt/checkpoint_epoch_${EPOCH}.pth"

BANK="outputs/elastic_bev/${EXP_NAME}/bn_bank/causal_prefix_bank_e${EPOCH}_n${N}.pth"

PROFILE_ROOT="outputs/elastic_bev/${EXP_NAME}/all84_contention_profile_v6/e${EPOCH}_n${N}"

CTRL_CSV="${PROFILE_ROOT}/controller_remaining_latency_table.csv"

LEVELS_JSON="outputs/elastic_bev/${EXP_NAME}/contention_calibration_v6/smooth_microchain_priority/contention_levels.json"

OUT_ROOT="outputs/elastic_bev/${EXP_NAME}/system_ablation/static_vs_dynamic/${HZ}Hz_seed${TRACE_SEED}"

POLICIES=(
    full
    best_static
    tv_dynamic
    oracle_level
)

for f in \
    "${FULL_CFG}" \
    "${FULL_CKPT}" \
    "${ELASTIC_CKPT}" \
    "${BANK}" \
    "${CTRL_CSV}" \
    "${LEVELS_JSON}" \
    tools/eval_tv_policy_ablation.py
do
    if [[ ! -f "${f}" ]]; then
        echo "[ERROR] missing: ${f}" >&2
        exit 2
    fi
done

mkdir -p "${OUT_ROOT}"

IFS=',' read -r -a LEVEL_ARRAY <<< "${PRESSURE_LEVELS}"

for raw in "${LEVEL_ARRAY[@]}"
do
    LEVEL="$(echo "${raw}" | xargs)"

    case "${LEVEL}" in
        L1|L2|L3|L4)
            ;;
        *)
            echo "[ERROR] invalid level: ${LEVEL}" >&2
            exit 3
            ;;
    esac

    echo
    echo "=========================================================================="
    echo "STATIC-vs-DYNAMIC ABLATION | ${LEVEL}"
    echo "=========================================================================="

    for POLICY in "${POLICIES[@]}"
    do
        OUT="${OUT_ROOT}/${LEVEL}/${POLICY}"

        rm -rf "${OUT}"
        mkdir -p "${OUT}"

        echo
        echo "----------------------------------------------------------------------"
        echo "${LEVEL} | ${POLICY}"
        echo "----------------------------------------------------------------------"

        python tools/eval_tv_policy_ablation.py \
            --full_cfg "${FULL_CFG}" \
            --full_ckpt "${FULL_CKPT}" \
            --elastic_ckpt "${ELASTIC_CKPT}" \
            --prefix_bn_bank "${BANK}" \
            --controller_csv "${CTRL_CSV}" \
            --levels_json "${LEVELS_JSON}" \
            --policy "${POLICY}" \
            --pressure_level "${LEVEL}" \
            --pressure_fraction "${PRESSURE_FRACTION}" \
            --trace_seed "${TRACE_SEED}" \
            --input_hz "${HZ}" \
            --runtime_warmup_frames "${WARMUP}" \
            --max_frames "${MAX_FRAMES}" \
            --workers 0 \
            --seed 1024 \
            --control_guard_per_boundary_ms "${GUARD}" \
            --output_dir "${OUT}" \
            2>&1 | tee "${OUT}/run.log"

        [[ -f "${OUT}/summary.json" ]] || {
            echo "[ERROR] missing summary: ${OUT}/summary.json" >&2
            exit 4
        }
    done

    # ------------------------------------------------------------------
    # Exact Random50 trace equality across all four policies
    # ------------------------------------------------------------------

    REF="${OUT_ROOT}/${LEVEL}/full/contention_trace.csv"

    for POLICY in \
        best_static \
        tv_dynamic \
        oracle_level
    do
        TEST="${OUT_ROOT}/${LEVEL}/${POLICY}/contention_trace.csv"

        if cmp -s "${REF}" "${TEST}"; then
            echo "[TRACE PASS] ${LEVEL}: full == ${POLICY}"
        else
            echo "[ERROR] trace mismatch: ${LEVEL}/${POLICY}" >&2
            exit 10
        fi
    done

    # Compare to already frozen TV formal trace if present.
    FORMAL_TRACE="outputs/elastic_bev/${EXP_NAME}/formal_streaming_random50/${HZ}Hz_forward_only_seed${TRACE_SEED}/L0_${LEVEL}_p${PRESSURE_FRACTION}/contention_trace.csv"

    DYNAMIC_TRACE="${OUT_ROOT}/${LEVEL}/tv_dynamic/contention_trace.csv"

    if [[ -f "${FORMAL_TRACE}" ]]; then
        if cmp -s "${FORMAL_TRACE}" "${DYNAMIC_TRACE}"; then
            echo "[TRACE PASS] ${LEVEL}: ablation == frozen formal TV"
        else
            echo "[ERROR] ablation trace != frozen formal TV" >&2
            exit 11
        fi
    else
        echo "[WARN] frozen formal trace not found:"
        echo "       ${FORMAL_TRACE}"
    fi
done

# ======================================================================
# Aggregate result table
# ======================================================================

python - "${OUT_ROOT}" "${PRESSURE_LEVELS}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(
    sys.argv[1]
)

levels = [
    x.strip()
    for x
    in sys.argv[2].split(",")
    if x.strip()
]

policies = (
    "full",
    "best_static",
    "tv_dynamic",
    "oracle_level",
)

rows = []

for level in levels:
    for policy in policies:
        p = (
            root
            /
            level
            /
            policy
            /
            "summary.json"
        )

        if not p.is_file():
            raise RuntimeError(
                f"missing: {p}"
            )

        s = json.loads(
            p.read_text()
        )

        f = s[
            "forward_latency"
        ]

        q = s[
            "stream_sap_3d_moderate_R40"
        ]

        ctrl = s[
            "controller_cpu_ms"
        ]

        rows.append({
            "level":
                level,

            "policy":
                policy,

            "Macro":
                q["Macro"],

            "Car":
                q["Car"],

            "Pedestrian":
                q["Pedestrian"],

            "Cyclist":
                q["Cyclist"],

            "drop_rate":
                s["drop_rate"],

            "deadline_miss_rate":
                s[
                    "deadline_miss_rate"
                ],

            "p50_ms":
                f["p50_ms"],

            "p90_ms":
                f["p90_ms"],

            "p99_ms":
                f["p99_ms"],

            "controller_mean_ms":
                ctrl.get(
                    "mean_ms",
                    0.0,
                ),

            "processed_frames":
                s[
                    "processed_frames"
                ],
        })

out = (
    root
    /
    "policy_ablation_summary.csv"
)

with out.open(
    "w",
    newline="",
) as f:
    w = csv.DictWriter(
        f,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    w.writeheader()
    w.writerows(rows)

print()
print("=" * 120)

print(
    "STATIC-vs-DYNAMIC POLICY ABLATION"
)

print("=" * 120)

print(
    "Level\tPolicy\tMacro\tDrop%\tMiss%\tp50\tp90\tp99"
)

for r in rows:
    print(
        f"{r['level']}\t"
        f"{r['policy']}\t"
        f"{r['Macro']:.4f}\t"
        f"{100*r['drop_rate']:.3f}\t"
        f"{100*r['deadline_miss_rate']:.3f}\t"
        f"{r['p50_ms']:.3f}\t"
        f"{r['p90_ms']:.3f}\t"
        f"{r['p99_ms']:.3f}"
    )

print()
print(f"CSV = {out}")
print("=" * 120)
PY

echo
echo "[PASS] static-vs-dynamic ablation complete"
