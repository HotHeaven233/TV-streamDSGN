#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
source scripts/stream_exp/00_env.sh


CFG="${1:-configs/stream/kitti_models/stream_dsgn_r18-token_prev_next-lasp_style_15ep.yaml}"
TRAIN_EXP="${2:-lasp_style_15ep}"
START_EP="${3:-1}"
END_EP="${4:-15}"
WORKERS="${5:-4}"

SELECT_EXP="lasp_best_selection"
EVAL_TAG="best_selection"

PARSER="tools/parse_lasp_offline_log.py"


# ======================================================================
# Basic preflight
# ======================================================================

for F in \
    "${CFG}" \
    "${PARSER}"
do
    if [[ ! -f "${F}" ]]; then
        echo "[ERROR] missing:"
        echo "        ${F}"
        exit 2
    fi
done


python - "${CFG}" <<'PY'
import sys
from pathlib import Path
import yaml

p = Path(sys.argv[1])

cfg = yaml.safe_load(
    p.read_text()
)

metrics = (
    cfg[
        "MODEL"
    ][
        "POST_PROCESSING"
    ][
        "EVAL_METRIC"
    ]
)

if metrics != ["offline_3d"]:
    raise RuntimeError(
        "LASP checkpoint selection must use "
        "EVAL_METRIC=['offline_3d']; "
        f"got {metrics}"
    )

print(
    "[PASS] validation protocol = offline_3d only"
)
PY


if ! [[ "${START_EP}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] invalid START_EP=${START_EP}"
    exit 3
fi

if ! [[ "${END_EP}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] invalid END_EP=${END_EP}"
    exit 3
fi

if ! [[ "${WORKERS}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] invalid WORKERS=${WORKERS}"
    exit 3
fi

if (( START_EP < 1 || END_EP < START_EP )); then
    echo "[ERROR] invalid epoch range ${START_EP}..${END_EP}"
    exit 3
fi


CFG_TAG="$(basename "${CFG}" .yaml)"

TRAIN_ROOT="outputs/stream_kitti_models/${CFG_TAG}.${TRAIN_EXP}"

CKPT_DIR="${TRAIN_ROOT}/ckpt"

SELECT_ROOT="outputs/stream_kitti_models/${CFG_TAG}.${SELECT_EXP}"

RESULTS="${TRAIN_ROOT}/lasp_validation.tsv"

BEST_EPOCH_FILE="${TRAIN_ROOT}/best_epoch.txt"
BEST_CKPT_FILE="${TRAIN_ROOT}/best_checkpoint.txt"
BEST_JSON_FILE="${TRAIN_ROOT}/best_validation.json"


if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "[ERROR] missing checkpoint directory:"
    echo "        ${CKPT_DIR}"
    exit 4
fi


# ======================================================================
# Check ALL checkpoints before doing any work
# ======================================================================

echo
echo "======================================================================"
echo "LASP CHECKPOINT PREFLIGHT"
echo "======================================================================"

for EP in $(seq "${START_EP}" "${END_EP}")
do
    CKPT="${CKPT_DIR}/checkpoint_epoch_${EP}.pth"

    if [[ ! -f "${CKPT}" ]]; then
        echo "[ERROR] missing checkpoint:"
        echo "        ${CKPT}"
        exit 5
    fi

    SIZE="$(
        du -h "${CKPT}" \
        | awk '{print $1}'
    )"

    printf \
        '[OK] epoch %02d  %s\n' \
        "${EP}" \
        "${SIZE}"
done

echo
echo "[PASS] all requested checkpoints exist"


# ======================================================================
# Write result table atomically.
#
# Existing selector logs are reused ONLY if:
#
#   1. exactly one log exists
#   2. it contains Evaluation done.
#   3. offline_3d exists
#   4. strict AP_R40 parser succeeds
#
# Otherwise that epoch is rerun.
# ======================================================================

mkdir -p "${TRAIN_ROOT}"

TMP_RESULTS="${RESULTS}.tmp.$$"

trap '
    rm -f "${TMP_RESULTS:-}"
' EXIT

printf \
    'epoch\tCar\tPedestrian\tCyclist\tMacro\n' \
    > "${TMP_RESULTS}"


for EP in $(seq "${START_EP}" "${END_EP}")
do
    CKPT="${CKPT_DIR}/checkpoint_epoch_${EP}.pth"

    EPOCH_ROOT="${SELECT_ROOT}/eval/epoch_${EP}"

    echo
    echo "======================================================================"
    echo "LASP OFFLINE VALIDATION | EPOCH ${EP}"
    echo "======================================================================"


    # ------------------------------------------------------------------
    # First try to reuse an already completed result.
    # ------------------------------------------------------------------

    mapfile -t EXISTING_LOGS < <(
        find \
            "${EPOCH_ROOT}" \
            -type f \
            -path "*/${EVAL_TAG}/log_eval.txt" \
            -print \
            2>/dev/null \
            | sort
    )

    REUSE=0
    ROW=""


    if (( ${#EXISTING_LOGS[@]} == 1 )); then
        EXISTING_LOG="${EXISTING_LOGS[0]}"

        echo \
            "[CHECK] existing log: ${EXISTING_LOG}"

        if ROW="$(
            python "${PARSER}" \
                --log "${EXISTING_LOG}" \
                --epoch "${EP}" \
                --format tsv
        )"
        then
            REUSE=1

            echo \
                "[REUSE] valid completed epoch ${EP} result"

        else
            echo \
                "[INFO] existing epoch ${EP} log is incomplete or invalid; rerunning"
        fi
    elif (( ${#EXISTING_LOGS[@]} > 1 )); then
        echo \
            "[INFO] multiple old logs found for epoch ${EP}; rerunning cleanly"
    fi


    # ------------------------------------------------------------------
    # Rerun only if reusable result was not found.
    # ------------------------------------------------------------------

    if (( REUSE == 0 )); then

        rm -rf "${EPOCH_ROOT}"

        python tools/test.py \
            --cfg_file "${CFG}" \
            --ckpt "${CKPT}" \
            --batch_size 1 \
            --workers "${WORKERS}" \
            --exp_name "${SELECT_EXP}" \
            --eval_tag "${EVAL_TAG}"


        mapfile -t NEW_LOGS < <(
            find \
                "${EPOCH_ROOT}" \
                -type f \
                -path "*/${EVAL_TAG}/log_eval.txt" \
                -print \
                2>/dev/null \
                | sort
        )

        if (( ${#NEW_LOGS[@]} != 1 )); then
            echo
            echo "[ERROR] expected exactly one evaluation log"
            echo "        epoch=${EP}"
            echo "        found=${#NEW_LOGS[@]}"
            echo

            find \
                "${EPOCH_ROOT}" \
                -maxdepth 6 \
                -type f \
                -print \
                2>/dev/null \
                || true

            exit 6
        fi

        LOG="${NEW_LOGS[0]}"

        echo \
            "[LOG] ${LOG}"

        ROW="$(
            python "${PARSER}" \
                --log "${LOG}" \
                --epoch "${EP}" \
                --format tsv
        )"
    fi


    # ------------------------------------------------------------------
    # Validate parser output shape before committing the row.
    # ------------------------------------------------------------------

    python - "${ROW}" "${EP}" <<'PY'
import math
import sys

row = sys.argv[1]
expected_epoch = int(sys.argv[2])

parts = row.split("\t")

if len(parts) != 5:
    raise RuntimeError(
        f"invalid LASP TSV row: {row!r}"
    )

epoch = int(parts[0])

if epoch != expected_epoch:
    raise RuntimeError(
        f"epoch mismatch: "
        f"expected={expected_epoch}, got={epoch}"
    )

values = [
    float(x)
    for x in parts[1:]
]

if not all(
    math.isfinite(x)
    for x in values
):
    raise RuntimeError(
        f"non-finite LASP result: {row}"
    )

print(
    f"[PASS] epoch {epoch:02d}: "
    f"Car={values[0]:.4f} "
    f"Ped={values[1]:.4f} "
    f"Cyc={values[2]:.4f} "
    f"Macro={values[3]:.4f}"
)
PY


    printf \
        '%s\n' \
        "${ROW}" \
        >> "${TMP_RESULTS}"
done


# ======================================================================
# Commit completed table only after ALL epochs succeeded.
# ======================================================================

mv \
    "${TMP_RESULTS}" \
    "${RESULTS}"

trap - EXIT


# ======================================================================
# Select validation best.
#
# Primary criterion:
#   strict 3D AP_R40 Moderate Macro
#
# Exact tie:
#   earlier epoch
# ======================================================================

python - \
    "${RESULTS}" \
    "${TRAIN_ROOT}" <<'PY'
import json
import math
import sys
from pathlib import Path


table = Path(
    sys.argv[1]
)

root = Path(
    sys.argv[2]
)

if not table.is_file():
    raise FileNotFoundError(
        table
    )


lines = table.read_text().splitlines()

if not lines:
    raise RuntimeError(
        "empty validation table"
    )


expected_header = (
    "epoch\tCar\tPedestrian\tCyclist\tMacro"
)

if lines[0].strip() != expected_header:
    raise RuntimeError(
        "unexpected TSV header: "
        f"{lines[0]!r}"
    )


rows = []

seen_epochs = set()

for line in lines[1:]:

    if not line.strip():
        continue

    parts = line.split("\t")

    if len(parts) != 5:
        raise RuntimeError(
            f"invalid TSV row: {line!r}"
        )

    ep, car, ped, cyc, macro = parts

    row = {
        "epoch":
            int(ep),

        "Car":
            float(car),

        "Pedestrian":
            float(ped),

        "Cyclist":
            float(cyc),

        "Macro":
            float(macro),
    }


    if row["epoch"] in seen_epochs:
        raise RuntimeError(
            f"duplicate epoch: {row['epoch']}"
        )

    seen_epochs.add(
        row["epoch"]
    )


    for key in (
        "Car",
        "Pedestrian",
        "Cyclist",
        "Macro",
    ):
        if not math.isfinite(
            row[key]
        ):
            raise RuntimeError(
                f"non-finite {key}: {row}"
            )


    recomputed = (
        row["Car"]
        + row["Pedestrian"]
        + row["Cyclist"]
    ) / 3.0


    if abs(
        recomputed
        - row["Macro"]
    ) > 1e-5:
        raise RuntimeError(
            "stored Macro does not match "
            f"class mean: {row}"
        )


    rows.append(
        row
    )


if not rows:
    raise RuntimeError(
        "no validation rows"
    )


best = sorted(
    rows,
    key=lambda x: (
        -x["Macro"],
        x["epoch"],
    ),
)[0]


ckpt = (
    root
    / "ckpt"
    / f"checkpoint_epoch_{best['epoch']}.pth"
)


if not ckpt.is_file():
    raise FileNotFoundError(
        ckpt
    )


(
    root
    / "best_epoch.txt"
).write_text(
    f"{best['epoch']}\n"
)


(
    root
    / "best_checkpoint.txt"
).write_text(
    f"{ckpt}\n"
)


payload = {
    "selection_metric":
        "strict_3d_AP_R40_Moderate_macro",

    "tie_break":
        "earlier_epoch",

    "classes": [
        "Car",
        "Pedestrian",
        "Cyclist",
    ],

    "num_evaluated_epochs":
        len(rows),

    "best":
        best,

    "checkpoint":
        str(ckpt),
}


(
    root
    / "best_validation.json"
).write_text(
    json.dumps(
        payload,
        indent=2,
    )
    + "\n"
)


print()
print(
    "======================================================================"
)
print(
    "LASP VALIDATION SELECTION COMPLETE"
)
print(
    "======================================================================"
)
print(
    f"best epoch = {best['epoch']}"
)
print(
    f"Car        = {best['Car']:.6f}"
)
print(
    f"Pedestrian = {best['Pedestrian']:.6f}"
)
print(
    f"Cyclist    = {best['Cyclist']:.6f}"
)
print(
    f"Macro      = {best['Macro']:.6f}"
)
print(
    f"checkpoint = {ckpt}"
)
print(
    "======================================================================"
)
PY


echo
echo "[VALIDATION TABLE]"

if command -v column >/dev/null 2>&1
then
    column \
        -t \
        -s $'\t' \
        "${RESULTS}"
else
    cat "${RESULTS}"
fi


echo
echo "[BEST]"

cat "${BEST_JSON_FILE}"

echo
echo "[PASS] LASP checkpoint selection complete."
