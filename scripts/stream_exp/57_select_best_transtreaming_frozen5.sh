#!/usr/bin/env bash
set -euo pipefail

cd /data/jhb/workspace/streamDSGN
source scripts/stream_exp/00_env.sh

CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-transtreaming_tat_v2_frozen_shared_5ep.yaml"

TRAIN_ROOT="outputs/stream_kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-transtreaming_tat_v2_frozen_shared_5ep.transtreaming_tat_v2_frozen_shared_5ep"

SELECT_EXP="transtreaming_frozen5_best_selection"
EVAL_TAG="best_selection"

CFG_TAG="$(basename "${CFG}" .yaml)"

# tools/test.py:
#   cfg.EXP_GROUP_PATH = '_'.join(cfg_file.split('/')[1:-1])
#
# configs/stream/kitti_models/xxx.yaml
# ->
# stream_kitti_models
EXP_GROUP="stream_kitti_models"

SELECT_ROOT="outputs/${EXP_GROUP}/${CFG_TAG}.${SELECT_EXP}"

echo "================================================================================"
echo "TRANSTREAMING FROZEN-5 CHECKPOINT SELECTION"
echo "================================================================================"
echo "CFG        = ${CFG}"
echo "TRAIN_ROOT = ${TRAIN_ROOT}"
echo "SELECT_ROOT= ${SELECT_ROOT}"
echo

if [ ! -f "${CFG}" ]; then
    echo "[ERROR] missing config:"
    echo "  ${CFG}"
    exit 2
fi

if [ ! -d "${TRAIN_ROOT}/ckpt" ]; then
    echo "[ERROR] missing checkpoint directory:"
    echo "  ${TRAIN_ROOT}/ckpt"
    exit 2
fi

# --------------------------------------------------------------------
# Verify all five checkpoints BEFORE doing any evaluation.
# --------------------------------------------------------------------

for EP in 1 2 3 4 5
do
    CKPT="${TRAIN_ROOT}/ckpt/checkpoint_epoch_${EP}.pth"

    if [ ! -f "${CKPT}" ]; then
        echo "[ERROR] missing checkpoint:"
        echo "  ${CKPT}"
        exit 3
    fi

    echo "[FOUND] epoch ${EP}: ${CKPT}"
done

echo
echo "[PASS] all 5 checkpoints exist"
echo

# Start with clean selection output.
rm -rf "${SELECT_ROOT}"

# --------------------------------------------------------------------
# Evaluate every checkpoint explicitly.
#
# tools/test.py itself creates:
#
# outputs/stream_kitti_models/
#   <cfg_tag>.<exp_name>/
#     eval/epoch_<N>/val/<eval_tag>/log_eval.txt
# --------------------------------------------------------------------

for EP in 1 2 3 4 5
do
    CKPT="${TRAIN_ROOT}/ckpt/checkpoint_epoch_${EP}.pth"

    echo
    echo "================================================================================"
    echo "EVALUATING EPOCH ${EP}/5"
    echo "================================================================================"

    python tools/test.py \
        --cfg_file "${CFG}" \
        --ckpt "${CKPT}" \
        --batch_size 1 \
        --workers 4 \
        --exp_name "${SELECT_EXP}" \
        --eval_tag "${EVAL_TAG}"

    LOG="${SELECT_ROOT}/eval/epoch_${EP}/val/${EVAL_TAG}/log_eval.txt"

    if [ ! -f "${LOG}" ]; then
        echo "[ERROR] tools/test.py finished but expected log is missing:"
        echo "  ${LOG}"
        exit 4
    fi

    if ! grep -q "Car AP_R40@0.70" "${LOG}"; then
        echo "[ERROR] epoch ${EP} log does not contain Car AP_R40:"
        echo "  ${LOG}"
        exit 5
    fi

    if ! grep -q "Pedestrian AP_R40@0.50" "${LOG}"; then
        echo "[ERROR] epoch ${EP} log does not contain Pedestrian AP_R40:"
        echo "  ${LOG}"
        exit 5
    fi

    if ! grep -q "Cyclist AP_R40@0.50" "${LOG}"; then
        echo "[ERROR] epoch ${EP} log does not contain Cyclist AP_R40:"
        echo "  ${LOG}"
        exit 5
    fi

    echo "[PASS] epoch ${EP} evaluation log:"
    echo "  ${LOG}"
done

# --------------------------------------------------------------------
# Parse only strict KITTI 3D AP_R40 Moderate:
#
# Car:
#   AP_R40@0.70,0.70,0.70
#
# Ped/Cyc:
#   AP_R40@0.50,0.50,0.50
#
# AP tuple order:
#   Easy, Moderate, Hard
#
# -> take second number.
# --------------------------------------------------------------------

python - \
    "${TRAIN_ROOT}" \
    "${SELECT_ROOT}" \
    "${EVAL_TAG}" <<'PY'
import json
import re
import sys
from pathlib import Path

train_root = Path(sys.argv[1])
select_root = Path(sys.argv[2])
eval_tag = sys.argv[3]


def extract_moderate(text, cls):
    if cls == "Car":
        header_re = re.compile(
            r"Car AP_R40@0\.70,\s*0\.70,\s*0\.70:"
        )
    elif cls in ("Pedestrian", "Cyclist"):
        header_re = re.compile(
            rf"{cls} AP_R40@0\.50,\s*0\.50,\s*0\.50:"
        )
    else:
        raise ValueError(cls)

    lines = text.splitlines()

    for i, line in enumerate(lines):
        if not header_re.search(line):
            continue

        # The expected 3d AP line is immediately after bbox/bev,
        # but scan a bounded window defensively.
        for j in range(i + 1, min(i + 10, len(lines))):
            m = re.search(
                r"\b3d\s+AP:\s*"
                r"([0-9]+(?:\.[0-9]+)?),\s*"
                r"([0-9]+(?:\.[0-9]+)?),\s*"
                r"([0-9]+(?:\.[0-9]+)?)",
                lines[j],
            )

            if m:
                easy = float(m.group(1))
                moderate = float(m.group(2))
                hard = float(m.group(3))

                return {
                    "easy": easy,
                    "moderate": moderate,
                    "hard": hard,
                }

        raise RuntimeError(
            f"Found {cls} strict AP_R40 header "
            "but no following 3d AP line"
        )

    raise RuntimeError(
        f"Strict AP_R40 block not found for {cls}"
    )


rows = []

for ep in range(1, 6):
    log_path = (
        select_root
        / "eval"
        / f"epoch_{ep}"
        / "val"
        / eval_tag
        / "log_eval.txt"
    )

    if not log_path.is_file():
        raise FileNotFoundError(
            f"Missing expected evaluation log: {log_path}"
        )

    text = log_path.read_text(
        errors="ignore"
    )

    car = extract_moderate(
        text,
        "Car",
    )["moderate"]

    ped = extract_moderate(
        text,
        "Pedestrian",
    )["moderate"]

    cyc = extract_moderate(
        text,
        "Cyclist",
    )["moderate"]

    macro = (
        car + ped + cyc
    ) / 3.0

    rows.append({
        "epoch": ep,
        "Car": car,
        "Pedestrian": ped,
        "Cyclist": cyc,
        "Macro": macro,
        "log": str(log_path),
    })


print()
print("=" * 84)
print("STRICT 3D AP_R40 MODERATE")
print("=" * 84)

print(
    f"{'Epoch':>7}"
    f"{'Car':>12}"
    f"{'Ped':>12}"
    f"{'Cyc':>12}"
    f"{'Macro':>12}"
)

for r in rows:
    print(
        f"{r['epoch']:7d}"
        f"{r['Car']:12.4f}"
        f"{r['Pedestrian']:12.4f}"
        f"{r['Cyclist']:12.4f}"
        f"{r['Macro']:12.4f}"
    )


best = max(
    rows,
    key=lambda x: x["Macro"],
)

best_epoch = int(
    best["epoch"]
)

best_ckpt = (
    train_root
    / "ckpt"
    / f"checkpoint_epoch_{best_epoch}.pth"
)

if not best_ckpt.is_file():
    raise FileNotFoundError(
        best_ckpt
    )


(train_root / "best_epoch.txt").write_text(
    f"{best_epoch}\n"
)

(train_root / "best_checkpoint.txt").write_text(
    f"{best_ckpt}\n"
)

(train_root / "best_validation.json").write_text(
    json.dumps(
        {
            "selection_metric":
                "mean strict 3D AP_R40 Moderate "
                "over Car/Pedestrian/Cyclist",

            "best":
                best,

            "all_epochs":
                rows,

            "best_checkpoint":
                str(best_ckpt),
        },
        indent=2,
    )
    + "\n"
)


print()
print("=" * 84)
print(f"BEST EPOCH      = {best_epoch}")
print(f"BEST Car        = {best['Car']:.4f}")
print(f"BEST Pedestrian = {best['Pedestrian']:.4f}")
print(f"BEST Cyclist    = {best['Cyclist']:.4f}")
print(f"BEST Macro      = {best['Macro']:.4f}")
print(f"BEST CKPT       = {best_ckpt}")
print("=" * 84)

print()
print("[PASS] best checkpoint selected and recorded")
PY

echo
echo "================================================================================"
echo "SELECTION COMPLETE"
echo "================================================================================"
cat "${TRAIN_ROOT}/best_epoch.txt"
cat "${TRAIN_ROOT}/best_checkpoint.txt"
