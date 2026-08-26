#!/usr/bin/env bash

set -euo pipefail


ROOT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

cd "${ROOT_DIR}"


PYTHON_BIN="${PYTHON_BIN:-python}"

OUT_ROOT="${OUT_ROOT:-outputs/single_roi}"

FORCE="${FORCE:-0}"

PRINT_EVERY="${PRINT_EVERY:-200}"


SIZES=(
    96
    128
    160
    192
    224
)


echo "========================================"
echo "Observation 1"
echo "Spatial redundancy + task sensitivity"
echo "========================================"
echo "repo      : ${ROOT_DIR}"
echo "python    : ${PYTHON_BIN}"
echo "out root  : ${OUT_ROOT}"
echo "force     : ${FORCE}"
echo


run_roi()
{
    local size="$1"

    local out_dir="${OUT_ROOT}/oracle_reuse_${size}"
    local summary="${out_dir}/summary.json"

    if [[ "${FORCE}" != "1" && -f "${summary}" ]]; then
        echo
        echo "[SKIP] ${size}x${size}"
        echo "       found ${summary}"
        return
    fi

    echo
    echo "========================================"
    echo "Running ROI ${size}x${size}"
    echo "========================================"

    mkdir -p "${out_dir}"

    "${PYTHON_BIN}" -u \
        tools/eval_single_roi_bev_semantic.py \
        --center-strategy oracle \
        --outside-mode reuse \
        --roi-h "${size}" \
        --roi-w "${size}" \
        --print-every "${PRINT_EVERY}" \
        --output "${out_dir}" \
        2>&1 | tee "${out_dir}/run.log"

    test -f "${summary}" || {
        echo "[ERROR] Missing ${summary}"
        exit 1
    }
}


run_full()
{
    local out_dir="${OUT_ROOT}/full_baseline"
    local summary="${out_dir}/summary.json"

    if [[ "${FORCE}" != "1" && -f "${summary}" ]]; then
        echo
        echo "[SKIP] Full baseline"
        echo "       found ${summary}"
        return
    fi

    echo
    echo "========================================"
    echo "Running Full baseline"
    echo "========================================"

    mkdir -p "${out_dir}"

    # roi-h / roi-w have no semantic effect in Full mode.
    "${PYTHON_BIN}" -u \
        tools/eval_single_roi_bev_semantic.py \
        --center-strategy full \
        --outside-mode reuse \
        --roi-h 160 \
        --roi-w 160 \
        --print-every "${PRINT_EVERY}" \
        --output "${out_dir}" \
        2>&1 | tee "${out_dir}/run.log"

    test -f "${summary}" || {
        echo "[ERROR] Missing ${summary}"
        exit 1
    }
}


for size in "${SIZES[@]}"; do
    run_roi "${size}"
done


run_full


echo
echo "========================================"
echo "Plotting Figure 1"
echo "========================================"

"${PYTHON_BIN}" -u \
    tools/plot_fig1.py \
    --root "${OUT_ROOT}" \
    --sizes "${SIZES[@]}" \
    --roi-pattern 'oracle_reuse_{size}' \
    --full-dir full_baseline \
    --outdir "${OUT_ROOT}/observation1_fig1"


echo
echo "========================================"
echo "Observation 1 complete"
echo "========================================"
echo
echo "Results:"
echo "  ${OUT_ROOT}/observation1_fig1/fig1_data.csv"
echo "  ${OUT_ROOT}/observation1_fig1/fig1a_feature_l1.pdf"
echo "  ${OUT_ROOT}/observation1_fig1/fig1b_ap3d.pdf"
echo
