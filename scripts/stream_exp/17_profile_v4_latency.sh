#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-20}"
RUNTIME_OPT="${2:-cached_gn}"
FRAMES="${3:-160}"
SCHEDULE="${4:-}"

ELASTIC_CKPT="outputs/elastic_bev/elastic_bev_v3/ckpt/checkpoint_epoch_${EPOCH}.pth"

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"
require_file "${ELASTIC_CKPT}"

run_one() {
    local sched="$1"
    local tag
    tag="$(echo "${sched}" | tr ',' '_')"
    local out_dir="outputs/elastic_v4_latency_probe/e${EPOCH}/${RUNTIME_OPT}"
    mkdir -p "${out_dir}"

    echo
    echo "======================================================================"
    echo "v4 latency probe | ${RUNTIME_OPT} | ${sched}"
    echo "======================================================================"

    python tools/profile_elastic_v4_latency.py \
        --full_cfg "${FULL_CFG}" \
        --full_ckpt "${FULL_CKPT}" \
        --elastic_ckpt "${ELASTIC_CKPT}" \
        --fixed_schedule "${sched}" \
        --runtime_opt "${RUNTIME_OPT}" \
        --warmup 20 \
        --frames "${FRAMES}" \
        --output "${out_dir}/${tag}.json"
}

if [ -n "${SCHEDULE}" ]; then
    run_one "${SCHEDULE}"
else
    # Full is intentionally native original K3 in v4.
    run_one "1.0,1.0,1.0,1.0,1.0,1.0"
    run_one "0.75,0.75,0.75,0.75,0.75,0.75"
    run_one "0.5,0.5,0.5,0.5,0.5,0.5"
    run_one "0.25,0.25,0.25,0.25,0.25,0.25"

    # Representative native-Full-prefix -> Elastic suffix profiles.
    run_one "1.0,1.0,0.75,0.5,0.5,0.25"
    run_one "1.0,1.0,1.0,0.75,0.5,0.25"
fi

