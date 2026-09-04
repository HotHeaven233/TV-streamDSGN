#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EPOCH="${1:-12}"
HZ="${2:-10}"

for W in \
    "1.0,1.0,1.0,1.0,1.0,1.0" \
    "0.75,0.75,0.75,0.75,0.75,0.75" \
    "0.5,0.5,0.5,0.5,0.5,0.5" \
    "0.25,0.25,0.25,0.25,0.25,0.25" \
    "1.0,1.0,0.75,0.5,0.5,0.25" \
    "1.0,1.0,1.0,0.75,0.5,0.25"
do
    "${SCRIPT_DIR}/19_test_elastic_v4_bn.sh" \
        "${EPOCH}" elastic_fixed "${HZ}" "${W}"
done

