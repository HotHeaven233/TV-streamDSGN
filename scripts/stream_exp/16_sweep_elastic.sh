#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPOCH="${1:-20}"
HZ="${2:-10}"

# Exact existing K3 Full baseline.
"${SCRIPT_DIR}/15_test_elastic.sh" "${EPOCH}" baseline_full "${HZ}"

# Uniform width ablations across all six elastic stages.
"${SCRIPT_DIR}/15_test_elastic.sh" "${EPOCH}" elastic_fixed "${HZ}" 1,1,1,1,1,1
"${SCRIPT_DIR}/15_test_elastic.sh" "${EPOCH}" elastic_fixed "${HZ}" 0.75,0.75,0.75,0.75,0.75,0.75
"${SCRIPT_DIR}/15_test_elastic.sh" "${EPOCH}" elastic_fixed "${HZ}" 0.5,0.5,0.5,0.5,0.5,0.5
"${SCRIPT_DIR}/15_test_elastic.sh" "${EPOCH}" elastic_fixed "${HZ}" 0.25,0.25,0.25,0.25,0.25,0.25

# Two progressive examples: keep early visual stages wider and compress later.
"${SCRIPT_DIR}/15_test_elastic.sh" "${EPOCH}" elastic_fixed "${HZ}" 1,0.75,0.75,0.5,0.5,0.25
"${SCRIPT_DIR}/15_test_elastic.sh" "${EPOCH}" elastic_fixed "${HZ}" 0.75,0.75,0.5,0.5,0.25,0.25

# Runtime timing-aware controller.
"${SCRIPT_DIR}/15_test_elastic.sh" "${EPOCH}" dynamic "${HZ}"
