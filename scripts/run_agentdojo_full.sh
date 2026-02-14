#!/usr/bin/env bash
# run_agentdojo_full.sh -- Run AgentDojo full benchmark (all suites).
#
# Environment variables:
#   MODEL    -- model key from configs/models.yaml  (default: model_a)
#   DEFENSE  -- defense key from configs/defenses.yaml (default: rvu)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Activate venv if available.
if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

MODEL="${MODEL:-model_a}"
DEFENSE="${DEFENSE:-rvu}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="agentdojo_full_${MODEL}_${DEFENSE}_${TIMESTAMP}"
OUT_DIR="${REPO_ROOT}/artifacts/runs/${RUN_ID}"
mkdir -p "${OUT_DIR}"

echo "================================================================"
echo "  AgentDojo Full Benchmark"
echo "  Model   : ${MODEL}"
echo "  Defense : ${DEFENSE}"
echo "  Run ID  : ${RUN_ID}"
echo "  Output  : ${OUT_DIR}"
echo "================================================================"

python3 -m rvu.harness.agentdojo_runner \
    --config "${REPO_ROOT}/configs/agentdojo.yaml" \
    --model "${MODEL}" \
    --defense "${DEFENSE}" \
    --mode full \
    --logdir "${OUT_DIR}" \
    "$@"

echo ""
echo "Done. Results at: ${OUT_DIR}"
