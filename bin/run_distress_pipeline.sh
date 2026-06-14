#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_distress_pipeline.sh — Unix entry-point for the Distressed Stock Analysis
# ──────────────────────────────────────────────────────────────────────────────
# Usage:
#   ./bin/run_distress_pipeline.sh [--config PATH] [--no-graphs] [--no-save]
#
# Environment variables (optional):
#   DISTRESSED_DATA_DIR   Override data cache directory
#   DATA_DIR              Fallback for data cache directory
#   DISTRESSED_OUTPUT_DIR Override output directory
#   OUTPUT_DIR            Fallback for output directory
#   CONDA_PREFIX          Conda environment (default: qf)
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Activate conda environment ───────────────────────────────────────────────
ENV_NAME="${CONDA_DEFAULT_ENV:-qf}"
if command -v conda &>/dev/null; then
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME" 2>/dev/null || echo "Warning: could not activate conda env '$ENV_NAME'"
fi

# ── Run the pipeline ─────────────────────────────────────────────────────────
cd "$PROJECT_ROOT"

export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT"

echo "═══════════════════════════════════════════════════════════════"
echo "  Distressed Stock Analysis Pipeline"
echo "  Project: $PROJECT_ROOT"
echo "  Env:     $ENV_NAME"
echo "═══════════════════════════════════════════════════════════════"

python -m qf.timeseries.distress_analysis "$@"

echo ""
echo "Pipeline finished. Output in: $(python -c "import os; print(os.environ.get('DISTRESSED_OUTPUT_DIR', os.environ.get('OUTPUT_DIR', '$PROJECT_ROOT/distress_output')))")"
