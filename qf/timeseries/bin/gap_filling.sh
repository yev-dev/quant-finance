#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# gap_filling.sh — Launcher for gap_filling.py
#
# Usage:
#   ./bin/gap_filling.sh                    # runs with default args (MSFT)
#   ./bin/gap_filling.sh --target AAPL      # single ticker
#   ./bin/gap_filling.sh --target ALL        # all sectors
#   ./bin/gap_filling.sh --target MSFT,AAPL  # comma-separated list
#
# Environment:
#   RESULT_PATH    — root output directory (default: ./results)
#   CONDA_ENV      — conda environment name (default: qf)
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_PATH="${RESULT_PATH:-${SCRIPT_DIR}/results}"
CONDA_ENV="${CONDA_ENV:-qf}"

# Resolve conda
if command -v conda &>/dev/null; then
    echo "  → Using conda env '${CONDA_ENV}'"
    exec conda run -n "${CONDA_ENV}" python "${SCRIPT_DIR}/gap_filling.py" \
        --data-dir "${SCRIPT_DIR}/notebooks/data" \
        --output-dir "${RESULT_PATH}" \
        "$@"
elif command -v python3 &>/dev/null; then
    echo "  → Using system python3"
    exec python3 "${SCRIPT_DIR}/gap_filling.py" \
        --data-dir "${SCRIPT_DIR}/notebooks/data" \
        --output-dir "${RESULT_PATH}" \
        "$@"
else
    echo "ERROR: No Python interpreter found."
    exit 1
fi
