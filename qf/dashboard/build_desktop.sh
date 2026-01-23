#!/usr/bin/env bash
# Small helper to build a standalone executable with PyInstaller.
# Usage: ./build_desktop.sh [output-dir]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-$HERE/dist}"
mkdir -p "$OUT_DIR"

# Use the Python interpreter that has the app's deps installed
PY="${PYTHON:-python3}"

# Common hidden imports for pywebview and streamlit; add more if your build fails
HIDDEN_IMPORTS=(--hidden-import=webview --hidden-import=gi --hidden-import=webview.platforms.gtk)

# Include the dashboard sources (templates/static files) if any - adjust as needed
DATA=("$HERE/launcher.py:.")

echo "Building desktop launcher into $OUT_DIR"

"$PY" -m PyInstaller \ 
  --onefile \ 
  --noconfirm \ 
  --add-data "${DATA[0]}" \ 
  "${HIDDEN_IMPORTS[@]}" \ 
  --name dashboard-launcher \ 
  "$HERE/launcher.py"

mv dist/dashboard-launcher "$OUT_DIR/" || true
echo "Build complete: $OUT_DIR/dashboard-launcher"
