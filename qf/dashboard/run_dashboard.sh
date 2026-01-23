#!/usr/bin/env bash
"""
run_dashboard.sh

Helper to run the Streamlit dashboard with a chosen Conda environment and operation.

Usage examples:
  ./run_dashboard.sh --env qf --operation run-dashboard
  ./run_dashboard.sh --env qf --operation run-dashboard-schk
  ./run_dashboard.sh --env qf --operation run-dashboard-develop --port 8600
  ./run_dashboard.sh --helper

Operations:
  run-dashboard         : start the dashboard (default server options)
  run-dashboard-schk    : run python -m py_compile on qf/dashboard/*.py inside the env
  run-dashboard-develop : start dashboard with file watcher (watchdog) and print the command (useful for dev)

Options:
  --env <name>         : conda environment name (default: qf)
  --port <port>        : server port (default: 8501)
  --helper             : show this help text
"""

set -euo pipefail

ENV_NAME="qf"
PORT=8501
OPERATION="run-dashboard"

print_help() {
  sed -n '1,200p' "$0" | sed -n '1,200p'
  echo ""
  echo "Usage: $0 [--env <name>] [--port <port>] [--operation <op>] [--helper]"
  echo "Operations: run-dashboard | run-dashboard-schk | run-dashboard-develop"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="$2"; shift 2;;
    --port)
      PORT="$2"; shift 2;;
    --operation)
      OPERATION="$2"; shift 2;;
    --helper|-h|--help)
      print_help; exit 0;;
    *)
      echo "Unknown arg: $1"; print_help; exit 2;;
  esac
done

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
CD_CMD=(cd "$BASE_DIR")

case "$OPERATION" in
  run-dashboard)
    CMD=(conda run -n "$ENV_NAME" streamlit run dashboard.py --server.port "$PORT")
    echo "Starting dashboard in env '$ENV_NAME' on port $PORT..."
    echo "Command: ${CMD[*]}"
    cd "$BASE_DIR" && "${CMD[@]}"
    ;;

  run-dashboard-schk)
    CMD=(conda run -n "$ENV_NAME" python -m py_compile qf/dashboard/*.py)
    echo "Running syntax check (py_compile) in env '$ENV_NAME'..."
    echo "Command: ${CMD[*]}"
    cd "$BASE_DIR" && "${CMD[@]}"
    ;;

  run-dashboard-develop)
    # development run with file watcher and runOnSave
    CMD=(conda run -n "$ENV_NAME" streamlit run dashboard.py --server.fileWatcherType=watchdog --server.runOnSave=true --server.port "$PORT")
    echo "Development dashboard (watch mode) in env '$ENV_NAME' on port $PORT"
    echo "Command: ${CMD[*]}"
    echo "You can open: http://127.0.0.1:$PORT"
    cd "$BASE_DIR" && "${CMD[@]}"
    ;;

  *)
    echo "Unknown operation: $OPERATION"; print_help; exit 2;;
esac
