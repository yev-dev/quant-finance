# Dashboard

A Streamlit-based dashboard for discovering and running Python runner modules as subprocesses. It scans the `runners/` package for modules, infers CLI parameters (via argparse), and executes them inside the `qf` conda environment. The app also provides logs of recent runs and a simple editor for the Streamlit config.

Note: The Streamlit Deploy/Share button is intentionally hidden to prevent accidental cloud deployment from this operational dashboard.

## Reconciliation tab

- Compare CSVs by a reconciliation key:
  - Mode "Directories": provide paths to a BEFORE and AFTER directory; it will reconcile CSV files present in both by filename and aggregate differences.
  - Mode "Files": provide a BEFORE and AFTER CSV file path to compare.
- Parameters:
  - Reconciliation key (CSV column name) — required.
  - Optional fields (comma-separated) — restricts comparison to the specified columns; leave empty to compare all shared columns.
- Output: A text report is shown and can be downloaded as a `.txt` file.

## Folder layout

- `dashboard.py` — Streamlit UI with tabs (Script Runners, Logs, Config)
- `scripts_runner.py` — Non-UI helpers (discover modules, parse argparse, build commands, run subprocess)
- `runners/` — Package containing your runnable Python modules (e.g., `first_model.py`, `second_model.py`)
- `.streamlit/config.toml` — Streamlit server configuration (auto-reload and watcher)

## Requirements

- Conda (Miniconda or Anaconda) installed and available on PATH
- Conda environment named `qf` with Python and dependencies
- Python packages in `qf` env:
  - `streamlit`
  - `watchdog` (for robust file watching)

## Setup

1. Ensure the `qf` environment exists:

```bash
conda env list
conda create -n qf python=3.11  # if needed
```

2. Install required packages into `qf`:

```bash
conda run -n qf pip install streamlit watchdog
```

3. (Optional) Verify Streamlit inside `qf`:

```bash
conda run -n qf python -c "import streamlit, watchdog; print('OK')"
```

## How to run

From the dashboard directory:

```bash
cd /Users/yevgeniy/Development/Projects/FinancialEngineering/quant-finance/qf/dashboard

# Option A: Run Streamlit via conda run (recommended for reliability)
conda run -n qf streamlit run dashboard.py

# Option B: Activate env and run (works if your shell supports conda activate)
# conda activate qf
# streamlit run dashboard.py
```

## Using the Script Runners tab

- Place your modules in `runners/` (e.g., `runners/first_model.py`).
- Define CLI arguments with `argparse.add_argument(...)` in each runner module.
- In the dashboard:
  - Select a module from the dropdown.
  - Review detected parameters in "Detected Parameters".
  - Fill inputs in the form and click "Run Module".
- The command that will be executed looks like:

```bash
conda run -n qf python -m runners.<module_name> [args]
```

## Logs tab

- Shows a history of recent runs (command, exit code, stdout, stderr).
- Click "Clear Logs" to wipe the history.

## Config tab

- Displays and lets you edit `.streamlit/config.toml`.
- Defaults provided by this project:

```toml
[server]
fileWatcherType = "watchdog"
runOnSave = true
```

- When you click "Save Config", the app writes changes and will auto-reload if the watcher is enabled.

## Auto-reload and file watching

- The dashboard imports runner modules so Streamlit’s watcher detects changes in `runners/` as well as `dashboard.py`.
- With `watchdog` and `runOnSave` enabled, saving files retriggers the app automatically.

## Troubleshooting

- "Import streamlit could not be resolved": install `streamlit` in the `qf` environment and run the app via `conda run -n qf`.
- "conda: command not found": install Miniconda/Anaconda and ensure `conda` is on PATH.
- Env `qf` doesn’t exist: create it via `conda create -n qf python=3.11`.
- Modules not detected: ensure `.py` files are inside `runners/` and that `runners/__init__.py` exists.
- Subprocess fails: check the command shown in the UI; verify dates/args and that required packages are installed in the `qf` env.

## Example runner modules

Minimal examples (place in `runners/`):

```python
# runners/first_model.py
import argparse
from datetime import datetime

def _parse_date(s: str):
    return datetime.strptime(s, "%Y-%m-%d").date()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", required=True, type=_parse_date)
    p.add_argument("--end-date", required=True, type=_parse_date)
    a = p.parse_args()
    print(f"[first_model] start={a.start_date} end={a.end_date}")
```

```python
# runners/second_model.py
import argparse
from datetime import datetime

def _parse_date(s: str):
    return datetime.strptime(s, "%Y-%m-%d").date()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", required=True, type=_parse_date)
    p.add_argument("--end-date", required=True, type=_parse_date)
    p.add_argument("--operation-type", default=None)
    a = p.parse_args()
    print(f"[second_model] start={a.start_date} end={a.end_date} op={a.operation_type}")
```

Run them from the dashboard UI by selecting the module and providing arguments.
