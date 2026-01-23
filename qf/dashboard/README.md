# Dashboard

A Streamlit-based dashboard for discovering and running Python runner modules as subprocesses. It scans the `runners/` package for modules, infers CLI parameters (via argparse), and executes them inside the `qf` conda environment. The app also provides logs of recent runs and a simple editor for the Streamlit config.

Note: The Streamlit Deploy/Share button is intentionally hidden to prevent accidental cloud deployment from this operational dashboard.

## Table of Contents

- [Dashboard tabs overview](#dashboard-tabs-overview)
  - [Script Runners](#script-runners)
  - [Logs](#logs)
  - [Config](#config)
  - [Reconciliation](#reconciliation)
  - [Tools](#tools)
  - [Docs](#docs)
- [Centralized logs](#centralized-logs)
- [Quick start](#quick-start)
- [Configuration reference (excerpt)](#configuration-reference-excerpt)
- [Troubleshooting](#troubleshooting)
- [Reconciliation tab](#reconciliation-tab)
- [Folder layout](#folder-layout)
- [Requirements](#requirements)
- [Setup](#setup)
- [How to run](#how-to-run)
- [Using the Script Runners tab](#using-the-script-runners-tab)
- [Logs tab](#logs-tab)
- [Config tab](#config-tab)
- [Auto-reload and file watching](#auto-reload-and-file-watching)
- [Example runner modules](#example-runner-modules)

## Dashboard tabs overview

This dashboard is organized into tabs. Here’s what each tab contains and how to use it:

### Script Runners
- Discover and run modules from the configured “scripts package” (defaults to `runners` or the package(s) defined in `.streamlit/config.toml`).
- Argparse-backed modules: parameters are auto-detected and rendered as inputs.
- Non-argparse modules: enter date tokens (COB_DATE, RUN_DATE, START_DATE, END_DATE) and choose how to pass them (argv tokens or environment variables).
- Environment selector: loads values from `~/.dashboard/config.ini` (e.g., `CONDA_ENV`, `CONFIG_PATH`, `RESULT_PATH`).
- Backend choice: run via Conda (`conda run -n <env> python -m ...`) or via the environment’s Python interpreter path.
- Config preset editor: generate or load a per-module JSON preset; includes a raw JSON editor to modify any node and buttons to Validate, Apply, Save, Create default, Dry run, and Run with Config.
- Detach mode: start long runs asynchronously and view live tails; you can terminate, refresh, or attach the tail to the dashboard log.
- Scripts package selection: if `.streamlit/config.toml` has a list for `runners_package`, pick one from the dropdown or enter a custom value; switching refreshes the discovered module list.
- Historical Log Runs: shows a timeline of executed commands with combined stdout+stderr.

### Logs
- Shows the tail of the centralized dashboard log at `~/.dashboard/logs/dashboard.log`.
- Use Refresh to reload the tail.
- All detached runs write their stdout/stderr to `~/.dashboard/logs/runs/*.log` and can auto-stream into the main log when auto-attach is enabled.

### Config
- Environment Config (`~/.dashboard/config.ini`): edit existing keys for selected env (DEV/UAT/PROD). Save persists to the home config.
- Validation: checks whether the selected env’s `CONDA_ENV` exists.
- Streamlit Config: edit `.streamlit/config.toml` in-place.
  - Supported keys include:
    - `runners_package` (string or array): one or more importable packages to discover modules from.
    - `dashboard_name`: title rendered at the top of the app.
    - `default_notebook_dir`: default folder for starting Jupyter.
    - `auto_attach` (bool): whether new processes auto-stream their logs into dashboard.log.
    - `auto_attach_interval` (float): seconds between tail checks.

### Reconciliation
- Compare CSVs by a reconciliation key.
  - Mode “Directories”: reconcile CSVs found in both BEFORE and AFTER directories by filename; aggregate differences.
  - Mode “Files”: directly compare a BEFORE file to an AFTER file.
- Optional “Preset Env” to auto-fill directories via `RESULT_PATH` from env config.
- Built-in directory/file browsers and CSV uploaders.
- Export the text report via the Download button.

### Tools
- Check Environment: runs `check_environment.py` (from a parent `tools/` folder) inside the selected conda env and shows output.
- Jupyter Notebook: start a Jupyter server in a chosen directory and env; detached with live tails.
- Terminal Script: run any shell command (optionally inside conda); detached with live tails and termination controls.
- Kill Process by PID: send termination signals manually.
- Terminate All: stop all tracked processes started from the dashboard.
- Running Processes: interactive table with multi-select terminate.
- Logs Maintenance: “Clear All Logs” truncates `dashboard.log`, removes rotated logs, and clears `~/.dashboard/logs/runs/*`.

### Docs
- Renders this `README.md` within the app for quick reference (Docs tab).

## Centralized logs
- Main dashboard log: `~/.dashboard/logs/dashboard.log` (rotated).
- Per-run logs: `~/.dashboard/logs/runs/*.out.log` and `*.err.log`.
- Auto-attach tailers: when enabled, background tailers stream new output into the main dashboard log; defaults are controlled via `.streamlit/config.toml`.

## Quick start
1. Ensure the `qf` conda environment exists with `streamlit` and `watchdog` installed.
2. Launch the app (via VS Code task or):

```bash
conda run -n qf streamlit run dashboard.py --server.fileWatcherType=watchdog --server.runOnSave=true
```

3. Open Script Runners, pick a module, fill parameters, and run.
4. Check Logs for dashboard.log tail; use Tools for Jupyter or Terminal runs.

## Configuration reference (excerpt)

```toml
[server]
fileWatcherType = "watchdog"
runOnSave = true

[dashboard]
runners_package = [
  "qf.dashboard.runners",
  "qf.tools",
  "qf.scripts"
]
dashboard_name = "Operational Dashboard"
default_notebook_dir = "~"
auto_attach = true
auto_attach_interval = 2.0
```

## Troubleshooting
- Conda env not found: ensure `conda env list` shows your env; update `CONDA_ENV` in `~/.dashboard/config.ini`.
- No modules discovered: check that your scripts reside in the configured `runners_package` and are importable.
- Permission errors on logs: ensure your user can create `~/.dashboard/logs` and `runs` subfolder.
- Streamlit not reloading: verify `fileWatcherType = "watchdog"` and `runOnSave = true` are set; ensure the app is launched via the intended environment.

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

## Run helper scripts

For convenience this repo includes small helper scripts to run the dashboard using a specified Conda environment and a few common operations.

POSIX (bash/zsh):

```bash
# Start the dashboard in conda env `qf` (default port 8501)
./run_dashboard.sh --env qf --operation run-dashboard

# Run a quick syntax check (py_compile) inside the env
./run_dashboard.sh --env qf --operation run-dashboard-schk

# Development run with watchdog file-watcher and printed command
./run_dashboard.sh --env qf --operation run-dashboard-develop --port 8600

# Show helper text
./run_dashboard.sh --helper
```

Windows (CMD):

```cmd
REM Start the dashboard (uses conda run)
run_dashboard.bat --env qf --operation run-dashboard

REM Syntax check
run_dashboard.bat --env qf --operation run-dashboard-schk

REM Development run with watchdog
run_dashboard.bat --env qf --operation run-dashboard-develop --port 8600

REM Show help
run_dashboard.bat --helper
```

Windows (PowerShell):

```powershell
# If you prefer PowerShell use the run_dashboard.ps1 helper
.
.\run_dashboard.ps1 -Env qf -Operation run-dashboard
# or
.\run_dashboard.ps1 -Env qf -Operation run-dashboard-develop -Port 8600
```

## Quick developer checks

Before opening the UI you can run a couple of quick checks locally to catch syntax issues and ensure Streamlit will use the file-watcher you expect.

- Syntax (quick compile check):

  ```bash
  # From the repository root or the dashboard folder
  python -m py_compile qf/dashboard/dashboard.py
  # If this completes with no output the file compiles to bytecode (no syntax errors).
  ```

- Run Streamlit with the watchdog file watcher (auto-reload on save):

  ```bash
  # Recommended: run Streamlit from the qf conda env so the app uses the correct interpreter
  conda run -n qf streamlit run dashboard.py --server.fileWatcherType=watchdog --server.runOnSave=true

  # Or activate the env then run:
  # conda activate qf
  # streamlit run dashboard.py --server.fileWatcherType=watchdog --server.runOnSave=true
  ```

If you see Streamlit failing to reload on save, confirm `watchdog` is installed in `qf`:

```bash
conda run -n qf pip show watchdog || conda run -n qf pip install watchdog
```

## Packaging for Windows

If you want a native Windows executable (single-file) for the dashboard launcher, use PyInstaller from a Windows environment that has the required packages installed (recommended inside the `qf` environment). The repository includes helper scripts in this folder:

- `build_desktop_windows.bat` — simple .bat wrapper to run PyInstaller with sensible defaults.
- `build_desktop_windows.ps1` — PowerShell helper with optional parameters.

High-level steps (on Windows):

1. Open a Developer PowerShell for Visual Studio or ensure the required MSVC build tools are available (PyInstaller may require the Microsoft Visual C++ Redistributable and build toolchain when bundling C extensions).
2. Activate the Python environment that contains `streamlit` and `pywebview`:

```powershell
conda activate qf
pip install pywebview pyinstaller requests
```

3. From the `qf/dashboard` folder run the bundled helper (PowerShell):

```powershell
./build_desktop_windows.ps1 -OutputDir .\dist
```

Or using the batch file:

```cmd
build_desktop_windows.bat
```

Notes and common issues on Windows
- Visual C++: If the build fails with missing MSVC tools, install "Build Tools for Visual Studio" or the full Visual Studio with C++ workload.
- VC Redistributable: The target machines may need the Visual C++ Redistributable installed to run the final .exe.
- Anti-virus: Some Windows anti-virus products flag single-file executables produced by PyInstaller; codesign the binary if distributing widely.
- Testing: Run the produced `dashboard-launcher.exe` and verify it starts the server and opens a window. If the window is blank or fails, check the `stdout`/`stderr` logs that PyInstaller can produce when `--console` is enabled.
- Hidden imports/data: If modules (e.g., `webview.platforms.win32`) fail to import at runtime, add `--hidden-import` options to the PyInstaller call or adjust the included `datas` in the `.spec`.

If you want, I can further tune the `.spec` file for Windows (add common hidden-imports and data files) and test a small build locally in this environment if you grant permission to run PyInstaller.




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
