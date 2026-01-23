Run Dashboard helper scripts

This repository includes small helper scripts to run the Streamlit dashboard found in `qf/dashboard`.
They make it easy to run the app, run a quick syntax check, or start a development instance with the file-watcher.

Examples (run these from the repository root):

POSIX (bash/zsh):

```bash
cd qf/dashboard
./run_dashboard.sh --env qf --operation run-dashboard

# syntax check
./run_dashboard.sh --env qf --operation run-dashboard-schk

# development run (watch mode) printing the command and exposing the port
./run_dashboard.sh --env qf --operation run-dashboard-develop --port 8600
```

Windows (CMD):

```cmd
cd qf\dashboard
run_dashboard.bat --env qf --operation run-dashboard

# syntax check
run_dashboard.bat --env qf --operation run-dashboard-schk

# development run
run_dashboard.bat --env qf --operation run-dashboard-develop --port 8600
```

PowerShell:

```powershell
cd qf\dashboard
.\run_dashboard.ps1 -Env qf -Operation run-dashboard
```

These helpers simply call `conda run -n <env> streamlit run dashboard.py` (or the equivalent py_compile check). They are convenient defaults — feel free to adapt their flags to your local workflow.
