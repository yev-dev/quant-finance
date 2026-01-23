param(
    [string]$Env = "qf",
    [int]$Port = 8501,
    [string]$Operation = "run-dashboard",
    [switch]$Helper
)

function Show-Help {
    Write-Host "Usage: .\run_dashboard.ps1 -Env <name> -Operation <op> [-Port <port>] [-Helper]"
    Write-Host "Operations: run-dashboard | run-dashboard-schk | run-dashboard-develop"
}

if ($Helper) { Show-Help; exit 0 }

switch ($Operation) {
    "run-dashboard" {
        $cmd = "conda run -n $Env streamlit run dashboard.py --server.port $Port"
        Write-Host "Starting dashboard in env '$Env' on port $Port"
        Write-Host "Command: $cmd"
        & conda run -n $Env streamlit run dashboard.py --server.port $Port
    }
    "run-dashboard-schk" {
        $cmd = "conda run -n $Env python -m py_compile qf/dashboard/*.py"
        Write-Host "Running py_compile in env '$Env'"
        Write-Host "Command: $cmd"
        & conda run -n $Env python -m py_compile qf/dashboard/*.py
    }
    "run-dashboard-develop" {
        $cmd = "conda run -n $Env streamlit run dashboard.py --server.fileWatcherType=watchdog --server.runOnSave=true --server.port $Port"
        Write-Host "Development run (watch mode) in env '$Env' on port $Port"
        Write-Host "Command: $cmd"
        & conda run -n $Env streamlit run dashboard.py --server.fileWatcherType=watchdog --server.runOnSave=true --server.port $Port
    }
    Default {
        Write-Error "Unknown operation: $Operation"
        Show-Help
        exit 2
    }
}
