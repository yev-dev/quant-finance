@echo off
rem run_dashboard.bat
rem Helper to run the Streamlit dashboard with a chosen Conda environment and operation.

setlocal ENABLEDELAYEDEXPANSION

set "ENV_NAME=qf"
set "PORT=8501"
set "OPERATION=run-dashboard"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--env" (
  set "ENV_NAME=%~2"
  shift
  shift
  goto parse_args
)
if /I "%~1"=="--port" (
  set "PORT=%~2"
  shift
  shift
  goto parse_args
)
if /I "%~1"=="--operation" (
  set "OPERATION=%~2"
  shift
  shift
  goto parse_args
)
if /I "%~1"=="--helper" (
  goto helper
)
echo Unknown argument: %~1
goto usage

:args_done

:run_op
if /I "%OPERATION%"=="run-dashboard" (
  echo Starting dashboard in env '%ENV_NAME%' on port %PORT%
  echo Command: conda run -n %ENV_NAME% streamlit run dashboard.py --server.port %PORT%
  conda run -n %ENV_NAME% streamlit run dashboard.py --server.port %PORT%
  goto end
)

if /I "%OPERATION%"=="run-dashboard-schk" (
  echo Running py_compile check in env '%ENV_NAME%'
  echo Command: conda run -n %ENV_NAME% python -m py_compile qf/dashboard/*.py
  conda run -n %ENV_NAME% python -m py_compile qf/dashboard/*.py
  goto end
)

if /I "%OPERATION%"=="run-dashboard-develop" (
  echo Development run (watch mode) in env '%ENV_NAME%' on port %PORT%
  echo Command: conda run -n %ENV_NAME% streamlit run dashboard.py --server.fileWatcherType=watchdog --server.runOnSave=true --server.port %PORT%
  conda run -n %ENV_NAME% streamlit run dashboard.py --server.fileWatcherType=watchdog --server.runOnSave=true --server.port %PORT%
  goto end
)

echo Unknown operation: %OPERATION%
goto usage

:helper
echo Usage: run_dashboard.bat [--env <name>] [--port <port>] [--operation <op>] [--helper]
echo Operations: run-dashboard ^| run-dashboard-schk ^| run-dashboard-develop
goto end

:usage
echo.
echo Usage: run_dashboard.bat [--env ^<name^>] [--port ^<port^>] [--operation ^<op^>] [--helper]
echo.
echo Example: run_dashboard.bat --env qf --operation run-dashboard-develop --port 8600

:end
endlocal
