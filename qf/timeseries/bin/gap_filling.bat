@echo off
REM ─────────────────────────────────────────────────────────────────
REM gap_filling.bat — Launcher for gap_filling.py (Windows)
REM
REM Usage:
REM   bin\gap_filling.bat                    -- runs with default args (MSFT)
REM   bin\gap_filling.bat --target AAPL      -- single ticker
REM   bin\gap_filling.bat --target ALL        -- all sectors
REM   bin\gap_filling.bat --target MSFT,AAPL  -- comma-separated list
REM
REM Environment:
REM   RESULT_PATH    — root output directory (default: .\results)
REM   CONDA_ENV      — conda environment name (default: qf)
REM ─────────────────────────────────────────────────────────────────
setlocal

set "SCRIPT_DIR=%~dp0.."
if "%RESULT_PATH%"=="" set "RESULT_PATH=%SCRIPT_DIR%\results"
if "%CONDA_ENV%"=="" set "CONDA_ENV=qf"

echo  → Using conda env '%CONDA_ENV%'
call conda run -n "%CONDA_ENV%" python "%SCRIPT_DIR%\gap_filling.py" ^
    --data-dir "%SCRIPT_DIR%\notebooks\data" ^
    --output-dir "%RESULT_PATH%" ^
    %*

if %ERRORLEVEL% neq 0 (
    echo ERROR: gap_filling.py exited with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
