@echo off
REM ────────────────────────────────────────────────────────────────────────────
REM run_distress_pipeline.bat — Windows entry-point for the Distressed Stock
REM                            Analysis Pipeline
REM ────────────────────────────────────────────────────────────────────────────
REM Usage:
REM   bin\run_distress_pipeline.bat [--config PATH] [--no-graphs] [--no-save]
REM
REM Environment variables (optional):
REM   DISTRESSED_DATA_DIR   Override data cache directory
REM   DATA_DIR              Fallback for data cache directory
REM   DISTRESSED_OUTPUT_DIR Override output directory
REM   OUTPUT_DIR            Fallback for output directory
REM ────────────────────────────────────────────────────────────────────────────

@echo ═══════════════════════════════════════════════════════════════
@echo   Distressed Stock Analysis Pipeline
@echo ═══════════════════════════════════════════════════════════════

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

set PYTHONPATH=%PYTHONPATH%;%PROJECT_ROOT%

python -m qf.timeseries.distress_analysis %*

echo.
echo Pipeline finished.
echo Output directory: %DISTRESSED_OUTPUT_DIR%  (or %%PROJECT_ROOT%%\distress_output)
