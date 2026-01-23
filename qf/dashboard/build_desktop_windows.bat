@echo off
REM Build a standalone Windows executable with PyInstaller (batch helper)
SETLOCAL

REM Use Python from the active environment. If conda isn't activated, set PY accordingly.
IF "%PY%"=="" SET "PY=python"

echo Using Python: %PY%

REM Ensure pyinstaller is installed in the active env
%PY% -m pip install --upgrade pyinstaller pywebview requests >NUL 2>&1

REM Where to output the built exe
SET OUTDIR=dist
mkdir %OUTDIR% 2>NUL || REM ignore

REM Common hidden imports (add more if runtime import errors appear)
SET HIDDEN=--hidden-import=webview --hidden-import=webview.platforms.win32

REM Include the launcher script as a data file so sys._MEIPASS access works if needed
SET ADDFILES=--add-data "launcher.py;."

echo Building dashboard-launcher.exe ...
%PY% -m PyInstaller --onefile --noconfirm --name dashboard-launcher %HIDDEN% %ADDFILES% launcher.py

IF EXIST dist\dashboard-launcher.exe (
  move /Y dist\dashboard-launcher.exe %OUTDIR%\dashboard-launcher.exe >NUL
  echo Build complete: %OUTDIR%\dashboard-launcher.exe
) ELSE (
  echo Build failed; check PyInstaller output above.
)

ENDLOCAL