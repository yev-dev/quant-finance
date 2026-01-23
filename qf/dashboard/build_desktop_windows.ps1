param(
    [string]$OutputDir = "dist",
    [string]$Python = "python",
    [switch]$Console
)

Write-Host "Using Python: $Python"

Write-Host "Ensuring PyInstaller and runtime deps are installed..."
& $Python -m pip install --upgrade pyinstaller pywebview requests | Out-Null

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

$hiddenImports = @("webview", "webview.platforms.win32")
$hiddenArgs = $hiddenImports | ForEach-Object { "--hidden-import=$_" }

$addData = "launcher.py;."

$cmd = @($Python, "-m", "PyInstaller", "--onefile", "--noconfirm", "--name", "dashboard-launcher") + $hiddenArgs + @("--add-data", $addData)
if ($Console) { $cmd += "--console" }
$cmd += "launcher.py"

Write-Host "Running: $($cmd -join ' ')"
& $cmd

if (Test-Path "dist\dashboard-launcher.exe") {
    Move-Item -Force "dist\dashboard-launcher.exe" "$OutputDir\dashboard-launcher.exe"
    Write-Host "Build complete: $OutputDir\dashboard-launcher.exe"
} else {
    Write-Error "Build failed. Check PyInstaller output above."
}
