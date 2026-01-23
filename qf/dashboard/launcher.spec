# PyInstaller spec file for building the dashboard launcher.
from pathlib import Path
import sys

block_cipher = None

here = Path(__file__).resolve().parent

a = Analysis(
    [str(here / 'launcher.py')],
    pathex=[str(here)],
    binaries=[],
    datas=[(str(here / 'launcher.py'), '.')],
    hiddenimports=['webview', 'gi', 'webview.platforms.gtk'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='dashboard-launcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
