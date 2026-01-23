#!/usr/bin/env python3
"""
Desktop launcher for the Streamlit dashboard.

This script starts the Streamlit server pointing at `dashboard.py`, waits until it
becomes reachable, then opens a native webview window (pywebview). When the window
is closed the Streamlit process is terminated.

Designed to be packaged with PyInstaller (onefile) or run directly from the repo.
"""
from __future__ import annotations

import os
import sys
import time
import signal
import socket
import subprocess
from pathlib import Path
from typing import Optional

try:
    import requests
except Exception:  # pragma: no cover - requests may not be available at import-time
    requests = None

try:
    import webview  # pywebview
except Exception:
    webview = None


def _base_dir() -> Path:
    """Return the base directory where the dashboard sources are located.

    When packaged by PyInstaller in onefile mode, resources are unpacked to sys._MEIPASS.
    """
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def find_free_port(start: int = 8501, end: int = 9000) -> int:
    for p in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("No free port found")


def start_streamlit(port: int, base_dir: Path, extra_args: Optional[list[str]] = None) -> subprocess.Popen:
    """Start Streamlit as a background process using the same Python interpreter.

    Returns the subprocess.Popen instance.
    """
    script = base_dir / "dashboard.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(script), "--server.headless", "true", "--server.port", str(port), "--server.enableCORS", "false"]
    # prefer watchdog for reliable reloads when available
    cmd += ["--server.fileWatcherType", "watchdog", "--server.runOnSave", "true"]
    if extra_args:
        cmd += extra_args
    env = os.environ.copy()
    # Ensure PYTHONUNBUFFERED to allow logs to stream (helpful during packaging)
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Start in a new process group so we can signal the whole group later
    proc = subprocess.Popen(cmd, cwd=str(base_dir), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    return proc


def wait_for_url(url: str, timeout: float = 30.0, interval: float = 0.25) -> bool:
    if requests is None:
        # import lazily if not available
        try:
            import requests as _requests

            rmod = _requests
        except Exception:
            rmod = None
    else:
        rmod = requests

    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if rmod:
                r = rmod.get(url, timeout=2)
                if r.status_code < 500:
                    return True
            else:
                # fallback to urllib
                from urllib.request import urlopen

                with urlopen(url, timeout=2) as f:
                    if f.status < 500:
                        return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def terminate_process(proc: subprocess.Popen) -> None:
    try:
        # send SIGTERM to the process group
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


def main() -> None:
    base = _base_dir()
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"

    if webview is None:
        print("pywebview is not installed. Install with: pip install pywebview", file=sys.stderr)

    proc = None
    try:
        proc = start_streamlit(port, base)
        ok = wait_for_url(url, timeout=30)
        if not ok:
            print("Streamlit did not become available in time. Check logs.", file=sys.stderr)
            # dump a bit of stderr to help debugging
            if proc and proc.stderr:
                try:
                    err = proc.stderr.read()
                    print("--- Streamlit stderr ---", file=sys.stderr)
                    print(err, file=sys.stderr)
                except Exception:
                    pass
            sys.exit(1)

        # Create and start a webview window; this call blocks until the window is closed.
        if webview:
            window = webview.create_window(os.environ.get("STREAMLIT_DESKTOP_TITLE", "Dashboard"), url, width=1200, height=900)
            webview.start()
        else:
            # If pywebview isn't available simply open the default browser
            import webbrowser

            webbrowser.open(url)
            print(f"Opened browser at {url}. Close the browser tab to stop the server.")
            # Wait until the process exits or user interrupts
            try:
                while proc.poll() is None:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
    finally:
        if proc:
            terminate_process(proc)
            try:
                proc.wait(timeout=5)
            except Exception:
                pass


if __name__ == "__main__":
    main()
