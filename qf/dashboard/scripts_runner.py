#!/usr/bin/env python3
"""
scripts_runner.py

Non-UI logic used by the Streamlit dashboard:
- Discover Python scripts in a directory
- Parse argparse add_argument() calls to infer CLI parameters
- Build command lines from parameter values
- Run scripts in a subprocess and return results
"""

import os
import ast
import shlex
import subprocess
import logging
import configparser
import json
import signal
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ArgSpec:
    name: str
    flags: List[str]
    positional: bool
    arg_type: str = "str"  # "str" | "int" | "float" | "bool"
    default: Optional[Any] = None
    choices: Optional[List[Any]] = None
    help_text: Optional[str] = None
    action: Optional[str] = None  # e.g., "store_true"

    def preferred_flag(self) -> Optional[str]:
        if self.positional:
            return None
        long_flags = [f for f in self.flags if f.startswith("--")]
        if long_flags:
            return sorted(long_flags, key=len, reverse=True)[0]
        return self.flags[0] if self.flags else None


SELF_NAME = os.path.basename(__file__)
RUNNERS_PACKAGE = "runners"
CONDA_ENV_NAME = "qf"
HOME_CONFIG_DIRNAME = ".qf_dashboard"
HOME_CONFIG_FILENAME = "config.ini"
RUNNERS_DIR = os.path.join(os.path.dirname(__file__), RUNNERS_PACKAGE)
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
RUN_LOGS_DIR = os.path.join(LOGS_DIR, "runs")


def _to_name_from_flag(flag: str) -> str:
    # --start-date -> start_date; -s -> s
    if flag.startswith("--"):
        return flag[2:].replace("-", "_")
    if flag.startswith("-"):
        return flag[1:]
    return flag


def parse_argparse_args_for_module(module_name: str) -> List[ArgSpec]:
    """Parse argparse add_argument calls from a module within the runners package.

    module_name should be the filename without .py (e.g., "first_model").
    """
    py_file = os.path.join(RUNNERS_DIR, f"{module_name}.py")
    try:
        with open(py_file, "r", encoding="utf-8") as f:
            src = f.read()
    except OSError:
        return []

    try:
        tree = ast.parse(src, filename=py_file)
    except SyntaxError:
        return []

    argspecs: List[ArgSpec] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != "add_argument":
                continue
            # Collect positional/flag strings
            flags: List[str] = []
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    flags.append(a.value)
                elif isinstance(a, ast.Str):  # py<3.8
                    flags.append(a.s)
            # Keyword args
            kw: Dict[str, Any] = {}
            for k in node.keywords:
                key = k.arg
                val = k.value
                kw[key] = val

            # Determine name/dest
            dest_name: Optional[str] = None
            if "dest" in kw:
                v = kw["dest"]
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    dest_name = v.value
                elif isinstance(v, ast.Str):
                    dest_name = v.s

            # Positional if no leading '-' in flags
            positional = not any(f.startswith("-") for f in flags)

            # Choose canonical name
            if dest_name:
                name = dest_name
            elif flags:
                name = _to_name_from_flag(sorted(flags, key=len, reverse=True)[0])
            else:
                name = f"arg_{len(argspecs)+1}"

            # Type
            arg_type = "str"
            if "type" in kw:
                v = kw["type"]
                if isinstance(v, ast.Name):
                    if v.id in ("int", "float", "str"):
                        arg_type = v.id
                elif isinstance(v, ast.Attribute):
                    if v.attr in ("int", "float", "str"):
                        arg_type = v.attr
            # action
            action: Optional[str] = None
            if "action" in kw:
                v = kw["action"]
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    action = v.value
                elif isinstance(v, ast.Str):
                    action = v.s
                if action == "store_true":
                    arg_type = "bool"

            # default
            default: Optional[Any] = None
            if "default" in kw:
                v = kw["default"]
                if isinstance(v, ast.Constant):
                    default = v.value
                elif isinstance(v, ast.Str):
                    default = v.s

            # choices
            choices: Optional[List[Any]] = None
            if "choices" in kw:
                v = kw["choices"]
                if isinstance(v, (ast.List, ast.Tuple)):
                    vals = []
                    for elt in v.elts:
                        if isinstance(elt, ast.Constant):
                            vals.append(elt.value)
                        elif isinstance(elt, ast.Str):
                            vals.append(elt.s)
                    choices = vals

            help_text: Optional[str] = None
            if "help" in kw:
                v = kw["help"]
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    help_text = v.value
                elif isinstance(v, ast.Str):
                    help_text = v.s

            argspecs.append(
                ArgSpec(
                    name=name,
                    flags=flags,
                    positional=positional,
                    arg_type=arg_type,
                    default=default,
                    choices=choices,
                    help_text=help_text,
                    action=action,
                )
            )

    return argspecs


def list_runner_modules() -> List[str]:
    """List module names available in the runners package directory."""
    if not os.path.isdir(RUNNERS_DIR):
        return []
    modules: List[str] = []
    for entry in os.listdir(RUNNERS_DIR):
        if not entry.endswith(".py"):
            continue
        if entry.startswith("__"):
            continue
        modules.append(entry[:-3])  # strip .py
    return sorted(modules)


def _home_config_dir() -> str:
    return os.path.join(os.path.expanduser("~"), HOME_CONFIG_DIRNAME)


def get_env_config_path() -> str:
    """Return the path to the home-sourced environment config file."""
    return os.path.join(_home_config_dir(), HOME_CONFIG_FILENAME)


def ensure_default_env_config() -> None:
    """Create a default config.ini with DEV/UAT/PROD sections if missing."""
    cfg_path = get_env_config_path()
    if os.path.exists(cfg_path):
        return
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    cp = configparser.ConfigParser()
    cp["DEV"] = {
        "API_URL": "http://localhost:8000",
        "API_KEY": "dev-key",
        "LOG_LEVEL": "INFO",
        "CONDA_ENV": CONDA_ENV_NAME,
        "CONFIG_PATH": "",
    }
    cp["UAT"] = {
        "API_URL": "https://uat.api.example.com",
        "API_KEY": "uat-key",
        "LOG_LEVEL": "INFO",
        "CONDA_ENV": CONDA_ENV_NAME,
        "CONFIG_PATH": "",
    }
    cp["PROD"] = {
        "API_URL": "https://api.example.com",
        "API_KEY": "prod-key",
        "LOG_LEVEL": "WARNING",
        "CONDA_ENV": CONDA_ENV_NAME,
        "CONFIG_PATH": "",
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        cp.write(f)


def list_env_names() -> List[str]:
    """List environment names (sections) from the home config file."""
    cfg_path = get_env_config_path()
    cp = configparser.ConfigParser()
    try:
        cp.read(cfg_path)
    except Exception:
        return []
    return list(cp.sections())


def get_env_for(env_name: str) -> Dict[str, str]:
    """Load environment variables (as strings) for the given env section."""
    cfg_path = get_env_config_path()
    cp = configparser.ConfigParser()
    cp.read(cfg_path)
    if env_name not in cp:
        return {}
    section = cp[env_name]
    env_vars: Dict[str, str] = {}
    for key in section:
        env_vars[key.upper()] = str(section.get(key))
    return env_vars


def save_env_for(env_name: str, updates: Dict[str, str]):
    """Persist environment variables for a given section.

    IMPORTANT: Only existing keys in the target section are updated.
    New keys are NOT added. Returns a tuple: (updated_keys, skipped_keys).
    """
    cfg_path = get_env_config_path()
    cp = configparser.ConfigParser()
    try:
        cp.read(cfg_path)
    except Exception:
        # If reading fails, start fresh
        cp = configparser.ConfigParser()
    if env_name not in cp:
        raise ValueError(f"Environment section '{env_name}' not found in config.ini")
    section = cp[env_name]
    updated_keys = []
    skipped_keys = []
    for k, v in updates.items():
        lk = k.lower()
        if lk in section:
            section[lk] = str(v)
            updated_keys.append(k)
        else:
            skipped_keys.append(k)
    if updated_keys:
        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
        with open(cfg_path, "w", encoding="utf-8") as f:
            cp.write(f)
    return updated_keys, skipped_keys


from typing import Tuple


def conda_env_exists(env_name: str) -> Tuple[bool, str]:
    """Check if a conda environment with the given name exists.

    Returns (exists, detail). If conda is not available or the query fails,
    returns (False, reason).
    """
    if not env_name:
        return False, "No environment name provided"
    try:
        proc = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False, "conda executable not found on PATH"
    except Exception as e:
        return False, f"Failed to query conda: {e}"

    if proc.returncode != 0:
        return False, f"conda returned {proc.returncode}: {proc.stderr.strip()}"
    try:
        data = json.loads(proc.stdout)
        env_paths = data.get("envs", [])
        for p in env_paths:
            try:
                if os.path.basename(p.rstrip(os.sep)) == env_name:
                    return True, p
            except Exception:
                continue
        return False, f"Environment '{env_name}' not found"
    except json.JSONDecodeError:
        return False, "Failed to parse conda JSON output"


def build_command(
    module_name: str,
    specs: List[ArgSpec],
    values: Dict[str, Any],
    conda_env_name: Optional[str] = None,
) -> List[str]:
    """Build a conda-run command to run a module from the runners package in the desired env.

    If conda_env_name is None, defaults to CONDA_ENV_NAME.
    """
    env_name = (conda_env_name or CONDA_ENV_NAME).strip()
    cmd: List[str] = [
        "conda",
        "run",
        "-n",
        env_name,
        "python",
        "-m",
        f"{RUNNERS_PACKAGE}.{module_name}",
    ]
    # Positional first in encounter order
    for spec in specs:
        val = values.get(spec.name)
        if spec.positional:
            if spec.arg_type == "bool":
                if val is not None:
                    cmd.append(str(val))
            else:
                if val is not None and str(val) != "":
                    cmd.append(str(val))
    # Options next
    for spec in specs:
        if spec.positional:
            continue
        flag = spec.preferred_flag()
        if not flag:
            continue
        val = values.get(spec.name)
        if spec.arg_type == "bool" and spec.action == "store_true":
            if bool(val):
                cmd.append(flag)
        else:
            if val is not None and str(val) != "":
                cmd.extend([flag, str(val)])
    return cmd


def run_subprocess(cmd: List[str], cwd: Optional[str] = None, extra_env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    logger = logging.getLogger("dashboard.runners")
    joined = " ".join(shlex.quote(c) for c in cmd)
    logger.info("Executing: %s (cwd=%s)", joined, cwd or os.getcwd())
    try:
        env = os.environ.copy()
        if extra_env:
            env.update({k: str(v) for k, v in extra_env.items()})
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except Exception:
        logger.exception("Subprocess execution failed")
        raise
    logger.info("Finished with return code: %s", proc.returncode)
    if proc.stdout:
        logger.debug("Stdout:\n%s", proc.stdout[:4000])
    if proc.stderr:
        logger.debug("Stderr:\n%s", proc.stderr[:4000])
    return proc


# --- Asynchronous process management for long-running tasks ---

# Keep track of active processes by PID (lives in process memory; suitable for Streamlit session lifecycle)
ACTIVE_PROCS: Dict[int, subprocess.Popen] = {}


def start_subprocess(
    cmd: List[str],
    cwd: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
    log_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Start a subprocess asynchronously, redirecting output to files.

    Returns dict with keys: pid, stdout_path, stderr_path, started_at.
    """
    logger = logging.getLogger("dashboard.runners")
    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    os.makedirs(RUN_LOGS_DIR, exist_ok=True)
    prefix = log_prefix or datetime.now().strftime("%Y%m%d-%H%M%S")
    stdout_path = os.path.join(RUN_LOGS_DIR, f"{prefix}.out.log")
    stderr_path = os.path.join(RUN_LOGS_DIR, f"{prefix}.err.log")

    # Open files and launch process; child inherits FDs, safe to close in parent after spawn
    with open(stdout_path, "w", encoding="utf-8") as out, open(stderr_path, "w", encoding="utf-8") as err:
        try:
            p = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=out,
                stderr=err,
                text=True,
                env=env,
                start_new_session=True,  # new process group for group termination
            )
        except Exception:
            logger.exception("Failed to start subprocess")
            raise

    ACTIVE_PROCS[p.pid] = p
    joined = " ".join(shlex.quote(c) for c in cmd)
    logger.info("Started PID %s: %s (cwd=%s)", p.pid, joined, cwd or os.getcwd())
    return {
        "pid": p.pid,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }


def get_status(pid: int) -> Dict[str, Any]:
    """Get running status and return code (if finished) for a PID we started."""
    p = ACTIVE_PROCS.get(pid)
    if not p:
        return {"known": False, "running": False, "returncode": None}
    rc = p.poll()
    return {"known": True, "running": rc is None, "returncode": rc}


def terminate_process(pid: int, force_after: float = 5.0) -> bool:
    """Terminate a running subprocess by PID. Returns True if terminated."""
    logger = logging.getLogger("dashboard.runners")
    p = ACTIVE_PROCS.get(pid)
    if not p:
        return False
    # Send SIGTERM to the whole process group
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        ACTIVE_PROCS.pop(pid, None)
        return True
    except Exception:
        logger.exception("Failed to send SIGTERM to PID %s", pid)
        # Fallback to terminate the single process
        try:
            p.terminate()
        except Exception:
            pass

    # Wait briefly for graceful shutdown
    deadline = time.time() + max(0.1, force_after)
    while time.time() < deadline:
        if p.poll() is not None:
            ACTIVE_PROCS.pop(pid, None)
            logger.info("Process %s terminated with rc=%s", pid, p.returncode)
            return True
        time.sleep(0.1)

    # Force kill the group
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass
    ACTIVE_PROCS.pop(pid, None)
    logger.info("Process %s force-killed", pid)
    return True
