#!/usr/bin/env python3
"""
Streamlit Dashboard: Dynamic Script Runner

- Scans all Python scripts in the same directory
- Parses argparse add_argument() calls to derive parameters
- Renders dynamic UI inputs for each parameter
- Runs selected script in a subprocess with provided arguments

Usage:
  streamlit run dashboard.py

Notes:
- Parameter parsing uses a best-effort AST analysis of argparse patterns.
- If a script has no argparse usage detected, it will be run without parameters.
- Supported types: str, int, float, choices; flags via action='store_true'.
"""

import os
import importlib
import shlex
import logging
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List
from datetime import datetime

import streamlit as st

from scripts_runner import (
    ArgSpec,
    list_runner_modules,
    parse_argparse_args_for_module,
    build_command,
    run_subprocess,
    ensure_default_env_config,
    list_env_names,
    get_env_for,
    save_env_for,
    get_env_config_path,
    conda_env_exists,
    start_subprocess,
    get_status,
    terminate_process,
)

# Point to the runners directory (one level down from this file)
SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runners")
# The dashboard directory (parent of runners)
DASHBOARD_DIR = os.path.dirname(SCRIPT_DIR)
LOG_DIR = os.path.join(DASHBOARD_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "dashboard.log")
UPLOADS_DIR = os.path.join(DASHBOARD_DIR, "uploads")
def setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # File handler with rotation
    fh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3)
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh.setFormatter(fmt)
    # Stream handler (optional; helps during dev)
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(fmt)
    # Avoid duplicate handlers
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        logger.addHandler(fh)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(sh)


# Non-UI logic is now provided by scripts_runner module


def render_inputs(specs: List[ArgSpec]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for spec in specs:
        label = spec.name.replace("_", " ").title()
        help_text = spec.help_text or ""
        if spec.arg_type == "bool" and spec.action == "store_true":
            values[spec.name] = st.checkbox(label, value=bool(spec.default or False), help=help_text)
        elif spec.choices:
            default_ix = 0
            if spec.default in spec.choices:
                default_ix = spec.choices.index(spec.default)
            values[spec.name] = st.selectbox(label, spec.choices, index=default_ix, help=help_text)
        elif spec.arg_type == "int":
            default_val = int(spec.default) if isinstance(spec.default, (int, float, str)) and str(spec.default).isdigit() else 0
            values[spec.name] = st.number_input(label, value=default_val, step=1, help=help_text)
        elif spec.arg_type == "float":
            default_val = float(spec.default) if isinstance(spec.default, (int, float, str)) else 0.0
            values[spec.name] = st.number_input(label, value=default_val, format="%f", help=help_text)
        else:
            default_val = spec.default if isinstance(spec.default, str) else ""
            values[spec.name] = st.text_input(label, value=default_val, help=help_text)
    return values


# --- Tab renderers for modularity ---
def render_script_runners_tab() -> None:
    st.header("Dynamic Python Script Runner")
    st.caption("Scans runner modules, infers CLI parameters (argparse), and runs them.")

    # Environment selection (dynamically loaded from config.ini)
    st.subheader("Environment")
    envs = list_env_names() or ["DEV", "UAT", "PROD"]
    # Ensure selected env exists; otherwise fall back to first
    if st.session_state["selected_env"] not in envs:
        st.session_state["selected_env"] = envs[0]
    selected_env = st.selectbox(
        "Select environment",
        envs,
        index=envs.index(st.session_state["selected_env"]),
    )
    st.session_state["selected_env"] = selected_env
    env_vars = get_env_for(selected_env)
    if env_vars:
        with st.expander("Loaded Environment Variables", expanded=False):
            for k, v in env_vars.items():
                st.write(f"- {k} = {v}")

    modules = list_runner_modules()
    if not modules:
        st.warning("No runner modules found in the 'runners' package.")
        return

    # Import modules so Streamlit's watcher tracks changes
    for _m in modules:
        try:
            importlib.import_module(f"runners.{_m}")
        except Exception:
            pass

    module_name = st.selectbox("Select a module", modules)

    with st.expander("Detected Parameters", expanded=True):
        specs = parse_argparse_args_for_module(module_name)
        if specs:
            for s in specs:
                st.markdown(
                    f"- {'positional' if s.positional else 'option'}: **{s.name}** "
                    f"{' (' + ', '.join(s.flags) + ')' if s.flags else ''} "
                    f"type={s.arg_type}"
                )
        else:
            st.info("No argparse parameters detected; module will run without arguments.")

    st.subheader("Inputs")
    with st.form("params_form"):
        values = render_inputs(specs)
        run_btn = st.form_submit_button("Run Module")

    # Show active run section if any
    if "active_run" in st.session_state and st.session_state["active_run"]:
        active = st.session_state["active_run"]
        pid = active.get("pid")
        status = get_status(pid)
        if status.get("running"):
            st.info(f"Module '{active.get('module')}' is running (PID {pid}).")
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("Terminate Run", key="terminate_btn"):
                    terminate_process(pid)
                    st.success("Termination signal sent.")
            with col2:
                st.caption("Live output (tail)")
                # Tail stdout/stderr if available
                out_path = active.get("stdout_path")
                err_path = active.get("stderr_path")
                try:
                    if out_path and os.path.exists(out_path):
                        with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                            out_lines = f.readlines()[-200:]
                        st.code("".join(out_lines) or "(no output yet)")
                except Exception:
                    pass
                try:
                    if err_path and os.path.exists(err_path):
                        with open(err_path, "r", encoding="utf-8", errors="ignore") as f:
                            err_lines = f.readlines()[-200:]
                        if err_lines:
                            st.subheader("Stderr (tail)")
                            st.code("".join(err_lines))
                except Exception:
                    pass
        else:
            # Finished; record log entry and clear active
            rc = status.get("returncode")
            out_text = ""
            err_text = ""
            try:
                if active.get("stdout_path") and os.path.exists(active["stdout_path"]):
                    with open(active["stdout_path"], "r", encoding="utf-8", errors="ignore") as f:
                        out_text = f.read()
            except Exception:
                pass
            try:
                if active.get("stderr_path") and os.path.exists(active["stderr_path"]):
                    with open(active["stderr_path"], "r", encoding="utf-8", errors="ignore") as f:
                        err_text = f.read()
            except Exception:
                pass
            st.success(f"Run finished (rc={rc}).")
            st.session_state["run_logs"].append({
                "time": datetime.now().isoformat(timespec="seconds"),
                "module": active.get("module"),
                "cmd": active.get("cmd"),
                "returncode": rc,
                "stdout": out_text,
                "stderr": err_text,
            })
            # Show outputs once
            if out_text:
                st.subheader("Stdout")
                st.code(out_text)
            if err_text:
                st.subheader("Stderr")
                st.code(err_text)
            st.session_state["active_run"] = None

    if run_btn and not st.session_state.get("active_run"):
        conda_env = env_vars.get("CONDA_ENV") or "qf"
        exists, detail = conda_env_exists(conda_env)
        if not exists:
            st.error(f"Conda environment '{conda_env}' not found. Details: {detail}")
            return
        cmd = build_command(module_name, specs, values, conda_env_name=conda_env)
        cmd_str = " ".join(shlex.quote(c) for c in cmd)
        st.write("Command:")
        st.code(cmd_str)
        # Start asynchronously so it can be terminated
        try:
            prefix = f"{datetime.now():%Y%m%d-%H%M%S}_{module_name}"
            info = start_subprocess(cmd, cwd=DASHBOARD_DIR, extra_env=env_vars, log_prefix=prefix)
        except Exception as e:
            logging.getLogger("dashboard").exception("Run failed for module %s", module_name)
            st.error(f"Failed to start subprocess: {e}")
            return
        st.session_state["active_run"] = {
            **info,
            "module": module_name,
            "cmd": cmd_str,
        }
        st.success(f"Started module '{module_name}' (PID {info['pid']}). Use 'Terminate Run' to stop.")


def render_logs_tab() -> None:
    st.header("Run Logs")
    logs = list(st.session_state.get("run_logs", []))
    if not logs:
        st.info("No runs yet.")
    else:
        for i, entry in enumerate(reversed(logs), 1):
            with st.expander(f"[{i}] {entry['time']} • {entry['module']} • rc={entry['returncode']}"):
                st.code(entry["cmd"], language="bash")
                if entry["stdout"]:
                    st.subheader("Stdout")
                    st.code(entry["stdout"])
                if entry["stderr"]:
                    st.subheader("Stderr")
                    st.code(entry["stderr"])
    if st.button("Clear Logs"):
        st.session_state["run_logs"] = []
        st.success("Logs cleared.")

    # Show tail of persistent log file (optional)
    st.subheader("Dashboard Log (tail)")
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            tail = "".join(lines[-200:])  # last 200 lines
            st.code(tail or "(empty)")
        else:
            st.caption("Log file not created yet.")
    except Exception as e:
        st.error(f"Failed to read log file: {e}")


def render_config_tab() -> None:
    st.header("Environment Config (config.ini)")
    st.caption(f"Path: {get_env_config_path()}")
    envs = list_env_names() or ["DEV", "UAT", "PROD"]
    if st.session_state["selected_env"] not in envs:
        st.session_state["selected_env"] = envs[0]
    env_choice = st.selectbox("Environment section", envs, index=envs.index(st.session_state["selected_env"]))
    env_current = get_env_for(env_choice)
    with st.form("env_config_form"):
        st.subheader(f"Edit values for [{env_choice}]")
        st.caption("Only existing keys can be updated. Adding new keys is disabled.")
        updated: Dict[str, str] = {}
        # Existing keys only
        for k in sorted(env_current.keys()):
            updated[k] = st.text_input(k, value=str(env_current.get(k, "")))
        save_env_btn = st.form_submit_button("Save Environment")
    if save_env_btn:
        try:
            updated_keys, skipped_keys = save_env_for(env_choice, updated)
            if updated_keys:
                st.success(f"Updated {len(updated_keys)} key(s) in [{env_choice}].")
            if skipped_keys:
                st.info(f"Skipped {len(skipped_keys)} new/nonexistent key(s): {', '.join(skipped_keys)}")
            # Keep session selection in sync
            st.session_state["selected_env"] = env_choice
        except Exception as e:
            st.error(f"Failed to save environment: {e}")

    # Validate configuration button
    st.subheader("Validation")
    if st.button("Validate Configuration"):
        envs = list_env_names() or []
        if not envs:
            st.error("No environments defined in config.ini")
        else:
            current = st.session_state.get("selected_env", envs[0])
            env_vars_chk = get_env_for(current)
            conda_env_chk = env_vars_chk.get("CONDA_ENV") or "qf"
            ok, detail = conda_env_exists(conda_env_chk)
            if ok:
                st.success(f"Conda environment '{conda_env_chk}' exists at: {detail}")
            else:
                st.error(f"Conda environment '{conda_env_chk}' not found. Details: {detail}")

    # Move Streamlit config editor below environment config
    st.header("Streamlit Config")
    config_path = os.path.join(DASHBOARD_DIR, ".streamlit", "config.toml")
    existing = ""
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                existing = f.read()
    except Exception as e:
        st.error(f"Failed to read config: {e}")
    with st.form("config_form"):
        edited = st.text_area("config.toml", value=existing, height=300)
        save_btn = st.form_submit_button("Save Config")
    if save_btn:
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(edited)
            st.success("Config saved. App will auto-reload on save if watcher is enabled.")
        except Exception as e:
            st.error(f"Failed to save config: {e}")


def render_reconciliation_tab() -> None:
    st.header("Reconciliation")
    st.caption("Compare before/after CSVs by a reconciliation key; aggregate differences. Supports directories or single files.")

    # Ensure module is importable and watched
    try:
        importlib.import_module("reconciliation")
    except Exception:
        pass

    mode = st.radio("Mode", ["Directories", "Files"], horizontal=True)
    key = st.text_input("Reconciliation key (CSV column)", value="id")
    fields_raw = st.text_input("Optional fields to compare (comma-separated)", value="")
    fields = [f.strip() for f in fields_raw.split(",") if f.strip()] or None

    if mode == "Directories":
        before_dir = st.text_input("Before directory", value="", placeholder="/path/to/before/")
        col_b1, col_b2 = st.columns([1, 3])
        with col_b1:
            if st.button("Browse Before dir", key="browse_before_dir_btn"):
                st.session_state.setdefault("browse_before_dir_active", True)
                st.session_state.setdefault("browse_before_dir_root", DASHBOARD_DIR)
        with col_b2:
            if st.session_state.get("browse_before_dir_active"):
                st.info("Browsing for BEFORE directory")
                root = st.text_input("Current folder", value=st.session_state.get("browse_before_dir_root", DASHBOARD_DIR), key="browse_before_dir_root_input")
                try:
                    dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
                except Exception:
                    dirs = []
                sel_dir = st.selectbox("Subdirectories", options=[".."] + dirs, key="browse_before_dir_sel")
                nav = st.button("Open", key="browse_before_dir_open")
                if nav:
                    new_root = os.path.abspath(os.path.join(root, sel_dir)) if sel_dir != ".." else os.path.abspath(os.path.join(root, ".."))
                    st.session_state["browse_before_dir_root"] = new_root
                pick = st.button("Select this folder", key="browse_before_dir_pick")
                if pick:
                    st.session_state["browse_before_dir_active"] = False
                    st.session_state["recon_before_dir"] = st.session_state.get("browse_before_dir_root", DASHBOARD_DIR)
                    before_dir = st.session_state["recon_before_dir"]

        after_dir = st.text_input("After directory", value="", placeholder="/path/to/after/")
        col_a1, col_a2 = st.columns([1, 3])
        with col_a1:
            if st.button("Browse After dir", key="browse_after_dir_btn"):
                st.session_state.setdefault("browse_after_dir_active", True)
                st.session_state.setdefault("browse_after_dir_root", DASHBOARD_DIR)
        with col_a2:
            if st.session_state.get("browse_after_dir_active"):
                st.info("Browsing for AFTER directory")
                root = st.text_input("Current folder", value=st.session_state.get("browse_after_dir_root", DASHBOARD_DIR), key="browse_after_dir_root_input")
                try:
                    dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
                except Exception:
                    dirs = []
                sel_dir = st.selectbox("Subdirectories", options=[".."] + dirs, key="browse_after_dir_sel")
                nav = st.button("Open", key="browse_after_dir_open")
                if nav:
                    new_root = os.path.abspath(os.path.join(root, sel_dir)) if sel_dir != ".." else os.path.abspath(os.path.join(root, ".."))
                    st.session_state["browse_after_dir_root"] = new_root
                pick = st.button("Select this folder", key="browse_after_dir_pick")
                if pick:
                    st.session_state["browse_after_dir_active"] = False
                    st.session_state["recon_after_dir"] = st.session_state.get("browse_after_dir_root", DASHBOARD_DIR)
                    after_dir = st.session_state["recon_after_dir"]
        run_btn = st.button("Run Reconciliation (Directories)")
        if run_btn:
            if not key.strip():
                st.error("Reconciliation key is required.")
            else:
                try:
                    from reconciliation import reconcile_directories
                    bd = (st.session_state.get("recon_before_dir") or before_dir).strip()
                    ad = (st.session_state.get("recon_after_dir") or after_dir).strip()
                    report = reconcile_directories(bd, ad, key.strip(), fields)
                    st.subheader("Report")
                    st.code(report)
                    st.download_button(
                        label="Download report",
                        data=report,
                        file_name=f"reconciliation_dirs_{datetime.now():%Y%m%d-%H%M%S}.txt",
                    )
                except Exception as e:
                    logging.getLogger("dashboard").exception("Reconciliation (dirs) failed")
                    st.error(f"Reconciliation failed: {e}")
    else:
        before_file = st.text_input("Before file", value="", placeholder="/path/to/before.csv")
        col_fb1, col_fb2 = st.columns([1, 3])
        with col_fb1:
            if st.button("Browse Before file", key="browse_before_file_btn"):
                st.session_state.setdefault("browse_before_file_active", True)
                st.session_state.setdefault("browse_before_file_root", DASHBOARD_DIR)
        with col_fb2:
            if st.session_state.get("browse_before_file_active"):
                st.info("Browsing for BEFORE file")
                root = st.text_input("Current folder", value=st.session_state.get("browse_before_file_root", DASHBOARD_DIR), key="browse_before_file_root_input")
                try:
                    dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
                    files = sorted([f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f)) and f.endswith(".csv")])
                except Exception:
                    dirs, files = [], []
                sel_dir = st.selectbox("Subdirectories", options=[".."] + dirs, key="browse_before_file_sel_dir")
                nav = st.button("Open", key="browse_before_file_open")
                if nav:
                    new_root = os.path.abspath(os.path.join(root, sel_dir)) if sel_dir != ".." else os.path.abspath(os.path.join(root, ".."))
                    st.session_state["browse_before_file_root"] = new_root
                sel_file = st.selectbox("CSV files", options=files, key="browse_before_file_sel_file")
                pickf = st.button("Select this file", key="browse_before_file_pick")
                if pickf and sel_file:
                    st.session_state["browse_before_file_active"] = False
                    st.session_state["recon_before_file"] = os.path.join(st.session_state.get("browse_before_file_root", DASHBOARD_DIR), sel_file)
                    before_file = st.session_state["recon_before_file"]
        after_file = st.text_input("After file", value="", placeholder="/path/to/after.csv")
        col_fa1, col_fa2 = st.columns([1, 3])
        with col_fa1:
            if st.button("Browse After file", key="browse_after_file_btn"):
                st.session_state.setdefault("browse_after_file_active", True)
                st.session_state.setdefault("browse_after_file_root", DASHBOARD_DIR)
        with col_fa2:
            if st.session_state.get("browse_after_file_active"):
                st.info("Browsing for AFTER file")
                root = st.text_input("Current folder", value=st.session_state.get("browse_after_file_root", DASHBOARD_DIR), key="browse_after_file_root_input")
                try:
                    dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
                    files = sorted([f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f)) and f.endswith(".csv")])
                except Exception:
                    dirs, files = [], []
                sel_dir = st.selectbox("Subdirectories", options=[".."] + dirs, key="browse_after_file_sel_dir")
                nav = st.button("Open", key="browse_after_file_open")
                if nav:
                    new_root = os.path.abspath(os.path.join(root, sel_dir)) if sel_dir != ".." else os.path.abspath(os.path.join(root, ".."))
                    st.session_state["browse_after_file_root"] = new_root
                sel_file = st.selectbox("CSV files", options=files, key="browse_after_file_sel_file")
                pickf = st.button("Select this file", key="browse_after_file_pick")
                if pickf and sel_file:
                    st.session_state["browse_after_file_active"] = False
                    st.session_state["recon_after_file"] = os.path.join(st.session_state.get("browse_after_file_root", DASHBOARD_DIR), sel_file)
                    after_file = st.session_state["recon_after_file"]
        # Drag-and-drop uploaders (optional)
        st.subheader("Or drag and drop CSV files")
        before_upload = st.file_uploader("Upload BEFORE CSV", type=["csv"], key="upload_before_csv")
        after_upload = st.file_uploader("Upload AFTER CSV", type=["csv"], key="upload_after_csv")
        run_btn = st.button("Run Reconciliation (Files)")
        if run_btn:
            if not key.strip():
                st.error("Reconciliation key is required.")
            else:
                try:
                    from reconciliation import reconcile_files
                    bf = (st.session_state.get("recon_before_file") or before_file).strip()
                    af = (st.session_state.get("recon_after_file") or after_file).strip()
                    # Prefer uploaded files when provided
                    if before_upload is not None:
                        os.makedirs(UPLOADS_DIR, exist_ok=True)
                        bf_path = os.path.join(UPLOADS_DIR, f"{datetime.now():%Y%m%d-%H%M%S}_before_{before_upload.name}")
                        try:
                            data = before_upload.getvalue()
                        except Exception:
                            data = before_upload.read()
                        with open(bf_path, "wb") as f:
                            f.write(data)
                        bf = bf_path
                    if after_upload is not None:
                        os.makedirs(UPLOADS_DIR, exist_ok=True)
                        af_path = os.path.join(UPLOADS_DIR, f"{datetime.now():%Y%m%d-%H%M%S}_after_{after_upload.name}")
                        try:
                            data = after_upload.getvalue()
                        except Exception:
                            data = after_upload.read()
                        with open(af_path, "wb") as f:
                            f.write(data)
                        af = af_path
                    report = reconcile_files(bf, af, key.strip(), fields)
                    st.subheader("Report")
                    st.code(report)
                    st.download_button(
                        label="Download report",
                        data=report,
                        file_name=f"reconciliation_files_{datetime.now():%Y%m%d-%H%M%S}.txt",
                    )
                except Exception as e:
                    logging.getLogger("dashboard").exception("Reconciliation (files) failed")
                    st.error(f"Reconciliation failed: {e}")


def render_tools_tab() -> None:
    st.header("Tools")
    st.subheader("Jupyter Notebook")
    st.caption("Start Jupyter Notebook in a chosen conda environment and directory. Environment variables from config.ini are passed to the notebook server.")

    # Choose environment
    envs = list_env_names() or ["DEV", "UAT", "PROD"]
    if st.session_state.get("selected_env") not in envs:
        st.session_state["selected_env"] = envs[0]
    nb_env = st.selectbox("Environment", envs, index=envs.index(st.session_state["selected_env"]))
    st.session_state["selected_env"] = nb_env
    nb_env_vars = get_env_for(nb_env)
    conda_env = nb_env_vars.get("CONDA_ENV") or "qf"
    # Default start path from config.ini NOTEBOOK_PATH if present
    default_nb_path = nb_env_vars.get("NOTEBOOK_PATH") or DASHBOARD_DIR
    nb_path = st.text_input("Start directory", value=default_nb_path)
    
    st.caption("Notebook will be started detached; check logs/runs for output.")
    start_nb = st.button("Start Jupyter Notebook")
    if start_nb:
        exists, detail = conda_env_exists(conda_env)
        if not exists:
            st.error(f"Conda environment '{conda_env}' not found. Details: {detail}")
        else:
            # cmd = ["conda", "run", "-n", conda_env, "jupyter", "notebook", "--no-browser"]
            cmd = ["conda", "run", "-n", conda_env, "jupyter", "notebook"]
            try:
                prefix = f"{datetime.now():%Y%m%d-%H%M%S}_jupyter"
                info = start_subprocess(cmd, cwd=nb_path or DASHBOARD_DIR, extra_env=nb_env_vars, log_prefix=prefix)
                st.success(f"Jupyter Notebook started (PID {info['pid']}) in '{nb_path}'.")
                st.caption("Open the printed URL from logs (uploads/logs). If using token authentication, copy token from logs.")
            except Exception as e:
                logging.getLogger("dashboard").exception("Failed to start Jupyter Notebook")
                st.error(f"Failed to start Jupyter Notebook: {e}")


def main() -> None:
    # Branding: Operational Dashboard with icon
    st.set_page_config(page_title="Operational Dashboard", layout="wide", page_icon="📈")
    # Hide Streamlit's Deploy/Share control in the toolbar
    st.markdown(
        """
        <style>
        /* Hide the Deploy/Share button if present */
        div[data-testid="stDeployButton"] { display: none !important; }
        /* Fallback selectors for older/newer Streamlit versions */
        button[title="Share"], a[href*="share.streamlit.io"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    logo_path = os.path.join(DASHBOARD_DIR, "assets", "logo.svg")
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        try:
            if os.path.exists(logo_path):
                # use_column_width is deprecated; set an explicit pixel width
                st.image(logo_path, width=160)
        except Exception:
            pass
    with col_title:
        st.markdown("# Operational Dashboard")
    setup_logging()
    logging.getLogger("dashboard").info("Dashboard started")

    # Initialize logs store in session state
    if "run_logs" not in st.session_state:
        st.session_state["run_logs"] = []

    # Ensure home config exists
    try:
        ensure_default_env_config()
    except Exception:
        logging.getLogger("dashboard").exception("Failed to ensure default env config")
    # Persist selected environment in session (initialize lazily)
    if "selected_env" not in st.session_state:
        existing_envs = list_env_names() or ["DEV", "UAT", "PROD"]
        st.session_state["selected_env"] = existing_envs[0]

    tab_runner, tab_logs, tab_config, tab_recon, tab_tools = st.tabs(["Script Runners", "Logs", "Config", "Reconciliation", "Tools"])

    # --- Script Runners Tab ---
    with tab_runner:
        render_script_runners_tab()

    # --- Logs Tab ---
    with tab_logs:
        render_logs_tab()

    # --- Config Tab ---
    with tab_config:
        render_config_tab()

    # --- Reconciliation Tab ---
    with tab_recon:
        render_reconciliation_tab()

    # --- Tools Tab ---
    with tab_tools:
        render_tools_tab()


if __name__ == "__main__":
    main()
