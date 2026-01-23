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
import json
import importlib
import shlex
import logging
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List
from datetime import datetime
from copy import deepcopy

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
    get_runners_package,
    get_runners_packages_list,
    _get_streamlit_config,
    set_runners_source_from_env,
    set_runners_package,
    ACTIVE_PROCS,
    attach_log_tail_to_dashboard,
    enable_auto_attach,
    disable_auto_attach,
    auto_attach_enabled,
)

# Point to the runners directory (one level down from this file)
SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runners")
# The dashboard directory (parent of runners)
DASHBOARD_DIR = os.path.dirname(SCRIPT_DIR)
# Centralize logs to user home: ~/.dashboard/logs
HOME_LOG_DIR = os.path.join(os.path.expanduser("~"), ".dashboard", "logs")
LOG_DIR = HOME_LOG_DIR
LOG_FILE = os.path.join(LOG_DIR, "dashboard.log")
UPLOADS_DIR = os.path.join(DASHBOARD_DIR, "uploads")

def _get_auto_attach_settings() -> tuple[bool, float]:
    """Return (enabled_by_default, interval_seconds) from .streamlit/config.toml.

    Supports keys at top-level or under the [dashboard] table:
      - auto_attach (bool)
      - auto_attach_interval (float)
    Defaults: enabled=True, interval=2.0
    """
    try:
        cfg = _get_streamlit_config() or {}
        enabled_val = None
        interval_val = None
        if isinstance(cfg, dict):
            enabled_val = cfg.get("auto_attach")
            interval_val = cfg.get("auto_attach_interval")
            db = cfg.get("dashboard") if isinstance(cfg.get("dashboard"), dict) else None
            if db is not None:
                if enabled_val is None:
                    enabled_val = db.get("auto_attach")
                if interval_val is None:
                    interval_val = db.get("auto_attach_interval")
        enabled = bool(enabled_val) if enabled_val is not None else True
        try:
            interval = float(interval_val) if interval_val is not None else 2.0
        except Exception:
            interval = 2.0
        return enabled, interval
    except Exception:
        return True, 2.0

# Resolve preferences once per process; Streamlit reruns will reuse these
AUTO_ATTACH_DEFAULT, AUTO_ATTACH_INTERVAL = _get_auto_attach_settings()

def _next_ui_key(prefix: str) -> str:
    """Return a unique Streamlit widget key for this session using a counter."""
    try:
        st.session_state.setdefault("_ui_key_counter", 0)
        st.session_state["_ui_key_counter"] += 1
        return f"{prefix}_{st.session_state['_ui_key_counter']}"
    except Exception:
        # Fallback to time-based key if session state isn't available
        return f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
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
    # Scripts package(s) resolved from config; prefer first but allow selection
    runners_pkg = get_runners_package()
    pkg_options = get_runners_packages_list() or [runners_pkg]
    # Allow a session-only override (user may change package in the UI without saving)
    if st.session_state.get("runners_package_override"):
        runners_pkg = st.session_state.get("runners_package_override")
    st.caption(
        f"Scans scripts from package: '{runners_pkg}', infers CLI parameters (argparse), and runs them."
    )

    # Environment selection (dynamically loaded from config.ini)
    st.subheader("Environment")
    envs = list_env_names() or ["PROD", "UAT", "DEV"]
    # Ensure selected env exists; otherwise fall back to first
    if st.session_state.get("selected_env") not in envs:
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

    # Optional: validate resolved scripts package inside the selected conda env
    col_pkg1, col_pkg2, col_pkg3 = st.columns([3, 1, 1])
    # Session-only select or custom scripts package input
    with col_pkg1:
        rp_key = "runners_package_override"
        default_rp = st.session_state.get(rp_key, runners_pkg)
        # Select from config-provided options
        try:
            idx = pkg_options.index(default_rp) if default_rp in pkg_options else 0
        except Exception:
            idx = 0
        selected_pkg = st.selectbox("Scripts package (from config)", options=pkg_options, index=idx, key="runners_pkg_select")
        # Dynamically apply selection to refresh module list
        try:
            prev_sel = st.session_state.get("runners_pkg_select_prev")
            if selected_pkg and selected_pkg != prev_sel:
                set_runners_package(selected_pkg)
                st.session_state[rp_key] = selected_pkg
                st.session_state["runners_pkg_select_prev"] = selected_pkg
                # Try set source from current env for resolution
                try:
                    sel_env = st.session_state.get("selected_env", (list_env_names() or ["DEV"]) [0])
                    env_vars_sel = get_env_for(sel_env)
                    conda_env_name = env_vars_sel.get("CONDA_ENV") or "qf"
                    set_runners_source_from_env(conda_env_name, selected_pkg)
                except Exception:
                    pass
                try:
                    st.rerun()
                except Exception:
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass
        except Exception:
            pass
        # Optional custom entry
        _ = st.text_input("Scripts package (custom)", value=default_rp, key="runners_pkg_input")
    with col_pkg2:
        if st.button("Apply selected (session only)"):
            new_pkg = st.session_state.get("runners_pkg_select", default_rp)
            try:
                set_runners_package(new_pkg)
                st.session_state[rp_key] = new_pkg
                st.success(f"Using scripts package '{new_pkg}' for this session.")
                try:
                    st.rerun()
                except Exception:
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"Failed to apply scripts package: {e}")
        if st.button("Apply custom (session only)"):
            new_pkg = st.session_state.get("runners_pkg_input", default_rp)
            try:
                set_runners_package(new_pkg)
                st.session_state[rp_key] = new_pkg
                st.success(f"Using scripts package '{new_pkg}' for this session.")
                try:
                    st.rerun()
                except Exception:
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"Failed to apply scripts package: {e}")
    with col_pkg3:
        if st.button("Validate scripts package in Env"):
            envs_list = list_env_names() or ["DEV", "UAT", "PROD"]
            sel_env = st.session_state.get("selected_env", envs_list[0] if envs_list else "DEV")
            env_vars_sel = get_env_for(sel_env)
            conda_env_name = env_vars_sel.get("CONDA_ENV") or "qf"
            from scripts_runner import validate_runners_package_in_env  # local import
            ok, detail, pkg_dir, mods = validate_runners_package_in_env(conda_env_name, runners_pkg)
            count = len(mods)
            if ok:
                st.success(f"{sel_env}: {count} modules (resolved at {pkg_dir})")
            else:
                st.error(f"{sel_env}: {count} modules — {detail}")

    # (Environment selection moved earlier) Ensure selected env exists; otherwise fall back to first
    envs = list_env_names() or ["PROD", "UAT", "DEV"]
    if st.session_state.get("selected_env") not in envs:
        st.session_state["selected_env"] = envs[0]
    # keep env_vars set from earlier block
    # env_vars is already defined above after selection

    # Ensure discovery points at the selected environment's package location
    conda_env_for_pkg = (env_vars.get("CONDA_ENV") or "qf") if env_vars else "qf"
    # Try to set runners source from the env; ignore failure (falls back to local resolution)
    try:
        set_runners_source_from_env(conda_env_for_pkg, runners_pkg)
    except Exception:
        pass

    # Runner backend selection
    st.subheader("Runner backend")
    backend_options = {
        "Conda (by name)": "conda",
        "Interpreter path (env python)": "python",
    }
    default_backend_key = next((k for k, v in backend_options.items() if v == st.session_state.get("runner_backend", "python")), "Interpreter path (env python)")
    chosen_backend_key = st.selectbox("Backend", options=list(backend_options.keys()), index=list(backend_options.keys()).index(default_backend_key))
    st.session_state["runner_backend"] = backend_options[chosen_backend_key]

    modules = list_runner_modules()
    if not modules:
        st.warning(f"No modules found in the scripts package '{runners_pkg}'.")
        return

    # Import modules so Streamlit's watcher tracks changes
    # runners_pkg already resolved above
    for _m in modules:
        try:
            importlib.import_module(f"{runners_pkg}.{_m}")
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
            st.caption(
                "Special-case inputs for non-argparse modules (e.g., third_model): "
                "provide date tokens or environment values."
            )

    # --- Optional: Config file editor/runner ---
    # Determine config directory: prefer CONFIG_PATH from selected env; else fallback to bundled config/
    cfg_path_env = (env_vars or {}).get("CONFIG_PATH", "").strip() if env_vars else ""
    if cfg_path_env:
        # Expand ~ and environment variables, then resolve relative paths against dashboard dir
        expanded = os.path.expanduser(os.path.expandvars(cfg_path_env))
        base_config_dir = expanded if os.path.isabs(expanded) else os.path.join(DASHBOARD_DIR, expanded)
    else:
        base_config_dir = os.path.join(DASHBOARD_DIR, "config")

    # Dynamically create a nested path under the configured directory for this module
    # e.g., <CONFIG_PATH>/<module_name>/<module_name>.json
    config_dir = os.path.join(base_config_dir, module_name)
    # Use a _config suffix for module config filenames to avoid name clashes
    config_path = os.path.join(config_dir, f"{module_name}_config.json")
    config_data = None
    generated_from_specs = False
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception as e:
            st.warning(f"Failed to read config for {module_name}: {e}")
    else:
        # Dynamically populate a default preset based on the selected module
        # CONFIG_PATH acts only as the base directory; file name is derived from the module
        try:
            if specs:
                # argparse-backed module: build args list with discovered defaults/help
                args_list = []
                for s in specs:
                    first_flag = s.flags[0] if s.flags else ""
                    args_list.append({
                        "name": s.name,
                        "flag": first_flag,
                        "type": s.arg_type or "string",
                        "value": s.default if s.default is not None else "",
                        "help": s.help_text or "",
                    })
                config_data = {
                    "noArgparse": False,
                    "args": args_list,
                    "env": {},
                }
            else:
                # Non-argparse: provide token defaults
                tokens = ["RUN_DATE", "COB_DATE", "START_DATE", "END_DATE"]
                config_data = {
                    "noArgparse": True,
                    "inputs": {"tokens": tokens},
                    "defaults": {k: "" for k in tokens},
                    "passMode": "argv tokens",
                    "env": {},
                }
            generated_from_specs = True
        except Exception as e:
            st.warning(f"Failed to generate default config for {module_name}: {e}")

    with st.expander("Config preset (optional)", expanded=False):
        st.caption("CONFIG_PATH should point to a base directory. The preset file path is derived from the selected module name.")
        st.caption(f"Config directory: {config_dir}")
        # Show the computed default config file path and allow override
        st.code(config_path)
        # Allow user to override the concrete JSON file path (stored in session per-module)
        cfg_file_key = f"cfg_file_{module_name}"
        default_cfg_file = st.session_state.get(cfg_file_key, config_path)
        config_path_input = st.text_input("Config JSON file (override)", value=default_cfg_file, key=cfg_file_key)

        # Inform if default directory doesn't exist (we'll create it on save)
        if not os.path.exists(os.path.dirname(config_path_input)):
            st.info("Config directory not found. It will be created when you save or if you change the path to an existing folder.")

        # Validation button: check file existence and JSON validity
        if st.button("Validate JSON file", key=f"cfg_validate_{module_name}"):
            try:
                if not os.path.exists(config_path_input):
                    st.error(f"File does not exist: {config_path_input}")
                else:
                    with open(config_path_input, "r", encoding="utf-8") as jf:
                        json.load(jf)
                    st.success(f"Valid JSON file: {config_path_input}")
            except Exception as e:
                st.error(f"JSON validation failed: {e}")
        if config_data is None:
            st.info("No config file found and no defaults available.")
            # If CONFIG_PATH is set but a preset exists in bundled fallback, surface a hint
            if cfg_path_env:
                fallback_dir = os.path.join(DASHBOARD_DIR, "config")
                fallback_path = os.path.join(fallback_dir, f"{module_name}_config.json")
                if os.path.exists(fallback_path):
                    st.warning(
                        "A preset exists in the bundled directory but not under your CONFIG_PATH. "
                        "Either clear CONFIG_PATH to use the bundled presets or click 'Save Config' to create a copy under CONFIG_PATH.\n\n"
                        f"Bundled preset found at: {fallback_path}"
                    )
        else:
            if generated_from_specs:
                st.info("This preset was generated dynamically from the module's parameters. Click 'Save Config' to persist it under CONFIG_PATH.")

            # Working copy in session to support add/remove/edit before saving
            work_key = f"cfg_work_{module_name}"
            if work_key not in st.session_state:
                st.session_state[work_key] = deepcopy(config_data)
            working = st.session_state[work_key]

            col_reset, _ = st.columns([1, 5])
            with col_reset:
                if st.button("Reset preset from file", key="cfg_reset_btn"):
                    st.session_state[work_key] = deepcopy(config_data)
                    try:
                        st.rerun()
                    except Exception:
                        try:
                            st.experimental_rerun()
                        except Exception:
                            pass

            no_argparse = bool(working.get("noArgparse"))
            if not no_argparse:
                # Show discovered arguments as a read-only summary. Editing arguments in the
                # config preset is disabled to avoid accidental mismatches with the module's
                # argparse signature. To change arguments, update the module source instead.
                st.subheader("Arguments (read-only)")
                if "args" not in working or not isinstance(working.get("args"), list):
                    working["args"] = []
                # Prepare a simple table view
                rows = []
                for a in working.get("args", []):
                    rows.append({
                        "name": a.get("name", ""),
                        "flag": a.get("flag", ""),
                        "type": a.get("type", "string"),
                        "default": a.get("value", ""),
                        "help": a.get("help", ""),
                    })
                if not rows:
                    st.info("No arguments detected for this module.")
                else:
                    try:
                        st.dataframe(rows, use_container_width=True)
                    except Exception:
                        st.write(rows)
            else:
                st.subheader("Non-argparse inputs")
                if "inputs" not in working or not isinstance(working.get("inputs"), dict):
                    working["inputs"] = {"tokens": ["RUN_DATE", "COB_DATE", "START_DATE", "END_DATE"]}
                if "defaults" not in working or not isinstance(working.get("defaults"), dict):
                    working["defaults"] = {k: "" for k in working["inputs"].get("tokens", [])}
                tokens = list(working.get("inputs", {}).get("tokens", []))
                tokens_text = st.text_area("Tokens (one per line)", value="\n".join(tokens), height=100, key="cfg_tokens_text")
                new_tokens = [t.strip() for t in tokens_text.splitlines() if t.strip()]
                working.setdefault("inputs", {})["tokens"] = new_tokens
                # Sync defaults with tokens
                old_defaults = dict(working.get("defaults", {}))
                new_defaults = {k: old_defaults.get(k, "") for k in new_tokens}
                for k in new_tokens:
                    new_defaults[k] = st.text_input(f"default for {k}", value=str(new_defaults.get(k, "")), key=f"cfg_def_{k}")
                working["defaults"] = new_defaults
                pm = str(working.get("passMode", "argv tokens"))
                working["passMode"] = st.radio(
                    "Pass values as",
                    ["argv tokens", "environment variables"],
                    index=0 if pm == "argv tokens" else 1,
                    key="cfg_pass_mode",
                )

            st.subheader("Environment overrides (optional)")
            if "env" not in working or not isinstance(working.get("env"), dict):
                working["env"] = {}
            # Existing env entries
            for k in sorted(list(working["env"].keys())):
                col_ev1, col_ev2 = st.columns([4, 1])
                with col_ev1:
                    val = st.text_input(f"ENV {k}", value=str(working["env"].get(k, "")), key=f"cfg_env_val_{k}")
                    working["env"][k] = val
                with col_ev2:
                    if st.button("Remove", key=f"cfg_env_remove_{k}"):
                        try:
                            del working["env"][k]
                            try:
                                st.rerun()
                            except Exception:
                                try:
                                    st.experimental_rerun()
                                except Exception:
                                    pass
                        except Exception:
                            pass
            # Add new env variable
            st.markdown("Add new environment variable")
            col_ne1, col_ne2, col_ne3 = st.columns([2, 2, 1])
            with col_ne1:
                new_env_key = st.text_input("Key", value="", key="cfg_env_new_key")
            with col_ne2:
                new_env_val = st.text_input("Value", value="", key="cfg_env_new_val")
            with col_ne3:
                if st.button("Add", key="cfg_env_add"):
                    if new_env_key.strip():
                        working["env"][new_env_key.strip()] = new_env_val
                        st.session_state["cfg_env_new_key"] = ""
                        st.session_state["cfg_env_new_val"] = ""
                        try:
                            st.rerun()
                        except Exception:
                            try:
                                st.experimental_rerun()
                            except Exception:
                                pass

            # Advanced editor: modify any JSON node directly
            st.subheader("Raw JSON editor (advanced)")
            st.caption("Edit the full JSON preset. Useful for custom structures beyond the guided form.")
            # Keep a separate text buffer in session so typing doesn't immediately overwrite the working copy
            # Avoid passing a `value=` argument when the widget key is managed via session_state;
            # Streamlit raises if a widget is created with a default value while the same key
            # already exists in session_state. Initialize the session key only when missing,
            # then let the widget read/write the session_state automatically by using the key
            # without supplying `value=`.
            raw_key = f"cfg_raw_json_{module_name}"
            if raw_key not in st.session_state:
                try:
                    st.session_state[raw_key] = json.dumps(working, indent=2)
                except Exception:
                    st.session_state[raw_key] = "{}"
            # Increase editor height by ~30% for better visibility
            # Note: do NOT pass `value=` here when using `key=` tied to session_state.
            raw_text = st.text_area("JSON", height=312, key=raw_key)
            col_rj1, col_rj2 = st.columns([1, 1])
            with col_rj1:
                if st.button("Validate JSON", key=f"cfg_raw_validate_{module_name}"):
                    try:
                        json.loads(raw_text)
                    except Exception as e:
                        st.error(f"Invalid JSON: {e}")
                    else:
                        st.success("JSON is valid.")
            with col_rj2:
                if st.button("Apply to preset", key=f"cfg_raw_apply_{module_name}"):
                    try:
                        parsed = json.loads(raw_text)
                        # Replace the working copy with the parsed JSON so downstream Save/Run uses it
                        working = parsed
                        st.session_state[work_key] = deepcopy(parsed)
                        st.success("Applied raw JSON to preset.")
                        try:
                            st.rerun()
                        except Exception:
                            try:
                                st.experimental_rerun()
                            except Exception:
                                pass
                    except Exception as e:
                        st.error(f"Failed to apply JSON: {e}")

            # Persist working copy back into session and provide actions
            st.session_state[work_key] = working

            # Detach toggle for config-based run
            detach_cfg = st.checkbox("Detach mode", value=True, key="cfg_detach_mode", help="Run asynchronously when enabled; otherwise run synchronously and show output inline.")
            col_s, col_r = st.columns(2)
            with col_s:
                if st.button("Save Config", key="cfg_save_btn"):
                    # Use the possibly-overridden config path input when saving
                    target_path = config_path_input
                    try:
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        with open(target_path, "w", encoding="utf-8") as f:
                            json.dump(working, f, indent=2)
                        # Persist the chosen file path in session state
                        st.session_state[cfg_file_key] = target_path
                        st.success(f"Config saved to {target_path}.")
                    except Exception as e:
                        st.error(f"Failed to save config: {e}")
                # Quick create when file doesn't exist yet
                if not os.path.exists(config_path_input):
                    if st.button("Create default preset here", key="cfg_create_default_btn"):
                        target_path = config_path_input
                        try:
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            with open(target_path, "w", encoding="utf-8") as f:
                                json.dump(working, f, indent=2)
                            st.session_state[cfg_file_key] = target_path
                            st.success(f"Default preset created at {target_path}.")
                        except Exception as e:
                            st.error(f"Failed to create preset: {e}")
            with col_r:
                # Dry run: show command and environment but do not execute
                if st.button("Dry run (show config)", key=f"cfg_dryrun_{module_name}"):
                    conda_env_cfg = (get_env_for(st.session_state["selected_env"]).get("CONDA_ENV") or "qf").strip()
                    ok, detail = conda_env_exists(conda_env_cfg)
                    if not ok:
                        st.error(f"Conda environment '{conda_env_cfg}' not found. Details: {detail}")
                    else:
                        effective_env_cfg = dict(get_env_for(st.session_state["selected_env"]))
                        if working.get("env"):
                            effective_env_cfg.update({k: str(v) for k, v in working.get("env", {}).items()})
                        if not working.get("noArgparse"):
                            cfg_values = {}
                            for a in working.get("args", []):
                                name = str(a.get("name", "")).strip()
                                if name:
                                    cfg_values[name] = a.get("value")
                            cmd_cfg = build_command(module_name, specs, cfg_values, conda_env_name=conda_env_cfg, backend=st.session_state.get("runner_backend", "python"))
                        else:
                            rpkg = runners_pkg
                            cmd_cfg = [
                                "conda", "run", "-n", conda_env_cfg, "python", "-m", f"{rpkg}.{module_name}"
                            ]
                            defaults = working.get("defaults", {})
                            provided = {k: v for k, v in defaults.items() if str(v).strip()}
                            if working.get("passMode", "argv tokens") == "argv tokens":
                                for k, v in provided.items():
                                    cmd_cfg.append(f"{k}:{v}")
                            else:
                                effective_env_cfg.update(provided)
                        cmd_str_cfg = " ".join(shlex.quote(c) for c in cmd_cfg)
                        with st.expander("Dry run (config): command & environment", expanded=True):
                            st.markdown("**Command (not executed):**")
                            st.code(cmd_str_cfg)
                            st.markdown("**Effective environment variables:**")
                            st.json(effective_env_cfg)
                            st.markdown("**Config values provided:**")
                            if not working.get("noArgparse"):
                                st.json({a.get('name'): a.get('value') for a in working.get('args', [])})
                            else:
                                st.json({k: v for k, v in working.get('defaults', {}).items()})

                if st.button("Run with Config", key="cfg_run_btn"):
                    # Build command/env from working copy
                    conda_env_cfg = (get_env_for(st.session_state["selected_env"]).get("CONDA_ENV") or "qf").strip()
                    ok, detail = conda_env_exists(conda_env_cfg)
                    if not ok:
                        st.error(f"Conda environment '{conda_env_cfg}' not found. Details: {detail}")
                    else:
                        effective_env_cfg = dict(get_env_for(st.session_state["selected_env"]))
                        if working.get("env"):
                            effective_env_cfg.update({k: str(v) for k, v in working.get("env", {}).items()})
                        if not working.get("noArgparse"):
                            cfg_values = {}
                            for a in working.get("args", []):
                                name = str(a.get("name", "")).strip()
                                if name:
                                    cfg_values[name] = a.get("value")
                            cmd_cfg = build_command(module_name, specs, cfg_values, conda_env_name=conda_env_cfg)
                        else:
                            # Build base command according to backend
                            cmd_cfg = build_command(module_name, [], {}, conda_env_name=conda_env_cfg, backend=st.session_state.get("runner_backend", "python"))
                            defaults = working.get("defaults", {})
                            provided = {k: v for k, v in defaults.items() if str(v).strip()}
                            if working.get("passMode", "argv tokens") == "argv tokens":
                                for k, v in provided.items():
                                    cmd_cfg.append(f"{k}:{v}")
                            else:
                                effective_env_cfg.update(provided)
                        cmd_str_cfg = " ".join(shlex.quote(c) for c in cmd_cfg)
                        st.write("Command (config):")
                        st.code(cmd_str_cfg)
                        # Log config-based run request
                        try:
                            logging.getLogger("dashboard").info(
                                "RunWithConfig requested: module=%s env=%s mode=%s cmd=%s",
                                module_name,
                                conda_env_cfg,
                                "detached" if detach_cfg else "sync",
                                cmd_str_cfg,
                            )
                        except Exception:
                            pass
                        if detach_cfg:
                            try:
                                prefix = f"{datetime.now():%Y%m%d-%H%M%S}_{module_name}_cfg"
                                info = start_subprocess(cmd_cfg, cwd=DASHBOARD_DIR, extra_env=effective_env_cfg, log_prefix=prefix)
                            except Exception as e:
                                logging.getLogger("dashboard").exception("Run with config failed for module %s", module_name)
                                st.error(f"Failed to start subprocess: {e}")
                            else:
                                # Prevent identical config-based run duplicates
                                if any(r.get("cmd") == cmd_str_cfg for r in st.session_state.get("active_runs", [])):
                                    st.warning("An identical run (same module + parameters) is already active. Change parameters to run concurrently.")
                                else:
                                    st.session_state.setdefault("active_runs", []).append({**info, "module": module_name, "cmd": cmd_str_cfg, "detached": True})
                                try:
                                    logging.getLogger("dashboard").info(
                                        "Started run (config): module=%s pid=%s stdout=%s stderr=%s",
                                        module_name,
                                        info.get("pid"),
                                        info.get("stdout_path"),
                                        info.get("stderr_path"),
                                    )
                                except Exception:
                                    pass
                                st.success(f"Started module '{module_name}' from config (PID {info['pid']}). Running in detached mode.")
                                # Show initial tail without rerun (combine stdout + stderr in Log generation)
                                out_path = info.get("stdout_path")
                                err_path = info.get("stderr_path")
                                st.subheader("Log generation (tail)")
                                combined_tail = ""
                                try:
                                    if out_path and os.path.exists(out_path):
                                        with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                                            out_lines = f.readlines()[-200:]
                                        combined_tail += "".join(out_lines)
                                except Exception:
                                    pass
                                try:
                                    if err_path and os.path.exists(err_path):
                                        with open(err_path, "r", encoding="utf-8", errors="ignore") as f:
                                            err_lines = f.readlines()[-200:]
                                        if err_lines:
                                            if combined_tail:
                                                combined_tail += "\n\nSTDERR (tail):\n"
                                            combined_tail += "".join(err_lines)
                                except Exception:
                                    pass
                                st.text_area("Stdout", value=combined_tail or "(no output yet)", height=300, key=f"initial_cfg_tail_{info['pid']}", label_visibility="collapsed")
                                # Start auto attach by default
                                try:
                                    if AUTO_ATTACH_DEFAULT:
                                        enable_auto_attach(info.get("pid"), label=module_name, interval=AUTO_ATTACH_INTERVAL)
                                except Exception:
                                    pass
                        else:
                            # Run synchronously and show outputs inline
                            try:
                                _t0_cfg = datetime.now()
                                result = run_subprocess(cmd_cfg, cwd=DASHBOARD_DIR, extra_env=effective_env_cfg)
                            except Exception as e:
                                logging.getLogger("dashboard").exception("Run with config failed for module %s", module_name)
                                st.error(f"Run failed: {e}")
                            else:
                                st.success(f"Run (config) finished (rc={result.returncode}).")
                                try:
                                    elapsed_cfg = (datetime.now() - _t0_cfg).total_seconds()
                                    logging.getLogger("dashboard").info(
                                        "Run (config) finished: module=%s rc=%s elapsed=%.3fs",
                                        module_name,
                                        result.returncode,
                                        elapsed_cfg,
                                    )
                                except Exception:
                                    pass
                                combined = (result.stdout or "")
                                if result.stderr:
                                    if combined:
                                        combined += "\n\nSTDERR:\n"
                                    combined += result.stderr
                                st.subheader("Log generation")
                                st.text_area("Stdout", value=combined or "(no output)", height=400, key=_next_ui_key(f"final_sync_cfg_log_{module_name}"), label_visibility="collapsed")
                                st.session_state["run_logs"].append({
                                    "time": datetime.now().isoformat(timespec="seconds"),
                                    "module": module_name,
                                    "cmd": cmd_str_cfg,
                                    "returncode": result.returncode,
                                    "stdout": result.stdout or "",
                                    "stderr": result.stderr or "",
                                })

    st.subheader("Inputs")
    with st.form("params_form"):
        values = render_inputs(specs)
        # Special-case UI when no argparse is present: collect tokens/env values
        run_date = cob_date = start_date = end_date = ""
        pass_mode = "argv tokens"
        # Detached mode: start process but do not display stdout/logs in the UI
        detached = st.checkbox("Detach mode (async)", value=True, help="Run asynchronously without forcing a page rerun. Logs will be shown below.")
        if not specs:
            st.markdown("#### Non-argparse inputs (optional)")
            st.caption("Any string formats are accepted; values are concatenated and separated with ':' by the module.")
            # Show RUN_DATE before COB_DATE per UI preference
            # Remove example date formats from placeholders to avoid prescriptive examples
            run_date = st.text_input("RUN_DATE (any format)", value="")
            cob_date = st.text_input("COB_DATE (any format)", value="")
            start_date = st.text_input("START_DATE (any format)", value="")
            end_date = st.text_input("END_DATE (any format)", value="")
            pass_mode = st.radio(
                "Pass values as",
                ["argv tokens", "environment variables"],
                index=0,
                help="Tokens are passed as NAME:VALUE on argv; env vars set RUN_DATE, COB_DATE, START_DATE, END_DATE",
            )
        # Two submit buttons in the form: Run and Dry run (show command and env)
        col_run_btn, col_dry_btn = st.columns([1, 1])
        with col_run_btn:
            run_btn = st.form_submit_button("Run Module")
        with col_dry_btn:
            dry_run_btn = st.form_submit_button("Dry run (show command)")

    # If user clicked Dry run from the parameter form, compute and display the
    # effective command and environment but do NOT start the subprocess.
    if 'dry_run_btn' in locals() and dry_run_btn:
        conda_env = env_vars.get("CONDA_ENV") or "qf"
        exists, detail = conda_env_exists(conda_env)
        if not exists:
            st.error(f"Conda environment '{conda_env}' not found. Details: {detail}")
        else:
            cmd = build_command(module_name, specs, values, conda_env_name=conda_env, backend=st.session_state.get("runner_backend", "python"))
            # If no argparse specs, optionally pass tokens/env values for non-argparse modules
            effective_env = dict(env_vars)
            provided = {}
            if not specs:
                provided = {
                    "RUN_DATE": run_date.strip(),
                    "COB_DATE": cob_date.strip(),
                    "START_DATE": start_date.strip(),
                    "END_DATE": end_date.strip(),
                }
                provided = {k: v for k, v in provided.items() if v}
                if provided:
                    if pass_mode == "argv tokens":
                        for k, v in provided.items():
                            cmd.append(f"{k}:{v}")
                    else:
                        effective_env.update(provided)
            cmd_str = " ".join(shlex.quote(c) for c in cmd)
            with st.expander("Dry run: command & environment", expanded=True):
                st.markdown("**Command (not executed):**")
                st.code(cmd_str)
                st.markdown("**Effective environment variables:**")
                st.json(effective_env)
                st.markdown("**Parameter values provided:**")
                st.json(values if values else provided)

    # Show active runs (zero or more). We track them in `st.session_state["active_runs"]`.
    if st.session_state.get("active_runs"):
        # Iterate over a copy because we may remove finished runs while iterating
        for active in list(st.session_state.get("active_runs", [])):
            pid = active.get("pid")
            status = get_status(pid)
            if status.get("running"):
                st.info(f"Module '{active.get('module')}' is running (PID {pid}).")
                col1, col2 = st.columns([1, 3])
                with col1:
                    # Auto-attach toggle (keyed by PID)
                    enabled = auto_attach_enabled(pid)
                    auto_on = st.checkbox("Auto attach to dashboard log", value=enabled, key=f"auto_attach_run_{pid}")
                    if auto_on and not enabled:
                        try:
                            enable_auto_attach(pid, label=active.get("module"), interval=AUTO_ATTACH_INTERVAL)
                        except Exception:
                            pass
                    elif (not auto_on) and enabled:
                        try:
                            disable_auto_attach(pid)
                        except Exception:
                            pass
                    if st.button("Terminate Run", key=f"terminate_btn_{pid}"):
                        terminate_process(pid)
                        st.success("Termination signal sent.")
                        try:
                            st.rerun()
                        except Exception:
                            try:
                                st.experimental_rerun()
                            except Exception:
                                pass
                    if st.button("Attach tail to dashboard log", key=f"attach_run_tail_btn_{pid}"):
                        ok = attach_log_tail_to_dashboard(pid, max_lines=200, label=active.get("module"))
                        if ok:
                            st.success("Attached current tail to dashboard.log")
                        else:
                            st.error("Failed to attach log tail for this PID")
                    if st.button("Refresh Log", key=f"refresh_run_log_btn_{pid}"):
                        try:
                            st.rerun()
                        except Exception:
                            try:
                                st.experimental_rerun()
                            except Exception:
                                pass
                with col2:
                    # Show logs (tail) regardless of detach mode; combine or separate per preference
                    st.caption("Log generation (tail)")
                    out_path = active.get("stdout_path")
                    err_path = active.get("stderr_path")
                    combined_tail = ""
                    try:
                        if out_path and os.path.exists(out_path):
                            with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                                out_lines = f.readlines()[-200:]
                            combined_tail += "".join(out_lines)
                    except Exception:
                        pass
                    try:
                        if err_path and os.path.exists(err_path):
                            with open(err_path, "r", encoding="utf-8", errors="ignore") as f:
                                err_lines = f.readlines()[-200:]
                            if err_lines:
                                if combined_tail:
                                    combined_tail += "\n\nSTDERR (tail):\n"
                                combined_tail += "".join(err_lines)
                    except Exception:
                        pass
                    st.text_area("Stdout", value=combined_tail or "(no output yet)", height=300, key=f"live_log_{pid}", label_visibility="collapsed")
            else:
                # Finished; record log entry and remove from active_runs
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
                try:
                    started_at = active.get("started_at")
                    elapsed_s = None
                    if started_at:
                        try:
                            dt0 = datetime.fromisoformat(str(started_at))
                            elapsed_s = (datetime.now() - dt0).total_seconds()
                        except Exception:
                            elapsed_s = None
                    logging.getLogger("dashboard").info(
                        "Run completed: module=%s rc=%s elapsed=%s cmd=%s",
                        active.get("module"),
                        rc,
                        f"{elapsed_s:.3f}s" if isinstance(elapsed_s, float) else "n/a",
                        active.get("cmd"),
                    )
                except Exception:
                    pass
                st.session_state["run_logs"].append({
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "module": active.get("module"),
                    "cmd": active.get("cmd"),
                    "returncode": rc,
                    "stdout": out_text,
                    "stderr": err_text,
                })
                if not active.get("detached"):
                    combined = out_text or ""
                    if err_text:
                        if combined:
                            combined += "\n\nSTDERR:\n"
                        combined += err_text
                    st.subheader("Log generation")
                    st.text_area("Stdout", value=combined or "(no output)", height=400, key=f"final_log_{pid}", label_visibility="collapsed")
                try:
                    attach_log_tail_to_dashboard(pid, max_lines=400, label=active.get("module"))
                except Exception:
                    pass
                # Remove finished run from session list
                try:
                    st.session_state["active_runs"] = [r for r in st.session_state.get("active_runs", []) if r.get("pid") != pid]
                except Exception:
                    st.session_state["active_runs"] = []

    if run_btn:
        conda_env = env_vars.get("CONDA_ENV") or "qf"
        exists, detail = conda_env_exists(conda_env)
        if not exists:
            st.error(f"Conda environment '{conda_env}' not found. Details: {detail}")
            return
        cmd = build_command(module_name, specs, values, conda_env_name=conda_env, backend=st.session_state.get("runner_backend", "python"))
        # If no argparse specs, optionally pass tokens/env values for non-argparse modules
        effective_env = dict(env_vars)
        if not specs:
            provided = {
                "RUN_DATE": run_date.strip(),
                "COB_DATE": cob_date.strip(),
                "START_DATE": start_date.strip(),
                "END_DATE": end_date.strip(),
            }
            provided = {k: v for k, v in provided.items() if v}
            if provided:
                if pass_mode == "argv tokens":
                    # Append NAME:VALUE tokens to the command
                    for k, v in provided.items():
                        cmd.append(f"{k}:{v}")
                else:
                    # Inject as environment variables
                    effective_env.update(provided)
        cmd_str = " ".join(shlex.quote(c) for c in cmd)
        # Prevent starting an identical run (same module + identical parameters) while
        # allow starting the same module concurrently when parameters differ.
        if any(r.get("cmd") == cmd_str for r in st.session_state.get("active_runs", [])):
            st.warning("An identical run is already active (same module and parameters).\nTo run concurrently, change the parameters.")
            return
        st.write("Command:")
        st.code(cmd_str)
        # Log run request details
        try:
            logging.getLogger("dashboard").info(
                "Run requested: module=%s env=%s mode=%s cmd=%s",
                module_name,
                (env_vars.get("CONDA_ENV") or "qf") if env_vars else "qf",
                "detached" if detached else "sync",
                cmd_str,
            )
        except Exception:
            pass
        if detached:
            # Start asynchronously so it can be terminated
            try:
                prefix = f"{datetime.now():%Y%m%d-%H%M%S}_{module_name}"
                info = start_subprocess(cmd, cwd=DASHBOARD_DIR, extra_env=effective_env, log_prefix=prefix)
            except Exception as e:
                logging.getLogger("dashboard").exception("Run failed for module %s", module_name)
                st.error(f"Failed to start subprocess: {e}")
                return
            # Append to active_runs so multiple concurrent runs are supported
            if any(r.get("cmd") == cmd_str for r in st.session_state.get("active_runs", [])):
                st.warning("An identical run (same module + parameters) is already active. Change parameters to run concurrently.")
                return
            st.session_state.setdefault("active_runs", []).append({
                **info,
                "module": module_name,
                "cmd": cmd_str,
                "detached": True,
            })
            # Log start information with PID and log paths
            try:
                logging.getLogger("dashboard").info(
                    "Started run: module=%s pid=%s stdout=%s stderr=%s",
                    module_name,
                    info.get("pid"),
                    info.get("stdout_path"),
                    info.get("stderr_path"),
                )
            except Exception:
                pass
            st.success(f"Started module '{module_name}' (PID {info['pid']}). Running in detached mode.")
            # Show initial tail without rerun
            out_path = info.get("stdout_path")
            err_path = info.get("stderr_path")
            # Show combined stdout + stderr tail
            try:
                combined_tail = ""
                if out_path and os.path.exists(out_path):
                    with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                        out_lines = f.readlines()[-200:]
                    combined_tail += "".join(out_lines)
                if err_path and os.path.exists(err_path):
                    with open(err_path, "r", encoding="utf-8", errors="ignore") as f:
                        err_lines = f.readlines()[-200:]
                    if err_lines:
                        if combined_tail:
                            combined_tail += "\n\nSTDERR (tail):\n"
                        combined_tail += "".join(err_lines)
                st.subheader("Log generation (tail)")
                st.text_area("Stdout", value=combined_tail or "(no output yet)", height=300, key=f"initial_tail_{info['pid']}", label_visibility="collapsed")
            except Exception:
                pass
            # Start auto attach by default
            try:
                enable_auto_attach(info.get("pid"), label=module_name, interval=2.0)
            except Exception:
                pass
        else:
            # Run synchronously and show output inline
            try:
                _t0 = datetime.now()
                result = run_subprocess(cmd, cwd=DASHBOARD_DIR, extra_env=effective_env)
            except Exception as e:
                logging.getLogger("dashboard").exception("Run failed for module %s", module_name)
                st.error(f"Run failed: {e}")
                return
            st.success(f"Run finished (rc={result.returncode}).")
            try:
                elapsed = (datetime.now() - _t0).total_seconds()
                logging.getLogger("dashboard").info(
                    "Run finished: module=%s rc=%s elapsed=%.3fs",
                    module_name,
                    result.returncode,
                    elapsed,
                )
            except Exception:
                pass
            if st.session_state.get("combine_stderr", True):
                combined = (result.stdout or "")
                if result.stderr:
                    if combined:
                        combined += "\n\nSTDERR:\n"
                    combined += result.stderr
                st.subheader("Log generation")
                st.text_area("Stdout", value=combined or "(no output)", height=400, key=_next_ui_key(f"final_sync_log_{module_name}"), label_visibility="collapsed")
            else:
                if result.stdout:
                    st.subheader("Log generation")
                    st.text_area("Stdout", value=result.stdout, height=400, key=_next_ui_key(f"final_sync_log_{module_name}"), label_visibility="collapsed")
                if result.stderr:
                    st.subheader("Stderr")
                    st.code(result.stderr)
            # Append to logs store
            st.session_state["run_logs"].append({
                "time": datetime.now().isoformat(timespec="seconds"),
                "module": module_name,
                "cmd": cmd_str,
                "returncode": result.returncode,
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
            })

    # --- Historical Log Runs (moved from Logs tab) ---
    st.subheader("Historical Log Runs")
    logs = list(st.session_state.get("run_logs", []))
    if not logs:
        st.info("No runs yet.")
    else:
        for i, entry in enumerate(reversed(logs), 1):
            with st.expander(f"[{i}] {entry['time']} • {entry['module']} • rc={entry['returncode']}"):
                st.code(entry["cmd"], language="bash")
                combined = entry.get("stdout", "")
                err = entry.get("stderr", "")
                if err:
                    if combined:
                        combined += "\n\nSTDERR:\n"
                    combined += err
                st.subheader("Log generation")
                st.text_area("Stdout", value=combined or "(no output)", height=300, key=_next_ui_key("hist_log"), label_visibility="collapsed")
    if st.button("Clear Logs"):
        st.session_state["run_logs"] = []
        st.success("Logs cleared.")

    # (Removed here — running processes are now shown under Script Runners tab)

def render_logs_tab() -> None:
    st.header("Logs")
    # Only show the tail of the dashboard log file in this tab
    st.subheader("Dashboard Log (tail)")
    # Manual refresh button to force reread of the log tail
    col_rl1, col_rl2 = st.columns([1, 3])
    with col_rl1:
        if st.button("Refresh", key="refresh_dashboard_log_btn"):
            try:
                st.rerun()
            except Exception:
                try:
                    st.experimental_rerun()
                except Exception:
                    pass
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


def render_readme_tab() -> None:
    st.header("Docs")
    readme_path = os.path.join(DASHBOARD_DIR, "README.md")
    try:
        if os.path.exists(readme_path):
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
            st.markdown(content)
        else:
            st.info("README.md not found in the dashboard directory.")
    except Exception as e:
        st.error(f"Failed to read README.md: {e}")


def render_reconciliation_tab() -> None:
    st.header("Reconciliation")
    st.caption("Compare before/after CSVs by a reconciliation key; aggregate differences. Supports directories or single files.")

    # Ensure module is importable and watched
    try:
        importlib.import_module("reconciliation")
    except Exception:
        pass

    # Available environments from config.ini (used to prepopulate directories via RESULT_PATH)
    envs = list_env_names() or ["DEV", "UAT", "PROD"]
    current_env = st.session_state.get("selected_env", envs[0])

    mode = st.radio("Mode", ["Directories", "Files"], horizontal=True)
    key = st.text_input("Reconciliation key (CSV column)", value="id")
    fields_raw = st.text_input("Optional fields to compare (comma-separated)", value="")
    fields = [f.strip() for f in fields_raw.split(",") if f.strip()] or None

    if mode == "Directories":
        # Optional single preset env (applies to both BEFORE and AFTER) using RESULT_PATH
        envs_with_resultpath = [e for e in envs if get_env_for(e).get("RESULT_PATH")]
        preset_env_choice = st.selectbox(
            "Preset Env (Before & After)",
            options=["(none)"] + envs_with_resultpath,
            index=0,
            help="Select an environment; its RESULT_PATH will prepopulate both directories",
        )
        if preset_env_choice != "(none)":
            cfg = get_env_for(preset_env_choice)
            rp = cfg.get("RESULT_PATH")
            if rp:
                st.session_state["recon_before_dir"] = rp
                st.session_state["recon_after_dir"] = rp
                st.session_state["browse_before_dir_active"] = False
                st.session_state["browse_after_dir_active"] = False
        before_dir = st.text_input(
            "Before directory",
            value=st.session_state.get("recon_before_dir", ""),
            placeholder="/path/to/before/",
        )
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

        after_dir = st.text_input(
            "After directory",
            value=st.session_state.get("recon_after_dir", ""),
            placeholder="/path/to/after/",
        )
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
    # --- Check Environment section (runs check_environment.py from parent tools dir) ---
    st.subheader("Check Environment")
    st.caption("Run a quick environment check using check_environment.py from the parent tools directory.")
    envs = list_env_names() or ["DEV", "UAT", "PROD"]
    sel_env_check = st.selectbox("Environment for check", envs, index=envs.index(st.session_state.get("selected_env", envs[0])))
    check_cfg = get_env_for(sel_env_check)
    conda_env_check = check_cfg.get("CONDA_ENV") or "qf"
    # Candidate locations for check_environment.py
    parent_tools_dir = os.path.join(os.path.dirname(DASHBOARD_DIR), "tools")
    candidates = [
        os.path.join(parent_tools_dir, "check_environment.py"),
        os.path.join(DASHBOARD_DIR, "tools", "check_environment.py"),
        os.path.join(os.path.dirname(DASHBOARD_DIR), "check_environment.py"),
    ]
    found_path = next((p for p in candidates if os.path.exists(p)), "")
    path_key = "check_env_path"
    default_path = st.session_state.get(path_key, found_path or os.path.join(parent_tools_dir, "check_environment.py"))
    check_path = st.text_input("Path to check_environment.py (override)", value=default_path, key=path_key)
    if st.button("Run environment check"):
        if not check_path or not os.path.exists(check_path):
            st.error(f"check_environment.py not found at: {check_path}")
        else:
            st.info(f"Running check with conda env '{conda_env_check}'...")
            try:
                cmd = ["conda", "run", "-n", conda_env_check, "python", check_path]
                proc = run_subprocess(cmd, cwd=os.path.dirname(check_path))
            except Exception as e:
                st.error(f"Failed to run check_environment.py: {e}")
            else:
                out = proc.stdout or ""
                err = proc.stderr or ""
                combined = out
                if err:
                    combined += "\n\nSTDERR:\n" + err
                with st.expander("Check Environment Output", expanded=True):
                    st.text_area("Output", value=combined or "(no output)", height=400, key="check_env_out", label_visibility="collapsed")

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
    # Default start path: prefer `.streamlit/config.toml` -> dashboard.default_notebook_dir or
    # top-level default_notebook_dir. Fall back to per-env NOTEBOOK_PATH, then to the user's home.
    try:
        _cfg = _get_streamlit_config() or {}
        cfg_nb = None
        if isinstance(_cfg, dict):
            # Prefer top-level key; otherwise check [dashboard] table
            cfg_nb = _cfg.get("default_notebook_dir")
            if (not cfg_nb) and isinstance(_cfg.get("dashboard"), dict):
                cfg_nb = _cfg["dashboard"].get("default_notebook_dir")
        if cfg_nb and str(cfg_nb).strip():
            # Expand ~ and env vars; resolve relative paths against dashboard dir
            expanded = os.path.expanduser(os.path.expandvars(str(cfg_nb).strip()))
            default_nb_path = expanded if os.path.isabs(expanded) else os.path.join(DASHBOARD_DIR, expanded)
        else:
            default_nb_path = nb_env_vars.get("NOTEBOOK_PATH") or os.path.expanduser("~")
    except Exception:
        default_nb_path = nb_env_vars.get("NOTEBOOK_PATH") or os.path.expanduser("~")
    # Persist selected notebook start path in session
    if "nb_path" not in st.session_state:
        st.session_state["nb_path"] = default_nb_path
    # Use an explicit widget key so we can programmatically update it when browsing
    if "nb_path_input" not in st.session_state:
        st.session_state["nb_path_input"] = st.session_state.get("nb_path", default_nb_path)
    nb_path = st.text_input("Start directory", value=st.session_state.get("nb_path_input", default_nb_path), key="nb_path_input")
    # Keep canonical nb_path in session synced with widget value
    st.session_state["nb_path"] = st.session_state.get("nb_path_input", default_nb_path)

    col_b1, col_b2 = st.columns([1, 3])
    with col_b1:
        if st.button("Browse directory", key="nb_browse_btn"):
            st.session_state.setdefault("nb_browse_active", True)
            # Initialize browse root to current notebook path or dashboard dir
            init_root = st.session_state.get("nb_path", default_nb_path) or DASHBOARD_DIR
            st.session_state.setdefault("nb_browse_root", init_root)
    with col_b2:
        if st.session_state.get("nb_browse_active"):
            st.info("Browsing for Notebook start directory")
            root = st.text_input(
                "Current folder",
                value=st.session_state.get("nb_browse_root", st.session_state.get("nb_path", default_nb_path)),
                key="nb_browse_root_input",
            )
            try:
                dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
            except Exception:
                dirs = []
            sel_dir = st.selectbox("Subdirectories", options=[".."] + dirs, key="nb_browse_sel")
            nav = st.button("Open", key="nb_browse_open")
            if nav:
                new_root = os.path.abspath(os.path.join(root, sel_dir)) if sel_dir != ".." else os.path.abspath(os.path.join(root, ".."))
                st.session_state["nb_browse_root"] = new_root
            pick = st.button("Select this folder", key="nb_browse_pick")
            if pick:
                    st.session_state["nb_browse_active"] = False
                    chosen = st.session_state.get("nb_browse_root", DASHBOARD_DIR)
                    # Update both the canonical path and the widget value so the text_input shows it
                    st.session_state["nb_path"] = chosen
                    st.session_state["nb_path_input"] = chosen
                    try:
                        st.rerun()
                    except Exception:
                        try:
                            st.experimental_rerun()
                        except Exception:
                            pass
    
    st.caption("Notebook will be started detached; check logs/runs for output.")
    open_in_browser = st.checkbox(
        "Open browser on start",
        value=False,
        key="nb_open_browser",
        help="If enabled, Jupyter will attempt to open a browser window when the server starts.")
    # Detect whether JupyterLab is available in the chosen conda env
    lab_available = False
    try:
        # fast check: try `jupyter lab --version` inside the conda env
        proc_check = run_subprocess(["conda", "run", "-n", conda_env, "jupyter", "lab", "--version"])
        lab_available = proc_check.returncode == 0
    except Exception:
        lab_available = False

    # Show Start Notebook button, then (if available) Start JupyterLab below it
    start_nb = st.button("Start Jupyter Notebook", key="start_nb_btn")
    if lab_available:
        start_lab = st.button("Start JupyterLab", key="start_lab_btn")
    else:
        st.write("JupyterLab not detected in selected environment")

    if start_nb or (locals().get("start_lab", False)):
        exists, detail = conda_env_exists(conda_env)
        if not exists:
            st.error(f"Conda environment '{conda_env}' not found. Details: {detail}")
        else:
            # Compose Jupyter start command and optionally open browser
            if start_nb:
                cmd = ["conda", "run", "-n", conda_env, "jupyter", "notebook"]
            else:
                cmd = ["conda", "run", "-n", conda_env, "jupyter", "lab"]
            if not open_in_browser:
                cmd.append("--no-browser")
            try:
                prefix = f"{datetime.now():%Y%m%d-%H%M%S}_jupyter"
                start_dir = st.session_state.get("nb_path", nb_path) or DASHBOARD_DIR
                info = start_subprocess(cmd, cwd=start_dir, extra_env=nb_env_vars, log_prefix=prefix)
                st.success(f"Jupyter Notebook started (PID {info['pid']}) in '{start_dir}'.")
                st.caption("Open the printed URL from logs (uploads/logs). If using token authentication, copy token from logs.")
                # Track active Jupyter session to enable termination and live tails
                try:
                    # Append to active_runs so Jupyter gets tracked alongside other runs
                    cmd_str = " ".join(shlex.quote(c) for c in cmd)
                    if any(r.get("cmd") == cmd_str for r in st.session_state.get("active_runs", [])):
                        st.warning("An identical Jupyter run is already active. Change options to run another instance.")
                    else:
                        st.session_state.setdefault("active_runs", []).append({**info, "module": "Jupyter", "cmd": cmd_str, "cwd": start_dir, "detached": True})
                except Exception:
                    pass
                # Start auto attach by default
                try:
                    enable_auto_attach(info.get("pid"), label="Jupyter", interval=2.0)
                except Exception:
                    pass
                try:
                    logging.getLogger("dashboard").info(
                        "Jupyter started: env=%s pid=%s cwd=%s open_browser=%s stdout=%s stderr=%s",
                        conda_env,
                        info.get("pid"),
                        start_dir,
                        bool(open_in_browser),
                        info.get("stdout_path"),
                        info.get("stderr_path"),
                    )
                except Exception:
                    pass
            except Exception as e:
                logging.getLogger("dashboard").exception("Failed to start Jupyter Notebook")
                st.error(f"Failed to start Jupyter Notebook: {e}")

    # Terminal Script Runner removed: terminal-style ad-hoc command execution is no longer supported
    # to keep the dashboard focused on module-based runs and managed subprocesses.
    # Active Jupyter runs are shown alongside other active runs in `active_runs`.
    # The `active_runs` UI above provides per-PID controls and log tails, so no separate
    # Jupyter-specific section is required here.

    # --- Global termination controls ---
    st.subheader("Kill Process by PID")
    st.caption("Send a termination signal to any running process by its PID. Use carefully.")
    kill_pid_str = st.text_input("PID", value=st.session_state.get("kill_pid_str", ""), key="tools_kill_pid_input")
    st.session_state["kill_pid_str"] = kill_pid_str
    if st.button("Kill PID", key="tools_kill_pid_btn"):
        try:
            pid = int(kill_pid_str)
        except Exception:
            st.error("Please enter a valid integer PID.")
        else:
            before = get_status(pid)
            if not before.get("known") and not before.get("running") and before.get("returncode") is None:
                st.warning(f"PID {pid} is not tracked by the dashboard. Attempting termination anyway.")
            try:
                terminate_process(pid)
            except Exception as e:
                logging.getLogger("dashboard").exception("Failed to terminate PID")
                st.error(f"Failed to terminate PID {pid}: {e}")
            else:
                after = get_status(pid)
                st.success(f"Termination signal sent to PID {pid}.")
                st.caption(f"Before: running={before.get('running')} rc={before.get('returncode')} | After: running={after.get('running')} rc={after.get('returncode')}")

    st.subheader("Terminate all active runs")
    st.caption("Stops all processes started via the dashboard (modules, terminal, Jupyter) that are still tracked.")
    if st.button("Terminate All", key="tools_terminate_all_btn"):
        pids = list(ACTIVE_PROCS.keys())
        if not pids:
            st.info("No active tracked processes.")
        else:
            terminated = []
            failed = []
            for pid in pids:
                try:
                    ok = terminate_process(pid)
                    if ok:
                        terminated.append(pid)
                    else:
                        failed.append(pid)
                except Exception:
                    failed.append(pid)
            if terminated:
                st.success(f"Terminated: {', '.join(str(p) for p in terminated)}")
            if failed:
                st.warning(f"Failed/Not found: {', '.join(str(p) for p in failed)}")

    # --- Running Processes table with multi-select terminate ---
    st.subheader("Running Processes")
    st.caption("Processes started by the dashboard and still running. Select multiple to terminate.")
    try:
        rows = []
        for pid, p in list(ACTIVE_PROCS.items()):
            stat = get_status(pid)
            if not stat.get("running"):
                continue
            # Build command string
            try:
                args = getattr(p, "args", None)
            except Exception:
                args = None
            if isinstance(args, (list, tuple)):
                try:
                    cmd_str = " ".join(shlex.quote(str(a)) for a in args)
                except Exception:
                    cmd_str = " ".join(str(a) for a in args)
            else:
                cmd_str = str(args) if args is not None else ""
            # Heuristic name
            name = "Process"
            lc = (cmd_str or "").lower()
            if "jupyter" in lc and "notebook" in lc:
                name = "Jupyter Notebook"
            elif " bash " in f" {lc} " and " -lc " in f" {lc} ":
                name = "Terminal"
            elif " -m " in f" {cmd_str} ":
                try:
                    parts = cmd_str.split()
                    if "-m" in parts:
                        mod = parts[parts.index("-m") + 1]
                        name = mod.split(".")[-1]
                except Exception:
                    pass
            elif isinstance(args, (list, tuple)) and args:
                name = os.path.basename(str(args[0]))
            # Truncate parameters for display
            params_display = cmd_str
            if len(params_display) > 200:
                params_display = params_display[:200] + " …"
            rows.append({
                "select": False,
                "pid": pid,
                "name": name,
                "parameters": params_display,
            })
        if not rows:
            st.info("No running processes.")
        else:
            edited = st.data_editor(
                rows,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "select": st.column_config.CheckboxColumn("Select", help="Mark to terminate"),
                    "pid": st.column_config.NumberColumn("PID"),
                    "name": st.column_config.TextColumn("Name"),
                    "parameters": st.column_config.TextColumn("Parameters"),
                },
                key="running_procs_editor",
            )
            # Collect selected PIDs
            selected_pids = [int(r.get("pid")) for r in edited if r.get("select")]
            c1, c2 = st.columns([1, 3])
            with c1:
                if st.button("Terminate Selected", key="terminate_selected_btn"):
                    if not selected_pids:
                        st.warning("No processes selected.")
                    else:
                        ok_list, fail_list = [], []
                        for spid in selected_pids:
                            try:
                                ok = terminate_process(spid)
                                (ok_list if ok else fail_list).append(spid)
                            except Exception:
                                fail_list.append(spid)
                        if ok_list:
                            st.success(f"Terminated: {', '.join(str(p) for p in ok_list)}")
                        if fail_list:
                            st.warning(f"Failed/Not found: {', '.join(str(p) for p in fail_list)}")
                        try:
                            st.rerun()
                        except Exception:
                            try:
                                st.experimental_rerun()
                            except Exception:
                                pass
            with c2:
                st.caption("Tip: Use the checkboxes to select one or more processes, then click 'Terminate Selected'.")
    except Exception as e:
        logging.getLogger("dashboard").exception("Failed to render running processes table")
        st.error(f"Failed to show running processes: {e}")

    # --- Logs Maintenance ---
    st.subheader("Logs Maintenance")
    st.caption("Manage logs stored under your home at ~/.dashboard/logs. Clearing logs does not stop active processes.")
    col_lm1, col_lm2 = st.columns([1, 3])
    with col_lm1:
        if st.button("Clear All Logs", key="tools_clear_all_logs_btn"):
            cleared = {
                "dashboard_log": False,
                "rotated_logs": 0,
                "run_logs": 0,
            }
            # Clear main dashboard log by truncating file; preserve file for handler
            try:
                os.makedirs(LOG_DIR, exist_ok=True)
                if os.path.exists(LOG_FILE):
                    with open(LOG_FILE, "w", encoding="utf-8") as f:
                        f.write("")
                    cleared["dashboard_log"] = True
            except Exception:
                logging.getLogger("dashboard").exception("Failed to truncate dashboard.log")
            # Remove rotated dashboard logs (e.g., dashboard.log.1, .2)
            try:
                for name in os.listdir(LOG_DIR):
                    if name.startswith("dashboard.log") and name != os.path.basename(LOG_FILE):
                        path = os.path.join(LOG_DIR, name)
                        try:
                            if os.path.isfile(path):
                                os.remove(path)
                                cleared["rotated_logs"] += 1
                        except Exception:
                            pass
            except Exception:
                logging.getLogger("dashboard").exception("Failed to remove rotated dashboard logs")
            # Clear run logs under ~/.dashboard/logs/runs
            runs_dir = os.path.join(LOG_DIR, "runs")
            try:
                if os.path.isdir(runs_dir):
                    for name in os.listdir(runs_dir):
                        path = os.path.join(runs_dir, name)
                        try:
                            if os.path.isfile(path):
                                os.remove(path)
                                cleared["run_logs"] += 1
                        except Exception:
                            pass
            except Exception:
                logging.getLogger("dashboard").exception("Failed to clear run logs")
            st.success(
                f"Cleared logs: dashboard.log={'yes' if cleared['dashboard_log'] else 'no'}, "
                f"rotated={cleared['rotated_logs']}, runs={cleared['run_logs']}"
            )
    with col_lm2:
        st.caption("This will empty the main dashboard log and delete all files in ~/.dashboard/logs/runs. Rotated dashboard logs are removed as well.")


def main() -> None:
    # Branding: Operational Dashboard with icon
    # Prefer a user-configured dashboard name from .streamlit/config.toml
    try:
        _cfg = _get_streamlit_config() or {}
        # Support both top-level and [dashboard] table keys
        dashboard_name = (
            (_cfg.get("dashboard") or {}).get("dashboard_name")
            if isinstance(_cfg.get("dashboard"), dict)
            else _cfg.get("dashboard_name")
        ) or "Operational Dashboard"
    except Exception:
        dashboard_name = "Operational Dashboard"
    # Use project logo as the page icon if available; fall back to an emoji
    logo_path = os.path.join(DASHBOARD_DIR, "assets", "logo.svg")
    page_icon_ref = logo_path if os.path.exists(logo_path) else "📈"
    st.set_page_config(page_title=dashboard_name, layout="wide", page_icon=page_icon_ref)
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
    # Logo displayed in header (separate from page icon)
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        try:
            if os.path.exists(logo_path):
                # use_column_width is deprecated; set an explicit pixel width
                st.image(logo_path, width=160)
        except Exception:
            pass
    with col_title:
        st.markdown(f"# {dashboard_name}")
    setup_logging()
    # Log startup only once per Streamlit session to avoid duplicate entries on reruns
    try:
        if not st.session_state.get("_dashboard_started_once"):
            logging.getLogger("dashboard").info("Dashboard started")
            st.session_state["_dashboard_started_once"] = True
    except Exception:
        # Fallback: log once without session guard
        logging.getLogger("dashboard").info("Dashboard started")

    # Initialize logs store in session state
    if "run_logs" not in st.session_state:
        st.session_state["run_logs"] = []
    # Track active runs in session state as a list of run info dicts
    if "active_runs" not in st.session_state:
        st.session_state["active_runs"] = []

    # Ensure home config exists
    try:
        ensure_default_env_config()
    except Exception:
        logging.getLogger("dashboard").exception("Failed to ensure default env config")
    # Persist selected environment in session (initialize lazily)
    if "selected_env" not in st.session_state:
        existing_envs = list_env_names() or ["DEV", "UAT", "PROD"]
        st.session_state["selected_env"] = existing_envs[0]

    tab_runner, tab_logs, tab_config, tab_recon, tab_tools, tab_docs = st.tabs(["Script Runners", "Logs", "Config", "Reconciliation", "Tools", "Docs"])

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

    # --- Docs Tab (always last) ---
    with tab_docs:
        render_readme_tab()


if __name__ == "__main__":
    main()
