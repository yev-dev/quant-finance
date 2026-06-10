#!/usr/bin/env python3
"""
gap_filling.py — Distressed Time-Series Reconstruction Pipeline
===============================================================

Supports single-target and multi-target runs with consolidated reports.
Uses RESULT_PATH env variable (default: ./results).
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, Matern
from scipy.stats import randint, uniform, rankdata
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit

# Optional: pykalman for Kalman filter methods
KALMAN_AVAILABLE = False
try:
    from pykalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    pass

warnings.filterwarnings("ignore")
np.random.seed(42)
sns_available = False
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    sns_available = True
except ImportError:
    pass

plt.rcParams.update({"figure.figsize": (14, 5), "font.size": 11})

# ── Column name constants ──
DIST_OBS_KEY = "DIST_OBS"
DIST_TRUE_KEY = "DIST_TRUE"
TARGET_ORIG_KEY = "TARGET_ORIG"

# ── Default config path (sibling to this file) ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_SCRIPT_DIR, "config.json")

# ── Backward-compatible default constants ──
def _load_config() -> dict:
    """Load configuration from the sibling config.json file.

    Returns a flat-enough dict of defaults.  Falls back to hard-coded
    values if ``config.json`` is missing or malformed.
    """
    try:
        with open(CONFIG_PATH) as fh:
            cfg = json.load(fh)
        return cfg
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

_CFG = _load_config()

def _override_config(config_path: str | Path) -> None:
    """Replace the global config with a custom config file."""
    global _CFG
    with open(str(config_path)) as fh:
        _CFG = json.load(fh)

def _c(*keys: str, default: Any = None) -> Any:
    """Safely traverse nested config dict.

    Examples
    --------
    >>> _c("peers", "lookback", default=80)
    """
    d: Any = _CFG
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d

# ── Backward-compatible default constants ──
DEFAULT_LOOKBACK    = _c("peers", "lookback", default=80)
DEFAULT_CLUSTER_THR = _c("peers", "cluster_thr", default=0.35)
DEFAULT_MAX_PEERS   = _c("peers", "max_peers", default=5)
DEFAULT_MIN_PEERS   = _c("peers", "min_peers", default=2)
DEFAULT_N_DAYS      = _c("data", "n_days", default=500)
DEFAULT_DIST_DEPTH  = _c("distress", "dist_depth", default=0.35)
DEFAULT_DIST_LENGTH = _c("distress", "dist_length", default=30)
DEFAULT_DIST_START_DATE = _c("distress", "dist_start_date", default=None)
DEFAULT_DIST_END_DATE   = _c("distress", "dist_end_date", default=None)
LOOKBACK_GRID       = _c("sweep", "lookback_grid", default=[40, 60, 80, 120])
THRESH_GRID         = _c("sweep", "threshold_grid", default=[0.25, 0.35, 0.45, 0.55])

# ── Result path ──
_RESULT_PATH_ENV = _c("paths", "result_path_env_var", default="RESULT_PATH")
_RESULT_PATH_DEF = _c("paths", "result_path_default", default="./results")
RESULT_PATH = Path(os.environ.get(_RESULT_PATH_ENV, Path.cwd() / _RESULT_PATH_DEF))

# ── Default data directory (relative to this script's location) ──
DEFAULT_DATA_DIR = os.path.join(_SCRIPT_DIR, "notebooks", "data")


# ═══════════════════════════════════════════════════════════════
#  1. DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_data(data_dir: str | Path = DEFAULT_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    instruments_df = pd.read_csv(data_dir / "instruments.csv")
    stocks_data_df = pd.read_csv(data_dir / "stock_data.csv")
    print(f"  Loaded {len(instruments_df)} instruments from 'instruments.csv'")
    print(f"  Loaded {stocks_data_df.shape[0]:,} rows from 'stock_data.csv'")
    return instruments_df, stocks_data_df


# ═══════════════════════════════════════════════════════════════
#  2. PEER SELECTION
# ═══════════════════════════════════════════════════════════════
def select_peers(target_ticker, instruments_df, stocks_data_df,
                 min_peer_history_days=None, min_peer_coverage=None,
                 max_model_peers=None, include_cross_sector=None):
    if min_peer_history_days is None:
        min_peer_history_days = _c("data", "min_peer_history_days", default=120)
    if min_peer_coverage is None:
        min_peer_coverage = _c("data", "min_peer_coverage", default=0.85)
    if max_model_peers is None:
        max_model_peers = _c("data", "max_model_peers", default=25)
    if include_cross_sector is None:
        include_cross_sector = _c("data", "include_cross_sector", default=False)
    meta = instruments_df.copy()
    meta["symbol"] = meta["symbol"].astype(str).str.upper()
    sector_lookup = meta.set_index("symbol")["sector"].to_dict()
    target_ticker = target_ticker.upper()
    target_sector = sector_lookup.get(target_ticker)

    prices_wide = stocks_data_df.copy()
    prices_wide["Date"] = pd.to_datetime(prices_wide["Date"])
    prices_wide = prices_wide.sort_values("Date").set_index("Date")
    prices_wide.columns = [str(c).upper() for c in prices_wide.columns]
    valid_tickers = [c for c in prices_wide.columns if c in set(meta["symbol"])]
    prices_wide = prices_wide[valid_tickers]
    if target_ticker not in prices_wide.columns:
        raise ValueError(f"{target_ticker} not in stocks_data_df")

    same_sector_all = [s for s in valid_tickers
                       if s != target_ticker and sector_lookup.get(s) == target_sector]
    cross_sector_all = [s for s in valid_tickers
                        if s != target_ticker and sector_lookup.get(s) != target_sector]
    target_ret_all = np.log(prices_wide[target_ticker] / prices_wide[target_ticker].shift(1))

    def _evaluate(cands):
        rows = []
        for t in cands:
            px = prices_wide[t]
            if len(pd.concat([prices_wide[target_ticker], px], axis=1).dropna()) < min_peer_history_days:
                continue
            ret_t = np.log(px / px.shift(1))
            overlap_ret = pd.concat([target_ret_all, ret_t], axis=1).dropna()
            if len(overlap_ret) < min_peer_history_days:
                continue
            corr = overlap_ret.iloc[:, 0].corr(overlap_ret.iloc[:, 1])
            if pd.isna(corr):
                continue
            vol_ratio = float(overlap_ret.iloc[:, 1].std() / overlap_ret.iloc[:, 0].std())
            mae = (overlap_ret.iloc[:, 0] - overlap_ret.iloc[:, 1]).abs().mean()
            coverage = prices_wide[t].notna().mean()
            vol_score = np.exp(-abs(np.log(vol_ratio)))
            perf_score = abs(corr) * coverage / (1.0 + mae)
            composite = 0.5 * (perf_score / (perf_score + 1e-10)) + 0.5 * vol_score
            rows.append({"ticker": t, "sector": sector_lookup.get(t),
                         "corr_to_target": float(corr), "abs_corr": float(abs(corr)),
                         "vol_ratio": vol_ratio, "vol_score": float(vol_score),
                         "ret_mae": float(mae), "coverage": float(coverage),
                         "overlap_days": int(len(overlap_ret)),
                         "peer_score": float(perf_score), "composite_score": float(composite)})
        if not rows:
            return pd.DataFrame(columns=["ticker", "sector", "corr_to_target", "abs_corr",
                                         "vol_ratio", "vol_score", "ret_mae", "coverage",
                                         "overlap_days", "peer_score", "composite_score"])
        return pd.DataFrame(rows).sort_values(
            ["composite_score", "peer_score", "abs_corr"], ascending=False).reset_index(drop=True)

    same_sector_perf = _evaluate(same_sector_all)
    cross_sector_perf = _evaluate(cross_sector_all) if include_cross_sector else pd.DataFrame()
    long_basket = same_sector_perf[same_sector_perf["coverage"] >= min_peer_coverage]["ticker"].tolist()
    if len(long_basket) < 3:
        long_basket = same_sector_perf["ticker"].tolist()
    if len(long_basket) < 3:
        raise ValueError("Not enough same-sector peers after screening")
    model_peer_cols = [t for t in same_sector_perf["ticker"].tolist() if t in long_basket][:max_model_peers]
    if len(model_peer_cols) < 5:
        raise ValueError("Model peer subset too small after ranking")
    cross_sector_peers = cross_sector_perf["ticker"].tolist() if include_cross_sector else []
    peers_df = pd.DataFrame(
        [{"ticker": target_ticker, "sector": target_sector, "peer_bucket": "target"}] +
        [{"ticker": t, "sector": sector_lookup.get(t), "peer_bucket": "same_sector"} for t in long_basket])
    peer_sector_map = peers_df.set_index("ticker")["sector"].to_dict()
    peer_bucket_map = peers_df.groupby("peer_bucket")["ticker"].apply(list).to_dict()
    performance_df = same_sector_perf.assign(peer_bucket="same_sector")
    if include_cross_sector and not cross_sector_perf.empty:
        performance_df = pd.concat(
            [performance_df, cross_sector_perf.assign(peer_bucket="cross_sector")], ignore_index=True)
    return {"target_ticker": target_ticker, "target_sector": target_sector,
            "same_sector_all": same_sector_all, "same_sector_peers": long_basket,
            "cross_sector_peers": cross_sector_peers, "model_peer_cols": model_peer_cols,
            "peers_df": peers_df, "peer_sector_map": peer_sector_map,
            "peer_bucket_map": peer_bucket_map, "performance_df": performance_df,
            "prices_wide": prices_wide}


# ═══════════════════════════════════════════════════════════════
#  3. REGRESSOR / DISTANCE / RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════
def get_regressor(model_type="ols", alpha=1.0, l1_ratio=0.5):
    """Return a regressor instance by type name.

    Parameters
    ----------
    model_type : str
        One of 'ols', 'ridge', 'elasticnet', 'elasticnet_cv'.
    alpha : float
        Regularisation strength (for ridge/elasticnet).
    l1_ratio : float
        Elastic Net mixing parameter (0=Ridge, 1=Lasso, 0.5=equal).

    Returns
    -------
    A scikit-learn regressor with .fit() / .predict() / .score() API.
    """
    model_type = model_type.lower()
    if model_type == "ols":
        return LinearRegression()
    elif model_type == "ridge":
        return Ridge(alpha=alpha, random_state=42)
    elif model_type == "elasticnet":
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=5000)
    elif model_type == "elasticnet_cv":
        return ElasticNetCV(l1_ratio=l1_ratio, cv=5, random_state=42, max_iter=5000)
    raise ValueError(f"Unknown model_type '{model_type}'. Use 'ols', 'ridge', 'elasticnet', or 'elasticnet_cv'.")


def build_peer_distance(ret_window, target_col="DIST", cluster_thr=DEFAULT_CLUSTER_THR,
                        min_peers=DEFAULT_MIN_PEERS, max_peers=DEFAULT_MAX_PEERS, peers_df=None):
    corr_mat = ret_window.corr()
    stocks = corr_mat.columns.tolist()
    corr_dist = ((1 - corr_mat) / 2).clip(lower=0, upper=1)
    np.fill_diagonal(corr_dist.values, 0.0)
    vol_series = ret_window.std()
    vol_diff = pd.DataFrame(0.0, index=stocks, columns=stocks)
    for si in stocks:
        for sj in stocks:
            vol_diff.loc[si, sj] = abs(vol_series[si] - vol_series[sj])
    max_v = vol_diff.to_numpy().max()
    if max_v > 0:
        vol_diff /= max_v
    tracking_dist = pd.DataFrame(0.0, index=stocks, columns=stocks)
    for si in stocks:
        for sj in stocks:
            tracking_dist.loc[si, sj] = (ret_window[si] - ret_window[sj]).std()
    max_t = tracking_dist.to_numpy().max()
    if max_t > 0:
        tracking_dist /= max_t
    down_mask = ret_window[target_col] < 0
    down_sim = pd.Series(0.0, index=stocks, dtype=float)
    target_down = ret_window.loc[down_mask, target_col]
    for s in stocks:
        if s == target_col:
            down_sim[s] = 1.0; continue
        pair = pd.concat([target_down, ret_window.loc[down_mask, s]], axis=1).dropna()
        if len(pair) >= 5:
            c = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            down_sim[s] = np.clip((c + 1) / 2, 0, 1) if pd.notna(c) else 0.0
    down_dist = pd.DataFrame(1.0, index=stocks, columns=stocks)
    for si in stocks:
        for sj in stocks:
            down_dist.loc[si, sj] = abs(down_sim[si] - down_sim[sj])
    dist_mat = (0.40 * corr_dist + 0.20 * vol_diff + 0.15 * tracking_dist + 0.25 * down_dist).clip(0, 1)
    np.fill_diagonal(dist_mat.values, 0.0)
    raw_corr = corr_mat.loc[target_col].drop(target_col)
    vol_sim = 1.0 - vol_diff.loc[target_col].drop(target_col)
    track_sim = 1.0 - tracking_dist.loc[target_col].drop(target_col)
    down_comp = down_sim.drop(target_col)
    peer_score = (0.35 * raw_corr.clip(0) + 0.20 * vol_sim + 0.20 * track_sim + 0.25 * down_comp).sort_values(ascending=False)
    link = linkage(squareform(dist_mat.values), method="ward")
    labels = fcluster(link, t=cluster_thr, criterion="distance")
    same_cluster = [s for s, cid in zip(stocks, labels)
                    if cid == labels[stocks.index(target_col)] and s != target_col and s in peer_score.index]
    if len(same_cluster) >= min_peers:
        peers = sorted(same_cluster, key=lambda s: peer_score[s], reverse=True)[:max_peers]
        rule = "cluster-first"
    else:
        peers = peer_score.index[:max_peers].tolist()
        rule = "fallback global score"
    diag = pd.DataFrame(index=peer_score.index)
    diag["corr_with_DIST"] = raw_corr
    diag["vol_similarity"] = vol_sim
    diag["tracking_similarity"] = track_sim
    diag["downside_similarity"] = down_comp
    if peers_df is not None:
        same_set = set(peers_df[peers_df["peer_bucket"] == "same_sector"]["ticker"])
        diag["same_sector"] = [s in same_set for s in diag.index]
    diag["peer_score"] = peer_score
    return dist_mat, peer_score, diag.sort_values("peer_score", ascending=False), peers, rule


def simple_fill_reconstruction(prices, gap_start, gap_end):
    anchor_pre = prices.iloc[gap_start - 1]
    anchor_post = prices.iloc[gap_end]
    ffill = prices.copy(); ffill.iloc[gap_start:gap_end] = anchor_pre
    bfill = prices.copy(); bfill.iloc[gap_start:gap_end] = anchor_post
    linear = prices.copy(); linear.iloc[gap_start:gap_end] = np.nan
    return {"Forward Fill": ffill, "Backward Fill": bfill,
            "Linear Interp": linear.interpolate(method="linear")}


def static_proxy_reconstruction(ret_window, ret_full, prices_obs, gap_start, gap_end,
                                 cluster_thr=DEFAULT_CLUSTER_THR, min_peers=DEFAULT_MIN_PEERS,
                                 max_peers=DEFAULT_MAX_PEERS, peers_df_override=None):
    _, _, _, peers, rule = build_peer_distance(ret_window, cluster_thr=cluster_thr,
                                               min_peers=min_peers, max_peers=max_peers,
                                               peers_df=peers_df_override)
    peer_cols = [c for c in peers if c in ret_window.columns]
    if not peer_cols:
        raise ValueError("No qualifying peers found for static proxy.")
    X = ret_window[peer_cols].values; y = ret_window["DIST"].values
    model = get_regressor("ols").fit(X, y)
    pred_log_ret = model.predict(ret_full[peer_cols].iloc[gap_start:gap_end].values)
    reconstructed = prices_obs.copy()
    for i in range(gap_end - gap_start):
        reconstructed.iloc[gap_start + i] = reconstructed.iloc[gap_start + i - 1] * np.exp(pred_log_ret[i])
    return reconstructed, {"peers": peer_cols, "coefs": dict(zip(peer_cols, model.coef_)),
                           "r2": model.score(X, y), "rule": rule}


def dynamic_proxy_reconstruction(ret_full, prices_obs, gap_start, gap_end, lookback=DEFAULT_LOOKBACK,
                                  cluster_thr=DEFAULT_CLUSTER_THR, min_peers=DEFAULT_MIN_PEERS,
                                  max_peers=DEFAULT_MAX_PEERS, peers_df_override=None):
    reconstructed = prices_obs.copy()
    daily_rows = []
    for day in range(gap_start, gap_end):
        win_start = max(0, day - lookback)
        window = ret_full.iloc[win_start:day].copy()
        if len(window) < 30:
            reconstructed.iloc[day] = reconstructed.iloc[day - 1]; continue
        _, _, _, peers, _ = build_peer_distance(window, cluster_thr=cluster_thr,
                                                min_peers=min_peers, max_peers=max_peers,
                                                peers_df=peers_df_override)
        peer_cols = [c for c in peers[:max_peers] if c in window.columns]
        if len(peer_cols) < min_peers:
            reconstructed.iloc[day] = reconstructed.iloc[day - 1]; continue
        X = window[peer_cols].values; y = window["DIST"].values
        model = get_regressor("ols").fit(X, y)
        pred_ret = model.predict(ret_full[peer_cols].iloc[day:day + 1].values)[0]
        reconstructed.iloc[day] = reconstructed.iloc[day - 1] * np.exp(pred_ret)
        daily_rows.append({"date": ret_full.index[day], "peers": ", ".join(peer_cols[:3]),
                           "n_peers": len(peer_cols), "r2": model.score(X, y), "pred_log_ret": pred_ret})
    return reconstructed, pd.DataFrame(daily_rows) if daily_rows else pd.DataFrame()


def feature_distance_matrix(ret_window, use_pca=False, pca_var=0.90):
    assets = ret_window.columns.tolist()
    features = {}
    for col in assets:
        r = ret_window[col].values
        features[col] = [r[-1], np.sum(r[-3:]), np.sum(r[-5:]), np.sum(r[-10:]),
                         np.std(r[-5:]), np.std(r[-10:]), np.min(r[-5:]), np.min(r[-10:]),
                         np.std(r[-5:][r[-5:] < 0]) if np.any(r[-5:] < 0) else 0.0,
                         np.mean(np.abs(np.diff(r[-5:])))]
    feat_scaled = StandardScaler().fit_transform(pd.DataFrame.from_dict(features, orient="index").fillna(0))
    if use_pca:
        feat_scaled = PCA(n_components=pca_var).fit_transform(feat_scaled)
    from sklearn.metrics.pairwise import euclidean_distances
    dist_arr = euclidean_distances(feat_scaled)
    dist_arr = (dist_arr + dist_arr.T) / 2.0
    max_val = dist_arr.max()
    if max_val > 0:
        dist_arr /= max_val
    np.fill_diagonal(dist_arr, 0.0)
    return pd.DataFrame(dist_arr, index=assets, columns=assets)


def ml_proxy_reconstruction(ret_window, ret_full, prices_obs, gap_start, gap_end,
                            use_pca=False, cluster_thr=DEFAULT_CLUSTER_THR, max_peers=DEFAULT_MAX_PEERS):
    dist_mat = feature_distance_matrix(ret_window, use_pca=use_pca)
    link = linkage(squareform(dist_mat.values), method="ward")
    labels = fcluster(link, t=cluster_thr, criterion="distance")
    stocks = dist_mat.columns.tolist()
    dist_cluster = labels[stocks.index("DIST")]
    same_cluster = [s for s, cid in zip(stocks, labels) if cid == dist_cluster and s != "DIST"]
    raw_corr = ret_window.corr().loc["DIST"].drop("DIST")
    if same_cluster:
        peers = sorted(same_cluster, key=lambda s: abs(raw_corr[s]), reverse=True)[:max_peers]
        rule = "feature-cluster-first"
    else:
        peers = raw_corr.abs().sort_values(ascending=False).index[:max_peers].tolist()
        rule = "feature-fallback global"
    peer_cols = [c for c in peers if c in ret_window.columns]
    if not peer_cols:
        raise ValueError("No peers found for ML proxy.")
    X = ret_window[peer_cols].values; y = ret_window["DIST"].values
    model = get_regressor("ols").fit(X, y)
    pred_log_ret = model.predict(ret_full[peer_cols].iloc[gap_start:gap_end].values)
    reconstructed = prices_obs.copy()
    for i in range(gap_end - gap_start):
        reconstructed.iloc[gap_start + i] = reconstructed.iloc[gap_start + i - 1] * np.exp(pred_log_ret[i])
    return reconstructed, {"peers": peer_cols, "coefs": dict(zip(peer_cols, model.coef_)),
                           "r2": model.score(X, y), "rule": rule,
                           "feature_type": "PCA" if use_pca else "Raw"}


def ml_proxy_optimisation(ret_df, prices_df, dist_start, dist_end,
                          lookback_grid=None, thresh_grid=None, default_max_peers=DEFAULT_MAX_PEERS):
    if lookback_grid is None:
        lookback_grid = _c("sweep", "lookback_grid", default=LOOKBACK_GRID)
    if thresh_grid is None:
        thresh_grid = _c("sweep", "threshold_grid", default=THRESH_GRID)
    sweep_true = prices_df[DIST_TRUE_KEY].iloc[dist_start:dist_end].values
    sweep_rows, best_rmse, best_params, best_series = [], np.inf, (None, None), None
    for lb in lookback_grid:
        for thr in thresh_grid:
            window = ret_df.iloc[max(0, dist_start - 1 - lb):dist_start - 1].copy()
            try:
                series, info = ml_proxy_reconstruction(window, ret_df, prices_df[DIST_OBS_KEY],
                                                       dist_start, dist_end, use_pca=False,
                                                       cluster_thr=thr, max_peers=default_max_peers)
            except Exception as e:
                sweep_rows.append({"Lookback": lb, "Threshold": thr, "RMSE": np.nan,
                                   "MAE": np.nan, "Ret Corr": np.nan, "R2_fit": np.nan,
                                   "N_Peers": 0, "Error": str(e)})
                continue
            pred = series.iloc[dist_start:dist_end].values
            rmse = float(np.sqrt(mean_squared_error(sweep_true, pred)))
            mae = float(np.mean(np.abs(sweep_true - pred)))
            pred_ret = np.log(pred / series.iloc[dist_start - 1:dist_end - 1].values)
            true_ret = np.log(sweep_true / prices_df[DIST_TRUE_KEY].iloc[dist_start - 1:dist_end - 1].values)
            ret_corr = float(np.corrcoef(pred_ret, true_ret)[0, 1])
            sweep_rows.append({"Lookback": lb, "Threshold": thr, "RMSE": rmse, "MAE": mae,
                               "Ret Corr": ret_corr, "R2_fit": info["r2"],
                               "N_Peers": len(info["peers"]), "Error": ""})
            if rmse < best_rmse:
                best_rmse, best_params, best_series = rmse, (lb, thr), series.copy()
    return (pd.DataFrame(sweep_rows).sort_values("RMSE").reset_index(drop=True),
            best_series, {"best_params": best_params, "best_rmse": best_rmse})


def ml_proxy_enhanced_optimisation(ret_df, prices_df, dist_start, dist_end,
                                   n_iter=None, random_state=None,
                                   default_max_peers=DEFAULT_MAX_PEERS):
    if n_iter is None:
        n_iter = _c("sweep", "n_iter_enhanced", default=50)
    if random_state is None:
        random_state = _c("sweep", "random_state", default=42)
    """Enhanced ML Proxy optimisation with random sampling, out-of-sample
    validation, enhanced metrics, and multi-objective composite scoring.

    Unlike the basic grid search (``ml_proxy_optimisation``), this function:

    1. Samples parameter combinations randomly from continuous ranges
       (lookback 30-120, threshold 0.20-0.60, max_peers 3-10).
    2. Splits the distress window 80/20 for out-of-sample validation.
    3. Tracks additional metrics: MAPE, Directional Accuracy, Max Drift,
       Tracking Error.
    4. Selects best params using a weighted composite rank (multi-objective).

    Parameters
    ----------
    ret_df : pd.DataFrame
        Full return panel.
    prices_df : pd.DataFrame
        Price panel with DIST_OBS and DIST_TRUE columns.
    dist_start, dist_end : int
        Distress window indices.
    n_iter : int
        Number of random parameter combinations to evaluate (default 50).
    random_state : int
        Seed for reproducible random sampling (default 42).
    default_max_peers : int
        Default max peers (fallback). Default DEFAULT_MAX_PEERS.

    Returns
    -------
    enhanced_df : pd.DataFrame
        Results with multi-objective composite scores, sorted by score.
    best_series : pd.Series
        Reconstruction from the best parameter combination.
    best_params : dict
        Best parameter values.
    """
    param_distributions = {
        'lookback': randint(30, 121),
        'cluster_thr': uniform(0.20, 0.40),
        'max_peers': randint(3, 11),
    }
    param_sampler = ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state)
    train_end = dist_start + int(0.8 * (dist_end - dist_start))
    test_start = train_end
    test_end = dist_end

    sweep_rows = []
    sweep_results = {}
    for params in param_sampler:
        lb = params['lookback']
        thr = params['cluster_thr']
        mp = params['max_peers']
        try:
            window = ret_df.iloc[max(0, dist_start - 1 - lb):dist_start - 1].copy()
            series, info = ml_proxy_reconstruction(window, ret_df, prices_df[DIST_OBS_KEY],
                                                   dist_start, dist_end, use_pca=False,
                                                   cluster_thr=thr, max_peers=mp)
        except Exception:
            continue
        pred_test = series.iloc[test_start:test_end].values
        true_test = prices_df[DIST_TRUE_KEY].iloc[test_start:test_end].values
        rmse = float(np.sqrt(mean_squared_error(true_test, pred_test)))
        mae = float(np.mean(np.abs(true_test - pred_test)))
        mape = float(np.mean(np.abs((true_test - pred_test) / true_test)) * 100)
        pred_ret = np.log(pred_test / series.iloc[test_start - 1:test_end - 1].values)
        true_ret = np.log(true_test / prices_df[DIST_TRUE_KEY].iloc[test_start - 1:test_end - 1].values)
        ret_corr = float(np.corrcoef(pred_ret, true_ret)[0, 1]) if len(pred_ret) > 1 else 0.0
        direction_accuracy = float(np.mean(np.sign(pred_ret) == np.sign(true_ret))) if len(pred_ret) > 0 else 0.0
        cum_err = np.cumsum(pred_test - true_test)
        max_drift = float(abs(np.min(cum_err))) if len(cum_err) > 0 else 0.0
        tracking_err = float(np.std(pred_ret - true_ret) * np.sqrt(252)) if len(pred_ret) > 1 else 0.0
        sweep_rows.append({'Lookback': lb, 'Threshold': thr, 'Max Peers': mp,
                           'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape,
                           'Ret Corr': ret_corr, 'Dir Accuracy': direction_accuracy,
                           'Max Drift': max_drift, 'Tracking Error': tracking_err})
        sweep_results[(lb, thr, mp)] = series

    if not sweep_rows:
        return pd.DataFrame(), None, {}
    enhanced_df = pd.DataFrame(sweep_rows)
    for col in ['rmse_rank', 'mape_rank', 'drift_rank', 'corr_rank', 'dir_rank', 'tracking_rank']:
        enhanced_df[col] = rankdata(enhanced_df[col.replace('rank', '') if 'rank' not in col else col])
    # Actually compute ranks properly
    enhanced_df['rmse_rank'] = rankdata(enhanced_df['RMSE'])
    enhanced_df['mape_rank'] = rankdata(enhanced_df['MAPE (%)'])
    enhanced_df['drift_rank'] = rankdata(enhanced_df['Max Drift'])
    enhanced_df['corr_rank'] = rankdata(-enhanced_df['Ret Corr'])
    enhanced_df['dir_rank'] = rankdata(-enhanced_df['Dir Accuracy'])
    enhanced_df['tracking_rank'] = rankdata(enhanced_df['Tracking Error'])
    enhanced_df['Composite Score'] = (
        0.35 * enhanced_df['rmse_rank'] + 0.25 * enhanced_df['mape_rank'] +
        0.20 * enhanced_df['drift_rank'] + 0.12 * enhanced_df['corr_rank'] +
        0.05 * enhanced_df['dir_rank'] + 0.03 * enhanced_df['tracking_rank'])
    enhanced_df = enhanced_df.sort_values('Composite Score').reset_index(drop=True)
    best = enhanced_df.iloc[0]
    best_params = {'lookback': int(best['Lookback']), 'threshold': float(best['Threshold']),
                   'max_peers': int(best['Max Peers'])}
    best_series = sweep_results.get((best['Lookback'], best['Threshold'], best['Max Peers']))
    return enhanced_df, best_series, best_params

def elastic_net_proxy_reconstruction(
    ret_window: pd.DataFrame,
    ret_full: pd.DataFrame,
    prices_obs: pd.Series,
    gap_start: int,
    gap_end: int,
    l1_ratio: float | None = None,
    use_cv: bool | None = None,
    alpha: float | None = None,
) -> tuple[pd.Series, dict]:
    if l1_ratio is None:
        l1_ratio = _c("models", "elastic_net", "l1_ratio", default=0.5)
    if use_cv is None:
        use_cv = _c("models", "elastic_net", "use_cv", default=True)
    if alpha is None:
        alpha = _c("models", "elastic_net", "alpha", default=1.0)
    """Reconstruct using Elastic Net for sparse peer selection.

    L1 penalty (Lasso) forces weak peer coefficients to exactly zero;
    L2 penalty (Ridge) shrinks and groups correlated peers.

    Parameters
    ----------
    ret_window : pd.DataFrame
        Pre-distress return window for model training (column 'DIST' = target).
    ret_full : pd.DataFrame
        Full return panel (includes distress window).
    prices_obs : pd.Series
        Observed price series (corrupted during distress).
    gap_start, gap_end : int
        Distress window indices.
    l1_ratio : float
        Elastic Net mixing: 0=Ridge, 1=Lasso, 0.5=equal (default 0.5).
    use_cv : bool
        Use cross-validation to select alpha (default True).
    alpha : float
        Regularisation strength (used only if use_cv=False).

    Returns
    -------
    reconstructed : pd.Series
        Reconstructed price series.
    info : dict
        Peer selection details and model diagnostics.
    """
    peer_cols = [c for c in ret_window.columns if c != 'DIST']
    X_train = ret_window[peer_cols].values
    y_train = ret_window['DIST'].values

    if use_cv:
        model = ElasticNetCV(l1_ratio=l1_ratio, cv=5, random_state=42, max_iter=5000)
    else:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=5000)

    model.fit(X_train, y_train)

    active_mask = np.abs(model.coef_) > 1e-6
    active_peers = [p for p, active in zip(peer_cols, active_mask) if active]

    if len(active_peers) == 0:
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_train, y_train)
        active_peers = peer_cols

    X_test = ret_full[peer_cols].iloc[gap_start:gap_end].values
    pred_log_ret = model.predict(X_test)

    reconstructed = prices_obs.copy()
    for i in range(gap_end - gap_start):
        reconstructed.iloc[gap_start + i] = (
            reconstructed.iloc[gap_start + i - 1] * np.exp(pred_log_ret[i])
        )

    info = {
        'active_peers': active_peers,
        'n_active': len(active_peers),
        'n_total_peers': len(peer_cols),
        'sparsity': 1.0 - len(active_peers) / len(peer_cols),
        'coefs': {p: c for p, c in zip(peer_cols, model.coef_) if abs(c) > 1e-6},
        'alpha_used': model.alpha_ if use_cv else alpha,
        'r2_train': model.score(X_train, y_train),
        'rule': 'elasticnet-cv' if use_cv else 'elasticnet-fixed',
    }
    return reconstructed, info


def gpr_reconstruction(
    ret_window: pd.DataFrame,
    ret_full: pd.DataFrame,
    prices_obs: pd.Series,
    gap_start: int,
    gap_end: int,
    kernel_type: str | None = None,
    n_restarts: int | None = None,
    max_peers: int = DEFAULT_MAX_PEERS,
) -> tuple[pd.Series, dict]:
    if kernel_type is None:
        kernel_type = _c("models", "gpr", "kernel_type", default='rbf')
    if n_restarts is None:
        n_restarts = _c("models", "gpr", "n_restarts", default=10)
    """Reconstruct using Gaussian Process Regression (non-parametric).

    GPR is a non-parametric method that captures non-linear relationships
    and provides uncertainty quantification.  Unlike OLS/Ridge which fit
    a fixed line, GPR fits a flexible function that adapts to the data.

    Parameters
    ----------
    ret_window : pd.DataFrame
        Pre-distress return window for model training.
    ret_full : pd.DataFrame
        Full return panel (includes distress window).
    prices_obs : pd.Series
        Observed price series (corrupted during distress).
    gap_start, gap_end : int
        Distress window indices.
    kernel_type : str
        Kernel type: 'rbf' (smooth) or 'matern' (robust). Default 'rbf'.
    n_restarts : int
        Number of optimizer restarts for hyperparameter tuning. Default 10.
    max_peers : int
        Maximum peers to use. Default DEFAULT_MAX_PEERS.

    Returns
    -------
    reconstructed : pd.Series
        Reconstructed price series.
    info : dict
        Kernel parameters, uncertainty estimates, and diagnostics.
    """
    corr_to_dist = ret_window.corr()['DIST'].drop('DIST').abs().sort_values(ascending=False)
    peer_cols = corr_to_dist.head(max_peers).index.tolist()

    X_train = ret_window[peer_cols].values
    y_train = ret_window['DIST'].values

    if kernel_type.lower() == 'rbf':
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + \
                 WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
    elif kernel_type.lower() == 'matern':
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + \
                 WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
    else:
        raise ValueError(f"Unknown kernel_type '{kernel_type}'. Use 'rbf' or 'matern'.")

    gpr = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=n_restarts, alpha=1e-10, random_state=42,
    )
    gpr.fit(X_train, y_train)

    X_test = ret_full[peer_cols].iloc[gap_start:gap_end].values
    pred_log_ret, pred_std = gpr.predict(X_test, return_std=True)

    reconstructed = prices_obs.copy()
    price_lower = prices_obs.copy()
    price_upper = prices_obs.copy()

    for i in range(gap_end - gap_start):
        reconstructed.iloc[gap_start + i] = (
            reconstructed.iloc[gap_start + i - 1] * np.exp(pred_log_ret[i])
        )
        ret_lower = pred_log_ret[i] - 1.96 * pred_std[i]
        ret_upper = pred_log_ret[i] + 1.96 * pred_std[i]
        price_lower.iloc[gap_start + i] = price_lower.iloc[gap_start + i - 1] * np.exp(ret_lower)
        price_upper.iloc[gap_start + i] = price_upper.iloc[gap_start + i - 1] * np.exp(ret_upper)

    info = {
        'peers': peer_cols,
        'kernel': str(gpr.kernel_),
        'kernel_type': kernel_type,
        'log_marginal_likelihood': gpr.log_marginal_likelihood_value_,
        'mean_prediction_std': np.mean(pred_std),
        'prediction_std': pred_std,
        'price_lower_95': price_lower,
        'price_upper_95': price_upper,
        'rule': f'gpr-{kernel_type}',
    }
    return reconstructed, info


def kalman_time_varying_beta_reconstruction(
    ret_window: pd.DataFrame,
    ret_full: pd.DataFrame,
    prices_obs: pd.Series,
    gap_start: int,
    gap_end: int,
    max_peers: int = DEFAULT_MAX_PEERS,
    process_noise: float | None = None,
) -> tuple[pd.Series, dict]:
    if process_noise is None:
        process_noise = _c("models", "kalman", "process_noise", default=1e-4)
    """Reconstruct using Kalman Filter for time-varying beta estimation.

    Instead of a static beta (OLS/Ridge), the Kalman Filter lets beta evolve
    over time: beta_t = beta_{t-1} + w_t, allowing adaptation to regime changes
    during the distress window.

    Requires ``pykalman``.  Falls back to static OLS if unavailable.

    Parameters
    ----------
    ret_window : pd.DataFrame
        Pre-distress return window for initial beta estimation.
    ret_full : pd.DataFrame
        Full return panel (includes distress window).
    prices_obs : pd.Series
        Observed price series (corrupted during distress).
    gap_start, gap_end : int
        Distress window indices.
    max_peers : int
        Maximum peers to use. Default DEFAULT_MAX_PEERS.
    process_noise : float
        Process covariance Q (how fast beta can change). Default 1e-4.

    Returns
    -------
    reconstructed : pd.Series
        Reconstructed price series.
    info : dict
        Beta evolution trajectory and diagnostics.
    """
    if not KALMAN_AVAILABLE:
        print('  ⚠️  pykalman not available. Falling back to static OLS.')
        return static_proxy_reconstruction(
            ret_window, ret_full, prices_obs, gap_start, gap_end,
            max_peers=max_peers)

    corr_to_dist = ret_window.corr()['DIST'].drop('DIST').abs().sort_values(ascending=False)
    peer_cols = corr_to_dist.head(max_peers).index.tolist()

    X_init = ret_window[peer_cols].values
    y_init = ret_window['DIST'].values
    beta_0 = np.linalg.lstsq(X_init, y_init, rcond=None)[0]
    residuals = y_init - X_init @ beta_0
    obs_noise = np.var(residuals)

    n_peers = len(peer_cols)

    # Full observation sequence
    full_start = max(0, gap_start - DEFAULT_LOOKBACK)
    X_full = ret_full[peer_cols].iloc[full_start:gap_end].values
    y_full = ret_full['DIST'].iloc[full_start:gap_end].values

    # Manual Kalman filtering loop (time-varying observation matrices)
    state_mean = beta_0
    state_cov = np.eye(n_peers) * 0.01
    filtered_means = []
    beta_trajectory = []

    for t in range(len(y_full)):
        H_t = X_full[t:t + 1]
        state_mean_pred = state_mean
        state_cov_pred = state_cov + np.eye(n_peers) * process_noise

        y_t = y_full[t]
        innovation = y_t - (H_t @ state_mean_pred)[0]
        innovation_cov = (H_t @ state_cov_pred @ H_t.T)[0, 0] + obs_noise
        kalman_gain = state_cov_pred @ H_t.T / innovation_cov

        state_mean = state_mean_pred + (kalman_gain.flatten() * innovation)
        state_cov = state_cov_pred - kalman_gain @ H_t @ state_cov_pred
        filtered_means.append(state_mean.copy())

    filtered_means = np.array(filtered_means)
    reconstructed = prices_obs.copy()
    distress_offset = gap_start - full_start

    for i in range(gap_end - gap_start):
        t = distress_offset + i
        beta_t = filtered_means[t]
        x_t = X_full[t]
        pred_log_ret = beta_t @ x_t
        reconstructed.iloc[gap_start + i] = (
            reconstructed.iloc[gap_start + i - 1] * np.exp(pred_log_ret)
        )
        beta_trajectory.append(beta_t)

    beta_df = pd.DataFrame(beta_trajectory, columns=peer_cols, index=range(gap_start, gap_end))

    pre_window_idx = slice(distress_offset - DEFAULT_LOOKBACK, distress_offset)
    y_pred_pre = np.sum(X_full[pre_window_idx] * filtered_means[pre_window_idx], axis=1)
    y_true_pre = y_full[pre_window_idx]
    r2_train = 1 - np.var(y_true_pre - y_pred_pre) / np.var(y_true_pre)

    info = {
        'peers': peer_cols,
        'beta_trajectory': beta_df,
        'beta_initial': dict(zip(peer_cols, beta_0)),
        'beta_final': dict(zip(peer_cols, beta_trajectory[-1])),
        'process_noise': process_noise,
        'obs_noise': obs_noise,
        'r2_train': r2_train,
        'rule': 'kalman-time-varying-beta',
    }
    return reconstructed, info


# ═══════════════════════════════════════════════════════════════
#  4. EVALUATION
# ═══════════════════════════════════════════════════════════════
def evaluate_reconstructions(recon, prices_df, dist_start, dist_end):
    rows = []
    for name, series in recon.items():
        pred = series.iloc[dist_start:dist_end].values
        true = prices_df[DIST_TRUE_KEY].iloc[dist_start:dist_end].values
        pred_ret = np.log(pred / series.iloc[dist_start - 1:dist_end - 1].values)
        true_ret = np.log(true / prices_df[DIST_TRUE_KEY].iloc[dist_start - 1:dist_end - 1].values)
        rows.append({"Method": name, "RMSE": np.sqrt(mean_squared_error(true, pred)),
                     "MAE": np.mean(np.abs(true - pred)),
                     "Max AE": np.max(np.abs(true - pred)),
                     "Return Corr": np.corrcoef(pred_ret, true_ret)[0, 1],
                     "Price Corr": np.corrcoef(pred, true)[0, 1],
                     "Vol Ratio": np.std(pred_ret) / np.std(true_ret) if np.std(true_ret) > 0 else np.nan,
                     "Final Gap": pred[-1] - true[-1]})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
#  5. DRIFT
# ═══════════════════════════════════════════════════════════════
def compute_drift_metrics(series, true, gap_start, gap_end):
    pv = series.iloc[gap_start:gap_end].values
    tv = true.iloc[gap_start:gap_end].values
    pp = series.iloc[gap_start - 1:gap_end - 1].values
    tp = true.iloc[gap_start - 1:gap_end - 1].values
    ret_err = np.log(pv / pp) - np.log(tv / tp)
    return pd.DataFrame({"price_error": pv - tv, "cum_error": np.cumsum(pv - tv),
                         "abs_error": np.abs(pv - tv), "return_error": ret_err,
                         "tracking_error": np.sqrt(np.cumsum(ret_err**2) / np.arange(1, len(ret_err) + 1))},
                        index=series.index[gap_start:gap_end])


def correct_drift_residual_bias(series, true_anchor, gap_start, gap_end, anchor_days=5):
    c = series.copy()
    bias = np.mean(series.iloc[gap_start:gap_start + anchor_days].values -
                   true_anchor.iloc[gap_start:gap_start + anchor_days].values)
    c.iloc[gap_start:gap_end] -= bias
    return c


def correct_drift_rolling_bias(series, true_anchor, gap_start, gap_end, window=5):
    c = series.copy()
    pv, tv = series.iloc[gap_start:gap_end].values, true_anchor.iloc[gap_start:gap_end].values
    cv = pv.copy()
    for i in range(1, len(cv)):
        lb = max(0, i - window)
        cv[i] = pv[i] - np.mean(cv[lb:i] - tv[lb:i])
    c.iloc[gap_start:gap_end] = cv
    return c


def correct_drift_error_feedback(series, peer_series, gap_start, gap_end, alpha=0.3):
    c = series.copy()
    pv = series.iloc[gap_start:gap_end].values
    for i in range(1, gap_end - gap_start):
        c.iloc[gap_start + i] = pv[i] + alpha * (c.iloc[gap_start + i - 1] - series.iloc[gap_start + i - 1])
    return c


def apply_drift_corrections(best_method, best_series, prices_df, dist_start, dist_end, model_peer_cols):
    corrected = {f"{best_method} (no correction)": best_series,
                 f"{best_method} + Residual Bias": correct_drift_residual_bias(
                     best_series, prices_df[DIST_TRUE_KEY], dist_start, dist_end),
                 f"{best_method} + Rolling Bias": correct_drift_rolling_bias(
                     best_series, prices_df[DIST_TRUE_KEY], dist_start, dist_end),
                 f"{best_method} + Error Feedback": correct_drift_error_feedback(
                     best_series, prices_df[model_peer_cols], dist_start, dist_end)}
    rows = []
    for name, series in corrected.items():
        pred = series.iloc[dist_start:dist_end].values
        true = prices_df[DIST_TRUE_KEY].iloc[dist_start:dist_end].values
        rows.append({"Method": name, "RMSE": np.sqrt(mean_squared_error(true, pred)),
                     "MAE": np.mean(np.abs(true - pred)),
                     "Max Drift": np.max(np.abs(pred - true)),
                     "Final Gap": pred[-1] - true[-1],
                     "End Drift %": ((pred[-1] - true[-1]) / true[-1]) * 100})
    return corrected, pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def cusum_drift_detection(errors, delta=None, threshold=None):
    if delta is None:
        delta = _c("drift_correction", "cusum_delta", default=0.2)
    if threshold is None:
        threshold = _c("drift_correction", "cusum_threshold", default=5.0)
    """Detect drift onset using CUSUM (Cumulative Sum) control chart.

    Parameters
    ----------
    errors : np.ndarray
        Prediction errors (absolute values recommended).
    delta : float
        Acceptable error level (baseline).
    threshold : float
        Alarm threshold — if CUSUM exceeds this, drift is detected.

    Returns
    -------
    dict with keys: 'cusum', 'drift_detected', 'drift_start_day', 'alarm_days', 'n_alarms'.
    """
    n = len(errors)
    S = np.zeros(n)
    drift_start = None
    alarm_days = []
    for t in range(n):
        if t == 0:
            S[t] = max(0, abs(errors[t]) - delta)
        else:
            S[t] = max(0, S[t - 1] + (abs(errors[t]) - delta))
        if S[t] > threshold:
            alarm_days.append(t)
            if drift_start is None:
                drift_start = t
    return {'cusum': S, 'drift_detected': drift_start is not None,
            'drift_start_day': drift_start, 'alarm_days': alarm_days, 'n_alarms': len(alarm_days)}


def kalman_drift_correction(reconstructed, true, gap_start, gap_end,
                            process_variance=None, use_smoother=True):
    if process_variance is None:
        process_variance = _c("drift_correction", "kalman_process_variance", default=1e-3)
    """Apply Kalman Filter to estimate and correct drift in reconstructed prices.

    Treats drift as a 1D latent state variable evolving as a random walk,
    with the prediction errors as noisy observations.

    Parameters
    ----------
    reconstructed : pd.Series
        Reconstructed price series (with drift).
    true : pd.Series
        True price series (for error calculation).
    gap_start, gap_end : int
        Distress window indices.
    process_variance : float
        Process noise Q (how fast drift can change). Default 1e-3.
    use_smoother : bool
        Use Kalman smoother (backward pass) for better estimates. Default True.

    Returns
    -------
    corrected : pd.Series
        Drift-corrected price series.
    info : dict
        Drift trajectory, uncertainty estimates, and diagnostics.
    """
    if not KALMAN_AVAILABLE:
        return reconstructed, {'error': 'pykalman not installed'}
    errors = (reconstructed.iloc[gap_start:gap_end].values - true.iloc[gap_start:gap_end].values)
    n = len(errors)
    obs_variance = np.var(errors) if np.var(errors) > 0 else 1.0
    kf = KalmanFilter(
        n_dim_obs=1, n_dim_state=1,
        initial_state_mean=[errors[0]],
        initial_state_covariance=[[obs_variance]],
        transition_matrices=[[1]],
        transition_covariance=[[process_variance]],
        observation_matrices=[[1]],
        observation_covariance=[[obs_variance]],
    )
    drift_est, drift_cov = (kf.smooth(errors.reshape(-1, 1)) if use_smoother
                            else kf.filter(errors.reshape(-1, 1)))
    drift_est = drift_est.flatten()
    drift_std = np.sqrt(drift_cov[:, 0, 0])
    corrected = reconstructed.copy()
    corrected.iloc[gap_start:gap_end] -= drift_est
    info = {'drift_estimate': drift_est, 'drift_std': drift_std,
            'drift_95_lower': drift_est - 1.96 * drift_std,
            'drift_95_upper': drift_est + 1.96 * drift_std,
            'process_variance': process_variance, 'obs_variance': obs_variance,
            'method': 'kalman-smoother' if use_smoother else 'kalman-filter'}
    return corrected, info


def drift_correction(reconstructed, true, gap_start, gap_end,
                     method=None, process_variance=None):
    if method is None:
        method = _c("drift_correction", "method", default='kalman')
    if process_variance is None:
        process_variance = _c("drift_correction", "kalman_process_variance", default=1e-3)
    """Universal drift correction returning DataFrame with multiple corrections.

    Parameters
    ----------
    reconstructed : pd.Series
        Reconstructed price series (with drift).
    true : pd.Series
        True price series.
    gap_start, gap_end : int
        Distress window indices.
    method : str
        'kalman', 'residual', 'rolling', 'feedback', or 'all'.
    process_variance : float
        Process noise Q for Kalman filter.

    Returns
    -------
    pd.DataFrame with columns 'original', 'true', and correction method columns.
    """
    result = pd.DataFrame({'original': reconstructed, 'true': true})
    if method in ('residual', 'all'):
        result['residual_bias'] = correct_drift_residual_bias(
            reconstructed, true, gap_start, gap_end)
    if method in ('rolling', 'all'):
        result['rolling_bias'] = correct_drift_rolling_bias(
            reconstructed, true, gap_start, gap_end)
    if method in ('kalman', 'all'):
        if KALMAN_AVAILABLE:
            corrected, _ = kalman_drift_correction(
                reconstructed, true, gap_start, gap_end, process_variance)
            result['kalman'] = corrected
        else:
            result['kalman'] = reconstructed
    return result


# ═══════════════════════════════════════════════════════════════
#  6. BACKTESTING
# ═══════════════════════════════════════════════════════════════
def backtest_mean_reversion(prices, lookback=None, threshold=None):
    if lookback is None:
        lookback = _c("backtest", "mean_reversion_lookback", default=5)
    if threshold is None:
        threshold = _c("backtest", "mean_reversion_threshold", default=0.01)
    ma = prices.rolling(lookback).mean()
    pos = pd.Series(0, index=prices.index, dtype=float)
    buy, sell = prices < ma * (1 - threshold), prices > ma * (1 + threshold)
    for i in range(1, len(prices)):
        if buy.iloc[i] and not pd.isna(ma.iloc[i]): pos.iloc[i] = 1.0
        elif sell.iloc[i] and not pd.isna(ma.iloc[i]): pos.iloc[i] = 0.0
        else: pos.iloc[i] = pos.iloc[i - 1]
    return (1 + pos.shift(1).fillna(0) * prices.pct_change().fillna(0)).cumprod()


def compute_performance(equity):
    dr = equity.pct_change().dropna()
    dd = (equity - equity.expanding().max()) / equity.expanding().max()
    return {"Total Return": equity.iloc[-1] - 1,
            "Sharpe Ratio": np.sqrt(252) * dr.mean() / dr.std() if dr.std() > 0 else 0,
            "Max Drawdown": dd.min(), "Final Equity": equity.iloc[-1]}


def run_backtests(recon, prices_df):
    eq_true = backtest_mean_reversion(prices_df[DIST_TRUE_KEY])
    eq_obs = backtest_mean_reversion(prices_df[DIST_OBS_KEY])
    pt, po = compute_performance(eq_true), compute_performance(eq_obs)
    results = [{"Method": n, **compute_performance(backtest_mean_reversion(s)),
                "Return vs True": compute_performance(backtest_mean_reversion(s))["Total Return"] - pt["Total Return"]}
               for n, s in recon.items()]
    return pd.DataFrame(results).sort_values("Sharpe Ratio", ascending=False).reset_index(drop=True), eq_true, eq_obs, pt, po


# ═══════════════════════════════════════════════════════════════
#  7. PLOTTING (saved as PNG in output dir)
# ═══════════════════════════════════════════════════════════════
def _plot_comparison_grid(summary_df, recon, prices_df, dates, dist_start, dist_end, out_dir):
    """Per-Method Comparison: actual vs predicted prices (distress window)."""
    zs, ze = max(0, dist_start - 10), min(len(dates), dist_end + 10)
    nm = len(summary_df); nc = min(4, nm); nr = int(np.ceil(nm / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(6 * nc, 4.5 * nr))
    axes = axes.flatten() if nm > 1 else [axes]
    colors = plt.cm.Set2(np.linspace(0, 1, nm))
    for idx, (_, row) in enumerate(summary_df.iterrows()):
        ax = axes[idx]; method = row["Method"]; series = recon[method]
        ax.plot(dates[zs:ze], prices_df[DIST_TRUE_KEY].iloc[zs:ze], "g--", lw=2, label="True", zorder=5)
        ax.plot(dates[zs:ze], prices_df[DIST_OBS_KEY].iloc[zs:ze], "#F44336", lw=0.8, alpha=0.35, label="Observed")
        ax.plot(dates[zs:ze], series.iloc[zs:ze], color=colors[idx], lw=2.5, label=method)
        ax.axvspan(dates[dist_start], dates[dist_end - 1], color="red", alpha=0.06)
        ax.text(0.03, 0.97, f"RMSE={row['RMSE']:.3f}  MAE={row['MAE']:.3f}\n"
                f"Ret %s={row['Return Corr']:.3f}  VR={row['Vol Ratio']:.3f}" % chr(961),
                transform=ax.transAxes, fontsize=7.5, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor=colors[idx]))
        ax.set_title(method, fontweight="bold", fontsize=10); ax.legend(fontsize=6.5, loc="lower left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, fontsize=7); ax.grid(alpha=0.15)
    for idx in range(nm, len(axes)):
        axes[idx].axis("off")
    fig.suptitle("Per-Method Comparison (Distress Window)", fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout(); fig.savefig(out_dir / "plot_comparison_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)



def _plot_drift(drift_metrics, drift_df, drift_series_dict, best_method,
                prices_df, dates, dist_start, dist_end, out_dir):
    """Drift Detection & Correction chart (Section 5 style).

    Three panels:
      1. Cumulative drift error over the distress window.
      2. CUSUM drift detection alarm signal.
      3. True vs best-predicted vs best-corrected price overlay.
    """
    zs = max(0, dist_start - 5)
    ze = min(len(dates), dist_end + 5)
    dist_dates = dates[dist_start:dist_end]

    # Find best correction
    best_corr_series = None
    best_corr_name = None
    if drift_series_dict and len(drift_series_dict) > 1:
        corr_names = [n for n in drift_series_dict if "(no correction)" not in n]
        if corr_names:
            best_rmse_c = np.inf
            for n in corr_names:
                p = drift_series_dict[n].iloc[dist_start:dist_end].values
                t = prices_df[DIST_TRUE_KEY].iloc[dist_start:dist_end].values
                rmse_c = float(np.sqrt(mean_squared_error(t, p)))
                if rmse_c < best_rmse_c:
                    best_rmse_c = rmse_c
                    best_corr_name = n
                    best_corr_series = drift_series_dict[n]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Cumulative drift error
    ax = axes[0]
    ax.plot(dist_dates, drift_metrics["cum_error"].values, color="#D32F2F", lw=2, marker="o", ms=3)
    ax.fill_between(dist_dates, 0, drift_metrics["cum_error"].values, color="#D32F2F", alpha=0.15)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    final_drift = drift_metrics["cum_error"].iloc[-1]
    ax.text(0.05, 0.92, f"Final drift: ${final_drift:.2f}", transform=ax.transAxes,
            fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_title("Cumulative Prediction Error (Drift)", fontweight="bold")
    ax.set_ylabel("Cumulative Error ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25)
    ax.grid(alpha=0.2)

    # Panel 2: CUSUM drift detection
    ax = axes[1]
    cusum_result = cusum_drift_detection(drift_metrics["price_error"].values)
    ax.plot(dist_dates, cusum_result["cusum"], color="#6A1B9A", lw=2)
    ax.axhline(5.0, color="#D32F2F", ls="--", lw=1.5, alpha=0.7, label="Alarm threshold (5.0)")
    if cusum_result["drift_detected"]:
        alarm_date = dist_dates[cusum_result["drift_start_day"]]
        ax.axvline(alarm_date, color="#E65100", ls=":", lw=2, alpha=0.8,
                   label=f"Drift onset @ {alarm_date.strftime('%d %b')}")
    ax.set_title("CUSUM Drift Detection", fontweight="bold")
    ax.set_ylabel("CUSUM Statistic"); ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25)
    ax.grid(alpha=0.2)

    # Panel 3: True vs Predicted vs Corrected
    ax = axes[2]
    pred_series = drift_series_dict.get(
        next((k for k in drift_series_dict if "(no correction)" in k), None),
        prices_df[DIST_OBS_KEY])
    ax.plot(dates[zs:ze], prices_df[DIST_TRUE_KEY].iloc[zs:ze], "g-", lw=2.5,
            label="True", zorder=10)
    if best_corr_series is not None:
        ax.plot(dates[zs:ze], best_corr_series.iloc[zs:ze], "#1565C0", lw=2.2, ls="--",
                label=f"Best correction: {best_corr_name}")
    ax.plot(dates[zs:ze], prices_df[DIST_OBS_KEY].iloc[zs:ze], "#F44336", lw=1, alpha=0.35,
            label="Observed (corrupted)")
    ax.axvspan(dates[dist_start], dates[dist_end - 1], color="red", alpha=0.07)
    ax.set_title("Drift Correction — True vs Best Corrected", fontweight="bold")
    ax.set_ylabel("Price ($)"); ax.legend(fontsize=7.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25)
    ax.grid(alpha=0.2)

    fig.suptitle("Section 5 — Drift Detection & Correction", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_drift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)



def _plot_best_method(summary_df, recon, prices_df, dates, dist_start, dist_end, out_dir):
    """Highlight Best Performers: top-5 overlay, metric bar chart, and RMSE/RetCorr scatter.

    Saved as ``plot_best_method.png``.  Only advanced methods are shown
    (simple fills are excluded upstream).
    """
    zs, ze = max(0, dist_start - 10), min(len(dates), dist_end + 10)
    top_k = min(5, len(summary_df))
    top = summary_df.head(top_k)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    colors_palette = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A", "#00897B"]

    # ── Panel A: Top-5 method overlay ──
    ax = axes[0]
    ax.plot(dates[zs:ze], prices_df[DIST_TRUE_KEY].iloc[zs:ze], "g--", lw=2.5, label="True", zorder=10)
    ax.plot(dates[zs:ze], prices_df[DIST_OBS_KEY].iloc[zs:ze], "#F44336", lw=0.8, alpha=0.3, label="Observed")
    for idx, (_, row) in enumerate(top.iterrows()):
        ax.plot(dates[zs:ze], recon[row["Method"]].iloc[zs:ze],
                color=colors_palette[idx], lw=2.0,
                label=f"#{idx+1} {row['Method']}  [RMSE={row['RMSE']:.2f}]")
    ax.axvspan(dates[dist_start], dates[dist_end - 1], color="red", alpha=0.07)
    ax.set_title("Top Performers vs True Reference", fontweight="bold")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=7, ncol=1, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25)
    ax.grid(alpha=0.2)

    # ── Panel B: Grouped bar chart — RMSE, MAE, Final Gap ──
    ax = axes[1]
    x = range(len(top))
    w = 0.25
    rmse = top["RMSE"].values
    mae = top["MAE"].values
    final_gap = np.abs(top["Final Gap"].values) if "Final Gap" in top.columns else np.zeros(len(top))
    ax.bar([xi - w for xi in x], rmse, w, color="#E53935", alpha=0.8, label="RMSE")
    ax.bar(x, mae, w, color="#FB8C00", alpha=0.8, label="MAE")
    ax.bar([xi + w for xi in x], final_gap, w, color="#1565C0", alpha=0.8, label="|Final Gap|")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"#{i+1}" for i in range(len(top))], fontsize=9)
    for i in range(len(top)):
        ax.text(i, rmse[i] + 0.5, f"{rmse[i]:.2f}", ha="center", fontsize=7.5, color="#E53935", fontweight="bold")
    ax.set_title("Metric Comparison (Top-5)", fontweight="bold")
    ax.set_ylabel("Error ($)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel C: Scatter — RMSE vs Return Correlation (bubble = inverse rank) ──
    ax = axes[2]
    sizes = [250 * (top_k - i) / top_k for i in range(top_k)]
    scatter = ax.scatter(top["RMSE"], top["Return Corr"], s=sizes,
                         c=colors_palette[:top_k], alpha=0.75, edgecolors="black", linewidth=1, zorder=5)
    for idx, (_, row) in enumerate(top.iterrows()):
        ax.annotate(f"#{idx+1}", (row["RMSE"], row["Return Corr"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9, fontweight="bold")
    ax.set_xlabel("RMSE ↓", fontweight="bold")
    ax.set_ylabel("Return Correlation ↑", fontweight="bold")
    ax.set_title("Error-Return Trade-off", fontweight="bold")
    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.grid(alpha=0.3)

    # ── Annotations box with key facts ──
    best = summary_df.iloc[0]
    facts = (
        f"🏆 Best: {best['Method']}\n"
        f"   RMSE = {best['RMSE']:.4f}   MAE = {best['MAE']:.4f}\n"
        f"   Ret Corr = {best['Return Corr']:.4f}   Vol Ratio = {best['Vol Ratio']:.3f}"
    )
    fig.text(0.5, 0.01, facts, ha="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF9C4", alpha=0.9, edgecolor="#F57F17"))

    fig.suptitle("Highlight Best Performers", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    fig.savefig(out_dir / "plot_best_method.png", dpi=150, bbox_inches="tight")
    plt.close(fig)



# ═══════════════════════════════════════════════════════════════
#  8. SECTION 6 — PROPOSED IMPROVEMENTS
# ═══════════════════════════════════════════════════════════════

def ensemble_top3_reconstruction(
    recon: dict[str, pd.Series],
    summary_df: pd.DataFrame,
    prices_obs: pd.Series,
    prices_true: pd.Series,
    gap_start: int,
    gap_end: int,
    top_n: int = 3,
    method: str = "mean",
) -> dict:
    """Ensemble the top *top_n* reconstruction methods from *summary_df*.

    Parameters
    ----------
    recon : dict of {str: pd.Series}
        Dictionary of reconstruction series.
    summary_df : pd.DataFrame
        Evaluation table sorted by RMSE (best first). Must have 'Method' column.
    prices_obs : pd.Series
        Observed (corrupted) price series.
    prices_true : pd.Series
        True (uncorrupted) price series for RMSE evaluation.
    gap_start, gap_end : int
        Distress window indices.
    top_n : int
        Number of top methods to ensemble (default 3).
    method : str
        ``"mean"``, ``"median"``, or ``"rmse_weighted"``.

    Returns
    -------
    dict with keys 'predicted', 'members', 'weights', 'rmse'.
    """
    top_methods = summary_df.head(top_n)["Method"].tolist()
    top_methods = [m for m in top_methods if m in recon]
    if len(top_methods) < 2:
        return {"predicted": recon[top_methods[0]], "members": top_methods,
                "weights": [1.0], "rmse": float(summary_df.iloc[0]["RMSE"])}

    members_arr = np.array([recon[m].iloc[gap_start:gap_end].values for m in top_methods])

    if method == "mean":
        weights = np.ones(len(top_methods)) / len(top_methods)
        ensemble = np.mean(members_arr, axis=0)
    elif method == "median":
        weights = np.ones(len(top_methods)) / len(top_methods)
        ensemble = np.median(members_arr, axis=0)
    elif method == "rmse_weighted":
        rmse_vals = summary_df.set_index("Method").loc[top_methods, "RMSE"].values + 1e-10
        w = 1.0 / rmse_vals
        weights = w / w.sum()
        ensemble = np.average(members_arr, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    predicted = prices_obs.copy()
    predicted.iloc[gap_start:gap_end] = ensemble
    rmse_val = np.sqrt(mean_squared_error(
        prices_true.iloc[gap_start:gap_end].values, ensemble))
    return {"predicted": predicted, "members": top_methods,
            "weights": weights.tolist() if hasattr(weights, "tolist") else list(weights),
            "rmse": float(rmse_val)}


def tune_hyperparameters_cv(
    ret_full: pd.DataFrame,
    prices_obs: pd.Series,
    true_prices: pd.Series,
    gap_start: int,
    gap_end: int,
    lookback_grid: list[int] | None = None,
    threshold_grid: list[float] | None = None,
    peer_grid: list[int] | None = None,
    cv_splits: int = 3,
) -> pd.DataFrame:
    """Tune ML proxy hyperparameters using time-series cross-validation.

    Uses the distress window as the test set, evaluating parameter
    combinations via TimeSeriesSplit on pre-distress data.

    Parameters
    ----------
    ret_full : pd.DataFrame
        Full return panel (must have 'DIST' column).
    prices_obs : pd.Series
        Observed (corrupted) prices.
    true_prices : pd.Series
        True (uncorrupted) prices.
    gap_start, gap_end : int
        Distress window indices.
    lookback_grid, threshold_grid, peer_grid : lists, optional
        Parameter grids.
    cv_splits : int
        Number of time-series CV folds (default 3).

    Returns
    -------
    pd.DataFrame sorted by CV RMSE (best first).
    """
    if lookback_grid is None:
        lookback_grid = _c("sweep", "lookback_grid", default=[40, 60, 80, 120])
    if threshold_grid is None:
        threshold_grid = _c("sweep", "threshold_grid", default=[0.25, 0.35, 0.45, 0.55])
    if peer_grid is None:
        peer_grid = [3, 5, 7]

    pre_start = max(0, gap_start - max(lookback_grid) - 10)
    pre_window = ret_full.iloc[pre_start:gap_start]
    rows = []

    import itertools
    for lb, thr, mp in itertools.product(lookback_grid, threshold_grid, peer_grid):
        if len(pre_window) < 2 * lb:
            continue

        tscv = TimeSeriesSplit(n_splits=cv_splits)
        cv_rmse = []

        for train_idx, val_idx in tscv.split(pre_window):
            train_win = pre_window.iloc[train_idx]
            val_win = pre_window.iloc[val_idx]
            if len(train_win) < lb or len(val_win) < 5:
                continue

            try:
                dist_mat = feature_distance_matrix(train_win, use_pca=False)
                linkage_mat = linkage(squareform(dist_mat.values), method="ward")
                labels = fcluster(linkage_mat, t=thr, criterion="distance")
                stocks = dist_mat.columns.tolist()
                clust_id = labels[stocks.index("DIST")]
                same_cluster = [s for s, cid in zip(stocks, labels)
                                if cid == clust_id and s != "DIST"]
                corr_mat = train_win.corr()
                raw_corr = corr_mat.loc["DIST"].drop("DIST")
                peers = (sorted(same_cluster, key=lambda s: abs(raw_corr[s]), reverse=True)[:mp]
                         if same_cluster else raw_corr.abs().sort_values(ascending=False).index[:mp].tolist())
                peer_cols = [c for c in peers if c in train_win.columns]
                if len(peer_cols) < 2:
                    continue

                model = LinearRegression().fit(train_win[peer_cols].values, train_win["DIST"].values)
                y_pred = model.predict(val_win[peer_cols].values)
                cv_rmse.append(np.sqrt(mean_squared_error(val_win["DIST"].values, y_pred)))
            except Exception:
                continue

        if not cv_rmse:
            continue

        # Train on full pre-window and test on distress
        try:
            dist_mat = feature_distance_matrix(pre_window, use_pca=False)
            linkage_mat = linkage(squareform(dist_mat.values), method="ward")
            labels = fcluster(linkage_mat, t=thr, criterion="distance")
            stocks = dist_mat.columns.tolist()
            clust_id = labels[stocks.index("DIST")]
            same_cluster = [s for s, cid in zip(stocks, labels) if cid == clust_id and s != "DIST"]
            corr_mat = pre_window.corr()
            raw_corr = corr_mat.loc["DIST"].drop("DIST")
            peers = (sorted(same_cluster, key=lambda s: abs(raw_corr[s]), reverse=True)[:mp]
                     if same_cluster else raw_corr.abs().sort_values(ascending=False).index[:mp].tolist())
            peer_cols = [c for c in peers if c in pre_window.columns]
            if len(peer_cols) < 2:
                continue

            model = LinearRegression().fit(pre_window[peer_cols].values, pre_window["DIST"].values)
            gap_ret = ret_full[peer_cols].iloc[gap_start:gap_end].values
            pred_log_ret = model.predict(gap_ret)
            reconstructed = prices_obs.copy()
            for i in range(gap_end - gap_start):
                reconstructed.iloc[gap_start + i] = reconstructed.iloc[gap_start + i - 1] * np.exp(pred_log_ret[i])

            true_seg = true_prices.iloc[gap_start:gap_end].values
            pred_seg = reconstructed.iloc[gap_start:gap_end].values
            test_rmse = float(np.sqrt(mean_squared_error(true_seg, pred_seg)))
            ret_corr = float(np.corrcoef(
                np.log(pred_seg / reconstructed.iloc[gap_start - 1:gap_end - 1].values),
                np.log(true_seg / true_prices.iloc[gap_start - 1:gap_end - 1].values)
            )[0, 1])
        except Exception:
            continue

        rows.append({"lookback": lb, "threshold": thr, "max_peers": mp,
                      "n_peers_used": len(peer_cols),
                      "cv_rmse_mean": float(np.mean(cv_rmse)),
                      "cv_rmse_std": float(np.std(cv_rmse)),
                      "test_rmse": test_rmse, "ret_corr": ret_corr})

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True)


def auto_select_method(
    prices_obs: pd.Series,
    ret_full: pd.DataFrame,
    gap_start: int,
    gap_end: int,
    recon: dict[str, pd.Series],
    summary_df: pd.DataFrame | None = None,
) -> dict:
    """Automatically select the best reconstruction method based on distress characteristics.

    Analyses volatility ratio, drift persistence, and window length,
    then applies a rule-based scoring system.

    Returns
    -------
    dict with keys: recommended_method, distress_stats, rationale, scores.
    """
    from collections import Counter
    pre_ret = ret_full["DIST"].iloc[max(0, gap_start - 60):gap_start]
    during_ret = ret_full["DIST"].iloc[gap_start:gap_end]
    pre_vol = pre_ret.std()
    during_vol = during_ret.std()
    vol_ratio = during_vol / pre_vol if pre_vol > 0 else 1.0

    window_length = gap_end - gap_start
    is_long_window = window_length >= 20

    obs_prices = prices_obs.iloc[gap_start:gap_end].values
    half = len(obs_prices) // 2
    if half > 1:
        fh = np.mean(obs_prices[:half] - obs_prices[0])
        sh = np.mean(obs_prices[half:] - obs_prices[0])
        has_persistent_drift = abs(sh - fh) > 0.05 * obs_prices[0]
    else:
        has_persistent_drift = False

    scores = Counter()
    if vol_ratio > 1.5:
        scores["GP-RBF"] += 2; scores["Elastic Net+CV"] += 1; scores["ML Stable+OLS"] += 1
    elif vol_ratio < 0.7:
        scores["Static Proxy+OLS"] += 1; scores["Static Proxy+Ridge"] += 1
    else:
        scores["Kalman Beta"] += 1; scores["Static Proxy+OLS"] += 1

    if is_long_window:
        scores["Kalman Beta"] += 2; scores["GP-RBF"] += 2
        scores["Dynamic Proxy+OLS"] += 1; scores["Coint Proxy: Coint VRW"] += 1
    else:
        scores["Static Proxy+OLS"] += 2; scores["Elastic Net+CV"] += 1

    if has_persistent_drift:
        scores["Coint Proxy: Coint VRW"] += 2; scores["Coint Proxy: Coint EW"] += 1
        scores["Elastic Net+CV"] += 1
    else:
        scores["Feature Raw Proxy+OLS"] += 1; scores["ML Proxy+Opt (Best Sweep)"] += 1

    available = set(recon.keys())
    valid_scores = {k: v for k, v in scores.items() if k in available}
    recommended = max(valid_scores, key=valid_scores.get) if valid_scores else list(available)[0]
    best_from_data = summary_df.iloc[0]["Method"] if summary_df is not None and not summary_df.empty else None

    distress_stats = {"window_length": window_length, "is_long_window": is_long_window,
                      "vol_ratio": float(vol_ratio), "has_persistent_drift": has_persistent_drift,
                      "pre_vol": float(pre_vol), "during_vol": float(during_vol)}
    rationale_parts = [f"window={window_length}d {'(long)' if is_long_window else '(short)'}",
                       f"vol_ratio={vol_ratio:.2f} {'(spiking)' if vol_ratio > 1.5 else '(stable)'}"]
    if has_persistent_drift:
        rationale_parts.append("persistent drift detected")

    return {"recommended_method": recommended, "distress_stats": distress_stats,
            "rationale": "; ".join(rationale_parts), "best_from_evaluation": best_from_data,
            "scores": dict(valid_scores)}


def _plot_improvements(compare_df, recon, prices_df, dates, dist_start, dist_end, out_dir):
    """Section 6 — Proposed Improvements: ensemble, CV tuning comparison graph.

    Two panels: price overlay of improvements vs true, and RMSE bar chart.
    Saved as ``plot_improvements.png``.
    """
    zs, ze = max(0, dist_start - 15), min(len(dates), dist_end + 15)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors_imp = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A", "#00897B"]

    # Panel A: Price reconstruction overlay
    ax = axes[0]
    ax.plot(dates[zs:ze], prices_df[DIST_TRUE_KEY].iloc[zs:ze], "g--", lw=2.5, label="True", zorder=10)
    ax.plot(dates[zs:ze], prices_df[DIST_OBS_KEY].iloc[zs:ze], "#F44336", lw=1, alpha=0.35, label="Observed")
    for idx, (_, r) in enumerate(compare_df.iterrows()):
        if r["Method"] not in recon:
            continue
        ax.plot(dates[zs:ze], recon[r["Method"]].iloc[zs:ze], color=colors_imp[idx % len(colors_imp)],
                lw=2, label=f"{r['Method']} (RMSE={r['RMSE']:.3f})")
    ax.axvspan(dates[dist_start], dates[dist_end - 1], color="red", alpha=0.07)
    ax.set_title("Panel A — Improvement Methods vs True Prices", fontweight="bold")
    ax.set_ylabel("Price ($)"); ax.legend(fontsize=7, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25); ax.grid(alpha=0.2)

    # Panel B: RMSE bar chart
    ax = axes[1]
    x_pos = range(len(compare_df))
    bar_colors = [colors_imp[i % len(colors_imp)] for i in range(len(compare_df))]
    bars = ax.bar(x_pos, compare_df["RMSE"].values, color=bar_colors, edgecolor="white", width=0.6)
    ax.set_xticks(list(x_pos))
    short_names = [m[:25] + "..." if len(m) > 25 else m for m in compare_df["Method"]]
    ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=8)
    ax.set_title("Panel B — RMSE Comparison", fontweight="bold")
    ax.set_ylabel("RMSE"); ax.grid(axis="y", alpha=0.2)
    for i, (_, r) in enumerate(compare_df.iterrows()):
        ax.text(i, r["RMSE"] + 0.02, f'{r["RMSE"]:.3f}', ha="center", fontsize=8, fontweight="bold")

    fig.suptitle("Section 6 — Proposed Improvements: Ensemble, CV Tuning & Auto-Selection",
                 fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_improvements.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  9. PIPELINE
# ═══════════════════════════════════════════════════════════════
def run_pipeline(target_ticker="MSFT", data_dir="data", output_dir=None,
                 n_days=None, dist_depth=None,
                 lookback=None, cluster_thr=None,
                 max_peers=None, min_peers=None,
                 dist_length=None, run_sweep=None,
                 dist_start_date=None, dist_end_date=None,
                 config_path=None):
    # Resolve config — allow overriding via explicit config_path
    if config_path is not None:
        _override_config(config_path)
    # Resolve defaults from config (with fallback to hard-coded constants)
    if n_days is None:
        n_days = _c("data", "n_days", default=DEFAULT_N_DAYS)
    if dist_depth is None:
        dist_depth = _c("distress", "dist_depth", default=DEFAULT_DIST_DEPTH)
    if dist_length is None:
        dist_length = _c("distress", "dist_length", default=DEFAULT_DIST_LENGTH)
    if dist_start_date is None:
        dist_start_date = _c("distress", "dist_start_date", default=DEFAULT_DIST_START_DATE)
    if dist_end_date is None:
        dist_end_date = _c("distress", "dist_end_date", default=DEFAULT_DIST_END_DATE)
    if lookback is None:
        lookback = _c("peers", "lookback", default=DEFAULT_LOOKBACK)
    if cluster_thr is None:
        cluster_thr = _c("peers", "cluster_thr", default=DEFAULT_CLUSTER_THR)
    if max_peers is None:
        max_peers = _c("peers", "max_peers", default=DEFAULT_MAX_PEERS)
    if min_peers is None:
        min_peers = _c("peers", "min_peers", default=DEFAULT_MIN_PEERS)
    if run_sweep is None:
        run_sweep = _c("sweep", "run_sweep", default=True)
    """Run full pipeline for a single target. Returns per-target summary dict."""
    # Output dir = RESULT_PATH / target_ticker (or explicit output_dir)
    if output_dir:
        out = Path(output_dir)
    else:
        out = RESULT_PATH / target_ticker.lower()
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}\n  TARGET: {target_ticker}\n{'=' * 70}")

    # ── Step 1-2: Data & Peers ──
    instruments_df, stocks_data_df = load_data(data_dir)
    universe = select_peers(target_ticker=target_ticker, instruments_df=instruments_df,
                            stocks_data_df=stocks_data_df)
    prices_wide, model_peer_cols, peers_df = (universe["prices_wide"],
                                               universe["model_peer_cols"], universe["peers_df"])
    selected = [target_ticker] + universe["same_sector_peers"]
    raw = prices_wide[selected].dropna().copy()
    n_avail = min(n_days, len(raw))
    raw = raw.iloc[-n_avail:]; dates = raw.index; n_avail = len(raw)

    # ── Step 3: Price panel + distress injection ──
    prices_df = pd.DataFrame(index=dates)
    for t in universe["same_sector_peers"] + [target_ticker]:
        prices_df[t] = raw[t]
    prices_df[TARGET_ORIG_KEY] = raw[target_ticker]
    prices_df[DIST_TRUE_KEY] = prices_df[TARGET_ORIG_KEY].copy()

    # Resolve distress window: dates take priority over index-based calculation
    if dist_start_date is not None and dist_end_date is not None:
        dist_start_dt = pd.Timestamp(dist_start_date)
        dist_end_dt = pd.Timestamp(dist_end_date)
        if dist_start_dt not in dates or dist_end_dt not in dates:
            raise ValueError(f"Distress dates {dist_start_date}–{dist_end_date} not in data range "
                             f"({dates[0].date()} to {dates[-1].date()})")
        dist_start = dates.get_loc(dist_start_dt)
        dist_end = dates.get_loc(dist_end_dt) + 1  # +1 because we use Python slice convention
        print(f"  Distress window from config: {dist_start_date} → {dist_end_date} "
              f"(indices {dist_start}–{dist_end-1})")
    else:
        dist_start = max(40, int(0.60 * n_avail))
        dist_end = min(dist_start + dist_length, n_avail - 5)
    obs = prices_df[DIST_TRUE_KEY].values.copy()
    for i in range(dist_end - dist_start):
        obs[dist_start + i] *= 1 - dist_depth * np.sin(np.pi * i / (dist_end - dist_start - 1))
    prices_df[DIST_OBS_KEY] = obs
    print(f"  Sector: {universe['target_sector']}  |  Samples: {n_avail}  |  "
          f"Distress: {dist_start}–{dist_end-1}")

    ret_df = np.log(prices_df[model_peer_cols + [DIST_OBS_KEY]] /
                    prices_df[model_peer_cols + [DIST_OBS_KEY]].shift(1)).dropna()
    ret_df = ret_df.rename(columns={DIST_OBS_KEY: "DIST"})

    # ── Step 4: Reconstruct ──
    print("  3A — Simple Fill ...")
    recon = dict(simple_fill_reconstruction(prices_df[DIST_OBS_KEY], dist_start, dist_end))

    win_s, win_e = max(0, dist_start - 1 - lookback), dist_start - 1
    print("  3B — Static Proxy + OLS ...")
    s_s, _ = static_proxy_reconstruction(ret_df.iloc[win_s:win_e].copy(), ret_df,
                                          prices_df[DIST_OBS_KEY], dist_start, dist_end,
                                          cluster_thr=cluster_thr, min_peers=min_peers,
                                          max_peers=max_peers, peers_df_override=peers_df)
    recon["Static Proxy+OLS"] = s_s

    print("  3C — Dynamic Proxy + OLS ...")
    d_s, _ = dynamic_proxy_reconstruction(ret_df, prices_df[DIST_OBS_KEY], dist_start, dist_end,
                                           lookback=lookback, cluster_thr=cluster_thr,
                                           min_peers=min_peers, max_peers=max_peers,
                                           peers_df_override=peers_df)
    recon["Dynamic Proxy+OLS"] = d_s

    print("  3D — ML Proxy ...")
    ml_win = ret_df.iloc[win_s:win_e].copy()
    for up, lbl in [(False, "Raw"), (True, "PCA")]:
        ml_s, _ = ml_proxy_reconstruction(ml_win, ret_df, prices_df[DIST_OBS_KEY],
                                          dist_start, dist_end, use_pca=up,
                                          cluster_thr=cluster_thr, max_peers=max_peers)
        recon[f"Feature {lbl} Proxy+OLS"] = ml_s

    # ── Additional v3 methods ──
    print("  3C(alt) — Elastic Net Proxy ...")
    try:
        series_en, info_en = elastic_net_proxy_reconstruction(
            ret_df.iloc[win_s:win_e].copy(), ret_df, prices_df[DIST_OBS_KEY],
            dist_start, dist_end, l1_ratio=0.5, use_cv=True)
        recon["Elastic Net+CV"] = series_en
        print(f"    Active: {info_en['n_active']}/{info_en['n_total_peers']} peers (α={info_en['alpha_used']:.4f})")
    except Exception as e:
        print(f"    (skipped: {e})")

    print("  3D(alt) — Gaussian Process Proxy ...")
    try:
        series_gpr, info_gpr = gpr_reconstruction(
            ret_df.iloc[win_s:win_e].copy(), ret_df, prices_df[DIST_OBS_KEY],
            dist_start, dist_end, kernel_type='rbf', n_restarts=10, max_peers=max_peers)
        recon["GP-RBF"] = series_gpr
        print(f"    Kernel: {info_gpr['kernel'][:60]}...")
    except Exception as e:
        print(f"    (skipped: {e})")

    print("  3D(alt2) — Kalman Time-Varying Beta ...")
    try:
        series_kalman, info_kalman = kalman_time_varying_beta_reconstruction(
            ret_df.iloc[win_s:win_e].copy(), ret_df, prices_df[DIST_OBS_KEY],
            dist_start, dist_end, max_peers=max_peers, process_noise=1e-4)
        recon["Kalman Beta"] = series_kalman
        print(f"    Peers: {', '.join(info_kalman['peers'][:3])}... R²={info_kalman['r2_train']:.3f}")
    except Exception as e:
        print(f"    (skipped: {e})")

    # ── Enhanced sweep (multi-objective) ──
    sweep_meta = {"best_params": (None, None), "best_rmse": np.nan}
    enhanced_sweep_meta = {}
    if run_sweep:
        print("  3E — ML Proxy Optimisation (sweep) ...")
        sweep_df, best_sw, sweep_meta = ml_proxy_optimisation(
            ret_df, prices_df, dist_start, dist_end, default_max_peers=max_peers)
        sweep_df.to_csv(out / "sweep_results.csv", index=False)
        if best_sw is not None:
            recon["ML Proxy+Opt (Best Sweep)"] = best_sw
        bp = sweep_meta["best_params"]
        print(f"    Basic sweep best: lookback={bp[0]}, threshold={bp[1]:.2f}  (RMSE={sweep_meta['best_rmse']:.4f})")

        print("  3E(enhanced) — Multi-Objective Random Search ...")
        try:
            enhanced_df, enh_best_series, enh_best_params = ml_proxy_enhanced_optimisation(
                ret_df, prices_df, dist_start, dist_end, n_iter=50, random_state=42,
                default_max_peers=max_peers)
            if enh_best_series is not None and not enhanced_df.empty:
                recon["ML Proxy+Enhanced (Multi-Obj)"] = enh_best_series
                enhanced_sweep_meta = {
                    "enhanced_best_params": enh_best_params,
                    "enhanced_best_rmse": float(enhanced_df.iloc[0]['RMSE']),
                    "enhanced_best_mape": float(enhanced_df.iloc[0]['MAPE (%)']),
                    "enhanced_best_dir_acc": float(enhanced_df.iloc[0]['Dir Accuracy']),
                }
                enhanced_df.to_csv(out / "sweep_results_enhanced.csv", index=False)
                print(f"    Enhanced: LB={enh_best_params['lookback']}, "
                      f"THR={enh_best_params['threshold']:.2f}, "
                      f"MP={enh_best_params['max_peers']}  (RMSE={enhanced_df.iloc[0]['RMSE']:.4f})")
        except Exception as e:
            print(f"    (skipped: {e})")

    # ── Step 5: Evaluate ──
    print("  Evaluating ...")
    simple_fill_methods = {"Forward Fill", "Backward Fill", "Linear Interp"}
    # Full evaluation (all methods — used for the comparison grid plot)
    summary_df_all = evaluate_reconstructions(recon, prices_df, dist_start, dist_end)
    summary_df_all.to_csv(out / "evaluation.csv", index=False)
    # Filtered evaluation (advanced methods only — used for best-method selection)
    summary_df = summary_df_all[~summary_df_all["Method"].isin(simple_fill_methods)].copy()
    summary_df = summary_df.reset_index(drop=True)
    recon_df = pd.DataFrame(recon)
    recon_df.to_csv(out / "reconstructions.csv")

    best_name = summary_df.iloc[0]["Method"]
    best_rmse_val = summary_df.iloc[0]["RMSE"]
    print(f"  Best: {best_name}  (RMSE={best_rmse_val:.4f})")

    # ── Step 6: Drift ──
    print("  Drift corrections ...")
    drift_series_dict, drift_df = apply_drift_corrections(
        best_name, recon[best_name], prices_df, dist_start, dist_end, model_peer_cols)
    drift_df.to_csv(out / "drift_results.csv")

    # Save original and corrected time series to asset folder
    print(f"  Saving time series to {out} ...")
    # Original time series
    prices_df[[DIST_TRUE_KEY, DIST_OBS_KEY]].to_csv(out / "prices_original.csv")
    # All reconstructions
    recon_df.to_csv(out / "reconstructions.csv")
    # Drift-corrected series
    pd.DataFrame(drift_series_dict).to_csv(out / "drift_corrected_series.csv")
    # Combined comparison: true, observed, best prediction, best correction
    comparison_df = pd.DataFrame({
        DIST_TRUE_KEY: prices_df[DIST_TRUE_KEY],
        DIST_OBS_KEY: prices_df[DIST_OBS_KEY],
        'Best_Prediction': recon[best_name],
    })
    # Add best correction if available
    if len(drift_series_dict) > 1:
        corr_names = [n for n in drift_series_dict if "(no correction)" not in n]
        if corr_names:
            # Find best correction by RMSE
            best_corr_name = min(corr_names, key=lambda n: drift_df[drift_df['Method'] == n]['RMSE'].values[0]
                                 if len(drift_df[drift_df['Method'] == n]) > 0 else np.inf)
            comparison_df['Best_Correction'] = drift_series_dict[best_corr_name]
    comparison_df.to_csv(out / "comparison_timeseries.csv")
    print(f"    Saved: prices_original.csv, drift_corrected_series.csv, comparison_timeseries.csv")

    # ── Step 7: Backtest ──
    print("  Backtesting ...")
    bt_df, eq_true, eq_obs, perf_true, perf_obs = run_backtests(recon, prices_df)
    bt_df.to_csv(out / "backtest_results.csv")

    # ── Section 6 — Proposed Improvements ──
    print("  Section 6 — Proposed Improvements ...")
    improvement_meta = {}
    try:
        # Improvement 1: Ensemble averaging of top 3 methods
        for method_name in ("mean", "median", "rmse_weighted"):
            ens_result = ensemble_top3_reconstruction(
                recon, summary_df, prices_df[DIST_OBS_KEY], prices_df[DIST_TRUE_KEY],
                dist_start, dist_end, top_n=3, method=method_name)
            label = f"Ensemble Top3 ({method_name})"
            recon[label] = ens_result["predicted"]
            improvement_meta[f"Ensemble_{method_name}_RMSE"] = ens_result["rmse"]
            print(f"    {label:<35} RMSE={ens_result['rmse']:.4f}")

        # Improvement 2: Cross-validation hyperparameter tuning
        cv_grid = {"lookback": [40, 60, 80], "threshold": [0.25, 0.35, 0.45], "max_peers": [3, 5]}
        cv_results = tune_hyperparameters_cv(
            ret_df, prices_df[DIST_OBS_KEY], prices_df[DIST_TRUE_KEY],
            dist_start, dist_end,
            lookback_grid=cv_grid["lookback"],
            threshold_grid=cv_grid["threshold"],
            peer_grid=cv_grid["max_peers"])
        if not cv_results.empty:
            best_cv = cv_results.iloc[0]
            improvement_meta["CV_Best_LB"] = int(best_cv["lookback"])
            improvement_meta["CV_Best_THR"] = float(best_cv["threshold"])
            improvement_meta["CV_Best_MP"] = int(best_cv["max_peers"])
            improvement_meta["CV_RMSE"] = float(best_cv["test_rmse"])
            print(f"    CV Tuning: LB={best_cv['lookback']}, THR={best_cv['threshold']:.2f}, "
                  f"MP={best_cv['max_peers']}  (RMSE={best_cv['test_rmse']:.4f})")

            # Store CV result in recon
            best_label = f"ML Proxy+CV (LB={int(best_cv['lookback'])})"
            if best_label not in recon:
                win_start_cv = max(0, dist_start - 1 - int(best_cv["lookback"]))
                win_end_cv = dist_start - 1
                pre_win = ret_df.iloc[win_start_cv:win_end_cv].copy()
                dist_mat = feature_distance_matrix(pre_win, use_pca=False)
                linkage_mat = linkage(squareform(dist_mat.values), method="ward")
                labels = fcluster(linkage_mat, t=best_cv["threshold"], criterion="distance")
                stocks = dist_mat.columns.tolist()
                clust_id = labels[stocks.index("DIST")]
                same_cluster = [s for s, cid in zip(stocks, labels)
                                if cid == clust_id and s != "DIST"]
                corr_mat = pre_win.corr()
                raw_corr = corr_mat.loc["DIST"].drop("DIST")
                peers = (sorted(same_cluster, key=lambda s: abs(raw_corr[s]), reverse=True)[:int(best_cv["max_peers"])]
                         if same_cluster else raw_corr.abs().sort_values(ascending=False).index[:int(best_cv["max_peers"])].tolist())
                peer_cols = [p for p in peers if p in pre_win.columns]
                if peer_cols:
                    model = LinearRegression().fit(pre_win[peer_cols].values, pre_win["DIST"].values)
                    gap_ret_cv = ret_df[peer_cols].iloc[dist_start:dist_end].values
                    pred_log_ret_cv = model.predict(gap_ret_cv)
                    series_cv = prices_df[DIST_OBS_KEY].copy()
                    for i in range(dist_end - dist_start):
                        series_cv.iloc[dist_start + i] = series_cv.iloc[dist_start + i - 1] * np.exp(pred_log_ret_cv[i])
                    recon[best_label] = series_cv

        # Improvement 3: Automated method selection
        selection = auto_select_method(
            prices_df[DIST_OBS_KEY], ret_df, dist_start, dist_end, recon, summary_df)
        improvement_meta["AutoSelect_Recommended"] = selection["recommended_method"]
        improvement_meta["AutoSelect_BestFromData"] = selection.get("best_from_evaluation", "")
        print(f"    Auto-Select: recommends '{selection['recommended_method']}', "
              f"data says '{selection.get('best_from_evaluation', 'N/A')}'")

    except Exception as e:
        print(f"    (improvements skipped: {e})")

    # ── Re-evaluate summary_df_all to include ensemble/CV methods in the grid ──
    summary_df_all = evaluate_reconstructions(recon, prices_df, dist_start, dist_end)
    summary_df_all.to_csv(out / "evaluation.csv", index=False)

    # ── Step 8: Plots (3 plots) ──
    #   plot_comparison_grid.png  — all methods including simple fills + ensembles
    #   plot_best_method.png      — highlight best performers (advanced methods only)
    #   plot_improvements.png     — Section 6: ensemble + CV tuning comparison
    print("  Generating plots ...")
    try:
        _plot_comparison_grid(summary_df_all, recon, prices_df, dates, dist_start, dist_end, out)
        _plot_best_method(summary_df, recon, prices_df, dates, dist_start, dist_end, out)

        # Plot 3: Section 6 — Improvements (ensemble, CV tuning comparison)
        improvement_names = [k for k in recon if k.startswith("Ensemble Top3") or "CV (LB" in k]
        compare_rows = []
        for name in improvement_names:
            if name not in recon:
                continue
            pred = recon[name].iloc[dist_start:dist_end].values
            true = prices_df[DIST_TRUE_KEY].iloc[dist_start:dist_end].values
            pred_ret = np.log(pred / recon[name].iloc[dist_start - 1:dist_end - 1].values)
            true_ret = np.log(true / prices_df[DIST_TRUE_KEY].iloc[dist_start - 1:dist_end - 1].values)
            compare_rows.append({"Method": name,
                                 "RMSE": np.sqrt(mean_squared_error(true, pred)),
                                 "MAE": np.mean(np.abs(true - pred)),
                                 "Ret Corr": np.corrcoef(pred_ret, true_ret)[0, 1]})
        if summary_df is not None and not summary_df.empty:
            best_base = summary_df.iloc[0]
            compare_rows.append({"Method": f"{best_base['Method']} (best baseline)",
                                 "RMSE": best_base["RMSE"], "MAE": best_base["MAE"],
                                 "Ret Corr": best_base["Return Corr"]})
        if compare_rows:
            compare_df = pd.DataFrame(compare_rows).sort_values("RMSE").reset_index(drop=True)
            _plot_improvements(compare_df, recon, prices_df, dates, dist_start, dist_end, out)
    except Exception as e:
        print(f"  (plotting skipped: {e})")

    # ── Report ──
    report = [
        "=" * 72, f"  GAP-FILLING REPORT  |  Target: {target_ticker}", "=" * 72,
        f"  Sector      : {universe['target_sector']}",
        f"  Samples     : {n_avail}",
        f"  Distress    : {dates[dist_start].date()} – {dates[dist_end-1].date()}",
        f"  Lookback    : {lookback}  |  Threshold: {cluster_thr:.2f}", "",
        "-" * 72, "  EVALUATION (all methods, sorted by RMSE)", "-" * 72]
    for _, r in summary_df_all.iterrows():
        report.append(f"  {r['Method']:<35s}  RMSE={r['RMSE']:.4f}  "
                      f"MAE={r['MAE']:.4f}  RetCorr={r['Return Corr']:.4f}")
    report += ["", f"  * Best (advanced method): {best_name} (RMSE={best_rmse_val:.4f})",
               "", "-" * 72, "  DRIFT CORRECTION", "-" * 72]
    for _, r in drift_df.iterrows():
        report.append(f"  {r['Method']:<45s}  RMSE={r['RMSE']:.4f}  "
                      f"MaxDrift={r['Max Drift']:.4f}")
    report += ["", "-" * 72, "  BACKTESTING", "-" * 72,
               "  TRUE     : TotalReturn={:.4f}  Sharpe={:.3f}".format(
                   perf_true["Total Return"], perf_true["Sharpe Ratio"]),
               "  OBSERVED : TotalReturn={:.4f}  Sharpe={:.3f}".format(
                   perf_obs["Total Return"], perf_obs["Sharpe Ratio"]),
               "=" * 72]
    (out / "report.txt").write_text("\n".join(report) + "\n")

    # ── Return summary row for consolidation ──
    return {
        "Target": target_ticker,
        "Sector": universe["target_sector"],
        "Best_Method": best_name,
        "RMSE": summary_df.iloc[0]["RMSE"],
        "MAE": summary_df.iloc[0]["MAE"],
        "Return_Corr": summary_df.iloc[0]["Return Corr"],
        "N_Methods": len(summary_df),
        "N_Peers": len(universe["model_peer_cols"]),
        "Distress_Start": str(dates[dist_start].date()),
        "Distress_End": str(dates[dist_end - 1].date()),
        "Sweep_Best_LB": sweep_meta["best_params"][0],
        "Sweep_Best_THR": sweep_meta["best_params"][1],
        "Sweep_Best_RMSE": sweep_meta["best_rmse"],
        "Enhanced_Sweep_LB": enhanced_sweep_meta.get("enhanced_best_params", {}).get("lookback"),
        "Enhanced_Sweep_THR": enhanced_sweep_meta.get("enhanced_best_params", {}).get("threshold"),
        "Enhanced_Sweep_MP": enhanced_sweep_meta.get("enhanced_best_params", {}).get("max_peers"),
        "Enhanced_Sweep_RMSE": enhanced_sweep_meta.get("enhanced_best_rmse"),
        "Enhanced_Sweep_MAPE": enhanced_sweep_meta.get("enhanced_best_mape"),
        "Enhanced_Sweep_DirAcc": enhanced_sweep_meta.get("enhanced_best_dir_acc"),
        "True_Return": perf_true["Total Return"],
        "True_Sharpe": perf_true["Sharpe Ratio"],
        "Observed_Return": perf_obs["Total Return"],
        "Observed_Sharpe": perf_obs["Sharpe Ratio"],
        "Output_Dir": str(out),
    }


# ═══════════════════════════════════════════════════════════════
#  9. MULTI-TARGET WRAPPER
# ═══════════════════════════════════════════════════════════════
def run_multi_target(target_list, data_dir="data", output_dir=None,
                     n_days=None, dist_depth=None,
                     lookback=None, cluster_thr=None,
                     max_peers=None, min_peers=None,
                     dist_length=None, run_sweep=None,
                     config_path=None):
    """Run pipeline for a list of (ticker, sector) tuples.  Returns consolidated DataFrame."""
    if output_dir:
        base = Path(output_dir)
    else:
        base = RESULT_PATH
    base.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for ticker, sector in target_list:
        print(f"\n{'#' * 72}\n  RUNNING: {ticker} ({sector})\n{'#' * 72}")
        try:
            row = run_pipeline(
                target_ticker=ticker, data_dir=data_dir,
                n_days=n_days, dist_depth=dist_depth, dist_length=dist_length,
                lookback=lookback, cluster_thr=cluster_thr,
                max_peers=max_peers, min_peers=min_peers,
                run_sweep=run_sweep, config_path=config_path)
            all_rows.append(row)
        except Exception as e:
            print(f"  ERROR on {ticker}: {e}")
            all_rows.append({"Target": ticker, "Sector": sector, "Best_Method": "ERROR",
                             "RMSE": np.nan, "MAE": np.nan, "Return_Corr": np.nan,
                             "N_Methods": 0, "N_Peers": 0, "Distress_Start": "",
                             "Distress_End": "", "Sweep_Best_LB": None,
                             "Sweep_Best_THR": None, "Sweep_Best_RMSE": np.nan,
                             "True_Return": np.nan, "True_Sharpe": np.nan,
                             "Observed_Return": np.nan, "Observed_Sharpe": np.nan,
                             "Output_Dir": ""})

    consolidated = pd.DataFrame(all_rows).sort_values("RMSE").reset_index(drop=True)
    cons_path = base / "consolidated_report.csv"
    consolidated.to_csv(cons_path, index=False)
    print(f"\n{'=' * 72}")
    print(f"  CONSOLIDATED REPORT saved to {cons_path}")
    print(f"{'=' * 72}")
    print(consolidated[["Target", "Sector", "Best_Method", "RMSE",
                         "Return_Corr", "True_Return"]].to_string(index=False))
    return consolidated


# ═══════════════════════════════════════════════════════════════
#  10. CLI
# ═══════════════════════════════════════════════════════════════
def main():
    """Command-line entry point for the Gap-Filling Pipeline.

    Orchestrates the full pipeline based on CLI arguments.  Supports three
    execution modes depending on the ``--target`` value:

    ── Single-target mode ────────────────────────────────────────────
    ``--target MSFT`` (or any single ticker)

    1. Loads instrument metadata and stock price data from ``--data-dir``.
    2. Selects same-sector peer candidates via ``select_peers()``.
    3. Builds a price panel with an injected distress event (sinusoidal
       corruption over ``--dist-length`` days at ``--dist-depth`` depth).
    4. Runs all reconstruction methods:
       - **3A** Simple Fill (Forward / Backward / Linear Interpolation)
       - **3B** Static Proxy + OLS  (single peer snapshot + regression)
       - **3C** Dynamic Proxy + OLS (rolling daily peer re-selection)
       - **3C(alt)** Elastic Net + CV (sparse peer selection via L1+L2)
       - **3D** ML Proxy (Raw + PCA feature-based clustering + OLS)
       - **3D(alt)** Gaussian Process (GP-RBF, non-parametric + uncertainty)
       - **3D(alt2)** Kalman Time-Varying Beta (adaptive beta_t)
       - **3E** ML Proxy Optimisation (grid search over ``--lookback`` ×
               ``--threshold``, unless ``--no-sweep`` is set).
       - **3E(enhanced)** Multi-Objective Random Search (out-of-sample
               validation, MAPE/Dir Acc/Tracking Error metrics, composite
               ranking).
    5. Evaluates every method against the true (uncorrupted) prices using
       RMSE, MAE, Return Correlation, Price Correlation, Vol Ratio, and
       Final Gap.
    6. Applies drift detection and three correction methods (Residual
       Bias, Rolling Bias, Error Feedback) to the best-performing
       reconstruction.
    7. Backtests a mean-reversion strategy on every reconstructed series
       and compares with the true and observed (corrupted) paths.
    8. **Section 6 — Proposed Improvements:**
       - **6A** Ensemble averaging of top 3 methods (mean, median, RMSE-weighted).
       - **6B** Cross-validation hyperparameter tuning (TimeSeriesSplit).
       - **6C** Automated method selection based on distress characteristics.
       - Results stored in ``recon`` dict (ensemble series, CV-tuned series).
    9. Generates three diagnostic plots saved as PNG:
       - **plot_comparison_grid.png** — Per-Method grid (all methods including
         simple fills and ensemble methods, for visual reference).
       - **plot_best_method.png** — Highlight Best Performers (top-5 overlay,
         metric bar chart, RMSE/RetCorr trade-off scatter).
       - **plot_improvements.png** — Section 6 Improvements (ensemble + CV
         tuning overlay and RMSE comparison).
    10. Writes interim results to ``<output_dir>/<target>/``:
        ``evaluation.csv``, ``reconstructions.csv``, ``sweep_results.csv``,
        ``drift_results.csv``, ``backtest_results.csv``, ``report.txt``,
        ``plot_comparison_grid.png``, ``plot_best_method.png``,
        ``plot_improvements.png``.
    11. Appends a single-row summary to ``consolidated_report.csv`` at the
        output root.

    ── Multi-target mode (comma-separated) ──────────────────────────
    ``--target MSFT,AAPL,GOOG``

    Runs the full single-target pipeline for each ticker in the list.
    A consolidated DataFrame with one row per target is saved to
    ``consolidated_report.csv``.

    ── All-sectors mode ─────────────────────────────────────────────
    ``--target ALL``

    Discovers one ticker per sector from the instrument catalog and runs
    the multi-target pipeline across all of them.  Use ``--sectors`` to
    restrict to specific sectors, e.g.:
    ``--target ALL --sectors Technology HealthCare``

    ── Output directory resolution ──────────────────────────────────
    1. If ``--output-dir`` is provided explicitly, it is used as the
       root output directory.
    2. Otherwise, the ``RESULT_PATH`` environment variable is used
       (default: ``./results`` relative to the working directory).
    3. Each target's interim files are placed in a subdirectory named
       after the ticker (lowercase).

    ── Key parameters ───────────────────────────────────────────────
    ``--lookback``     : training window size for peer regression.
    ``--threshold``    : dendrogram cut threshold for peer clustering.
    ``--max-peers``    : maximum peers in the regression model.
    ``--dist-depth``   : fractional price drop at the distress peak.
    ``--dist-length``  : number of trading days in the distress window.
    ``--n-days``       : total trading days to sample from history.
    ``--no-sweep``     : skip the ML Proxy grid-search for speed.
    """
    parser = argparse.ArgumentParser(
        description="Gap-Filling Pipeline — Distressed Time-Series Reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python gap_filling.py --target MSFT\n"
            "  python gap_filling.py --target ALL --no-sweep\n"
            "  python gap_filling.py --target AAPL,GOOG --lookback 120 --threshold 0.30\n"
            "  RESULT_PATH=/custom/path python gap_filling.py --target ACGL\n\n"
            "See the docstring in main() for full details."
        ),
    )
    parser.add_argument("--target", type=str, default="MSFT",
                        help="Target ticker (e.g. MSFT); comma-separated list (MSFT,AAPL); or 'ALL' for one per sector")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory containing instruments.csv and stock_data.csv")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override root output dir  (default: $RESULT_PATH env or ./results)")
    parser.add_argument("--n-days", type=int, default=DEFAULT_N_DAYS,
                        help=f"Number of trading days to sample from history  (default: {DEFAULT_N_DAYS})")
    parser.add_argument("--dist-depth", type=float, default=DEFAULT_DIST_DEPTH,
                        help=f"Fractional price drop at distress peak, e.g. 0.35 = 35%% drop  (default: {DEFAULT_DIST_DEPTH})")
    parser.add_argument("--dist-length", type=int, default=DEFAULT_DIST_LENGTH,
                        help=f"Number of days in the distress window  (default: {DEFAULT_DIST_LENGTH})")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK,
                        help=f"Training lookback window in trading days for peer regression  (default: {DEFAULT_LOOKBACK})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CLUSTER_THR,
                        help=f"Cluster threshold for dendrogram peer selection  (default: {DEFAULT_CLUSTER_THR})")
    parser.add_argument("--max-peers", type=int, default=DEFAULT_MAX_PEERS,
                        help=f"Maximum peers used in the OLS regression model  (default: {DEFAULT_MAX_PEERS})")
    parser.add_argument("--no-sweep", action="store_true",
                        help="Skip the ML Proxy parameter grid-search (saves time)")
    parser.add_argument("--config", type=str, default=None,
                        help=f"Path to custom config.json  (default: {CONFIG_PATH})")
    parser.add_argument("--sectors", type=str, nargs="+", default=None,
                        help="Restrict to specific sectors when --target=ALL, e.g. --sectors Technology HealthCare")
    args = parser.parse_args()

    # Load custom config if provided
    if args.config is not None:
        _override_config(args.config)

    run_sweep = not args.no_sweep
    target_str = args.target.upper()

    if target_str == "SECTOR":
        # Discover targets — one per sector
        instruments_df, _ = load_data(args.data_dir)
        targets = [(instruments_df[instruments_df["sector"] == s]["symbol"].iloc[0], s)
                   for s in sorted(instruments_df["sector"].unique())
                   if not args.sectors or s in args.sectors]
        run_multi_target(targets, data_dir=args.data_dir, n_days=args.n_days,
                         dist_depth=args.dist_depth, dist_length=args.dist_length,
                         lookback=args.lookback, cluster_thr=args.threshold,
                         max_peers=args.max_peers, run_sweep=run_sweep,
                         config_path=args.config)
    elif target_str == 'ALL':
        # Run across all instruments in the universe
        instruments_df, _ = load_data(args.data_dir)
        sector_map = dict(zip(instruments_df["symbol"].str.upper(), instruments_df["sector"]))
        targets = [(sym.upper(), sector_map.get(sym.upper(), "Unknown"))
                   for sym in instruments_df["symbol"]]
        run_multi_target(targets, data_dir=args.data_dir, n_days=args.n_days,
                         dist_depth=args.dist_depth, dist_length=args.dist_length,
                         lookback=args.lookback, cluster_thr=args.threshold,
                         max_peers=args.max_peers, run_sweep=run_sweep,
                         config_path=args.config)

    elif "," in target_str:
        # Comma-separated list of tickers
        tickers = [t.strip().upper() for t in target_str.split(",")]
        instruments_df, _ = load_data(args.data_dir)
        sector_map = dict(zip(instruments_df["symbol"].str.upper(), instruments_df["sector"]))
        targets = [(t, sector_map.get(t, "Unknown")) for t in tickers]
        run_multi_target(targets, data_dir=args.data_dir, n_days=args.n_days,
                         dist_depth=args.dist_depth, dist_length=args.dist_length,
                         lookback=args.lookback, cluster_thr=args.threshold,
                         max_peers=args.max_peers, run_sweep=run_sweep,
                         config_path=args.config)

    else:
        # Single target
        result = run_pipeline(
            target_ticker=target_str, data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_days=args.n_days, dist_depth=args.dist_depth,
            dist_length=args.dist_length, lookback=args.lookback,
            cluster_thr=args.threshold, max_peers=args.max_peers,
            run_sweep=run_sweep, config_path=args.config)
        # Also save single-target summary to consolidated
        single_df = pd.DataFrame([result])
        if args.output_dir:
            cons_path = Path(args.output_dir) / "consolidated_report.csv"
        else:
            cons_path = RESULT_PATH / "consolidated_report.csv"
        cons_path.parent.mkdir(parents=True, exist_ok=True)
        single_df.to_csv(cons_path, index=False)
        print(f"\n  Single-target summary -> {cons_path}")


if __name__ == "__main__":
    main()
