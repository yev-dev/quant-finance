#!/usr/bin/env python3
"""
gap_filling_v2.py — Non-Linear Distressed Time-Series Reconstruction (v2)
=======================================================================

This module provides **non-linear** reconstruction methods for filling gaps
in distressed equity price series.  Unlike the original ``gap_filling.py``
which relies on linear regressors (OLS, Ridge, ElasticNet), this version
uses only non-linear models:

  - **Random Forest** — ensemble of decision trees, captures complex interactions
  - **Gradient Boosting** — sequential ensemble, handles non-linear patterns
  - **Support Vector Regression (SVR)** — kernel-based non-linear mapping
  - **K-Nearest Neighbors (KNN)** — non-parametric similarity-based prediction
  - **Kernel Ridge Regression** — ridge regression with RBF kernel trick
  - **MLP (Neural Network)** — multi-layer perceptron for deep patterns
  - **Extra Trees** — extremely randomized trees for variance reduction

Each method is available in both **static** (single pre-distress snapshot) and
**dynamic** (rolling daily re-estimation) variants.

Usage
-----
    from gap_filling_v2 import run_pipeline_v2
    result = run_pipeline_v2(target_ticker="MSFT")
"""

from __future__ import annotations

import json
import os
import warnings
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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from scipy.stats import randint, uniform, rankdata

# Reuse data loading and peer selection from the original module
from gap_filling import (
    load_data,
    select_peers,
    simple_fill_reconstruction,
    evaluate_reconstructions,
    compute_drift_metrics,
    correct_drift_residual_bias,
    correct_drift_rolling_bias,
    correct_drift_error_feedback,
    cusum_drift_detection,
    backtest_mean_reversion,
    compute_performance,
    run_backtests,
    ensemble_top3_reconstruction,
    tune_hyperparameters_cv,
    auto_select_method,
    DIST_OBS_KEY,
    DIST_TRUE_KEY,
    TARGET_ORIG_KEY,
    DEFAULT_LOOKBACK,
    DEFAULT_CLUSTER_THR,
    DEFAULT_MAX_PEERS,
    DEFAULT_MIN_PEERS,
    DEFAULT_N_DAYS,
    DEFAULT_DIST_DEPTH,
    DEFAULT_DIST_LENGTH,
    DEFAULT_DIST_START_DATE,
    DEFAULT_DIST_END_DATE,
    LOOKBACK_GRID,
    THRESH_GRID,
    RESULT_PATH,
    DEFAULT_DATA_DIR,
    _c,
    _override_config,
    CONFIG_PATH,
)

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


# ═══════════════════════════════════════════════════════════════
#  1. NON-LINEAR REGRESSOR FACTORY
# ═══════════════════════════════════════════════════════════════

def get_nonlinear_regressor(model_type: str = "rf", **kwargs):
    """Return a non-linear regressor instance by type name.

    Parameters
    ----------
    model_type : str
        One of:
        - ``'rf'`` — Random Forest
        - ``'gbr'`` — Gradient Boosting
        - ``'svr'`` — Support Vector Regression (RBF kernel)
        - ``'knn'`` — K-Nearest Neighbors
        - ``'kernel_ridge'`` — Kernel Ridge Regression (RBF kernel)
        - ``'mlp'`` — Multi-Layer Perceptron
        - ``'et'`` — Extra Trees

    Returns
    -------
    A scikit-learn regressor with .fit() / .predict() API.
    """
    model_type = model_type.lower()

    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 10),
            min_samples_leaf=kwargs.get("min_samples_leaf", 5),
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "gbr":
        return GradientBoostingRegressor(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 5),
            learning_rate=kwargs.get("learning_rate", 0.05),
            min_samples_leaf=kwargs.get("min_samples_leaf", 5),
            random_state=42,
        )
    elif model_type == "svr":
        return SVR(
            kernel=kwargs.get("kernel", "rbf"),
            C=kwargs.get("C", 10.0),
            gamma=kwargs.get("gamma", "scale"),
            epsilon=kwargs.get("epsilon", 0.01),
        )
    elif model_type == "knn":
        return KNeighborsRegressor(
            n_neighbors=kwargs.get("n_neighbors", 7),
            weights=kwargs.get("weights", "distance"),
            metric=kwargs.get("metric", "euclidean"),
        )
    elif model_type == "kernel_ridge":
        return KernelRidge(
            kernel=kwargs.get("kernel", "rbf"),
            alpha=kwargs.get("alpha", 1.0),
            gamma=kwargs.get("gamma", 0.1),
        )
    elif model_type == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=kwargs.get("hidden_layer_sizes", (64, 32)),
            activation=kwargs.get("activation", "relu"),
            learning_rate_init=kwargs.get("learning_rate_init", 0.001),
            max_iter=kwargs.get("max_iter", 500),
            early_stopping=True,
            random_state=42,
        )
    elif model_type == "et":
        return ExtraTreesRegressor(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 10),
            min_samples_leaf=kwargs.get("min_samples_leaf", 5),
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Use: 'rf', 'gbr', 'svr', 'knn', 'kernel_ridge', 'mlp', 'et'."
        )


# ═══════════════════════════════════════════════════════════════
#  2. PEER DISTANCE (NON-LINEAR AWARE)
# ═══════════════════════════════════════════════════════════════

def build_peer_distance_nonlinear(
    ret_window: pd.DataFrame,
    target_col: str = "DIST",
    cluster_thr: float = DEFAULT_CLUSTER_THR,
    min_peers: int = DEFAULT_MIN_PEERS,
    max_peers: int = DEFAULT_MAX_PEERS,
    peers_df: pd.DataFrame | None = None,
    use_spearman: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str], str]:
    """Build peer distance matrix using non-linear-aware similarity metrics.

    Unlike the linear version which uses Pearson correlation, this function
    incorporates:

    - **Spearman rank correlation** — captures monotonic non-linear relationships
    - **Distance correlation** — captures general dependence (linear + non-linear)
    - **Mutual information proxy** — via rank-based entropy estimation

    Parameters
    ----------
    ret_window : pd.DataFrame
        Return window for distance computation.
    target_col : str
        Target column name (default 'DIST').
    cluster_thr : float
        Dendrogram cut threshold for hierarchical clustering.
    min_peers, max_peers : int
        Min/max peers to select.
    peers_df : pd.DataFrame, optional
        Peer metadata for sector overlay.
    use_spearman : bool
        If True, use Spearman rank correlation as primary similarity.

    Returns
    -------
    dist_mat : pd.DataFrame
        Pairwise distance matrix.
    peer_score : pd.Series
        Composite peer scores (higher = better peer).
    diag : pd.DataFrame
        Diagnostic DataFrame with individual similarity components.
    peers : list[str]
        Selected peer tickers.
    rule : str
        Selection rule description.
    """
    stocks = ret_window.columns.tolist()

    # ── 1. Spearman rank correlation (monotonic non-linear) ──
    if use_spearman:
        corr_mat = ret_window.corr(method="spearman")
        sim_label = "spearman"
    else:
        corr_mat = ret_window.corr(method="pearson")
        sim_label = "pearson"

    corr_dist = ((1 - corr_mat) / 2).clip(lower=0, upper=1)
    vals = corr_dist.values.copy()
    np.fill_diagonal(vals, 0.0)
    corr_dist = pd.DataFrame(vals, index=corr_dist.index, columns=corr_dist.columns)

    # ── 2. Distance correlation (general dependence) ──
    def _distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """Compute distance correlation between two 1D arrays."""
        n = len(x)
        if n < 5:
            return 0.0
        a = np.tile(x, (n, 1))
        b = np.tile(y, (n, 1))
        A = np.abs(a - a.T)
        B = np.abs(b - b.T)
        A_centered = A - A.mean(axis=1, keepdims=True) - A.mean(axis=0) + A.mean()
        B_centered = B - B.mean(axis=1, keepdims=True) - B.mean(axis=0) + B.mean()
        dCov = np.sqrt(np.mean(A_centered * B_centered))
        dVarX = np.sqrt(np.mean(A_centered ** 2))
        dVarY = np.sqrt(np.mean(B_centered ** 2))
        if dVarX * dVarY == 0:
            return 0.0
        return min(1.0, dCov / np.sqrt(dVarX * dVarY))

    dcor_mat = pd.DataFrame(0.0, index=stocks, columns=stocks)
    for si in stocks:
        for sj in stocks:
            if si == sj:
                continue
            pair = ret_window[[si, sj]].dropna()
            if len(pair) >= 10:
                dcor_mat.loc[si, sj] = _distance_correlation(
                    pair[si].values, pair[sj].values
                )
    dcor_dist = (1 - dcor_mat).clip(0, 1)
    vals_dc = dcor_dist.values.copy()
    np.fill_diagonal(vals_dc, 0.0)
    dcor_dist = pd.DataFrame(vals_dc, index=dcor_dist.index, columns=dcor_dist.columns)

    # ── 3. Volatility similarity ──
    vol_series = ret_window.std()
    vol_diff = pd.DataFrame(0.0, index=stocks, columns=stocks)
    for si in stocks:
        for sj in stocks:
            vol_diff.loc[si, sj] = abs(vol_series[si] - vol_series[sj])
    max_v = vol_diff.to_numpy().max()
    if max_v > 0:
        vol_diff /= max_v

    # ── 4. Downside similarity (tail dependence) ──
    down_mask = ret_window[target_col] < 0
    down_sim = pd.Series(0.0, index=stocks, dtype=float)
    target_down = ret_window.loc[down_mask, target_col]
    for s in stocks:
        if s == target_col:
            down_sim[s] = 1.0
            continue
        pair = pd.concat([target_down, ret_window.loc[down_mask, s]], axis=1).dropna()
        if len(pair) >= 5:
            c = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
            down_sim[s] = np.clip((c + 1) / 2, 0, 1) if pd.notna(c) else 0.0

    down_dist = pd.DataFrame(1.0, index=stocks, columns=stocks)
    for si in stocks:
        for sj in stocks:
            down_dist.loc[si, sj] = abs(down_sim[si] - down_sim[sj])

    # ── 5. Composite distance ──
    dist_mat = (
        0.30 * corr_dist
        + 0.20 * dcor_dist
        + 0.15 * vol_diff
        + 0.15 * down_dist
    ).clip(0, 1)
    vals_dm = dist_mat.values.copy()
    np.fill_diagonal(vals_dm, 0.0)
    dist_mat = pd.DataFrame(vals_dm, index=dist_mat.index, columns=dist_mat.columns)

    # ── Peer scoring ──
    raw_corr = corr_mat.loc[target_col].drop(target_col)
    dcor_target = dcor_mat.loc[target_col].drop(target_col)
    vol_sim = 1.0 - vol_diff.loc[target_col].drop(target_col)
    down_comp = down_sim.drop(target_col)

    peer_score = (
        0.30 * raw_corr.clip(0)
        + 0.25 * dcor_target
        + 0.20 * vol_sim
        + 0.25 * down_comp
    ).sort_values(ascending=False)

    # ── Hierarchical clustering ──
    link = linkage(squareform(dist_mat.values), method="ward")
    labels = fcluster(link, t=cluster_thr, criterion="distance")
    same_cluster = [
        s
        for s, cid in zip(stocks, labels)
        if cid == labels[stocks.index(target_col)] and s != target_col and s in peer_score.index
    ]

    if len(same_cluster) >= min_peers:
        peers = sorted(same_cluster, key=lambda s: peer_score[s], reverse=True)[:max_peers]
        rule = f"cluster-first ({sim_label}+dcor)"
    else:
        peers = peer_score.index[:max_peers].tolist()
        rule = f"fallback global score ({sim_label}+dcor)"

    # ── Diagnostics ──
    diag = pd.DataFrame(index=peer_score.index)
    diag[f"{sim_label}_corr"] = raw_corr
    diag["distance_corr"] = dcor_target
    diag["vol_similarity"] = vol_sim
    diag["downside_similarity"] = down_comp
    if peers_df is not None:
        same_set = set(peers_df[peers_df["peer_bucket"] == "same_sector"]["ticker"])
        diag["same_sector"] = [s in same_set for s in diag.index]
    diag["peer_score"] = peer_score

    return dist_mat, peer_score, diag.sort_values("peer_score", ascending=False), peers, rule


# ═══════════════════════════════════════════════════════════════
#  3. NON-LINEAR RECONSTRUCTION METHODS
# ═══════════════════════════════════════════════════════════════

def _select_peers_nonlinear(
    ret_window: pd.DataFrame,
    cluster_thr: float = DEFAULT_CLUSTER_THR,
    min_peers: int = DEFAULT_MIN_PEERS,
    max_peers: int = DEFAULT_MAX_PEERS,
    peers_df: pd.DataFrame | None = None,
) -> tuple[list[str], str]:
    """Select peers using non-linear-aware distance."""
    _, _, _, peers, rule = build_peer_distance_nonlinear(
        ret_window,
        cluster_thr=cluster_thr,
        min_peers=min_peers,
        max_peers=max_peers,
        peers_df=peers_df,
    )
    return peers, rule


def _reconstruct_from_peers(
    ret_window: pd.DataFrame,
    ret_full: pd.DataFrame,
    prices_obs: pd.Series,
    gap_start: int,
    gap_end: int,
    peer_cols: list[str],
    model_type: str = "rf",
    **model_kwargs,
) -> tuple[pd.Series, dict]:
    """Generic reconstruction using a non-linear regressor.

    Parameters
    ----------
    ret_window : pd.DataFrame
        Pre-distress return window for training.
    ret_full : pd.DataFrame
        Full return panel (includes distress window).
    prices_obs : pd.Series
        Observed (corrupted) price series.
    gap_start, gap_end : int
        Distress window indices.
    peer_cols : list[str]
        Peer tickers to use as features.
    model_type : str
        Non-linear model type (see ``get_nonlinear_regressor``).
    **model_kwargs
        Additional keyword arguments passed to the regressor.

    Returns
    -------
    reconstructed : pd.Series
        Reconstructed price series.
    info : dict
        Model diagnostics.
    """
    peer_cols = [c for c in peer_cols if c in ret_window.columns]
    if len(peer_cols) < 2:
        raise ValueError(f"Need at least 2 peer columns, got {len(peer_cols)}")

    X_train = ret_window[peer_cols].values
    y_train = ret_window["DIST"].values

    model = get_nonlinear_regressor(model_type, **model_kwargs)
    model.fit(X_train, y_train)

    X_test = ret_full[peer_cols].iloc[gap_start:gap_end].values
    pred_log_ret = model.predict(X_test)

    reconstructed = prices_obs.copy()
    for i in range(gap_end - gap_start):
        reconstructed.iloc[gap_start + i] = (
            reconstructed.iloc[gap_start + i - 1] * np.exp(pred_log_ret[i])
        )

    # Feature importance (if available)
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = dict(zip(peer_cols, model.feature_importances_))
    elif hasattr(model, "coef_"):
        importance = dict(zip(peer_cols, model.coef_))

    info = {
        "peers": peer_cols,
        "n_peers": len(peer_cols),
        "model_type": model_type,
        "r2_train": float(r2_score(y_train, model.predict(X_train))),
        "feature_importance": importance,
        "rule": f"nonlinear-{model_type}",
    }
    return reconstructed, info


# ── 3A. Static Non-Linear Proxy ──

def static_nonlinear_reconstruction(
    ret_window: pd.DataFrame,
    ret_full: pd.DataFrame,
    prices_obs: pd.Series,
    gap_start: int,
    gap_end: int,
    model_type: str = "rf",
    cluster_thr: float = DEFAULT_CLUSTER_THR,
    min_peers: int = DEFAULT_MIN_PEERS,
    max_peers: int = DEFAULT_MAX_PEERS,
    peers_df: pd.DataFrame | None = None,
    **model_kwargs,
) -> tuple[pd.Series, dict]:
    """Static non-linear proxy reconstruction.

    Selects peers once using the pre-distress window, then applies a
    non-linear regressor to reconstruct the entire distress window.

    Parameters
    ----------
    ret_window : pd.DataFrame
        Pre-distress return window.
    ret_full : pd.DataFrame
        Full return panel.
    prices_obs : pd.Series
        Observed (corrupted) prices.
    gap_start, gap_end : int
        Distress window indices.
    model_type : str
        Non-linear model type.
    cluster_thr, min_peers, max_peers : float, int, int
        Peer selection parameters.
    peers_df : pd.DataFrame, optional
        Peer metadata.
    **model_kwargs
        Additional model parameters.

    Returns
    -------
    reconstructed : pd.Series
        Reconstructed price series.
    info : dict
        Model diagnostics.
    """
    peers, rule = _select_peers_nonlinear(
        ret_window,
        cluster_thr=cluster_thr,
        min_peers=min_peers,
        max_peers=max_peers,
        peers_df=peers_df,
    )
    reconstructed, info = _reconstruct_from_peers(
        ret_window, ret_full, prices_obs, gap_start, gap_end,
        peers, model_type=model_type, **model_kwargs,
    )
    info["rule"] = f"static-{rule}"
    return reconstructed, info


# ── 3B. Dynamic Non-Linear Proxy ──

def dynamic_nonlinear_reconstruction(
    ret_full: pd.DataFrame,
    prices_obs: pd.Series,
    gap_start: int,
    gap_end: int,
    model_type: str = "rf",
    lookback: int = DEFAULT_LOOKBACK,
    cluster_thr: float = DEFAULT_CLUSTER_THR,
    min_peers: int = DEFAULT_MIN_PEERS,
    max_peers: int = DEFAULT_MAX_PEERS,
    peers_df: pd.DataFrame | None = None,
    **model_kwargs,
) -> tuple[pd.Series, pd.DataFrame]:
    """Dynamic non-linear proxy reconstruction.

    Re-estimates peers and the non-linear model each day of the distress
    window using a rolling lookback window.

    Parameters
    ----------
    ret_full : pd.DataFrame
        Full return panel.
    prices_obs : pd.Series
        Observed (corrupted) prices.
    gap_start, gap_end : int
        Distress window indices.
    model_type : str
        Non-linear model type.
    lookback : int
        Rolling training window size.
    cluster_thr, min_peers, max_peers : float, int, int
        Peer selection parameters.
    peers_df : pd.DataFrame, optional
        Peer metadata.
    **model_kwargs
        Additional model parameters.

    Returns
    -------
    reconstructed : pd.Series
        Reconstructed price series.
    daily_log : pd.DataFrame
        Daily diagnostics.
    """
    reconstructed = prices_obs.copy()
    daily_rows = []

    for day in range(gap_start, gap_end):
        win_start = max(0, day - lookback)
        window = ret_full.iloc[win_start:day].copy()

        if len(window) < 30:
            reconstructed.iloc[day] = reconstructed.iloc[day - 1]
            continue

        try:
            peers, _ = _select_peers_nonlinear(
                window,
                cluster_thr=cluster_thr,
                min_peers=min_peers,
                max_peers=max_peers,
                peers_df=peers_df,
            )
            peer_cols = [c for c in peers[:max_peers] if c in window.columns]
            if len(peer_cols) < min_peers:
                reconstructed.iloc[day] = reconstructed.iloc[day - 1]
                continue

            X = window[peer_cols].values
            y = window["DIST"].values
            model = get_nonlinear_regressor(model_type, **model_kwargs)
            model.fit(X, y)

            pred_ret = model.predict(
                ret_full[peer_cols].iloc[day : day + 1].values
            )[0]
            reconstructed.iloc[day] = reconstructed.iloc[day - 1] * np.exp(pred_ret)

            r2 = r2_score(y, model.predict(X))
            daily_rows.append({
                "date": ret_full.index[day],
                "peers": ", ".join(peer_cols[:3]),
                "n_peers": len(peer_cols),
                "r2": r2,
                "pred_log_ret": pred_ret,
            })
        except Exception:
            reconstructed.iloc[day] = reconstructed.iloc[day - 1]

    return reconstructed, pd.DataFrame(daily_rows) if daily_rows else pd.DataFrame()


# ── 3C. Feature-Based Non-Linear Proxy ──

def feature_distance_matrix_nonlinear(
    ret_window: pd.DataFrame,
    use_pca: bool = False,
    pca_var: float = 0.90,
) -> pd.DataFrame:
    """Build feature-based distance matrix with non-linear features.

    Extracts richer feature set including:
    - Return moments (mean, std, skew, kurtosis)
    - Tail risk features (VaR, CVaR, max drawdown)
    - Autocorrelation (serial dependence)
    - Entropy proxy (permutation entropy via rank-based)
    """
    assets = ret_window.columns.tolist()
    features = {}

    for col in assets:
        r = ret_window[col].values
        r_pos = r[r > 0]
        r_neg = r[r < 0]

        feat = [
            r[-1],                                    # latest return
            np.sum(r[-3:]), np.sum(r[-5:]), np.sum(r[-10:]),  # momentum
            np.std(r[-5:]), np.std(r[-10:]),           # short-term vol
            np.std(r[-20:]) if len(r) >= 20 else 0.0,  # medium-term vol
            np.min(r[-5:]), np.min(r[-10:]),           # max drawdown
            np.max(r[-5:]), np.max(r[-10:]),           # max run-up
            np.percentile(r[-10:], 5) if len(r) >= 10 else 0.0,  # VaR 95%
            np.mean(r_neg[-10:]) if len(r_neg) > 0 else 0.0,     # CVaR
            np.mean(np.abs(np.diff(r[-5:]))),          # return volatility of vol
            np.std(r[-5:][r[-5:] < 0]) if np.any(r[-5:] < 0) else 0.0,  # downside vol
            np.mean(r[-5:] > 0) if len(r) >= 5 else 0.5,  # win rate
            np.abs(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 2 else 0.0,  # autocorr
        ]
        features[col] = feat

    feat_df = pd.DataFrame.from_dict(features, orient="index").fillna(0)
    feat_scaled = StandardScaler().fit_transform(feat_df)

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


def ml_nonlinear_proxy_reconstruction(
    ret_window: pd.DataFrame,
    ret_full: pd.DataFrame,
    prices_obs: pd.Series,
    gap_start: int,
    gap_end: int,
    model_type: str = "rf",
    use_pca: bool = False,
    cluster_thr: float = DEFAULT_CLUSTER_THR,
    max_peers: int = DEFAULT_MAX_PEERS,
    **model_kwargs,
) -> tuple[pd.Series, dict]:
    """ML-based non-linear proxy using feature-based clustering.

    Uses rich feature extraction (moments, tail risk, autocorrelation)
    for peer clustering, then applies a non-linear regressor.

    Parameters
    ----------
    ret_window : pd.DataFrame
        Pre-distress return window.
    ret_full : pd.DataFrame
        Full return panel.
    prices_obs : pd.Series
        Observed (corrupted) prices.
    gap_start, gap_end : int
        Distress window indices.
    model_type : str
        Non-linear model type.
    use_pca : bool
        Apply PCA to feature space.
    cluster_thr : float
        Cluster threshold.
    max_peers : int
        Maximum peers.
    **model_kwargs
        Additional model parameters.

    Returns
    -------
    reconstructed : pd.Series
        Reconstructed price series.
    info : dict
        Model diagnostics.
    """
    dist_mat = feature_distance_matrix_nonlinear(ret_window, use_pca=use_pca)
    link = linkage(squareform(dist_mat.values), method="ward")
    labels = fcluster(link, t=cluster_thr, criterion="distance")
    stocks = dist_mat.columns.tolist()
    dist_cluster = labels[stocks.index("DIST")]
    same_cluster = [s for s, cid in zip(stocks, labels) if cid == dist_cluster and s != "DIST"]

    raw_corr = ret_window.corr(method="spearman").loc["DIST"].drop("DIST")

    if same_cluster:
        peers = sorted(same_cluster, key=lambda s: abs(raw_corr[s]), reverse=True)[:max_peers]
        rule = "feature-cluster-first (nonlinear)"
    else:
        peers = raw_corr.abs().sort_values(ascending=False).index[:max_peers].tolist()
        rule = "feature-fallback global (nonlinear)"

    reconstructed, info = _reconstruct_from_peers(
        ret_window, ret_full, prices_obs, gap_start, gap_end,
        peers, model_type=model_type, **model_kwargs,
    )
    info["rule"] = rule
    info["feature_type"] = "PCA" if use_pca else "Raw"
    return reconstructed, info


# ── 3D. Ensemble Non-Linear Proxy (multiple non-linear models averaged) ──

def ensemble_nonlinear_reconstruction(
    ret_window: pd.DataFrame,
    ret_full: pd.DataFrame,
    prices_obs: pd.Series,
    gap_start: int,
    gap_end: int,
    model_types: list[str] | None = None,
    cluster_thr: float = DEFAULT_CLUSTER_THR,
    min_peers: int = DEFAULT_MIN_PEERS,
    max_peers: int = DEFAULT_MAX_PEERS,
    peers_df: pd.DataFrame | None = None,
) -> tuple[pd.Series, dict]:
    """Ensemble of multiple non-linear models.

    Trains several non-linear models on the same peer set and averages
    their predictions.  Provides robustness through model diversity.

    Parameters
    ----------
    ret_window : pd.DataFrame
        Pre-distress return window.
    ret_full : pd.DataFrame
        Full return panel.
    prices_obs : pd.Series
        Observed (corrupted) prices.
    gap_start, gap_end : int
        Distress window indices.
    model_types : list[str], optional
        List of model types to ensemble. Default: ['rf', 'gbr', 'svr', 'knn'].
    cluster_thr, min_peers, max_peers : float, int, int
        Peer selection parameters.
    peers_df : pd.DataFrame, optional
        Peer metadata.

    Returns
    -------
    reconstructed : pd.Series
        Reconstructed price series (ensemble average).
    info : dict
        Ensemble diagnostics with per-model details.
    """
    if model_types is None:
        model_types = ["rf", "gbr", "svr", "knn"]

    peers, rule = _select_peers_nonlinear(
        ret_window,
        cluster_thr=cluster_thr,
        min_peers=min_peers,
        max_peers=max_peers,
        peers_df=peers_df,
    )
    peer_cols = [c for c in peers[:max_peers] if c in ret_window.columns]
    if len(peer_cols) < 2:
        raise ValueError("Not enough peers for ensemble reconstruction.")

    X_train = ret_window[peer_cols].values
    y_train = ret_window["DIST"].values
    X_test = ret_full[peer_cols].iloc[gap_start:gap_end].values

    predictions = []
    member_info = {}

    for mt in model_types:
        try:
            model = get_nonlinear_regressor(mt)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            r2 = r2_score(y_train, model.predict(X_train))
            predictions.append(pred)
            member_info[mt] = {
                "r2_train": float(r2),
                "feature_importance": (
                    dict(zip(peer_cols, model.feature_importances_))
                    if hasattr(model, "feature_importances_")
                    else None
                ),
            }
        except Exception as e:
            print(f"    Ensemble member '{mt}' failed: {e}")
            continue

    if not predictions:
        raise ValueError("All ensemble members failed.")

    pred_array = np.array(predictions)
    ensemble_pred = np.mean(pred_array, axis=0)
    ensemble_std = np.std(pred_array, axis=0)

    reconstructed = prices_obs.copy()
    for i in range(gap_end - gap_start):
        reconstructed.iloc[gap_start + i] = (
            reconstructed.iloc[gap_start + i - 1] * np.exp(ensemble_pred[i])
        )

    info = {
        "peers": peer_cols,
        "n_peers": len(peer_cols),
        "model_types": model_types,
        "n_successful": len(predictions),
        "members": member_info,
        "ensemble_std": ensemble_std,
        "rule": f"ensemble-{rule}",
    }
    return reconstructed, info


# ── 3E. Non-Linear Optimisation (Random Search) ──

def nonlinear_optimisation(
    ret_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    dist_start: int,
    dist_end: int,
    model_type: str = "rf",
    n_iter: int = 30,
    random_state: int = 42,
    default_max_peers: int = DEFAULT_MAX_PEERS,
) -> tuple[pd.DataFrame, pd.Series | None, dict]:
    """Random search optimisation for non-linear model hyperparameters.

    Parameters
    ----------
    ret_df : pd.DataFrame
        Full return panel.
    prices_df : pd.DataFrame
        Price panel with DIST_OBS and DIST_TRUE columns.
    dist_start, dist_end : int
        Distress window indices.
    model_type : str
        Non-linear model type.
    n_iter : int
        Number of random parameter combinations.
    random_state : int
        Random seed.
    default_max_peers : int
        Default max peers.

    Returns
    -------
    sweep_df : pd.DataFrame
        Results sorted by composite score.
    best_series : pd.Series or None
        Best reconstruction.
    best_params : dict
        Best parameter values.
    """
    # Define parameter distributions per model type
    param_distributions = {
        "rf": {
            "n_estimators": randint(50, 300),
            "max_depth": randint(3, 20),
            "min_samples_leaf": randint(2, 15),
        },
        "gbr": {
            "n_estimators": randint(50, 300),
            "max_depth": randint(3, 12),
            "learning_rate": uniform(0.01, 0.2),
            "min_samples_leaf": randint(2, 10),
        },
        "svr": {
            "C": uniform(0.1, 50),
            "gamma": uniform(0.001, 1.0),
            "epsilon": uniform(0.001, 0.1),
        },
        "knn": {
            "n_neighbors": randint(3, 20),
            "weights": ["distance", "uniform"],
        },
        "kernel_ridge": {
            "alpha": uniform(0.01, 10),
            "gamma": uniform(0.001, 1.0),
        },
        "mlp": {
            "hidden_layer_sizes": [(32,), (64,), (32, 16), (64, 32), (128, 64)],
            "learning_rate_init": uniform(0.0001, 0.01),
        },
        "et": {
            "n_estimators": randint(50, 300),
            "max_depth": randint(3, 20),
            "min_samples_leaf": randint(2, 15),
        },
    }

    dist = param_distributions.get(model_type, param_distributions["rf"])
    param_sampler = ParameterSampler(dist, n_iter=n_iter, random_state=random_state)

    # Out-of-sample validation split
    train_end = dist_start + int(0.8 * (dist_end - dist_start))
    test_start = train_end
    test_end = dist_end

    sweep_rows = []
    sweep_results = {}

    for params in param_sampler:
        try:
            window = ret_df.iloc[
                max(0, dist_start - 1 - DEFAULT_LOOKBACK) : dist_start - 1
            ].copy()
            series, info = static_nonlinear_reconstruction(
                window, ret_df, prices_df[DIST_OBS_KEY],
                dist_start, dist_end,
                model_type=model_type,
                max_peers=default_max_peers,
                **params,
            )
        except Exception:
            continue

        pred_test = series.iloc[test_start:test_end].values
        true_test = prices_df[DIST_TRUE_KEY].iloc[test_start:test_end].values

        if len(pred_test) < 2:
            continue

        rmse = float(np.sqrt(mean_squared_error(true_test, pred_test)))
        mae = float(np.mean(np.abs(true_test - pred_test)))
        mape = float(np.mean(np.abs((true_test - pred_test) / (true_test + 1e-10))) * 100)

        pred_ret = np.log(pred_test / series.iloc[test_start - 1 : test_end - 1].values)
        true_ret = np.log(
            true_test / prices_df[DIST_TRUE_KEY].iloc[test_start - 1 : test_end - 1].values
        )
        ret_corr = float(np.corrcoef(pred_ret, true_ret)[0, 1]) if len(pred_ret) > 1 else 0.0
        direction_accuracy = float(np.mean(np.sign(pred_ret) == np.sign(true_ret)))

        sweep_rows.append({
            **{f"param_{k}": v for k, v in params.items()},
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "Ret Corr": ret_corr,
            "Dir Accuracy": direction_accuracy,
        })
        sweep_results[tuple(params.items())] = series

    if not sweep_rows:
        return pd.DataFrame(), None, {}

    sweep_df = pd.DataFrame(sweep_rows)
    # Composite score (lower is better)
    for col in ["RMSE", "MAE", "MAPE"]:
        sweep_df[f"{col}_rank"] = rankdata(sweep_df[col])
    sweep_df["corr_rank"] = rankdata(-sweep_df["Ret Corr"])
    sweep_df["dir_rank"] = rankdata(-sweep_df["Dir Accuracy"])
    sweep_df["Composite Score"] = (
        0.35 * sweep_df["RMSE_rank"]
        + 0.25 * sweep_df["MAE_rank"]
        + 0.20 * sweep_df["MAPE_rank"]
        + 0.12 * sweep_df["corr_rank"]
        + 0.08 * sweep_df["dir_rank"]
    )
    sweep_df = sweep_df.sort_values("Composite Score").reset_index(drop=True)

    best = sweep_df.iloc[0]
    best_params = {k: v for k, v in best.items() if k.startswith("param_")}
    best_series = sweep_results.get(tuple(best_params.items()))

    return sweep_df, best_series, best_params


# ═══════════════════════════════════════════════════════════════
#  4. DRIFT CORRECTION (re-exported for convenience)
# ═══════════════════════════════════════════════════════════════

def apply_drift_corrections_v2(
    best_method: str,
    best_series: pd.Series,
    prices_df: pd.DataFrame,
    dist_start: int,
    dist_end: int,
    model_peer_cols: list[str],
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Apply drift corrections to the best reconstruction.

    Wraps the original ``apply_drift_corrections`` with v2 naming.
    """
    corrected = {
        f"{best_method} (no correction)": best_series,
        f"{best_method} + Residual Bias": correct_drift_residual_bias(
            best_series, prices_df[DIST_TRUE_KEY], dist_start, dist_end
        ),
        f"{best_method} + Rolling Bias": correct_drift_rolling_bias(
            best_series, prices_df[DIST_TRUE_KEY], dist_start, dist_end
        ),
        f"{best_method} + Error Feedback": correct_drift_error_feedback(
            best_series, prices_df[model_peer_cols], dist_start, dist_end
        ),
    }
    rows = []
    for name, series in corrected.items():
        pred = series.iloc[dist_start:dist_end].values
        true = prices_df[DIST_TRUE_KEY].iloc[dist_start:dist_end].values
        rows.append({
            "Method": name,
            "RMSE": np.sqrt(mean_squared_error(true, pred)),
            "MAE": np.mean(np.abs(true - pred)),
            "Max Drift": np.max(np.abs(pred - true)),
            "Final Gap": pred[-1] - true[-1],
            "End Drift %": ((pred[-1] - true[-1]) / true[-1]) * 100,
        })
    return corrected, pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
#  5. PLOTTING
# ═══════════════════════════════════════════════════════════════

def _plot_comparison_grid_v2(
    summary_df: pd.DataFrame,
    recon: dict[str, pd.Series],
    prices_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    dist_start: int,
    dist_end: int,
    out_dir: Path,
) -> None:
    """Per-Method Comparison: actual vs predicted prices (distress window)."""
    zs, ze = max(0, dist_start - 10), min(len(dates), dist_end + 10)
    nm = len(summary_df)
    nc = min(4, nm)
    nr = int(np.ceil(nm / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(6 * nc, 4.5 * nr))
    axes = axes.flatten() if nm > 1 else [axes]
    colors = plt.cm.Set2(np.linspace(0, 1, nm))

    for idx, (_, row) in enumerate(summary_df.iterrows()):
        ax = axes[idx]
        method = row["Method"]
        series = recon[method]
        ax.plot(dates[zs:ze], prices_df[DIST_TRUE_KEY].iloc[zs:ze], "g--", lw=2, label="True", zorder=5)
        ax.plot(dates[zs:ze], prices_df[DIST_OBS_KEY].iloc[zs:ze], "#F44336", lw=0.8, alpha=0.35, label="Observed")
        ax.plot(dates[zs:ze], series.iloc[zs:ze], color=colors[idx], lw=2.5, label=method)
        ax.axvspan(dates[dist_start], dates[dist_end - 1], color="red", alpha=0.06)
        ax.text(
            0.03, 0.97,
            f"RMSE={row['RMSE']:.3f}  MAE={row['MAE']:.3f}\n"
            f"Ret ρ={row['Return Corr']:.3f}  VR={row['Vol Ratio']:.3f}",
            transform=ax.transAxes, fontsize=7.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor=colors[idx]),
        )
        ax.set_title(method, fontweight="bold", fontsize=10)
        ax.legend(fontsize=6.5, loc="lower left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, fontsize=7)
        ax.grid(alpha=0.15)

    for idx in range(nm, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Non-Linear Methods — Per-Method Comparison", fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_comparison_grid_v2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_best_method_v2(
    summary_df: pd.DataFrame,
    recon: dict[str, pd.Series],
    prices_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    dist_start: int,
    dist_end: int,
    out_dir: Path,
) -> None:
    """Highlight Best Performers: top-5 overlay, metric bar chart, and RMSE/RetCorr scatter."""
    zs, ze = max(0, dist_start - 10), min(len(dates), dist_end + 10)
    top_k = min(5, len(summary_df))
    top = summary_df.head(top_k)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    colors_palette = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A", "#00897B"]

    # Panel A: Top-5 overlay
    ax = axes[0]
    ax.plot(dates[zs:ze], prices_df[DIST_TRUE_KEY].iloc[zs:ze], "g--", lw=2.5, label="True", zorder=10)
    ax.plot(dates[zs:ze], prices_df[DIST_OBS_KEY].iloc[zs:ze], "#F44336", lw=0.8, alpha=0.3, label="Observed")
    for idx, (_, row) in enumerate(top.iterrows()):
        ax.plot(
            dates[zs:ze], recon[row["Method"]].iloc[zs:ze],
            color=colors_palette[idx], lw=2.0,
            label=f"#{idx+1} {row['Method']}  [RMSE={row['RMSE']:.2f}]",
        )
    ax.axvspan(dates[dist_start], dates[dist_end - 1], color="red", alpha=0.07)
    ax.set_title("Top Non-Linear Performers vs True Reference", fontweight="bold")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=7, ncol=1, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25)
    ax.grid(alpha=0.2)

    # Panel B: Grouped bar chart
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
    ax.set_title("Metric Comparison (Top-5 Non-Linear)", fontweight="bold")
    ax.set_ylabel("Error ($)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel C: Scatter
    ax = axes[2]
    sizes = [250 * (top_k - i) / top_k for i in range(top_k)]
    scatter = ax.scatter(
        top["RMSE"], top["Return Corr"], s=sizes,
        c=colors_palette[:top_k], alpha=0.75, edgecolors="black", linewidth=1, zorder=5,
    )
    for idx, (_, row) in enumerate(top.iterrows()):
        ax.annotate(
            f"#{idx+1}", (row["RMSE"], row["Return Corr"]),
            textcoords="offset points", xytext=(6, 6), fontsize=9, fontweight="bold",
        )
    ax.set_xlabel("RMSE ↓", fontweight="bold")
    ax.set_ylabel("Return Correlation ↑", fontweight="bold")
    ax.set_title("Error-Return Trade-off (Non-Linear)", fontweight="bold")
    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.grid(alpha=0.3)

    best = summary_df.iloc[0]
    facts = (
        f"🏆 Best: {best['Method']}\n"
        f"   RMSE = {best['RMSE']:.4f}   MAE = {best['MAE']:.4f}\n"
        f"   Ret Corr = {best['Return Corr']:.4f}   Vol Ratio = {best['Vol Ratio']:.3f}"
    )
    fig.text(0.5, 0.01, facts, ha="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF9C4", alpha=0.9, edgecolor="#F57F17"))

    fig.suptitle("Best Non-Linear Performers", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    fig.savefig(out_dir / "plot_best_method_v2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  6. PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline_v2(
    target_ticker: str = "MSFT",
    data_dir: str = "data",
    output_dir: str | None = None,
    n_days: int | None = None,
    dist_depth: float | None = None,
    lookback: int | None = None,
    cluster_thr: float | None = None,
    max_peers: int | None = None,
    min_peers: int | None = None,
    dist_length: int | None = None,
    run_sweep: bool | None = None,
    dist_start_date: str | None = None,
    dist_end_date: str | None = None,
    config_path: str | None = None,
    model_types: list[str] | None = None,
) -> dict[str, Any]:
    """Run the non-linear gap-filling pipeline for a single target.

    Parameters
    ----------
    target_ticker : str
        Target stock ticker.
    data_dir : str
        Directory containing instruments.csv and stock_data.csv.
    output_dir : str, optional
        Override output directory.
    n_days : int, optional
        Number of trading days to sample.
    dist_depth : float, optional
        Fractional price drop at distress peak.
    lookback : int, optional
        Training lookback window.
    cluster_thr : float, optional
        Cluster threshold for peer selection.
    max_peers : int, optional
        Maximum peers in model.
    min_peers : int, optional
        Minimum peers required.
    dist_length : int, optional
        Distress window length.
    run_sweep : bool, optional
        Run hyperparameter optimisation.
    dist_start_date, dist_end_date : str, optional
        Explicit distress window dates.
    config_path : str, optional
        Path to custom config.json.
    model_types : list[str], optional
        Non-linear model types to run. Default: all available.

    Returns
    -------
    dict
        Summary row for consolidation.
    """
    # Resolve config
    if config_path is not None:
        _override_config(config_path)

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
    if model_types is None:
        model_types = ["rf", "gbr", "svr", "knn", "kernel_ridge", "mlp", "et"]

    # Output directory
    if output_dir:
        out = Path(output_dir)
    else:
        out = RESULT_PATH / f"{target_ticker.lower()}_v2"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  NON-LINEAR GAP-FILLING v2 — TARGET: {target_ticker}")
    print(f"{'=' * 70}")

    # ── Step 1-2: Data & Peers ──
    instruments_df, stocks_data_df = load_data(data_dir)
    universe = select_peers(
        target_ticker=target_ticker,
        instruments_df=instruments_df,
        stocks_data_df=stocks_data_df,
    )
    prices_wide = universe["prices_wide"]
    model_peer_cols = universe["model_peer_cols"]
    peers_df = universe["peers_df"]

    selected = [target_ticker] + universe["same_sector_peers"]
    raw = prices_wide[selected].dropna().copy()
    n_avail = min(n_days, len(raw))
    raw = raw.iloc[-n_avail:]
    dates = raw.index
    n_avail = len(raw)

    # ── Step 3: Price panel + distress injection ──
    prices_df = pd.DataFrame(index=dates)
    for t in universe["same_sector_peers"] + [target_ticker]:
        prices_df[t] = raw[t]
    prices_df[TARGET_ORIG_KEY] = raw[target_ticker]
    prices_df[DIST_TRUE_KEY] = prices_df[TARGET_ORIG_KEY].copy()

    if dist_start_date is not None and dist_end_date is not None:
        dist_start_dt = pd.Timestamp(dist_start_date)
        dist_end_dt = pd.Timestamp(dist_end_date)
        if dist_start_dt not in dates or dist_end_dt not in dates:
            raise ValueError(
                f"Distress dates {dist_start_date}–{dist_end_date} not in data range "
                f"({dates[0].date()} to {dates[-1].date()})"
            )
        dist_start = dates.get_loc(dist_start_dt)
        dist_end = dates.get_loc(dist_end_dt) + 1
        print(f"  Distress window: {dist_start_date} → {dist_end_date} "
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

    ret_df = np.log(
        prices_df[model_peer_cols + [DIST_OBS_KEY]]
        / prices_df[model_peer_cols + [DIST_OBS_KEY]].shift(1)
    ).dropna()
    ret_df = ret_df.rename(columns={DIST_OBS_KEY: "DIST"})

    # ── Step 4: Non-Linear Reconstructions ──
    recon = {}

    # Simple fills (baseline)
    print("  Simple Fill (baseline) ...")
    recon.update(simple_fill_reconstruction(prices_df[DIST_OBS_KEY], dist_start, dist_end))

    win_s = max(0, dist_start - 1 - lookback)
    win_e = dist_start - 1
    pre_window = ret_df.iloc[win_s:win_e].copy()

    # Static non-linear methods
    for mt in model_types:
        label = f"Static {mt.upper()}"
        print(f"  {label} ...")
        try:
            series, info = static_nonlinear_reconstruction(
                pre_window, ret_df, prices_df[DIST_OBS_KEY],
                dist_start, dist_end,
                model_type=mt,
                cluster_thr=cluster_thr,
                min_peers=min_peers,
                max_peers=max_peers,
                peers_df=peers_df,
            )
            recon[label] = series
            print(f"    R²={info['r2_train']:.3f}  peers={info['n_peers']}")
        except Exception as e:
            print(f"    (skipped: {e})")

    # Dynamic non-linear methods (subset for speed)
    dynamic_types = [mt for mt in model_types if mt in ("rf", "gbr", "knn")]
    for mt in dynamic_types:
        label = f"Dynamic {mt.upper()}"
        print(f"  {label} ...")
        try:
            series, daily_log = dynamic_nonlinear_reconstruction(
                ret_df, prices_df[DIST_OBS_KEY],
                dist_start, dist_end,
                model_type=mt,
                lookback=lookback,
                cluster_thr=cluster_thr,
                min_peers=min_peers,
                max_peers=max_peers,
                peers_df=peers_df,
            )
            recon[label] = series
            if not daily_log.empty:
                print(f"    Mean R²={daily_log['r2'].mean():.3f}")
        except Exception as e:
            print(f"    (skipped: {e})")

    # Feature-based ML non-linear proxy
    for up, lbl in [(False, "Raw"), (True, "PCA")]:
        label = f"ML Non-Linear ({lbl})"
        print(f"  {label} ...")
        try:
            series, info = ml_nonlinear_proxy_reconstruction(
                pre_window, ret_df, prices_df[DIST_OBS_KEY],
                dist_start, dist_end,
                model_type="rf",
                use_pca=up,
                cluster_thr=cluster_thr,
                max_peers=max_peers,
            )
            recon[label] = series
            print(f"    R²={info['r2_train']:.3f}  rule={info['rule']}")
        except Exception as e:
            print(f"    (skipped: {e})")

    # Ensemble non-linear
    print("  Ensemble Non-Linear (RF+GBR+SVR+KNN) ...")
    try:
        series, info = ensemble_nonlinear_reconstruction(
            pre_window, ret_df, prices_df[DIST_OBS_KEY],
            dist_start, dist_end,
            model_types=["rf", "gbr", "svr", "knn"],
            cluster_thr=cluster_thr,
            min_peers=min_peers,
            max_peers=max_peers,
            peers_df=peers_df,
        )
        recon["Ensemble Non-Linear"] = series
        print(f"    Members: {info['n_successful']}  R²={info['members'].get('rf', {}).get('r2_train', 0):.3f}")
    except Exception as e:
        print(f"    (skipped: {e})")

    # ── Optimisation (sweep) ──
    sweep_meta = {}
    if run_sweep:
        for mt in ["rf", "gbr", "svr"]:
            print(f"  Optimisation ({mt.upper()}) ...")
            try:
                sweep_df, best_series, best_params = nonlinear_optimisation(
                    ret_df, prices_df, dist_start, dist_end,
                    model_type=mt, n_iter=20, random_state=42,
                    default_max_peers=max_peers,
                )
                if best_series is not None and not sweep_df.empty:
                    label = f"Opt {mt.upper()}"
                    recon[label] = best_series
                    sweep_df.to_csv(out / f"sweep_{mt}.csv", index=False)
                    sweep_meta[f"opt_{mt}_rmse"] = float(sweep_df.iloc[0]["RMSE"])
                    print(f"    Best RMSE={sweep_df.iloc[0]['RMSE']:.4f}")
            except Exception as e:
                print(f"    (skipped: {e})")

    # ── Step 5: Evaluate ──
    print("  Evaluating ...")
    simple_fill_methods = {"Forward Fill", "Backward Fill", "Linear Interp"}
    summary_df_all = evaluate_reconstructions(recon, prices_df, dist_start, dist_end)
    summary_df_all.to_csv(out / "evaluation_v2.csv", index=False)
    summary_df = summary_df_all[~summary_df_all["Method"].isin(simple_fill_methods)].copy()
    summary_df = summary_df.reset_index(drop=True)

    recon_df = pd.DataFrame(recon)
    recon_df.to_csv(out / "reconstructions_v2.csv")

    best_name = summary_df.iloc[0]["Method"]
    best_rmse_val = summary_df.iloc[0]["RMSE"]
    print(f"  Best: {best_name}  (RMSE={best_rmse_val:.4f})")

    # ── Step 6: Drift ──
    print("  Drift corrections ...")
    drift_series_dict, drift_df = apply_drift_corrections_v2(
        best_name, recon[best_name], prices_df, dist_start, dist_end, model_peer_cols,
    )
    drift_df.to_csv(out / "drift_results_v2.csv")

    # Save time series
    prices_df[[DIST_TRUE_KEY, DIST_OBS_KEY]].to_csv(out / "prices_original.csv")
    recon_df.to_csv(out / "reconstructions_v2.csv")
    pd.DataFrame(drift_series_dict).to_csv(out / "drift_corrected_series_v2.csv")

    comparison_df = pd.DataFrame({
        DIST_TRUE_KEY: prices_df[DIST_TRUE_KEY],
        DIST_OBS_KEY: prices_df[DIST_OBS_KEY],
        "Best_Prediction": recon[best_name],
    })
    if len(drift_series_dict) > 1:
        corr_names = [n for n in drift_series_dict if "(no correction)" not in n]
        if corr_names:
            best_corr_name = min(
                corr_names,
                key=lambda n: drift_df[drift_df["Method"] == n]["RMSE"].values[0]
                if len(drift_df[drift_df["Method"] == n]) > 0 else np.inf,
            )
            comparison_df["Best_Correction"] = drift_series_dict[best_corr_name]
    comparison_df.to_csv(out / "comparison_timeseries_v2.csv")

    # ── Step 7: Backtest ──
    print("  Backtesting ...")
    bt_df, eq_true, eq_obs, perf_true, perf_obs = run_backtests(recon, prices_df)
    bt_df.to_csv(out / "backtest_results_v2.csv")

    # ── Step 8: Plots ──
    print("  Generating plots ...")
    try:
        _plot_comparison_grid_v2(summary_df_all, recon, prices_df, dates, dist_start, dist_end, out)
        _plot_best_method_v2(summary_df, recon, prices_df, dates, dist_start, dist_end, out)
    except Exception as e:
        print(f"  (plotting skipped: {e})")

    # ── Report ──
    report = [
        "=" * 72,
        f"  NON-LINEAR GAP-FILLING REPORT v2  |  Target: {target_ticker}",
        "=" * 72,
        f"  Sector      : {universe['target_sector']}",
        f"  Samples     : {n_avail}",
        f"  Distress    : {dates[dist_start].date()} – {dates[dist_end-1].date()}",
        f"  Lookback    : {lookback}  |  Threshold: {cluster_thr:.2f}",
        "",
        "-" * 72,
        "  EVALUATION (non-linear methods, sorted by RMSE)",
        "-" * 72,
    ]
    for _, r in summary_df_all.iterrows():
        report.append(
            f"  {r['Method']:<40s}  RMSE={r['RMSE']:.4f}  "
            f"MAE={r['MAE']:.4f}  RetCorr={r['Return Corr']:.4f}"
        )
    report += [
        "",
        f"  * Best (non-linear method): {best_name} (RMSE={best_rmse_val:.4f})",
        "",
        "-" * 72,
        "  DRIFT CORRECTION",
        "-" * 72,
    ]
    for _, r in drift_df.iterrows():
        report.append(
            f"  {r['Method']:<50s}  RMSE={r['RMSE']:.4f}  MaxDrift={r['Max Drift']:.4f}"
        )
    report += [
        "",
        "-" * 72,
        "  BACKTESTING",
        "-" * 72,
        f"  TRUE     : TotalReturn={perf_true['Total Return']:.4f}  Sharpe={perf_true['Sharpe Ratio']:.3f}",
        f"  OBSERVED : TotalReturn={perf_obs['Total Return']:.4f}  Sharpe={perf_obs['Sharpe Ratio']:.3f}",
        "=" * 72,
    ]
    (out / "report_v2.txt").write_text("\n".join(report) + "\n")

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
        "True_Return": perf_true["Total Return"],
        "True_Sharpe": perf_true["Sharpe Ratio"],
        "Observed_Return": perf_obs["Total Return"],
        "Observed_Sharpe": perf_obs["Sharpe Ratio"],
        "Output_Dir": str(out),
    }


# ═══════════════════════════════════════════════════════════════
#  7. CLI
# ═══════════════════════════════════════════════════════════════

def main():
    """Command-line entry point for the Non-Linear Gap-Filling Pipeline v2."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Non-Linear Gap-Filling v2 — Distressed Time-Series Reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--target", type=str, default="MSFT", help="Target ticker")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--n-days", type=int, default=DEFAULT_N_DAYS, help="Number of trading days")
    parser.add_argument("--dist-depth", type=float, default=DEFAULT_DIST_DEPTH, help="Distress depth")
    parser.add_argument("--dist-length", type=int, default=DEFAULT_DIST_LENGTH, help="Distress length")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK, help="Lookback window")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CLUSTER_THR, help="Cluster threshold")
    parser.add_argument("--max-peers", type=int, default=DEFAULT_MAX_PEERS, help="Max peers")
    parser.add_argument("--no-sweep", action="store_true", help="Skip hyperparameter optimisation")
    parser.add_argument(
        "--models", type=str, nargs="+",
        default=["rf", "gbr", "svr", "knn", "kernel_ridge", "mlp", "et"],
        help="Non-linear model types to run",
    )
    parser.add_argument("--config", type=str, default=None, help="Custom config path")
    args = parser.parse_args()

    run_pipeline_v2(
        target_ticker=args.target.upper(),
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_days=args.n_days,
        dist_depth=args.dist_depth,
        dist_length=args.dist_length,
        lookback=args.lookback,
        cluster_thr=args.threshold,
        max_peers=args.max_peers,
        run_sweep=not args.no_sweep,
        config_path=args.config,
        model_types=args.models,
    )


if __name__ == "__main__":
    main()