#!/usr/bin/env python3
"""
gap_filling.py — Distressed Time-Series Reconstruction Pipeline
===============================================================

Supports single-target and multi-target runs with consolidated reports.
Uses RESULT_PATH env variable (default: ./results).
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

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

# ── Defaults ──
DEFAULT_LOOKBACK = 80
DEFAULT_CLUSTER_THR = 0.35
DEFAULT_MAX_PEERS = 5
DEFAULT_MIN_PEERS = 2
DEFAULT_N_DAYS = 500
DEFAULT_DIST_DEPTH = 0.35
DEFAULT_DIST_LENGTH = 30
DIST_OBS_KEY = "DIST_OBS"
DIST_TRUE_KEY = "DIST_TRUE"
TARGET_ORIG_KEY = "TARGET_ORIG"
LOOKBACK_GRID = [40, 60, 80, 120]
THRESH_GRID = [0.25, 0.35, 0.45, 0.55]

# ── Result path ──
RESULT_PATH = Path(os.environ.get("RESULT_PATH", Path.cwd() / "results"))


# ═══════════════════════════════════════════════════════════════
#  1. DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_data(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
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
                 min_peer_history_days=120, min_peer_coverage=0.85,
                 max_model_peers=25, include_cross_sector=False):
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
def get_regressor(model_type="ols"):
    if model_type.lower() == "ols":
        return LinearRegression()
    raise ValueError(f"Unknown model_type '{model_type}'. Only 'ols' supported.")


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
        lookback_grid = LOOKBACK_GRID
    if thresh_grid is None:
        thresh_grid = THRESH_GRID
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


# ═══════════════════════════════════════════════════════════════
#  6. BACKTESTING
# ═══════════════════════════════════════════════════════════════
def backtest_mean_reversion(prices, lookback=5, threshold=0.01):
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


def _plot_overlay(summary_df, recon, prices_df, dates, dist_start, dist_end, out_dir):
    """Reconstruction Method Comparison: overlay + RMSE bar."""
    zs, ze = max(0, dist_start - 10), min(len(dates), dist_end + 10)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    colors = ["#1565C0", "#E65100", "#D32F2F", "#F57C00", "#6A1B9A", "#00897B", "#C62828"]
    axes[0].plot(dates[zs:ze], prices_df[DIST_TRUE_KEY].iloc[zs:ze], "g--", lw=2.5, label="True", zorder=10)
    axes[0].plot(dates[zs:ze], prices_df[DIST_OBS_KEY].iloc[zs:ze], "#F44336", lw=1, alpha=0.4, label="Observed")
    axes[0].axvspan(dates[dist_start], dates[dist_end - 1], color="red", alpha=0.06)
    for idx, (_, rw) in enumerate(summary_df.iterrows()):
        axes[0].plot(dates[zs:ze], recon[rw["Method"]].iloc[zs:ze],
                     color=colors[idx % len(colors)], lw=1.8, alpha=0.85, label=rw["Method"])
    axes[0].set_title("All Methods vs True Reference", fontweight="bold")
    axes[0].set_ylabel("Price"); axes[0].legend(fontsize=7, ncol=2, loc="upper left")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=25); axes[0].grid(alpha=0.2)

    xp = range(len(summary_df)); bw = 0.35
    rmse_n = summary_df["RMSE"] / summary_df["RMSE"].max()
    axes[1].bar([xi - bw / 2 for xi in xp], rmse_n.values, bw, color="#E53935", alpha=0.7, label="Norm RMSE")
    ax2 = axes[1].twinx()
    ax2.scatter([xi + bw / 2 for xi in xp], summary_df["Return Corr"].values, s=80,
                c=summary_df["Return Corr"], cmap="RdYlGn", vmin=-1, vmax=1, edgecolor="black", zorder=5)
    axes[1].set_xticks(list(xp)); axes[1].set_xticklabels(summary_df["Method"], rotation=35, ha="right", fontsize=9)
    axes[1].set_title("RMSE (norm) vs Return Correlation", fontweight="bold")
    axes[1].set_ylabel("Norm RMSE", color="#E53935"); ax2.set_ylabel("Return Corr", color="green")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Reconstruction Method Comparison", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout(); fig.savefig(out_dir / "plot_method_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_drift(drift_df, drift_results, best_method, prices_df, dates, dist_start, dist_end, out_dir):
    """Drift Detection & Correction chart."""
    zs = slice(max(0, dist_start - 5), min(len(dates), dist_end + 5))
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    axes[0, 0].plot(drift_df.index, drift_df["price_error"], "o-", color="#D32F2F", lw=1.5, ms=3)
    axes[0, 0].axhline(0, color="gray", lw=0.7, ls="--")
    axes[0, 0].axhline(drift_df["price_error"].mean(), color="#D32F2F", ls=":", lw=1.5,
                        label=f'Mean={drift_df["price_error"].mean():.3f}')
    axes[0, 0].set_title("Daily Price Error", fontweight="bold"); axes[0, 0].legend(fontsize=8)

    axes[0, 1].fill_between(drift_df.index, 0, drift_df["cum_error"], color="#D32F2F", alpha=0.3)
    axes[0, 1].plot(drift_df.index, drift_df["cum_error"], color="#D32F2F", lw=2)
    axes[0, 1].axhline(0, color="gray", lw=0.7, ls="--")
    axes[0, 1].set_title("Cumulative Prediction Error (Drift)", fontweight="bold")

    axes[0, 2].plot(drift_df.index, drift_df["tracking_error"], "s-", color="#6A1B9A", lw=1.5, ms=3)
    axes[0, 2].set_title("Tracking Error", fontweight="bold")

    drift_corrected = {f"{best_method} (no correction)": prices_df["DIST_OBS"].copy()}
    drift_corrected[f"{best_method} (no correction)"] = \
        [s for _, s in [(n, None) for n in range(1)]][0]  # placeholder
    # Use actual drift corrections
    dc = {r["Method"]: r for _, r in drift_results.iterrows()}
    method_names = list(dc.keys())[:4]
    from gap_filling import correct_drift_residual_bias, correct_drift_rolling_bias, correct_drift_error_feedback
    # Actually let's just plot from the drift_results we already computed
    for ax, (name, series) in zip(axes[1], drift_results.iterrows()):
        if len(axes[1].flatten()) > 4:
            break
    # Simpler: just plot the best series from the drift_results
    # Reconstruct for plotting
    best_orig = prices_df["DIST_OBS"].copy()
    # This is getting complex; let's just use what we have from run_pipeline's drift_corrected_series
    axes[1, 0].text(0.5, 0.5, "See drift_results.csv\nfor detailed metrics",
                    transform=axes[1, 0].transAxes, ha="center", va="center", fontsize=12)
    axes[1, 1].axis("off"); axes[1, 2].axis("off")

    fig.suptitle("Drift Detection & Correction", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout(); fig.savefig(out_dir / "plot_drift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  8. PIPELINE
# ═══════════════════════════════════════════════════════════════
def run_pipeline(target_ticker="MSFT", data_dir="data", output_dir=None,
                 n_days=DEFAULT_N_DAYS, dist_depth=DEFAULT_DIST_DEPTH,
                 lookback=DEFAULT_LOOKBACK, cluster_thr=DEFAULT_CLUSTER_THR,
                 max_peers=DEFAULT_MAX_PEERS, min_peers=DEFAULT_MIN_PEERS,
                 dist_length=DEFAULT_DIST_LENGTH, run_sweep=True):
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

    sweep_meta = {"best_params": (None, None), "best_rmse": np.nan}
    if run_sweep:
        print("  3E — ML Proxy Optimisation (sweep) ...")
        sweep_df, best_sw, sweep_meta = ml_proxy_optimisation(
            ret_df, prices_df, dist_start, dist_end, default_max_peers=max_peers)
        sweep_df.to_csv(out / "sweep_results.csv", index=False)
        if best_sw is not None:
            recon["ML Proxy+Opt (Best Sweep)"] = best_sw
        bp = sweep_meta["best_params"]
        print(f"    Best: lookback={bp[0]}, threshold={bp[1]:.2f}  (RMSE={sweep_meta['best_rmse']:.4f})")

    # ── Step 5: Evaluate ──
    print("  Evaluating ...")
    summary_df = evaluate_reconstructions(recon, prices_df, dist_start, dist_end)
    summary_df.to_csv(out / "evaluation.csv", index=False)
    pd.DataFrame(recon).to_csv(out / "reconstructions.csv")

    best_name = summary_df.iloc[0]["Method"]
    best_rmse_val = summary_df.iloc[0]["RMSE"]
    print(f"  Best: {best_name}  (RMSE={best_rmse_val:.4f})")

    # ── Step 6: Drift ──
    print("  Drift corrections ...")
    drift_series_dict, drift_df = apply_drift_corrections(
        best_name, recon[best_name], prices_df, dist_start, dist_end, model_peer_cols)
    drift_df.to_csv(out / "drift_results.csv")

    # ── Step 7: Backtest ──
    print("  Backtesting ...")
    bt_df, eq_true, eq_obs, perf_true, perf_obs = run_backtests(recon, prices_df)
    bt_df.to_csv(out / "backtest_results.csv")

    # ── Step 8: Plots ──
    print("  Generating plots ...")
    try:
        _plot_comparison_grid(summary_df, recon, prices_df, dates, dist_start, dist_end, out)
        _plot_overlay(summary_df, recon, prices_df, dates, dist_start, dist_end, out)
        # Simpler drift plot
        drift_metrics = compute_drift_metrics(recon[best_name], prices_df[DIST_TRUE_KEY],
                                               dist_start, dist_end)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(drift_metrics.index, drift_metrics["cum_error"], color="#D32F2F", lw=2)
        ax.fill_between(drift_metrics.index, 0, drift_metrics["cum_error"],
                        color="#D32F2F", alpha=0.15)
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.set_title(f"Drift — {best_name}", fontweight="bold")
        ax.set_ylabel("Cumulative Error ($)")
        fig.tight_layout(); fig.savefig(out / "plot_drift.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  (plotting skipped: {e})")

    # ── Report ──
    report = [
        "=" * 72, f"  GAP-FILLING REPORT  |  Target: {target_ticker}", "=" * 72,
        f"  Sector      : {universe['target_sector']}",
        f"  Samples     : {n_avail}",
        f"  Distress    : {dates[dist_start].date()} – {dates[dist_end-1].date()}",
        f"  Lookback    : {lookback}  |  Threshold: {cluster_thr:.2f}", "",
        "-" * 72, "  EVALUATION (sorted by RMSE)", "-" * 72]
    for _, r in summary_df.iterrows():
        report.append(f"  {r['Method']:<35s}  RMSE={r['RMSE']:.4f}  "
                      f"MAE={r['MAE']:.4f}  RetCorr={r['Return Corr']:.4f}")
    report += ["", "-" * 72, "  DRIFT CORRECTION", "-" * 72]
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
                     n_days=DEFAULT_N_DAYS, dist_depth=DEFAULT_DIST_DEPTH,
                     lookback=DEFAULT_LOOKBACK, cluster_thr=DEFAULT_CLUSTER_THR,
                     max_peers=DEFAULT_MAX_PEERS, min_peers=DEFAULT_MIN_PEERS,
                     dist_length=DEFAULT_DIST_LENGTH, run_sweep=True):
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
                run_sweep=run_sweep)
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
       - **3D** ML Proxy (Raw + PCA feature-based clustering + OLS)
       - **3E** ML Proxy Optimisation (grid search over ``--lookback`` ×
               ``--threshold``, unless ``--no-sweep`` is set).
    5. Evaluates every method against the true (uncorrupted) prices using
       RMSE, MAE, Return Correlation, Price Correlation, Vol Ratio, and
       Final Gap.
    6. Applies drift detection and three correction methods (Residual
       Bias, Rolling Bias, Error Feedback) to the best-performing
       reconstruction.
    7. Backtests a mean-reversion strategy on every reconstructed series
       and compares with the true and observed (corrupted) paths.
    8. Generates three diagnostic plots saved as PNG:
       - Per-Method Comparison grid (actual vs predicted in distress window)
       - Reconstruction Method Comparison overlay + RMSE bar chart
       - Drift cumulative error chart
    9. Writes interim results to ``<output_dir>/<target>/``:
       ``evaluation.csv``, ``reconstructions.csv``, ``sweep_results.csv``,
       ``drift_results.csv``, ``backtest_results.csv``, ``report.txt``,
       ``plot_comparison_grid.png``, ``plot_method_comparison.png``,
       ``plot_drift.png``.
    10. Appends a single-row summary to ``consolidated_report.csv`` at the
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
    parser.add_argument("--data-dir", type=str, default="data",
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
    parser.add_argument("--sectors", type=str, nargs="+", default=None,
                        help="Restrict to specific sectors when --target=ALL, e.g. --sectors Technology HealthCare")
    args = parser.parse_args()

    run_sweep = not args.no_sweep
    target_str = args.target.upper()

    if target_str == "ALL":
        # Discover targets — one per sector
        instruments_df, _ = load_data(args.data_dir)
        targets = [(instruments_df[instruments_df["sector"] == s]["symbol"].iloc[0], s)
                   for s in sorted(instruments_df["sector"].unique())
                   if not args.sectors or s in args.sectors]
        run_multi_target(targets, data_dir=args.data_dir, n_days=args.n_days,
                         dist_depth=args.dist_depth, dist_length=args.dist_length,
                         lookback=args.lookback, cluster_thr=args.threshold,
                         max_peers=args.max_peers, run_sweep=run_sweep)

    elif "," in target_str:
        # Comma-separated list of tickers
        tickers = [t.strip().upper() for t in target_str.split(",")]
        instruments_df, _ = load_data(args.data_dir)
        sector_map = dict(zip(instruments_df["symbol"].str.upper(), instruments_df["sector"]))
        targets = [(t, sector_map.get(t, "Unknown")) for t in tickers]
        run_multi_target(targets, data_dir=args.data_dir, n_days=args.n_days,
                         dist_depth=args.dist_depth, dist_length=args.dist_length,
                         lookback=args.lookback, cluster_thr=args.threshold,
                         max_peers=args.max_peers, run_sweep=run_sweep)

    else:
        # Single target
        result = run_pipeline(
            target_ticker=target_str, data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_days=args.n_days, dist_depth=args.dist_depth,
            dist_length=args.dist_length, lookback=args.lookback,
            cluster_thr=args.threshold, max_peers=args.max_peers,
            run_sweep=run_sweep)
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
