"""
distress_analysis.py — Distressed Stock Price Reconstruction Module

A comprehensive toolkit for:
  1. Loading distressed stock configurations
  2. Cached data downloads from Yahoo Finance
  3. Altman Z-Score computation and distress classification
  4. Market-based distress indicators (Merton DD, volatility regime, correlation breakdown)
  5. Composite market distress scoring with peer-based index auto-detection
  6. Price reconstruction using healthy-peer regression (CAPM + Sector)
  7. Risk metrics (Historical VaR, portfolio VaR with RWA impact)
  8. Visualization (price reconstruction, VaR comparison, Z-Score dashboard, sector R²)
  9. Cross-sector R² validation (empirical proof for peer-based reconstruction)
  10. Multi-ticker market distress scanning

All functions follow clean interfaces: data in → data out.
Cached downloads use Parquet format with a `reload` flag.
Visualization functions use a `show_graphs` flag (default True).
"""

import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from scipy import stats

# Suppress font glyph warnings (e.g. emoji in labels on systems without emoji fonts)
warnings.filterwarnings("ignore", message="Glyph.*missing from font")
warnings.filterwarnings("ignore", message="findfont: Font family.*not found")
# Suppress yfinance FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


try:
    from scipy.optimize import brentq
    from scipy.stats import norm
    _HAS_SCIPY_FULL = True
except ImportError:
    _HAS_SCIPY_FULL = False

try:
    from sklearn.linear_model import LinearRegression
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION & HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_ZONE = {"distress": 0, "grey": 1, "safe": 2}
ZONE_COLORS = {"Distress": "#e74c3c", "Grey": "#f39c12", "Safe": "#27ae60", "Unknown": "#95a5a6"}

DISTRESS_THRESHOLD = 1.81
GREY_THRESHOLD = 2.99

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "distressed_stocks.json"
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "distressed_data"


def load_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    """Load distressed stocks configuration from JSON.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to JSON config file. Defaults to distressed_stocks.json
        in the same directory as this module.

    Returns
    -------
    dict with keys: stocks, sector_universes, global_settings
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def get_stock_config(ticker: str, config: Optional[dict] = None) -> dict:
    """Get configuration for a single distressed stock.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g., 'RIG', 'ATO.PA').
    config : dict, optional
        Pre-loaded config. If None, loads from default path.

    Returns
    -------
    dict with keys: name, sector, peers, market_index, distress_start,
                   distress_end, known_event, event_type
    """
    if config is None:
        config = load_config()
    stock = config["stocks"].get(ticker)
    if stock is None:
        raise KeyError(f"Ticker '{ticker}' not found in config. Available: {list(config['stocks'].keys())}")
    return stock


def list_distressed_tickers(config: Optional[dict] = None) -> List[str]:
    """Return a list of all distressed tickers in the config."""
    if config is None:
        config = load_config()
    return list(config["stocks"].keys())


# ══════════════════════════════════════════════════════════════════════════════
# 1. CACHED DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def _ticker_key(tickers: List[str]) -> str:
    """Generate a stable filesystem-safe key from a ticker list."""
    return "_".join(sorted(t.upper().replace(".", "_").replace("^", "IDX_") for t in tickers))


def cached_download(
    tickers: List[str],
    start: str,
    end: str,
    *,
    reload: bool = False,
    data_dir: Optional[Union[str, Path]] = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Download adjusted close prices for tickers, caching results to disk.

    Checks `distressed_data/{key}_{start}_{end}.parquet` first.
    If missing or reload=True, downloads from Yahoo Finance and caches.

    Parameters
    ----------
    tickers : list of str
        Yahoo Finance ticker symbols.
    start : str
        Start date 'YYYY-MM-DD'.
    end : str
        End date 'YYYY-MM-DD'.
    reload : bool
        If True, forces re-download even if cached file exists.
    data_dir : str or Path, optional
        Cache directory. Defaults to distressed_data/ next to this module.
    progress : bool
        Show yfinance progress bar.

    Returns
    -------
    pd.DataFrame of adjusted close prices, columns = tickers.
    """
    data_path = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    data_path.mkdir(parents=True, exist_ok=True)

    cache_file = data_path / f"{_ticker_key(tickers)}_{start}_{end}.parquet"

    if cache_file.exists() and not reload:
        df = pd.read_parquet(cache_file)
        return df

    # Download
    try:
        data = yf.download(tickers, start=start, end=end, progress=progress, auto_adjust=False)
    except Exception as e:
        err_msg = str(e)[:80]
        logger.warning(f"Download failed for {tickers}: {err_msg}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=tickers)

    if data.empty:
        logger.warning(f"Empty data for {tickers}")
        return pd.DataFrame(columns=tickers)

    # Extract close prices
    try:
        if len(tickers) == 1:
            prices = data[["Close"]].copy()
            prices.columns = tickers
        elif isinstance(data.columns, pd.MultiIndex) and "Adj Close" in data.columns.levels[0]:
            prices = data["Adj Close"].copy()
        elif isinstance(data.columns, pd.MultiIndex) and "Close" in data.columns.levels[0]:
            prices = data["Close"].copy()
        else:
            # Single-level columns — return as-is
            prices = data.copy()
    except (KeyError, IndexError) as e:
        logger.warning(f"Cannot extract prices for {tickers}: {e}")
        return pd.DataFrame(columns=tickers)

    prices = prices.ffill()
    prices.to_parquet(cache_file)
    return prices


def download_price_history(
    ticker: str,
    start: str,
    end: str,
    *,
    reload: bool = False,
    data_dir: Optional[Union[str, Path]] = None,
) -> pd.Series:
    """Download price history for a single ticker (returns Series)."""
    df = cached_download([ticker], start=start, end=end, reload=reload, data_dir=data_dir, progress=False)
    if ticker in df.columns:
        return df[ticker].dropna()
    return pd.Series(dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# 2. ALTMAN Z-SCORE (Fundamental-Based Distress Detection)
# ══════════════════════════════════════════════════════════════════════════════

def _safe_get(series: pd.Series, *keys: str) -> float:
    """Safely extract a value from a pandas Series, trying multiple key names."""
    for key in keys:
        try:
            val = series.get(key)
            if val is not None and not (isinstance(val, (float, np.floating)) and np.isnan(val)):
                return float(val)
        except (KeyError, TypeError, ValueError):
            continue
    return np.nan


def compute_altman_z_score(
    working_capital: float,
    total_assets: float,
    retained_earnings: float,
    ebit: float,
    market_cap: float,
    total_liabilities: float,
    revenue: float,
) -> float:
    """Compute Altman Z-Score from financial statement components.

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    """
    if total_assets <= 0:
        return np.nan
    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_cap / total_liabilities if total_liabilities > 0 else np.nan
    x5 = revenue / total_assets
    return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5


def classify_z_score(z: float) -> str:
    """Classify Z-Score: 'Safe' (>2.99), 'Grey' (1.81-2.99), 'Distress' (<1.81)."""
    if np.isnan(z):
        return "Unknown"
    if z > 2.99:
        return "Safe"
    elif z >= 1.81:
        return "Grey"
    else:
        return "Distress"


ZONE_COLORS = {"Safe": "#2ecc71", "Grey": "#f39c12", "Distress": "#e74c3c", "Unknown": "#95a5a6"}
DISTRESS_THRESHOLD = 1.81
GREY_THRESHOLD = 2.99


def extract_financials_from_yahoo(ticker: yf.Ticker) -> Dict[str, float]:
    """Extract Altman Z-Score components from Yahoo Finance balance sheet & income statement."""
    result = {
        "working_capital": np.nan, "total_assets": np.nan, "retained_earnings": np.nan,
        "ebit": np.nan, "market_cap": np.nan, "total_liabilities": np.nan, "revenue": np.nan,
    }
    try:
        bs = ticker.balance_sheet
        if bs is not None and not bs.empty:
            latest = bs.iloc[:, 0]
            result["total_assets"] = _safe_get(latest, "Total Assets", "Total assets", "TotalAssets", "totalAssets")
            result["total_liabilities"] = _safe_get(latest,
                "Total Liabilities Net Minority Interest", "Total Liabilities", "Total liabilities",
                "TotalDebt", "Total Debt", "Long Term Debt And Capital Lease Obligation")
            if np.isnan(result["total_liabilities"]):
                std = _safe_get(latest, "Total Debt", "TotalDebt", "Short Term Debt")
                ltd = _safe_get(latest, "Long Term Debt", "LongTermDebt", "Long Term Debt And Capital Lease Obligation")
                if not np.isnan(std) or not np.isnan(ltd):
                    result["total_liabilities"] = (0 if np.isnan(std) else std) + (0 if np.isnan(ltd) else ltd)
            current_assets = _safe_get(latest, "Current Assets", "Current assets", "CurrentAssets", "currentAssets")
            current_liabilities = _safe_get(latest,
                "Current Liabilities", "Current liabilities", "CurrentLiabilities", "currentLiabilities")
            if not np.isnan(current_assets) and not np.isnan(current_liabilities):
                result["working_capital"] = current_assets - current_liabilities
            result["retained_earnings"] = _safe_get(latest,
                "Retained Earnings", "Retained earnings", "RetainedEarnings", "retainedEarnings",
                "Stockholders Equity", "StockholdersEquity", "Total Equity Gross Minority Interest")
    except Exception:
        pass
    try:
        income = ticker.financials
        if income is not None and not income.empty:
            latest_inc = income.iloc[:, 0]
            result["revenue"] = _safe_get(latest_inc,
                "Total Revenue", "Total revenue", "Revenue", "revenue",
                "Operating Revenue", "Operating revenue", "Sales", "Net Sales")
            result["ebit"] = _safe_get(latest_inc,
                "EBIT", "ebit", "Earnings Before Interest and Taxes",
                "Operating Income", "Operating income", "OperatingIncome", "operatingIncome",
                "Operating Income (Loss)", "Operating Profit")
            if np.isnan(result["ebit"]):
                gp = _safe_get(latest_inc, "Gross Profit", "GrossProfit")
                opex = _safe_get(latest_inc, "Operating Expense", "OperatingExpense",
                                 "Total Operating Expenses", "Total Expenses")
                if not np.isnan(gp) and not np.isnan(opex):
                    result["ebit"] = gp - opex
    except Exception:
        pass
    try:
        info = ticker.info
        if info:
            result["market_cap"] = info.get("marketCap", np.nan)
            if np.isnan(result["market_cap"]) or result["market_cap"] == 0:
                prev_close = info.get("previousClose", None)
                shares = info.get("sharesOutstanding", None)
                if prev_close and shares:
                    result["market_cap"] = prev_close * shares
    except Exception:
        pass
    return result


def compute_z_score_timeseries(ticker_str: str) -> pd.DataFrame:
    """Compute Altman Z-Score time-series from quarterly filings."""
    ticker = yf.Ticker(ticker_str)
    try:
        bs_q = ticker.quarterly_balance_sheet
        inc_q = ticker.quarterly_financials
    except Exception as e:
        print(f"  ⚠️  Quarterly data unavailable for {ticker_str}: {e}")
        return pd.DataFrame()
    if bs_q is None or bs_q.empty:
        return pd.DataFrame()
    try:
        shares = ticker.info.get("sharesOutstanding", None)
    except Exception:
        shares = None

    records = []
    for col in bs_q.columns:
        date = col.date() if hasattr(col, "date") else col
        bs_row = bs_q[col]
        total_assets = _safe_get(bs_row, "Total Assets", "Total assets", "TotalAssets", "totalAssets")
        total_liabilities = _safe_get(bs_row,
            "Total Liabilities Net Minority Interest", "Total Liabilities", "Total liabilities", "TotalDebt", "Total Debt")
        if np.isnan(total_liabilities):
            std = _safe_get(bs_row, "Total Debt", "TotalDebt", "Short Term Debt")
            ltd = _safe_get(bs_row, "Long Term Debt", "LongTermDebt")
            if not np.isnan(std) or not np.isnan(ltd):
                total_liabilities = (0 if np.isnan(std) else std) + (0 if np.isnan(ltd) else ltd)
        current_assets = _safe_get(bs_row, "Current Assets", "Current assets", "CurrentAssets", "currentAssets")
        current_liabs = _safe_get(bs_row, "Current Liabilities", "Current liabilities", "CurrentLiabilities", "currentLiabilities")
        wc = (current_assets - current_liabs) if not np.isnan(current_assets) and not np.isnan(current_liabs) else np.nan
        retained = _safe_get(bs_row,
            "Retained Earnings", "Retained earnings", "RetainedEarnings", "retainedEarnings",
            "Stockholders Equity", "StockholdersEquity", "Total Equity Gross Minority Interest")
        ebit = np.nan
        revenue = np.nan
        if inc_q is not None and col in inc_q.columns:
            inc_row = inc_q[col]
            ebit = _safe_get(inc_row, "EBIT", "ebit", "Earnings Before Interest and Taxes",
                             "Operating Income", "Operating income", "OperatingIncome", "operatingIncome",
                             "Operating Income (Loss)", "Operating Profit")
            if np.isnan(ebit):
                gp = _safe_get(inc_row, "Gross Profit", "GrossProfit")
                opex = _safe_get(inc_row, "Operating Expense", "OperatingExpense", "Total Operating Expenses", "Total Expenses")
                if not np.isnan(gp) and not np.isnan(opex):
                    ebit = gp - opex
            revenue = _safe_get(inc_row,
                "Total Revenue", "Total revenue", "Revenue", "revenue",
                "Operating Revenue", "Operating revenue", "Sales", "Net Sales")
        mkt_cap = np.nan
        try:
            hist = ticker.history(start=date - timedelta(days=7), end=date + timedelta(days=7))
            if not hist.empty and "Close" in hist.columns and shares:
                mkt_cap = hist["Close"].dropna().iloc[-1] * shares
        except Exception:
            pass
        z = compute_altman_z_score(working_capital=wc, total_assets=total_assets,
                                    retained_earnings=retained, ebit=ebit,
                                    market_cap=mkt_cap, total_liabilities=total_liabilities, revenue=revenue)
        records.append({
            "date": pd.Timestamp(date), "z_score": z, "zone": classify_z_score(z),
            "x1_wc_ta": wc / total_assets if total_assets > 0 else np.nan,
            "x2_re_ta": retained / total_assets if total_assets > 0 else np.nan,
            "x3_ebit_ta": ebit / total_assets if total_assets > 0 else np.nan,
            "x4_mve_tl": mkt_cap / total_liabilities if total_liabilities > 0 else np.nan,
            "x5_sales_ta": revenue / total_assets if total_assets > 0 else np.nan,
            "market_cap": mkt_cap,
        })
    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. MARKET-BASED DISTRESS INDICATORS (No Fundamentals Required)
# ══════════════════════════════════════════════════════════════════════════════

def compute_merton_dd(
    equity_value: float,
    equity_vol: float,
    debt_face_value: float,
    risk_free_rate: float = 0.03,
    time_to_maturity: float = 1.0,
    max_iter: int = 100,
) -> dict:
    """Compute Merton Distance-to-Default (DD) and implied Probability of Default."""
    if equity_value <= 0 or equity_vol <= 0 or debt_face_value <= 0:
        return {"DD": np.nan, "PD": np.nan, "asset_value": np.nan, "asset_vol": np.nan}

    def _merton_eq_err(asset_vol):
        V_guess = equity_value + debt_face_value * np.exp(-risk_free_rate * time_to_maturity)
        def _eq_err(V):
            d1 = (np.log(V / debt_face_value) + (risk_free_rate + 0.5 * asset_vol**2) * time_to_maturity) / (asset_vol * np.sqrt(time_to_maturity))
            d2 = d1 - asset_vol * np.sqrt(time_to_maturity)
            return V * scipy_norm.cdf(d1) - debt_face_value * np.exp(-risk_free_rate * time_to_maturity) * scipy_norm.cdf(d2) - equity_value
        try:
            V_sol = brentq(_eq_err, equity_value * 0.5, equity_value * 5.0, maxiter=max_iter)
        except (ValueError, RuntimeError):
            return np.inf
        d1 = (np.log(V_sol / debt_face_value) + (risk_free_rate + 0.5 * asset_vol**2) * time_to_maturity) / (asset_vol * np.sqrt(time_to_maturity))
        return (equity_vol * equity_value) / (V_sol * scipy_norm.cdf(d1)) - asset_vol

    try:
        sigma_V = brentq(_merton_eq_err, equity_vol * 0.1, equity_vol * 3.0, maxiter=max_iter)
    except (ValueError, RuntimeError):
        sigma_V = equity_vol * equity_value / (equity_value + debt_face_value * np.exp(-risk_free_rate * time_to_maturity))

    def _eq_err_V(V):
        d1 = (np.log(V / debt_face_value) + (risk_free_rate + 0.5 * sigma_V**2) * time_to_maturity) / (sigma_V * np.sqrt(time_to_maturity))
        d2 = d1 - sigma_V * np.sqrt(time_to_maturity)
        return V * scipy_norm.cdf(d1) - debt_face_value * np.exp(-risk_free_rate * time_to_maturity) * scipy_norm.cdf(d2) - equity_value

    try:
        V = brentq(_eq_err_V, equity_value * 0.5, equity_value * 5.0, maxiter=max_iter)
    except (ValueError, RuntimeError):
        V = equity_value + debt_face_value * np.exp(-risk_free_rate * time_to_maturity)

    dd = (np.log(V / debt_face_value) + (risk_free_rate - 0.5 * sigma_V**2) * time_to_maturity) / (sigma_V * np.sqrt(time_to_maturity))
    return {"DD": dd, "PD": scipy_norm.cdf(-dd), "asset_value": V, "asset_vol": sigma_V}


def detect_volatility_regime(
    prices: pd.Series, short_window: int = 20, long_window: int = 252, vol_spike_multiple: float = 2.0,
) -> pd.DataFrame:
    """Detect volatility regime: short-term vol vs long-term baseline."""
    log_ret = np.log(prices / prices.shift(1))
    vol_short = log_ret.rolling(short_window).std() * np.sqrt(252)
    vol_long = log_ret.rolling(long_window).std() * np.sqrt(252)
    return pd.DataFrame({
        "vol_short_ann": vol_short, "vol_long_ann": vol_long,
        "vol_ratio": vol_short / vol_long,
        "distress_signal": vol_short / vol_long >= vol_spike_multiple,
    }, index=prices.index)


def detect_correlation_breakdown(
    target_prices: pd.Series, peer_prices: pd.DataFrame,
    window: int = 60, breakdown_threshold: float = 0.3,
) -> pd.DataFrame:
    """Detect distress as drop in correlation with sector peers."""
    target_ret = np.log(target_prices / target_prices.shift(1))
    peer_ret = np.log(peer_prices / peer_prices.shift(1))
    peer_mean_ret = peer_ret.mean(axis=1)
    rolling_corr = target_ret.rolling(window).corr(peer_mean_ret)
    if peer_ret.shape[1] >= 2:
        pairwise = []
        for i in range(len(peer_ret.columns)):
            for j in range(i + 1, len(peer_ret.columns)):
                pairwise.append(peer_ret.iloc[:, i].rolling(window).corr(peer_ret.iloc[:, j]))
        avg_peer_corr = pd.concat(pairwise, axis=1).mean(axis=1)
    else:
        avg_peer_corr = pd.Series(1.0, index=target_ret.index)
    return pd.DataFrame({
        "corr_with_peers": rolling_corr, "avg_peer_corr": avg_peer_corr,
        "corr_breakdown": rolling_corr < breakdown_threshold,
        "corr_differential": rolling_corr - avg_peer_corr,
    }, index=target_prices.index)


def detect_tail_risk_events(
    prices: pd.Series, var_window: int = 252, var_confidence: float = 0.99,
    breach_lookback: int = 20, breach_threshold: int = 5,
) -> pd.DataFrame:
    """Detect distress from VaR breach clustering."""
    log_ret = np.log(prices / prices.shift(1))
    rolling_var = log_ret.rolling(var_window).quantile(1 - var_confidence)
    breach = log_ret < rolling_var
    rolling_breaches = breach.rolling(breach_lookback).sum()
    def _max_dd_30d(x):
        return (x.iloc[-1] / x.max() - 1) if x.max() > 0 else 0
    return pd.DataFrame({
        "log_return": log_ret, "VaR": rolling_var, "breach": breach,
        "breach_count": rolling_breaches,
        "tail_distress": rolling_breaches >= breach_threshold,
        "max_drawdown_30d": prices.rolling(30).apply(_max_dd_30d),
    }, index=prices.index)


def detect_distress_periods(
    z_score_df: pd.DataFrame, distress_threshold: float = 1.81, min_distress_quarters: int = 1,
) -> pd.DataFrame:
    """Identify contiguous distress periods in a Z-Score time-series."""
    df = z_score_df.copy()
    df["is_distressed"] = df["z_score"] < distress_threshold
    episode_id = 0
    in_episode = False
    episode_ids = []
    for distressed in df["is_distressed"]:
        if distressed and not in_episode:
            episode_id += 1
            in_episode = True
        elif not distressed:
            in_episode = False
        episode_ids.append(episode_id if distressed else 0)
    df["distress_episode_id"] = episode_ids
    for eid in range(1, episode_id + 1):
        mask = df["distress_episode_id"] == eid
        if mask.sum() < min_distress_quarters:
            df.loc[mask, "distress_episode_id"] = 0
            df.loc[mask, "is_distressed"] = False
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPOSITE MARKET DISTRESS SCORE
# ══════════════════════════════════════════════════════════════════════════════

_EXCHANGE_TO_INDEX = {
    ".PA": "^FCHI", ".DE": "^GDAXI", ".L": "^FTSE", ".MI": "FTSEMIB.MI",
    ".MC": "^SMSI", ".SW": "^SSMI", ".AS": "^AEX", ".T": "^N225",
    ".KS": "^KS11", ".TW": "^TWII", ".NS": "^NSEI", ".AX": "^AXJO",
    ".TO": "^GSPTSE", ".SA": "^BVSP", ".MX": "^MXX", ".HK": "^HSI",
}


def compute_market_distress_score(
    prices: pd.Series,
    peer_prices: pd.DataFrame,
    debt: Optional[float] = None,
    shares_outstanding: Optional[float] = None,
    market_index_prices: Optional[pd.Series] = None,
    auto_index: bool = True,
) -> pd.DataFrame:
    """Combine 6 market-based signals into a composite distress score (0-1).

    Signals: volatility spike, correlation breakdown, tail risk,
             Merton DD, drawdown severity, market index underperformance.
    """
    result = pd.DataFrame(index=prices.index)

    # 1. Volatility spike
    vol_regime = detect_volatility_regime(prices)
    result["vol_signal"] = np.clip(vol_regime["vol_ratio"] / 5.0, 0, 1)

    # 2. Correlation breakdown
    if peer_prices is not None and not peer_prices.empty:
        corr_bd = detect_correlation_breakdown(prices, peer_prices)
        result["corr_signal"] = (1 - np.clip(corr_bd["corr_with_peers"], 0, 1))
    else:
        result["corr_signal"] = 0.0

    # 3. Tail risk
    tail = detect_tail_risk_events(prices)
    result["tail_signal"] = np.clip(tail["breach_count"] / 10.0, 0, 1)

    # 4. Merton DD
    if debt is not None and shares_outstanding is not None and debt > 0:
        dd_results = []
        log_ret = np.log(prices / prices.shift(1))
        for i in range(252, len(prices)):
            eq_vol = log_ret.iloc[i-252:i].std() * np.sqrt(252)
            eq_val = prices.iloc[i] * shares_outstanding
            m = compute_merton_dd(eq_val, eq_vol, debt)
            dd_results.append({"date": prices.index[i], "DD": m["DD"]})
        if dd_results:
            dd_df = pd.DataFrame(dd_results).set_index("date")
            result["merton_dd"] = dd_df["DD"].reindex(result.index)
            result["dd_signal"] = np.clip((2.0 - result["merton_dd"].fillna(2.0)) / 2.0, 0, 1)
        else:
            result["merton_dd"] = np.nan
            result["dd_signal"] = 0.0
    else:
        result["merton_dd"] = np.nan
        result["dd_signal"] = 0.0

    # 5. Drawdown
    result["drawdown_30d"] = tail["max_drawdown_30d"]
    result["ddraw_signal"] = np.clip(-result["drawdown_30d"] / 0.50, 0, 1)

    # 6. Market Index Underperformance (with auto-fallback)
    _index_available = False
    _index_name = None
    aligned_index = None

    if market_index_prices is not None:
        aligned_index = market_index_prices.reindex(prices.index).ffill()
        _index_name = "explicit"
        _index_available = True
    elif auto_index:
        try:
            ticker_name = getattr(prices, 'name', '')
            _index_name = None
            for suffix, idx in _EXCHANGE_TO_INDEX.items():
                if isinstance(ticker_name, str) and ticker_name.upper().endswith(suffix.upper()):
                    _index_name = idx
                    break
            if _index_name is None:
                _index_name = "^GSPC"
            idx_slice = yf.download(_index_name,
                                     start=prices.index[0].strftime("%Y-%m-%d"),
                                     end=prices.index[-1].strftime("%Y-%m-%d"),
                                     progress=False, auto_adjust=False)
            if not idx_slice.empty:
                idx_prices = idx_slice["Close"] if "Close" in idx_slice.columns else idx_slice.iloc[:, 0]
                aligned_index = idx_prices.reindex(prices.index).ffill()
                _index_available = bool(aligned_index.notna().sum() > 10) if isinstance(aligned_index, pd.Series) else False
        except Exception:
            _index_available = False

    result["_index_used"] = _index_name if _index_available else "none"

    if _index_available and aligned_index is not None:
        stock_ret = np.log(prices / prices.shift(1))
        index_ret = np.log(aligned_index / aligned_index.shift(1))
        excess_ret = (stock_ret - index_ret).rolling(60).sum()
        result["index_excess_60d"] = excess_ret
        result["index_signal"] = np.clip(-excess_ret / 0.30, 0, 1)
    else:
        result["index_excess_60d"] = 0.0
        result["index_signal"] = 0.0

    # Composite
    signal_cols = ["vol_signal", "corr_signal", "tail_signal", "dd_signal", "ddraw_signal"]
    has_index = result["index_signal"].std() > 0.01
    if has_index:
        signal_cols.append("index_signal")
    result["composite_distress"] = result[signal_cols].mean(axis=1)
    result["n_signals"] = len(signal_cols)

    result["distress_level"] = pd.cut(
        result["composite_distress"], bins=[-0.01, 0.2, 0.4, 0.6, 1.0],
        labels=["Low", "Elevated", "High", "Critical"],
    )
    result["is_distressed"] = result["composite_distress"] > 0.4

    # Episode tracking
    episode_id = 0
    in_episode = False
    episode_ids = []
    for flag in result["is_distressed"]:
        if flag and not in_episode:
            episode_id += 1
            in_episode = True
        elif not flag:
            in_episode = False
        episode_ids.append(episode_id if flag else 0)
    result["distress_episode_id"] = episode_ids
    for eid in range(1, episode_id + 1):
        mask = result["distress_episode_id"] == eid
        if mask.sum() < 5:
            result.loc[mask, "distress_episode_id"] = 0
            result.loc[mask, "is_distressed"] = False
    return result


def extract_distress_date_ranges(
    df: pd.DataFrame, episode_col: str = "distress_episode_id",
    date_col: Optional[str] = "date",
) -> pd.DataFrame:
    """Extract start/end dates for each distress episode."""
    records = []
    dates = pd.to_datetime(df[date_col]) if date_col and date_col in df.columns else pd.to_datetime(df.index)
    episode_ids = sorted(df[df[episode_col] > 0][episode_col].unique())
    for eid in episode_ids:
        mask = df[episode_col] == eid
        episode_dates = dates[mask]
        if len(episode_dates) == 0:
            continue
        rec = {
            "episode_id": int(eid),
            "start_date": episode_dates.min(), "end_date": episode_dates.max(),
            "duration_days": (episode_dates.max() - episode_dates.min()).days,
            "n_observations": mask.sum(),
        }
        if "z_score" in df.columns:
            rec["mean_z_score"] = df.loc[mask, "z_score"].mean()
        if "composite_distress" in df.columns:
            rec["mean_composite"] = df.loc[mask, "composite_distress"].mean()
            rec["max_composite"] = df.loc[mask, "composite_distress"].max()
        records.append(rec)
    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values("start_date").reset_index(drop=True)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 5. PRICE RECONSTRUCTION (Peer-Based Gap Filling)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_pre_distress_betas(
    target_returns: pd.Series, peer_returns: pd.DataFrame,
) -> Tuple[np.ndarray, float, float]:
    """Estimate linear relationship between stock and peers using pre-distress data."""
    common_idx = target_returns.dropna().index.intersection(peer_returns.dropna().index)
    y = target_returns.loc[common_idx].values
    X = peer_returns.loc[common_idx].values
    mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
    y, X = y[mask], X[mask]
    if len(y) < 20:
        raise ValueError(f"Insufficient training data: only {len(y)} observations")
    model = LinearRegression().fit(X, y)
    return model.coef_, model.intercept_, model.score(X, y)


def reconstruct_distress_prices(
    target_prices: pd.Series,
    peer_prices: pd.DataFrame,
    distress_mask: pd.Series,
    pre_distress_end: pd.Timestamp,
    use_log: bool = True,
) -> pd.DataFrame:
    """Reconstruct prices during distress using pre-distress peer regression.

    Returns DataFrame with columns: actual, reconstructed, is_distressed, combined
    """
    result = pd.DataFrame({"actual": target_prices}, index=target_prices.index)
    # Ensure distress_mask is a pandas Series indexed by target_prices
    if not isinstance(distress_mask, pd.Series):
        distress_mask = pd.Series(distress_mask, index=target_prices.index)
    elif not distress_mask.index.equals(target_prices.index):
        distress_mask = distress_mask.reindex(target_prices.index).fillna(False).astype(bool)
    result["is_distressed"] = distress_mask.astype(bool)
    if use_log:
        target_ret = np.log(target_prices / target_prices.shift(1))
        peer_ret = np.log(peer_prices / peer_prices.shift(1))
    else:
        target_ret = target_prices.pct_change()
        peer_ret = peer_prices.pct_change()

    train_mask = target_ret.index <= pre_distress_end
    betas, alpha, r2 = estimate_pre_distress_betas(target_ret[train_mask], peer_ret[train_mask])

    reconstructed_ret = pd.Series(np.nan, index=target_ret.index)
    distress_idx = target_ret.index[distress_mask]
    for dt in distress_idx:
        peer_row = peer_ret.loc[dt].values
        if not np.any(np.isnan(peer_row)):
            reconstructed_ret.loc[dt] = alpha + np.dot(betas, peer_row)

    reconstructed_prices = pd.Series(np.nan, index=target_prices.index)
    if len(distress_idx) == 0:
        # No distress period — just return actual prices
        result["reconstructed"] = target_prices
        result["combined"] = target_prices
        return result

    anchor_idx = target_prices.index[target_prices.index < distress_idx[0]]
    anchor_price = target_prices.loc[anchor_idx[-1]] if len(anchor_idx) > 0 else target_prices.iloc[0]
    if len(anchor_idx) > 0:
        reconstructed_prices.loc[anchor_idx[-1]] = anchor_price
    else:
        reconstructed_prices.iloc[0] = anchor_price

    for i in range(len(target_prices.index) - 1):
        dt = target_prices.index[i]
        next_dt = target_prices.index[i + 1]
        if distress_mask.loc[next_dt]:
            if not np.isnan(reconstructed_ret.loc[next_dt]):
                reconstructed_prices.loc[next_dt] = (
                    reconstructed_prices.loc[dt] * np.exp(reconstructed_ret.loc[next_dt]) if use_log
                    else reconstructed_prices.loc[dt] * (1 + reconstructed_ret.loc[next_dt]))
            else:
                reconstructed_prices.loc[next_dt] = reconstructed_prices.loc[dt]
        else:
            reconstructed_prices.loc[next_dt] = target_prices.loc[next_dt]

    result["reconstructed"] = reconstructed_prices
    result["combined"] = result["reconstructed"].where(result["is_distressed"], result["actual"])
    result.attrs["betas"] = betas
    result.attrs["alpha"] = alpha
    result.attrs["r2"] = r2
    return result


def compute_sector_adjusted_prices(
    target_prices: pd.Series,
    peer_prices: pd.DataFrame,
    distress_start: pd.Timestamp,
    distress_end: pd.Timestamp,
    pre_distress_lookback_days: int = 252,
) -> pd.DataFrame:
    """Full pipeline: compute sector-adjusted prices for a distressed stock."""
    pre_start = distress_start - pd.Timedelta(days=int(pre_distress_lookback_days * 1.5))
    prices_slice = target_prices.loc[pre_start:]
    peers_slice = peer_prices.loc[pre_start:]
    distress_mask = pd.Series(False, index=prices_slice.index)
    distress_mask.loc[distress_start:distress_end] = True
    return reconstruct_distress_prices(
        target_prices=prices_slice, peer_prices=peers_slice,
        distress_mask=distress_mask,
        pre_distress_end=distress_start - pd.Timedelta(days=1), use_log=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. RISK METRICS — VaR & RWA
# ══════════════════════════════════════════════════════════════════════════════

def compute_historical_var(
    returns: pd.Series, confidence: float = 0.99, window: int = 260,
) -> pd.Series:
    """Compute rolling Historical VaR at given confidence level."""
    return returns.rolling(window).quantile(1 - confidence)


def compute_portfolio_var(
    price_df: pd.DataFrame, weights: Optional[dict] = None,
    confidence: float = 0.99, window: int = 260,
) -> dict:
    """Compute VaR for a basket of stocks.

    Parameters
    ----------
    price_df : pd.DataFrame
        Daily prices, one column per ticker.
    weights : dict, optional
        Portfolio weights. If None, equal-weighted.
    confidence : float
        VaR confidence level.
    window : int
        Rolling window in trading days.

    Returns
    -------
    dict with keys: VaR_latest, VaR_series, portfolio_returns, weights
    """
    returns = price_df.pct_change().dropna()
    if weights is None:
        w = {c: 1.0 / len(price_df.columns) for c in price_df.columns}
    else:
        w = weights
    portfolio_ret = sum(returns[c] * w.get(c, 0) for c in returns.columns)
    var_series = compute_historical_var(portfolio_ret, confidence, window)
    return {
        "VaR_latest": var_series.iloc[-1],
        "VaR_series": var_series,
        "portfolio_returns": portfolio_ret,
        "weights": w,
    }


def compute_var_comparison(
    reconstructed: pd.DataFrame,
    peer_prices: pd.DataFrame,
    weights: Optional[dict] = None,
    confidence: float = 0.99,
    window: int = 260,
) -> pd.DataFrame:
    """Compare VaR using raw distressed prices vs reconstructed prices.

    Returns a DataFrame with VaR_raw, VaR_reconstructed, and VaR_ratio.
    """
    raw_prices = pd.DataFrame({
        "Distressed": reconstructed["actual"],
    })
    for p in peer_prices.columns:
        raw_prices[p] = peer_prices[p]

    recon_prices = raw_prices.copy()
    recon_prices["Distressed"] = reconstructed["combined"]

    raw_var = compute_portfolio_var(raw_prices, weights, confidence, window)
    recon_var = compute_portfolio_var(recon_prices, weights, confidence, window)

    comparison = pd.DataFrame({
        "VaR_raw": raw_var["VaR_series"],
        "VaR_reconstructed": recon_var["VaR_series"],
    })
    comparison["VaR_ratio"] = comparison["VaR_raw"] / comparison["VaR_reconstructed"]
    comparison["VaR_overstatement_pct"] = (comparison["VaR_ratio"] - 1) * 100
    return comparison


# ══════════════════════════════════════════════════════════════════════════════
# 7. VISUALIZATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_distressed_vs_peers(
    distressed_ticker: str,
    peer_tickers: List[str],
    start: str,
    end: str,
    distress_start: Optional[str] = None,
    distress_end: Optional[str] = None,
    market_index_ticker: Optional[str] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (16, 7),
    title: Optional[str] = None,
    show_graphs: bool = True,
) -> plt.Figure:
    """Plot distressed stock price against healthy peers (greyed).

    Parameters
    ----------
    show_graphs : bool
        If False, returns the Figure without calling plt.show().
    """
    all_tickers = [distressed_ticker] + peer_tickers
    if market_index_ticker and market_index_ticker not in all_tickers:
        all_tickers.append(market_index_ticker)

    prices = cached_download(all_tickers, start=start, end=end, reload=False, progress=False)

    if prices.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14)
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        plot_data = prices.div(prices.iloc[0]) * 100
        ylabel = "Price Indexed to 100"
    else:
        plot_data = prices
        ylabel = "Price ($)"

    # Peers in grey
    for peer in peer_tickers:
        if peer in plot_data.columns:
            ps = plot_data[peer].dropna()
            ax.plot(ps.index, ps.values, color="#b0b0b0", linewidth=1.2, alpha=0.7,
                    label=f"  {peer}" if len(peer_tickers) <= 6 else None)

    # Market index in grey dashed
    if market_index_ticker and market_index_ticker in plot_data.columns:
        idx_ps = plot_data[market_index_ticker].dropna()
        ax.plot(idx_ps.index, idx_ps.values, color="#808080", linewidth=1.0, alpha=0.5,
                linestyle=":", label=f"  {market_index_ticker} (index)")

    # Distressed stock in bold red
    ts = plot_data[distressed_ticker].dropna()
    ax.plot(ts.index, ts.values, color="#c0392b", linewidth=2.8,
            label=f"🔴 {distressed_ticker} (DISTRESSED)", zorder=5)

    # Shade distress period
    if distress_start and distress_end:
        d_start = pd.Timestamp(distress_start)
        d_end = pd.Timestamp(distress_end)
        ax.axvspan(d_start, d_end, alpha=0.12, color="#e74c3c", label="Distress Period")
        ax.axvline(x=d_start, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(x=d_end, color="#27ae60", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title or f"{distressed_ticker} vs Peers", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if show_graphs:
        plt.show()
    return fig


def plot_multi_distressed_comparison(
    distressed_tickers: List[str],
    peer_groups: List[List[str]],
    start: str,
    end: str,
    normalize: bool = True,
    figsize: Tuple[int, int] = (16, 12),
    show_graphs: bool = True,
) -> plt.Figure:
    """Multi-panel comparison: one subplot per distressed stock vs peers."""
    n = len(distressed_tickers)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]
    for idx, (ticker, peers) in enumerate(zip(distressed_tickers, peer_groups)):
        ax = axes[idx]
        all_t = [ticker] + peers
        prices = cached_download(all_t, start=start, end=end, reload=False, progress=False)
        if prices.empty:
            ax.text(0.5, 0.5, f"{ticker}: no data", ha="center", va="center")
            continue
        plot_data = prices.div(prices.iloc[0]) * 100 if normalize else prices
        for peer in peers:
            if peer in plot_data.columns:
                ps = plot_data[peer].dropna()
                ax.plot(ps.index, ps.values, color="#b0b0b0", linewidth=1, alpha=0.6)
        ts = plot_data[ticker].dropna()
        ax.plot(ts.index, ts.values, color="#c0392b", linewidth=2.5, zorder=5)
        ax.set_ylabel("Index (100)" if normalize else "Price ($)", fontsize=10)
        ax.set_title(f"{ticker} vs {len(peers)} peers", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date", fontsize=12)
    fig.suptitle("Distressed Stocks vs Healthy Peers", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    if show_graphs:
        plt.show()
    return fig


def plot_reconstructed_prices(
    reconstructed: pd.DataFrame,
    ticker: str,
    show_graphs: bool = True,
) -> plt.Figure:
    """3-panel reconstruction chart: full history + distress zoom + ratio."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 1.5, 1.2])
    df = reconstructed.copy()
    r2 = df.attrs.get("r2", 0)
    distress_mask = df["is_distressed"]
    distress_region = df[distress_mask]

    # Panel 1: Full Price History
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df["actual"], color="#2c3e50", linewidth=1.8, label="Actual Price", zorder=3)
    ax1.plot(df.index, df["combined"], color="#27ae60", linewidth=2.2, label="Combined Series", zorder=4, alpha=0.95)
    if distress_mask.any():
        ax1.plot(df.loc[distress_mask].index, df.loc[distress_mask, "reconstructed"],
                 color="#e74c3c", linewidth=3.0, linestyle="--",
                 label="🟠 Sector-Implied (during distress)", zorder=6, alpha=0.95)
        ax1.plot(df.loc[distress_mask].index, df.loc[distress_mask, "actual"],
                 color="#e74c3c", linewidth=1.5, linestyle=":", alpha=0.6, zorder=2)
    if not distress_region.empty:
        ax1.axvspan(distress_region.index[0], distress_region.index[-1], alpha=0.08, color="#e74c3c")
        mid_idx = len(distress_region) // 2
        mid_actual = distress_region["actual"].iloc[mid_idx]
        mid_recon = distress_region["reconstructed"].iloc[mid_idx]
        if not np.isnan(mid_recon) and mid_recon > 0:
            gap_pct = (mid_actual / mid_recon - 1) * 100
            ax1.annotate(f"Distress Gap: {'+' if gap_pct > 0 else ''}{gap_pct:.1f}%",
                        xy=(distress_region.index[mid_idx], min(mid_actual, mid_recon)),
                        fontsize=10, fontweight="bold", color="#c0392b", ha="center", va="top",
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#e74c3c", alpha=0.95))
    ax1.set_ylabel("Price ($)", fontsize=13, fontweight="bold")
    ax1.set_title(f"{ticker} — Price Reconstruction (Pre-distress R² = {r2:.3f})", fontsize=15, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Distress Zoom
    ax2 = fig.add_subplot(gs[1])
    if not distress_region.empty:
        pre_idx = max(0, df.index.get_loc(distress_region.index[0]) - 10)
        post_idx = min(len(df) - 1, df.index.get_loc(distress_region.index[-1]) + 5)
        zoom_df = df.iloc[pre_idx:post_idx + 1]
        ax2.plot(zoom_df.index, zoom_df["actual"], color="#2c3e50", linewidth=2.0, label="Actual")
        zoom_distress = zoom_df["is_distressed"]
        if zoom_distress.any():
            ax2.plot(zoom_df.loc[zoom_distress].index, zoom_df.loc[zoom_distress, "reconstructed"],
                     color="#e74c3c", linewidth=3.5, linestyle="--", label="🟠 Sector-Implied")
            ax2.plot(zoom_df.loc[zoom_distress].index, zoom_df.loc[zoom_distress, "actual"],
                     color="#e74c3c", linewidth=1.5, linestyle=":", alpha=0.5)
        ax2.fill_between(zoom_df.index, zoom_df["actual"], zoom_df["reconstructed"],
                         where=(zoom_df["is_distressed"] & zoom_df["reconstructed"].notna()),
                         color="#e74c3c", alpha=0.15, label="Distress Discount")
        ax2.axvspan(distress_region.index[0], distress_region.index[-1], alpha=0.06, color="#e74c3c")
        ax2.set_title("🔍 Distress Window — Zoomed View", fontsize=14, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No distress period detected", ha="center", va="center",
                 fontsize=14, color="grey", transform=ax2.transAxes)
    ax2.set_ylabel("Price ($)", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Ratio
    ax3 = fig.add_subplot(gs[2])
    ratio = df["actual"] / df["reconstructed"]
    ratio_normal = ratio.where(~distress_mask)
    ratio_distress = ratio.where(distress_mask)
    ax3.axhline(y=1.0, color="#27ae60", linestyle="-", linewidth=2, alpha=0.7, label="Parity (1.0)")
    if ratio_normal.notna().any():
        ax3.plot(ratio_normal.index, ratio_normal.values, color="#3498db", linewidth=1, alpha=0.5, label="Ratio (normal)")
    if ratio_distress.notna().any():
        ax3.plot(ratio_distress.index, ratio_distress.values, color="#e74c3c", linewidth=2.2,
                 marker="o", markersize=4, label="Ratio (distress)", zorder=5)
        ax3.fill_between(ratio_distress.index, 1.0, ratio_distress.values,
                         where=(ratio_distress < 1.0), color="#e74c3c", alpha=0.15, interpolate=True)
    if not distress_region.empty:
        ax3.axvspan(distress_region.index[0], distress_region.index[-1], alpha=0.06, color="#e74c3c")
    ax3.set_ylabel("Actual ÷ Reconstructed", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Date", fontsize=13, fontweight="bold")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, alpha=0.3)
    rmin, rmax = ratio.min(), ratio.max()
    pad = max(0.08, (rmax - rmin) * 0.12)
    ax3.set_ylim(max(0.4, rmin - pad), min(2.5, rmax + pad))
    fig.tight_layout()
    if show_graphs:
        plt.show()
    return fig


def plot_var_comparison(
    var_comparison: pd.DataFrame,
    ticker: str,
    figsize: Tuple[int, int] = (16, 7),
    show_graphs: bool = True,
) -> plt.Figure:
    """Plot VaR comparison: raw vs reconstructed prices."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.plot(var_comparison.index, var_comparison["VaR_raw"], color="#e74c3c", linewidth=2, label="VaR (raw prices)")
    ax1.plot(var_comparison.index, var_comparison["VaR_reconstructed"], color="#27ae60", linewidth=2, label="VaR (reconstructed)")
    ax1.set_ylabel("VaR (99%)", fontsize=12, fontweight="bold")
    ax1.set_title(f"{ticker} — VaR: Raw vs Reconstructed", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(var_comparison.index, var_comparison["VaR_overstatement_pct"], color="#e74c3c", linewidth=2)
    ax2.fill_between(var_comparison.index, 0, var_comparison["VaR_overstatement_pct"],
                     where=(var_comparison["VaR_overstatement_pct"] > 0), color="#e74c3c", alpha=0.2)
    ax2.axhline(y=0, color="#27ae60", linestyle="--", linewidth=1)
    ax2.set_ylabel("VaR Overstatement (%)", fontsize=12, fontweight="bold")
    ax2.set_title(f"{ticker} — VaR Overstatement from Raw Prices", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if show_graphs:
        plt.show()
    return fig


def plot_z_score_dashboard(
    z_score_df: pd.DataFrame, ticker: str,
    price_series: Optional[pd.Series] = None, show_graphs: bool = True,
) -> plt.Figure:
    """Z-Score evolution with zone shading and optional price overlay."""
    n_panels = 2 if price_series is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    df = z_score_df.copy()
    ax1 = axes[0]
    ax1.axhline(y=DISTRESS_THRESHOLD, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Distress ({DISTRESS_THRESHOLD})")
    ax1.axhline(y=GREY_THRESHOLD, color="#f39c12", linestyle="--", linewidth=1.5, label=f"Grey ({GREY_THRESHOLD})")
    for _, row in df.iterrows():
        if row["zone"] == "Distress":
            ax1.axvspan(row["date"] - pd.Timedelta(days=45), row["date"] + pd.Timedelta(days=45),
                        alpha=0.1, color="#e74c3c", zorder=0)
    for zone, color in ZONE_COLORS.items():
        mask = df["zone"] == zone
        if mask.any():
            ax1.scatter(df.loc[mask, "date"], df.loc[mask, "z_score"],
                       c=color, s=80, label=zone, zorder=5, edgecolors="white", linewidth=0.5)
    ax1.plot(df["date"], df["z_score"], color="#2c3e50", alpha=0.3, linewidth=1, zorder=1)
    ax1.fill_between(df["date"], 0, df["z_score"], alpha=0.05, color="#3498db")
    ax1.set_ylabel("Altman Z-Score", fontsize=12, fontweight="bold")
    ax1.set_title(f"{ticker} — Altman Z-Score Evolution", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    if price_series is not None and n_panels > 1:
        ax2 = axes[1]
        ax2.plot(price_series.index, price_series.values, color="#2c3e50", linewidth=1.5, label="Price")
        for _, row in df.iterrows():
            if row["zone"] == "Distress":
                ax2.axvspan(row["date"] - pd.Timedelta(days=45), row["date"] + pd.Timedelta(days=45),
                            alpha=0.15, color="#e74c3c", zorder=0)
        ax2.set_ylabel("Price ($)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_title(f"{ticker} — Price with Distress Overlay", fontsize=14, fontweight="bold")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    if show_graphs:
        plt.show()
    return fig


def plot_sector_r2_validation(
    r2_results: pd.DataFrame,
    show_graphs: bool = True,
) -> plt.Figure:
    """Plot sector R² comparison: CAPM vs Peer Regression."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    x = np.arange(len(r2_results))
    width = 0.2
    ax1.bar(x - 1.5*width, r2_results["R²_CAPM"], width, label="CAPM (Market only)", color="#3498db", alpha=0.8)
    ax1.bar(x - 0.5*width, r2_results["R²_ETF"], width, label="Market + Sector ETF", color="#9b59b6", alpha=0.8)
    ax1.bar(x + 0.5*width, r2_results["R²_PeerPort"], width, label="Market + Peer Portfolio", color="#f39c12", alpha=0.8)
    ax1.bar(x + 1.5*width, r2_results["R²_FullPeers"], width, label="Market + Individual Peers", color="#27ae60", alpha=0.8)
    ax1.set_ylabel("R² (Variance Explained)", fontsize=12, fontweight="bold")
    ax1.set_title("How Much of Stock Returns Does the Sector Explain?", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(r2_results["Test_Stock"].values, fontsize=10)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")
    ax2.bar(x, r2_results["R²_CAPM"], 0.5, label="Market only", color="#3498db", alpha=0.6)
    ax2.bar(x, r2_results["Sector_Premium_Full"], 0.5, bottom=r2_results["R²_CAPM"],
            label="Sector Premium", color="#27ae60", alpha=0.8)
    ax2.set_ylabel("R²", fontsize=12, fontweight="bold")
    ax2.set_title("Sector Premium: Additional Variance Explained by Peers", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(r2_results["Test_Stock"].values, fontsize=10)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    for i, row in r2_results.iterrows():
        ax2.text(i, row["R²_FullPeers"] + 0.01, f"{row['R²_FullPeers']*100:.0f}%", ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    if show_graphs:
        plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 8. CROSS-SECTOR R² VALIDATION (Empirical Proof for Peer Regression)
# ══════════════════════════════════════════════════════════════════════════════

_SECTOR_TESTS = {
    "Energy (Oil & Gas E&P)": {"stocks": ["XOM", "CVX", "COP", "EOG", "OXY"], "etf": "XLE", "test_stock": "OXY"},
    "Technology": {"stocks": ["AAPL", "MSFT", "NVDA", "AMD", "CRM"], "etf": "XLK", "test_stock": "AMD"},
    "Financials": {"stocks": ["JPM", "GS", "BAC", "C", "MS"], "etf": "XLF", "test_stock": "C"},
    "Healthcare": {"stocks": ["JNJ", "PFE", "MRK", "ABBV", "LLY"], "etf": "XLV", "test_stock": "PFE"},
    "Consumer Discretionary": {"stocks": ["AMZN", "TSLA", "HD", "NKE", "SBUX"], "etf": "XLY", "test_stock": "NKE"},
    "Industrials": {"stocks": ["CAT", "GE", "BA", "HON", "UPS"], "etf": "XLI", "test_stock": "BA"},
}


def run_sector_r2_validation(
    lookback_years: int = 3,
    market_proxy: str = "SPY",
    show_graphs: bool = True,
) -> pd.DataFrame:
    """Run cross-sector R² validation: CAPM vs Peer Regression.

    Returns DataFrame with columns: Sector, Test_Stock, R²_CAPM, R²_ETF,
    R²_PeerPort, R²_FullPeers, Sector_Premium_Full.
    """
    all_results = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(lookback_years * 365 * 1.1))

    for sector_name, cfg in _SECTOR_TESTS.items():
        stocks = cfg["stocks"]
        etf = cfg["etf"]
        test_stock = cfg["test_stock"]
        peers = [s for s in stocks if s != test_stock]
        all_tickers = [test_stock] + peers + [etf, market_proxy]

        try:
            prices = cached_download(all_tickers, start=start_date.strftime("%Y-%m-%d"),
                                      end=end_date.strftime("%Y-%m-%d"), progress=False)
        except Exception as e:
            print(f"  ⚠️  {sector_name}: download failed — {e}")
            continue

        if prices.empty or len(prices) < 252:
            continue

        rets = np.log(prices / prices.shift(1)).dropna()
        test_ret = rets[test_stock]
        market_ret = rets[market_proxy]
        etf_ret = rets[etf]
        peer_rets = rets[peers]

        common = test_ret.dropna().index.intersection(market_ret.dropna().index)
        common = common.intersection(etf_ret.dropna().index)
        for p in peers:
            common = common.intersection(rets[p].dropna().index)

        y = test_ret.loc[common].values
        X_mkt = market_ret.loc[common].values.reshape(-1, 1)

        r2_capm = LinearRegression().fit(X_mkt, y).score(X_mkt, y)
        X_etf = np.column_stack([market_ret.loc[common].values, etf_ret.loc[common].values])
        r2_etf = LinearRegression().fit(X_etf, y).score(X_etf, y)
        peer_port_ret = peer_rets.mean(axis=1).loc[common].values
        X_peer_port = np.column_stack([market_ret.loc[common].values, peer_port_ret])
        r2_peer_port = LinearRegression().fit(X_peer_port, y).score(X_peer_port, y)
        X_peers = market_ret.loc[common].values.reshape(-1, 1)
        for p in peers:
            X_peers = np.column_stack([X_peers, rets[p].loc[common].values])
        r2_peers = LinearRegression().fit(X_peers, y).score(X_peers, y)

        all_results.append({
            "Sector": sector_name, "Test_Stock": test_stock,
            "R²_CAPM": round(r2_capm, 4), "R²_ETF": round(r2_etf, 4),
            "R²_PeerPort": round(r2_peer_port, 4), "R²_FullPeers": round(r2_peers, 4),
            "Sector_Premium_Full": round(r2_peers - r2_capm, 4),
        })

    df_results = pd.DataFrame(all_results)
    if show_graphs:
        plot_sector_r2_validation(df_results, show_graphs=True)
    return df_results


# ══════════════════════════════════════════════════════════════════════════════
# 9. MARKET DISTRESS SCAN (Multi-Ticker)
# ══════════════════════════════════════════════════════════════════════════════

_BUILTIN_PEER_MAP = {
    "XOM": ["CVX", "COP", "EOG", "SLB", "OXY"], "CVX": ["XOM", "COP", "EOG", "SLB", "OXY"],
    "COP": ["XOM", "CVX", "EOG", "SLB", "OXY"], "EOG": ["XOM", "CVX", "COP", "SLB", "OXY"],
    "SLB": ["XOM", "CVX", "COP", "EOG", "HAL"], "OXY": ["XOM", "CVX", "COP", "EOG", "SLB"],
    "RIG": ["SLB", "HAL", "BKR", "NOV", "OXY"], "HP": ["SLB", "HAL", "BKR", "NOV", "XOM"],
    "PTEN": ["SLB", "HAL", "BKR", "NOV", "OXY"], "WTI": ["OXY", "COP", "EOG", "XOM", "CVX"],
    "WMT": ["TGT", "COST", "AMZN", "M", "KSS"], "TGT": ["WMT", "COST", "AMZN", "M", "KSS"],
    "COST": ["WMT", "TGT", "AMZN", "M", "KSS"], "M": ["KSS", "JWN", "TGT", "WMT", "GPS"],
    "GPS": ["AEO", "ANF", "ROST", "TJX", "M"], "JWN": ["M", "KSS", "DDS", "TGT", "GPS"],
    "KSS": ["M", "JWN", "TGT", "ROST", "GPS"], "BBBYQ": ["RH", "WSM", "AMZN", "WMT", "TGT"],
    "AMZN": ["WMT", "TGT", "COST", "M", "BABA"], "BABA": ["AMZN", "JD", "PDD", "NIO", "BIDU"],
}


def run_market_scan(
    tickers: List[str],
    peer_map: Optional[Dict[str, List[str]]] = None,
    lookback_years: int = 4,
    top_n: int = 20,
    show_graphs: bool = True,
    figsize: Tuple[int, int] = (16, 8),
) -> pd.DataFrame:
    """Run market distress score across a ticker universe and produce bar chart + table.

    Returns DataFrame with columns: Ticker, Composite, Level, Merton_DD, Latest_Price, n_signals.
    """
    if peer_map is None:
        peer_map = _BUILTIN_PEER_MAP

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_years * 365)
    records = []

    for idx, tkr in enumerate(tickers):
        print(f"  [{idx+1}/{len(tickers)}] {tkr}...", end=" ")
        try:
            peer_list = peer_map.get(tkr, [])
            if len(peer_list) == 0:
                print(" - no peers")
                continue
            all_t = [tkr] + peer_list[:6]
            prices = cached_download(all_t, start=start_date.strftime("%Y-%m-%d"),
                                      end=end_date.strftime("%Y-%m-%d"), progress=False)
            if prices.empty or tkr not in prices.columns:
                print(" - no price data")
                continue
            target_p = prices[tkr].dropna()
            peer_p = prices[[c for c in peer_list[:6] if c in prices.columns]].dropna()

            tkr_yf = yf.Ticker(tkr)
            try:
                bs = tkr_yf.balance_sheet
                if bs is not None and not bs.empty and bs.shape[1] > 0:
                    debt = _safe_get(bs.iloc[:, 0],
                        "Total Liabilities Net Minority Interest", "Total Liabilities",
                        "Total liabilities", "TotalDebt", "Total Debt")
                else:
                    debt = np.nan
            except Exception:
                debt = np.nan
            try:
                shares = tkr_yf.info.get("sharesOutstanding", None)
            except Exception:
                shares = None

            result = compute_market_distress_score(
                prices=target_p, peer_prices=peer_p,
                debt=debt if not np.isnan(debt) else None, shares_outstanding=shares,
            )
            composite = result["composite_distress"].iloc[-1]
            level = result["distress_level"].iloc[-1]
            merton_dd = result["merton_dd"].iloc[-1] if "merton_dd" in result.columns else np.nan

            records.append({
                "Ticker": tkr, "Composite": round(composite, 3), "Level": level,
                "Merton_DD": round(merton_dd, 2),
                "Latest_Price": round(prices[tkr].dropna().iloc[-1], 2),
                "n_signals": int(result["n_signals"].iloc[0]),
                "color": "#c0392b" if composite > 0.4 else ("#f39c12" if composite > 0.2 else "#27ae60"),
            })
            print(f"→ {composite:.3f} ({level})")
        except Exception as e:
            print(f": {str(e)[:60]}")
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("Composite", ascending=False)
    if show_graphs:
        plot_df = df.head(top_n).iloc[::-1]
        fig, (ax_bar, ax_table) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2, 1.2]})
        ax_bar.barh(range(len(plot_df)), plot_df["Composite"].values,
                    color=plot_df["color"].values, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax_bar.set_yticks(range(len(plot_df)))
        ax_bar.set_yticklabels(plot_df["Ticker"].values, fontsize=10)
        ax_bar.axvline(x=0.4, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7, label="Distress (0.4)")
        ax_bar.axvline(x=0.2, color="#f39c12", linestyle="--", linewidth=1, alpha=0.7, label="Elevated (0.2)")
        ax_bar.set_xlabel("Composite Distress Score", fontsize=12, fontweight="bold")
        ax_bar.set_title(f"Market Distress Scan — {len(df)} tickers", fontsize=14, fontweight="bold")
        ax_bar.legend(loc="lower right", fontsize=8)
        ax_bar.grid(True, alpha=0.3, axis="x")
        for i, (_, row) in enumerate(plot_df.iterrows()):
            ax_bar.text(row["Composite"] + 0.01, i, f" {row['Level']}", va="center", fontsize=8,
                       color=row["color"], fontweight="bold")
        table_df = df[["Ticker", "Composite", "Level", "Merton_DD", "Latest_Price", "n_signals"]].copy()
        table_df.columns = ["Ticker", "Composite", "Level", "Merton DD", "Price $", "Sig#"]
        ax_table.axis("off")
        tbl = ax_table.table(cellText=table_df.values, colLabels=table_df.columns,
                              cellLoc="center", loc="center", colWidths=[0.12, 0.12, 0.12, 0.12, 0.10, 0.06])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.3)
        for i in range(len(table_df)):
            level = table_df["Level"].iloc[i]
            color = "#ffe0e0" if level in ("Critical", "High") else "#fff3cd" if level == "Elevated" else "#d5f5e3"
            tbl[(i + 1, 2)].set_facecolor(color)
        fig.tight_layout()
        plt.show()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 10. STATIONARITY TEST (Augmented Dickey-Fuller)
# ══════════════════════════════════════════════════════════════════════════════

def test_stationarity(
    series: pd.Series, significance: float = 0.05,
) -> dict:
    """Run Augmented Dickey-Fuller test on a time series.

    Returns dict with: is_stationary, adf_statistic, p_value, critical_values.
    """
    from statsmodels.tsa.stattools import adfuller
    clean = series.dropna()
    if len(clean) < 20:
        return {"is_stationary": False, "error": "insufficient data"}
    result = adfuller(clean.values, autolag="AIC")
    return {
        "is_stationary": result[1] < significance,
        "adf_statistic": result[0],
        "p_value": result[1],
        "critical_values": result[4],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 11. CROSS-VALIDATION: VaR Breach Summary Across All Stocks
# ══════════════════════════════════════════════════════════════════════════════

def compute_cross_validation_breaches(
    config: dict,
    tickers: Optional[List[str]] = None,
    confidence: float = 0.99,
    window: int = 260,
    show_graphs: bool = True,
) -> pd.DataFrame:
    """Run cross-validation breach analysis across all distressed stocks.

    For each stock, estimates pre-distress betas (market index + peers),
    reconstructs prices, computes 260-day rolling VaR at the given confidence
    level, and counts breaches before and after reconstruction.

    Parameters
    ----------
    config : dict
        Loaded distressed stocks config.
    tickers : list of str, optional
        Tickers to include. Defaults to all in config.
    confidence : float
        VaR confidence level (default 0.99 for 99%).
    window : int
        Rolling window in trading days (default 260).
    show_graphs : bool
        If True, displays a before/after breach comparison chart.

    Returns
    -------
    pd.DataFrame with columns: Ticker, Name, Sector, R²_Peers, R²_IndexOnly,
    R²_PeersIndex, MarketBeta, BreachBefore, BreachAfter, BreachChange, etc.
    """
    if tickers is None:
        tickers = list(config["stocks"].keys())

    expected_rate = (1 - confidence) * 100
    results = []

    for tkr in tickers:
        try:
            cfg = config["stocks"].get(tkr)
            if cfg is None:
                continue

            ds = pd.Timestamp(cfg["distress_start"])
            de = pd.Timestamp(cfg["distress_end"])
            dl_start = (ds - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
            dl_end   = (de + pd.DateOffset(years=1)).strftime("%Y-%m-%d")
            idx = cfg["market_index"]
            peers = cfg["peers"][:6]
            all_t = [tkr] + peers + [idx]

            prices = cached_download(all_t, start=dl_start, end=dl_end, progress=False)
            tp = prices[tkr].dropna()
            pp = prices[[c for c in peers if c in prices.columns]].dropna()

            tr = np.log(tp / tp.shift(1))
            pr = np.log(pp / pp.shift(1))

            # Pre-distress training window
            pre_mask = tp.index < ds
            train_tr = tr[pre_mask].dropna()
            train_pr = pr.reindex(train_tr.index).dropna()
            common = train_tr.index.intersection(train_pr.index)

            # Model 1: Peers only
            r2_peers = np.nan
            if len(common) > 20 and pp.shape[1] > 0:
                Xp = train_pr.loc[common].values
                yp = train_tr.loc[common].values
                mask = ~(np.isnan(yp) | np.any(np.isnan(Xp), axis=1))
                if mask.sum() > 20:
                    r2_peers = LinearRegression().fit(Xp[mask], yp[mask]).score(Xp[mask], yp[mask])

            # Model 2: Index only (CAPM)
            r2_idx = np.nan
            market_beta = np.nan
            if idx in prices.columns:
                idx_ret = np.log(prices[idx] / prices[idx].shift(1)).dropna()
                ci = train_tr.index.intersection(idx_ret.index)
                if len(ci) > 20:
                    Xi = idx_ret.loc[ci].values.reshape(-1, 1)
                    yi = train_tr.loc[ci].values
                    model = LinearRegression().fit(Xi, yi)
                    r2_idx = model.score(Xi, yi)
                    market_beta = model.coef_[0]

            # Model 3: Peers + Index
            r2_both = np.nan
            if idx in prices.columns and pp.shape[1] > 0:
                idx_ret = np.log(prices[idx] / prices[idx].shift(1)).dropna()
                both_df = pd.concat([idx_ret, pr], axis=1).dropna()
                ci = train_tr.index.intersection(both_df.index)
                if len(ci) > 20:
                    Xb = both_df.loc[ci].values
                    yb = train_tr.loc[ci].values
                    mask = ~(np.isnan(yb) | np.any(np.isnan(Xb), axis=1))
                    if mask.sum() > 20:
                        r2_both = LinearRegression().fit(Xb[mask], yb[mask]).score(Xb[mask], yb[mask])

            # Reconstruction (peers only)
            ret_common = tr.dropna().index.intersection(pr.dropna().index)
            betas_p, alpha_p, _ = estimate_pre_distress_betas(
                tr.loc[ret_common.intersection(train_tr.index)],
                pr.loc[ret_common.intersection(train_pr.index)]
            )
            d_mask = (tp.index >= ds) & (tp.index <= de)
            recon = reconstruct_distress_prices(tp, pp, d_mask, ds)

            # Build portfolio and compute breaches
            raw_pr = pd.DataFrame({tkr: recon["actual"]})
            for p in pp.columns:
                raw_pr[p] = pp[p]
            recon_pr = raw_pr.copy()
            recon_pr[tkr] = recon["combined"]

            w = {c: 1.0 / len(raw_pr.columns) for c in raw_pr.columns}
            raw_ret = raw_pr.pct_change().dropna()
            recon_ret = recon_pr.pct_change().dropna()
            raw_port = sum(raw_ret[c] * w[c] for c in raw_ret.columns)
            recon_port = sum(recon_ret[c] * w[c] for c in recon_ret.columns)

            raw_var_series = raw_port.rolling(window).apply(lambda x: np.percentile(x, 100 * (1 - confidence)), raw=False)
            recon_var_series = recon_port.rolling(window).apply(lambda x: np.percentile(x, 100 * (1 - confidence)), raw=False)

            ci = raw_port.index.intersection(recon_var_series.dropna().index)
            raw_p = raw_port.loc[ci]
            recon_p = recon_port.loc[ci]
            raw_v = raw_var_series.loc[ci]
            recon_v = recon_var_series.loc[ci]

            n_total = len(ci)
            breach_raw = (raw_p < raw_v).sum()
            breach_recon = (recon_p < recon_v).sum()
            rate_raw = breach_raw / n_total * 100 if n_total > 0 else 0
            rate_recon = breach_recon / n_total * 100 if n_total > 0 else 0

            # ── Accuracy: deviation from expected breach rate ────────────────
            # A well-calibrated VaR model should breach at rate = 1-confidence.
            # Accuracy = |observed - expected| — LOWER is better calibrated.
            accuracy_raw  = abs(rate_raw - expected_rate)
            accuracy_recon = abs(rate_recon - expected_rate)
            accuracy_change = accuracy_recon - accuracy_raw  # negative = better

            # ── Binomial test p-value (two-sided, vs expected rate) ──────────
            # Tests H0: breach rate = expected rate (i.e., model is calibrated)
            from scipy.stats import binomtest
            if n_total > 0:
                pval_raw = binomtest(breach_raw, n_total, p=expected_rate/100, alternative='two-sided').pvalue
                pval_recon = binomtest(breach_recon, n_total, p=expected_rate/100, alternative='two-sided').pvalue
            else:
                pval_raw = pval_recon = 1.0

            results.append({
                "Ticker": tkr,
                "Name": cfg.get("name", ""),
                "Sector": cfg.get("sector", ""),
                "Market Index": cfg.get("market_index", ""),
                "N Peers": pp.shape[1],
                "R² Peers": round(r2_peers, 4) if not np.isnan(r2_peers) else None,
                "R² Index": round(r2_idx, 4) if not np.isnan(r2_idx) else None,
                "R² Peers+Index": round(r2_both, 4) if not np.isnan(r2_both) else None,
                "Market Beta": round(market_beta, 3) if not np.isnan(market_beta) else None,
                "Total Days": n_total,
                "Breaches Before": int(breach_raw),
                "Breaches After": int(breach_recon),
                "Rate Before %": round(rate_raw, 2),
                "Rate After %": round(rate_recon, 2),
                "Expected Rate %": round(expected_rate, 1),
                "Breach Change": int(breach_recon - breach_raw),
                "Accuracy Before": round(accuracy_raw, 3),
                "Accuracy After": round(accuracy_recon, 3),
                "Accuracy Change": round(accuracy_change, 3),
                "P-Value Before": round(pval_raw, 4),
                "P-Value After": round(pval_recon, 4),
                # More accurate = lower deviation from expected rate
                "More Accurate": "After" if accuracy_recon < accuracy_raw else ("Before" if accuracy_raw < accuracy_recon else "Same"),
            })
        except Exception as e:
            results.append({
                "Ticker": tkr, "Name": cfg.get("name", ""), "Sector": cfg.get("sector", ""),
                "Breaches Before": None, "Breaches After": None, "Error": str(e)[:60],
            })

    df = pd.DataFrame(results).sort_values("Breaches Before", ascending=False, na_position="last")

    if show_graphs and not df.empty:
        valid = df.dropna(subset=["Breaches Before", "Breaches After"])
        if len(valid) > 0:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, max(5, len(valid) * 0.4)))
            y = np.arange(len(valid))
            h = 0.35

            # Panel 1: Breach counts before vs after
            ax1.barh(y - h/2, valid["Breaches Before"].values, h, color="#e74c3c", alpha=0.85, label="Before reconstruction")
            ax1.barh(y + h/2, valid["Breaches After"].values, h, color="#27ae60", alpha=0.85, label="After reconstruction")
            ax1.set_yticks(y)
            ax1.set_yticklabels([f'{r["Ticker"]} ({r["Sector"][:12]})' for _, r in valid.iterrows()], fontsize=9)
            ax1.set_xlabel("VaR Breaches", fontsize=12, fontweight="bold")
            ax1.set_title(f"VaR Breaches Before vs After ({window}d rolling, {int(confidence*100)}% CI)",
                          fontsize=13, fontweight="bold")
            ax1.legend(loc="lower right", fontsize=10)
            ax1.grid(True, alpha=0.3, axis="x")

            # Panel 2: Breach change
            ax2.barh(y, valid["Breach Change"].values, h/2,
                     color=["#e74c3c" if v > 0 else "#27ae60" for v in valid["Breach Change"].values],
                     alpha=0.8)
            ax2.axvline(x=0, color="black", linewidth=0.8)
            ax2.set_yticks(y)
            ax2.set_yticklabels([f'{r["Ticker"]} ({r["Sector"][:12]})' for _, r in valid.iterrows()], fontsize=9)
            ax2.set_xlabel("Breach Change (After − Before)", fontsize=12, fontweight="bold")
            ax2.set_title("Change in VaR Breaches (negative = fewer breaches)",
                          fontsize=13, fontweight="bold")
            ax2.grid(True, alpha=0.3, axis="x")

            # Panel 3: Accuracy — deviation from expected breach rate (%)
            # Lower deviation = better calibrated. Expected line at 0.
            ax3.barh(y - h/2, valid["Accuracy Before"].values, h, color="#e74c3c", alpha=0.85,
                     label="Before (raw prices)")
            ax3.barh(y + h/2, valid["Accuracy After"].values, h, color="#27ae60", alpha=0.85,
                     label="After (reconstructed)")
            expected_rate_line = (1 - confidence) * 100
            ax3.axvline(x=0, color="black", linewidth=0.8)
            ax3.set_yticks(y)
            ax3.set_yticklabels([f'{r["Ticker"]} ({r["Sector"][:12]})' for _, r in valid.iterrows()], fontsize=9)
            ax3.set_xlabel(f"|Observed − Expected| (target: {expected_rate_line:.0f}%)", fontsize=12, fontweight="bold")
            ax3.set_title("VaR Accuracy: Deviation from Expected Breach Rate\n(lower = better calibrated)",
                          fontsize=13, fontweight="bold")
            ax3.legend(loc="lower right", fontsize=10)
            ax3.grid(True, alpha=0.3, axis="x")

            fig.tight_layout()
            plt.show()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN (for standalone testing)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 12. SOURCE MAPPING — Cross-vendor ticker mapping for Reuters & Bloomberg
# ══════════════════════════════════════════════════════════════════════════════

# Built-in name map for peers & market indices not in the config's "name" field
# This is maintained here so it can be reused across notebooks and mappings.
_BUILTIN_NAME_MAP: Dict[str, str] = {
    # ── Energy ───────────────────────────────────────────────────────────────
    "XOM": "Exxon Mobil Corporation", "CVX": "Chevron Corporation",
    "COP": "ConocoPhillips", "EOG": "EOG Resources, Inc.",
    "SLB": "Schlumberger Limited", "HAL": "Halliburton Company",
    "BKR": "Baker Hughes Company", "NOV": "NOV Inc.",
    "BP": "BP p.l.c.", "SHEL": "Shell plc", "TTE": "TotalEnergies SE",
    # ── Consumer ─────────────────────────────────────────────────────────────
    "M": "Macy's Inc.", "KSS": "Kohl's Corporation", "JWN": "Nordstrom, Inc.",
    "TGT": "Target Corporation", "WMT": "Walmart Inc.", "DDS": "Dillard's, Inc.",
    "AEO": "American Eagle Outfitters, Inc.", "ANF": "Abercrombie & Fitch Co.",
    "ROST": "Ross Stores, Inc.", "TJX": "The TJX Companies, Inc.",
    "KMX": "CarMax, Inc.", "AN": "AutoNation, Inc.", "PAG": "Penske Automotive Group, Inc.",
    "GPII": "Group 1 Automotive, Inc.",
    # ── Airlines ─────────────────────────────────────────────────────────────
    "DAL": "Delta Air Lines, Inc.", "UAL": "United Airlines Holdings, Inc.",
    "LUV": "Southwest Airlines Co.", "AAL": "American Airlines Group Inc.",
    "ALK": "Alaska Air Group, Inc.", "JBLU": "JetBlue Airways Corporation",
    "AF.PA": "Air France-KLM SA", "ICAGY": "International Consolidated Airlines Group SA",
    # ── Technology ───────────────────────────────────────────────────────────
    "AMD": "Advanced Micro Devices, Inc.", "NVDA": "NVIDIA Corporation",
    "QCOM": "QUALCOMM Incorporated", "MRVL": "Marvell Technology, Inc.",
    "TSM": "Taiwan Semiconductor Manufacturing Company Limited",
    "STM": "STMicroelectronics N.V.", "IFNNY": "Infineon Technologies AG",
    "CAP.PA": "Capgemini SE", "SOP.PA": "Sopra Steria Group SA",
    "INFY": "Infosys Limited", "WIT": "Wipro Limited",
    "ACN": "Accenture plc", "CTSH": "Cognizant Technology Solutions Corporation",
    # ── Market Indices ───────────────────────────────────────────────────────
    "^GSPC": "S&P 500 Index", "^FCHI": "CAC 40 Index",
    "^GDAXI": "DAX Index", "^FTSE": "FTSE 100 Index",
    "FTSEMIB.MI": "FTSE MIB Index", "^SMSI": "IBEX 35 Index",
    "^SSMI": "Swiss Market Index", "^AEX": "AEX Index",
    "^N225": "Nikkei 225 Index", "^KS11": "KOSPI Composite Index",
    "^TWII": "Taiwan Weighted Index", "^NSEI": "Nifty 50 Index",
    "^AXJO": "ASX 200 Index", "^GSPTSE": "S&P/TSX Composite Index",
    "^BVSP": "Bovespa Index", "^MXX": "IPC Mexico Index",
    "^HSI": "Hang Seng Index",
    # ── Sector ETFs ──────────────────────────────────────────────────────────
    "XLE": "Energy Select Sector SPDR Fund", "SPY": "SPDR S&P 500 ETF Trust",
    "XLF": "Financial Select Sector SPDR Fund", "XLK": "Technology Select Sector SPDR Fund",
    "XLY": "Consumer Discretionary Select Sector SPDR Fund",
    "XLI": "Industrial Select Sector SPDR Fund",
    # ── Common Healthy Stocks (scan controls) ────────────────────────────────
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation",
    "JNJ": "Johnson & Johnson", "JPM": "JPMorgan Chase & Co.",
    "KO": "The Coca-Cola Company", "PG": "The Procter & Gamble Company",
}


def build_source_mapping(
    config: Optional[dict] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Build a cross-vendor ticker mapping for all entities in the config.

    Collects every ticker referenced in the distressed stocks config
    (distressed stocks, their peers, and market indices), attaches the
    known company/index name from the config or built-in name map, and
    provides empty columns for Reuters RIC and Bloomberg ticker to be
    filled in manually or programmatically.

    Parameters
    ----------
    config : dict, optional
        Pre-loaded config. If None, loads from default path.
    output_path : str or Path, optional
        If provided, writes the mapping to a JSON file.
        Defaults to ``source_config.json`` next to the config file.

    Returns
    -------
    pd.DataFrame with columns:
        yahoo_ticker, name, type, used_by, used_by_count,
        reuters_ric, bloomberg_ticker, bloomberg_name,
        isin, sedol, cusip
    """
    if config is None:
        config = load_config()

    stocks = config.get("stocks", {})

    # ── Collect all tickers with their roles ──────────────────────────────────
    # id -> {"name": ..., "type": ..., "used_by": set()}
    registry: Dict[str, Dict[str, Any]] = {}

    for tkr, cfg in stocks.items():
        # The distressed stock itself
        if tkr not in registry:
            registry[tkr] = {"name": cfg.get("name", ""), "type": "Distressed Stock", "used_by": set()}
        registry[tkr]["used_by"].add(tkr)

        # Peers
        for peer in cfg.get("peers", []):
            if peer not in registry:
                registry[peer] = {"name": _BUILTIN_NAME_MAP.get(peer, ""), "type": "Peer", "used_by": set()}
            registry[peer]["used_by"].add(tkr)

        # Market index
        idx = cfg.get("market_index", "")
        if idx and idx not in registry:
            registry[idx] = {"name": _BUILTIN_NAME_MAP.get(idx, idx), "type": "Market Index", "used_by": set()}
        if idx:
            registry[idx]["used_by"].add(tkr)

    # ── Build DataFrame ──────────────────────────────────────────────────────
    rows = []
    for tkr, info in sorted(registry.items()):
        used_list = sorted(info["used_by"])
        rows.append({
            "yahoo_ticker": tkr,
            "name": info["name"] or "",
            "type": info["type"],
            "used_by": ", ".join(used_list),
            "used_by_count": len(used_list),
            "reuters_ric": "",
            "bloomberg_ticker": "",
            "bloomberg_name": "",
            "isin": "",
            "sedol": "",
            "cusip": "",
        })

    df = pd.DataFrame(rows)

    # ── Write JSON ───────────────────────────────────────────────────────────
    if output_path is None:
        output_path = _DEFAULT_CONFIG_PATH.parent / "source_config.json"
    else:
        output_path = Path(output_path)

    payload = {
        "_description": (
            "Cross-vendor ticker mapping for distressed stocks, peers, and market indices. "
            "Yahoo Finance tickers are the primary key. "
            "Fill in reuters_ric, bloomberg_ticker, isin, etc. for each entity."
        ),
        "_version": "1.0",
        "_generated_by": "build_source_mapping() in distress_analysis.py",
        "_total_entities": len(df),
        "_breakdown": {
            "Distressed Stock": int((df["type"] == "Distressed Stock").sum()),
            "Peer": int((df["type"] == "Peer").sum()),
            "Market Index": int((df["type"] == "Market Index").sum()),
        },
        "entities": [
            {
                "yahoo_ticker": r["yahoo_ticker"],
                "name": r["name"] or None,
                "type": r["type"],
                "used_by": r["used_by"].split(", ") if r["used_by"] else [],
                "reuters_ric": r["reuters_ric"] or None,
                "bloomberg_ticker": r["bloomberg_ticker"] or None,
                "bloomberg_name": r["bloomberg_name"] or None,
                "isin": r["isin"] or None,
                "sedol": r["sedol"] or None,
                "cusip": r["cusip"] or None,
            }
            for _, r in df.iterrows()
        ],
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Source mapping written to %s (%d entities)", output_path, len(df))
    return df
