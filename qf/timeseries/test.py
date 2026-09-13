import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# =============================================================================
# 1. DATA ACQUISITION UTILITY (Retained and enhanced)
# =============================================================================


def download_ticker_data(ticker_symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame | None:
    """
    Downloads historical stock price data for a given ticker symbol from Yahoo Finance.
    Returns OHLCV DataFrame or None on failure.
    """
    print(f"\n--- Attempting to download data for {ticker_symbol} ---")
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(start=start_date, end=end_date)

        if hist.empty:
            print(f"⚠️ Warning: No data found for {ticker_symbol}.")
            return None
        else:
            # Focus on closing price for most predictive models
            df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            print(f"✅ Successfully downloaded {len(df)} records for {ticker_symbol}.")
            return df

    except Exception as e:
        print(f"❌ Error downloading data for {ticker_symbol}: {e}")
        return None

# =============================================================================
# 2. CORE METHODOLOGY MODULES (Synthesizing Notebook Logic)
# =============================================================================

def select_and_score_peers(target_data: pd.DataFrame, peer_tickers: list[str], distress_window_dates: tuple[pd.Timestamp, pd.Timestamp]) -> tuple[list[str], pd.DataFrame]:
    """
    Proprietary Step 1: Selects and scores relevant peers based on historical correlation/co-movement 
    within a defined distress window (e.g., worst quartile performance).

    Args:
        target_data: The primary stock's data.
        peer_tickers: List of other ticker symbols to compare against.
        distress_window_dates: (start, end) for calculating correlation stability.

    Returns:
        tuple[list[str], pd.DataFrame]: A tuple containing [scored_peers] and a summary DataFrame.
    """
    print("\n>>> MODULE STEP 1: Peer Selection & Scoring (Correlation/Volatility Analysis)")
    # TODO: Integrate logic from 'gap_filling_timeseries_modelling_v2.ipynb' here.
    # Requires downloading data for ALL peers over the window, calculating pairwise correlations, 
    # and filtering based on statistical metrics (e.g., correlation > X AND lowest Z-score deviation).
    print(">>> Placeholder: Peer scoring logic executed. Using mock results.")
    
    scored_peers = peer_tickers[:min(2, len(peer_tickers))] # Mock selecting up to 2 peers
    summary_df = pd.DataFrame({
        'Peer': scored_peers,
        'Score_Metric_A': np.random.uniform(0.5, 1.0, len(scored_peers)).round(3),
        'Correlation_Est': np.random.uniform(-0.8, 0.8, len(scored_peers)).round(2)
    })
    return scored_peers, summary_df


def reconstruct_gap_value(target_data: pd.DataFrame, peer_scores: pd.DataFrame, method: str = 'WEIGHTED_CORR') -> pd.Series | None:
    """
    Proprietary Step 2: Uses weighted regression/averaging (derived from distress reconstruction) 
    to estimate missing values in the target stock's closing price series.

    Args:
        target_data: The primary stock's data, which MUST have NaN gaps where filling is needed.
        peer_scores: DataFrame containing peer scores used for weighting/prediction.
        method: Reconstruction method ('AVERAGE', 'WEIGHTED_CORR').

    Returns:
        pd.Series | None: A Series with filled values, or None if no gaps found.
    """# ... existing code (up to line 137) ...
if __name__ == '__main__':
    print("===============================================================")
    print("!!! QUANTIFIABLE FINANCIAL TIME SERIES ANALYSIS PIPELINE v3 !!!")
    print("===============================================================")

    # --- CONFIGURATION ---
    TARGET_TICKER = "INTC"  # Intel (as a proxy for a distressed/volatile ticker)
    PEER_TICKERS = ["AMD", "NVDA", "TSM", "INTC"] # Note: INTC is target, others are peers
    START_DATE = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    # Define a 'distress window' for correlation analysis (e.g., last 30 days)
    distress_window = (
        pd.Timestamp(END_DATE) - timedelta(days=30),
        pd.Timestamp(END_DATE)
    )

    # --- STEP 1: DATA ACQUISITION ---
    target_df = download_ticker_data(TARGET_TICKER, START_DATE, END_DATE)
    
    if target_df is not None:
        # ARTIFICIAL DISTRESS SIMULATION: Inject NaN gaps into the target stock
        # We simulate "missing data" which often occurs during periods of extreme volatility/distress
        print(f"\n[SIMULATION] Injecting artificial NaN gaps into {TARGET_TICKER}...")
        target_df.loc[target_df.index[20:35], 'Close'] = np.nan
        target_df.loc[target_df.index[50:55], 'Close'] = np.nan

        # --- STEP 2: PEER SELECTION & SCORING ---
        # In a production environment, this would download all peer data and calculate metrics.
        # For this demo, we use the existing module logic.
        selected_peers, summary_stats = select_and_score_peers(
            target_df, 
            [p for p in PEER_TICKERS if p != TARGET_TICKER], 
            distress_window
        )
        print("\n--- Peer Scoring Summary ---")
        print(summary_stats)

        # --- STEP 3: RECONSTRUCTION ---
        reconstructed_series = reconstruct_gap_value(
            target_df, 
            summary_stats, 
            method='WEIGHTED_CORR'
        )

        if reconstructed_series is not None:
            # --- STEP 4: VISUALIZATION ---
            plot_reconstruction(target_df['Close'], reconstructed_series, TARGET_TICKER)
        else:
            print("❌ Reconstruction failed: No gaps to fill.")
    else:
        print("❌ Pipeline aborted: Target data unavailable.")

    print("\n===============================================================")
    print("!!! PIPELINE EXECUTION COMPLETE !!!")
    print("===============================================================")
    print(f"\n>>> MODULE STEP 2: Gap Reconstruction ({method} Method)")
    # TODO: Integrate core gap-filling logic from 'distress_reconstruction_methodology.ipynb' here.
    
    if target_data['Close'].isnull().any():
        print("   [INFO] Detected NaN gaps in the Closing Price.")
        
        if method == 'WEIGHTED_CORR':
            # Actual logic would fit a model: Y = Beta0 + Sum(Beta_i * Peer_i)
            reconstructed_values = target_data['Close'].fillna(method='ffill').iloc[target_data.index] # Fallback for mock
        else: # AVERAGE fallback
             reconstructed_values = target_data['Close'].interpolate(method='linear')

        return reconstructed_values
    else:
        print("   [INFO] No gaps detected in the 'Close' price column; no reconstruction needed.")
        return None


# =============================================================================
# 3. ANALYSIS & VISUALIZATION (Proof Generation)
# =============================================================================

def plot_reconstruction(original_series: pd.Series, reconstructed_series: pd.Series, ticker: str):
    """Generates comparative plots for visual verification of the reconstruction."""
    print(f"\n*** Generating Visualization for {ticker} ***")
    plt.figure(figsize=(14, 7))
    
    # Use a mask to only plot 'real' points and highlight where gaps were/filled
    mask = original_series.notna()
    sns.lineplot(x=original_series.index[mask], y=original_series[mask], label='Original Known Price', color='#3a6073')
    
    # Plot the reconstructed series over the entire date range
    plt.plot(original_series.index, reconstructed_series, label='Reconstructed/Filled Price (Model Output)', linewidth=2.5, alpha=0.9, color='#D41E2B')
    
    plt.title(f'{ticker} - Model-Assisted Time Series Reconstruction')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# =============================================================================
# 4. MAIN EXECUTION BLOCK (Workflow Example)
# =============================================================================

if __name__ == '__main__':
    print("===============================================================")
    print("!!! QUANTIFIABLE FINANCIAL TIME SERIES ANALYSIS PIPELINE v3 !!!")
    print("===============================================================")

    # --- CONFIGURATION ---
    TARGET_TICKER = "INTC"  # Intel (as a proxy for a distressed/volatile ticker)
    PEER_TICKERS = ["AMD", "NVDA", "TSM", "INTC"] # Note: INTC is target, others are peers
    START_DATE = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    # Define a 'distress window' for correlation analysis (e.g., last 30 days)
    distress_window = (
        pd.Timestamp(END_DATE) - timedelta(days=30),
        pd.Timestamp(END_DATE)
    )

    # --- STEP 1: DATA ACQUISITION ---
    target_df = download_ticker_data(TARGET_TICKER, START_DATE, END_DATE)
    
    if target_df is not None:
        # ARTIFICIAL DISTRESS SIMULATION: Inject NaN gaps into the target stock
        # We simulate "missing data" which often occurs during periods of extreme volatility/distress
        print(f"\n[SIMULATION] Injecting artificial NaN gaps into {TARGET_TICKER}...")
        target_df.loc[target_df.index[20:35], 'Close'] = np.nan
        target_df.loc[target_df.index[50:55], 'Close'] = np.nan

        # --- STEP 2: PEER SELECTION & SCORING ---
        # In a production environment, this would download all peer data and calculate metrics.
        # For this demo, we use the existing module logic.
        selected_peers, summary_stats = select_and_score_peers(
            target_df, 
            [p for p in PEER_TICKERS if p != TARGET_TICKER], 
            distress_window
        )
        print("\n--- Peer Scoring Summary ---")
        print(summary_stats)

        # --- STEP 3: RECONSTRUCTION ---
        reconstructed_series = reconstruct_gap_value(
            target_df, 
            summary_stats, 
            method='WEIGHTED_CORR'
        )

        if reconstructed_series is not None:
            # --- STEP 4: VISUALIZATION ---
            plot_reconstruction(target_df['Close'], reconstructed_series, TARGET_TICKER)
        else:
            print("❌ Reconstruction failed: No gaps to fill.")
    else:
        print("❌ Pipeline aborted: Target data unavailable.")

    print("\n===============================================================")
    print("!!! PIPELINE EXECUTION COMPLETE !!!")
    print("==========================================")