
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os
import requests
from typing import List, Optional, Union


def get_instruments_data(columns=None):
    
    # S&P500 dataframe: list of tickers

    if columns is None:
        columns = ['symbol', 'security_name', 'sector', 'sub_industry', 'date_added']
    sp_df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

    sp_df.rename(columns={
        'Symbol': 'symbol',
        'Security': 'security_name',
        'GICS Sector': 'sector',
        'GICS Sub-Industry': 'sub_industry',
        'Date added': 'date_added'
    }, inplace=True)

    sp_df['symbol'] = sp_df['symbol'].str.replace('.', '-')
    
    return sp_df[columns]

def _extract_key_statistics(default_key_statistics):
    if not default_key_statistics:
        return {}
    return {
        "market_cap": default_key_statistics.get("marketCap"),
        "enterprise_value": default_key_statistics.get("enterpriseValue"),
        "trailing_pe": default_key_statistics.get("trailingPE"),
        "forward_pe": default_key_statistics.get("forwardPE"),
        "peg_ratio": default_key_statistics.get("pegRatio"),
        "price_to_book": default_key_statistics.get("priceToBook"),
        "dividend_yield": default_key_statistics.get("dividendYield"),
        "dividend_rate": default_key_statistics.get("dividendRate"),
        "fifty_two_week_high": default_key_statistics.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": default_key_statistics.get("fiftyTwoWeekLow"),
        "fifty_two_week_change": default_key_statistics.get("fiftyTwoWeekChange"),
        "beta": default_key_statistics.get("beta"),
    }

def _extract_summary_detail(summary_detail):
    if not summary_detail:
        return {}
    return {
        "previous_close": summary_detail.get("previousClose"),
        "open": summary_detail.get("open"),
        "day_high": summary_detail.get("dayHigh"),
        "day_low": summary_detail.get("dayLow"),
        "fifty_two_week_high": summary_detail.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": summary_detail.get("fiftyTwoWeekLow"),
        "fifty_two_week_change": summary_detail.get("fiftyTwoWeekChange"),
        "volume": summary_detail.get("volume"),
        "average_volume": summary_detail.get("averageVolume"),
        "market_cap": summary_detail.get("marketCap"),
        "beta": summary_detail.get("beta"),
        "dividend_yield": summary_detail.get("dividendYield"),
        "dividend_rate": summary_detail.get("dividendRate"),
        "ex_dividend_date": summary_detail.get("exDividendDate"),
    }

def _extract_esg_scores(esg_scores):
    if not esg_scores:
        return {}
    return {
        "environmental": esg_scores.get("environmentalScore"),
        "social": esg_scores.get("socialScore"),
        "governance": esg_scores.get("governanceScore"),
        "highest_controversy": esg_scores.get("highestControversy"),
        "esg_score": esg_scores.get("totalEsg"),
        "peer_count": esg_scores.get("peerCount"),
        "peer_group": esg_scores.get("peerGroup"),
        "rating_year": esg_scores.get("ratingYear"),
        "rating_month": esg_scores.get("ratingMonth"),
        "adult": esg_scores.get("adult"),
        "animal_testing": esg_scores.get("animalTesting"),
        "gambling": esg_scores.get("gambling"),
        "small_arm": esg_scores.get("smallArms"),
        "furLeather": esg_scores.get("furLeather"),
        "coal": esg_scores.get("coal"),
        "controversial_weapons": esg_scores.get("controversialWeapons"),
        "tobacco": esg_scores.get("tobacco"),
    }

def _extract_financial_data(financial_data):
    if not financial_data:
        return {}
    return {
        "current_price": financial_data.get("currentPrice"),
        "return_on_equity": financial_data.get("returnOnEquity"),
        "return_on_assets": financial_data.get("returnOnAssets"),
        "debt_to_equity": financial_data.get("debtToEquity"),
        "current_ratio": financial_data.get("currentRatio"),
        "quick_ratio": financial_data.get("quickRatio"),
        "revenue": financial_data.get("totalRevenue"),
        "ebitda": financial_data.get("ebitda"),
        "free_cash_flow": financial_data.get("freeCashflow"),
    }

def get_yahoo_query_data(tickers):
    """
    Fetch detailed stock data for given tickers from Yahoo Finance.
    Args:
        tickers (str or list): Stock ticker symbols.
    Returns:
        dict: Dictionary containing detailed stock data for each ticker."""

    from yahooquery import Ticker

    ticker_data = {}

    if isinstance(tickers, str):
        tickers = [tickers]

    for ticker in tickers:
        try:
            data = Ticker(ticker).all_modules[ticker]
            ticker_data[ticker] = {
                "symbol": ticker,
                "long_name": data["quoteType"]["longName"],
                "summary": data["assetProfile"]["longBusinessSummary"],
                "summary_detail": _extract_summary_detail(data.get("summaryDetail")),
                "esg_scores": _extract_esg_scores(data.get("esgScores")),
                "financial_data": _extract_financial_data(data.get("financialData")),
            }
        except Exception:
            print(f"Not able to collect data for {ticker}")

    return ticker_data


def get_yahoo_query_historical_data(tickers, period='10y', identifier_list=None):
    """
    Fetch historical stock data for given tickers from Yahoo Finance.
    Args:
        tickers (str or list): Stock ticker symbols.
        period (str): Period for which to fetch data (e.g., '1d', '1mo', '1y', '5y', '10y').
        identifier_list (str): Identifier for the data to fetch (e.g., ['open', 'high', 'low', 'close', 'volume', 'adjclose', 'dividends']).
    Returns:
        pd.DataFrame: DataFrame containing historical stock data."""

    from yahooquery import Ticker

    if identifier_list is None:
        identifier_list = ['open', 'high', 'low', 'close', 'volume', 'adjclose', 'dividends']

    stock_data = Ticker(tickers, asynchronous=True).history(period=period) if identifier_list is None else Ticker(tickers, asynchronous=True).history(period=period)[identifier_list]

    # stock_data = stock_data.unstack(level=0)


    stock_data = stock_data.unstack(level=0)
    stock_data.columns = stock_data.columns.droplevel(1)
    stock_data.stack().reset_index().rename(index=str, columns={"level_1": "Ticker"}).sort_values(['Ticker','Date'])

    stock_data.index = pd.to_datetime(stock_data.index, format='mixed', utc=True)
    stock_data.index = stock_data.index.tz_localize(None)

    if isinstance(tickers, str):
        tickers = [tickers]

    if isinstance(stock_data, pd.DataFrame):
        stock_data = stock_data[tickers]
    elif isinstance(stock_data, pd.Series):
        stock_data = pd.DataFrame(stock_data)

    return stock_data


def get_yahoo_data(tickers_list, start_date, end_date, identifier: str = 'Close'):
    """
    Fetch historical data for a given ticker from Yahoo Finance.
    
    Args:
        tickers_list (list): List of stock ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame containing historical stock data.
    """

    allowed = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    if identifier not in allowed:
        raise ValueError(f"Identifier must be one of {allowed}")

    # Variables instantiation
    batch_size = 20
    loop_size = int(len(tickers_list) // batch_size) + (1 if len(tickers_list) % batch_size else 0)
    yahoo_data_df: Optional[pd.DataFrame] = None

    for t in range(loop_size):  # Batch download
        m = t * batch_size
        n = (t + 1) * batch_size
        batch_list = tickers_list[m:n]
        if not batch_list:
            continue
        dl = yf.download(
            tickers=batch_list,
            start=start_date,
            end=end_date,
            interval="1d",
            group_by='column',
            auto_adjust=True,
            prepost=True,
            proxy=None
        )
        # Extract the requested field for the batch
        try:
            fld_df = dl[identifier]
        except KeyError:
            # If the identifier isn't present, skip this batch
            continue
        # Normalize index: timezone-naive, deduplicate dates, sort
        fld_df.index = pd.to_datetime(fld_df.index, format='mixed', utc=True)
        fld_df.index = fld_df.index.tz_localize(None)
        fld_df = fld_df.groupby(level=0).last().sort_index()
        # Single-ticker case may return Series; normalize to DataFrame with ticker column
        if isinstance(fld_df, pd.Series):
            fld_df = fld_df.to_frame(name=batch_list[0])
        # Accumulate across batches via column-wise concat (outer)
        if yahoo_data_df is None:
            yahoo_data_df = fld_df
        else:
            yahoo_data_df = pd.concat([yahoo_data_df, fld_df], axis=1)

    # Final sanity: sort index and columns
    if yahoo_data_df is None:
        return pd.DataFrame()
    yahoo_data_df = yahoo_data_df.sort_index()
    # Ensure columns are sorted alphabetically by ticker
    yahoo_data_df = yahoo_data_df.reindex(sorted(yahoo_data_df.columns), axis=1)
    return yahoo_data_df


def get_yahoo_data_multi(tickers_list,
                         start_date,
                         end_date,
                         identifiers: Optional[List[str]] = None,
                         output_format: str = 'wide',
                         column_order: str = 'ticker_field') -> pd.DataFrame:
    """
    Fetch multiple OHLCV fields for a list of tickers over a date range.

    Downloads in batches using yfinance and returns either:
    - wide: one row per date, columns per (ticker, field) with suffixed names
    - long (tidy): columns [Date, Ticker, Field, Value]

    Args:
        tickers_list: List of ticker symbols.
        start_date, end_date: Date strings 'YYYY-MM-DD' or Timestamps.
        identifiers: List of fields to retrieve, defaults to
            ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        output_format: 'wide' or 'long'.
        column_order: For wide format, either 'ticker_field' (default) or 'field_ticker'.

    Returns:
        DataFrame in the requested format.
    """
    allowed = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    if identifiers is None:
        identifiers = allowed
    else:
        bad = [f for f in identifiers if f not in allowed]
        if bad:
            raise ValueError(f"Unsupported identifiers: {bad}. Allowed: {allowed}")

    # Batch setup
    batch_size = 20
    loop_size = int(len(tickers_list) // batch_size) + (1 if len(tickers_list) % batch_size else 0)

    # Aggregate per-field DataFrames across batches: dict[field] -> DataFrame(date x tickers)
    field_frames: dict[str, pd.DataFrame] = {}

    for t in range(loop_size):
        m = t * batch_size
        n = (t + 1) * batch_size
        batch_list = tickers_list[m:n]
        if not batch_list:
            continue
        dl = yf.download(
            tickers=batch_list,
            start=start_date,
            end=end_date,
            interval="1d",
            group_by='column',
            auto_adjust=True,
            prepost=True,
            proxy=None
        )

        for field in identifiers:
            try:
                fld_df = dl[field]
            except KeyError:
                # field not available for this batch
                continue
            # Sanitize index: tz-naive, deduplicate, sort
            fld_df.index = pd.to_datetime(fld_df.index, format='mixed', utc=True)
            fld_df.index = fld_df.index.tz_localize(None)
            fld_df = fld_df.groupby(level=0).last().sort_index()
            # Single-ticker case -> Series
            if isinstance(fld_df, pd.Series):
                fld_df = fld_df.to_frame(name=batch_list[0])
            # Accumulate horizontally per field
            if field in field_frames:
                field_frames[field] = pd.concat([field_frames[field], fld_df], axis=1)
            else:
                field_frames[field] = fld_df

    # If nothing collected
    if not field_frames:
        return pd.DataFrame()

    # Ensure columns are unique tickers per field and sorted
    for field, df in field_frames.items():
        # Deduplicate columns if yfinance produced duplicates accidentally
        df = df.loc[:, ~df.columns.duplicated()].copy()
        df = df.sort_index()
        df = df.reindex(sorted(df.columns), axis=1)
        field_frames[field] = df

    if output_format == 'wide':
        # Build suffixed column names and concat
        wide_frames = []
        for field, df in field_frames.items():
            if column_order == 'field_ticker':
                renamed = {col: f"{field}_{col}" for col in df.columns}
            else:  # ticker_field
                renamed = {col: f"{col}_{field}" for col in df.columns}
            wf = df.rename(columns=renamed)
            wide_frames.append(wf)
        wide_df = pd.concat(wide_frames, axis=1).sort_index()
        # Sort columns alphabetically
        wide_df = wide_df.reindex(sorted(wide_df.columns), axis=1)
        return wide_df
    elif output_format == 'long':
        # Stack each field into long format with Field column
        long_frames = []
        for field, df in field_frames.items():
            lf = (
                df.stack()
                  .reset_index()
                  .rename(columns={'Date': 'Date', 0: 'Value', 'level_1': 'Ticker'})
            )
            # In case index name isn't 'Date'
            if 'Date' not in lf.columns:
                # The first column from reset_index() is the date index
                date_col = lf.columns[0]
                lf = lf.rename(columns={date_col: 'Date'})
            lf['Field'] = field
            long_frames.append(lf[['Date', 'Ticker', 'Field', 'Value']])
        long_df = pd.concat(long_frames, axis=0, ignore_index=True)
        long_df = long_df.sort_values(['Date', 'Ticker', 'Field']).reset_index(drop=True)
        return long_df
    else:
        raise ValueError("output_format must be 'wide' or 'long'")


def get_yahoo_query_full_data(tickers = None, features_list=None):


    if tickers is None:
        # Get S&P 500 tickers if no tickers are provided   
        tickers = get_instruments_data() 


    if isinstance(tickers, pd.DataFrame):
        tickers = tickers['symbol'].tolist()


    full_data_dict = get_yahoo_query_data(tickers)


    financial_data_df = pd.DataFrame.from_dict(
        {ticker: data['financial_data'] for ticker, data in full_data_dict.items()},
        orient='index'
    )

    financial_data_df.index.name = 'symbol'
    financial_data_df.reset_index(inplace=True)

    summary_detail_df = pd.DataFrame.from_dict(
        {ticker: data['summary_detail'] for ticker, data in full_data_dict.items()},
        orient='index'
    )
    summary_detail_df.index.name = 'symbol'
    summary_detail_df.reset_index(inplace=True) 

    esg_scores_df = pd.DataFrame.from_dict(
        {ticker: data['esg_scores'] for ticker, data in full_data_dict.items()},
        orient='index'
    )
    esg_scores_df.index.name = 'symbol'
    esg_scores_df.reset_index(inplace=True)


    return_data_df = pd.DataFrame.from_dict(
        {ticker: {'symbol': data.get('symbol'), 'long_name': data.get('long_name'), 'summary': data.get('summary')}
         for ticker, data in full_data_dict.items()},
        orient='index'
    )
    return_data_df = summary_detail_df.merge(return_data_df, on='symbol', how='left', validate='one_to_one')
    return_data_df = financial_data_df.merge(return_data_df, on='symbol', how='left', validate='one_to_one')    
    return_data_df = return_data_df.merge(esg_scores_df, on='symbol', how='left', validate='one_to_one')

    if features_list:
        return return_data_df[features_list]

    return return_data_df


def _get_historical_shares_outstanding(ticker: str,
                                       start_date: Optional[Union[str, pd.Timestamp]] = None,
                                       end_date: Optional[Union[str, pd.Timestamp]] = None) -> pd.Series:
    """
    Retrieve historical shares outstanding for a given ticker.

    Tries yfinance Ticker.get_shares_full (quarterly granularity). If unavailable,
    falls back to a constant series using current sharesOutstanding from Ticker.info.

    Returns a Series indexed by date with shares outstanding as float.
    """
    t = yf.Ticker(ticker)
    shares = None
    try:
        # get_shares_full typically returns a DataFrame with columns like 'SharesOutstanding'
        df = t.get_shares_full(start=start_date, end=end_date)
        if isinstance(df, pd.DataFrame):
            # Standardize column name and ensure datetime index (timezone-naive)
            col = 'SharesOutstanding' if 'SharesOutstanding' in df.columns else df.columns[0]
            if 'Date' in df.columns:
                df = df.set_index('Date')
            # Normalize to UTC then drop timezone to match price index
            df.index = pd.to_datetime(df.index, utc=True)
            df.index = df.index.tz_localize(None)
            # Deduplicate index by keeping the last entry per date
            shares = (df[col]
                      .astype(float)
                      .groupby(level=0)
                      .last()
                      .sort_index())
            # Normalize to midnight to align with daily price index
            shares.index = shares.index.normalize()
        elif isinstance(df, pd.Series):
            # Normalize series index to timezone-naive
            idx = pd.to_datetime(df.index, utc=True)
            idx = idx.tz_localize(None)
            s = pd.Series(df.values, index=idx, name=df.name)
            shares = (s.astype(float)
                      .groupby(level=0)
                      .last()
                      .sort_index())
            shares.index = shares.index.normalize()
    except Exception:
        shares = None

    if shares is None or shares.empty:
        # Fallback to static shares outstanding
        try:
            info = t.info
            so = info.get('sharesOutstanding')
            if so is not None:
                # Build a constant series across the requested date range
                if start_date is None or end_date is None:
                    # If no range, just return a single-point series
                    idx = pd.Index([pd.Timestamp.today().normalize()], name='Date')
                else:
                    idx = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq='D')
                shares = pd.Series(float(so), index=idx, name='SharesOutstanding')
            else:
                # Ensure we return an empty Series rather than None
                shares = pd.Series(dtype=float)
        except Exception:
            shares = pd.Series(dtype=float)

    return shares


def _compute_market_cap(price: pd.Series, shares: pd.Series) -> pd.Series:
    """
    Compute market cap time series by aligning price and shares series.

    - Reindexes shares to price index with forward/backward fill.
    - Market cap = price * shares.
    Returns a Series with the same index as price.
    """
    if price.empty:
        return pd.Series(dtype=float)
    # Ensure unique, sorted indices to avoid reindex errors
    price = (price.astype(float)
             .groupby(level=0)
             .last()
             .sort_index())
    # Handle None gracefully
    if shares is None:
        shares = pd.Series(dtype=float)
    shares = (shares.astype(float)
              .groupby(level=0)
              .last()
              .sort_index())

    # Align shares to price timeline
    price_index_unique = price.index.unique()
    shares_aligned = shares.reindex(price_index_unique).ffill().bfill()
    mc = (price * shares_aligned).rename('market_cap')

    return mc


def get_historical_market_cap(tickers_input: Union[str, List[str]],
                              start_date: Union[str, pd.Timestamp],
                              end_date: Union[str, pd.Timestamp],
                              price_identifier: str = 'Close',
                              batch_size: int = 15) -> pd.DataFrame:
    """
    Fetch historical market capitalization for one or more tickers.

    Market cap is computed as price * shares outstanding.
    - Price: yfinance download (daily, auto-adjusted) using `price_identifier` column.
    - Shares: yfinance Ticker.get_shares_full (quarterly), forward-filled to daily;
      fallback to constant sharesOutstanding when historical series not available.

    Args:
        tickers: Single ticker or list of tickers.
        start_date: Start date (YYYY-MM-DD or Timestamp).
        end_date: End date (YYYY-MM-DD or Timestamp).
        price_identifier: Column to use from downloaded prices (default 'Close').
        batch_size: Number of tickers per batch for price download.

    Returns:
        DataFrame indexed by date with columns per ticker containing market cap.
    """
    if isinstance(tickers_input, str):
        tickers_list: List[str] = [tickers_input]
    else:
        tickers_list = list(tickers_input)

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    mc_df = pd.DataFrame()
    loop_size = int(len(tickers_list) // batch_size) + (1 if len(tickers_list) % batch_size else 0)

    for t in range(loop_size):
        m = t * batch_size
        n = (t + 1) * batch_size
        batch_list = tickers_list[m:n]
        if not batch_list:
            continue
        try:
            prices = yf.download(
                tickers=batch_list,
                start=start_date,
                end=end_date,
                interval="1d",
                group_by='column',
                auto_adjust=True,
                prepost=True,
                proxy=None
            )[price_identifier]
        except (KeyError, ValueError, OSError, TimeoutError, requests.exceptions.RequestException) as e:
            # Catch specific likely errors from yfinance/download or network requests
            print(f"Price download failed for batch {t+1}: {e}")
            continue
            

        # Ensure datetime index without timezone and deduplicate by last
        prices.index = pd.to_datetime(prices.index, format='mixed', utc=True)
        prices.index = prices.index.tz_localize(None)
        prices = prices.groupby(level=0).last().sort_index()

        # If single ticker, prices may be a Series; normalize to DataFrame
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=batch_list[0])

        batch_mc = {}
        for ticker in batch_list:
            if ticker not in prices.columns:
                print(f"Skipping {ticker}: price column not found")  
                continue
            price_series = prices[ticker].dropna()
            shares_series = _get_historical_shares_outstanding(ticker, start_date, end_date)
            if shares_series is None or shares_series.empty:
                print(f"Skipping {ticker}: no shares outstanding data")
                continue
            mc_series = _compute_market_cap(price_series, shares_series)
            batch_mc[ticker] = mc_series

        if batch_mc:
            batch_df = pd.DataFrame(batch_mc)
            mc_df = mc_df.join(batch_df, how='outer') if not mc_df.empty else batch_df
            # Deduplicate index after join
            mc_df = mc_df.groupby(level=0).last().sort_index()

    return mc_df.sort_index()


def export_market_cap_to_csv(mc_df: pd.DataFrame, path: str, include_index: bool = True) -> str:
    """
    Export market cap DataFrame to CSV.

    - Ensures destination directory exists.
    - Returns the absolute file path written.
    """
    if mc_df is None or mc_df.empty:
        raise ValueError("market cap DataFrame is empty; nothing to export")
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    mc_df.to_csv(abs_path, index=include_index)
    return abs_path


    # Force requests to use the base components URL (e.g. https://finance.yahoo.com/quote/%5ESBF120/components/)
    _original_requests_get = requests.get

    def _requests_get_components_base(url, *args, **kwargs):
        try:
            if isinstance(url, str) and url.startswith("https://finance.yahoo.com/quote/") and "/components?p=" in url:
                base = url.split("?", 1)[0]
                if not base.endswith("/"):
                    base = base + "/"
                url = base
        except Exception:
            pass
        return _original_requests_get(url, *args, **kwargs)

    requests.get = _requests_get_components_base

def get_yahoo_components_for_index_symbol(index_symbol: str,
                                            columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Retrieve index components from Yahoo Finance for a given index symbol.

    Args:
        index_symbol: Index symbol as used on Yahoo Finance (e.g. "^GSPC", "^DJI").
        columns: Optional[List[str]] = None

    Returns:
        pd.DataFrame of components (empty DataFrame on failure).
    """
    from requests.utils import requote_uri

    if not index_symbol or not isinstance(index_symbol, str):
        return pd.DataFrame()

    try:    
        # Ensure symbol is safely embedded in the URL (handles '^' etc.)
        quoted = requote_uri(index_symbol)
        url = f"https://finance.yahoo.com/quote/{quoted}/components?p={quoted}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible)"}
        # https://finance.yahoo.com/quote/%5ESBF120/components/
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception:
        return pd.DataFrame()

    try:
        tables = pd.read_html(resp.text)
    except Exception:
        return pd.DataFrame()

    if not tables:
        return pd.DataFrame()

    # Prefer table that looks like a components table (has a Symbol/Ticker column)
    comp_df = None
    for t in tables:
        cols_l = [str(c).lower() for c in t.columns]
        if any(c in cols_l for c in ("symbol", "ticker")) or any("symbol" in str(c).lower() for c in t.columns):
            comp_df = t.copy()
            break
    if comp_df is None:
        comp_df = tables[0].copy()

    # Normalize common column names
    rename_map = {}
    for c in comp_df.columns:
        lc = str(c).lower()
        if "symbol" in lc or "ticker" in lc:
            rename_map[c] = "symbol"
        elif "name" == lc or "company" in lc:
            rename_map[c] = "name"
        elif "weight" in lc:
            rename_map[c] = "weight"
        elif "sector" in lc:
            rename_map[c] = "sector"
        elif "industry" in lc or "sub-industry" in lc or "sub industry" in lc:
            rename_map[c] = "industry"

    comp_df.rename(columns=rename_map, inplace=True)

    # Basic cleaning: symbol formatting and numeric conversion for weight
    if "symbol" in comp_df.columns:
        comp_df["symbol"] = comp_df["symbol"].astype(str).str.replace(".", "-", regex=False).str.strip()

    if "weight" in comp_df.columns:
        comp_df["weight"] = (
            comp_df["weight"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        comp_df["weight"] = pd.to_numeric(comp_df["weight"], errors="coerce") / 100.0

    # Trim whitespace from string columns
    for col in comp_df.select_dtypes(include="object").columns:
        comp_df[col] = comp_df[col].str.strip()

    # Optionally pick subset of columns
    if columns:
        available = [c for c in columns if c in comp_df.columns]
        comp_df = comp_df[available]

    # Ensure index is simple integer index
    comp_df = comp_df.reset_index(drop=True)

    return comp_df

def plot_time_series(series_df: Union[pd.DataFrame, pd.Series],
                    tickers: Union[str, List[str]],
                    start_date: Optional[Union[str, pd.Timestamp]] = None,
                    end_date: Optional[Union[str, pd.Timestamp]] = None,
                    normalize: bool = False,
                    log_scale: bool = True,
                    title: Optional[str] = None,
                    ylabel: Optional[str] = None):
    """
    Plot time series for given tickers from any DataFrame (or Series) with a date index.

    Args:
        mc_df: DataFrame (or Series) with date index. Columns should correspond to tickers/series names.
        tickers: Single ticker or list of tickers to plot (column names or the Series name).
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        normalize: If True, normalize each series to 1.0 at its first non-NaN value in the plot window.
        log_scale: If True, use logarithmic y-axis.
        title: Optional plot title.
        ylabel: Optional y-axis label (defaults to "Value").

    Returns:
        (fig, ax): Matplotlib figure and axes objects.
    """
    if series_df is None or (isinstance(series_df, pd.DataFrame) and series_df.empty):
        raise ValueError("input time series is empty; cannot plot")

    # Normalize input to a DataFrame
    if isinstance(series_df, pd.Series):
        # Determine column name for Series
        if isinstance(tickers, str):
            col_name = tickers
        else:
            col_name = series_df.name or 'series'
        df = series_df.to_frame(name=col_name)
    else:
        df = series_df.copy()

    if isinstance(tickers, str):
        tickers_list: List[str] = [tickers]
    else:
        tickers_list = list(tickers)

    # Filter date range
    if start_date is not None:
        df = df.loc[pd.to_datetime(start_date):]
    if end_date is not None:
        df = df.loc[:pd.to_datetime(end_date)]

    # Select columns present
    present = [t for t in tickers_list if t in df.columns]
    missing = [t for t in tickers_list if t not in df.columns]
    if missing:
        print(f"Warning: missing series not in DataFrame: {missing}")
    if not present:
        raise ValueError("none of the requested series are present in the input DataFrame")

    plot_df = df[present]
    if normalize:
        # Normalize each series by its first non-NaN value
        plot_df = plot_df.apply(lambda s: s / s.dropna().iloc[0] if s.dropna().size else s)

    try:
        import seaborn as sns
        sns.set(style="whitegrid")
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(ax=ax)
    if log_scale:
        ax.set_yscale('log')
    ax.set_xlabel("Date")
    ylab = (ylabel or "Value") + (" (normalized)" if normalize else "")
    ax.set_ylabel(ylab)
    if title is None:
        title = f"Time Series: {', '.join(present)}"
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # instruments_df = get_instruments_data()
    # instruments_df.to_csv('/Users/yevgeniy/Development/projects/quant/quant-finance/qf/clustering/data/instruments.csv', index=False)
    # instruments_df = pd.read_csv(r'/Users/yevgeniy/Development/Projects/FinancialEngineering/quant-finance/qf/clustering/data/instruments.csv')
    # tickers = instruments_df['symbol'].tolist()
    # market_cap_hist = get_historical_market_cap(
    #     tickers_input=tickers,
    #     start_date='2010-01-01',
    #     end_date='2024-06-01',
    #     price_identifier='Adj Close',
    #     batch_size=20
    # )
    # saving_path_mc = r'/Users/yevgeniy/Development/projects/quant/quant-finance/qf/clustering/data/market_cap_data.csv'
    # market_cap_hist.to_csv(saving_path_mc, index=True)
    # print(f"Market cap data saved to {saving_path_mc}")
    # fundamentals_df = get_yahoo_query_full_data(tickers=tickers)
    # saving_path = r'/Users/yevgeniy/Development/projects/quant/quant-finance/qf/clustering/data/fundamentals_data.csv'
    # fundamentals_df.to_csv(saving_path, index=False)
    # print(f"Fundamentals data saved to {saving_path}")
    # print(fundamentals_df.head())
    from qf.core.data import get_historical_market_cap

    mc = get_historical_market_cap(
        tickers_input=['AAPL','MSFT','GOOGL'],
        start_date='2021-01-01',
        end_date='2023-12-31'
    )
    print(mc.head())


    


    