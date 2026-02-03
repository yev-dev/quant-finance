import pandas as pd
import numpy as np
import types
import yfinance as yf
import pytest

from qf.core.data import get_yahoo_data_multi


def _make_dl_return(dates, tickers):
    # Build dict-like object with per-field DataFrames
    close_df = pd.DataFrame(
        {
            tickers[0]: np.arange(1, len(dates) + 1, dtype=float),
            tickers[1]: np.arange(101, 101 + len(dates), dtype=float),
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )
    vol_df = pd.DataFrame(
        {
            tickers[0]: np.full(len(dates), 10_000, dtype=int),
            tickers[1]: np.full(len(dates), 20_000, dtype=int),
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )
    return {
        "Close": close_df,
        "Volume": vol_df,
        # Optional other fields
    }


def test_get_yahoo_data_multi_wide(monkeypatch):
    dates = pd.date_range("2024-01-01", "2024-01-05", freq="D")
    tickers = ["AAA", "BBB"]

    def fake_download(*args, **kwargs):
        return _make_dl_return(dates, tickers)

    monkeypatch.setattr(yf, "download", fake_download)

    df = get_yahoo_data_multi(
        tickers_list=tickers,
        start_date=str(dates.min().date()),
        end_date=str(dates.max().date()),
        identifiers=["Close", "Volume"],
        output_format="wide",
        column_order="ticker_field",
    )

    # Shape: 5 dates x 4 columns
    assert df.shape == (len(dates), 4)
    # Column names
    assert set(df.columns) == {"AAA_Close", "AAA_Volume", "BBB_Close", "BBB_Volume"}
    # Index equals dates (timezone-naive)
    assert df.index.equals(pd.DatetimeIndex(dates))
    # Values match the fake generator
    assert df.loc[dates[0], "AAA_Close"] == 1.0
    assert df.loc[dates[-1], "BBB_Close"] == float(101 + len(dates) - 1)
    assert df.loc[dates[0], "AAA_Volume"] == 10_000
    assert df.loc[dates[0], "BBB_Volume"] == 20_000


def test_get_yahoo_data_multi_long(monkeypatch):
    dates = pd.date_range("2024-01-01", "2024-01-05", freq="D")
    tickers = ["AAA", "BBB"]

    def fake_download(*args, **kwargs):
        return _make_dl_return(dates, tickers)

    monkeypatch.setattr(yf, "download", fake_download)

    df = get_yahoo_data_multi(
        tickers_list=tickers,
        start_date=str(dates.min().date()),
        end_date=str(dates.max().date()),
        identifiers=["Close", "Volume"],
        output_format="long",
    )

    # Expected rows = dates * tickers * fields = 5 * 2 * 2 = 20
    assert df.shape[0] == len(dates) * len(tickers) * 2
    # Columns
    assert list(df.columns) == ["Date", "Ticker", "Field", "Value"]
    # Sample assertions
    sample = df[(df["Date"] == dates[0]) & (df["Ticker"] == "AAA") & (df["Field"] == "Close")]
    assert sample.iloc[0]["Value"] == 1.0
    sample_vol = df[(df["Date"] == dates[0]) & (df["Ticker"] == "BBB") & (df["Field"] == "Volume")]
    assert sample_vol.iloc[0]["Value"] == 20_000
