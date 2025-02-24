import pytest
import pandas as pd
import numpy as np
from qf.timeseries.backtest import Backtest

@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    actual = pd.Series(np.random.randn(100), index=dates)
    forecast = pd.Series(np.random.randn(100), index=dates)
    alpha = 0.05
    return actual, forecast, alpha

def test_hit_series(sample_data):
    actual, forecast, alpha = sample_data
    bt = Backtest(actual, forecast, alpha)
    hits = bt.hit_series()
    assert len(hits) == len(actual)
    assert all((hits == 0) | (hits == 1))

def test_number_of_hits(sample_data):
    actual, forecast, alpha = sample_data
    bt = Backtest(actual, forecast, alpha)
    hits = bt.hit_series()
    assert bt.number_of_hits() == hits.sum()

def test_hit_rate(sample_data):
    actual, forecast, alpha = sample_data
    bt = Backtest(actual, forecast, alpha)
    hits = bt.hit_series()
    assert bt.hit_rate() == hits.mean()

def test_expected_hits(sample_data):
    actual, forecast, alpha = sample_data
    bt = Backtest(actual, forecast, alpha)
    assert bt.expected_hits() == len(actual) * alpha

def test_duration_series(sample_data):
    actual, forecast, alpha = sample_data
    bt = Backtest(actual, forecast, alpha)
    durations = bt.duration_series()
    assert isinstance(durations, np.ndarray)

def test_tick_loss(sample_data):
    actual, forecast, alpha = sample_data
    bt = Backtest(actual, forecast, alpha)
    loss = bt.tick_loss()
    assert isinstance(loss, float)

def test_smooth_loss(sample_data):
    actual, forecast, alpha = sample_data
    bt = Backtest(actual, forecast, alpha)
    loss = bt.smooth_loss()
    assert isinstance(loss, float)

def test_quadratic_loss(sample_data):
    actual, forecast, alpha = sample_data
    bt = Backtest(actual, forecast, alpha)
    loss = bt.quadratic_loss()
    assert isinstance(loss, float)

def test_firm_loss(sample_data):
    actual, forecast, alpha = sample_data
    bt = Backtest(actual, forecast, alpha)
    loss = bt.firm_loss()
    assert isinstance(loss, float)

def test_lr_bt(sample_data):
    actual, forecast, alpha = sample_data
    bt = Backtest(actual, forecast, alpha)
    df = bt.lr_bt()
    assert isinstance(df, pd.DataFrame)
    assert "Statistic" in df.columns
    assert "p-value" in df.columns

def test_dq_bt(sample_data):
    actual, forecast, alpha = sample_data
    bt = Backtest(actual, forecast, alpha)
    series = bt.dq_bt()
    assert isinstance(series, pd.Series)
    assert "Statistic" in series.index
    assert "p-value" in series.index