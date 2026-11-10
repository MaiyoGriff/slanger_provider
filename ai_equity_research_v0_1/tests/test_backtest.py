import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest, run_strategy_comparison, slice_evaluation_period
from src.features import calculate_features


def synthetic_prices(periods=800):
    dates = pd.bdate_range("2018-01-01", periods=periods)
    x = np.arange(len(dates), dtype=float)
    return pd.DataFrame({
        "AAA": 100 * (1.0010 ** x),
        "BBB": 100 * (1.0005 ** x),
        "CCC": 100 * (1.0002 ** x),
        "SPY": 100 * (1.0004 ** x),
    }, index=dates)


def test_backtest_returns_expected_columns_and_finite_equity():
    prices = synthetic_prices()
    result = run_backtest(prices, calculate_features(prices), top_n=2, transaction_cost_bps=5)
    expected = {"strategy_return", "gross_return", "transaction_cost", "turnover", "benchmark_return", "strategy_equity", "benchmark_equity"}
    assert expected.issubset(result.columns)
    assert result.index.equals(prices.index)
    assert np.isfinite(result["strategy_equity"].iloc[-1])
    assert np.isfinite(result["benchmark_equity"].iloc[-1])
    assert (result["transaction_cost"] >= 0).all()


def test_comparison_has_hedge_momentum_and_benchmark():
    prices = synthetic_prices()
    result = run_strategy_comparison(prices, calculate_features(prices), top_n=2)
    assert {"hedge_return", "momentum_return", "benchmark_return", "hedge_equity", "momentum_equity", "benchmark_equity"}.issubset(result.columns)
    assert result.index.equals(prices.index)


def test_signal_does_not_earn_return_before_next_close_execution():
    prices = synthetic_prices()
    result = run_backtest(prices, calculate_features(prices), top_n=1)
    # Before the first eligible signal/holding period, no position should be earned.
    first_nonzero = result.index[result["turnover"] > 0][0]
    assert result.loc[first_nonzero, "transaction_cost"] > 0
    next_date = result.index[result.index.get_loc(first_nonzero) + 1]
    assert pd.isna(result.loc[first_nonzero, "gross_return"]) or result.loc[first_nonzero, "gross_return"] == 0
    assert result.loc[next_date, "gross_return"] >= 0


def test_period_slice_is_chronological_and_inclusive():
    prices = synthetic_prices()
    result = run_strategy_comparison(prices, calculate_features(prices), top_n=2)
    sliced = slice_evaluation_period(result, "2020-01-01", "2020-12-31")
    assert not sliced.empty
    assert sliced.index.min() >= pd.Timestamp("2020-01-01")
    assert sliced.index.max() <= pd.Timestamp("2020-12-31")


def test_backtest_validates_arguments():
    prices = synthetic_prices(20)
    features = calculate_features(prices)
    with pytest.raises(ValueError, match="top_n"):
        run_backtest(prices, features, top_n=0)
    with pytest.raises(ValueError, match="transaction_cost"):
        run_backtest(prices, features, transaction_cost_bps=-1)
