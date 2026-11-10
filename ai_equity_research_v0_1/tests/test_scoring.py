import numpy as np
import pandas as pd
import pytest

from src.scoring import percentile_score, rank_stocks


def test_percentile_scores_are_bounded_and_ordered():
    scores = percentile_score(pd.Series({"A": 10.0, "B": 20.0, "C": 30.0}))
    assert scores.between(0, 100).all()
    assert scores["A"] < scores["B"] < scores["C"]


def test_missing_values_remain_missing():
    scores = percentile_score(pd.Series({"A": 10.0, "B": float("nan"), "C": 30.0}))
    assert pd.isna(scores["B"])


def test_empty_series():
    assert percentile_score(pd.Series(dtype=float)).empty


def test_rank_stocks_returns_descending_scores_and_expected_formula():
    dates = pd.bdate_range("2020-01-01", periods=320)
    prices = pd.DataFrame({
        "AAA": [100 + i * 0.5 for i in range(320)],
        "BBB": [100 + i * 0.3 for i in range(320)],
        "CCC": [100 + i * 0.1 for i in range(320)],
    }, index=dates)
    returns = prices.pct_change(fill_method=None)
    features = {
        "momentum_12_1": prices.shift(21) / prices.shift(252) - 1.0,
        "momentum_6m": prices / prices.shift(126) - 1.0,
        "volatility_63d": returns.rolling(63, min_periods=50).std() * (252 ** 0.5),
        "daily_returns": returns,
    }
    ranking = rank_stocks(features, min_history_days=252)
    assert ranking["score"].is_monotonic_decreasing
    assert ranking.index[0] == "AAA"
    assert np.allclose(ranking["score"], 0.75 * ranking["momentum_score"] + 0.25 * ranking["risk_score"])


def test_rank_stocks_rejects_missing_date():
    dates = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame({"AAA": range(100, 400)}, index=dates, dtype=float)
    returns = prices.pct_change(fill_method=None)
    features = {
        "momentum_12_1": prices.shift(21) / prices.shift(252) - 1.0,
        "volatility_63d": returns.rolling(63, min_periods=50).std() * (252 ** 0.5),
        "daily_returns": returns,
    }
    with pytest.raises(ValueError, match="No feature observations"):
        rank_stocks(features, as_of="2025-01-01")
