import numpy as np
import pandas as pd
import pytest

from src.report import performance_summary


def test_performance_summary_includes_strategy_and_benchmark_metrics():
    dates = pd.bdate_range("2022-01-03", periods=260)
    strategy = pd.Series([0.001] * 130 + [-0.002] * 130, index=dates)
    benchmark = pd.Series([0.0005] * 260, index=dates)
    bt = pd.DataFrame({
        "strategy_return": strategy,
        "benchmark_return": benchmark,
        "transaction_cost": [0.0001] * 260,
        "turnover": [0.2] * 260,
    })
    summary = performance_summary(bt)
    assert summary["strategy_observations"] == 260
    assert summary["benchmark_observations"] == 260
    assert summary["strategy_maximum_drawdown"] < 0
    assert summary["strategy_annualized_volatility"] > 0
    assert summary["benchmark_annualized_volatility"] == pytest.approx(0.0)
    assert summary["total_transaction_cost_return_units"] == pytest.approx(0.026)
    assert "strategy_sortino" in summary
    assert "strategy_calmar" in summary


def test_performance_summary_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        performance_summary(pd.DataFrame({"strategy_return": [0.01]}))


def test_performance_summary_empty_input():
    with pytest.raises(ValueError, match="empty"):
        performance_summary(pd.DataFrame(columns=["strategy_return", "benchmark_return"]))
