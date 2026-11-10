"""Out-of-sample validation and benchmark reporting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import period_performance, run_strategy_comparison
from src.features import calculate_features


def forward_returns(prices: pd.DataFrame, horizon_days: int = 126) -> pd.DataFrame:
    """Calculate future simple returns, aligned to the signal date."""
    if prices.empty:
        raise ValueError("prices is empty")
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")
    prices = prices.sort_index()
    return prices.shift(-horizon_days).div(prices).sub(1.0)


def rank_ic_series(
    prices: pd.DataFrame,
    features: dict[str, pd.DataFrame] | None = None,
    horizon_days: int = 126,
    min_history_days: int = 252,
    benchmark: str = "SPY",
) -> pd.Series:
    """Monthly Spearman correlation of Hedge score vs subsequent stock returns."""
    from src.scoring import rank_stocks

    if benchmark not in prices.columns:
        raise ValueError(f"Missing benchmark: {benchmark}")
    prices = prices.sort_index()
    features = features or calculate_features(prices)
    future = forward_returns(prices, horizon_days)
    month_ends = pd.Series(prices.index, index=prices.index).groupby(prices.index.to_period("M")).last().tolist()
    values: dict[pd.Timestamp, float] = {}
    for date in month_ends:
        try:
            ranking = rank_stocks(features, as_of=date, min_history_days=min_history_days)
        except ValueError:
            continue
        scores = ranking["score"].drop(index=benchmark, errors="ignore")
        outcomes = future.loc[date].drop(index=benchmark, errors="ignore")
        common = scores.index.intersection(outcomes.dropna().index)
        if len(common) < 3:
            continue
        ic = scores.loc[common].corr(outcomes.loc[common], method="spearman")
        if pd.notna(ic):
            values[pd.Timestamp(date)] = float(ic)
    return pd.Series(values, name="rank_ic", dtype=float).sort_index()


def summarize_rank_ic(ic: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(ic, errors="coerce").dropna()
    if values.empty:
        return {"ic_observations": 0, "mean_rank_ic": float("nan"), "median_rank_ic": float("nan"), "positive_rank_ic_fraction": float("nan")}
    return {
        "ic_observations": int(len(values)),
        "mean_rank_ic": float(values.mean()),
        "median_rank_ic": float(values.median()),
        "positive_rank_ic_fraction": float((values > 0).mean()),
    }


def run_oos_validation(
    prices: pd.DataFrame,
    benchmark: str = "SPY",
    top_n: int = 10,
    transaction_cost_bps: float = 5.0,
    min_history_days: int = 252,
    evaluation_start: str = "2024-01-01",
    evaluation_end: str | None = None,
    periods: dict[str, tuple[str, str | None]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run strategies on full history, evaluate only dates in held-out window.

    Features and portfolio signals use only information available at each date.
    The evaluation window should be chosen before inspecting its results.
    """
    features = calculate_features(prices)
    full = run_strategy_comparison(
        prices, features, benchmark, top_n, transaction_cost_bps, min_history_days
    )
    end_date = pd.Timestamp(evaluation_end) if evaluation_end else full.index.max()
    start_date = pd.Timestamp(evaluation_start)
    oos = full.loc[(full.index >= start_date) & (full.index <= end_date)].copy()
    if oos.empty:
        raise ValueError("Out-of-sample window contains no rows; check dates and price history")
    periods = periods or {"out_of_sample": (evaluation_start, evaluation_end)}
    period_table = period_performance(full, periods)
    ic = rank_ic_series(prices, features, min_history_days=min_history_days, benchmark=benchmark)
    ic = ic.loc[(ic.index >= start_date) & (ic.index <= end_date)]
    ic_summary = summarize_rank_ic(ic)
    return full, period_table, {"evaluation_start": str(start_date.date()), "evaluation_end": str(end_date.date()), **ic_summary}
