"""Backtesting utilities for Hedge.

Signals are formed at a month-end close. A target is assumed executed at the
next available close; transaction costs are charged on that execution date,
and the new portfolio earns close-to-close returns only after execution.
This conservative convention avoids using a signal to earn the return that
occurred before the assumed trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.scoring import rank_stocks


def _validate_inputs(
    prices: pd.DataFrame,
    benchmark: str,
    top_n: int,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("prices must not be empty")
    if benchmark not in prices.columns:
        raise ValueError(f"Missing benchmark: {benchmark}")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps cannot be negative")
    if not prices.index.is_monotonic_increasing:
        prices = prices.sort_index()
    if prices.index.has_duplicates:
        raise ValueError("prices index contains duplicate dates")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices index must be a DatetimeIndex")
    return prices.astype(float)


def _monthly_signal_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    dates = pd.Series(index, index=index)
    return dates.groupby(index.to_period("M")).last().tolist()


def _select_tickers(
    features: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    benchmark: str,
    top_n: int,
    min_history_days: int,
    strategy: str,
) -> list[str]:
    if strategy == "hedge":
        ranking = rank_stocks(features, as_of=date, min_history_days=min_history_days)
        candidates = ranking.index.tolist()
    elif strategy == "momentum":
        momentum = features["momentum_12_1"]
        returns = features["daily_returns"]
        if date not in momentum.index:
            return []
        history = returns.loc[:date].notna().sum()
        eligible = momentum.loc[date].dropna()
        eligible = eligible[history.reindex(eligible.index).fillna(0) >= min_history_days]
        candidates = eligible.sort_values(ascending=False, kind="mergesort").index.tolist()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return [ticker for ticker in candidates if ticker != benchmark][:top_n]


def _run_one_strategy(
    prices: pd.DataFrame,
    features: dict[str, pd.DataFrame],
    benchmark: str,
    top_n: int,
    transaction_cost_bps: float,
    min_history_days: int,
    strategy: str,
) -> pd.DataFrame:
    """Run one long-only, equal-weight monthly strategy."""
    dates = prices.index
    daily_returns = prices.pct_change(fill_method=None)
    targets = pd.DataFrame(np.nan, index=dates, columns=prices.columns, dtype=float)
    costs = pd.Series(0.0, index=dates, dtype=float)
    turnover = pd.Series(0.0, index=dates, dtype=float)
    target_weights = pd.Series(0.0, index=prices.columns, dtype=float)
    cost_rate = transaction_cost_bps / 10_000.0

    for signal_date in _monthly_signal_dates(dates):
        signal_pos = dates.get_loc(signal_date)
        # Need a later session on which to execute the signal.
        if signal_pos + 1 >= len(dates):
            continue
        try:
            selected = _select_tickers(
                features, signal_date, benchmark, top_n, min_history_days, strategy
            )
        except (ValueError, KeyError):
            continue
        if not selected:
            continue

        new_weights = pd.Series(0.0, index=prices.columns, dtype=float)
        new_weights.loc[selected] = 1.0 / len(selected)
        execution_date = dates[signal_pos + 1]
        # Target changes and costs are booked on the assumed execution date.
        day_turnover = float((new_weights - target_weights).abs().sum())
        turnover.loc[execution_date] += day_turnover
        costs.loc[execution_date] += day_turnover * cost_rate
        targets.loc[execution_date] = new_weights
        target_weights = new_weights

    # Carry the executed holdings forward. The shift ensures a trade executed
    # at the close on day t earns the close-to-close return ending on day t+1.
    executed_weights = targets.ffill().fillna(0.0)
    held_weights = executed_weights.shift(1).fillna(0.0)
    # Missing returns for held assets are not silently treated as zero: flag the
    # affected portfolio day as NaN so data-quality problems remain visible.
    asset_returns = daily_returns
    weighted_returns = held_weights * asset_returns
    missing_held_return = ((held_weights.abs() > 1e-12) & asset_returns.isna()).any(axis=1)
    gross = weighted_returns.sum(axis=1, min_count=1)
    gross.loc[held_weights.abs().sum(axis=1) == 0] = 0.0
    gross.loc[missing_held_return] = np.nan
    net = gross - costs

    result = pd.DataFrame(
        {
            f"{strategy}_gross_return": gross,
            f"{strategy}_transaction_cost": costs,
            f"{strategy}_turnover": turnover,
            f"{strategy}_return": net,
        },
        index=dates,
    )
    return result


def run_backtest(
    prices: pd.DataFrame,
    features: dict[str, pd.DataFrame],
    benchmark: str = "SPY",
    top_n: int = 10,
    transaction_cost_bps: float = 5.0,
    min_history_days: int = 252,
) -> pd.DataFrame:
    """Run Hedge strategy and include SPY buy-and-hold returns.

    Compatibility wrapper retaining the v0.2 output column names. Monthly
    signals are executed on the next available close and costs are charged on
    that execution date. Missing returns on held assets are kept as NaN.
    """
    prices = _validate_inputs(prices, benchmark, top_n, transaction_cost_bps)
    hedge = _run_one_strategy(
        prices, features, benchmark, top_n, transaction_cost_bps,
        min_history_days, "hedge",
    )
    returns = prices.pct_change(fill_method=None)
    result = pd.DataFrame(index=prices.index)
    result["strategy_return"] = hedge["hedge_return"]
    result["gross_return"] = hedge["hedge_gross_return"]
    result["transaction_cost"] = hedge["hedge_transaction_cost"]
    result["turnover"] = hedge["hedge_turnover"]
    result["benchmark_return"] = returns[benchmark]
    result.loc[returns[benchmark].isna(), "benchmark_return"] = np.nan
    result["strategy_equity"] = (1.0 + result["strategy_return"].fillna(0.0)).cumprod()
    result["benchmark_equity"] = (1.0 + result["benchmark_return"].fillna(0.0)).cumprod()
    return result


def run_strategy_comparison(
    prices: pd.DataFrame,
    features: dict[str, pd.DataFrame],
    benchmark: str = "SPY",
    top_n: int = 10,
    transaction_cost_bps: float = 5.0,
    min_history_days: int = 252,
) -> pd.DataFrame:
    """Compare Hedge, momentum-only and buy-and-hold benchmark daily returns."""
    prices = _validate_inputs(prices, benchmark, top_n, transaction_cost_bps)
    hedge = _run_one_strategy(
        prices, features, benchmark, top_n, transaction_cost_bps,
        min_history_days, "hedge",
    )
    momentum = _run_one_strategy(
        prices, features, benchmark, top_n, transaction_cost_bps,
        min_history_days, "momentum",
    )
    returns = prices.pct_change(fill_method=None)
    result = pd.concat([hedge, momentum], axis=1)
    result["benchmark_return"] = returns[benchmark]
    for col in ("hedge_return", "momentum_return", "benchmark_return"):
        result[f"{col.removesuffix('_return')}_equity"] = (1.0 + result[col].fillna(0.0)).cumprod()
    return result


def slice_evaluation_period(
    results: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Slice a precomputed backtest to an evaluation period.

    The full history should be used to calculate features and signals so the
    strategy has its required lookback at the start date. Only the returned
    period is used to calculate evaluation metrics.
    """
    start_date = pd.Timestamp(start)
    if end is None:
        return results.loc[results.index >= start_date].copy()
    end_date = pd.Timestamp(end)
    if end_date < start_date:
        raise ValueError("end must be on or after start")
    return results.loc[(results.index >= start_date) & (results.index <= end_date)].copy()


def period_performance(
    results: pd.DataFrame,
    periods: dict[str, tuple[str, str | None]],
) -> pd.DataFrame:
    """Create per-period metrics for Hedge, momentum-only, and benchmark."""
    from src.report import performance_summary

    rows: list[dict[str, object]] = []
    for period_name, (start, end) in periods.items():
        sample = slice_evaluation_period(results, start, end)
        row: dict[str, object] = {"period": period_name, "start": start, "end": end or "latest"}
        if sample.empty:
            row["observations"] = 0
            row["note"] = "No observations in period"
            rows.append(row)
            continue
        row["observations"] = len(sample)
        for label, return_col, cost_col, turnover_col in (
            ("hedge", "hedge_return", "hedge_transaction_cost", "hedge_turnover"),
            ("momentum", "momentum_return", "momentum_transaction_cost", "momentum_turnover"),
            ("benchmark", "benchmark_return", None, None),
        ):
            inputs = pd.DataFrame(
                {"strategy_return": sample[return_col], "benchmark_return": sample["benchmark_return"]},
                index=sample.index,
            )
            if cost_col:
                inputs["transaction_cost"] = sample[cost_col]
            if turnover_col:
                inputs["turnover"] = sample[turnover_col]
            summary = performance_summary(inputs)
            for metric, value in summary.items():
                if metric.startswith("strategy_"):
                    row[f"{label}_{metric.removeprefix('strategy_')}"] = value
            if cost_col:
                row[f"{label}_total_transaction_cost_return_units"] = summary.get("total_transaction_cost_return_units", float("nan"))
                row[f"{label}_total_one_way_turnover"] = summary.get("total_one_way_turnover", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)
