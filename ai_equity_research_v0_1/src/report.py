from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
TRADING_DAYS_PER_YEAR = 252


def export_ranking(ranking: pd.DataFrame) -> None:
    """Write the current cross-sectional ranking as CSV and HTML."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    ranking.to_csv(OUTPUT_DIR / "latest_ranking.csv", index_label="ticker")
    ranking.to_html(
        OUTPUT_DIR / "latest_ranking.html",
        index=True,
        border=0,
        float_format=lambda value: f"{value:.3f}",
    )


def _series_metrics(returns: pd.Series, prefix: str, risk_free_rate: float = 0.0) -> dict[str, Any]:
    """Calculate metrics from periodic simple returns; risk_free_rate is annualized."""
    r = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if r.empty:
        return {
            f"{prefix}_total_return": float("nan"),
            f"{prefix}_annualized_return": float("nan"),
            f"{prefix}_annualized_volatility": float("nan"),
            f"{prefix}_sharpe": float("nan"),
            f"{prefix}_sortino": float("nan"),
            f"{prefix}_maximum_drawdown": float("nan"),
            f"{prefix}_calmar": float("nan"),
            f"{prefix}_win_rate": float("nan"),
            f"{prefix}_best_day": float("nan"),
            f"{prefix}_worst_day": float("nan"),
            f"{prefix}_observations": 0,
        }

    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    # Use elapsed calendar time when a DatetimeIndex is available; otherwise use 252-day convention.
    if isinstance(r.index, pd.DatetimeIndex) and len(r) > 1:
        elapsed_years = (r.index[-1] - r.index[0]).days / 365.25
    else:
        elapsed_years = len(r) / TRADING_DAYS_PER_YEAR
    cagr = float(equity.iloc[-1] ** (1.0 / elapsed_years) - 1.0) if elapsed_years > 0 and equity.iloc[-1] > 0 else float("nan")
    daily_vol = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
    annual_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR) if np.isfinite(daily_vol) else float("nan")
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    excess = r - daily_rf
    excess_std = float(excess.std(ddof=1)) if len(excess) > 1 else float("nan")
    sharpe = float(excess.mean() / excess_std * np.sqrt(TRADING_DAYS_PER_YEAR)) if excess_std > 0 else float("nan")
    downside = np.minimum(excess.to_numpy(), 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sortino = float(excess.mean() * TRADING_DAYS_PER_YEAR / downside_deviation) if downside_deviation > 0 else float("nan")
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 and np.isfinite(cagr) else float("nan")

    return {
        f"{prefix}_total_return": total_return,
        f"{prefix}_annualized_return": cagr,
        f"{prefix}_annualized_volatility": float(annual_vol),
        f"{prefix}_sharpe": sharpe,
        f"{prefix}_sortino": sortino,
        f"{prefix}_maximum_drawdown": max_drawdown,
        f"{prefix}_calmar": calmar,
        f"{prefix}_win_rate": float((r > 0).mean()),
        f"{prefix}_best_day": float(r.max()),
        f"{prefix}_worst_day": float(r.min()),
        f"{prefix}_observations": int(len(r)),
    }


def performance_summary(backtest: pd.DataFrame, risk_free_rate: float = 0.0) -> dict[str, Any]:
    """Summarize strategy and benchmark returns, costs and turnover.

    Returns are assumed to be daily simple returns. Sharpe and Sortino use an
    annualized risk-free rate (zero by default). This is a diagnostic report,
    not a guarantee of future performance.
    """
    required = {"strategy_return", "benchmark_return"}
    missing = required.difference(backtest.columns)
    if missing:
        raise ValueError(f"Backtest is missing required columns: {sorted(missing)}")
    if backtest.empty:
        raise ValueError("Backtest is empty; insufficient returns to evaluate.")
    if risk_free_rate <= -1:
        raise ValueError("risk_free_rate must be greater than -1.")

    summary: dict[str, Any] = {}
    summary.update(_series_metrics(backtest["strategy_return"], "strategy", risk_free_rate))
    summary.update(_series_metrics(backtest["benchmark_return"], "benchmark", risk_free_rate))
    strategy_total = summary["strategy_total_return"]
    benchmark_total = summary["benchmark_total_return"]
    summary["total_return_difference_vs_benchmark"] = (
        float(strategy_total - benchmark_total)
        if np.isfinite(strategy_total) and np.isfinite(benchmark_total)
        else float("nan")
    )
    if "transaction_cost" in backtest.columns:
        summary["total_transaction_cost_return_units"] = float(
            pd.to_numeric(backtest["transaction_cost"], errors="coerce").fillna(0).sum()
        )
    if "turnover" in backtest.columns:
        turnover = pd.to_numeric(backtest["turnover"], errors="coerce").dropna()
        summary["average_daily_turnover"] = float(turnover.mean()) if not turnover.empty else float("nan")
        summary["total_one_way_turnover"] = float(turnover.sum()) if not turnover.empty else float("nan")
    summary["risk_free_rate_annual"] = float(risk_free_rate)
    summary["annualization_trading_days"] = TRADING_DAYS_PER_YEAR
    return summary
