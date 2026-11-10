"""Final holdout evaluation and shadow-paper-trading utilities.

The holdout compares predeclared configurations without selecting parameters on
holdout returns. HAC/Newey-West standard errors reduce (but do not eliminate)
serial-correlation concerns. Paper trading writes a local shadow ledger only;
no broker, orders, or live capital are involved.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.backtest import run_backtest

DEFAULT_CONFIGS = [
    {"name": "top5_126d", "top_n": 5, "min_history_days": 126},
    {"name": "top10_126d", "top_n": 10, "min_history_days": 126},
    {"name": "top5_252d", "top_n": 5, "min_history_days": 252},
    {"name": "top10_252d", "top_n": 10, "min_history_days": 252},
    {"name": "top20_252d", "top_n": 20, "min_history_days": 252},
]


def _hac_mean_test(values: pd.Series, max_lags: int | None = None) -> tuple[float, float]:
    """Two-sided normal-approximation test for mean > 0 using Newey-West SE."""
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 30:
        return np.nan, np.nan
    centered = x - x.mean()
    lag_count = max_lags if max_lags is not None else int(4 * (n / 100) ** (2 / 9))
    lag_count = min(max(0, lag_count), n - 1)
    long_run_var = float(centered @ centered / n)
    for lag in range(1, lag_count + 1):
        weight = 1 - lag / (lag_count + 1)
        gamma = float(centered[lag:] @ centered[:-lag] / n)
        long_run_var += 2 * weight * gamma
    se = np.sqrt(max(long_run_var, 0.0) / n)
    if se <= 0 or not np.isfinite(se):
        return np.nan, np.nan
    z = float(x.mean() / se)
    return z, float(2 * norm.sf(abs(z)))


def _summary(returns: pd.Series, benchmark: pd.Series) -> dict:
    aligned = pd.concat([returns.rename("strategy"), benchmark.rename("benchmark")], axis=1).dropna()
    if aligned.empty:
        return {"observations": 0, "net_return_pct": np.nan, "benchmark_return_pct": np.nan,
                "excess_return_pct": np.nan, "annualized_sharpe": np.nan,
                "max_drawdown_pct": np.nan, "hac_excess_z": np.nan, "hac_excess_p_value": np.nan,
                "significance_note": "Insufficient aligned observations"}
    r, b = aligned.strategy, aligned.benchmark
    equity = (1 + r).cumprod()
    vol = r.std(ddof=1)
    sharpe = r.mean() / vol * np.sqrt(252) if len(r) > 1 and vol > 0 else np.nan
    z, p = _hac_mean_test(r - b)
    return {"observations": len(aligned), "net_return_pct": float((equity.iloc[-1] - 1) * 100),
            "benchmark_return_pct": float(((1 + b).prod() - 1) * 100),
            "excess_return_pct": float(((1 + r).prod() - (1 + b).prod()) * 100),
            "annualized_sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
            "max_drawdown_pct": float((equity / equity.cummax() - 1).min() * 100),
            "hac_excess_z": z, "hac_excess_p_value": p,
            "significance_note": "HAC test on daily arithmetic excess returns; exploratory, not proof of alpha" if pd.notna(p) else "Need at least 30 aligned observations"}


def evaluate_final_holdout(prices: pd.DataFrame, features: dict[str, pd.DataFrame], *,
                           benchmark: str = "SPY", transaction_cost_bps: float = 10.0,
                           holdout_fraction: float = 0.20,
                           configs: list[dict] | None = None) -> tuple[pd.DataFrame, str]:
    """Evaluate fixed configurations on the final chronological slice only.

    Returns per-configuration metrics and the holdout start date. Configurations
    must be declared before looking at these results; do not re-tune on this slice.
    """
    if not 0.10 <= holdout_fraction <= 0.40:
        raise ValueError("holdout_fraction must be between 0.10 and 0.40")
    if not isinstance(prices.index, pd.DatetimeIndex) or not prices.index.is_monotonic_increasing:
        raise ValueError("prices must have an increasing DatetimeIndex")
    if benchmark not in prices:
        raise ValueError(f"Missing benchmark: {benchmark}")
    configs = configs or DEFAULT_CONFIGS
    split = int(len(prices) * (1 - holdout_fraction))
    if split < 252 or len(prices) - split < 30:
        raise ValueError("Need at least 252 pre-holdout sessions and 30 holdout sessions")
    start_date = prices.index[split]
    rows = []
    for cfg in configs:
        frame = run_backtest(prices, features, benchmark=benchmark, top_n=int(cfg["top_n"]),
            transaction_cost_bps=float(transaction_cost_bps), min_history_days=int(cfg["min_history_days"]))
        # Warm up holdings/signals on the full history; only score the final slice.
        sample = frame.loc[frame.index >= start_date]
        metrics = _summary(sample["strategy_return"], sample["benchmark_return"])
        rows.append({"config": cfg["name"], "top_n": int(cfg["top_n"]),
            "min_history_days": int(cfg["min_history_days"]), "holdout_start": start_date.date().isoformat(),
            "holdout_end": prices.index[-1].date().isoformat(), "transaction_cost_bps": float(transaction_cost_bps), **metrics})
    result = pd.DataFrame(rows)
    # Benjamini-Hochberg false-discovery-rate adjustment across tested configs.
    result["hac_excess_q_value"] = np.nan
    valid = result["hac_excess_p_value"].notna()
    pvals = result.loc[valid, "hac_excess_p_value"].astype(float)
    if not pvals.empty:
        ordered = pvals.sort_values()
        m = len(ordered)
        raw_q = ordered.to_numpy() * m / np.arange(1, m + 1)
        monotone_q = np.minimum.accumulate(raw_q[::-1])[::-1].clip(0, 1)
        q_series = pd.Series(monotone_q, index=ordered.index)
        result.loc[q_series.index, "hac_excess_q_value"] = q_series
    return result.sort_values(["hac_excess_q_value", "excess_return_pct"], ascending=[True, False], na_position="last"), start_date.date().isoformat()


def update_shadow_paper_ledger(prices: pd.DataFrame, features: dict[str, pd.DataFrame], *,
                               ledger_path: str | Path, config: dict,
                               benchmark: str = "SPY", transaction_cost_bps: float = 10.0,
                               start_date: str | None = None) -> pd.DataFrame:
    """Append idempotent daily shadow-portfolio returns; this does not place trades."""
    frame = run_backtest(prices, features, benchmark=benchmark, top_n=int(config["top_n"]),
        transaction_cost_bps=float(transaction_cost_bps), min_history_days=int(config["min_history_days"]))
    if start_date:
        frame = frame.loc[frame.index >= pd.Timestamp(start_date)]
    new = pd.DataFrame({"date": frame.index.strftime("%Y-%m-%d"), "config": config["name"],
        "strategy_return": frame["strategy_return"].to_numpy(),
        "benchmark_return": frame["benchmark_return"].to_numpy(),
        "transaction_cost": frame["transaction_cost"].to_numpy(),
        "turnover": frame["turnover"].to_numpy()})
    new = new.dropna(subset=["strategy_return", "benchmark_return"])
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(subset=["date", "config"], keep="last").sort_values(["config", "date"])
    new["strategy_equity"] = new.groupby("config")["strategy_return"].transform(lambda x: (1 + x).cumprod())
    new["benchmark_equity"] = new.groupby("config")["benchmark_return"].transform(lambda x: (1 + x).cumprod())
    new.to_csv(path, index=False)
    return new
