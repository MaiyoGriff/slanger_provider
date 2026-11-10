"""Walk-forward parameter search with regime diagnostics and net-of-cost ranking.

This is research prioritization only: it never places trades or changes live weights.
Configurations are selected using past training windows, then scored on the next
unseen window. All returns are the existing backtest's net returns.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.backtest import run_backtest

DEFAULT_CONFIGS = [
    {"name": "top5_126d", "top_n": 5, "min_history_days": 126},
    {"name": "top10_126d", "top_n": 10, "min_history_days": 126},
    {"name": "top5_252d", "top_n": 5, "min_history_days": 252},
    {"name": "top10_252d", "top_n": 10, "min_history_days": 252},
    {"name": "top20_252d", "top_n": 20, "min_history_days": 252},
]


def _metrics(returns: pd.Series, periods_per_year: int = 252) -> dict:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return {"observations": len(r), "net_return_pct": np.nan, "annualized_sharpe": np.nan,
                "max_drawdown_pct": np.nan, "win_rate_pct": np.nan}
    equity = (1 + r).cumprod()
    vol = r.std(ddof=1)
    sharpe = (r.mean() / vol * np.sqrt(periods_per_year)) if vol > 0 else np.nan
    dd = equity / equity.cummax() - 1
    return {"observations": len(r), "net_return_pct": float((equity.iloc[-1] - 1) * 100),
            "annualized_sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
            "max_drawdown_pct": float(dd.min() * 100), "win_rate_pct": float((r > 0).mean() * 100)}


def classify_market_regimes(prices: pd.DataFrame, benchmark: str = "SPY") -> pd.Series:
    """Classify dates using trailing benchmark trend and volatility, without look-ahead."""
    if benchmark not in prices:
        raise ValueError(f"Missing benchmark: {benchmark}")
    ret = prices[benchmark].pct_change(fill_method=None)
    trend = prices[benchmark].pct_change(126, fill_method=None)
    vol = ret.rolling(63, min_periods=40).std() * np.sqrt(252)
    high_vol = vol > vol.rolling(252, min_periods=80).quantile(.75)
    regime = pd.Series("sideways", index=prices.index, dtype="object")
    regime.loc[trend > 0.05] = "bull"
    regime.loc[trend < -0.05] = "bear"
    regime.loc[high_vol.fillna(False)] = regime.loc[high_vol.fillna(False)].map(lambda x: f"high_vol_{x}")
    return regime.rename("market_regime")


def run_adaptive_search(prices: pd.DataFrame, features: dict[str, pd.DataFrame], benchmark: str = "SPY",
                        transaction_cost_bps: float = 10.0, configs: list[dict] | None = None,
                        min_train_days: int = 756, test_days: int = 126) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select configs on expanding historical windows and evaluate on next unseen fold.

    Returns (fold-level OOS results, regime-level OOS summaries). Cost sensitivity
    is explicit and passed to every backtest. Selection objective is training Sharpe,
    with drawdown as a tie-breaker; no OOS returns are used for that fold's selection.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices index must be a DatetimeIndex")
    if benchmark not in prices:
        raise ValueError(f"Missing benchmark: {benchmark}")
    if min_train_days < 252 or test_days < 20:
        raise ValueError("min_train_days must be >=252 and test_days >=20")
    configs = configs or DEFAULT_CONFIGS
    if not configs:
        raise ValueError("At least one configuration is required")
    cache = {}
    for cfg in configs:
        name = cfg["name"]
        cache[name] = run_backtest(prices, features, benchmark=benchmark,
            top_n=int(cfg["top_n"]), transaction_cost_bps=float(transaction_cost_bps),
            min_history_days=int(cfg["min_history_days"]))
    regimes = classify_market_regimes(prices, benchmark)
    rows = []
    start = min_train_days
    fold = 0
    while start < len(prices):
        stop = min(start + test_days, len(prices))
        if stop - start < 20:
            break
        train_start = max(0, start - min_train_days)
        scores = []
        for cfg in configs:
            frame = cache[cfg["name"]]
            train = frame["strategy_return"].iloc[train_start:start]
            m = _metrics(train)
            scores.append((cfg, m))
        # Robust selection: positive train return required; prefer Sharpe then smaller drawdown.
        eligible = [(c, m) for c, m in scores if pd.notna(m["annualized_sharpe"]) and m["net_return_pct"] > 0]
        if eligible:
            chosen, train_metrics = max(eligible, key=lambda cm: (cm[1]["annualized_sharpe"], cm[1]["max_drawdown_pct"]))
        else:
            chosen, train_metrics = max(scores, key=lambda cm: (cm[1]["annualized_sharpe"] if pd.notna(cm[1]["annualized_sharpe"]) else -np.inf, cm[1]["max_drawdown_pct"] if pd.notna(cm[1]["max_drawdown_pct"]) else -np.inf))
        oos = cache[chosen["name"]]["strategy_return"].iloc[start:stop]
        met = _metrics(oos)
        oos_regimes = regimes.iloc[start:stop]
        for regime in sorted(oos_regimes.dropna().unique()):
            mask = oos_regimes == regime
            rm = _metrics(oos.loc[mask])
            rows.append({"fold": fold, "train_start": prices.index[train_start].date().isoformat(),
                "train_end": prices.index[start-1].date().isoformat(), "test_start": prices.index[start].date().isoformat(),
                "test_end": prices.index[stop-1].date().isoformat(), "selected_config": chosen["name"],
                "top_n": int(chosen["top_n"]), "min_history_days": int(chosen["min_history_days"]),
                "transaction_cost_bps": float(transaction_cost_bps), "regime": regime,
                "train_sharpe": train_metrics["annualized_sharpe"], **{f"oos_{k}": v for k,v in rm.items()}})
        fold += 1
        start = stop
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame(columns=["regime", "folds", "oos_net_return_pct", "oos_sharpe", "oos_max_drawdown_pct", "research_priority"])
    summary_rows = []
    for regime, g in detail.groupby("regime"):
        # Weighted priority discourages tiny-sample/regime overconfidence.
        n = int(g["oos_observations"].sum())
        sharpe = pd.to_numeric(g["oos_annualized_sharpe"], errors="coerce").mean()
        net = pd.to_numeric(g["oos_net_return_pct"], errors="coerce").mean()
        dd = pd.to_numeric(g["oos_max_drawdown_pct"], errors="coerce").mean()
        priority = (sharpe if pd.notna(sharpe) else -10) + (net / 100 if pd.notna(net) else -10) + (dd / 100 if pd.notna(dd) else -10)
        summary_rows.append({"regime": regime, "folds": len(g), "oos_observations": n,
            "oos_net_return_pct": net, "oos_sharpe": sharpe, "oos_max_drawdown_pct": dd,
            "research_priority": priority, "priority_note": "Research ranking only; require adequate sample and stability review"})
    summary = pd.DataFrame(summary_rows).sort_values("research_priority", ascending=False)
    return detail, summary
