"""Event-driven, local-only paper trading simulator for Hedge.

Signals are frozen at a month-end close and execute at the next available close
because the bundled source data contains adjusted-close panels, not intraday/open
quotes. A real OHLC/open feed can be added later; this module never claims fills
at prices that are not in the supplied data. No broker or live orders are used.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd


def _monthly_signal_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(index, index=index)
    return s.groupby(index.to_period("M")).last().tolist()


def _read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns or [])


def update_paper_simulator(
    prices: pd.DataFrame, *, output_dir: str | Path, benchmark: str = "SPY",
    top_n: int = 10, min_history_days: int = 252,
    transaction_cost_bps: float = 10.0, initial_cash: float = 100_000.0,
    start_date: str | None = None, max_position_weight: float = 0.20,
    max_gross_exposure: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """Append new frozen forecasts and update fills, portfolio and risk ledgers.

    Forecast keys are (signal_date, strategy, ticker); existing forecast values are
    preserved on reruns. Executions occur at the next available adjusted close.
    Returns are mark-to-market daily, with turnover-based transaction costs.
    """
    if prices is None or prices.empty:
        raise ValueError("Price data is empty")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must have a DatetimeIndex")
    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark {benchmark!r} is missing")
    if top_n < 1 or min_history_days < 1 or initial_cash <= 0:
        raise ValueError("top_n, min_history_days and initial_cash must be positive")
    if transaction_cost_bps < 0 or not 0 < max_position_weight <= 1 or not 0 < max_gross_exposure <= 1:
        raise ValueError("Invalid costs or exposure limits")
    px = prices.sort_index().astype(float).copy()
    px = px.loc[~px.index.duplicated(keep="last")]
    px = px.dropna(axis=1, how="all")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    strategy = f"momentum_top{top_n}_{min_history_days}d"
    pred_path, fill_path = out / "paper_predictions.csv", out / "paper_fills.csv"
    daily_path, positions_path = out / "paper_daily_ledger.csv", out / "paper_positions.csv"
    predictions = _read(pred_path, ["signal_date", "execution_date", "strategy", "ticker", "score_at_signal", "predicted_direction", "execution_price", "realized_exit_date", "realized_exit_price", "realized_return", "directional_win"])
    fills = _read(fill_path, ["date", "strategy", "ticker", "side", "weight_delta", "reference_price", "transaction_cost_bps", "estimated_cost_cash", "status"])
    daily_old = _read(daily_path)
    positions_old = _read(positions_path)

    # Create only new, immutable signal snapshots. Momentum is a ranking score, not
    # a calibrated expected-return forecast; predicted direction is its sign.
    score = px.shift(21) / px.shift(252) - 1.0
    signals = _monthly_signal_dates(px.index)
    existing_keys = set()
    if not predictions.empty:
        existing_keys = set(zip(predictions.signal_date.astype(str), predictions.strategy.astype(str), predictions.ticker.astype(str)))
    new_pred_rows, new_fill_rows = [], []
    for signal_date in signals:
        if start_date and signal_date < pd.Timestamp(start_date):
            continue
        pos = px.index.get_loc(signal_date)
        if not isinstance(pos, (int, np.integer)) or pos + 1 >= len(px.index):
            continue
        execution_date = px.index[pos + 1]
        hist_count = px.loc[:signal_date].notna().sum()
        s = score.loc[signal_date].dropna()
        s = s[(hist_count.reindex(s.index).fillna(0) >= min_history_days) & (s.index != benchmark)]
        chosen = s.sort_values(ascending=False, kind="mergesort").head(top_n)
        if chosen.empty:
            continue
        raw_weight = min(max_gross_exposure / len(chosen), max_position_weight)
        total_weight = raw_weight * len(chosen)
        for ticker, val in chosen.items():
            key = (signal_date.strftime("%Y-%m-%d"), strategy, str(ticker))
            if key not in existing_keys:
                new_pred_rows.append({"signal_date": key[0], "execution_date": execution_date.strftime("%Y-%m-%d"),
                    "strategy": strategy, "ticker": str(ticker), "score_at_signal": float(val),
                    "predicted_direction": int(np.sign(val)), "target_weight": float(raw_weight),
                    "execution_price": float(px.at[execution_date, ticker]) if pd.notna(px.at[execution_date, ticker]) else np.nan,
                    "realized_exit_date": None, "realized_exit_price": np.nan, "realized_return": np.nan,
                    "directional_win": np.nan, "forecast_note": "Momentum score sign; not a calibrated expected return"})
                existing_keys.add(key)

    if new_pred_rows:
        predictions = pd.concat([predictions, pd.DataFrame(new_pred_rows)], ignore_index=True, sort=False)
    if not predictions.empty:
        predictions["signal_date"] = pd.to_datetime(predictions["signal_date"], errors="coerce")
        predictions["execution_date"] = pd.to_datetime(predictions["execution_date"], errors="coerce")
        predictions["realized_exit_date"] = pd.to_datetime(predictions.get("realized_exit_date"), errors="coerce")
        # Realized horizon is from execution until the next signal's execution date.
        # Existing resolved outcomes are immutable; only unresolved forecasts mature.
        for (strat, signal_date), idx in predictions.groupby(["strategy", "signal_date"], dropna=False).groups.items():
            g = predictions.loc[idx]
            exec_date = g["execution_date"].iloc[0]
            future_execs = predictions.loc[(predictions.strategy == strat) & (predictions.execution_date > exec_date), "execution_date"]
            exit_date = future_execs.min() if not future_execs.empty else pd.NaT
            if pd.isna(exit_date) or exit_date not in px.index:
                continue
            for row_idx, row in g.iterrows():
                if pd.notna(row.get("realized_return")):
                    continue
                ticker = row["ticker"]
                if ticker not in px or pd.isna(px.at[exec_date, ticker]) or pd.isna(px.at[exit_date, ticker]):
                    continue
                entry = float(px.at[exec_date, ticker]); exit_px = float(px.at[exit_date, ticker])
                realized = exit_px / entry - 1.0
                predictions.at[row_idx, "realized_exit_date"] = exit_date
                predictions.at[row_idx, "realized_exit_price"] = exit_px
                predictions.at[row_idx, "realized_return"] = realized
                predictions.at[row_idx, "directional_win"] = bool(np.sign(row["predicted_direction"]) == np.sign(realized)) if realized != 0 else np.nan
        predictions["signal_date"] = pd.to_datetime(predictions["signal_date"]).dt.strftime("%Y-%m-%d")
        predictions["execution_date"] = pd.to_datetime(predictions["execution_date"]).dt.strftime("%Y-%m-%d")
        predictions["realized_exit_date"] = pd.to_datetime(predictions["realized_exit_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        predictions.to_csv(pred_path, index=False)

    # Replay the full observed period deterministically for portfolio-level equity.
    # Holdings are effective only after next-close execution; returns begin after that close.
    target_by_date: dict[pd.Timestamp, dict[str, float]] = {}
    if not predictions.empty:
        pp = predictions.copy()
        pp["execution_date"] = pd.to_datetime(pp["execution_date"], errors="coerce")
        for date, group in pp.groupby("execution_date"):
            target_by_date[date] = {str(r.ticker): float(r.target_weight) for r in group.itertuples() if pd.notna(r.target_weight)}
    current: dict[str, float] = {}; rows = []; fill_rows = []
    equity = initial_cash; peak = initial_cash; prev_prices = None
    returns = px.pct_change(fill_method=None)
    for i, date in enumerate(px.index):
        # First mark the portfolio held from the prior close through today's close.
        # A signal executed at today's close cannot earn today's close-to-close return.
        daily_gross = 0.0; missing = False
        if i > 0 and current:
            for ticker, weight in current.items():
                r = returns.at[date, ticker] if ticker in returns else np.nan
                if pd.isna(r): missing = True
                else: daily_gross += weight * float(r)
        target = target_by_date.get(date)
        turnover = 0.0; cost_cash = 0.0
        if target is not None:
            tickers = set(current) | set(target)
            deltas = {t: target.get(t, 0.0) - current.get(t, 0.0) for t in tickers}
            turnover = sum(abs(v) for v in deltas.values())
            cost_cash = equity * turnover * transaction_cost_bps / 10_000.0
            for ticker, delta in deltas.items():
                if abs(delta) > 1e-10:
                    fill_rows.append({"date": date.strftime("%Y-%m-%d"), "strategy": strategy, "ticker": ticker,
                        "side": "BUY" if delta > 0 else "SELL", "weight_delta": delta,
                        "reference_price": float(px.at[date, ticker]) if ticker in px and pd.notna(px.at[date, ticker]) else np.nan,
                        "transaction_cost_bps": transaction_cost_bps, "estimated_cost_cash": abs(delta) * equity * transaction_cost_bps / 10_000.0,
                        "status": "SIMULATED_NEXT_AVAILABLE_CLOSE"})
            current = target.copy()
        net_ret = daily_gross - (cost_cash / equity if equity else 0.0)
        equity *= (1.0 + net_ret); peak = max(peak, equity)
        dd = equity / peak - 1.0
        bench_ret = returns.at[date, benchmark]
        rows.append({"date": date.strftime("%Y-%m-%d"), "strategy": strategy, "cash_start": initial_cash,
            "gross_exposure": sum(current.values()), "holdings_count": len(current), "gross_return": np.nan if missing else daily_gross,
            "net_return": np.nan if missing else net_ret, "benchmark_return": bench_ret,
            "turnover": turnover, "transaction_cost_cash": cost_cash, "equity": equity,
            "drawdown_pct": dd * 100.0, "missing_held_price": bool(missing),
            "execution_model": "next available adjusted close; close-only data proxy"})
        # For next session PnL, weights held through the session remain current.
    daily = pd.DataFrame(rows)
    if not daily_old.empty:
        # The replay is authoritative for mark-to-market, and stable for the same data/config.
        pass
    daily.to_csv(daily_path, index=False)
    fills_df = pd.DataFrame(fill_rows)
    # Idempotent fills keyed by date/strategy/ticker/side.
    if not fills.empty:
        fills_df = pd.concat([fills, fills_df], ignore_index=True, sort=False)
    if not fills_df.empty:
        fills_df = fills_df.drop_duplicates(["date", "strategy", "ticker", "side"], keep="last").sort_values(["date", "ticker"])
        fills_df.to_csv(fill_path, index=False)
    # Current position snapshot based on latest executed target.
    latest_date = px.index[-1]
    latest_target_date = max([d for d in target_by_date if d <= latest_date], default=None)
    pos_rows = []
    if latest_target_date is not None:
        for ticker, weight in target_by_date[latest_target_date].items():
            pos_rows.append({"as_of_date": latest_date.strftime("%Y-%m-%d"), "last_execution_date": latest_target_date.strftime("%Y-%m-%d"),
                "strategy": strategy, "ticker": ticker, "weight": weight,
                "latest_price": float(px[ticker].dropna().iloc[-1]) if ticker in px and px[ticker].notna().any() else np.nan,
                "market_value_proxy": equity * weight})
    positions = pd.DataFrame(pos_rows)
    positions.to_csv(positions_path, index=False)
    # Risk/forecast metrics: mark only matured prediction rows.
    realized = pd.to_numeric(predictions.get("realized_return", pd.Series(dtype=float)), errors="coerce")
    win = pd.to_numeric(predictions.get("directional_win", pd.Series(dtype=float)), errors="coerce")
    summary = pd.DataFrame([{
        "strategy": strategy, "as_of_date": latest_date.strftime("%Y-%m-%d"), "initial_cash": initial_cash,
        "latest_equity": float(daily.equity.iloc[-1]) if not daily.empty else initial_cash,
        "total_return_pct": float((daily.equity.iloc[-1] / initial_cash - 1) * 100) if not daily.empty else 0.0,
        "max_drawdown_pct": float(daily.drawdown_pct.min()) if not daily.empty else 0.0,
        "annualized_volatility_pct": float(pd.to_numeric(daily.net_return, errors="coerce").std() * np.sqrt(252) * 100) if len(daily) > 1 else np.nan,
        "resolved_forecasts": int(realized.notna().sum()), "forecast_directional_win_rate_pct": float(win.mean() * 100) if win.notna().any() else np.nan,
        "mean_realized_forecast_return_pct": float(realized.mean() * 100) if realized.notna().any() else np.nan,
        "benchmark_total_return_pct": float(((1 + pd.to_numeric(daily.benchmark_return, errors="coerce").dropna()).prod() - 1) * 100) if daily.benchmark_return.notna().any() else np.nan,
        "open_forecasts": int(realized.isna().sum()), "execution_price_note": "Adjusted close proxy; not bid/ask or open fills"}])
    summary.to_csv(out / "paper_risk_summary.csv", index=False)
    return {"predictions": predictions, "fills": fills_df, "daily": daily, "positions": positions, "summary": summary}
