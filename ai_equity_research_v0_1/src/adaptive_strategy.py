"""Auditable strategy outcome tracking and conservative adaptive selection.

This module does not mutate model weights or guarantee learning. It evaluates
matured, timestamped predictions and maintains an auditable strategy ledger.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED = {"date", "ticker", "predicted_return", "actual_forward_return"}


def evaluate_prediction_history(predictions: pd.DataFrame) -> pd.DataFrame:
    """Score matured forecasts; positive/negative directional hit rate and returns.

    A prediction is a directional win when forecast and realized return have
    the same sign. Zero realized returns are neutral, not wins or losses.
    """
    cols = ["date", "strategy", "n", "directional_wins", "directional_losses",
            "win_rate_pct", "mean_actual_return", "top_quintile_mean_return"]
    if predictions is None or predictions.empty or not REQUIRED.issubset(predictions.columns):
        return pd.DataFrame(columns=cols)
    df = predictions.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("predicted_return", "actual_forward_return"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["strategy"] = df["strategy"].fillna("default").astype(str) if "strategy" in df else "default"
    df = df.dropna(subset=["date", "predicted_return", "actual_forward_return"])
    df = df[df["actual_forward_return"] != 0].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["win"] = np.sign(df["predicted_return"]) == np.sign(df["actual_forward_return"])
    rows = []
    for (date, strategy), g in df.groupby(["date", "strategy"]):
        k = max(1, int(np.ceil(len(g) * .2)))
        top = g.nlargest(k, "predicted_return")
        rows.append({
            "date": date, "strategy": strategy, "n": len(g),
            "directional_wins": int(g["win"].sum()),
            "directional_losses": int((~g["win"]).sum()),
            "win_rate_pct": float(g["win"].mean() * 100),
            "mean_actual_return": float(g["actual_forward_return"].mean()),
            "top_quintile_mean_return": float(top["actual_forward_return"].mean()),
        })
    return pd.DataFrame(rows, columns=cols).sort_values(["date", "strategy"])


def update_strategy_ledger(predictions: pd.DataFrame, ledger_path: str | Path,
                           loss_limit: int = 5, min_resolved: int = 20) -> pd.DataFrame:
    """Append newly matured outcomes and flag strategies after consecutive losses.

    Input rows are individual forecast outcomes with strategy, date, ticker,
    predicted_return and actual_forward_return. Duplicate date/strategy/ticker
    outcomes are ignored. Disabled strategies remain disabled pending explicit
    review; this function never silently reactivates a strategy.
    """
    if loss_limit < 1 or min_resolved < 1:
        raise ValueError("loss_limit and min_resolved must be positive")
    p = Path(ledger_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    required = REQUIRED | {"strategy"}
    if predictions is None or predictions.empty or not required.issubset(predictions.columns):
        old = pd.read_csv(p) if p.exists() else pd.DataFrame()
        return old
    new = predictions.copy()
    new["date"] = pd.to_datetime(new["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    new["strategy"] = new["strategy"].astype(str)
    for c in ("predicted_return", "actual_forward_return"):
        new[c] = pd.to_numeric(new[c], errors="coerce")
    new = new.dropna(subset=["date", "ticker", "strategy", "predicted_return", "actual_forward_return"])
    new["win"] = (np.sign(new["predicted_return"]) == np.sign(new["actual_forward_return"])) & (new["actual_forward_return"] != 0)
    new["loss"] = (np.sign(new["predicted_return"]) != np.sign(new["actual_forward_return"])) & (new["actual_forward_return"] != 0)
    if p.exists():
        old = pd.read_csv(p)
        # Keep prior status/history; append only unseen outcome keys.
        keys = ["date", "strategy", "ticker"]
        seen = set(map(tuple, old[keys].astype(str).to_numpy())) if set(keys).issubset(old.columns) else set()
        new = new[[tuple(row) not in seen for row in new[keys].astype(str).to_numpy()]]
        ledger = pd.concat([old, new], ignore_index=True, sort=False)
    else:
        ledger = new
    if ledger.empty:
        ledger.to_csv(p, index=False)
        return ledger
    if "disabled" not in ledger:
        ledger["disabled"] = False
    ledger["disabled"] = ledger["disabled"].fillna(False).astype(bool)
    for strategy, idx in ledger.groupby("strategy", sort=False).groups.items():
        g = ledger.loc[idx].sort_values("date")
        losses = g["loss"].fillna(False).astype(bool).to_numpy()
        streak = 0
        for loss in losses[::-1]:
            if loss:
                streak += 1
            else:
                break
        if streak >= loss_limit:
            ledger.loc[idx, "disabled"] = True
        # record current consecutive loss streak for transparency
        ledger.loc[idx, "consecutive_losses"] = streak
    ledger["resolved_count"] = ledger.groupby("strategy")["actual_forward_return"].transform("count")
    ledger["mean_realized_return"] = ledger.groupby("strategy")["actual_forward_return"].transform("mean")
    ledger["win_rate_pct"] = ledger.groupby("strategy")["win"].transform("mean") * 100
    ledger["eligible_for_promotion"] = (
        (ledger["resolved_count"] >= min_resolved)
        & (ledger["mean_realized_return"] > 0)
        & (~ledger["disabled"])
    )
    ledger.to_csv(p, index=False)
    return ledger


def strategy_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    """One row per strategy; promotion is evidence-gated, not just a high win rate."""
    if ledger is None or ledger.empty or "strategy" not in ledger:
        return pd.DataFrame(columns=["strategy", "resolved_count", "win_rate_pct", "mean_realized_return", "consecutive_losses", "disabled", "eligible_for_promotion"])
    rows = []
    for name, g in ledger.groupby("strategy"):
        g = g.sort_values("date")
        losses = g["loss"].fillna(False).astype(bool).to_numpy() if "loss" in g else []
        streak = 0
        for loss in losses[::-1]:
            if loss: streak += 1
            else: break
        rows.append({
            "strategy": name,
            "resolved_count": len(g),
            "win_rate_pct": pd.to_numeric(g.get("win"), errors="coerce").mean() * 100 if "win" in g else np.nan,
            "mean_realized_return": pd.to_numeric(g.get("actual_forward_return"), errors="coerce").mean(),
            "consecutive_losses": streak,
            "disabled": bool(g.get("disabled", pd.Series(False, index=g.index)).fillna(False).astype(bool).any()),
            "eligible_for_promotion": bool(len(g) >= 20 and pd.to_numeric(g.get("actual_forward_return"), errors="coerce").mean() > 0 and not g.get("disabled", pd.Series(False, index=g.index)).fillna(False).astype(bool).any()),
        })
    return pd.DataFrame(rows).sort_values(["eligible_for_promotion", "mean_realized_return"], ascending=[False, False])
