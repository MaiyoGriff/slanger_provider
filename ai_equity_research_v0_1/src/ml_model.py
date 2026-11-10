"""Leakage-aware walk-forward ML research baseline for Hedge v0.5.

This is a research prototype, not an investment recommendation. Labels are
forward returns over horizon_days. At each rebalance date, training rows are
restricted to labels whose outcome window has fully ended before that date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

FEATURE_NAMES = ("momentum_12_1", "momentum_6m", "volatility_63d")


def build_ml_panel(prices: pd.DataFrame, features: dict[str, pd.DataFrame],
                   horizon_days: int = 126) -> pd.DataFrame:
    """Build date/ticker rows with price-only features and forward-return label."""
    if prices.empty:
        raise ValueError("prices must not be empty")
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    prices = prices.sort_index()
    future = prices.shift(-horizon_days) / prices - 1.0
    frames = []
    for name in FEATURE_NAMES:
        if name not in features:
            raise ValueError(f"Missing feature: {name}")
    for ticker in prices.columns:
        frame = pd.DataFrame({
            "date": prices.index,
            "ticker": ticker,
            "momentum_12_1": features["momentum_12_1"][ticker].to_numpy(),
            "momentum_6m": features["momentum_6m"][ticker].to_numpy(),
            "volatility_63d": features["volatility_63d"][ticker].to_numpy(),
            "forward_return": future[ticker].to_numpy(),
        })
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel


def walk_forward_predictions(prices: pd.DataFrame, features: dict[str, pd.DataFrame],
                             horizon_days: int = 126, min_train_rows: int = 250,
                             rebalance: str = "month_end", random_state: int = 42) -> pd.DataFrame:
    """Predict cross-sectional forward returns at monthly dates without label leakage.

    For date t, training labels must have fully matured by t-horizon_days.
    Output columns: date, ticker, predicted_return, actual_forward_return.
    """
    panel = build_ml_panel(prices, features, horizon_days)
    dates = pd.DatetimeIndex(prices.sort_index().index)
    if rebalance != "month_end":
        raise ValueError("Only rebalance='month_end' is supported")
    signal_dates = pd.Series(dates, index=dates).groupby(dates.to_period("M")).last().tolist()
    output = []
    for date in signal_dates:
        # Find the historical cutoff position; labels ending after the cutoff are excluded.
        pos = dates.get_loc(date)
        cutoff_pos = pos - horizon_days
        if cutoff_pos < 0:
            continue
        cutoff_date = dates[cutoff_pos]
        train = panel[(panel["date"] <= cutoff_date)].dropna(subset=list(FEATURE_NAMES) + ["forward_return"])
        current = panel[(panel["date"] == date)].dropna(subset=list(FEATURE_NAMES))
        if len(train) < min_train_rows or current.empty:
            continue
        model = RandomForestRegressor(
            n_estimators=150, min_samples_leaf=5, max_features=1.0,
            random_state=random_state, n_jobs=-1,
        )
        model.fit(train[list(FEATURE_NAMES)], train["forward_return"])
        preds = model.predict(current[list(FEATURE_NAMES)])
        result = current[["date", "ticker", "forward_return"]].copy()
        result["predicted_return"] = preds
        result = result.rename(columns={"forward_return": "actual_forward_return"})
        output.append(result)
    if not output:
        return pd.DataFrame(columns=["date", "ticker", "predicted_return", "actual_forward_return"])
    return pd.concat(output, ignore_index=True).sort_values(["date", "predicted_return"], ascending=[True, False])


def latest_ml_ranking(prices: pd.DataFrame, features: dict[str, pd.DataFrame],
                      horizon_days: int = 126, min_train_rows: int = 250,
                      random_state: int = 42) -> pd.DataFrame:
    """Fit on matured historical labels and score eligible stocks at latest date."""
    panel = build_ml_panel(prices, features, horizon_days)
    dates = pd.DatetimeIndex(prices.sort_index().index)
    date = dates[-1]
    pos = dates.get_loc(date)
    cutoff_pos = pos - horizon_days
    if cutoff_pos < 0:
        raise ValueError("Insufficient history to form matured training labels")
    cutoff_date = dates[cutoff_pos]
    train = panel[(panel["date"] <= cutoff_date)].dropna(subset=list(FEATURE_NAMES) + ["forward_return"])
    current = panel[(panel["date"] == date)].dropna(subset=list(FEATURE_NAMES)).copy()
    if len(train) < min_train_rows:
        raise ValueError(f"Insufficient training rows: {len(train)} < {min_train_rows}")
    if current.empty:
        raise ValueError("No eligible stocks have complete current features")
    model = RandomForestRegressor(n_estimators=150, min_samples_leaf=5,
                                  max_features=1.0, random_state=random_state, n_jobs=-1)
    model.fit(train[list(FEATURE_NAMES)], train["forward_return"])
    current["predicted_return"] = model.predict(current[list(FEATURE_NAMES)])
    current["rank"] = current["predicted_return"].rank(method="min", ascending=False).astype(int)
    current["model_version"] = "v0.5-random-forest-price-only"
    return current.set_index("ticker")[["predicted_return", "rank", *FEATURE_NAMES, "model_version"]].sort_values("rank")
