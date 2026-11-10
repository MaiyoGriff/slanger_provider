import pandas as pd


def percentile_score(series: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank scaled to 0–100; missing values stay missing."""
    return series.rank(pct=True, method="average") * 100.0


def rank_stocks(features: dict[str, pd.DataFrame], as_of: str | pd.Timestamp | None = None,
                min_history_days: int = 252) -> pd.DataFrame:
    momentum = features["momentum_12_1"]
    volatility = features["volatility_63d"]
    returns = features["daily_returns"]
    if momentum.empty:
        raise ValueError("Momentum feature matrix is empty.")
    date = pd.Timestamp(as_of) if as_of is not None else momentum.index.max()
    if date not in momentum.index:
        raise ValueError(f"No feature observations for {date.date()}")
    history_count = returns.loc[:date].notna().sum()
    raw = pd.DataFrame({
        "momentum_12_1": momentum.loc[date],
        "volatility_63d": volatility.loc[date],
        "history_days": history_count,
    })
    raw = raw[
        (raw["history_days"] >= min_history_days)
        & raw["momentum_12_1"].notna()
        & raw["volatility_63d"].notna()
    ].copy()
    if raw.empty:
        raise ValueError("No stocks meet the data eligibility rules.")
    raw["momentum_score"] = percentile_score(raw["momentum_12_1"])
    raw["risk_score"] = 100.0 - percentile_score(raw["volatility_63d"])
    raw["score"] = 0.75 * raw["momentum_score"] + 0.25 * raw["risk_score"]
    raw["rank"] = raw["score"].rank(method="min", ascending=False).astype(int)
    raw["model_version"] = "v0.1-price-only"
    raw.index.name = "ticker"
    return raw.sort_values(["rank", "score"], ascending=[True, False])
