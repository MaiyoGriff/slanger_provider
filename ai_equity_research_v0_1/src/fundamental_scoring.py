"""Transparent cross-sectional fundamental scoring; scores are not forecasts."""
from __future__ import annotations
import numpy as np
import pandas as pd

VALUE_LOWER_IS_BETTER = ["trailing_pe", "forward_pe", "price_to_book", "price_to_sales", "ev_to_ebitda"]
QUALITY_HIGHER_IS_BETTER = ["return_on_equity", "profit_margin", "operating_margin", "free_cash_flow", "current_ratio"]
GROWTH_HIGHER_IS_BETTER = ["revenue_growth", "earnings_growth"]

def _percentile(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Negative multiples and impossible values are not meaningful for the relevant valuation metric.
    if not higher_is_better:
        values = values.where(values > 0)
    ranked = values.rank(pct=True, method="average") * 100.0
    return ranked if higher_is_better else 100.0 - ranked

def score_fundamentals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Return 0-100 value/quality/growth scores and a weighted composite.

    Composite = 40% value, 35% quality, 25% growth. Missing factor values are
    averaged only over available metrics; absent whole groups remain NaN and
    are not silently treated as good scores.
    """
    if fundamentals.empty:
        raise ValueError("fundamentals is empty")
    result = fundamentals.copy()
    for column in VALUE_LOWER_IS_BETTER:
        if column in result:
            result[f"{column}_score"] = _percentile(result[column], higher_is_better=False)
    for column in QUALITY_HIGHER_IS_BETTER:
        if column in result:
            result[f"{column}_score"] = _percentile(result[column], higher_is_better=True)
    for column in GROWTH_HIGHER_IS_BETTER:
        if column in result:
            result[f"{column}_score"] = _percentile(result[column], higher_is_better=True)

    groups = {
        "value_score": VALUE_LOWER_IS_BETTER,
        "quality_score": QUALITY_HIGHER_IS_BETTER,
        "growth_score": GROWTH_HIGHER_IS_BETTER,
    }
    for output, columns in groups.items():
        score_columns = [f"{c}_score" for c in columns if f"{c}_score" in result]
        result[output] = result[score_columns].mean(axis=1, skipna=True) if score_columns else np.nan
    result["fundamental_score"] = (
        0.40 * result["value_score"]
        + 0.35 * result["quality_score"]
        + 0.25 * result["growth_score"]
    )
    # Require at least one available metric in each category to avoid ranking
    # a company as a combined value/quality/growth candidate with a missing category.
    result.loc[result[["value_score", "quality_score", "growth_score"]].isna().any(axis=1), "fundamental_score"] = np.nan
    result["fundamental_rank"] = result["fundamental_score"].rank(method="min", ascending=False).astype("Int64")
    return result.sort_values(["fundamental_rank", "fundamental_score"], ascending=[True, False], na_position="last")
