import numpy as np
import pandas as pd
import pytest

from src.fundamental_scoring import score_fundamentals

def sample_fundamentals():
    return pd.DataFrame({
        "trailing_pe": [10.0, 20.0, 30.0, np.nan],
        "forward_pe": [9.0, 18.0, 25.0, np.nan],
        "price_to_book": [1.0, 2.0, 3.0, np.nan],
        "price_to_sales": [2.0, 4.0, 6.0, np.nan],
        "ev_to_ebitda": [6.0, 12.0, 18.0, np.nan],
        "return_on_equity": [0.20, 0.12, 0.05, np.nan],
        "profit_margin": [0.20, 0.12, 0.05, np.nan],
        "operating_margin": [0.25, 0.15, 0.07, np.nan],
        "free_cash_flow": [100.0, 50.0, 10.0, np.nan],
        "current_ratio": [2.0, 1.5, 1.0, np.nan],
        "revenue_growth": [0.25, 0.12, 0.02, np.nan],
        "earnings_growth": [0.30, 0.10, -0.05, np.nan],
    }, index=pd.Index(["AAA", "BBB", "CCC", "DDD"], name="ticker"))

def test_scores_rank_stronger_value_quality_growth_higher():
    ranked = score_fundamentals(sample_fundamentals())
    assert ranked.loc["AAA", "value_score"] > ranked.loc["CCC", "value_score"]
    assert ranked.loc["AAA", "quality_score"] > ranked.loc["CCC", "quality_score"]
    assert ranked.loc["AAA", "growth_score"] > ranked.loc["CCC", "growth_score"]
    assert ranked.index[0] == "AAA"

def test_composite_formula_and_bounds():
    ranked = score_fundamentals(sample_fundamentals())
    row = ranked.loc["AAA"]
    assert row["fundamental_score"] == pytest.approx(
        0.40 * row["value_score"] + 0.35 * row["quality_score"] + 0.25 * row["growth_score"]
    )
    assert ranked["fundamental_score"].dropna().between(0, 100).all()

def test_missing_entire_factor_groups_not_given_fake_score():
    data = pd.DataFrame({"trailing_pe": [10.0, 20.0]}, index=["AAA", "BBB"])
    ranked = score_fundamentals(data)
    assert ranked["fundamental_score"].isna().all()

def test_empty_input_raises():
    with pytest.raises(ValueError, match="fundamentals is empty"):
        score_fundamentals(pd.DataFrame())
