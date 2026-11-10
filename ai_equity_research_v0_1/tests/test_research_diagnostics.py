import numpy as np
import pandas as pd

from src.research_diagnostics import build_research_consensus, evaluate_ml_vs_momentum, data_quality_summary


def test_consensus_prefers_consistently_high_ranked_ticker_and_tracks_layers():
    quant = pd.DataFrame({"ticker": ["AAA", "BBB"], "rank": [1, 2], "score": [90, 10]})
    fund = pd.DataFrame({"ticker": ["AAA", "BBB"], "fundamental_rank": [2, 1], "fundamental_score": [40, 80]})
    ml = pd.DataFrame({"ticker": ["AAA", "BBB"], "rank": [1, 2], "predicted_return": [.2, -.1]})
    result = build_research_consensus(quant, fund, ml).set_index("ticker")
    assert result.loc["AAA", "layers_available"] == 3
    assert result.loc["BBB", "layers_available"] == 3
    assert result.loc["AAA", "consensus_rank"] == 1
    assert result.loc["AAA", "signal_disagreement"] > 0


def test_consensus_keeps_missing_layers_missing():
    quant = pd.DataFrame({"ticker": ["AAA"], "rank": [1]})
    result = build_research_consensus(quant, None, None)
    assert result.loc[0, "layers_available"] == 1
    assert result.loc[0, "consensus_percentile"] == 100.0


def test_ml_evaluation_compares_ml_and_momentum():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-31"] * 5),
        "ticker": list("ABCDE"),
        "predicted_return": [5, 4, 3, 2, 1],
        "momentum_12_1": [1, 2, 3, 4, 5],
        "actual_forward_return": [0.5, 0.4, 0.3, 0.2, 0.1],
    })
    result = evaluate_ml_vs_momentum(frame)
    assert len(result) == 1
    assert result.iloc[0]["ml_rank_ic"] > 0.99
    assert result.iloc[0]["momentum_rank_ic"] < -0.99
    assert result.iloc[0]["ml_top_quintile_mean_return"] > result.iloc[0]["momentum_top_quintile_mean_return"]


def test_data_quality_summary_reports_coverage():
    prices = pd.DataFrame({"AAA": [1.0, 2.0, 3.0], "BBB": [1.0, np.nan, 3.0]}, index=pd.date_range("2025-01-01", periods=3))
    summary = data_quality_summary(prices)
    assert summary["available"] is True
    assert summary["rows"] == 3
    assert summary["tickers"] == 2
    assert summary["missing_cell_fraction"] == 1 / 6
    assert summary["tickers_with_under_252_observations"] == 2
