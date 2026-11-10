import pandas as pd
from src.adaptive_strategy import evaluate_prediction_history, update_strategy_ledger, strategy_summary


def test_prediction_scorecard_ignores_unmatured_and_scores_direction():
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01", "2024-02-01"],
        "ticker": ["AAA", "BBB", "AAA"],
        "predicted_return": [0.1, -0.1, 0.2],
        "actual_forward_return": [0.05, 0.02, None],
        "strategy": ["s", "s", "s"],
    })
    out = evaluate_prediction_history(df)
    assert len(out) == 1
    assert out.iloc[0]["directional_wins"] == 1
    assert out.iloc[0]["directional_losses"] == 1
    assert out.iloc[0]["win_rate_pct"] == 50


def test_strategy_disabled_after_five_consecutive_losses(tmp_path):
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5).astype(str),
        "ticker": ["AAA"] * 5, "strategy": ["bad"] * 5,
        "predicted_return": [0.1] * 5, "actual_forward_return": [-0.01] * 5,
    })
    ledger = update_strategy_ledger(df, tmp_path / "ledger.csv", loss_limit=5, min_resolved=3)
    assert ledger["disabled"].all()
    assert strategy_summary(ledger).iloc[0]["consecutive_losses"] == 5


def test_ledger_deduplicates_outcomes(tmp_path):
    df = pd.DataFrame({"date": ["2024-01-01"], "ticker": ["AAA"], "strategy": ["s"],
                       "predicted_return": [0.1], "actual_forward_return": [0.02]})
    path = tmp_path / "ledger.csv"
    update_strategy_ledger(df, path)
    ledger = update_strategy_ledger(df, path)
    assert len(ledger) == 1
