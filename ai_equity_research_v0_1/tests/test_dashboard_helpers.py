from pathlib import Path

def test_dashboard_mentions_all_research_layers():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    for phrase in ["Quantitative ranking", "Fundamentals", "ML predictions", "Backtest & validation", "Data & limitations"]:
        assert phrase in source

def test_dashboard_has_no_live_execution_claim():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "does not place orders" in source


def test_dashboard_has_v07_decision_support():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "Research decision support" in source
    assert "ML versus momentum baseline" in source
    assert "Price-data coverage" in source
