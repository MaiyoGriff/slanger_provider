import numpy as np
import pandas as pd
import pytest

from src.features import calculate_features
from src.ml_model import build_ml_panel, walk_forward_predictions


def prices_fixture(n=700):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(7)
    values = {}
    for i, ticker in enumerate(["AAA", "BBB", "CCC"]):
        rets = rng.normal(0.0003 * (i + 1), 0.01, n)
        values[ticker] = 100 * np.cumprod(1 + rets)
    return pd.DataFrame(values, index=idx)


def test_build_panel_has_expected_columns_and_forward_labels():
    prices = prices_fixture()
    panel = build_ml_panel(prices, calculate_features(prices), horizon_days=20)
    assert set(["date", "ticker", "forward_return", "momentum_12_1"]).issubset(panel.columns)
    assert len(panel) == prices.size
    assert panel.groupby("ticker")["forward_return"].apply(lambda x: x.tail(20).isna().all()).all()


def test_invalid_horizon_raises():
    with pytest.raises(ValueError):
        build_ml_panel(prices_fixture(), calculate_features(prices_fixture()), horizon_days=0)


def test_walk_forward_returns_expected_schema():
    prices = prices_fixture()
    out = walk_forward_predictions(prices, calculate_features(prices), horizon_days=20,
                                   min_train_rows=80, random_state=1)
    assert set(out.columns) == {"date", "ticker", "predicted_return", "actual_forward_return"}
    assert out.empty or out["predicted_return"].notna().all()
    # All prediction dates should be later than the last training-label cutoff by construction.\n