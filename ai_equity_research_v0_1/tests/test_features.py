import pandas as pd

from src.features import calculate_features


def test_feature_shapes_and_volatility():
    dates = pd.bdate_range("2020-01-01", periods=320)
    prices = pd.DataFrame({"AAA": range(100, 420)}, index=dates, dtype=float)
    features = calculate_features(prices)
    assert features["momentum_12_1"].shape == prices.shape
    assert features["volatility_63d"].shape == prices.shape
    assert pd.isna(features["daily_returns"].iloc[0, 0])
    assert pd.notna(features["momentum_12_1"].iloc[-1, 0])
