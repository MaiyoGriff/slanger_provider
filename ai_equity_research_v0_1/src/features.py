import pandas as pd


def calculate_features(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Calculate price-only features using adjusted close prices."""
    if prices.empty:
        raise ValueError("Price matrix is empty.")
    prices = prices.sort_index()
    daily_returns = prices.pct_change(fill_method=None)
    momentum_12_1 = prices.shift(21) / prices.shift(252) - 1.0
    momentum_6m = prices / prices.shift(126) - 1.0
    volatility_63d = daily_returns.rolling(63, min_periods=50).std() * (252 ** 0.5)
    return {
        "momentum_12_1": momentum_12_1,
        "momentum_6m": momentum_6m,
        "volatility_63d": volatility_63d,
        "daily_returns": daily_returns,
    }
