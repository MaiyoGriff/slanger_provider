import numpy as np
import pandas as pd
import pytest

from src.validation import forward_returns, summarize_rank_ic


def test_forward_returns_aligns_future_return_to_signal_date():
    prices = pd.DataFrame({"AAA": [100.0, 110.0, 121.0]}, index=pd.bdate_range("2024-01-01", periods=3))
    result = forward_returns(prices, horizon_days=1)
    assert result.iloc[0, 0] == pytest.approx(0.10)
    assert result.iloc[1, 0] == pytest.approx(0.10)
    assert np.isnan(result.iloc[2, 0])


def test_forward_returns_rejects_invalid_horizon():
    with pytest.raises(ValueError, match="horizon_days"):
        forward_returns(pd.DataFrame({"A": [1.0, 2.0]}), 0)


def test_rank_ic_summary_handles_empty_and_nonempty():
    empty = summarize_rank_ic(pd.Series(dtype=float))
    assert empty["ic_observations"] == 0
    summary = summarize_rank_ic(pd.Series([0.2, -0.1, 0.3]))
    assert summary["ic_observations"] == 3
    assert summary["positive_rank_ic_fraction"] == pytest.approx(2 / 3)
