import pandas as pd
import pytest
import pickle

@pytest.fixture
def parquet_stub(monkeypatch):
    def to_parquet(self, path, *args, **kwargs):
        with open(path, "wb") as f: pickle.dump(self, f)
    def read_parquet(path, *args, **kwargs):
        with open(path, "rb") as f: return pickle.load(f)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", to_parquet)
    monkeypatch.setattr(pd, "read_parquet", read_parquet)
from src import market_data


def test_update_daily_prices_merges_cache_and_excludes_current_date(tmp_path, monkeypatch, parquet_stub):
    cache = tmp_path / "prices.parquet"
    old = pd.DataFrame({"AAA": [10.0, 11.0], "SPY": [100.0, 101.0]},
                       index=pd.to_datetime(["2025-01-02", "2025-01-03"]))
    old.to_parquet(cache)
    def fake_download(ticker, start, end):
        assert end == "2025-01-07"
        return pd.Series([12.0, 13.0], index=pd.to_datetime(["2025-01-06", "2025-01-07"]), name=ticker)
    monkeypatch.setattr(market_data, "_download_yahoo", fake_download)
    panel, meta = market_data.update_daily_prices(["AAA", "SPY"], start_date="2025-01-01",
        cache_path=cache, provider="yfinance", benchmark="SPY", today="2025-01-07",
        max_stale_calendar_days=5)
    assert pd.Timestamp("2025-01-07") not in panel.index
    assert panel.loc[pd.Timestamp("2025-01-06"), "SPY"] == 12.0
    assert meta["coverage_fraction"] == 1.0


def test_stale_benchmark_does_not_overwrite_cache(tmp_path, monkeypatch, parquet_stub):
    cache = tmp_path / "prices.parquet"
    old = pd.DataFrame({"SPY": [100.0]}, index=pd.to_datetime(["2025-01-01"]))
    old.to_parquet(cache)
    before = cache.read_bytes()
    def fake_download(ticker, start, end):
        return pd.Series([100.0], index=pd.to_datetime(["2025-01-01"]), name=ticker)
    monkeypatch.setattr(market_data, "_download_yahoo", fake_download)
    with pytest.raises(RuntimeError, match="stale"):
        market_data.update_daily_prices(["SPY"], cache_path=cache, provider="yfinance",
            benchmark="SPY", today="2025-01-20", max_stale_calendar_days=5)
    assert cache.read_bytes() == before


def test_missing_benchmark_fails_closed(tmp_path, monkeypatch, parquet_stub):
    cache = tmp_path / "prices.parquet"
    monkeypatch.setattr(market_data, "_download_yahoo", lambda ticker, start, end:
        pd.Series([10.0], index=pd.to_datetime(["2025-01-06"]), name=ticker))
    with pytest.raises(RuntimeError, match="benchmark SPY missing"):
        market_data.update_daily_prices(["AAA"], cache_path=cache, provider="yfinance",
            benchmark="SPY", today="2025-01-07")
    assert not cache.exists()
