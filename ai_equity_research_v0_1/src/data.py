from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_prices(tickers: list[str], start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    """Download adjusted close prices for a prototype research universe.

    yfinance/Yahoo data should not be assumed to have commercial redistribution rights.
    """
    if not tickers:
        raise ValueError("The ticker list is empty.")
    prices: dict[str, pd.Series] = {}
    for ticker in sorted(set(t.strip().upper() for t in tickers if t.strip())):
        try:
            frame = yf.download(
                ticker, start=start, end=end, auto_adjust=True,
                actions=False, progress=False, threads=False
            )
            if frame.empty or "Close" not in frame.columns:
                print(f"WARNING: no usable price data for {ticker}")
                continue
            series = frame["Close"]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = pd.to_numeric(series, errors="coerce").dropna()
            series = series[~series.index.duplicated(keep="last")]
            if not series.empty:
                prices[ticker] = series.rename(ticker)
        except Exception as exc:
            print(f"WARNING: failed to download {ticker}: {exc}")
    if not prices:
        raise RuntimeError("No price data downloaded. Check tickers, network, and provider status.")
    result = pd.concat(prices.values(), axis=1).sort_index()
    result.index = pd.to_datetime(result.index)
    result.index.name = "date"
    result = result[~result.index.duplicated(keep="last")]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(RAW_DIR / "adjusted_close.parquet")
    return result
