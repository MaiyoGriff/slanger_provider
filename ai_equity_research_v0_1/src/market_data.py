"""Validated daily adjusted-close ingestion for research and paper trading.

Yahoo Finance via yfinance is a best-effort, unofficial personal-research source.
Alpha Vantage is an optional documented API, but its free tier is rate limited.
Neither source is represented as an exchange-certified production feed.
"""
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data" / "raw" / "adjusted_close.parquet"


def _download_yahoo(ticker: str, start: str, end: str) -> pd.Series:
    import yfinance as yf
    frame = yf.download(ticker, start=start, end=end, auto_adjust=True,
                         actions=False, progress=False, threads=False, timeout=20)
    if frame is None or frame.empty or "Close" not in frame:
        raise RuntimeError(f"No daily prices returned for {ticker}")
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close = close[close > 0]
    if close.empty:
        raise RuntimeError(f"No valid positive closes for {ticker}")
    return close.rename(ticker)


def _download_alpha_vantage(ticker: str, start: str, end: str, api_key: str) -> pd.Series:
    # Optional provider; use only for a small configured universe due to free-tier quotas.
    import requests
    response = requests.get("https://www.alphavantage.co/query", params={
        "function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": ticker,
        "outputsize": "full", "apikey": api_key,
    }, timeout=30)
    response.raise_for_status()
    payload = response.json()
    key = "Time Series (Daily)"
    if key not in payload:
        raise RuntimeError(f"Alpha Vantage returned no daily series for {ticker}: {payload.get('Note') or payload.get('Information') or payload.get('Error Message') or 'unknown response'}")
    frame = pd.DataFrame.from_dict(payload[key], orient="index")
    col = "5. adjusted close" if "5. adjusted close" in frame else "4. close"
    series = pd.to_numeric(frame[col], errors="coerce").dropna()
    series.index = pd.to_datetime(series.index).normalize()
    series = series.loc[(series.index >= pd.Timestamp(start)) & (series.index < pd.Timestamp(end))]
    if series.empty:
        raise RuntimeError(f"No valid Alpha Vantage prices for {ticker}")
    return series.sort_index().rename(ticker)


def update_daily_prices(tickers: list[str], *, start_date: str = "2015-01-01",
                        cache_path: str | Path = DEFAULT_CACHE,
                        provider: str | None = None, max_stale_calendar_days: int = 5,
                        min_coverage_fraction: float = 0.80,
                        benchmark: str = "SPY", today: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Incrementally update cache, validate freshness/coverage, then atomically persist.

    The current calendar date is excluded so an in-progress daily bar cannot be
    mistaken for a finalized close. Market holidays/weekends are allowed by the
    configurable staleness threshold. A failed validation never overwrites cache.
    """
    tickers = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
    if not tickers:
        raise ValueError("No tickers configured")
    provider = (provider or os.getenv("HEDGE_PRICE_PROVIDER", "yfinance")).lower()
    if provider not in {"yfinance", "alpha_vantage"}:
        raise ValueError("HEDGE_PRICE_PROVIDER must be yfinance or alpha_vantage")
    now_date = pd.Timestamp(today or datetime.now(timezone.utc).date()).normalize()
    # End is exclusive; exclude today's potentially incomplete candle.
    end = now_date.strftime("%Y-%m-%d")
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.DataFrame()
    if cache_path.exists():
        try:
            old = pd.read_parquet(cache_path)
            old.index = pd.to_datetime(old.index).tz_localize(None).normalize()
            old = old.sort_index()
        except Exception as exc:
            raise RuntimeError(f"Existing price cache could not be read; refusing to overwrite: {exc}") from exc
    if not old.empty:
        recent_start = max(pd.Timestamp(start_date), old.index.max() - pd.Timedelta(days=10))
    else:
        recent_start = pd.Timestamp(start_date)
    fetch_start = recent_start.strftime("%Y-%m-%d")
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    if provider == "alpha_vantage" and not api_key:
        raise RuntimeError("Set ALPHA_VANTAGE_API_KEY to use Alpha Vantage")

    new_series: dict[str, pd.Series] = {}
    errors: dict[str, str] = {}
    for ticker in tickers:
        try:
            if provider == "yfinance":
                s = _download_yahoo(ticker, fetch_start, end)
            else:
                s = _download_alpha_vantage(ticker, fetch_start, end, api_key)
            new_series[ticker] = s
        except Exception as exc:
            errors[ticker] = str(exc)
    if not new_series:
        raise RuntimeError(f"No tickers downloaded from {provider}; cache was not changed. Errors: {errors}")
    fresh = pd.concat(new_series.values(), axis=1).sort_index()
    if not old.empty:
        combined = pd.concat([old, fresh], axis=0, sort=False)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        # Preserve old cached history for configured tickers when provider fetch fails.
        for ticker in tickers:
            if ticker in old.columns and ticker not in combined.columns:
                combined[ticker] = old[ticker]
    else:
        combined = fresh
    combined.index = pd.to_datetime(combined.index).tz_localize(None).normalize()
    combined = combined.loc[combined.index < now_date]
    combined = combined.apply(pd.to_numeric, errors="coerce")
    combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index()
    available = [t for t in tickers if t in combined and combined[t].notna().any()]
    coverage = len(available) / len(tickers)
    if benchmark not in available:
        raise RuntimeError(f"Required benchmark {benchmark} missing; cache was not changed")
    if coverage < min_coverage_fraction:
        raise RuntimeError(f"Price coverage {coverage:.0%} below minimum {min_coverage_fraction:.0%}; cache was not changed. Errors: {errors}")
    benchmark_dates = combined[benchmark].dropna().index
    if benchmark_dates.empty:
        raise RuntimeError("Benchmark has no valid observations")
    latest = benchmark_dates.max()
    stale_days = (now_date - latest).days
    if stale_days > max_stale_calendar_days:
        raise RuntimeError(f"Benchmark data is stale: last close {latest.date()}, {stale_days} calendar days old. Cache was not changed.")
    stale_tickers = {}
    for ticker in available:
        ticker_dates = combined[ticker].dropna().index
        age = (now_date - ticker_dates.max()).days if len(ticker_dates) else 10**6
        if age > max_stale_calendar_days:
            stale_tickers[ticker] = {"last_date": str(ticker_dates.max().date()) if len(ticker_dates) else None, "age_days": age}
    if stale_tickers:
        raise RuntimeError(f"One or more configured tickers have stale data; cache was not changed: {stale_tickers}")
    # Avoid corrupting the known-good cache: write temp then replace atomically.
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    combined.to_parquet(tmp)
    tmp.replace(cache_path)
    metadata = {
        "provider": provider, "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "latest_benchmark_date": latest.strftime("%Y-%m-%d"), "stale_calendar_days": stale_days,
        "coverage_fraction": coverage, "configured_tickers": len(tickers),
        "available_tickers": len(available), "failed_tickers": errors,
        "price_type": "auto-adjusted daily close", "cache_path": str(cache_path),
    }
    return combined, metadata
