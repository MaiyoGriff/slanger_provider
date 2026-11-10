"""Low-cost fundamental data retrieval for research prototypes.

Uses yfinance's Yahoo Finance interface. This is not a licensed institutional
point-in-time fundamentals feed. Check provider terms before commercial use.
"""
from __future__ import annotations

from typing import Any
import pandas as pd
import yfinance as yf

INFO_FIELDS = {
    "company_name": "longName",
    "sector": "sector",
    "industry": "industry",
    "country": "country",
    "currency": "currency",
    "market_cap": "marketCap",
    "trailing_pe": "trailingPE",
    "forward_pe": "forwardPE",
    "price_to_book": "priceToBook",
    "price_to_sales": "priceToSalesTrailing12Months",
    "ev_to_ebitda": "enterpriseToEbitda",
    "return_on_equity": "returnOnEquity",
    "profit_margin": "profitMargins",
    "operating_margin": "operatingMargins",
    "revenue_growth": "revenueGrowth",
    "earnings_growth": "earningsGrowth",
    "debt_to_equity": "debtToEquity",
    "free_cash_flow": "freeCashflow",
    "operating_cash_flow": "operatingCashflow",
    "total_cash": "totalCash",
    "total_debt": "totalDebt",
    "current_ratio": "currentRatio",
    "shares_outstanding": "sharesOutstanding",
    "fiscal_year_end": "lastFiscalYearEnd",
    "most_recent_quarter": "mostRecentQuarter",
    "financial_currency": "financialCurrency",
    "quote_type": "quoteType",
}

def _safe_number(value: Any) -> float:
    try:
        number = float(value)
        return number if pd.notna(number) else float("nan")
    except (TypeError, ValueError):
        return float("nan")

def fetch_fundamentals(tickers: list[str], *, strict: bool = False) -> pd.DataFrame:
    """Fetch latest available summary fundamentals for each symbol.

    Failed symbols are skipped by default so fundamental data outages do not
    break the established price-only v0.3 pipeline. Set strict=True to raise
    if no symbol returns any data. Values are latest provider values, not
    point-in-time historical snapshots.
    """
    if not tickers:
        raise ValueError("The fundamentals ticker list is empty.")
    rows: list[dict[str, Any]] = []
    for raw_symbol in sorted(set(str(t).strip().upper() for t in tickers if str(t).strip())):
        try:
            info = yf.Ticker(raw_symbol).get_info()
            if not isinstance(info, dict) or not info:
                print(f"WARNING: no fundamental data for {raw_symbol}")
                continue
            row: dict[str, Any] = {"ticker": raw_symbol}
            for output, source in INFO_FIELDS.items():
                value = info.get(source)
                if output not in {"company_name", "sector", "industry", "country", "currency", "financial_currency", "quote_type"}:
                    value = _safe_number(value)
                row[output] = value
            rows.append(row)
        except Exception as exc:
            print(f"WARNING: fundamental fetch failed for {raw_symbol}: {exc}")
    if not rows:
        if strict:
            raise RuntimeError("No fundamental data downloaded. Check symbols, network, and provider status.")
        return pd.DataFrame(columns=["ticker", *INFO_FIELDS.keys()]).set_index("ticker")
    result = pd.DataFrame(rows).set_index("ticker").sort_index()
    result.index.name = "ticker"
    return result
