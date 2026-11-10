"""Explainable cross-layer diagnostics for Hedge v0.7.

These helpers align existing rankings; they do not turn them into a calibrated
probability or an expected-return guarantee. Missing layers remain missing.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _ticker_index(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None
    out = frame.copy()
    if "ticker" in out.columns:
        out = out.set_index("ticker")
    out.index = out.index.astype(str)
    out.index.name = "ticker"
    return out[~out.index.duplicated(keep="last")]


def build_research_consensus(quant: pd.DataFrame | None,
                             fundamentals: pd.DataFrame | None = None,
                             ml: pd.DataFrame | None = None) -> pd.DataFrame:
    """Combine ordinal ranks transparently; lower ranks are better.

    Consensus is the mean percentile rank across available layers. At least one
    layer is required; the number of contributing layers is explicit.
    """
    layers = {"quant": _ticker_index(quant), "fundamental": _ticker_index(fundamentals), "ml": _ticker_index(ml)}
    available = {k: v for k, v in layers.items() if v is not None}
    if not available:
        return pd.DataFrame(columns=["ticker", "layers_available", "consensus_percentile", "consensus_rank", "signal_disagreement"])
    all_tickers = sorted(set().union(*(set(v.index) for v in available.values())))
    result = pd.DataFrame(index=pd.Index(all_tickers, name="ticker"))
    percentile_cols = []
    for name, frame in available.items():
        rank_col = "rank" if "rank" in frame.columns else ("fundamental_rank" if "fundamental_rank" in frame.columns else None)
        score_col = {"quant": "score", "fundamental": "fundamental_score", "ml": "predicted_return"}.get(name)
        if rank_col:
            ranks = pd.to_numeric(frame[rank_col], errors="coerce")
            pct = ranks.rank(pct=True, ascending=True)
        elif score_col and score_col in frame:
            scores = pd.to_numeric(frame[score_col], errors="coerce")
            pct = scores.rank(pct=True, ascending=False)
        else:
            continue
        col = f"{name}_percentile"
        result[col] = pct.reindex(result.index)
        percentile_cols.append(col)
        rank = ranks if rank_col else scores.rank(ascending=False, method="min")
        result[f"{name}_rank"] = rank.reindex(result.index)
    if not percentile_cols:
        return pd.DataFrame(columns=["ticker", "layers_available", "consensus_percentile", "consensus_rank", "signal_disagreement"])
    vals = result[percentile_cols]
    result["layers_available"] = vals.notna().sum(axis=1)
    result["consensus_percentile"] = vals.mean(axis=1, skipna=True) * 100
    result["consensus_rank"] = result["consensus_percentile"].rank(method="min", ascending=True).astype("Int64")
    result["signal_disagreement"] = vals.std(axis=1, ddof=0) * 100
    result = result.sort_values(["consensus_rank", "signal_disagreement"], ascending=[True, True])
    return result.reset_index()


def evaluate_ml_vs_momentum(predictions: pd.DataFrame | None) -> pd.DataFrame:
    """Per-date rank IC and top-quintile outcomes for ML vs momentum baseline."""
    cols = ["date", "n_stocks", "ml_rank_ic", "momentum_rank_ic", "ml_top_quintile_mean_return", "momentum_top_quintile_mean_return"]
    if predictions is None or predictions.empty:
        return pd.DataFrame(columns=cols)
    required = {"date", "ticker", "predicted_return", "actual_forward_return"}
    if not required.issubset(predictions.columns):
        return pd.DataFrame(columns=cols)
    rows = []
    for date, g in predictions.groupby("date"):
        g = g.copy()
        g["actual_forward_return"] = pd.to_numeric(g["actual_forward_return"], errors="coerce")
        g["predicted_return"] = pd.to_numeric(g["predicted_return"], errors="coerce")
        if "momentum_12_1" not in g:
            g["momentum_12_1"] = np.nan
        g["momentum_12_1"] = pd.to_numeric(g["momentum_12_1"], errors="coerce")
        g = g.dropna(subset=["actual_forward_return", "predicted_return"])
        if len(g) < 3:
            continue
        k = max(1, int(np.ceil(len(g) * .2)))
        ml_top = g.nlargest(k, "predicted_return")
        mom = g.dropna(subset=["momentum_12_1"])
        mom_top = mom.nlargest(max(1, int(np.ceil(len(mom) * .2))), "momentum_12_1") if not mom.empty else mom
        rows.append({"date": pd.Timestamp(date), "n_stocks": len(g),
                     "ml_rank_ic": g["predicted_return"].corr(g["actual_forward_return"], method="spearman"),
                     "momentum_rank_ic": g["momentum_12_1"].corr(g["actual_forward_return"], method="spearman"),
                     "ml_top_quintile_mean_return": ml_top["actual_forward_return"].mean(),
                     "momentum_top_quintile_mean_return": mom_top["actual_forward_return"].mean() if not mom_top.empty else np.nan})
    return pd.DataFrame(rows, columns=cols).sort_values("date") if rows else pd.DataFrame(columns=cols)


def data_quality_summary(prices: pd.DataFrame | None) -> dict[str, object]:
    """Basic coverage diagnostics for a downloaded price matrix."""
    if prices is None or prices.empty:
        return {"available": False, "rows": 0, "tickers": 0, "missing_cell_fraction": np.nan, "latest_date": None, "tickers_with_under_252_observations": 0}
    px = prices.apply(pd.to_numeric, errors="coerce")
    counts = px.notna().sum()
    return {"available": True, "rows": int(len(px)), "tickers": int(px.shape[1]),
            "missing_cell_fraction": float(px.isna().to_numpy().mean()),
            "latest_date": str(pd.Timestamp(px.index.max()).date()),
            "tickers_with_under_252_observations": int((counts < 252).sum())}
