"""Scheduled daily refresh: validate prices, then update paper simulation only."""
from __future__ import annotations
import json
import logging
from pathlib import Path
import yaml
from src.market_data import update_daily_prices
from src.paper_trading import update_paper_simulator

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
OUT = ROOT / "outputs"
LOG_DIR.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(LOG_DIR / "daily_update.log"), logging.StreamHandler()])
log = logging.getLogger("hedge.daily_update")


def main() -> None:
    with (ROOT / "config" / "settings.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get("daily_update", {})
    tickers = cfg["universe"]["tickers"]
    benchmark = cfg["data"].get("benchmark", "SPY")
    cache = ROOT / data_cfg.get("cache_path", "data/raw/adjusted_close.parquet")
    prices, metadata = update_daily_prices(
        tickers, start_date=cfg["data"].get("start_date", "2015-01-01"),
        cache_path=cache, provider=data_cfg.get("provider", "yfinance"),
        max_stale_calendar_days=int(data_cfg.get("max_stale_calendar_days", 5)),
        min_coverage_fraction=float(data_cfg.get("min_coverage_fraction", 0.80)),
        benchmark=benchmark,
    )
    metadata_path = OUT / "price_feed_status.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    prices.to_parquet(OUT / "prices_snapshot.parquet")
    # Only simulate when provider validation passed. Costs are one-way estimates
    # charged on turnover; half-spread, slippage, and commission are additive.
    cost_cfg = data_cfg.get("execution_costs", {})
    spread = float(cost_cfg.get("half_spread_bps", 5.0))
    slippage = float(cost_cfg.get("slippage_bps", 5.0))
    commission = float(cost_cfg.get("commission_bps", 1.0))
    total_bps = spread + slippage + commission
    paper_cfg = cfg.get("paper_trading", {})
    result = update_paper_simulator(
        prices.astype(float), output_dir=OUT, benchmark=benchmark,
        top_n=int(paper_cfg.get("top_n", cfg["model"].get("top_n", 10))),
        min_history_days=int(paper_cfg.get("min_history_days", cfg["model"].get("min_history_days", 252))),
        transaction_cost_bps=total_bps,
        initial_cash=float(paper_cfg.get("initial_cash", 100000.0)),
        start_date=paper_cfg.get("start_date"),
        max_position_weight=float(paper_cfg.get("max_position_weight", 0.20)),
        max_gross_exposure=float(paper_cfg.get("max_gross_exposure", 1.0)),
    )
    cost_meta = {"half_spread_bps": spread, "slippage_bps": slippage,
                 "commission_bps": commission, "combined_one_way_cost_bps": total_bps,
                 "execution_note": "Daily adjusted-close proxy; costs are assumptions, not observed quotes/fills."}
    (OUT / "paper_execution_assumptions.json").write_text(json.dumps(cost_meta, indent=2), encoding="utf-8")
    log.info("Daily update successful: provider=%s latest=%s coverage=%.1f%%; paper equity=%.2f; assumed cost=%s bps",
             metadata["provider"], metadata["latest_benchmark_date"], metadata["coverage_fraction"]*100,
             float(result["summary"].iloc[0]["latest_equity"]), total_bps)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("Daily update FAILED; paper simulator was not updated after failed price validation")
        raise
