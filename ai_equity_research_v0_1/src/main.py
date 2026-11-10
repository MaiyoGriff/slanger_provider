from pathlib import Path

import pandas as pd
import yaml

from src.backtest import period_performance, run_backtest, run_strategy_comparison, slice_evaluation_period
from src.data import download_prices
from src.features import calculate_features
from src.fundamentals import fetch_fundamentals
from src.fundamental_scoring import score_fundamentals
from src.report import export_ranking, performance_summary
from src.scoring import rank_stocks
from src.validation import rank_ic_series, summarize_rank_ic
from src.ml_model import latest_ml_ranking, walk_forward_predictions

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True, parents=True)


def _strategy_summary(results: pd.DataFrame, start: str, end: str | None = None) -> dict:
    sample = slice_evaluation_period(results, start, end)
    if sample.empty:
        raise ValueError(f"No observations in evaluation window {start} to {end or 'latest'}")
    summary = {}
    for label, return_col, cost_col, turnover_col in (
        ("hedge", "hedge_return", "hedge_transaction_cost", "hedge_turnover"),
        ("momentum", "momentum_return", "momentum_transaction_cost", "momentum_turnover"),
        ("benchmark", "benchmark_return", None, None),
    ):
        frame = pd.DataFrame({"strategy_return": sample[return_col], "benchmark_return": sample["benchmark_return"]}, index=sample.index)
        if cost_col:
            frame["transaction_cost"] = sample[cost_col]
        if turnover_col:
            frame["turnover"] = sample[turnover_col]
        metrics = performance_summary(frame)
        for key, value in metrics.items():
            if key.startswith("strategy_") or key in {"total_transaction_cost_return_units", "average_daily_turnover", "total_one_way_turnover"}:
                summary[f"{label}_{key}"] = value
    summary["evaluation_start"] = str(sample.index.min().date())
    summary["evaluation_end"] = str(sample.index.max().date())
    summary["observations"] = int(len(sample))
    return summary


def main() -> None:
    with (ROOT / "config" / "settings.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    prices = download_prices(config["universe"]["tickers"], start=config["data"]["start_date"])
    prices.to_parquet(OUTPUTS / "prices_snapshot.parquet")

    # Fundamentals are additive and fail-soft: an API outage must not prevent
    # the already-working price-only v0.3 pipeline from completing.
    fundamental_cfg = config.get("fundamentals", {})
    if fundamental_cfg.get("enabled", True):
        fundamental_tickers = fundamental_cfg.get("tickers", config["universe"]["tickers"])
        try:
            fundamentals = fetch_fundamentals(fundamental_tickers, strict=False)
            if not fundamentals.empty:
                fundamentals.to_csv(OUTPUTS / "fundamentals_raw.csv", index_label="ticker")
                scored_fundamentals = score_fundamentals(fundamentals)
                scored_fundamentals.to_csv(OUTPUTS / "fundamental_ranking.csv", index_label="ticker")
                print("\nLatest fundamental ranking (latest provider data; not point-in-time):")
                print(scored_fundamentals.head(10).to_string())
            else:
                print("WARNING: fundamental fetch returned no rows; continuing price-only pipeline.")
        except Exception as exc:
            print(f"WARNING: fundamental analysis skipped; continuing price-only pipeline: {exc}")
    benchmark = config["data"]["benchmark"]
    if benchmark not in prices.columns:
        raise RuntimeError(f"Benchmark {benchmark!r} missing from downloaded data.")
    features = calculate_features(prices)

    # v0.5 ML is additive and fail-soft; baseline outputs remain available if
    # training is not possible with the configured history/universe.
    ml_cfg = config.get("machine_learning", {})
    if ml_cfg.get("enabled", True):
        try:
            horizon = int(ml_cfg.get("horizon_days", config["model"].get("horizon_days", 126)))
            min_train_rows = int(ml_cfg.get("min_train_rows", 250))
            ml_predictions = walk_forward_predictions(
                prices, features, horizon_days=horizon, min_train_rows=min_train_rows,
                random_state=int(ml_cfg.get("random_state", 42)),
            )
            ml_predictions["strategy"] = "random_forest_price_only"
            ml_predictions.to_csv(OUTPUTS / "ml_walk_forward_predictions.csv", index=False)
            latest_ml = latest_ml_ranking(
                prices, features, horizon_days=horizon, min_train_rows=min_train_rows,
                random_state=int(ml_cfg.get("random_state", 42)),
            )
            latest_ml.to_csv(OUTPUTS / "ml_latest_ranking.csv", index_label="ticker")
            print("\nLatest ML research ranking (not an investment recommendation):")
            print(latest_ml.head(10).to_string())
        except Exception as exc:
            print(f"WARNING: ML research layer skipped; baseline pipeline continues: {exc}")

    min_history = int(config["model"].get("min_history_days", 252))
    ranking = rank_stocks(features, min_history_days=min_history)
    export_ranking(ranking)

    top_n = int(config["model"]["top_n"])
    costs = float(config["backtest"]["transaction_cost_bps"])
    # Keep the legacy output for the existing dashboard, now with next-close execution.
    legacy = run_backtest(prices, features, benchmark, top_n, costs, min_history)
    legacy.to_csv(OUTPUTS / "backtest.csv", index_label="date")
    legacy_summary = performance_summary(legacy)
    with (OUTPUTS / "performance_summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(legacy_summary, handle, sort_keys=False)

    comparison = run_strategy_comparison(prices, features, benchmark, top_n, costs, min_history)
    comparison.to_csv(OUTPUTS / "comparison_backtest.csv", index_label="date")

    validation_cfg = config.get("validation", {})
    oos_start = str(validation_cfg.get("evaluation_start", "2024-01-01"))
    oos_end = validation_cfg.get("evaluation_end")
    oos_summary = _strategy_summary(comparison, oos_start, oos_end)
    with (OUTPUTS / "oos_summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(oos_summary, handle, sort_keys=False)

    periods = validation_cfg.get("periods", {
        "2019_2020": ["2019-01-01", "2020-12-31"],
        "2021_2022": ["2021-01-01", "2022-12-31"],
        "2023": ["2023-01-01", "2023-12-31"],
        "out_of_sample": [oos_start, oos_end],
    })
    period_map = {name: (values[0], values[1] if len(values) > 1 else None) for name, values in periods.items()}
    period_performance(comparison, period_map).to_csv(OUTPUTS / "period_performance.csv", index=False)

    ic = rank_ic_series(prices, features, horizon_days=int(config["model"].get("horizon_days", 126)), min_history_days=min_history, benchmark=benchmark)
    ic.to_csv(OUTPUTS / "rank_ic.csv", header=True, index_label="date")
    ic_oos = ic.loc[ic.index >= pd.Timestamp(oos_start)]
    with (OUTPUTS / "rank_ic_summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summarize_rank_ic(ic_oos), handle, sort_keys=False)

    print("\nLatest ranking:")
    print(ranking.head(10).to_string())
    print("\nFull-period Hedge vs SPY summary:")
    print(pd.Series(legacy_summary).to_string())
    print(f"\nOut-of-sample comparison ({oos_start} to {oos_end or 'latest'}):")
    print(pd.Series(oos_summary).to_string())
    print("\nPeriod performance:")
    print(pd.read_csv(OUTPUTS / "period_performance.csv").to_string(index=False))
    print(f"\nFiles written to: {OUTPUTS.resolve()}")


if __name__ == "__main__":
    main()
