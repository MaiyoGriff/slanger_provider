"""Integrated Hedge research dashboard. Run with: streamlit run app.py"""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from src.research_diagnostics import build_research_consensus, evaluate_ml_vs_momentum, data_quality_summary
from src.adaptive_strategy import evaluate_prediction_history, update_strategy_ledger, strategy_summary
from src.adaptive_engine import run_adaptive_search
from src.adaptive_validation import evaluate_final_holdout, update_shadow_paper_ledger, DEFAULT_CONFIGS as VALIDATION_CONFIGS
from src.paper_trading import update_paper_simulator
from src.features import calculate_features

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
st.set_page_config(page_title="Hedge Research Platform v1.0", layout="wide")
st.title("Hedge — Adaptive Quant Research v1.2")
st.caption("Research prototype only. Scores are not investment advice; no live execution. Check data licensing and point-in-time limitations before commercial use.")


def load_csv(name, parse_dates=None):
    path = OUT / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, parse_dates=parse_dates or [])
    except Exception as exc:
        st.warning(f"Could not read {name}: {exc}")
        return None

ranking = load_csv("latest_ranking.csv")
fund = load_csv("fundamental_ranking.csv")
ml = load_csv("ml_latest_ranking.csv")
comparison = load_csv("comparison_backtest.csv", ["date"])
periods = load_csv("period_performance.csv")
ic = load_csv("rank_ic.csv")
ml_history = load_csv("ml_walk_forward_predictions.csv", ["date"])
try:
    prices_snapshot = pd.read_parquet(OUT / "prices_snapshot.parquet") if (OUT / "prices_snapshot.parquet").exists() else None
except Exception:
    prices_snapshot = None

pages = ["Overview", "Quantitative ranking", "Fundamentals", "ML predictions", "Backtest & validation", "Research decision support", "Prediction scorecard & adaptation", "Data & limitations"]
page = st.sidebar.radio("Research section", pages)

if page == "Overview":
    st.subheader("Research pipeline status")
    files = {"Price ranking": ranking, "Fundamental ranking": fund, "ML ranking": ml,
             "Strategy comparison": comparison, "Period performance": periods, "Rank IC": ic}
    cols = st.columns(3)
    for i, (name, frame) in enumerate(files.items()):
        with cols[i % 3]:
            st.metric(name, "Available" if frame is not None and not frame.empty else "Not available")
    st.markdown("### Current research layers")
    st.write("Price momentum and risk scoring → fundamental value/quality/growth → experimental ML ranking → historical strategy comparison.")
    st.info("Run `python -m src.main` from the project root to refresh outputs. A file being available does not mean the model is validated.")
    feed_status = OUT / "price_feed_status.json"
    if feed_status.exists():
        import json
        try:
            feed = json.loads(feed_status.read_text(encoding="utf-8"))
            st.caption(f"Last validated daily feed: {feed.get('provider')} · latest benchmark date {feed.get('latest_benchmark_date')} · coverage {feed.get('coverage_fraction', 0):.0%}")
        except Exception:
            st.warning("Price feed status metadata could not be read.")
    st.caption("Scheduled refresh: run `python -m src.daily_update`; install cron with `bash scripts/install_daily_cron.sh`. The job stops if freshness/coverage checks fail.")
    if ranking is not None and not ranking.empty:
        st.markdown("### Latest quantitative ranking")
        st.dataframe(ranking.head(10), use_container_width=True, hide_index=True)
elif page == "Quantitative ranking":
    st.subheader("Price-based ranking")
    if ranking is None or ranking.empty:
        st.warning("No ranking output found. Run `python -m src.main` first.")
    else:
        st.dataframe(ranking, use_container_width=True, hide_index=True)
        st.download_button("Download ranking CSV", ranking.to_csv(index=False).encode(), "latest_ranking.csv", "text/csv")
elif page == "Fundamentals":
    st.subheader("Fundamental value, quality and growth")
    if fund is None or fund.empty:
        st.warning("No fundamental ranking found. Check provider access and `fundamentals.enabled` in config/settings.yaml.")
    else:
        st.caption("Provider's latest available metrics; not point-in-time historical fundamentals.")
        st.dataframe(fund, use_container_width=True, hide_index=False)
        st.download_button("Download fundamental ranking", fund.to_csv(index=False).encode(), "fundamental_ranking.csv", "text/csv")
elif page == "ML predictions":
    st.subheader("Experimental forward-return predictions")
    if ml is None or ml.empty:
        st.warning("No ML ranking found. Run the pipeline and inspect its ML warnings.")
    else:
        st.warning("Predictions are model estimates, not calibrated probabilities or guaranteed returns. Validate against simple baselines before use.")
        st.dataframe(ml, use_container_width=True, hide_index=False)
        st.download_button("Download ML ranking", ml.to_csv(index=False).encode(), "ml_latest_ranking.csv", "text/csv")
elif page == "Research decision support":
    st.subheader("Cross-layer research view")
    st.caption("Ranks are aligned for comparison only. Consensus is not a calibrated return forecast or a buy/sell signal.")
    consensus = build_research_consensus(ranking, fund, ml)
    if consensus.empty:
        st.warning("No ranking layers are available. Run the research pipeline first.")
    else:
        st.markdown("### Cross-layer consensus and disagreement")
        st.write("Consensus percentile is the mean percentile rank across available layers (lower is better). Signal disagreement is the standard deviation of those percentiles. Missing layers are not imputed.")
        st.dataframe(consensus, use_container_width=True, hide_index=True)
        st.download_button("Download research consensus", consensus.to_csv(index=False).encode(), "research_consensus.csv", "text/csv")
    st.markdown("### ML versus momentum baseline")
    if ml_history is None or ml_history.empty:
        st.info("No walk-forward prediction history found. ML comparison will appear after a successful pipeline run.")
    else:
        evaluation = evaluate_ml_vs_momentum(ml_history)
        if evaluation.empty:
            st.warning("Prediction file is missing required columns or has too few valid observations.")
        else:
            st.dataframe(evaluation.tail(60), use_container_width=True, hide_index=True)
            st.download_button("Download ML baseline evaluation", evaluation.to_csv(index=False).encode(), "ml_vs_momentum_evaluation.csv", "text/csv")
            st.caption("Forward returns overlap when the forecast horizon exceeds rebalance frequency; rows are not independent observations. This diagnostic is descriptive, not proof of predictive edge.")
    st.markdown("### Price-data coverage")
    quality = data_quality_summary(prices_snapshot)
    qcols = st.columns(4)
    qcols[0].metric("Price rows", quality["rows"])
    qcols[1].metric("Tickers", quality["tickers"])
    qcols[2].metric("Missing price cells", "—" if pd.isna(quality["missing_cell_fraction"]) else f"{quality["missing_cell_fraction"]:.1%}")
    qcols[3].metric("Tickers under 252 observations", quality["tickers_with_under_252_observations"])
    st.write(f"Latest date in snapshot: {quality["latest_date"] or "not available"}")
elif page == "Backtest & validation":
    st.subheader("Strategy comparison and validation")
    if comparison is not None and not comparison.empty:
        equity_cols = [c for c in ["hedge_equity", "momentum_equity", "benchmark_equity"] if c in comparison]
        if "date" in comparison and equity_cols:
            long = comparison.melt(id_vars="date", value_vars=equity_cols, var_name="Portfolio", value_name="Growth of $1")
            st.plotly_chart(px.line(long, x="date", y="Growth of $1", color="Portfolio"), use_container_width=True)
        st.dataframe(comparison.tail(30), use_container_width=True, hide_index=True)
        st.download_button("Download comparison returns", comparison.to_csv(index=False).encode(), "comparison_backtest.csv", "text/csv")
    else:
        st.warning("Comparison backtest is unavailable. Run the pipeline first.")
    if periods is not None and not periods.empty:
        st.markdown("### Performance by period")
        st.dataframe(periods, use_container_width=True, hide_index=True)
    if ic is not None and not ic.empty:
        st.markdown("### Rank information coefficient")
        st.dataframe(ic.tail(60), use_container_width=True, hide_index=True)
    st.markdown("### Adaptive parameter search (walk-forward, net of costs)")
    st.caption("Each fold chooses parameters using only its prior training window, then evaluates the selected configuration on the next unseen window. This ranks research candidates; it does not execute trades.")
    if prices_snapshot is None or prices_snapshot.empty:
        st.info("Adaptive search needs outputs/prices_snapshot.parquet. Run the research pipeline first.")
    else:
        prices_for_search = prices_snapshot.copy()
        if "date" in prices_for_search.columns:
            prices_for_search["date"] = pd.to_datetime(prices_for_search["date"], errors="coerce")
            prices_for_search = prices_for_search.set_index("date")
        prices_for_search.index = pd.to_datetime(prices_for_search.index, errors="coerce")
        prices_for_search = prices_for_search.loc[~prices_for_search.index.isna()].sort_index()
        bench_options = [c for c in prices_for_search.columns if c.upper() == "SPY"]
        if not bench_options:
            st.warning("SPY benchmark column is required for regime classification and benchmark-relative research.")
        else:
            cost_bps = st.number_input("Transaction cost (basis points per turnover unit)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
            train_days = st.select_slider("Minimum training window (trading days)", options=[504, 756, 1008, 1260], value=756)
            test_days = st.select_slider("Out-of-sample fold length (trading days)", options=[63, 126, 189, 252], value=126)
            if st.button("Run adaptive parameter search", type="primary"):
                with st.spinner("Backtesting parameter configurations and scoring unseen folds..."):
                    try:
                        feat = calculate_features(prices_for_search.astype(float))
                        detail, regime_summary = run_adaptive_search(prices_for_search.astype(float), feat,
                            benchmark=bench_options[0], transaction_cost_bps=cost_bps,
                            min_train_days=int(train_days), test_days=int(test_days))
                        if detail.empty:
                            st.warning("Not enough price history for the selected walk-forward settings.")
                        else:
                            OUT.mkdir(parents=True, exist_ok=True)
                            detail.to_csv(OUT / "adaptive_walk_forward_results.csv", index=False)
                            regime_summary.to_csv(OUT / "adaptive_regime_priority.csv", index=False)
                            st.markdown("#### Out-of-sample fold results")
                            st.dataframe(detail, use_container_width=True, hide_index=True)
                            st.markdown("#### Market-regime research priority")
                            st.dataframe(regime_summary, use_container_width=True, hide_index=True)
                            st.download_button("Download walk-forward results", detail.to_csv(index=False).encode(), "adaptive_walk_forward_results.csv", "text/csv")
                            st.download_button("Download regime priorities", regime_summary.to_csv(index=False).encode(), "adaptive_regime_priority.csv", "text/csv")
                            st.warning("Prioritize further testing, not capital allocation, based on these rankings. Use untouched final holdouts, realistic spread/slippage assumptions, point-in-time universes and stability checks before any deployment.")
                    except Exception as exc:
                        st.error(f"Adaptive search failed: {exc}")
            st.markdown("### Final untouched holdout & statistical checks")
            st.caption("Compare predeclared configurations on the last chronological slice. Treat this as a one-time test: changing parameters after viewing these results contaminates the holdout. HAC p-values are exploratory and do not correct for all strategy-search or data biases.")
            holdout_fraction = st.slider("Final holdout fraction", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
            if st.button("Evaluate fixed configurations on final holdout"):
                with st.spinner("Evaluating fixed configurations on the final holdout..."):
                    try:
                        feat = calculate_features(prices_for_search.astype(float))
                        holdout_results, holdout_start = evaluate_final_holdout(
                            prices_for_search.astype(float), feat, benchmark=bench_options[0],
                            transaction_cost_bps=cost_bps, holdout_fraction=holdout_fraction)
                        OUT.mkdir(parents=True, exist_ok=True)
                        holdout_results.to_csv(OUT / "final_holdout_results.csv", index=False)
                        st.info(f"Holdout window starts {holdout_start}. Do not use this same period repeatedly for tuning.")
                        st.dataframe(holdout_results, use_container_width=True, hide_index=True)
                        st.download_button("Download final holdout report", holdout_results.to_csv(index=False).encode(), "final_holdout_results.csv", "text/csv")
                    except Exception as exc:
                        st.error(f"Final holdout evaluation failed: {exc}")
            st.markdown("### Live-updating event-driven paper simulator")
            st.caption("Freezes monthly momentum signals at the signal close, simulates execution at the next available close in the supplied adjusted-close data, records fills and matured forecast outcomes, and tracks portfolio equity/drawdown/volatility. This is a local simulation only; no broker or live orders are connected.")
            paper_top_n = st.select_slider("Paper portfolio holdings", options=[5, 10, 15, 20], value=10, key="paper_top_n")
            paper_history = st.select_slider("Paper signal minimum history", options=[126, 252, 378], value=252, key="paper_history")
            paper_cash = st.number_input("Paper starting cash ($)", min_value=1000.0, max_value=10000000.0, value=100000.0, step=10000.0, key="paper_cash")
            paper_start_event = st.date_input("Start recording paper signals from", value=prices_for_search.index[max(0, len(prices_for_search)-252)].date(), key="paper_start_event")
            if st.button("Update event-driven paper simulator", type="primary"):
                try:
                    result = update_paper_simulator(prices_for_search.astype(float), output_dir=OUT,
                        benchmark=bench_options[0], top_n=int(paper_top_n), min_history_days=int(paper_history),
                        transaction_cost_bps=float(cost_bps), initial_cash=float(paper_cash),
                        start_date=paper_start_event.isoformat())
                    summary = result["summary"].iloc[0]
                    mcols = st.columns(4)
                    mcols[0].metric("Paper equity", f"${summary['latest_equity']:,.2f}")
                    mcols[1].metric("Total return", f"{summary['total_return_pct']:.2f}%")
                    mcols[2].metric("Max drawdown", f"{summary['max_drawdown_pct']:.2f}%")
                    mcols[3].metric("Forecast win rate", "—" if pd.isna(summary['forecast_directional_win_rate_pct']) else f"{summary['forecast_directional_win_rate_pct']:.1f}%")
                    st.markdown("#### Portfolio risk and benchmark")
                    st.dataframe(result["summary"], use_container_width=True, hide_index=True)
                    st.markdown("#### Simulated fills")
                    st.dataframe(result["fills"].tail(40), use_container_width=True, hide_index=True)
                    st.markdown("#### Forecasts vs realized outcomes")
                    st.dataframe(result["predictions"].tail(50), use_container_width=True, hide_index=True)
                    st.markdown("#### Portfolio equity and drawdown")
                    st.plotly_chart(px.line(result["daily"], x="date", y="equity", title="Paper account equity"), use_container_width=True)
                    st.plotly_chart(px.line(result["daily"], x="date", y="drawdown_pct", title="Drawdown (%)"), use_container_width=True)
                    for label, key in [("Download predictions", "predictions"), ("Download fills", "fills"), ("Download daily risk ledger", "daily"), ("Download positions", "positions"), ("Download risk summary", "summary")]:
                        st.download_button(label, result[key].to_csv(index=False).encode(), f"paper_{key}.csv", "text/csv", key=f"download_paper_{key}")
                    st.warning("Execution prices are adjusted-close proxies because the current dataset does not contain open/bid/ask quotes. Forecast direction is derived from a trailing momentum score, not a calibrated expected-return model. Paper results are hypothetical and may differ materially from tradable results.")
                except Exception as exc:
                    st.error(f"Paper simulator update failed: {exc}")

            st.markdown("### Shadow paper-trading ledger")
            st.caption("Creates a hypothetical, cost-adjusted portfolio-return ledger for observation. No broker connection, orders, or live capital are used. Choose a configuration before examining holdout results when possible.")
            paper_config_name = st.selectbox("Fixed paper configuration", [c["name"] for c in VALIDATION_CONFIGS], index=1)
            paper_start = st.date_input("Paper ledger start date", value=prices_for_search.index[max(0, len(prices_for_search)-126)].date())
            if st.button("Update shadow paper ledger"):
                try:
                    chosen_cfg = next(c for c in VALIDATION_CONFIGS if c["name"] == paper_config_name)
                    feat = calculate_features(prices_for_search.astype(float))
                    ledger = update_shadow_paper_ledger(prices_for_search.astype(float), feat,
                        ledger_path=OUT / "shadow_paper_trading_ledger.csv", config=chosen_cfg,
                        benchmark=bench_options[0], transaction_cost_bps=cost_bps,
                        start_date=paper_start.isoformat())
                    selected_ledger = ledger[ledger["config"] == paper_config_name].copy()
                    st.success(f"Shadow ledger updated: {len(selected_ledger)} daily records for {paper_config_name}.")
                    st.dataframe(selected_ledger.tail(60), use_container_width=True, hide_index=True)
                    st.download_button("Download shadow paper ledger", ledger.to_csv(index=False).encode(), "shadow_paper_trading_ledger.csv", "text/csv")
                except Exception as exc:
                    st.error(f"Shadow paper ledger update failed: {exc}")
elif page == "Prediction scorecard & adaptation":
    st.subheader("Prediction scorecard & strategy adaptation")
    st.caption("Only forecasts with matured realized returns are scored. Historical outcomes are evidence for research, not a guarantee of future performance.")
    if ml_history is None or ml_history.empty:
        st.warning("No walk-forward prediction history found. Run `python -m src.main` first.")
    else:
        history = ml_history.copy()
        if "strategy" not in history:
            history["strategy"] = "random_forest_price_only"
        scorecard = evaluate_prediction_history(history)
        if scorecard.empty:
            st.info("No matured forecast outcomes are available yet. The forward-return horizon must finish before a prediction can be scored.")
        else:
            st.markdown("### Timeline: prediction outcomes")
            st.dataframe(scorecard.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
            st.download_button("Download prediction scorecard", scorecard.to_csv(index=False).encode(), "prediction_scorecard.csv", "text/csv")
            st.markdown("### Strategy status")
            ledger_path = OUT / "strategy_outcome_ledger.csv"
            ledger = update_strategy_ledger(history, ledger_path, loss_limit=5, min_resolved=20)
            summary = strategy_summary(ledger)
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.caption("A strategy is flagged after five consecutive directional losses. Promotion requires at least 20 resolved outcomes, positive mean realized return, and no disable flag. A disabled strategy is not silently reactivated. Directional win rate alone is not enough: payoff size, costs, drawdown, sample size and out-of-sample stability matter.")
            st.download_button("Download strategy outcome ledger", ledger.to_csv(index=False).encode(), "strategy_outcome_ledger.csv", "text/csv")
            st.warning("This is a research gate, not an autonomous live-trading switch. Strategy changes should be validated on untouched walk-forward periods and include realistic spreads, slippage, fees and point-in-time data.")
else:
    st.subheader("Data, licensing and model limitations")
    st.markdown("""
- **Market data:** price history currently depends on the configured Yahoo Finance/yfinance workflow. Verify current provider terms and licensing before commercial use or redistribution.
- **Fundamentals:** latest reported metrics are not guaranteed to reflect what was publicly available on each historical date. Do not use them for historical backtests without point-in-time statement data and publication timestamps.
- **Universe bias:** a manually configured current ticker list can introduce survivorship and selection bias; global listings also require currency, exchange, corporate-action and delisting handling.
- **ML validity:** a walk-forward split helps control leakage but does not establish predictive power. Compare against SPY and momentum, include costs, and reserve an untouched test period.
- **Valuation:** percentile scores are relative screens, not intrinsic fair-value estimates. Sector-specific metrics and accounting differences matter.
- **Execution:** this dashboard does not place orders and is not connected to a broker.
""")

st.sidebar.caption("Hedge v1.1 · Adaptive research and paper simulation")
