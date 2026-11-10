# Hedge — Adaptive Quant Research v1.0

A research prototype with quantitative ranking, fundamental screening, walk-forward ML experiments, historical backtesting, and cross-layer decision-support diagnostics.

**Scope and safety**
- Research only: no broker connection, order routing, or live execution.
- The ranking is a price-only baseline, not a complete fundamental valuation or predictive AI model.
- Educational software, not investment advice. Historical performance does not guarantee future results.
- `yfinance` is for prototyping; review data terms and commercial redistribution rights before commercial use.
- The current ticker list is a smoke-test universe, not a representative point-in-time US equity universe.

## Install and test

Python 3.11 recommended (3.10+ should work).

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

`pytest.ini` configures the project root on Python's import path so `from src...` imports work when tests are run from the repository root.

## Run

```bash
python -m src.main
streamlit run app.py
```

The pipeline writes `outputs/latest_ranking.csv`, `outputs/latest_ranking.html`, `outputs/backtest.csv`, `outputs/performance_summary.yaml`, and `outputs/prices_snapshot.parquet`.

## Model formulas

- 12–1 momentum = adjusted close at t−21 sessions / adjusted close at t−252 sessions − 1.
- 6-month momentum = adjusted close at t / adjusted close at t−126 sessions − 1 (reported as an additional feature; not used in the v0.2 composite score).
- 63-day annualized volatility = standard deviation of daily returns over trailing 63 sessions × √252.
- Momentum score = cross-sectional percentile rank × 100.
- Low-volatility score = 100 − cross-sectional volatility percentile rank × 100.
- Composite score = 0.75 × momentum score + 0.25 × low-volatility score.

The backtest uses monthly ranking dates and prior-session weights for current-session returns, avoiding use of a closing price and that same day's return in the portfolio return. Transaction costs are a configurable turnover-based assumption, not a market-impact model.

## Performance report

`performance_summary.yaml` now reports strategy and benchmark total return, annualized return, annualized volatility, Sharpe ratio, Sortino ratio, maximum drawdown, Calmar ratio, win rate, best/worst daily return and observation count. It also reports total-return difference versus benchmark, transaction costs and turnover where available. The risk-free rate defaults to zero. Metrics are descriptive and can be unstable on short samples.

## Known limitations and next work

1. No point-in-time universe, delisted securities, or survivorship-bias controls.
2. No point-in-time fundamentals, valuation, estimates, news, or catalyst analysis.
3. A monthly portfolio backtest is not equivalent to evaluating a 126-session forecast. Add forward-return labels, rank-IC and walk-forward validation before making predictive claims.
4. Missing/stale prices, corporate actions, liquidity, slippage, taxes, sector exposures, and trading constraints need further validation.
5. The backtest remains a research prototype; inspect timing and missing-data behavior before using results to inform capital allocation.
6. No broker connection or live execution.

## Ordered v0.2 checklist

1. Run the automated tests and inspect any failures.
2. Verify downloaded price coverage and the actual ticker universe.
3. Audit ranking formulas and backtest signal/return alignment.
4. Review performance metrics and benchmark comparison.
5. Add out-of-sample validation, forward-return labels and rank-IC.
6. Add licensed, point-in-time fundamentals and valuation features.
7. Expand the universe and introduce a documented data-quality pipeline.


## v0.3 out-of-sample validation

The validation run compares three daily return streams: Hedge (75% 12–1 month momentum score, 25% low-volatility score), momentum-only (12–1 month momentum ranking), and SPY buy-and-hold. Monthly signals are formed at month-end and assumed executed at the next available close. Turnover costs are charged on the execution date; new weights earn close-to-close returns after that execution date. This is a simplified daily-close execution convention, not a fill-level simulator.

Configure `validation.evaluation_start` in `config/settings.yaml` before running and do not tune the strategy based on that final evaluation period. Full historical data is used to establish the feature lookback and generate point-in-time signals, but performance metrics for the held-out period are calculated only from dates on or after the configured start. Output files include `comparison_backtest.csv`, `oos_summary.yaml`, `period_performance.csv`, `rank_ic.csv`, and `rank_ic_summary.yaml`.

Important limitations remain: the default ticker list is a small present-day universe, so survivorship bias is material; the model does not yet handle delistings or point-in-time universe membership; missing returns on held assets are flagged as missing rather than imputed; and historical results do not establish future profitability. The momentum-only comparison is a benchmark strategy, not a trained model.


## v0.4 — fundamental research (development prototype)

The pipeline now optionally downloads latest available summary fundamentals for a starter global ticker sample and writes `outputs/fundamentals_raw.csv` and `outputs/fundamental_ranking.csv`. The existing v0.3 price ranking, backtest, comparison and validation outputs are preserved. Fundamental retrieval is fail-soft: if the provider fails or returns no rows, the price-only pipeline continues.

Fundamental score categories use cross-sectional percentiles: value (lower positive valuation multiples score better), quality (higher ROE, margins, free cash flow and current ratio score better), and growth (higher revenue/earnings growth score better). The composite is 40% value, 35% quality, and 25% growth. These are transparent initial weights, not empirically optimized weights. A company needs at least one usable metric in each category to receive a composite score. This score is a screening heuristic, not an intrinsic-value estimate or a return forecast. Negative/meaningless valuation multiples are excluded from valuation ranks; accounting definitions and sector comparability still require specialist treatment.

### Data, licensing and point-in-time caveats

This prototype uses `yfinance` to access Yahoo Finance data for development. Do not assume the feed permits commercial use, redistribution, or production reliance; review current provider terms and obtain a suitable licensed feed before commercial deployment. Global symbols, currencies, fiscal calendars, ADRs, local share classes and reporting standards need normalization. The starter list is not a complete investable universe and is not point-in-time. Latest `Ticker.info` values may include information reported after historical dates and can be revised; therefore `fundamental_ranking.csv` must NOT be used in historical backtests as though it were available at the time. A proper historical fundamental backtest requires filing/publication timestamps, historical point-in-time values, delisted constituents, corporate-action handling and currency normalization.

### Run v0.4

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.main
```

Set `fundamentals.enabled: false` in `config/settings.yaml` to disable fundamental retrieval without affecting v0.3. Output files are CSV for easy inspection. No broker connection or trade execution is included.


## Hedge v0.5: experimental predictive ML

v0.5 adds `src/ml_model.py`, a Random Forest regression baseline using price-only
features (`momentum_12_1`, `momentum_6m`, and `volatility_63d`) to estimate
126-trading-day forward returns. It writes `outputs/ml_walk_forward_predictions.csv`
and `outputs/ml_latest_ranking.csv` when enough training data are available.
The ML layer is optional and fail-soft: the existing v0.3 backtest and v0.4
fundamental outputs are not meant to be replaced by ML output.

### Install and run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.main
```

### Research and leakage limitations

- Walk-forward training only includes rows whose forward-return labels have
  matured before the prediction date. This prevents direct use of future labels.
- Features are currently price-only. Fundamental data are not joined into the
  model because the available Yahoo Finance snapshots are not point-in-time.
- The starter ticker universe is small and has survivorship/selection bias.
- A Random Forest on a small universe is exploratory; the prediction is not a
  calibrated probability and is not an investment recommendation.
- The model currently produces research rankings, not an ML portfolio backtest.
  Do not compare these predictions with the v0.3 strategy until a separate,
  cost-aware execution backtest is added and independently checked.
- Model settings must be selected on training/validation periods only. Keep the
  final out-of-sample window untouched.
- yfinance/Yahoo Finance usage, storage, and redistribution rights must be
  reviewed before commercial use.


## v0.6 — Integrated research dashboard

v0.6 adds a dashboard that surfaces existing v0.3–v0.5 outputs in one place: quantitative ranking, latest fundamental scores, experimental ML predictions, strategy-comparison backtests, period performance, and rank IC. It does not change the core ranking or ML algorithm. Missing output files are handled with helpful warnings.

### Run

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.main
streamlit run app.py
```

### Important limitations

- The dashboard is a presentation/integration layer, not evidence of predictive edge.
- Latest-provider fundamental data are not point-in-time historical data.
- Verify Yahoo Finance/yfinance terms and data licensing before commercial use or redistribution.
- A small manually specified global ticker list has selection/survivorship bias and does not represent a complete global universe.
- ML outputs must be validated on untouched dates against simple baselines, after costs, before being used for investment decisions.
- No brokerage connection or live execution is included.

## v0.7 — Research decision support and diagnostics

v0.7 adds a cross-layer research view to the integrated dashboard. It aligns the available quantitative, fundamental, and ML rankings by ticker; reports a mean percentile consensus and cross-layer disagreement; evaluates walk-forward ML rankings against a simple momentum baseline; and summarizes coverage in the saved price snapshot. Downloadable CSVs are available in the dashboard.

### Interpretation guardrails
- Consensus uses ordinal ranks, not calibrated return forecasts. A lower consensus percentile/rank is better only relative to the supplied universe and available layers.
- `signal_disagreement` is the cross-layer standard deviation of available percentile ranks. It is a disagreement indicator, not a probability of loss.
- ML evaluation compares rank IC and top-quintile forward returns with a momentum baseline. Forward-return horizons overlap across monthly observations, so observations are dependent; results are descriptive and must not be treated as proof of edge.
- Data coverage checks are basic diagnostics, not a full corporate-action, delisting, stale-price, or point-in-time audit.
- Fundamentals remain latest-provider snapshots, not point-in-time historical records. This release does not connect to a broker or execute trades.

Run tests with `python -m pytest -q`, refresh outputs with `python -m src.main`, and launch the dashboard with `streamlit run app.py` from the project root.


## Prediction scorecard and conservative adaptation (v0.8)

The dashboard now includes **Prediction scorecard & adaptation**. Once forward-return labels mature, it reports directional wins/losses, win rate, mean realized return and top-quintile realized return by forecast date. It keeps an append-only outcome ledger and flags a strategy after five consecutive directional losses. A strategy is only eligible for research promotion after at least 20 resolved outcomes, positive mean realized return, and no disable flag.

This is an auditable research control, not self-modifying model intelligence or an order-routing system. It does not automatically change production weights or execute trades. Review sample size, net returns, drawdown, turnover, costs, regime stability, and untouched out-of-sample results before promoting anything. Current ML outputs use overlapping 126-trading-day labels, so adjacent monthly forecasts are dependent and must not be treated as independent trades. The default directional win rate is diagnostic only; a 60% win rate can still lose money if losses are larger than wins.

Recommended research standard: leakage-aware walk-forward evaluation, matured labels only, explicit costs, simple baselines, fixed holdout periods, and a record of all tested strategies/parameters. Current data limitations (current ticker universe, latest fundamentals rather than point-in-time fundamentals) remain.

## Adaptive parameter search (v0.8)

The **Backtest & validation** dashboard now includes an adaptive walk-forward parameter search. It compares several `top_n` / minimum-history configurations, chooses each fold's configuration using only the preceding training window, then reports performance on the next unseen fold. Net returns include the configurable transaction-cost assumption. It also classifies market regimes using trailing benchmark trend and volatility and ranks regimes for further research.

Outputs are saved to `outputs/adaptive_walk_forward_results.csv` and `outputs/adaptive_regime_priority.csv`. This is research prioritization, not autonomous live trading or proof of a durable edge. Review the number of folds, sample sizes, drawdowns, benchmark-relative results, realistic spread/slippage, survivorship bias, and untouched final holdout before using any result. Regime labels are descriptive and are not used to select a configuration using future regime outcomes.


## v1.0 — final holdout, statistical checks, and shadow paper trading

The Backtest & validation dashboard now includes three additional safeguards:

- **Final chronological holdout:** evaluates a fixed, predeclared set of parameter configurations on the final portion of the price history. The holdout must be treated as a one-time evaluation; re-tuning after inspecting it contaminates the test.
- **HAC significance diagnostics:** reports a Newey–West heteroskedasticity/autocorrelation-consistent test for mean daily arithmetic excess returns versus the benchmark, plus Benjamini–Hochberg q-values across the tested configurations. These are exploratory diagnostics, not proof of alpha; they do not fix survivorship bias, data snooping across prior experiments, or model misspecification.
- **Shadow paper-trading ledger:** records hypothetical net daily returns, benchmark returns, transaction costs, turnover, and cumulative equity for a user-selected fixed configuration. Updates are idempotent by date/configuration. This is a backtest-derived shadow ledger, not broker-connected paper execution and not a fill simulator.

No broker, order-routing, or live trading integration is included. Before considering deployment, use point-in-time universe membership including delisted securities, corporate-action-safe data, realistic spreads/slippage/market impact, a frozen configuration, and a forward paper-trading period with operational monitoring.

## v1.1 event-driven paper simulator

The Backtest & validation page now includes an event-driven paper simulator. It freezes monthly momentum score snapshots, records next-available-close simulated fills, tracks a local position snapshot and daily account/risk ledger, applies configurable turnover-based transaction costs, and compares matured forecast direction with realized holding-period returns. It also exports `paper_predictions.csv`, `paper_fills.csv`, `paper_daily_ledger.csv`, `paper_positions.csv`, and `paper_risk_summary.csv` under `outputs/`.

**Execution-data limitation:** the current pipeline stores adjusted close prices only. The simulator therefore uses the next available adjusted close as a transparent execution-price proxy, not an opening auction, bid/ask, or guaranteed fill. Signal direction is based on the sign of a trailing momentum score; it is not a calibrated expected-return forecast. No broker integration or live orders are present. Treat all outputs as hypothetical research, not evidence of profitability.

## v1.2 — validated daily price refresh and scheduled paper trading

### Feed choice and limitations

The default feed is Yahoo Finance through `yfinance`, chosen because it is free for low-volume personal research, requires no API key, and has broad ticker coverage for this mixed-market starter universe. It is **unofficial, best-effort and not a guaranteed or licensed production feed**; endpoints can change, requests can be blocked, and terms may restrict automated/commercial use. There is no genuinely free feed that can be recommended as both globally comprehensive and production-grade/reliable. Review the current provider terms before use beyond personal research. An optional Alpha Vantage adapter is included for a small universe if you configure `ALPHA_VANTAGE_API_KEY`; its free tier has strict daily quotas, and endpoint entitlements may vary. Do not assume the free tier supports every adjusted-data endpoint.

### Sensible update process

- The scheduled job excludes the current calendar date so a still-forming daily candle is not treated as a completed close.
- The updater merges a recent overlap into the local Parquet cache rather than re-downloading the full history each run. The overlap permits corrections to recent bars.
- It validates required benchmark presence, minimum ticker coverage, positive prices, duplicate dates, and per-ticker freshness. If validation fails, it stops before writing a new cache or updating paper results.
- The cache is written to a temporary file and atomically replaced only after validation. Job logs are written to `logs/daily_update.log` and `logs/cron.log`; feed status goes to `outputs/price_feed_status.json`.
- After a validated refresh, it replays/updates the local paper simulator and writes `outputs/paper_execution_assumptions.json`.
- Default one-way execution cost assumptions are 5 bps half-spread + 5 bps slippage + 1 bp commission = 11 bps per unit of turnover. These are transparent placeholders, not observed quotes; adjust by asset/liquidity and test sensitivity (e.g. 5/11/20/30 bps total).

### Install the scheduler (Linux/Kali)

From the project root, after activating/creating `.venv` and installing `requirements.txt`:

```bash
python -m pytest -q
python -m src.daily_update   # manual first run
bash scripts/install_daily_cron.sh
crontab -l
```

The installer schedules the job at 06:30 on Monday through Saturday in the machine's local timezone. Choose a time after the latest close in the configured markets; cron uses the machine timezone. To remove the job, run `crontab -e` and delete the line containing `src.daily_update`. Check logs and `outputs/price_feed_status.json` after the first scheduled run.

The scheduled process is not a real-time feed or broker. Paper execution still uses adjusted daily closes as a proxy; spread/slippage/commission are modeled assumptions, and no live orders are sent. If a ticker has no sufficiently recent bar (for example because it was delisted or its symbol changed), the strict freshness check fails closed; investigate the symbol rather than blindly raising the threshold. For commercial or capital-deployment use, procure a properly licensed provider with documented SLAs and corporate-action/point-in-time handling.
