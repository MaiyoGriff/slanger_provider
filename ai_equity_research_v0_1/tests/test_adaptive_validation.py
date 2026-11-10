import numpy as np
import pandas as pd
from src.adaptive_validation import _hac_mean_test, evaluate_final_holdout, update_shadow_paper_ledger
from src.features import calculate_features


def make_prices(n=900):
    rng = np.random.default_rng(71)
    idx = pd.bdate_range('2019-01-01', periods=n)
    spy = 100 * np.cumprod(1 + rng.normal(.0003, .008, n))
    aaa = 25 * np.cumprod(1 + rng.normal(.0005, .012, n))
    bbb = 40 * np.cumprod(1 + rng.normal(.0002, .01, n))
    return pd.DataFrame({'SPY': spy, 'AAA': aaa, 'BBB': bbb}, index=idx)


def test_hac_test_reports_significance_for_persistent_positive_series():
    z, p = _hac_mean_test(pd.Series(np.full(100, .001) + np.random.default_rng(3).normal(0, .0001, 100)))
    assert z > 0
    assert p < .05


def test_final_holdout_is_chronological_and_reports_hac_metrics():
    prices = make_prices()
    results, start = evaluate_final_holdout(prices, calculate_features(prices), holdout_fraction=.2,
        configs=[{'name':'test_cfg','top_n':1,'min_history_days':126}])
    assert len(results) == 1
    assert results.iloc[0]['holdout_start'] == start
    assert results.iloc[0]['observations'] >= 30
    assert 'hac_excess_p_value' in results.columns


def test_shadow_paper_ledger_is_idempotent(tmp_path):
    prices = make_prices(500)
    features = calculate_features(prices)
    path = tmp_path / 'paper.csv'
    cfg = {'name':'paper_test','top_n':1,'min_history_days':126}
    first = update_shadow_paper_ledger(prices, features, ledger_path=path, config=cfg, start_date='2020-01-01')
    second = update_shadow_paper_ledger(prices, features, ledger_path=path, config=cfg, start_date='2020-01-01')
    assert len(first) == len(second)
    assert not second.duplicated(['date','config']).any()
    assert {'strategy_equity','benchmark_equity','transaction_cost'}.issubset(second.columns)
