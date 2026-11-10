import numpy as np
import pandas as pd
import pytest
from src.adaptive_engine import classify_market_regimes, run_adaptive_search
from src.features import calculate_features


def test_regime_classifier_is_aligned_and_named():
    idx = pd.bdate_range('2018-01-01', periods=400)
    prices = pd.DataFrame({'SPY': 100 * np.cumprod(1 + np.full(len(idx), .0005))}, index=idx)
    regimes = classify_market_regimes(prices)
    assert len(regimes) == len(prices)
    assert set(regimes.dropna().unique()).issubset({'bull','bear','sideways','high_vol_bull','high_vol_bear','high_vol_sideways'})


def test_adaptive_search_requires_sufficient_walk_forward_history():
    idx = pd.bdate_range('2020-01-01', periods=300)
    prices = pd.DataFrame({'SPY': np.linspace(100, 130, len(idx)), 'AAA': np.linspace(20, 35, len(idx))}, index=idx)
    with pytest.raises(ValueError):
        run_adaptive_search(prices, calculate_features(prices), min_train_days=100, test_days=30)


def test_adaptive_search_outputs_oos_folds_and_regime_summary():
    rng = np.random.default_rng(42)
    idx = pd.bdate_range('2015-01-01', periods=1100)
    spy = 100 * np.cumprod(1 + rng.normal(.0003, .008, len(idx)))
    aaa = 20 * np.cumprod(1 + rng.normal(.0005, .012, len(idx)))
    bbb = 30 * np.cumprod(1 + rng.normal(.0002, .01, len(idx)))
    prices = pd.DataFrame({'SPY': spy, 'AAA': aaa, 'BBB': bbb}, index=idx)
    detail, summary = run_adaptive_search(prices, calculate_features(prices), min_train_days=756, test_days=126,
        configs=[{'name':'small','top_n':1,'min_history_days':126}])
    assert not detail.empty
    assert {'selected_config','regime','oos_net_return_pct','oos_annualized_sharpe'}.issubset(detail.columns)
    assert not summary.empty
    assert summary['research_priority'].notna().any()
