import numpy as np
import pandas as pd
from src.paper_trading import update_paper_simulator


def sample_prices(n=620):
    idx = pd.bdate_range('2020-01-01', periods=n)
    t = np.arange(n)
    data = {
        'SPY': 100 * (1.0005 ** t),
        'AAA': 30 * (1.0012 ** t) * (1 + .005*np.sin(t/8)),
        'BBB': 40 * (1.0008 ** t) * (1 + .006*np.cos(t/9)),
        'CCC': 20 * (1.0002 ** t) * (1 + .008*np.sin(t/11)),
    }
    return pd.DataFrame(data, index=idx)


def test_simulator_creates_outputs_and_next_close_fills(tmp_path):
    result = update_paper_simulator(sample_prices(), output_dir=tmp_path, top_n=2,
        min_history_days=100, start_date='2021-01-01')
    assert not result['predictions'].empty
    assert not result['daily'].empty
    assert {'equity', 'drawdown_pct', 'net_return', 'transaction_cost_cash'}.issubset(result['daily'].columns)
    assert set(result['fills']['status']) == {'SIMULATED_NEXT_AVAILABLE_CLOSE'}
    assert (pd.to_datetime(result['predictions']['execution_date']) > pd.to_datetime(result['predictions']['signal_date'])).all()
    assert (tmp_path / 'paper_risk_summary.csv').exists()


def test_simulator_is_idempotent_for_prediction_keys(tmp_path):
    prices = sample_prices()
    a = update_paper_simulator(prices, output_dir=tmp_path, top_n=2, min_history_days=100, start_date='2021-01-01')
    n = len(a['predictions'])
    b = update_paper_simulator(prices, output_dir=tmp_path, top_n=2, min_history_days=100, start_date='2021-01-01')
    assert len(b['predictions']) == n
    assert not b['predictions'].duplicated(['signal_date', 'strategy', 'ticker']).any()


def test_no_future_execution_on_last_date(tmp_path):
    result = update_paper_simulator(sample_prices(300), output_dir=tmp_path, top_n=2,
        min_history_days=100, start_date='2020-01-01')
    assert (pd.to_datetime(result['predictions']['execution_date']) <= sample_prices(300).index[-1]).all()
