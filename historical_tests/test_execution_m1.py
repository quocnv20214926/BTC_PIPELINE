import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model_training'))
import execution_m1 as ex  # noqa: E402

T0 = 1_700_000_000_000 // ex.M5 * ex.M5


def market(n, p=100.0):
    t = T0 + np.arange(n) * ex.M1
    return {'t': t, 'o': np.full(n, p), 'h': np.full(n, p + 0.1), 'l': np.full(n, p - 0.1), 'c': np.full(n, p)}


def signals(times, e_long=0.3, e_short=-0.3, b=0.01):
    k = len(times)
    return pd.DataFrame({'decision_ms': np.asarray(times, dtype=np.int64), 'barrier_pct': b, 'e_long_1r': e_long,
                         'e_short_1r': e_short, 'e_long_2r': 0.0, 'e_short_2r': 0.0, 'p_touch': 0.5,
                         'p_up_touch': 0.6, 'close': 100.0, 'timestamp_ns': np.zeros(k, dtype=np.int64)})


class ExecutorTests(unittest.TestCase):
    cfg = ex.ExecConfig(mode='abs', min_score=0.05, fee_roundtrip=0.001)

    def test_only_confident_signals_enter(self):
        s = ex.score_signals(signals([T0, T0 + ex.M5], e_long=[0.3, 0.12]), self.cfg)
        self.assertEqual(s['enter'].tolist(), [True, False])           # 0.12 - 0.1 fee = 0.02 < 0.05

    def test_fixed_levels_timeout_and_independent_overlap(self):
        m = market(600)
        tr = ex.simulate(m, ex.score_signals(signals([T0, T0 + ex.M5]), self.cfg), self.cfg)
        self.assertEqual(len(tr), 2)                                   # overlapping trades both open
        self.assertTrue((tr['reason'] == 'timeout').all())
        self.assertTrue((tr['minutes'] == 240).all())
        self.assertAlmostEqual(tr['R'].iloc[0], -0.1)                  # flat price: only the fee
        self.assertEqual(ex.concurrency(tr), 2)

    def test_tp_sl_and_same_bar_is_stop(self):
        m = market(600)
        m['h'][10] = 101.5                                              # long TP (101) on bar 10
        tr = ex.simulate(m, ex.score_signals(signals([T0]), self.cfg), self.cfg)
        self.assertEqual((tr['reason'][0], tr['minutes'][0]), ('tp', 11))
        self.assertAlmostEqual(tr['R'][0], 0.9)
        m['l'][5] = 98.0
        m['h'][5] = 102.0
        tr = ex.simulate(m, ex.score_signals(signals([T0]), self.cfg), self.cfg)
        self.assertEqual(tr['reason'][0], 'sl')
        self.assertAlmostEqual(tr['R'][0], -1.1)

    def test_stop_gap_fills_at_open_and_short_side(self):
        m = market(600)
        m['o'][20], m['l'][20], m['h'][20] = 98.0, 97.5, 98.2
        tr = ex.simulate(m, ex.score_signals(signals([T0]), self.cfg), self.cfg)
        self.assertAlmostEqual(tr['exit'][0], 98.0)
        m = market(600)
        m['l'][30] = 98.9
        tr = ex.simulate(m, ex.score_signals(signals([T0], e_long=-0.3, e_short=0.3), self.cfg), self.cfg)
        self.assertEqual((tr['side'][0], tr['reason'][0]), (-1, 'tp'))

    def test_no_trade_without_full_horizon(self):
        tr = ex.simulate(market(100), ex.score_signals(signals([T0]), self.cfg), self.cfg)
        self.assertEqual(len(tr), 0)


if __name__ == '__main__':
    unittest.main()
