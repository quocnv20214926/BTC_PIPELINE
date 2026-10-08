import unittest

import numpy as np
import pandas as pd

from pipeline.core import COLLECT_KINDS_BY_TIMEFRAME, KINDS_BY_TIMEFRAME, metric, validate

H = 3_600_000
M5 = 300_000


class NewSourceTests(unittest.TestCase):
    def test_window_kinds_unchanged(self):
        # Thu thập thêm nguồn mới không được làm thay đổi cách tạo feature window.
        self.assertEqual(KINDS_BY_TIMEFRAME['5m'], ('candle', 'ratio', 'taker'))
        self.assertIn('oi', COLLECT_KINDS_BY_TIMEFRAME['5m'])
        self.assertIn('funding', COLLECT_KINDS_BY_TIMEFRAME['1h'])
        self.assertEqual(COLLECT_KINDS_BY_TIMEFRAME['1m'], ('candle',))

    def test_open_interest_is_end_stamped(self):
        ts = 10 * M5
        row = {'symbol': 'BTCUSDT', 'sumOpenInterest': '81234.5', 'sumOpenInterestValue': '6.9e9', 'timestamp': ts}
        e = metric('oi', '5m', row, 'test', 'backfill', received=ts + 1000)
        validate(e)
        self.assertEqual(e['period_end_ms'], ts)

    def test_top_trader_ratio(self):
        ts = 10 * M5
        row = {'symbol': 'BTCUSDT', 'longShortRatio': '1.25', 'longAccount': '0.5556', 'shortAccount': '0.4444',
               'timestamp': ts}
        e = metric('top_ratio', '5m', row, 'test', 'backfill', received=ts + 1000)
        validate(e)
        bad = dict(row, longAccount='0.9')
        with self.assertRaises(ValueError):
            validate(metric('top_ratio', '5m', bad, 'test', 'backfill', received=ts + 1000))

    def test_funding_known_at_settlement(self):
        settle = 100 * H + 3                      # fundingTime carries a few ms
        row = {'symbol': 'BTCUSDT', 'fundingTime': settle, 'fundingRate': '-0.00012', 'markPrice': '65000.1'}
        e = metric('funding', '1h', row, 'test', 'live', received=settle + 500)
        validate(e)                               # negative funding is valid
        self.assertEqual(e['period_end_ms'], 100 * H)
        self.assertLessEqual(e['period_end_ms'], e['received_at_ms'])
        old = dict(row, markPrice='')
        validate(metric('funding', '1h', old, 'test', 'backfill', received=settle + 500))


class ModelSignalTests(unittest.TestCase):
    def test_decision_uses_executor_rule_and_past_only(self):
        from pipeline.model_signals import decide
        import execution_m1 as ex
        n = 25 * 288
        t = np.arange(n) * M5
        hist = pd.DataFrame({'decision_ms': t, 'barrier_pct': 0.01, 'e_long_1r': np.linspace(-.1, .3, n),
                             'e_short_1r': -0.2, 'e_long_2r': 0.0, 'e_short_2r': 0.0})
        row = pd.Series({'timestamp': pd.Timestamp(int(t[-1]) + M5, unit='ms', tz='UTC'), 'barrier_pct': 0.01,
                         'e_long_1r': 0.2, 'e_short_1r': -0.2, 'e_long_2r': 0.0, 'e_short_2r': 0.0})
        absolute = ex.ExecConfig(mode='abs', min_score=0.05)
        d = decide(hist, row, absolute)                    # score = 0.2 - 0.1 = 0.1 > 0.05
        self.assertEqual((d['side'], d['enter']), (1, True))
        self.assertAlmostEqual(d['score'], 0.1)
        rolling = ex.ExecConfig(mode='rolling', top_frac=0.02)
        self.assertFalse(decide(hist, row, rolling)['enter'])      # top 2% of history is above 0.1
        self.assertFalse(decide(hist.iloc[:1000], row, rolling)['enter'])   # not enough history
        later = pd.concat([hist, pd.DataFrame({'decision_ms': [int(t[-1]) + 10 * M5], 'barrier_pct': [0.01],
                                               'e_long_1r': [9.0], 'e_short_1r': [0.0], 'e_long_2r': [0.0], 'e_short_2r': [0.0]})])
        self.assertEqual(decide(later, row, rolling), decide(hist, row, rolling))

    def test_event_contract(self):
        from pipeline.model_signals import to_event
        import execution_m1 as ex
        row = pd.Series({'timestamp': pd.Timestamp('2026-10-01 10:00', tz='UTC'), 'close': 80000.0, 'p_touch': 0.7,
                         'p_up_touch': 0.62, 'barrier_pct': 0.009, 'e_long_1r': 0.2})
        cfg = ex.ExecConfig()
        ev = to_event(row, {'side': 1, 'score': 0.09, 'threshold': 0.05, 'enter': True}, cfg, 'modelv2_5m', 'b.joblib')
        self.assertEqual(ev['event_time_ms'], int(pd.Timestamp('2026-10-01 10:05', tz='UTC').value // 1_000_000))
        self.assertEqual(ev['decision']['side'], 'LONG')
        self.assertTrue(ev['decision']['good'])
        self.assertEqual(ev['decision']['execution']['max_hold_min'], 240)
        off = to_event(row, {'side': -1, 'score': 0.01, 'threshold': 0.05, 'enter': False}, cfg, 'modelv2_5m', 'b')
        self.assertFalse(off['decision']['good'])
        self.assertEqual(off['decision']['side'], 'SHORT')


if __name__ == '__main__':
    unittest.main()
