import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model_training'))
import modelv3 as v3  # noqa: E402

T0 = 1_700_000_000_000 // v3.M5 * v3.M5


def synthetic_m1(days=24, seed=0):
    """Random walk 1m có các cú bùng nổ biến động để sinh sự kiện."""
    rng = np.random.default_rng(seed)
    n = days * 1440
    vol = np.full(n, 0.0004)
    for s in rng.integers(min(3000, n // 4), n - 300, size=days * 2):
        vol[s:s + rng.integers(5, 60)] *= rng.uniform(4, 10)
    lr = rng.normal(0, vol)
    c = 30000 * np.exp(np.cumsum(lr))
    o = np.concatenate([[c[0]], c[:-1]])
    wig = np.abs(rng.normal(0, vol)) * c
    h, l = np.maximum(o, c) + wig, np.minimum(o, c) - wig
    v = rng.uniform(50, 150, n) * (vol / 0.0004)
    return pd.DataFrame({'t': T0 + np.arange(n) * v3.M1, 'o': o, 'h': h, 'l': l, 'c': c, 'v': v,
                         'n': (v * 10).round(), 'tb': v * rng.uniform(0.3, 0.7, n)})


def tiny_bundle(m1, funding=None):
    cfg = v3.Config(xgb=dict(n_estimators=30, learning_rate=0.1, max_depth=2, min_child_weight=1.0, subsample=1.0,
                             colsample_bytree=1.0, reg_lambda=1.0, reg_alpha=0.0, early_stopping_rounds=10),
                    thr_grid=(-1.0,), min_val_trades=1)
    data = v3.build_dataset(m1, funding, cfg)
    return v3.train_final(data, cfg, val_days=6), data


class ZigzagTests(unittest.TestCase):
    def test_pivots_confirmed_only_after_one_percent_reversal(self):
        p = np.array([100, 100.5, 101, 101.6, 101.2, 100.5, 100.4, 99.9, 100.2, 101.0, 101.5, 102, 101.9])
        leg, last, age, ph1, ph2, pl1, pl2, ext = v3._zigzag(p, p, 0.01)
        self.assertEqual(leg[1], 0)                       # chưa xác nhận gì
        self.assertEqual((leg[2], last[2]), (1, 100.0))   # đáy 100 xác nhận khi giá lên 101
        self.assertEqual(leg[4], 1)                       # 101.2 chưa hồi đủ 1% từ 101.6
        self.assertEqual((leg[5], ph1[5]), (-1, 101.6))   # 100.5 <= 101.6 * 0.99 -> đỉnh xác nhận
        self.assertEqual((leg[9], pl1[9]), (1, 99.9))
        self.assertEqual(ext[11], 102.0)


class LabelTests(unittest.TestCase):
    def outcome(self, highs, lows, d=0.01, tp_mult=1.0, hold=240):
        n = hold + 5
        o = np.full(n, 100.0)
        h, l, c = np.full(n, 100.05), np.full(n, 99.95), np.full(n, 100.0)
        for k, x in highs.items():
            h[k] = x
        for k, x in lows.items():
            l[k] = x
        return v3._trade_outcomes(o, h, l, c, np.array([0]), np.array([d]), tp_mult, hold)

    def test_tp_equals_sl_long_and_short(self):
        cls, r, mins = self.outcome({10: 101.1}, {})
        self.assertEqual(cls[0].tolist(), [1, 0])           # long TP, short SL (cùng nến)
        self.assertAlmostEqual(r[0, 0], 1.0)
        self.assertAlmostEqual(r[0, 1], -1.0)
        self.assertEqual(mins[0, 0], 11)

    def test_same_bar_is_stop_and_timeout(self):
        cls, r, _ = self.outcome({7: 101.5}, {7: 98.5})
        self.assertEqual(cls[0].tolist(), [0, 0])
        cls, r, _ = self.outcome({}, {})
        self.assertEqual(cls[0].tolist(), [2, 2])
        self.assertAlmostEqual(r[0, 0], 0.0)


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m1 = synthetic_m1()
        cls.cfg = v3.Config()
        cls.f, cls.sp = v3.compute_bars(cls.m1, None, cls.cfg)

    def test_events_exist_and_pivot_one_percent_only(self):
        self.assertGreater(int(self.f['is_event'].sum()), 20)
        self.assertTrue(any(n.startswith('zz10_') for n in self.sp.names))
        self.assertFalse(any(n.startswith('zz15_') for n in self.sp.names))
        self.assertEqual(self.cfg.tp_mult, 1.0)

    def test_side_matrix_mirror(self):
        xl, xs = v3.side_matrix(self.f, self.sp, 1), v3.side_matrix(self.f, self.sp, -1)
        n_odd, n_even = len(self.sp.odd), len(self.sp.even)
        np.testing.assert_allclose(xs[:, :n_odd], -xl[:, :n_odd], equal_nan=True)
        np.testing.assert_allclose(xs[:, n_odd:n_odd + n_even], xl[:, n_odd:n_odd + n_even], equal_nan=True)
        k = n_odd + n_even                                      # first swap pair
        np.testing.assert_allclose(xs[:, k], xl[:, k + 1], equal_nan=True)

    def test_features_use_past_only(self):
        cut = len(self.m1) - 1440
        g, _ = v3.compute_bars(self.m1.iloc[:cut], None, self.cfg)
        a = self.f.iloc[len(g) - 200:len(g)][self.sp.names].to_numpy()
        b = g.iloc[-200:][self.sp.names].to_numpy()
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)


class BundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m1 = synthetic_m1(days=40, seed=1)
        cls.bundle, cls.data = tiny_bundle(cls.m1)

    def test_live_prediction_matches_batch_with_short_history(self):
        full = v3.predict_live(self.bundle, self.m1, None, last_n=100)
        short = v3.predict_live(self.bundle, self.m1.iloc[-15 * 1440:], None, last_n=100)
        self.assertTrue((full['is_event'].to_numpy() == short['is_event'].to_numpy()).all())
        self.assertLess(float((full['ev'] - short['ev'].to_numpy()).abs().max()), 0.02)
        self.assertTrue((full['tp_pct'] == full['sl_pct']).all())      # TP = SL

    def test_not_enough_history(self):
        with self.assertRaises(ValueError):
            v3.predict_live(self.bundle, self.m1.iloc[-3 * 1440:], None, last_n=1)


class WorkerTests(unittest.TestCase):
    def test_missing_runs_and_history_fill(self):
        from pipeline.model_signals_v3 import History, missing_runs
        t = T0 + np.array([0, 1, 2, 5, 6, 9]) * v3.M1
        self.assertEqual(missing_runs(t, T0, T0 + 10 * v3.M1),
                         [(T0 + 3 * v3.M1, T0 + 5 * v3.M1), (T0 + 7 * v3.M1, T0 + 9 * v3.M1)])
        full = synthetic_m1(days=1)
        calls = []

        def fetch(a, b):
            calls.append((a, b))
            return full[(full['t'] >= a) & (full['t'] < b)]
        hist = History(fetch)
        db_part = full.iloc[100:]
        got = hist.get(db_part, T0, T0 + 1440 * v3.M1)
        self.assertEqual(len(got), 1440)
        self.assertEqual(calls, [(T0, T0 + 100 * v3.M1)])
        hist.get(db_part, T0, T0 + 1440 * v3.M1)                  # second time: served from cache
        self.assertEqual(len(calls), 1)

    def test_event_contract(self):
        from pipeline.model_signals_v3 import to_event
        row = pd.Series({'decision_ms': T0 + v3.M5, 'close': 80000.0, 'sl_pct': 0.01, 'side': -1, 'ev': 0.05,
                         'p_tp': 0.4, 'p_sl': 0.25, 'ev_long': -0.1, 'ev_short': 0.05, 'threshold': 0.0,
                         'enter': True, 'is_event': True, 'E1': 1.0, 'E2': 0.0, 'E3': 1.0})
        ev = to_event(row, 1.0, 240, 'modelv3_event', 'model_v3.joblib')
        s = ev['signal']
        self.assertEqual(ev['decision']['side'], 'SHORT')
        self.assertTrue(ev['decision']['good'])
        self.assertAlmostEqual(s['short_tp_1r'], 79200.0)
        self.assertAlmostEqual(s['short_sl'], 80800.0)
        self.assertEqual(s['time_stop_bars'], 48)
        from pipeline.telegram_bot import format_signal
        text = format_signal(ev, 1, 0.001, 7, 'modelv3_event')
        self.assertIn('SHORT BTCUSDT', text)
        self.assertIn('TP: 79,200.0 (-1.00%)', text)
        self.assertIn('P(TP) 0.40 · P(SL) 0.25 · nến 5m bất thường, bất thường 1m', text)


@unittest.skipUnless(os.getenv('TEST_DATABASE_URL'), 'needs TEST_DATABASE_URL (empty PostgreSQL database)')
class WorkerDatabaseTests(unittest.TestCase):
    def test_worker_once_writes_every_closed_bar(self):
        import tempfile
        from psycopg.types.json import Jsonb
        from pipeline import storage
        from pipeline import model_signals_v3 as w
        from pipeline.core import INSTRUMENT
        os.environ['DATABASE_URL'] = os.environ['TEST_DATABASE_URL']
        m1 = synthetic_m1(days=40, seed=1)
        bundle, _ = tiny_bundle(m1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'b.joblib'
            v3.save_bundle(bundle, path)
            with storage.connect() as db:
                db.execute('DROP SCHEMA public CASCADE; CREATE SCHEMA public')
                db.execute(Path('sql/schema.sql').read_text())
                recent = m1.iloc[-3 * 1440:]                     # DB chỉ có 3 ngày, phần còn lại từ "REST"
                with db.transaction():
                    for r in recent.itertuples():
                        eid = f'c{r.t}'
                        db.execute("""INSERT INTO raw_events(event_id,instrument,timeframe,kind,period_start_ms,
                            received_at_ms,envelope) VALUES(%s,%s,'1m','candle',%s,%s,'{}')""", (eid, INSTRUMENT, r.t, r.t))
                        db.execute("INSERT INTO observations VALUES(%s,'1m','candle',%s,%s,%s,%s,'live',%s)",
                                   (INSTRUMENT, r.t, r.t + v3.M1, eid, r.t + v3.M1,
                                    Jsonb({'open': r.o, 'high': r.h, 'low': r.l, 'close': r.c, 'volume': r.v,
                                           'trade_count': r.n, 'taker_buy_base': r.tb})))
            hist = w.History(lambda a, b: m1[(m1['t'] >= a) & (m1['t'] < b)])
            fund = mock.Mock(get=mock.Mock(return_value=None))
            with mock.patch.dict(os.environ, {'MODEL_V3_BUNDLE': str(path)}):
                w.model_worker_v3(once=True, history=hist, funding=fund)
            with storage.connect() as db:
                rows = db.execute("SELECT event_time_ms, payload FROM signal_events ORDER BY event_time_ms").fetchall()
                outbox = db.execute('SELECT count(*) AS n FROM outbox').fetchone()['n']
        self.assertEqual(len(rows), 288)
        self.assertEqual(outbox, 288)
        self.assertEqual(rows[-1]['event_time_ms'], int(m1['t'].iloc[-1]) + v3.M1)
        expected = v3.predict_live(bundle, m1, None, last_n=288)
        got_good = [r['payload']['decision']['good'] for r in rows]
        self.assertEqual(got_good, expected['enter'].tolist())


if __name__ == '__main__':
    unittest.main()
