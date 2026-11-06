"""Kiểm thử chiến lược H1 live (pipeline/strategy_live.py, pipeline/strategy_exec.py).

1. ExecTests (không cần dữ liệu): TP / SL / gap qua SL / limit pullback (khớp ở open, khớp giữa nến, hết chờ -> market) /
   hết giờ giữ / luật danh mục.
2. SyntheticParityTests (không cần dữ liệu): bộ thực thi chạy lại tín hiệu ngẫu nhiên trên nến 5m giả phải cho đúng
   từng lệnh của strategy_h1.simulate + select_trades (cả next_open và limit_pullback) và cùng vốn cuối.
3. ReplayParityTests: như 2 nhưng với tín hiệu walk-forward thật (artifacts/1h_strategy_v2/walkforward_signals.csv).
4. LivePipelineTests: dữ liệu lịch sử được ghi vào PostgreSQL ĐÚNG như collector ghi từ REST, rồi worker tính tín hiệu
   từng giờ -> xác suất phải trùng lịch sử dự báo trong bundle, tín hiệu trùng walk-forward, lệnh paper (khớp bằng
   nến 5m) trùng bộ thực thi chạy lại. Cần TEST_DATABASE_URL (PostgreSQL trống) + historical_data + bundle v2.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model_training'))

from pipeline import strategy_exec as sx  # noqa: E402

H, M5, M1 = 3_600_000, 300_000, 60_000
HD = ROOT / 'historical_data'
ART = ROOT / 'artifacts' / '1h_strategy_v2'


def bar(t, o, h, l, c, step=M5):
    return sx.Bar(t, t + step, o, h, l, c)


def strategy_cfg():
    import strategy_h1 as st
    p = ART / 'config_locked.json'
    return st.load_locked(str(p)) if p.exists() else st.StrategyConfig()


class ExecTests(unittest.TestCase):
    p = sx.ExecParams(entry_mode='next_open', max_positions=2, max_hold_h=1)
    lp = sx.ExecParams(entry_mode='limit_pullback', max_positions=2, max_hold_h=1)

    def trade(self, side=1, b=0.01, entry=100.0, t0=10 * H):
        t = sx.Trade('t', 's', t0 - H, side, b)
        sx.open_trade(t, t0, entry, self.p, 1.0)
        return t

    def pending(self, side=1, b=0.01, decision=10 * H):
        return sx.Trade('t', 's', decision - H, side, b)

    def test_long_tp_then_r(self):
        t = self.trade()
        self.assertFalse(sx.step(t, bar(t.entry_ms, 100, 100.5, 99.8, 100.3), self.p))
        self.assertTrue(sx.step(t, bar(t.entry_ms + M5, 100.3, 101.2, 100.2, 101.0), self.p))
        self.assertEqual(t.status, 'TP')
        self.assertAlmostEqual(t.r_gross, 1.0)
        self.assertAlmostEqual(t.r_net, 1.0 - 0.001 / 0.01)          # taker vào + taker ra

    def test_same_bar_tp_and_sl_is_sl(self):
        t = self.trade()
        self.assertTrue(sx.step(t, bar(t.entry_ms, 100, 101.5, 98.9, 100.0), self.p))
        self.assertEqual(t.status, 'SL')
        self.assertAlmostEqual(t.r_gross, -1.0)

    def test_gap_through_sl_fills_at_open(self):
        t = self.trade(side=-1)
        self.assertTrue(sx.step(t, bar(t.entry_ms + M5, 101.5, 101.6, 101.0, 101.2), self.p))
        self.assertEqual(t.status, 'SL')
        self.assertAlmostEqual(t.r_gross, -np.log(101.5 / 100) / 0.01)

    def test_timeout_after_max_hold(self):
        t = self.trade()
        closed = [sx.step(t, bar(t.entry_ms + i * M5, 100, 100.1, 99.9, 100.05), self.p) for i in range(12)]
        self.assertEqual(closed, [False] * 11 + [True])
        self.assertEqual(t.status, 'TIMEOUT')
        self.assertEqual(t.exit_close_ms, t.entry_ms + H)
        self.assertAlmostEqual(t.exit_price, 100.05)

    def test_bars_processed_once(self):
        t = self.trade()
        sx.step(t, bar(t.entry_ms, 100, 100.2, 99.9, 100.1), self.p)
        self.assertFalse(sx.step(t, bar(t.entry_ms, 100, 120, 80, 100), self.p))   # cùng nến -> bỏ qua
        self.assertEqual(t.status, 'OPEN')

    def test_limit_open_through_limit_is_maker_at_open(self):
        t, H0 = self.pending(), 10 * H
        lim = 100 * np.exp(-0.05 * 0.01)
        self.assertFalse(sx.try_fill(t, bar(H0, 100, 100.2, 99.98, 100.1), self.lp, 1.0))
        self.assertTrue(sx.try_fill(t, bar(H0 + M5, 99.9, 100.0, 99.8, 99.95), self.lp, 1.0))
        self.assertLess(99.9, lim)
        self.assertEqual((t.status, t.entry, t.extra['maker'], t.extra['intrabar']), ('OPEN', 99.9, True, False))
        self.assertAlmostEqual(sx.fee_r(self.lp, t), 0.0007 / 0.01)

    def test_limit_intrabar_fill_skips_tp_on_fill_bar(self):
        t, H0 = self.pending(), 10 * H
        lim = 100 * np.exp(-0.05 * 0.01)
        b0 = bar(H0, 100, 102, 99.9, 101.5)                    # xuyên limit và cao hơn TP trong cùng nến
        self.assertTrue(sx.try_fill(t, b0, self.lp, 1.0))
        self.assertAlmostEqual(t.entry, lim)
        self.assertFalse(sx.step(t, b0, self.lp))              # nến khớp chỉ xét SL
        self.assertTrue(sx.step(t, bar(H0 + M5, 101.5, 102, 101, 101.8), self.lp))
        self.assertEqual(t.status, 'TP')

    def test_limit_times_out_to_market(self):
        t, H0 = self.pending(side=-1), 10 * H
        self.assertFalse(sx.try_fill(t, bar(H0, 100, 100.02, 99.9, 99.95), self.lp, 1.0))
        self.assertFalse(sx.try_fill(t, bar(H0 + M5, 99.95, 100.0, 99.8, 99.9), self.lp, 1.0))
        self.assertTrue(sx.try_fill(t, bar(H0 + 2 * M5, 99.7, 99.8, 99.6, 99.7), self.lp, 1.0))
        self.assertEqual((t.entry, t.extra['maker'], t.entry_ref), (99.7, False, 100))
        self.assertAlmostEqual(t.extra['slippage_r'], -np.log(99.7 / 100) / 0.01)

    def test_portfolio_rule_counts_trades_exiting_after_signal_hour(self):
        book = sx.Book(self.p)
        for k in range(2):
            t = sx.Trade(f't{k}', f's{k}', k * H, 1, 0.01)
            self.assertEqual(book.add_signal(t), (True, 'ok'))
            sx.open_trade(t, (k + 1) * H, 100.0, self.p, 1.0)
        self.assertEqual(book.admit(2 * H), (False, 'max_positions'))
        for t in book.trades:
            t.status, t.exit_ms, t.r_net = 'SL', 2 * H, -1.0
        self.assertEqual(book.admit(2 * H)[0], False)          # thoát ở nến mở lúc >= giờ tín hiệu -> vẫn tính
        self.assertEqual(book.admit(3 * H)[0], True)

    def test_min_gap(self):
        book = sx.Book(sx.ExecParams(min_gap_h=2))
        book.add_signal(sx.Trade('a', 'a', 0, 1, 0.01))
        self.assertEqual(book.admit(H), (False, 'min_gap'))
        self.assertEqual(book.admit(2 * H), (True, 'ok'))


def compare_replay(tc, st, cfg, d5, rows):
    """d5: nến 5m (open_time_ms, open, high, low, close); rows: tín hiệu (t_ns, side, b) -> replay == backtest."""
    import modelv2 as mv2
    D5 = mv2.prepare_candles(d5, 5)
    Dd = st.Data(cfg=cfg, df=None, d5=D5, ds=None, xd=None, dir_names=())
    S = st.simulate(Dd, rows.reset_index(drop=True), cfg)
    T = S[st.select_trades(S, cfg.max_positions, cfg.min_gap_bars)].reset_index(drop=True)
    t5 = mv2._ns_int(D5['timestamp']) // 1_000_000
    o, h, l, c = (D5[k].to_numpy() for k in ('open', 'high', 'low', 'close'))
    bars = (sx.Bar(int(t5[i]), int(t5[i]) + M5, o[i], h[i], l[i], c[i]) for i in range(len(D5)))
    sig = [(str(k), int(r.t_ns // 1_000_000), int(r.side), float(r.b)) for k, r in rows.iterrows()]
    book = sx.replay(sig, bars, sx.ExecParams.from_strategy(cfg))
    X = pd.DataFrame([t.to_dict() for t in book.trades if t.status in ('TP', 'SL', 'TIMEOUT')])
    T = T[T.kind != 4]                                        # lệnh bị cắt ở cuối dữ liệu
    tc.assertEqual(len(X), len(T))
    X = X.sort_values('hour_open_ms').reset_index(drop=True)
    T = T.sort_values('t_ns').reset_index(drop=True)
    np.testing.assert_array_equal(X['hour_open_ms'].to_numpy(), T.t_ns.to_numpy() // 1_000_000)
    np.testing.assert_array_equal(X['entry_ms'].to_numpy(), T.entry_ns.to_numpy() // 1_000_000)
    np.testing.assert_allclose(X['entry'].to_numpy(), T.entry.to_numpy(), rtol=0, atol=1e-9)
    np.testing.assert_allclose(X['r_net'].to_numpy(), T.R.to_numpy(), rtol=0, atol=1e-9)
    name = {1: 'TP', -1: 'SL', 3: 'TIMEOUT'}
    tc.assertTrue((T.kind.map(name).to_numpy() == X['status'].to_numpy()).all())
    return len(X)


class SyntheticParityTests(unittest.TestCase):
    def test_replay_equals_backtest_both_entry_modes(self):
        import strategy_h1 as st
        rng = np.random.default_rng(7)
        n = 12 * 24 * 40
        t0 = int(pd.Timestamp('2025-01-01', tz='UTC').value // 1_000_000)
        lc = np.log(50_000) + np.cumsum(rng.normal(0, 0.002, n))
        o = np.exp(np.r_[lc[0], lc[:-1]])
        c = np.exp(lc)
        h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.0015, n)))
        l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.0015, n)))
        d5 = pd.DataFrame({'open_time_ms': t0 + M5 * np.arange(n), 'open': o, 'high': h, 'low': l, 'close': c,
                           'volume': 1.0})
        hours = t0 + H * np.sort(rng.choice(np.arange(24 * 39), 300, replace=False))
        rows = pd.DataFrame({'t_ns': hours * 1_000_000, 'side': rng.choice([-1, 1], len(hours)),
                             'b': rng.uniform(0.006, 0.015, len(hours))})
        base = strategy_cfg()
        from dataclasses import replace
        for mode in ('next_open', 'limit_pullback'):
            with self.subTest(mode=mode):
                self.assertGreater(compare_replay(self, st, replace(base, entry_mode=mode), d5, rows), 50)


def five_minute_file():
    for name in ('BTCUSDT_5m_all.csv', 'BTCUSDT_5m_smoke.csv.gz'):
        if (HD / name).exists():
            return HD / name
    return None


def have_history():
    need = [HD / 'BTCUSDT_positioning_5m.csv', HD / 'BTCUSDT_funding.csv', ART / 'walkforward_signals.csv',
            ART / 'models_final.joblib']
    return all(p.exists() for p in need) and (five_minute_file() is not None)


@unittest.skipUnless(have_history(), 'cần historical_data + artifacts/1h_strategy_v2')
class ReplayParityTests(unittest.TestCase):
    def test_replay_equals_backtest_last_year(self):
        import strategy_h1 as st
        W = pd.read_csv(ART / 'walkforward_signals.csv')
        W = W[W['signal'].astype(bool)]
        W = W[W.t_ns >= W.t_ns.max() - 365 * 86_400 * 10**9]
        d5 = pd.read_csv(five_minute_file(), usecols=['open_time_ms', 'open', 'high', 'low', 'close', 'volume'])
        d5 = d5[d5['open_time_ms'] >= W.t_ns.min() // 1_000_000].reset_index(drop=True)
        self.assertGreater(compare_replay(self, st, strategy_cfg(), d5, W[['t_ns', 'side', 'b']]), 100)


def have_live_inputs():
    return (have_history() and os.getenv('TEST_DATABASE_URL') and (HD / 'BTCUSDT_metrics_5m.csv').exists()
            and (HD / 'BTCUSDT_premium_5m.csv').exists())


@unittest.skipUnless(have_live_inputs(), 'cần TEST_DATABASE_URL + historical_data (metrics, premium) + bundle v2')
class LivePipelineTests(unittest.TestCase):
    """Giả lập collector ghi 28 ngày dữ liệu REST vào DB rồi chạy worker theo từng giờ, tới hết dữ liệu của bundle."""
    DAYS_DB = 28
    HOURS = int(os.getenv('TEST_LIVE_HOURS', '168'))

    @classmethod
    def setUpClass(cls):
        import strategy_h1 as st
        os.environ['DATABASE_URL'] = os.environ['TEST_DATABASE_URL']
        os.environ.update({'STRATEGY_BUNDLE': str(ART / 'models_final.joblib'), 'STRATEGY_FILL_BARS': '5m',
                           'STRATEGY_HISTORY_DAYS': '200'})
        from pipeline.core import candle, metric, premium, validate
        from pipeline.storage import connect, init_schema, save_event
        os.chdir(ROOT)
        cls.bundle = st.load_bundle(ART / 'models_final.joblib')
        cls.END = int(cls.bundle['data_end_ns'] // 1_000_000) + H     # giờ quyết định của nến 1h cuối trong bundle
        init_schema()
        with connect() as db:
            db.execute('TRUNCATE strategy_trades, strategy_state, telegram_messages, outbox, signal_events, '
                       'observations, raw_events CASCADE')
        start = cls.END - cls.DAYS_DB * 86_400_000
        d5 = pd.read_csv(five_minute_file())
        cls.d5 = d5
        k = d5[(d5['open_time_ms'] >= start) & (d5['open_time_ms'] < cls.END)]
        m = pd.read_csv(HD / 'BTCUSDT_metrics_5m.csv')
        m['ts'] = pd.to_datetime(m['create_time'], utc=True).dt.as_unit('ms').astype('int64')
        m = m[(m['ts'] >= start) & (m['ts'] < cls.END)]
        pr = pd.read_csv(HD / 'BTCUSDT_premium_5m.csv')
        pr = pr[(pr['open_time'] >= start) & (pr['open_time'] < cls.END)]
        events = []
        rec = cls.END + 60_000
        for r in k.itertuples():
            row = [r.open_time_ms, r.open, r.high, r.low, r.close, r.volume, r.open_time_ms + M5 - 1,
                   r.close * r.volume, int(r.trades), r.taker_buy_base_asset_volume, 0]
            events.append(candle('5m', row, 'test', 'backfill', rec))
        for r in m.itertuples():
            ts = int(r.ts)
            events.append(metric('oi', '5m', {'timestamp': ts, 'sumOpenInterest': r.sum_open_interest,
                                              'sumOpenInterestValue': r.sum_open_interest_value}, 'test', 'backfill', rec))
            for kind, v in (('ratio', r.count_long_short_ratio), ('top_ratio', r.sum_toptrader_long_short_ratio),
                            ('top_acct_ratio', r.count_toptrader_long_short_ratio)):
                if not (v > 0):
                    continue
                la = round(v / (1 + v), 4)
                events.append(metric(kind, '5m', {'timestamp': ts, 'longShortRatio': v, 'longAccount': la,
                                                  'shortAccount': round(1 - la, 4)}, 'test', 'backfill', rec))
        for r in pr.itertuples():
            events.append(premium('5m', [r.open_time, r.open, r.high, r.low, r.close, '0', r.open_time + M5 - 1],
                                  'test', 'backfill', rec))
        with connect() as db:
            with db.transaction():
                for e in events:
                    validate(e)
                    save_event(db, e)

    def test_hourly_signals_match_bundle_and_trades_match_replay(self):
        from pipeline import strategy_live as sl
        from pipeline.storage import connect
        fund = pd.read_csv(HD / 'BTCUSDT_funding.csv')
        d5 = self.d5

        def fake_rest(path, params, client=None):
            assert path == '/fapi/v1/klines', path
            f = d5[(d5['open_time_ms'] >= params['startTime']) & (d5['open_time_ms'] <= params['endTime'])].head(1500)
            return [[int(r.open_time_ms), r.open, r.high, r.low, r.close, r.volume, 0, 0, int(r.trades),
                     r.taker_buy_base_asset_volume, 0, 0] for r in f.itertuples()]

        hist = self.bundle['history'].assign(h=lambda x: x.t_ns // 1_000_000).set_index('h')
        W = pd.read_csv(ART / 'walkforward_signals.csv').assign(h=lambda x: x.t_ns // 1_000_000).set_index('h')
        clock = {'now': 0}
        with mock.patch.object(sl, 'rest_get', side_effect=fake_rest), \
                mock.patch.object(sl, 'now_ms', side_effect=lambda: clock['now']), \
                mock.patch.object(sl.FundingCache, 'get',
                                  side_effect=lambda db=None: fund[fund['funding_time_ms'] < clock['now']]):
            with connect() as db:
                strat = sl.Strategy(db, sl.Settings())
                first = self.END - self.HOURS * H
                evs = []
                for Hh in range(first, self.END + 1, H):
                    clock['now'] = Hh + 120_000
                    ev = strat.run_hour(Hh)
                    self.assertIsNotNone(ev)
                    evs.append(ev)
                    hh = Hh - H
                    self.assertAlmostEqual(ev['signal']['p_touch'], float(hist.loc[hh, 'p_touch']), places=6)
                    self.assertAlmostEqual(ev['signal']['p_up'], float(hist.loc[hh, 'p_up']), places=6)
                    if hh in W.index:
                        self.assertEqual(ev['decision']['good'], bool(W.loc[hh, 'signal']))
                    clock['now'] = Hh + 30 * M1
                    strat.manage()
                clock['now'] = self.END + 2 * 86_400_000
                strat.manage()
                trades = db.execute("SELECT * FROM strategy_trades ORDER BY hour_open_ms").fetchall()
        good = [e for e in evs if e['decision']['good']]
        admitted = [e for e in good if e['decision']['admitted']]
        self.assertEqual(len(trades), len(admitted))
        self.assertGreater(len(trades), 0)
        cfg = self.bundle['config']
        sig = [(e['signal_id'], e['hour_open_ms'], 1 if e['decision']['side'] == 'LONG' else -1, e['signal']['b_pct'])
               for e in good]
        f = d5[d5['open_time_ms'] >= first]
        bars = (sx.Bar(int(r.open_time_ms), int(r.open_time_ms) + M5, r.open, r.high, r.low, r.close)
                for r in f.itertuples())
        ref = {t.hour_open_ms: t for t in sx.replay(sig, bars, sx.ExecParams.from_strategy(cfg)).trades}
        self.assertEqual(sorted(ref), [int(t['hour_open_ms']) for t in trades])
        for tr in trades:
            if tr['status'] in ('PENDING', 'OPEN'):
                continue
            t = ref[int(tr['hour_open_ms'])]
            self.assertEqual(t.status, tr['status'])
            self.assertAlmostEqual(t.entry, tr['entry'], places=9)
            self.assertAlmostEqual(t.r_net, tr['r_net'], places=9)


if __name__ == '__main__':
    unittest.main()
