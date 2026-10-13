"""Kiểm thử chiến lược H1 live (pipeline/strategy_live.py, pipeline/strategy_exec.py).

1. ExecTests (không cần dữ liệu): TP / SL / khớp ở giá mở khi gap / thoát nhanh / hết hạn / luật danh mục.
2. ReplayParityTests: bộ thực thi chạy lại tín hiệu walk-forward trên nến 5m phải cho đúng từng lệnh của
   strategy_h1.backtest_m5 (cần historical_data + artifacts/1h_strategy_v1).
3. LivePipelineTests: dữ liệu lịch sử được ghi vào PostgreSQL ĐÚNG như collector ghi từ REST (envelope, validate,
   observations), rồi worker tính tín hiệu từng giờ -> xác suất phải trùng walk-forward và lệnh paper phải giống
   bộ thực thi chạy lại. Cần TEST_DATABASE_URL (PostgreSQL trống) + historical_data (metrics, premium, 5m, funding).
"""

import math
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

H, M5 = 3_600_000, 300_000
HD = ROOT / 'historical_data'
ART = ROOT / 'artifacts' / '1h_strategy_v1'


def bar(t, o, h, l, c, step=M5, levels=True, anomaly=True):
    return sx.Bar(t, t + step, o, h, l, c, levels, anomaly)


class ExecTests(unittest.TestCase):
    p = sx.ExecParams(tp_mult=1.0, sl_mult=0.75, anomaly_range=0.0075, max_positions=2, max_hold_h=1)

    def trade(self, side=1, b=0.01, entry=100.0, t0=10 * H):
        t = sx.Trade('t', 's', t0 - H, side, b)
        sx.open_trade(t, t0, entry, self.p, 1.0)
        return t

    def test_long_tp_then_r(self):
        t = self.trade()
        self.assertFalse(sx.step(t, bar(t.entry_ms, 100, 100.5, 99.8, 100.3), self.p))
        self.assertTrue(sx.step(t, bar(t.entry_ms + M5, 100.3, 101.2, 100.2, 101.0), self.p))
        self.assertEqual(t.status, 'TP')
        self.assertAlmostEqual(t.r_gross, 1 / 0.75)
        self.assertAlmostEqual(t.r_net, 1 / 0.75 - 0.001 / 0.0075)

    def test_same_bar_tp_and_sl_is_sl(self):
        t = self.trade()
        self.assertTrue(sx.step(t, bar(t.entry_ms, 100, 101.5, 99.0, 100.0), self.p))
        self.assertEqual(t.status, 'SL')
        self.assertAlmostEqual(t.r_gross, -1.0)

    def test_gap_through_sl_fills_at_open(self):
        t = self.trade(side=-1)
        self.assertTrue(sx.step(t, bar(t.entry_ms + M5, 101.5, 101.6, 101.0, 101.2), self.p))
        self.assertEqual(t.status, 'SL')
        self.assertAlmostEqual(t.r_gross, -math.log(101.5 / 100) / 0.0075)

    def test_anomaly_exit_only_against_position(self):
        t = self.trade()
        self.assertFalse(sx.step(t, bar(t.entry_ms, 100, 100.6, 99.8, 100.5), self.p))   # cùng chiều
        self.assertTrue(sx.step(t, bar(t.entry_ms + M5, 100.4, 100.45, 99.6, 99.7), self.p))  # ngược, biên 0.85%
        self.assertEqual(t.status, 'ANOMALY')
        self.assertAlmostEqual(t.exit_price, 99.7)

    def test_timeout_after_max_hold(self):
        t = self.trade()
        closed = [sx.step(t, bar(t.entry_ms + i * M5, 100, 100.1, 99.9, 100.05), self.p) for i in range(12)]
        self.assertEqual(closed, [False] * 11 + [True])
        self.assertEqual(t.status, 'TIMEOUT')
        self.assertEqual(t.exit_close_ms, t.entry_ms + H)

    def test_bars_processed_once(self):
        t = self.trade()
        b0 = bar(t.entry_ms, 100, 100.2, 99.9, 100.1)
        sx.step(t, b0, self.p)
        self.assertFalse(sx.step(t, bar(t.entry_ms, 100, 120, 80, 100), self.p))   # cùng nến -> bỏ qua
        self.assertEqual(t.status, 'OPEN')

    def test_one_minute_levels_and_five_minute_anomaly(self):
        t = self.trade(t0=10 * H + 2 * 60_000)
        bars = [sx.Bar(10 * H + i * 60_000, 10 * H + (i + 1) * 60_000, 100, 100.1, 99.95, 100.0, True, False)
                for i in range(2, 5)]
        bars.append(sx.Bar(10 * H, 10 * H + M5, 100.5, 100.6, 99.7, 99.8, False, True))     # ngược, 0.9%
        out = [sx.step(t, b, self.p) for b in bars]
        self.assertEqual(out, [False, False, False, True])
        self.assertEqual(t.status, 'ANOMALY')

    def test_portfolio_rule_counts_trades_exiting_after_signal_hour(self):
        book = sx.Book(self.p)
        for k in range(2):
            t = sx.Trade(f't{k}', f's{k}', k * H, 1, 0.01)
            self.assertEqual(book.add_signal(t), (True, 'ok'))
            book.fill(t, (k + 1) * H, 100.0)
        self.assertEqual(book.admit(2 * H), (False, 'max_positions'))
        for t in book.trades:
            t.status, t.exit_ms, t.r_net, t.risk_frac = 'SL', 2 * H, -1.0, 0.01
        self.assertEqual(book.admit(2 * H)[0], False)          # thoát ở nến mở lúc >= giờ tín hiệu -> vẫn tính
        self.assertEqual(book.admit(3 * H)[0], True)


def have_history():
    need = [HD / 'BTCUSDT_positioning_5m.csv', HD / 'BTCUSDT_funding.csv', ART / 'walkforward_signals.csv',
            ART / 'models_final.joblib']
    return all(p.exists() for p in need) and (five_minute_file() is not None)


def five_minute_file():
    for name in ('BTCUSDT_5m_all.csv', 'BTCUSDT_5m_smoke.csv.gz'):
        if (HD / name).exists():
            return HD / name
    return None


@unittest.skipUnless(have_history(), 'cần historical_data + artifacts/1h_strategy_v1')
class ReplayParityTests(unittest.TestCase):
    def test_replay_equals_backtest_m5(self):
        import modelv2 as mv2
        import strategy_h1 as st
        cfg = st.StrategyConfig()
        W = pd.read_csv(ART / 'walkforward_signals.csv')
        W['timestamp'] = pd.to_datetime(W['timestamp'], utc=True)
        W = W[W['year'] == 2026].reset_index(drop=True)
        d5 = pd.read_csv(five_minute_file())
        d5 = d5[d5['open_time_ms'] >= int(W['timestamp'].min().value // 1_000_000)]
        h1 = st.aggregate_1h(d5)
        sd = st.StrategyData(df=h1, ds=None, x_dir=None, dir_names=(), pivots=None)
        W['bar'] = np.searchsorted(h1['timestamp'].to_numpy(), W['timestamp'].to_numpy())
        T, _ = st.backtest_m5(sd, W, cfg, d5)
        S = W[W['signal']].sort_values('timestamp')
        sig = [(str(i), int(r.timestamp.value // 1_000_000), int(r.side), float(r.b)) for i, r in S.iterrows()]
        D = mv2.prepare_candles(d5, 5)
        t5 = mv2._ns_int(D['timestamp']) // 1_000_000
        bars = (sx.Bar(int(t5[i]), int(t5[i]) + M5, *D.loc[i, ['open', 'high', 'low', 'close']]) for i in range(len(D)))
        book = sx.replay(sig, bars, sx.ExecParams.from_strategy(cfg))
        X = pd.DataFrame([t.to_dict() for t in book.trades if not t.is_open])
        X['entry_time'] = pd.to_datetime(X['entry_ms'], unit='ms', utc=True)
        M = T.merge(X, on='entry_time', how='inner')
        self.assertEqual(len(M), len(T))
        self.assertLess(float(np.abs(M['R'] - M['r_net']).max()), 1e-9)
        name = {'TP': 'TP', 'SL': 'SL', 'thoát nhanh 5m': 'ANOMALY', 'hết giờ': 'TIMEOUT'}
        self.assertTrue((M['result'].map(name) == M['status']).all())


def have_live_inputs():
    return (have_history() and os.getenv('TEST_DATABASE_URL') and (HD / 'BTCUSDT_metrics_5m.csv').exists()
            and (HD / 'BTCUSDT_premium_5m.csv').exists())


@unittest.skipUnless(have_live_inputs(), 'cần TEST_DATABASE_URL + historical_data (metrics, premium)')
class LivePipelineTests(unittest.TestCase):
    """Giả lập collector ghi 28 ngày dữ liệu REST vào DB rồi chạy worker theo từng giờ."""
    END = int(pd.Timestamp(os.getenv('TEST_LIVE_END', '2026-05-11'), tz='UTC').value // 1_000_000)
    DAYS_DB = 28
    HOURS = int(os.getenv('TEST_LIVE_HOURS', '168'))      # tuần 04-11/05/2026: 34 tín hiệu (đủ lệnh tối đa)

    @classmethod
    def setUpClass(cls):
        os.environ['DATABASE_URL'] = os.environ['TEST_DATABASE_URL']
        from pipeline.core import candle, metric, premium, validate
        from pipeline.storage import connect, init_schema, save_event
        os.chdir(ROOT)
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

    def test_hourly_signals_match_walkforward_and_trades_are_consistent(self):
        import strategy_h1 as st
        from pipeline import strategy_live as sl
        from pipeline.storage import connect
        fund = pd.read_csv(HD / 'BTCUSDT_funding.csv')
        d5 = self.d5

        def fake_rest(path, params, client=None):
            assert path == '/fapi/v1/klines', path
            f = d5[(d5['open_time_ms'] >= params['startTime']) & (d5['open_time_ms'] <= params['endTime'])].head(1500)
            return [[int(r.open_time_ms), r.open, r.high, r.low, r.close, r.volume, 0, 0, int(r.trades),
                     r.taker_buy_base_asset_volume, 0, 0] for r in f.itertuples()]

        W = pd.read_csv(ART / 'walkforward_signals.csv')
        W['h'] = pd.to_datetime(W['timestamp'], utc=True).astype('int64') // 1_000_000
        W = W.set_index('h')
        clock = {'now': 0}
        os.environ.update({'STRATEGY_ENTRY': 'bar_open', 'STRATEGY_HISTORY_DAYS': '200'})
        with mock.patch.object(sl, 'rest_get', side_effect=fake_rest), \
                mock.patch.object(sl, 'now_ms', side_effect=lambda: clock['now']), \
                mock.patch.object(sl.FundingCache, 'get',
                                  side_effect=lambda db=None: fund[fund['funding_time_ms'] < clock['now']]):
            with connect() as db:
                strat = sl.Strategy(db, sl.Settings())
                first = self.END - self.HOURS * H
                n_sig, evs = 0, []
                for Hh in range(first, self.END + 1, H):
                    clock['now'] = Hh + 120_000
                    ev = strat.run_hour(Hh)
                    self.assertIsNotNone(ev)
                    evs.append(ev)
                    w = W.loc[Hh - H]
                    self.assertAlmostEqual(ev['signal']['p_touch'], float(w['p_touch']), places=6)
                    self.assertAlmostEqual(ev['signal']['p_up'], float(w['p_up']), places=6)
                    self.assertEqual(ev['decision']['good'], bool(w['signal']))
                    n_sig += ev['decision']['good']
                    strat.manage()
                clock['now'] = self.END + 30 * 86_400_000
                trades = db.execute("SELECT * FROM strategy_trades ORDER BY hour_open_ms").fetchall()
        self.assertGreater(len(trades), 0)
        self.assertLessEqual(len(trades), n_sig)
        skipped = [e for e in evs if e['decision']['good'] and not e['decision']['admitted']]
        self.assertEqual(len(trades) + len(skipped), n_sig)
        cfg = st.StrategyConfig()
        p = sx.ExecParams.from_strategy(cfg)
        for tr in trades:
            if tr['status'] in ('PENDING', 'OPEN'):
                continue
            t = sx.Trade('x', 'x', tr['hour_open_ms'], tr['side'], tr['b_pct'])
            e = d5[d5['open_time_ms'] == tr['hour_open_ms'] + H].iloc[0]
            sx.open_trade(t, tr['hour_open_ms'] + H, float(e['open']), p, 1.0)
            f = d5[d5['open_time_ms'] >= t.entry_ms]
            for r in f.itertuples():
                if sx.step(t, sx.Bar(int(r.open_time_ms), int(r.open_time_ms) + M5, r.open, r.high, r.low, r.close), p):
                    break
            self.assertEqual(t.status, tr['status'])
            self.assertAlmostEqual(t.r_net, tr['r_net'], places=9)


if __name__ == '__main__':
    unittest.main()
