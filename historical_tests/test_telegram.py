import json
import os
import unittest
from unittest import mock

import httpx

from pipeline.core import INSTRUMENT

from pipeline.telegram_bot import (M1, M5, Telegram, TelegramError, evaluate_outcome, format_result, format_signal,
                                   summarize, trade_levels)

T0 = 1_760_000_000_000 // M5 * M5


def event(side='LONG', p=60_000.0, b=0.01, t0=T0, good=True, signal_id='sig1'):
    return {
        'signal_id': signal_id, 'event_time_ms': t0, 'close': p, 'source': 'modelv2_5m',
        'signal': {'barrier_pct': b, 'p_touch': 0.7, 'p_up_touch': 0.62 if side == 'LONG' else 0.38,
                   'long_sl': p * (1 - b), 'long_tp_1r': p * (1 + b), 'long_tp_2r': p * (1 + 2 * b),
                   'short_sl': p * (1 + b), 'short_tp_1r': p * (1 - b), 'short_tp_2r': p * (1 - 2 * b),
                   'e_long_1r': 0.1, 'e_short_1r': -0.1, 'e_long_2r': 0.05, 'e_short_2r': -0.2,
                   'time_stop_bars': 48},
        'decision': {'side': side, 'confidence': 0.12, 'score': 0.12, 'threshold': 0.05, 'good': good},
    }


def flat_bars(n, p=60_000.0, step=M1, t0=T0, spread=10.0):
    return [(t0 + i * step, p + spread, p - spread, p) for i in range(n)]


class OutcomeTests(unittest.TestCase):
    def test_long_take_profit(self):
        bars = flat_bars(10)
        bars[5] = (bars[5][0], 60_650.0, 59_990.0, 60_600.0)
        out = evaluate_outcome(event(), bars, M1, 1, 0.001)
        self.assertEqual(out['status'], 'TP')
        self.assertAlmostEqual(out['r_gross'], 1.0)
        self.assertAlmostEqual(out['r_net'], 0.9)          # fee 0.1% / barrier 1% = 0.1R
        self.assertEqual(out['minutes'], 6)

    def test_short_stop_and_same_bar_is_stop(self):
        bars = flat_bars(10)
        bars[3] = (bars[3][0], 60_700.0, 59_300.0, 60_000.0)   # touches SL and TP of a short
        out = evaluate_outcome(event('SHORT'), bars, M1, 1, 0.001)
        self.assertEqual(out['status'], 'SL')
        self.assertTrue(out['ambiguous'])
        self.assertAlmostEqual(out['r_gross'], -1.0)

    def test_timeout_and_open(self):
        bars = flat_bars(240)
        bars[-1] = (bars[-1][0], 60_310.0, 60_290.0, 60_300.0)
        out = evaluate_outcome(event(), bars, M1, 1, 0.0)
        self.assertEqual(out['status'], 'TIMEOUT')
        self.assertAlmostEqual(out['r_gross'], 0.5)
        self.assertEqual(out['exit_ms'], T0 + 48 * M5)
        live = evaluate_outcome(event(), bars[:100], M1, 1, 0.0)
        self.assertEqual(live['status'], 'OPEN')
        self.assertAlmostEqual(live['r_now'], 0.0)

    def test_gap_and_bars_before_signal_ignored(self):
        bars = [(T0 - M1, 99e9, 0, 1)] + flat_bars(5) + flat_bars(5, t0=T0 + 7 * M1)
        self.assertEqual(evaluate_outcome(event(), bars, M1, 1, 0.0)['status'], 'GAP')

    def test_two_r_target(self):
        bars = flat_bars(10)
        bars[2] = (bars[2][0], 60_700.0, 59_990.0, 60_600.0)    # 1R hit, 2R not
        self.assertEqual(evaluate_outcome(event(), bars, M1, 2, 0.0)['status'], 'OPEN')
        bars[4] = (bars[4][0], 61_250.0, 60_500.0, 61_100.0)
        out = evaluate_outcome(event(), bars, M1, 2, 0.0)
        self.assertEqual(out['status'], 'TP')
        self.assertAlmostEqual(out['r_gross'], 2.0)
        self.assertAlmostEqual(trade_levels(event(), 2)['rr'], 2.0)


class FormatTests(unittest.TestCase):
    def test_signal_message(self):
        text = format_signal(event('SHORT'), 1, 0.001, 7, 'modelv2_5m')
        self.assertIn('SHORT BTCUSDT', text)
        self.assertIn('SL: 60,600.0 (+1.00%)', text)
        self.assertIn('TP 1R: 59,400.0 (-1.00%) ← theo dõi', text)
        self.assertIn('phí ≈ 0.10R', text)
        self.assertIn('R kỳ vọng sau phí) +0.12R · ngưỡng +0.05R', text)

    def test_result_and_summary(self):
        out = evaluate_outcome(event(), [(T0, 60_700.0, 59_990.0, 60_600.0)], M1, 1, 0.001)
        self.assertIn('TP 1R', format_result(event(), out, 1, 7))
        st = summarize([out, dict(out, status='SL', r_net=-1.1)])
        self.assertEqual((st['n'], st['wins'], st['tp'], st['sl']), (2, 1, 1, 1))
        self.assertAlmostEqual(st['sum_r'], -0.2)


class ClientTests(unittest.TestCase):
    def test_retry_after_flood_limit_and_no_token_in_errors(self):
        calls = []

        def handler(request):
            calls.append(json.loads(request.content))
            if len(calls) == 1:
                return httpx.Response(429, json={'ok': False, 'parameters': {'retry_after': 0}})
            if calls[-1].get('chat_id') == 'bad':
                return httpx.Response(400, json={'ok': False, 'description': 'chat not found'})
            return httpx.Response(200, json={'ok': True, 'result': {'message_id': 42}})

        bot = Telegram('123:SECRET', httpx.Client(transport=httpx.MockTransport(handler)))
        with mock.patch('time.sleep'):
            self.assertEqual(bot.send('1', 'hi', reply_to=7), 42)
        self.assertEqual(calls[-1]['reply_parameters']['message_id'], 7)
        with self.assertRaises(TelegramError) as ctx:
            bot.send('bad', 'x')
        self.assertNotIn('SECRET', str(ctx.exception))


@unittest.skipUnless(os.getenv('TEST_DATABASE_URL'), 'needs TEST_DATABASE_URL (empty PostgreSQL database)')
class NotifierDatabaseTests(unittest.TestCase):
    """End-to-end on a real PostgreSQL: schema.sql, signal_events, observations, telegram_messages."""

    def setUp(self):
        from pathlib import Path
        from psycopg.types.json import Jsonb
        from pipeline import storage
        os.environ['DATABASE_URL'] = os.environ['TEST_DATABASE_URL']
        self.Jsonb = Jsonb
        self.db = storage.connect()
        self.db.execute('DROP SCHEMA public CASCADE; CREATE SCHEMA public')
        self.db.execute(Path('sql/schema.sql').read_text())
        self.sent = []

        def handler(request):
            body = json.loads(request.content)
            method = request.url.path.rsplit('/', 1)[-1]
            if method == 'getUpdates':
                return httpx.Response(200, json={'ok': True, 'result': []})
            self.sent.append(body)
            return httpx.Response(200, json={'ok': True, 'result': {'message_id': 100 + len(self.sent)}})

        from pipeline.telegram_bot import Notifier
        self.bot = Telegram('1:x', httpx.Client(transport=httpx.MockTransport(handler)))
        self.n = Notifier(self.db, self.bot, '555')
        self.n.source = 'modelv2_5m'                       # các dòng test được ghi với source v2

    def tearDown(self):
        self.db.close()

    def add_signal(self, ev):
        self.db.execute("""INSERT INTO signal_events(signal_id,instrument,signal_type,source,timeframe,event_time_ms,
            produced_at_ms,payload) VALUES(%s,%s,'model','modelv2_5m','5m',%s,%s,%s)""",
                        (ev['signal_id'], INSTRUMENT, ev['event_time_ms'], ev['event_time_ms'], self.Jsonb(ev)))

    def add_bars(self, bars, tf='1m'):
        step = M1 if tf == '1m' else M5
        for t, h, l, c in bars:
            self.db.execute("""INSERT INTO raw_events(event_id,instrument,timeframe,kind,period_start_ms,received_at_ms,
                envelope) VALUES(%s,%s,%s,'candle',%s,%s,'{}')""", (f'e{tf}{t}', INSTRUMENT, tf, t, t + step))
            self.db.execute("""INSERT INTO observations VALUES(%s,%s,'candle',%s,%s,%s,%s,'live',%s)""",
                            (INSTRUMENT, tf, t, t + step, f'e{tf}{t}', t + step,
                             self.Jsonb({'open': c, 'high': h, 'low': l, 'close': c, 'volume': 1})))

    def test_full_cycle(self):
        from pipeline import telegram_bot as tb
        now = T0 + 3 * M1
        old = event(t0=T0 - 3600_000, signal_id='old')              # too old: never sent
        a = event(signal_id='a')
        b = event('SHORT', t0=T0 + M5, signal_id='b')             # arrives while 'a' is open -> skipped
        weak = event(t0=T0 + 2 * M5, good=False, signal_id='weak')
        for ev in (old, a, b, weak):
            self.add_signal(ev)
        self.n.one_at_a_time = True
        with mock.patch.object(tb, 'now_ms', return_value=now):
            self.n.daily_hour = -1
            r = self.n.cycle()
        self.assertEqual(r, {'signals_sent': 1, 'results_sent': 0})
        signal_msgs = [m for m in self.sent if 'LONG BTCUSDT' in m['text']]
        self.assertEqual(len(signal_msgs), 1)
        kinds = {row['dedup_key']: row['kind'] for row in self.db.execute('SELECT * FROM telegram_messages')}
        self.assertEqual(kinds, {'signal:a': 'signal', 'skipped:b': 'skipped'})

        # price reaches TP on the 4th 1m bar -> reply under the signal message
        bars = flat_bars(6)
        bars[3] = (bars[3][0], 60_650.0, 59_995.0, 60_620.0)
        self.add_bars(bars)
        with mock.patch.object(tb, 'now_ms', return_value=T0 + 7 * M1):
            r = self.n.cycle()
            again = self.n.cycle()                                 # idempotent
        self.assertEqual(r['results_sent'], 1)
        self.assertEqual(again, {'signals_sent': 0, 'results_sent': 0})
        result = [m for m in self.sent if 'TP 1R' in m['text'] and 'reply_parameters' in m]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['reply_parameters']['message_id'], 101 + self.sent.index(signal_msgs[0]))
        with mock.patch.object(tb, 'now_ms', return_value=T0 + 7 * M1):
            status = self.n.report_text()
        self.assertIn('24h: 1 lệnh', status)
        self.assertIn('Không có lệnh đang mở', status)

    def test_falls_back_to_5m_and_no_data(self):
        from pipeline import telegram_bot as tb
        a = event('SHORT', signal_id='a')
        self.add_signal(a)
        with mock.patch.object(tb, 'now_ms', return_value=T0 + M1):
            self.n.cycle()
        self.add_bars(flat_bars(3, step=M5) + [(T0 + 3 * M5, 60_010.0, 59_350.0, 59_400.0)], tf='5m')
        with mock.patch.object(tb, 'now_ms', return_value=T0 + 5 * M5):
            self.assertEqual(self.n.cycle()['results_sent'], 1)
        out = self.db.execute("SELECT payload FROM telegram_messages WHERE kind='result'").fetchone()['payload']['outcome']
        self.assertEqual((out['status'], out['bar_ms']), ('TP', M5))

        b = event(t0=T0 + 10 * M5, signal_id='b')
        self.add_signal(b)
        with mock.patch.object(tb, 'now_ms', return_value=T0 + 10 * M5 + M1):
            self.n.cycle()
        with mock.patch.object(tb, 'now_ms', return_value=T0 + 10 * M5 + 5 * 3600_000):
            self.n.cycle()
        out = self.db.execute("SELECT payload FROM telegram_messages WHERE dedup_key='result:b'").fetchone()
        self.assertEqual(out['payload']['outcome']['status'], 'NO_DATA')


if __name__ == '__main__':
    unittest.main()
