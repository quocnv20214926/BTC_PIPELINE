import os
import unittest
from unittest import mock

import httpx

from pipeline.telegram_bot import (H1, Telegram, TelegramError, fmt_dur, format_close, format_open, format_skipped,
                                   format_summary, summarize)

T0 = 1_760_000_000_000 // H1 * H1


def trade(side=1, status='OPEN', r=None):
    e = 60_000.0
    b = 0.012
    return {'trade_id': 'tr1', 'signal_id': 'sig1', 'hour_open_ms': T0, 'side': side, 'b_pct': b,
            'status': status, 'entry_ms': T0 + H1 + 120_000, 'entry': e,
            'tp': e * (1 + side * b), 'sl': e * (1 - side * 0.75 * b),
            'exit_price': e * (1 + side * b) if status == 'TP' else None,
            'exit_close_ms': T0 + 5 * H1, 'r_gross': r, 'r_net': None if r is None else r - 0.111,
            'equity_after': 1.0123, 'state': {'slippage_r': -0.02},
            'params': {'fee_pct': 0.001, 'sl_mult': 0.75, 'anomaly_range': 0.0075, 'max_hold_h': 720}}


EV = {'source': 'strategy_h1', 'hour_open_ms': T0,
      'signal': {'p_touch': 0.71, 'p_up': 0.62, 'conf': 0.12},
      'decision': {'good': True, 'side': 'LONG', 'admitted': False, 'skip_reason': 'max_positions'}}


class FormatTests(unittest.TestCase):
    def test_open_message(self):
        text = format_open(trade(), EV, 7, open_count=3, equity=1.05)
        self.assertIn('LONG BTCUSDT', text)
        self.assertIn('60,000.0', text)
        self.assertIn('+1.20%', text)                 # TP = 1b
        self.assertIn('-0.90%', text)                 # SL = 0.75b
        self.assertIn('p_touch 0.71', text)
        self.assertIn('3 lệnh đang mở', text)
        self.assertIn('×1.050', text)

    def test_short_open_levels(self):
        text = format_open(trade(side=-1), EV, 7)
        self.assertIn('SHORT', text)
        self.assertIn('-1.20%', text)
        self.assertIn('+0.90%', text)

    def test_close_message(self):
        text = format_close(trade(status='TP', r=1.3333), 7)
        self.assertIn('TP', text)
        self.assertIn('+1.33R', text)
        self.assertIn('sau phí +1.22R', text)
        self.assertIn('giữ 3h 58m', text)
        anomaly = format_close(dict(trade(status='ANOMALY', r=-0.4), exit_price=59_900.0), 7)
        self.assertIn('Thoát nhanh', anomaly)

    def test_skipped_and_summary(self):
        self.assertIn('đã đủ số lệnh tối đa', format_skipped(EV, 7))
        st = summarize([dict(status='TP', r_net=1.2), dict(status='SL', r_net=-1.1), dict(status='ANOMALY', r_net=-0.3)])
        self.assertEqual((st['n'], st['wins'], st['tp'], st['sl'], st['anomaly']), (3, 1, 1, 1, 1))
        self.assertIn('-0.20R', format_summary('7 ngày', st))
        self.assertEqual(format_summary('x', summarize([])), 'x: chưa có lệnh nào đóng.')

    def test_duration(self):
        self.assertEqual(fmt_dur(45 * 60_000), '45m')
        self.assertEqual(fmt_dur(26 * 3_600_000), '1d 2h')


class ClientTests(unittest.TestCase):
    def test_retry_on_flood_limit_and_no_token_in_error(self):
        calls = []

        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(429, json={'ok': False, 'parameters': {'retry_after': 0}})
            return httpx.Response(200, json={'ok': True, 'result': {'message_id': 7}})
        bot = Telegram('SECRET', client=httpx.Client(transport=httpx.MockTransport(handler)))
        with mock.patch('pipeline.telegram_bot.time.sleep'):
            self.assertEqual(bot.send('1', 'hi'), 7)
        self.assertEqual(len(calls), 2)
        bad = Telegram('SECRET', client=httpx.Client(transport=httpx.MockTransport(
            lambda r: httpx.Response(400, json={'ok': False, 'description': 'Bad Request'}))))
        with self.assertRaises(TelegramError) as ctx:
            bad.send('1', 'hi')
        self.assertNotIn('SECRET', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
