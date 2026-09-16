import unittest

from pipeline.core import INTERVALS, KINDS_BY_TIMEFRAME, candle, validate


class TimeframeConfigurationTests(unittest.TestCase):
    def test_all_requested_timeframes_are_configured(self):
        self.assertEqual(set(INTERVALS), {'1m', '5m', '15m', '1h'})

    def test_one_minute_does_not_request_unavailable_metrics(self):
        self.assertEqual(KINDS_BY_TIMEFRAME['1m'], ('candle',))
        self.assertEqual(KINDS_BY_TIMEFRAME['5m'], ('candle', 'ratio', 'taker'))

    def test_one_minute_candle_keeps_native_taker_buy_fields(self):
        row = [0, '100', '110', '90', '105', '2', 59_999,
               '205', 10, '1.2', '123']
        event = candle('1m', row, 'test', 'backfill', received=60_000)
        validate(event)
        self.assertEqual(event['data']['taker_buy_base'], '1.2')
        self.assertEqual(event['data']['taker_buy_quote'], '123')


if __name__ == '__main__':
    unittest.main()
