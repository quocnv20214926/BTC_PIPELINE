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


class StrategySourceTests(unittest.TestCase):
    def test_strategy_kinds_collected_on_5m(self):
        for kind in ('oi', 'top_acct_ratio', 'top_ratio', 'ratio', 'premium', 'candle'):
            self.assertIn(kind, COLLECT_KINDS_BY_TIMEFRAME['5m'])

    def test_top_trader_account_ratio(self):
        ts = 10 * M5
        row = {'symbol': 'BTCUSDT', 'longShortRatio': '1.31', 'longAccount': '0.5671', 'shortAccount': '0.4329',
               'timestamp': ts}
        e = metric('top_acct_ratio', '5m', row, 'test', 'backfill', received=ts + 1000)
        validate(e)
        self.assertEqual(e['data']['source_timestamp_ms'], ts)

    def test_premium_index_kline_can_be_negative(self):
        from pipeline.core import premium
        t = 10 * M5
        row = [t, '-0.00012', '0.00003', '-0.00031', '-0.00020', '0', t + M5 - 1, '0', 12, '0', '0', '0']
        e = premium('5m', row, 'test', 'live', received=t + M5 + 1000)
        validate(e)
        self.assertEqual(e['period_start_ms'], t)
        self.assertEqual(e['data']['close'], '-0.0002')
        bad = list(row)
        bad[3] = '0.5'
        with self.assertRaises(ValueError):
            validate(premium('5m', bad, 'test', 'live', received=t + M5 + 1000))

    def test_rest_rows_build_training_positioning_table(self):
        """REST (timestamp = create_time) -> cùng bảng như dữ liệu data.binance.vision sau 2024-03-04."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model_training'))
        import positioning_data as pdm
        t0 = pdm.CUTOVER_MS + 100 * M5
        grid = np.arange(t0, t0 + 10 * M5, M5)
        ts = [t0 + i * M5 for i in range(10) if i != 4]                        # thiếu 1 dòng -> giữ giá trị cũ
        oi = [{'timestamp': t, 'sumOpenInterest': 1000 + i, 'sumOpenInterestValue': 0} for i, t in enumerate(ts)]
        ls = [{'timestamp': t, 'longShortRatio': 1.1} for t in ts]
        prem = [[int(t), 0.0001, 0.0002, 0.0, 0.00015] for t in grid]
        m = pdm.rest_metrics(oi, ls, ls, ls)
        out = pdm.align(m, pdm.rest_premium(prem), grid)
        self.assertEqual(list(out['open_time_ms']), list(grid))
        self.assertEqual(out['oi'].iloc[4], out['oi'].iloc[3])
        self.assertEqual(out['m_age_bars'].iloc[4], 1)
        self.assertTrue((out['prem_close'] == 0.00015).all())
        vision = pd.DataFrame({'create_ms': ts, 'oi': [r['sumOpenInterest'] for r in oi], 'oi_usd': 0,
                               'top_acct_ls': 1.1, 'top_pos_ls': 1.1, 'acct_ls': 1.1})
        same = pdm.align(vision, pdm.rest_premium(prem), grid)
        pd.testing.assert_frame_equal(out, same)

    def test_metric_owner_before_and_after_cutover(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model_training'))
        import positioning_data as pdm
        c = pdm.CUTOVER_MS
        self.assertEqual(list(pdm.metric_owner_open_ms(np.array([c - M5, c, c + M5]))), [c - 2 * M5, c, c + M5])


if __name__ == '__main__':
    unittest.main()
