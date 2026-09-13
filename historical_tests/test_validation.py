import unittest
from collect_historical import validate

class ValidationTests(unittest.TestCase):
    def row(self, t):
        return dict(open_time_ms=t, open='100', high='110', low='90', close='105', volume='1')
    def test_contiguous(self):
        validate([self.row(0),self.row(900000)],0,1800000,900000)
    def test_missing(self):
        with self.assertRaises(ValueError):
            validate([self.row(0)],0,1800000,900000)
    def test_duplicate(self):
        with self.assertRaises(ValueError):
            validate([self.row(0),self.row(0)],0,1800000,900000)
    def test_invalid_ohlc(self):
        row=self.row(0); row['high']='99'
        with self.assertRaises(ValueError):
            validate([row],0,900000,900000)
    def test_nan(self):
        row=self.row(0); row['volume']='NaN'
        with self.assertRaises(ValueError):
            validate([row],0,900000,900000)

if __name__=='__main__':
    unittest.main()

class ResumeTests(unittest.TestCase):
    def test_second_run_reuses_valid_csv_without_network(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from unittest.mock import patch
        from collect_historical import collect
        import contextlib
        import io
        start=1577836800000
        end=start+3600000
        def fake_get(path, params):
            step=900000 if params['interval']=='15m' else 3600000
            return [[t,'100','110','90','105','1',t+step-1] for t in range(params['startTime'],params['endTime']+1,step)]
        with TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            with patch('collect_historical.get',side_effect=fake_get), patch('collect_historical.time.sleep'):
                collect(Path(folder),start,end)
            with patch('collect_historical.get',side_effect=AssertionError('Should reuse saved CSV')):
                collect(Path(folder),start,end)

class TotalsTests(unittest.TestCase):
    def test_new_month_added_and_rerun_not_duplicated(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from unittest.mock import patch
        from collect_historical import collect
        import csv
        import contextlib
        import io
        start=1580511600000  # 2020-01-31 23:00 UTC
        def fake_get(path, params):
            step=900000 if params['interval']=='15m' else 3600000
            return [[t,'100','110','90','105','1',t+step-1] for t in range(params['startTime'],params['endTime']+1,step)]
        with TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), patch('collect_historical.get',side_effect=fake_get), patch('collect_historical.time.sleep'):
            folder=Path(folder)
            collect(folder,start,start+3600000)
            for _ in range(2):
                collect(folder,start,start+7200000)
                for tf, expected in [('15m',8),('1h',2)]:
                    with (folder/('BTCUSDT_'+tf+'_all.csv')).open() as f:
                        rows=list(csv.DictReader(f))
                    self.assertEqual(len(rows),expected)
                    self.assertEqual(len({r['open_time_ms'] for r in rows}),expected)
