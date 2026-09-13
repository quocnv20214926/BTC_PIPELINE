"""Standalone Binance USD-M OHLCV collector. Python standard library only."""
import argparse
import csv
import hashlib
import io
import json
import time
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

BASE = 'https://fapi.binance.com'
STEPS = {'15m': 900000, '1h': 3600000}
FIELDS = ['open_time_ms', 'open_time_utc', 'open', 'high', 'low', 'close', 'volume']

def get(path, params):
    url = BASE + path + '?' + urllib.parse.urlencode(params)
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=40) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code not in (418, 429, 500, 502, 503, 504):
                raise
            delay = max(float(exc.headers.get('Retry-After', '0')), min(60, 2 ** attempt))
        except (urllib.error.URLError, TimeoutError):
            delay = min(60, 2 ** attempt)
        print('Retry in', delay, 'seconds', flush=True)
        time.sleep(delay)
    raise RuntimeError('HTTP retry budget exhausted')

def utc(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()

def atomic(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(data, encoding='utf-8')
    temp.replace(path)

def validate(rows, start, end, step):
    expected = list(range(start, end, step))
    if [int(r['open_time_ms']) for r in rows] != expected:
        raise ValueError('Missing, duplicate or unordered candles in requested range')
    for r in rows:
        o, h, l, c, v = [Decimal(r[k]) for k in ('open', 'high', 'low', 'close', 'volume')]
        if not all(x.is_finite() for x in (o,h,l,c,v)) or min(o,h,l,c) <= 0 or v < 0 or h < max(o,c,l) or l > min(o,c,h):
            raise ValueError('Invalid OHLCV')

def merge_totals(folder, start, end):
    totals = []
    for tf, step in STEPS.items():
        rows = []
        finish = end // step * step
        for part in sorted((folder / tf).glob('BTCUSDT_' + tf + '_????-??.csv')):
            with part.open(encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != FIELDS:
                    raise ValueError(f'Unexpected columns: {part}')
                rows.extend(r for r in reader if start <= int(r['open_time_ms']) < finish)
        rows.sort(key=lambda r: int(r['open_time_ms']))
        validate(rows, start, finish, step)
        path = folder / ('BTCUSDT_' + tf + '_all.csv')
        buf = io.StringIO(newline='')
        writer = csv.DictWriter(buf, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        atomic(path, buf.getvalue())
        totals.append({'path':path.name, 'rows':len(rows),
                       'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return totals


def collect(folder, start, end):
    manifest = {'market':'Binance USD-M perpetual', 'symbol':'BTCUSDT', 'source':BASE+'/fapi/v1/klines', 'start_utc':utc(start), 'end_exclusive_utc':utc(end), 'volume_unit':'BTC', 'timezone':'UTC', 'files':[]}
    for tf, step in STEPS.items():
        finish = end // step * step
        cursor = start
        target = folder / tf
        target.mkdir(parents=True, exist_ok=True)
        while cursor < finish:
            date = datetime.fromtimestamp(cursor/1000, timezone.utc)
            nxt = datetime(date.year + (date.month == 12), date.month % 12 + 1, 1, tzinfo=timezone.utc)
            until = min(finish, int(nxt.timestamp()*1000))
            path = target / ('BTCUSDT_'+tf+'_'+date.strftime('%Y-%m')+'.csv')
            rows = []
            if path.exists():
                with path.open(encoding='utf-8', newline='') as f:
                    rows = list(csv.DictReader(f))
                # Reuse only a fully validated exact range, otherwise redownload this month.
                try:
                    validate(rows, cursor, until, step)
                except (ValueError, KeyError, ArithmeticError):
                    rows = []
            if not rows:
                page = cursor
                while page < until:
                    page_end = min(until, page+1500*step)
                    batch = get('/fapi/v1/klines', {'symbol':'BTCUSDT','interval':tf,'startTime':page,'endTime':page_end-1,'limit':1500})
                    if not isinstance(batch, list):
                        raise ValueError('Unexpected API response')
                    for r in batch:
                        t = int(r[0])
                        if not page <= t < page_end or int(r[6])+1 != t+step:
                            raise ValueError('API candle outside requested range')
                        rows.append(dict(zip(FIELDS, [t,utc(t),*r[1:6]])))
                    page = page_end
                    time.sleep(0.2)
                validate(rows, cursor, until, step)
                buf = io.StringIO(newline='')
                writer = csv.DictWriter(buf, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
                atomic(path, buf.getvalue())
            manifest['files'].append({'path':str(path.relative_to(folder)), 'rows':len(rows), 'first_utc':rows[0]['open_time_utc'], 'last_utc':rows[-1]['open_time_utc'], 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            print(tf, date.strftime('%Y-%m'), len(rows), 'candles OK', flush=True)
            cursor = until
        print(tf, 'complete', flush=True)
    manifest['totals'] = merge_totals(folder, start, end)
    manifest['completed_at_utc'] = utc(int(time.time()*1000))
    manifest['total_rows'] = sum(f['rows'] for f in manifest['files'])
    atomic(folder/'manifest.json', json.dumps(manifest, indent=2))
    print('DONE:', manifest['total_rows'], 'candles;', folder, flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--end', help='Exclusive UTC end date, YYYY-MM-DD; default latest closed candles')
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent/'historical_data'
    folder.mkdir(exist_ok=True)
    start = int(datetime(2020,1,1,tzinfo=timezone.utc).timestamp()*1000)
    end = int(datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc).timestamp()*1000) if args.end else int(get('/fapi/v1/time', {})['serverTime'])
    if end <= start:
        parser.error('end must be after 2020-01-01')
    collect(folder, start, end)

if __name__ == '__main__':
    main()
