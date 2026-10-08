"""Standalone Binance USD-M BTCUSDT collector (klines + funding rate). Python standard library only.

Klines keep every field Binance returns (not only OHLCV):
  quote_volume, trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume
-> taker buy volume = aggressive buying; volume - taker buy = aggressive selling (order-flow imbalance).
Monthly files written by the old OHLCV-only version are detected (missing columns) and re-downloaded.

Funding rate history (/fapi/v1/fundingRate, every 8h since listing) is written to
historical_data/BTCUSDT_funding.csv.

Usage:
  python collect_historical.py                         # all timeframes + funding, up to the latest closed candle
  python collect_historical.py --timeframes 5m 1h      # skip the heavy 1m download
  python collect_historical.py --end 2026-10-01 --no-funding
"""
import argparse
import csv
import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

BASE = 'https://fapi.binance.com'
STEPS = {'1m': 60000, '5m': 300000, '15m': 900000, '1h': 3600000}
FIELDS = ['open_time_ms', 'open_time_utc', 'open', 'high', 'low', 'close', 'volume',
          'quote_volume', 'trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
FUNDING_FIELDS = ['funding_time_ms', 'funding_time_utc', 'funding_rate', 'mark_price']
START_UTC = datetime(2020, 7, 1, tzinfo=timezone.utc)
FUNDING_FILE = 'BTCUSDT_funding.csv'


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


def write_csv(path, fields, rows):
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic(path, buf.getvalue())


def kline_row(r):
    """Binance kline array -> dict with every useful field."""
    t = int(r[0])
    return dict(zip(FIELDS, [t, utc(t), r[1], r[2], r[3], r[4], r[5], r[7], int(r[8]), r[9], r[10]]))


def validate(rows, start, end, step):
    expected = list(range(start, end, step))
    if [int(r['open_time_ms']) for r in rows] != expected:
        raise ValueError('Missing, duplicate or unordered candles in requested range')
    for r in rows:
        o, h, l, c, v = [Decimal(r[k]) for k in ('open', 'high', 'low', 'close', 'volume')]
        if not all(x.is_finite() for x in (o, h, l, c, v)) or min(o, h, l, c) <= 0 or v < 0 \
                or h < max(o, c, l) or l > min(o, c, h):
            raise ValueError('Invalid OHLCV')
        if 'taker_buy_base_asset_volume' in r:          # full kline layout (absent in old OHLCV-only rows)
            qv, tb, tq = [Decimal(r[k]) for k in ('quote_volume', 'taker_buy_base_asset_volume',
                                                   'taker_buy_quote_asset_volume')]
            if not all(x.is_finite() for x in (qv, tb, tq)) or min(qv, tb, tq) < 0 or tb > v or tq > qv \
                    or int(r['trades']) < 0:
                raise ValueError('Invalid volume / trade fields')


def read_month(path):
    """Rows of an existing monthly file, or [] when it has the old (OHLCV-only) layout."""
    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            return []
        return list(reader)


def merge_totals(folder, start, end, timeframes):
    totals = []
    for tf in timeframes:
        step = STEPS[tf]
        rows = []
        finish = end // step * step
        for part in sorted((folder / tf).glob('BTCUSDT_' + tf + '_????-??.csv')):
            # Ignore monthly files outside [start, finish) (e.g. leftovers of an older, earlier-start collection).
            year, month = map(int, part.stem.rsplit('_', 1)[1].split('-'))
            m0 = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)
            m1 = int(datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
            if m1 <= start or m0 >= finish:
                continue
            with part.open(encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != FIELDS:
                    raise ValueError(f'Unexpected columns (old layout?): {part}')
                rows.extend(r for r in reader if start <= int(r['open_time_ms']) < finish)
        rows.sort(key=lambda r: int(r['open_time_ms']))
        validate(rows, start, finish, step)
        path = folder / ('BTCUSDT_' + tf + '_all.csv')
        write_csv(path, FIELDS, rows)
        totals.append({'path': path.name, 'rows': len(rows),
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return totals


def collect_klines(folder, start, end, timeframes):
    files = []
    for tf in timeframes:
        step = STEPS[tf]
        finish = end // step * step
        cursor = start
        target = folder / tf
        target.mkdir(parents=True, exist_ok=True)
        while cursor < finish:
            date = datetime.fromtimestamp(cursor / 1000, timezone.utc)
            nxt = datetime(date.year + (date.month == 12), date.month % 12 + 1, 1, tzinfo=timezone.utc)
            until = min(finish, int(nxt.timestamp() * 1000))
            path = target / ('BTCUSDT_' + tf + '_' + date.strftime('%Y-%m') + '.csv')
            rows, source = [], 'download'
            if path.exists():
                rows = read_month(path)
                # Reuse only a fully validated exact range with the full column layout, otherwise redownload.
                try:
                    validate(rows, cursor, until, step)
                    source = 'cached'
                except (ValueError, KeyError, ArithmeticError, TypeError):
                    rows = []
            if not rows:
                page = cursor
                while page < until:
                    page_end = min(until, page + 1500 * step)
                    batch = get('/fapi/v1/klines', {'symbol': 'BTCUSDT', 'interval': tf, 'startTime': page,
                                                    'endTime': page_end - 1, 'limit': 1500})
                    if not isinstance(batch, list):
                        raise ValueError('Unexpected API response')
                    for r in batch:
                        t = int(r[0])
                        if not page <= t < page_end or int(r[6]) + 1 != t + step:
                            raise ValueError('API candle outside requested range')
                        rows.append(kline_row(r))
                    page = page_end
                    time.sleep(0.2)
                validate(rows, cursor, until, step)
                write_csv(path, FIELDS, rows)
            files.append({'path': str(path.relative_to(folder)), 'rows': len(rows),
                          'first_utc': rows[0]['open_time_utc'], 'last_utc': rows[-1]['open_time_utc'],
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            print(tf, date.strftime('%Y-%m'), len(rows), 'candles', source, flush=True)
            cursor = until
        print(tf, 'complete', flush=True)
    return files


def collect_funding(folder, start, end):
    """Full funding-rate history in [start, end). Funding is settled every 8h (time stamps may carry a few ms)."""
    rows, cursor = [], start
    while cursor < end:
        batch = get('/fapi/v1/fundingRate', {'symbol': 'BTCUSDT', 'startTime': cursor, 'endTime': end - 1,
                                             'limit': 1000})
        if not isinstance(batch, list):
            raise ValueError('Unexpected funding API response')
        if not batch:
            break
        for r in batch:
            t = int(r['fundingTime'])
            if start <= t < end:
                rate = Decimal(r['fundingRate'])
                if not rate.is_finite():
                    raise ValueError('Invalid funding rate')
                rows.append({'funding_time_ms': t, 'funding_time_utc': utc(t), 'funding_rate': r['fundingRate'],
                             'mark_price': r.get('markPrice') or ''})
        last = int(batch[-1]['fundingTime'])
        if last + 1 <= cursor:
            break
        cursor = last + 1
        time.sleep(0.2)
    rows = sorted({r['funding_time_ms']: r for r in rows}.values(), key=lambda r: r['funding_time_ms'])
    if not rows:
        raise ValueError('No funding rows returned')
    gaps = [(a['funding_time_utc'], b['funding_time_utc']) for a, b in zip(rows, rows[1:])
            if b['funding_time_ms'] - a['funding_time_ms'] > 9 * 3600 * 1000]
    if gaps:
        print(f'Warning: {len(gaps)} funding intervals longer than 9h, first: {gaps[0]}', flush=True)
    path = folder / FUNDING_FILE
    write_csv(path, FUNDING_FIELDS, rows)
    print('funding', len(rows), 'rows', rows[0]['funding_time_utc'], '->', rows[-1]['funding_time_utc'], flush=True)
    return {'path': path.name, 'rows': len(rows), 'first_utc': rows[0]['funding_time_utc'],
            'last_utc': rows[-1]['funding_time_utc'], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def collect(folder, start, end, timeframes=None, funding=False):
    timeframes = list(STEPS) if timeframes is None else timeframes
    manifest_path = folder / 'manifest.json'
    old = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    keep_files = [f for f in old.get('files', [])
                  if f['path'].replace('\\', '/').split('/')[0] not in timeframes]
    keep_totals = [t for t in old.get('totals', [])
                   if not any(t['path'] == f'BTCUSDT_{tf}_all.csv' for tf in timeframes)]
    manifest = {'market': 'Binance USD-M perpetual', 'symbol': 'BTCUSDT', 'source': BASE + '/fapi/v1/klines',
                'start_utc': utc(start), 'end_exclusive_utc': utc(end), 'volume_unit': 'BTC', 'timezone': 'UTC',
                'kline_fields': FIELDS, 'files': keep_files + collect_klines(folder, start, end, timeframes)}
    manifest['totals'] = keep_totals + merge_totals(folder, start, end, timeframes)
    if funding:
        manifest['funding'] = collect_funding(folder, start, end)
    elif 'funding' in old:
        manifest['funding'] = old['funding']
    manifest['completed_at_utc'] = utc(int(time.time() * 1000))
    manifest['total_rows'] = sum(f['rows'] for f in manifest['files'])
    atomic(manifest_path, json.dumps(manifest, indent=2))
    print('DONE:', manifest['total_rows'], 'candles;', folder, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--end', help='Exclusive UTC end date, YYYY-MM-DD; default latest closed candles')
    parser.add_argument('--timeframes', nargs='+', default=list(STEPS), choices=list(STEPS),
                        help='timeframes to collect (default: all)')
    parser.add_argument('--no-funding', action='store_true', help='skip the funding-rate history')
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent / 'historical_data'
    folder.mkdir(exist_ok=True)
    start = int(START_UTC.timestamp() * 1000)
    end = int(datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc).timestamp() * 1000) if args.end \
        else int(get('/fapi/v1/time', {})['serverTime'])
    if end <= start:
        parser.error(f'end must be after {START_UTC.date().isoformat()}')
    timeframes = [tf for tf in STEPS if tf in args.timeframes]
    collect(folder, start, end, timeframes, funding=not args.no_funding)


if __name__ == '__main__':
    main()
