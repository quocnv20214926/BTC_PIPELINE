"""Tải dữ liệu vị thế lịch sử từ data.binance.vision và dựng lại bảng train BTCUSDT_positioning_5m.csv.

Binance REST chỉ giữ ~30 ngày open interest / long-short ratio, nên dữ liệu dài hạn để TRAIN lấy từ file ngày của
data.binance.vision (cùng nguồn đã dùng để dựng bảng hiện tại). Live không cần script này: collector ghi REST vào
PostgreSQL và pipeline/strategy_live.py dựng bảng bằng cùng hàm model_training/positioning_data.align.

  historical_data/BTCUSDT_metrics_5m.csv   create_time, sum_open_interest, sum_open_interest_value,
                                           count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
                                           count_long_short_ratio, sum_taker_long_short_vol_ratio, date
  historical_data/BTCUSDT_premium_5m.csv   open_time, open, high, low, close, volume, close_time, quote_volume,
                                           count, taker_buy_volume, taker_buy_quote_volume, date

Chỉ tải các ngày còn thiếu (sau ngày cuối trong file) đến hôm qua (UTC); ngày chưa được công bố (404) bỏ qua.
Sau đó gọi positioning_data để dựng lại historical_data/BTCUSDT_positioning_5m.csv trên lưới nến 5m
(chạy collect_historical.py --timeframes 5m trước để có nến mới nhất).

Dùng:
  python collect_positioning.py                 # cập nhật hai file nguồn + dựng lại bảng vị thế
  python collect_positioning.py --no-rebuild    # chỉ tải
"""
import argparse
import csv
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

VISION = 'https://data.binance.vision/data/futures/um/daily'
FOLDER = Path(__file__).resolve().parent / 'historical_data'
METRICS = FOLDER / 'BTCUSDT_metrics_5m.csv'
PREMIUM = FOLDER / 'BTCUSDT_premium_5m.csv'
METRIC_FIELDS = ['create_time', 'sum_open_interest', 'sum_open_interest_value', 'count_toptrader_long_short_ratio',
                 'sum_toptrader_long_short_ratio', 'count_long_short_ratio', 'sum_taker_long_short_vol_ratio', 'date']
PREMIUM_FIELDS = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'count',
                  'taker_buy_volume', 'taker_buy_quote_volume', 'date']
FIRST = {'metrics': date(2020, 9, 1), 'premium': date(2019, 12, 31)}


def download(url):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code not in (429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f'download failed: {url}')


def read_zip_csv(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        text = z.open(z.namelist()[0]).read().decode('utf-8')
    rows = list(csv.reader(io.StringIO(text)))
    if rows and not rows[0][0].replace('-', '').replace(':', '').replace(' ', '').isdigit():
        rows = rows[1:]                                   # bỏ dòng tiêu đề (file mới có header)
    return rows


def last_date(path):
    if not path.exists():
        return None
    with path.open('rb') as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 4096))
        tail = f.read().decode('utf-8', errors='ignore').strip().splitlines()
    return date.fromisoformat(tail[-1].split(',')[-1]) if len(tail) > 1 else None


def append(path, fields, rows):
    new = not path.exists()
    with path.open('a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(fields)
        w.writerows(rows)


def update(kind, until):
    path = METRICS if kind == 'metrics' else PREMIUM
    start = (last_date(path) or FIRST[kind] - timedelta(days=1)) + timedelta(days=1)
    added = 0
    d = start
    while d <= until:
        ds = d.isoformat()
        if kind == 'metrics':
            url = f'{VISION}/metrics/BTCUSDT/BTCUSDT-metrics-{ds}.zip'
        else:
            url = f'{VISION}/premiumIndexKlines/BTCUSDT/5m/BTCUSDT-5m-{ds}.zip'
        blob = download(url)
        if blob is None:
            print(f'{kind} {ds}: chưa có trên data.binance.vision, dừng', flush=True)
            break
        rows = read_zip_csv(blob)
        if kind == 'metrics':                             # create_time, symbol, ... -> bỏ cột symbol
            out = [[r[0], *r[2:8], ds] for r in rows]
        else:
            out = [[*r[:11], ds] for r in rows]
        append(path, METRIC_FIELDS if kind == 'metrics' else PREMIUM_FIELDS, out)
        added += len(out)
        print(f'{kind} {ds}: +{len(out)} dòng', flush=True)
        d += timedelta(days=1)
        time.sleep(0.2)
    return added


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--until', help='ngày cuối (YYYY-MM-DD, UTC); mặc định hôm qua')
    ap.add_argument('--no-rebuild', action='store_true')
    a = ap.parse_args()
    FOLDER.mkdir(exist_ok=True)
    until = date.fromisoformat(a.until) if a.until else datetime.now(timezone.utc).date() - timedelta(days=1)
    for kind in ('metrics', 'premium'):
        print(kind, ': thêm', update(kind, until), 'dòng')
    if not a.no_rebuild:
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'model_training'))
        import pandas as pd
        import positioning_data as pdm
        k = pd.read_csv(FOLDER / 'BTCUSDT_5m_all.csv', usecols=lambda c: c in ('open_time_ms', 'close'))
        out = pdm.align(pdm.vision_metrics(pd.read_csv(METRICS)), pdm.vision_premium(pd.read_csv(PREMIUM)),
                        k['open_time_ms'], ref_close=k.set_index('open_time_ms')['close'])
        out.to_csv(FOLDER / 'BTCUSDT_positioning_5m.csv', index=False)
        print('BTCUSDT_positioning_5m.csv:', len(out), 'dòng')


if __name__ == '__main__':
    main()
