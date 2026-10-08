"""Worker `model-signals-v3`: model giao dịch theo sự kiện (model_training/modelv3.py) trên dữ liệu live.

Mỗi khi có nến 5m mới đóng (dựng từ nến 1m trong observations):
  1. đọc MODEL_V3_HISTORY_DAYS ngày nến 1m (mặc định 20, model cần >= 10 ngày): lấy từ DB, phút nào thiếu thì
     bổ sung từ REST klines của Binance,
  2. funding đã chốt (REST, ~40 ngày),
  3. modelv3.predict_live -> sự kiện E1/E2/E3, P(TP)/P(SL), R kỳ vọng sau phí, phía, mức SL/TP,
  4. ghi MỌI nến 5m vào signal_events (source modelv3_event): decision.good = có sự kiện VÀ R kỳ vọng >= ngưỡng
     của bundle. Lệnh: vào ở giá mở nến 1m kế tiếp, SL = TP = sl_pct, giữ tối đa 4h, các lệnh độc lập.
     Cùng transaction ghi outbox (topic signals.model.v1).

Biến môi trường:
  MODEL_V3_BUNDLE         artifacts/5m_event_v3/model_v3.joblib
  MODEL_V3_SOURCE         modelv3_event
  MODEL_V3_HISTORY_DAYS   20
  MODEL_V3_POLL_SECONDS   15
  MODEL_V3_MIN_EV         (tùy chọn) ghi đè ngưỡng R kỳ vọng của bundle
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from psycopg.types.json import Jsonb

from .core import INSTRUMENT, MODEL_SIGNALS, now_ms
from .storage import connect, heartbeat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model_training'))
import modelv3 as v3  # noqa: E402

log = logging.getLogger(__name__)
M1, M5 = 60_000, 300_000
SERVICE = 'model-signals-v3'
KLINES_URL = 'https://fapi.binance.com/fapi/v1/klines'
FUNDING_URL = 'https://fapi.binance.com/fapi/v1/fundingRate'


def env(name, default, cast=str):
    value = os.getenv(name)
    return cast(default if value in (None, '') else value)


# --------------------------------------------------------------------------- #
# Dữ liệu
# --------------------------------------------------------------------------- #
def db_m1(db, start_ms: int) -> pd.DataFrame:
    rows = db.execute("""SELECT period_start_ms, data FROM observations
        WHERE instrument=%s AND timeframe='1m' AND kind='candle' AND period_start_ms >= %s
        ORDER BY period_start_ms""", (INSTRUMENT, start_ms)).fetchall()
    if not rows:
        return pd.DataFrame(columns=['t', 'o', 'h', 'l', 'c', 'v', 'n', 'tb'])
    d = pd.DataFrame([r['data'] for r in rows])
    out = pd.DataFrame({'t': [int(r['period_start_ms']) for r in rows]})
    for src, dst in (('open', 'o'), ('high', 'h'), ('low', 'l'), ('close', 'c'), ('volume', 'v'),
                     ('trade_count', 'n'), ('taker_buy_base', 'tb')):
        out[dst] = pd.to_numeric(d[src], errors='coerce') if src in d else np.nan
    return out


def rest_m1(start_ms: int, end_ms: int, get=None) -> pd.DataFrame:
    """Nến 1m đã đóng có open time trong [start_ms, end_ms)."""
    get = get or (lambda params: httpx.get(KLINES_URL, params=params, timeout=25))
    rows, cursor = [], start_ms
    while cursor < end_ms:
        r = get({'symbol': 'BTCUSDT', 'interval': '1m', 'startTime': cursor, 'endTime': end_ms - 1, 'limit': 1500})
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + M1
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame(columns=['t', 'o', 'h', 'l', 'c', 'v', 'n', 'tb'])
    a = pd.DataFrame(rows)
    out = pd.DataFrame({'t': a[0].astype('int64'), 'o': pd.to_numeric(a[1]), 'h': pd.to_numeric(a[2]),
                        'l': pd.to_numeric(a[3]), 'c': pd.to_numeric(a[4]), 'v': pd.to_numeric(a[5]),
                        'n': pd.to_numeric(a[8]), 'tb': pd.to_numeric(a[9])})
    return out[(out['t'] >= start_ms) & (out['t'] < end_ms)]


def missing_runs(t: np.ndarray, start_ms: int, end_ms: int) -> list[tuple[int, int]]:
    """Các đoạn phút bị thiếu trong [start_ms, end_ms) (t: open time đã sắp xếp)."""
    have = set(int(x) for x in t)
    runs, run_start = [], None
    for m in range(start_ms, end_ms, M1):
        if m not in have:
            run_start = m if run_start is None else run_start
        elif run_start is not None:
            runs.append((run_start, m))
            run_start = None
    if run_start is not None:
        runs.append((run_start, end_ms))
    return runs


class History:
    """Nến 1m liên tục cho cửa sổ lịch sử: DB ưu tiên, REST lấp chỗ trống (cache phần REST)."""

    def __init__(self, fetch=rest_m1):
        self.fetch = fetch
        self.cache = pd.DataFrame(columns=['t', 'o', 'h', 'l', 'c', 'v', 'n', 'tb'])

    def get(self, db_frame: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
        frame = pd.concat([db_frame, self.cache]).drop_duplicates('t', keep='first')
        frame = frame[(frame['t'] >= start_ms) & (frame['t'] < end_ms)].sort_values('t')
        runs = missing_runs(frame['t'].to_numpy(), start_ms, end_ms)
        if runs:
            got = [self.fetch(a, b) for a, b in runs]
            got = [g for g in got if len(g)]
            if got:
                new = pd.concat(got)
                self.cache = pd.concat([self.cache, new]).drop_duplicates('t', keep='last')
                self.cache = self.cache[self.cache['t'] >= start_ms - 86_400_000]
                frame = pd.concat([frame, new]).drop_duplicates('t', keep='first').sort_values('t')
        return frame.reset_index(drop=True)


class Funding:
    def __init__(self, ttl_seconds=600):
        self.ttl, self.at, self.rows = ttl_seconds, 0.0, None

    def get(self):
        if self.rows is None or time.monotonic() - self.at > self.ttl:
            try:
                r = httpx.get(FUNDING_URL, params={'symbol': 'BTCUSDT', 'limit': 200}, timeout=20)
                r.raise_for_status()
                self.rows, self.at = v3.funding_table(pd.DataFrame(r.json())), time.monotonic()
            except Exception as exc:  # noqa: BLE001 - giữ bản tốt gần nhất
                log.warning('funding fetch failed: %s', exc)
        return self.rows


# --------------------------------------------------------------------------- #
# Event
# --------------------------------------------------------------------------- #
def to_event(row: pd.Series, tp_mult: float, hold_min: int, source: str, bundle_name: str) -> dict:
    event_time = int(row['decision_ms'])
    p, sl = float(row['close']), float(row['sl_pct'])
    side = 'LONG' if int(row['side']) == 1 else 'SHORT'
    f = lambda k: None if k not in row or pd.isna(row[k]) else float(row[k])  # noqa: E731
    sig = {
        'barrier_pct': sl, 'sl_pct': sl, 'tp_pct': sl * tp_mult,
        'long_sl': p * (1 - sl), 'long_tp_1r': p * (1 + tp_mult * sl), 'long_tp_2r': None,
        'short_sl': p * (1 + sl), 'short_tp_1r': p * (1 - tp_mult * sl), 'short_tp_2r': None,
        'time_stop_bars': hold_min // 5,
        'p_tp': f('p_tp'), 'p_sl': f('p_sl'),
        'p_tp_long': f('p_tp_long'), 'p_sl_long': f('p_sl_long'), 'p_tp_short': f('p_tp_short'), 'p_sl_short': f('p_sl_short'),
        'e_long_1r': f('ev_long'), 'e_short_1r': f('ev_short'),
        'E1': bool(row['E1']), 'E2': bool(row['E2']), 'E3': bool(row['E3']),
    }
    thr = None if not np.isfinite(row['threshold']) else float(row['threshold'])
    signal_id = hashlib.sha256(f'{source}/{event_time}'.encode()).hexdigest()
    return {
        'schema_version': 3, 'signal_id': signal_id, 'instrument': INSTRUMENT, 'signal_type': 'model',
        'source': source, 'timeframe': '5m', 'event_time_ms': event_time, 'produced_at_ms': now_ms(),
        'close': p, 'signal': sig,
        'decision': {'side': side, 'good': bool(row['enter']), 'is_event': bool(row['is_event']),
                     'confidence': float(row['ev']), 'score': float(row['ev']), 'threshold': thr,
                     'rule': f'modelv3: sự kiện bất thường & R kỳ vọng sau phí >= ngưỡng; TP = {tp_mult:g} x SL, '
                             f'giữ tối đa {hold_min} phút, lệnh độc lập',
                     'execution': {'entry': 'market, nến 1m kế tiếp', 'sl_pct': sl, 'tp_pct': sl * tp_mult,
                                   'max_hold_min': hold_min}},
        'model': {'bundle': bundle_name, 'strategy': v3.STRATEGY},
    }


def closed_5m_end(m1_last_open_ms: int) -> int:
    """Mốc đóng của nến 5m cuối cùng đã đủ 5 nến 1m."""
    return (m1_last_open_ms + M1) // M5 * M5


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def model_worker_v3(once: bool = False, history: History | None = None, funding: Funding | None = None):
    bundle_path = Path(env('MODEL_V3_BUNDLE', 'artifacts/5m_event_v3/model_v3.joblib'))
    source = env('MODEL_V3_SOURCE', 'modelv3_event')
    days = env('MODEL_V3_HISTORY_DAYS', '20', float)
    poll = env('MODEL_V3_POLL_SECONDS', '15', float)
    bundle = v3.load_bundle(bundle_path)
    if os.getenv('MODEL_V3_MIN_EV'):
        bundle['threshold'] = float(os.environ['MODEL_V3_MIN_EV'])
    cfg, _ = v3.bundle_parts(bundle)
    history, funding = history or History(), funding or Funding()
    log.info('%s: bundle=%s threshold=%s tp_mult=%s', SERVICE, bundle_path, bundle['threshold'], cfg.tp_mult)
    with connect() as db:
        while True:
            try:
                last = db.execute('SELECT max(event_time_ms) AS t FROM signal_events WHERE source=%s',
                                  (source,)).fetchone()['t']
                latest = db.execute("""SELECT max(period_start_ms) AS t FROM observations
                    WHERE instrument=%s AND timeframe='1m' AND kind='candle'""", (INSTRUMENT,)).fetchone()['t']
                end = closed_5m_end(int(latest)) if latest is not None else None
                if end is None or (last is not None and end <= last):
                    heartbeat(db, SERVICE, {'state': 'idle', 'last_signal_ms': last})
                else:
                    start = end - int(days * 86_400_000)
                    m1 = history.get(db_m1(db, start), start, end)
                    fund = funding.get()
                    n_new = 288 if last is None else int(max(1, min(288, (end - last) // M5)))
                    try:
                        frame = v3.predict_live(bundle, m1, fund, last_n=n_new)
                    except ValueError as exc:
                        heartbeat(db, SERVICE, {'state': 'warming_up', 'error': str(exc), 'm1': len(m1)})
                        log.warning('%s waiting: %s', SERVICE, exc)
                        if once:
                            return
                        time.sleep(poll)
                        continue
                    created = 0
                    with db.transaction():
                        for _, row in frame.iterrows():
                            if last is not None and row['decision_ms'] <= last:
                                continue
                            ev = to_event(row, cfg.tp_mult, cfg.hold_min, source, bundle_path.name)
                            ok = db.execute("""INSERT INTO signal_events(signal_id,instrument,signal_type,source,timeframe,
                                event_time_ms,produced_at_ms,payload) VALUES(%s,%s,'model',%s,'5m',%s,%s,%s)
                                ON CONFLICT DO NOTHING RETURNING signal_id""",
                                (ev['signal_id'], INSTRUMENT, source, ev['event_time_ms'], ev['produced_at_ms'],
                                 Jsonb(ev))).fetchone()
                            if ok:
                                db.execute("""INSERT INTO outbox(topic,message_key,dedup_key,payload)
                                    VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                                    (MODEL_SIGNALS, INSTRUMENT, 'model:' + ev['signal_id'], Jsonb(ev)))
                                created += 1
                    tail = frame.iloc[-1]
                    heartbeat(db, SERVICE, {'state': 'ok', 'created': created, 'm1': len(m1),
                                            'last_bar_close_ms': int(tail['decision_ms']),
                                            'is_event': bool(tail['is_event']), 'ev': float(tail['ev'])})
                    log.info('%s created=%s events=%s enter=%s', SERVICE, created, int(frame['is_event'].sum()),
                             int(frame['enter'].sum()))
                if once:
                    return
            except Exception as exc:  # noqa: BLE001
                heartbeat(db, SERVICE, {'state': 'error', 'error': str(exc)[:300]})
                log.exception('%s cycle failed', SERVICE)
                if once:
                    raise
            time.sleep(poll)
