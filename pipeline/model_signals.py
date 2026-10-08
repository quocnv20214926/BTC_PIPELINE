"""Worker `model-signals`: chạy model v2 (model_training/modelv2.py) trên dữ liệu live.

Mỗi khi có nến 5m mới đã đóng trong bảng observations, worker:
  1. đọc MODEL_HISTORY_DAYS ngày nến 5m (mặc định 30): phần có trong observations lấy từ DB, phần cũ
     hơn (collector chỉ backfill <= 28 ngày) được bổ sung một lần từ REST klines. Với >= 30 ngày, tín
     hiệu live khớp backtest (p_up_touch lệch < 0.002, p_touch lệch ~0.001); với 16 ngày p_touch lệch ~0.01,
  2. lấy lịch sử funding đã chốt (REST Binance, ~40 ngày - cần cho z-score 30 ngày),
  3. gọi modelv2.predict_frame với bundle đã train -> xác suất, R kỳ vọng, mức SL/TP,
  4. quyết định vào lệnh bằng ĐÚNG hàm của bộ thực thi (model_training/execution_m1.live_decision):
     độ confident = R kỳ vọng sau phí max(e_long, e_short) - phí/barrier; vào lệnh khi vượt ngưỡng
     (tuyệt đối, hoặc top % của 60 ngày tín hiệu trước đó: dự báo OOS + tín hiệu live đã lưu).
     SL/TP đặt một lần, giữ tối đa 4h, các lệnh độc lập (xem execution_m1.py),
  5. ghi vào signal_events và outbox (topic signals.model.v1) trong cùng transaction.

Biến môi trường:
  MODEL_BUNDLE            artifacts/5m_xgboost_signal_v2/models_final.joblib
  MODEL_SOURCE            modelv2_5m
  MODEL_HISTORY_DAYS      30
  MODEL_POLL_SECONDS      20
  EXEC_RR                 1       (mẫu R: 1 -> TP = SL = 1 barrier; 2 -> TP = 2, SL = 1)
  EXEC_MODE               abs     (abs | rolling)
  EXEC_MIN_SCORE          0.05    (R kỳ vọng sau phí tối thiểu)
  EXEC_TOP_FRAC           0.02    (rolling: top 2% của 60 ngày trước)
  EXEC_FEE_ROUNDTRIP      0.001
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

from .core import INSTRUMENT, INTERVALS, MODEL_SIGNALS, now_ms
from .storage import connect, heartbeat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model_training'))
import modelv2 as mv2  # noqa: E402
import execution_m1 as ex  # noqa: E402

log = logging.getLogger(__name__)
STEP = INTERVALS['5m']
FUNDING_URL = 'https://fapi.binance.com/fapi/v1/fundingRate'
KLINES_URL = 'https://fapi.binance.com/fapi/v1/klines'
HIST_FIELDS = ('barrier_pct', 'e_long_1r', 'e_short_1r', 'e_long_2r', 'e_short_2r')
SIGNAL_FIELDS = ('barrier_pct', 'p_touch', 'p_up_touch', 'p_up_first', 'p_dn_first', 'p_tie', 'p_none',
                 'p_ext_long', 'p_ext_short', 'mu_timeout', 'e_long_1r', 'e_short_1r', 'e_long_2r', 'e_short_2r',
                 'best_side', 'best_rr', 'best_e', 'long_sl', 'long_tp_1r', 'long_tp_2r', 'short_sl',
                 'short_tp_1r', 'short_tp_2r', 'time_stop_bars')


def env(name, default, cast=str):
    return cast(os.getenv(name, default))


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
def load_candles(db, since_ms: int) -> pd.DataFrame:
    """Closed 5m candles from observations, in the layout modelv2 expects."""
    rows = db.execute("""SELECT period_start_ms, data FROM observations
        WHERE instrument=%s AND timeframe='5m' AND kind='candle' AND period_start_ms >= %s
        ORDER BY period_start_ms""", (INSTRUMENT, since_ms)).fetchall()
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame([r['data'] for r in rows])
    out = pd.DataFrame({'open_time_ms': [r['period_start_ms'] for r in rows]})
    for col in ('open', 'high', 'low', 'close', 'volume'):
        out[col] = pd.to_numeric(d[col])
    if 'taker_buy_base' in d:
        out['taker_buy_base_asset_volume'] = pd.to_numeric(d['taker_buy_base'])
    if 'trade_count' in d:
        out['trades'] = pd.to_numeric(d['trade_count'])
    return out


class KlineBackfill:
    """Older 5m candles from REST to complete the model history (fetched once, extended when needed)."""

    def __init__(self):
        self.frame = pd.DataFrame()

    def get(self, start_ms: int, end_ms: int) -> pd.DataFrame:
        have = self.frame
        if len(have) and have['open_time_ms'].min() <= start_ms and have['open_time_ms'].max() >= end_ms - STEP:
            return have[(have['open_time_ms'] >= start_ms) & (have['open_time_ms'] < end_ms)]
        rows, cursor = [], start_ms
        while cursor < end_ms:
            r = httpx.get(KLINES_URL, params={'symbol': 'BTCUSDT', 'interval': '5m', 'startTime': cursor,
                                              'endTime': end_ms - 1, 'limit': 1500}, timeout=25)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            cursor = int(batch[-1][0]) + STEP
            time.sleep(0.2)
        cols = ['open_time_ms', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(rows, columns=cols)[['open_time_ms', 'open', 'high', 'low', 'close', 'volume', 'trades',
                                               'taker_buy_base_asset_volume']].apply(pd.to_numeric)
        self.frame = pd.concat([have, df]).drop_duplicates('open_time_ms').sort_values('open_time_ms')
        return self.frame[(self.frame['open_time_ms'] >= start_ms) & (self.frame['open_time_ms'] < end_ms)]


def assemble_history(db_candles: pd.DataFrame, rest: KlineBackfill, start_ms: int) -> pd.DataFrame:
    """DB candles have priority; the part before the first DB candle comes from REST."""
    first_db = int(db_candles['open_time_ms'].min()) if len(db_candles) else None
    if first_db is not None and first_db <= start_ms:
        return db_candles
    older = rest.get(start_ms, first_db if first_db is not None else now_ms() // STEP * STEP)
    return pd.concat([older, db_candles]).drop_duplicates('open_time_ms', keep='last').sort_values('open_time_ms')


class FundingCache:
    """Funding history via REST (the DB only keeps ~28 days, the z-score needs 30)."""

    def __init__(self, ttl_seconds: int = 600):
        self.ttl, self.at, self.rows = ttl_seconds, 0.0, None

    def get(self) -> pd.DataFrame | None:
        if self.rows is None or time.monotonic() - self.at > self.ttl:
            try:
                r = httpx.get(FUNDING_URL, params={'symbol': 'BTCUSDT', 'limit': 120}, timeout=20)
                r.raise_for_status()
                self.rows, self.at = pd.DataFrame(r.json()), time.monotonic()
            except Exception as exc:  # noqa: BLE001 - keep the last good copy
                log.warning('funding fetch failed: %s', exc)
        return self.rows


def oos_history(bundle_path: Path) -> pd.DataFrame:
    """OOS predictions shipped with the bundle (p_touch, p_up_touch per bar) - warm start for thresholds."""
    parts = []
    for f in sorted(bundle_path.parent.glob('predictions_*.npz')):
        z = np.load(f)
        parts.append(pd.DataFrame({'decision_ms': z['timestamp_ns'] // 1_000_000 + STEP,
                                   **{k: z[k] for k in HIST_FIELDS}}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['decision_ms', *HIST_FIELDS])


def stored_history(db, source: str, since_ms: int) -> pd.DataFrame:
    cols = ', '.join(f"(payload->'signal'->>'{k}')::float AS {k}" for k in HIST_FIELDS)
    rows = db.execute(f"""SELECT event_time_ms AS decision_ms, {cols}
        FROM signal_events WHERE source=%s AND event_time_ms >= %s""", (source, since_ms)).fetchall()
    return pd.DataFrame(rows, columns=['decision_ms', *HIST_FIELDS])


# --------------------------------------------------------------------------- #
# Signal logic (pure functions, unit-testable)
# --------------------------------------------------------------------------- #
def exec_config() -> ex.ExecConfig:
    return ex.ExecConfig(rr=env('EXEC_RR', '1', int), tp_mult=env('EXEC_RR', '1', float), sl_mult=1.0,
                         max_hold_min=240, mode=env('EXEC_MODE', 'abs'), min_score=env('EXEC_MIN_SCORE', '0.05', float),
                         top_frac=env('EXEC_TOP_FRAC', '0.02', float),
                         fee_roundtrip=env('EXEC_FEE_ROUNDTRIP', '0.001', float))


def decide(history: pd.DataFrame, row: pd.Series, cfg: ex.ExecConfig) -> dict:
    """Quyết định của bộ thực thi cho một nến (history: chỉ các nến TRƯỚC nến này)."""
    r = {k: row[k] for k in HIST_FIELDS}
    r['decision_ms'] = int(pd.Timestamp(row['timestamp']).value // 1_000_000) + STEP
    return ex.live_decision(history, r, cfg)


def to_event(row: pd.Series, dec: dict, cfg: ex.ExecConfig, source: str, bundle_name: str) -> dict:
    """signal_events / Kafka payload of one decision bar."""
    event_time = int(pd.Timestamp(row['timestamp']).value // 1_000_000) + STEP
    sig = {k: (float(row[k]) if k in row and pd.notna(row[k]) else None) for k in SIGNAL_FIELDS}
    signal_id = hashlib.sha256(f'{source}/{event_time}'.encode()).hexdigest()
    return {
        'schema_version': 2, 'signal_id': signal_id, 'instrument': INSTRUMENT, 'signal_type': 'model',
        'source': source, 'timeframe': '5m', 'event_time_ms': event_time, 'produced_at_ms': now_ms(),
        'close': float(row['close']),
        'signal': sig,
        'decision': {'side': 'LONG' if dec['side'] == 1 else 'SHORT', 'good': bool(dec['enter']),
                     'confidence': float(dec['score']), 'score': float(dec['score']),
                     'threshold': dec.get('threshold'), 'reason': dec.get('reason'),
                     'rule': f'execution_m1: {cfg.label}, max hold {cfg.max_hold_min} min, independent trades',
                     'execution': {'rr': cfg.rr, 'tp_mult': cfg.tp_mult, 'sl_mult': cfg.sl_mult,
                                   'max_hold_min': cfg.max_hold_min}},
        'model': {'bundle': bundle_name, 'strategy': mv2.STRATEGY},
    }


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def model_worker(once: bool = False):
    bundle_path = Path(env('MODEL_BUNDLE', 'artifacts/5m_xgboost_signal_v2/models_final.joblib'))
    source = env('MODEL_SOURCE', 'modelv2_5m')
    history_days = env('MODEL_HISTORY_DAYS', '30', float)
    poll = env('MODEL_POLL_SECONDS', '20', float)
    cfg = exec_config()
    thr_days = cfg.window_days

    bundle = mv2.load_bundle(bundle_path)
    needs_funding = any(n.startswith('fund_') for n in bundle['dir_names'])
    oos = oos_history(bundle_path)
    funding = FundingCache()
    rest = KlineBackfill()
    log.info('model-signals: bundle=%s funding=%s oos_rows=%s rule=%s', bundle_path, needs_funding, len(oos), cfg.label)

    with connect() as db:
        while True:
            try:
                last = db.execute('SELECT max(event_time_ms) AS t FROM signal_events WHERE source=%s',
                                  (source,)).fetchone()['t']
                latest = db.execute("""SELECT max(period_start_ms) AS t FROM observations
                    WHERE instrument=%s AND timeframe='5m' AND kind='candle'""", (INSTRUMENT,)).fetchone()['t']
                if latest is None or (last is not None and latest + STEP <= last):
                    heartbeat(db, 'model-signals', {'state': 'idle', 'last_signal_ms': last})
                else:
                    start = latest - int(history_days * 86_400_000)
                    candles = assemble_history(load_candles(db, start), rest, start)
                    fund = funding.get() if needs_funding else None
                    if needs_funding and fund is None:
                        raise RuntimeError('funding history unavailable')
                    n_new = 288 if last is None else max(1, min(288, (latest + STEP - last) // STEP))
                    try:
                        frame = mv2.predict_frame(bundle, candles, last_n=int(n_new), funding=fund)
                    except ValueError as exc:     # not enough history yet (warm-up) or a data gap
                        heartbeat(db, 'model-signals', {'state': 'warming_up', 'error': str(exc),
                                                        'candles': len(candles)})
                        log.warning('model-signals waiting: %s (candles=%s)', exc, len(candles))
                        if once:
                            return
                        time.sleep(poll)
                        continue
                    hist = pd.concat([oos, stored_history(db, source, latest - int(thr_days * 86_400_000))])
                    created = 0
                    with db.transaction():
                        for _, row in frame.iterrows():
                            ev_time = int(pd.Timestamp(row['timestamp']).value // 1_000_000) + STEP
                            if last is not None and ev_time <= last:
                                continue
                            ev = to_event(row, decide(hist, row, cfg), cfg, source, bundle_path.name)
                            ok = db.execute("""INSERT INTO signal_events(signal_id,instrument,signal_type,source,timeframe,
                                event_time_ms,produced_at_ms,payload) VALUES(%s,%s,'model',%s,'5m',%s,%s,%s)
                                ON CONFLICT DO NOTHING RETURNING signal_id""",
                                (ev['signal_id'], INSTRUMENT, source, ev_time, ev['produced_at_ms'], Jsonb(ev))).fetchone()
                            if ok:
                                db.execute("""INSERT INTO outbox(topic,message_key,dedup_key,payload)
                                    VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                                    (MODEL_SIGNALS, INSTRUMENT, 'model:' + ev['signal_id'], Jsonb(ev)))
                                created += 1
                                hist = pd.concat([hist, pd.DataFrame([{'decision_ms': ev_time,
                                    **{k: ev['signal'][k] for k in HIST_FIELDS}}])])
                    last_ev = frame.iloc[-1]
                    heartbeat(db, 'model-signals', {'state': 'ok', 'created': created, 'candles': len(candles),
                                                    'last_bar': str(last_ev['timestamp']),
                                                    'p_touch': float(last_ev['p_touch']),
                                                    'p_up_touch': float(last_ev['p_up_touch'])})
                    log.info('model-signals created=%s last_bar=%s', created, last_ev['timestamp'])
                if once:
                    return
            except Exception as exc:  # noqa: BLE001
                heartbeat(db, 'model-signals', {'state': 'error', 'error': str(exc)})
                log.exception('model-signals cycle failed')
                if once:
                    raise
            time.sleep(poll)
