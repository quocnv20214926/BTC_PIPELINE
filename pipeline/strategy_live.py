"""Worker `strategy-h1`: chạy chiến lược H1 (touch + hướng B, model_training/strategy_h1.py) trên dữ liệu live và
quản lý lệnh paper bằng bộ thực thi pipeline/strategy_exec.py (khớp backtest_m5).

Mỗi giờ H (nến 1h [H-1h, H) vừa đóng):
  1. Chờ đủ dữ liệu: 12 nến 5m của giờ đó + dòng vị thế của nến 5m cuối giờ (oi, top_acct_ratio, top_ratio,
     ratio, premium). Quá STRATEGY_MAX_WAIT_S giây thì vẫn tính (bảng vị thế giữ giá trị cũ <= 15 phút, như train).
  2. Dựng dữ liệu đúng như train:
       - nến 5m (DB, phần thiếu lấy REST) -> strategy_h1.aggregate_1h -> nến 1h (STRATEGY_HISTORY_DAYS ngày;
         >= 180 ngày cho xác suất trùng khớp backtest, kiểm định trong historical_tests/test_strategy_live.py)
       - bảng vị thế 5m bằng model_training/positioning_data.align (cùng hàm dựng bảng train)
       - funding đã chốt (REST /fapi/v1/fundingRate, 200 kỳ gần nhất)
  3. strategy_h1.predict_live -> p_touch, p_up, độ tự tin, tín hiệu, hướng, b.
  4. Ghi signal_events (source strategy_h1, timeframe 1h) + outbox (signals.model.v1) trong cùng transaction.
     Có tín hiệu -> bộ thực thi nhận / bỏ (tối đa 5 lệnh, cách >= 1h) và tạo lệnh PENDING trong strategy_trades.
Mỗi vòng lặp: khớp giá vào cho lệnh PENDING (open nến 1m đầu tiên sau lúc ra quyết định) và theo dõi lệnh OPEN:
TP / SL trên nến 1m (thiếu nến 1m thì dùng nến 5m), thoát nhanh trên nến 5m, hết hạn sau max_hold_bars giờ.

Hệ thống KHÔNG đặt lệnh thật: mọi lệnh là paper (forward test).

Biến môi trường:
  STRATEGY_BUNDLE           artifacts/1h_strategy_v1/models_final.joblib
  STRATEGY_HISTORY_DAYS     200
  STRATEGY_MAX_WAIT_S       600     chờ dữ liệu vị thế tối đa sau khi giờ đóng
  STRATEGY_POLL_SECONDS     15
  STRATEGY_CATCHUP_HOURS    6       khi khởi động lại: tính bù tín hiệu cho tối đa N giờ đã qua
  STRATEGY_MAX_ENTRY_DELAY_MIN 10   tín hiệu ra quyết định trễ hơn H + N phút thì không vào lệnh (ghi lý do 'late')
  STRATEGY_ENTRY            next_1m (next_1m | bar_open)
  STRATEGY_START_EQUITY     1.0
  POSITIONING_TS_SHIFT_MS   0       create_time (data.binance.vision) = timestamp REST + giá trị này
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from psycopg.types.json import Jsonb

from .core import INSTRUMENT, INTERVALS, MODEL_SIGNALS, STRATEGY_SOURCE, now_ms
from .storage import connect, heartbeat
from . import strategy_exec as sx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model_training'))
import positioning_data as pdm  # noqa: E402
import strategy_h1 as st  # noqa: E402

log = logging.getLogger(__name__)
M1, M5, H1 = INTERVALS['1m'], INTERVALS['5m'], INTERVALS['1h']
REST = 'https://fapi.binance.com'
POS_KINDS = {'oi': 'oi', 'top_acct_ratio': 'top_acct', 'top_ratio': 'top_pos', 'ratio': 'acct'}


def env(name, default, cast=str):
    v = os.getenv(name)
    return cast(default if v in (None, '') else v)


class Settings:
    def __init__(self):
        self.bundle_path = Path(env('STRATEGY_BUNDLE', 'artifacts/1h_strategy_v1/models_final.joblib'))
        self.source = env('STRATEGY_SOURCE', STRATEGY_SOURCE)
        self.history_days = env('STRATEGY_HISTORY_DAYS', '200', float)
        self.max_wait_ms = int(env('STRATEGY_MAX_WAIT_S', '600', float) * 1000)
        self.poll = env('STRATEGY_POLL_SECONDS', '15', float)
        self.catchup_h = env('STRATEGY_CATCHUP_HOURS', '6', int)
        self.max_entry_delay_ms = int(env('STRATEGY_MAX_ENTRY_DELAY_MIN', '10', float) * 60_000)
        self.entry = env('STRATEGY_ENTRY', 'next_1m')
        self.start_equity = env('STRATEGY_START_EQUITY', '1.0', float)
        self.ts_shift_ms = env('POSITIONING_TS_SHIFT_MS', '0', int)
        if self.entry not in ('next_1m', 'bar_open'):
            raise ValueError('STRATEGY_ENTRY must be next_1m or bar_open')


# --------------------------------------------------------------------------- #
# Dữ liệu từ DB / REST
# --------------------------------------------------------------------------- #
def db_candles(db, tf: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Nến đã đóng, open time trong [start_ms, end_ms)."""
    rows = db.execute("""SELECT period_start_ms, data FROM observations WHERE instrument=%s AND timeframe=%s
        AND kind='candle' AND period_start_ms >= %s AND period_start_ms < %s ORDER BY period_start_ms""",
                      (INSTRUMENT, tf, start_ms, end_ms)).fetchall()
    return candles_frame([r['period_start_ms'] for r in rows], [r['data'] for r in rows])


def candles_frame(times, datas) -> pd.DataFrame:
    cols = ['open_time_ms', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume', 'trades']
    if not len(times):
        return pd.DataFrame(columns=cols)
    d = pd.DataFrame(list(datas))
    out = pd.DataFrame({'open_time_ms': np.asarray(times, np.int64)})
    for c in ('open', 'high', 'low', 'close', 'volume'):
        out[c] = pd.to_numeric(d[c]).to_numpy()
    out['taker_buy_base_asset_volume'] = pd.to_numeric(d.get('taker_buy_base')).to_numpy()
    out['trades'] = pd.to_numeric(d.get('trade_count')).to_numpy()
    return out[cols]


def rest_get(path: str, params: dict, client: httpx.Client | None = None):
    c = client or httpx
    for attempt in range(6):
        try:
            r = c.get(REST + path, params=params, timeout=25)
            if r.status_code in (418, 429) or r.status_code >= 500:
                time.sleep(min(float(r.headers.get('Retry-After', 2 ** attempt)), 120))
                continue
            r.raise_for_status()
            return r.json()
        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f'Binance retry budget exceeded: {path}')


class KlineHistory:
    """Nến 5m cũ hơn phần DB đang giữ (collector chỉ backfill <= 28 ngày) lấy từ REST, giữ trong bộ nhớ."""

    def __init__(self):
        self.frame = pd.DataFrame(columns=['open_time_ms'])

    def fill(self, start_ms: int, end_ms: int) -> pd.DataFrame:
        have = set(self.frame['open_time_ms'].astype('int64')) if len(self.frame) else set()
        need = [t for t in range(start_ms, end_ms, M5) if t not in have]
        if need:
            rows, cursor, last = [], need[0], need[-1] + M5
            while cursor < last:
                batch = rest_get('/fapi/v1/klines', {'symbol': 'BTCUSDT', 'interval': '5m', 'startTime': cursor,
                                                     'endTime': last - 1, 'limit': 1500})
                if not batch:
                    break
                rows.extend(batch)
                cursor = int(batch[-1][0]) + M5
                time.sleep(0.2)
            if rows:
                df = pd.DataFrame({'open_time_ms': [int(r[0]) for r in rows],
                                   **{k: [float(r[i]) for r in rows] for k, i in
                                      (('open', 1), ('high', 2), ('low', 3), ('close', 4), ('volume', 5),
                                       ('trades', 8), ('taker_buy_base_asset_volume', 9))}})
                df = df[df['open_time_ms'] + M5 <= now_ms()]          # chỉ nến đã đóng
                self.frame = (pd.concat([self.frame, df]).drop_duplicates('open_time_ms', keep='last')
                              .sort_values('open_time_ms').reset_index(drop=True))
        f = self.frame
        return f[(f['open_time_ms'] >= start_ms) & (f['open_time_ms'] < end_ms)]


def history_5m(db, rest: KlineHistory, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Nến 5m liên tục cho [start_ms, end_ms): DB ưu tiên, khoảng thiếu lấy REST."""
    dbf = db_candles(db, '5m', start_ms, end_ms)
    have = set(dbf['open_time_ms'].astype('int64'))
    missing = [t for t in range(start_ms, end_ms, M5) if t not in have]
    if missing:
        older = rest.fill(missing[0], missing[-1] + M5)
        older = older[~older['open_time_ms'].isin(have)]
        dbf = pd.concat([older, dbf])
    cols = ['open_time_ms', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume', 'trades']
    return dbf[cols].drop_duplicates('open_time_ms', keep='last').sort_values('open_time_ms').reset_index(drop=True)


def positioning_frame(db, grid_open_ms: np.ndarray, start_ms: int, end_ms: int, ts_shift_ms: int,
                      ref_close: pd.Series | None = None) -> pd.DataFrame:
    """Bảng vị thế 5m giống BTCUSDT_positioning_5m.csv, dựng từ observations (REST) bằng positioning_data.align."""
    rows = db.execute("""SELECT kind, data FROM observations WHERE instrument=%s AND timeframe='5m'
        AND kind IN ('oi','top_acct_ratio','top_ratio','ratio') AND period_start_ms >= %s AND period_start_ms < %s""",
                      (INSTRUMENT, start_ms - 2 * M5, end_ms + 2 * M5)).fetchall()
    by = {k: [] for k in POS_KINDS}
    for r in rows:
        d = dict(r['data'])
        d['timestamp'] = int(d['source_timestamp_ms'])
        by[r['kind']].append(d)
    metrics = pdm.rest_metrics(by['oi'], by['top_acct_ratio'], by['top_ratio'], by['ratio'], ts_shift_ms)
    prem = db.execute("""SELECT period_start_ms, data FROM observations WHERE instrument=%s AND timeframe='5m'
        AND kind='premium' AND period_start_ms >= %s AND period_start_ms < %s ORDER BY period_start_ms""",
                      (INSTRUMENT, start_ms, end_ms)).fetchall()
    premium = pdm.rest_premium([[r['period_start_ms'], r['data']['open'], r['data']['high'], r['data']['low'],
                                 r['data']['close']] for r in prem])
    return pdm.align(metrics, premium, grid_open_ms, ref_close=ref_close)


class FundingCache:
    def __init__(self, ttl_s: int = 600):
        self.ttl, self.at, self.frame = ttl_s, 0.0, None

    def get(self, db=None) -> pd.DataFrame:
        if self.frame is None or time.monotonic() - self.at > self.ttl:
            try:
                rows = rest_get('/fapi/v1/fundingRate', {'symbol': 'BTCUSDT', 'limit': 200})
                self.frame = pd.DataFrame({'funding_time_ms': [int(r['fundingTime']) for r in rows],
                                           'funding_rate': [float(r['fundingRate']) for r in rows]})
                self.at = time.monotonic()
            except Exception as exc:  # noqa: BLE001 - dùng bản cũ / DB
                log.warning('funding REST failed: %s', exc)
                if self.frame is None and db is not None:
                    self.frame = funding_from_db(db)
        return self.frame


def funding_from_db(db) -> pd.DataFrame:
    rows = db.execute("""SELECT data FROM observations WHERE instrument=%s AND kind='funding'
        ORDER BY period_start_ms""", (INSTRUMENT,)).fetchall()
    return pd.DataFrame({'funding_time_ms': [int(r['data']['source_timestamp_ms']) for r in rows],
                         'funding_rate': [float(r['data']['fundingRate']) for r in rows]})


# --------------------------------------------------------------------------- #
# Tín hiệu
# --------------------------------------------------------------------------- #
def hour_readiness(db, H: int, ts_shift_ms: int) -> dict:
    """Đủ 12 nến 5m của giờ [H-1h, H) và dòng vị thế của nến 5m cuối giờ (open H-5m) chưa?"""
    n5 = db.execute("""SELECT count(*) AS n FROM observations WHERE instrument=%s AND timeframe='5m' AND kind='candle'
        AND period_start_ms >= %s AND period_start_ms < %s""", (INSTRUMENT, H - H1, H)).fetchone()['n']
    last = H - M5
    want_ts = last - ts_shift_ms                       # timestamp REST tương ứng với nến 5m cuối giờ
    got = {}
    for kind in POS_KINDS:
        row = db.execute("""SELECT 1 FROM observations WHERE instrument=%s AND timeframe='5m' AND kind=%s
            AND (data->>'source_timestamp_ms')::bigint = %s LIMIT 1""", (INSTRUMENT, kind, want_ts)).fetchone()
        got[kind] = row is not None
    got['premium'] = db.execute("""SELECT 1 FROM observations WHERE instrument=%s AND timeframe='5m' AND kind='premium'
        AND period_start_ms=%s""", (INSTRUMENT, last)).fetchone() is not None
    return {'candles_5m': int(n5), 'positioning': got, 'ready': n5 == 12 and all(got.values())}


def compute_signal(bundle: dict, candles_5m: pd.DataFrame, positioning: pd.DataFrame, funding: pd.DataFrame,
                   H: int) -> pd.Series:
    """Đầu ra model cho nến 1h [H-1h, H). candles_5m: nến 5m kết thúc ở H."""
    h1 = st.aggregate_1h(candles_5m)
    if not len(h1) or int(pd.Timestamp(h1['timestamp'].iloc[-1]).value // 1_000_000) != H - H1:
        raise ValueError(f'nến 1h {pd.to_datetime(H - H1, unit="ms", utc=True)} chưa đủ 12 nến 5m')
    P = st.predict_live(bundle, h1, positioning, funding, last_n=1)
    row = P.iloc[-1]
    if int(pd.Timestamp(row['timestamp']).value // 1_000_000) != H - H1:
        raise ValueError('model không có dự báo cho giờ vừa đóng (thiếu lịch sử / gap)')
    return row


def signal_event(row: pd.Series, H: int, bundle: dict, settings: Settings, quality: dict) -> dict:
    cfg = bundle['config']
    side = int(row['side'])
    b = float(row['b'])
    close = float(row['close'])
    sid = hashlib.sha256(f'{settings.source}/{H}'.encode()).hexdigest()
    return {
        'schema_version': 3, 'signal_id': sid, 'instrument': INSTRUMENT, 'signal_type': 'model',
        'source': settings.source, 'timeframe': '1h', 'event_time_ms': H, 'hour_open_ms': H - H1,
        'produced_at_ms': now_ms(), 'close': close,
        'signal': {'p_touch': float(row['p_touch']), 'p_up': float(row['p_up']), 'conf': float(row['conf']),
                   'b_pct': b, 'thr_touch': float(bundle['thr_touch']), 'thr_conf': float(bundle['thr_conf']),
                   'touch_ok': bool(row['p_touch'] >= bundle['thr_touch']),
                   'conf_ok': bool(row['conf'] >= bundle['thr_conf'])},
        'decision': {'good': bool(row['signal']), 'side': 'LONG' if side > 0 else 'SHORT',
                     'tp_pct': cfg.tp_mult * b, 'sl_pct': cfg.sl_mult * b,
                     'tp_ref': close * math.exp(side * cfg.tp_mult * b),
                     'sl_ref': close * math.exp(-side * cfg.sl_mult * b),
                     'rule': (f'touch top {cfg.touch_top:.0%} + B top {cfg.dir_top:.0%}; TP {cfg.tp_mult}b, SL '
                              f'{cfg.sl_mult}b, thoát nhanh nến 5m ngược chiều >= {cfg.anomaly_range:.2%}, '
                              f'tối đa {cfg.max_positions} lệnh, giữ tối đa {cfg.max_hold_bars}h')},
        'data_quality': quality,
        'model': {'bundle': settings.bundle_path.name, 'strategy': bundle.get('strategy'),
                  'trained_fold': bundle.get('trained_fold')},
    }


# --------------------------------------------------------------------------- #
# Lệnh paper (DB <-> strategy_exec)
# --------------------------------------------------------------------------- #
TRADE_COLS = ('trade_id', 'signal_id', 'hour_open_ms', 'side', 'status', 'entry_ms', 'entry', 'entry_ref', 'tp', 'sl',
              'risk_frac', 'exit_ms', 'exit_close_ms', 'exit_price', 'r_gross', 'r_net', 'equity_after', 'last_bar_ms')


def load_trades(db, source: str, since_hour_ms: int | None = None) -> list[sx.Trade]:
    """Lệnh đang mở; với since_hour_ms: thêm các lệnh đã thoát ở nến mở lúc >= since_hour_ms (luật danh mục)."""
    if since_hour_ms is None:
        q, args = """SELECT * FROM strategy_trades WHERE source=%s AND status IN ('PENDING','OPEN')
                     ORDER BY hour_open_ms""", (source,)
    else:
        q, args = """SELECT * FROM strategy_trades WHERE source=%s AND (status IN ('PENDING','OPEN')
                     OR (status <> 'CANCELLED' AND exit_ms >= %s)) ORDER BY hour_open_ms""", (source, since_hour_ms)
    rows = db.execute(q, args).fetchall()
    out = []
    for r in rows:
        t = sx.Trade(trade_id=r['trade_id'], signal_id=r['signal_id'], hour_open_ms=int(r['hour_open_ms']),
                     side=int(r['side']), b=float(r['b_pct']))
        for k in TRADE_COLS[4:]:
            setattr(t, k, r[k])
        t.extra = dict(r['state'] or {})
        out.append(t)
    return out


def save_trade(db, t: sx.Trade, source: str, decision_ms: int | None = None, params: dict | None = None):
    vals = {k: getattr(t, k) for k in TRADE_COLS}
    db.execute("""INSERT INTO strategy_trades(trade_id,source,signal_id,hour_open_ms,decision_ms,side,b_pct,status,
            entry_ms,entry,entry_ref,tp,sl,risk_frac,exit_ms,exit_close_ms,exit_price,r_gross,r_net,equity_after,
            last_bar_ms,state,params)
        VALUES(%(trade_id)s,%(source)s,%(signal_id)s,%(hour_open_ms)s,%(decision_ms)s,%(side)s,%(b)s,%(status)s,
            %(entry_ms)s,%(entry)s,%(entry_ref)s,%(tp)s,%(sl)s,%(risk_frac)s,%(exit_ms)s,%(exit_close_ms)s,
            %(exit_price)s,%(r_gross)s,%(r_net)s,%(equity_after)s,%(last_bar_ms)s,%(state)s,%(params)s)
        ON CONFLICT (trade_id) DO UPDATE SET status=excluded.status, entry_ms=excluded.entry_ms, entry=excluded.entry,
            entry_ref=excluded.entry_ref, tp=excluded.tp, sl=excluded.sl, risk_frac=excluded.risk_frac,
            exit_ms=excluded.exit_ms, exit_close_ms=excluded.exit_close_ms, exit_price=excluded.exit_price,
            r_gross=excluded.r_gross, r_net=excluded.r_net, equity_after=excluded.equity_after,
            last_bar_ms=excluded.last_bar_ms, state=excluded.state, updated_at=now()""",
               {**vals, 'source': source, 'decision_ms': decision_ms if decision_ms is not None else now_ms(),
                'b': t.b, 'state': Jsonb(t.extra), 'params': Jsonb(params or {})})


def get_state(db, source: str, start_equity: float) -> dict:
    row = db.execute('SELECT * FROM strategy_state WHERE source=%s', (source,)).fetchone()
    if row is None:
        db.execute('INSERT INTO strategy_state(source,equity) VALUES(%s,%s) ON CONFLICT DO NOTHING',
                   (source, start_equity))
        return {'equity': start_equity, 'last_entry_hour_ms': None}
    return {'equity': float(row['equity']), 'last_entry_hour_ms': row['last_entry_hour_ms']}


def set_state(db, source: str, equity: float, last_entry_hour_ms: int | None):
    db.execute("""UPDATE strategy_state SET equity=%s, last_entry_hour_ms=%s, updated_at=now() WHERE source=%s""",
               (equity, last_entry_hour_ms, source))


def bars_for(db, start_ms: int, end_ms: int, rest: KlineHistory | None = None) -> list[sx.Bar]:
    """Nến để theo dõi lệnh trong [start_ms, end_ms): mỗi khoảng 5m đã đóng -> 5 nến 1m (TP/SL) + nến 5m (thoát
    nhanh); thiếu nến 1m -> nến 5m dùng cho cả TP/SL (bảo thủ). Nến 5m thiếu trong DB lấy từ REST; chỉ xử lý
    đến khoảng 5m liên tục cuối cùng (không nhảy qua chỗ trống)."""
    start5 = start_ms - start_ms % M5
    end5 = min(end_ms, now_ms()) // M5 * M5 - M5            # open time của nến 5m đã đóng gần nhất
    f5 = db_candles(db, '5m', start5, end5 + M5)
    if rest is not None and end5 >= start5:
        have = set(f5['open_time_ms'].astype('int64'))
        if any(t not in have for t in range(start5, end5 + M5, M5)):
            extra = rest.fill(start5, end5 + M5)
            f5 = (pd.concat([f5, extra[~extra['open_time_ms'].isin(have)]]).sort_values('open_time_ms')
                  .reset_index(drop=True))
    times = f5['open_time_ms'].astype('int64').to_numpy()
    if len(times):
        expected = start5 + M5 * np.arange(len(times))
        cut = np.flatnonzero(times != expected)
        if len(cut):
            f5 = f5.iloc[:cut[0]]                          # dừng ở chỗ trống đầu tiên
    f1 = db_candles(db, '1m', start5, end5 + M5)
    one = {int(r.open_time_ms): r for r in f1.itertuples()}
    bars = []
    for r in f5.itertuples():
        t5 = int(r.open_time_ms)
        mins = [one.get(t5 + i * M1) for i in range(5)]
        if all(m is not None for m in mins):
            bars += [sx.Bar(int(m.open_time_ms), int(m.open_time_ms) + M1, m.open, m.high, m.low, m.close, True, False)
                     for m in mins]
            bars.append(sx.Bar(t5, t5 + M5, r.open, r.high, r.low, r.close, False, True))
        else:
            bars.append(sx.Bar(t5, t5 + M5, r.open, r.high, r.low, r.close, True, True))
    bars.sort(key=lambda b: (b.close_ms, not b.levels))
    return bars


class Strategy:
    def __init__(self, db, settings: Settings):
        self.db, self.s = db, settings
        self.bundle = st.load_bundle(settings.bundle_path)
        self.cfg = self.bundle['config']
        self.p = sx.ExecParams.from_strategy(self.cfg)
        self.rest = KlineHistory()
        self.funding = FundingCache()
        self.params = {k: getattr(self.p, k) for k in self.p.__dataclass_fields__}

    # -- tín hiệu mỗi giờ ---------------------------------------------------- #
    def pending_hours(self) -> list[int]:
        now = now_ms()
        last_closed = now // H1 * H1
        row = self.db.execute('SELECT max(event_time_ms) AS t FROM signal_events WHERE source=%s',
                              (self.s.source,)).fetchone()
        first = last_closed - (self.s.catchup_h - 1) * H1
        if row['t'] is not None:
            first = max(first, int(row['t']) + H1)
        return list(range(first, last_closed + 1, H1))

    def run_hour(self, H: int) -> dict | None:
        ready = hour_readiness(self.db, H, self.s.ts_shift_ms)
        late = now_ms() - H
        if not ready['ready'] and late < self.s.max_wait_ms:
            return None                                        # chờ dữ liệu
        if ready['candles_5m'] < 12:
            if late < 6 * H1:
                return None                                    # nến 5m chưa về (collector trễ) -> chờ tiếp
        start = H - int(self.s.history_days * 86_400_000)
        start -= start % M5
        c5 = history_5m(self.db, self.rest, start, H)
        grid = c5['open_time_ms'].to_numpy(np.int64)
        pos_start = H - 20 * 86_400_000
        pos = positioning_frame(self.db, grid[grid >= pos_start], pos_start, H, self.s.ts_shift_ms,
                                ref_close=c5.set_index('open_time_ms')['close'])
        try:
            row = compute_signal(self.bundle, c5, pos, self.funding.get(self.db), H)
        except ValueError as exc:
            if late < 6 * H1:
                raise
            return self.record_missing(H, str(exc))         # giờ không tính được (thiếu dữ liệu lâu) -> bỏ qua
        last_pos = pos[pos['open_time_ms'] == H - M5]
        quality = {'ready_on_time': ready['ready'], 'positioning_found': ready['positioning'],
                   'decided_after_s': round(late / 1000, 1),
                   'pos_age_bars': (None if last_pos.empty or pd.isna(last_pos['m_age_bars'].iloc[0])
                                    else float(last_pos['m_age_bars'].iloc[0]))}
        ev = signal_event(row, H, self.bundle, self.s, quality)
        decision = ev['decision']
        with self.db.transaction():
            trade = None
            if decision['good']:
                decision['admitted'], decision['skip_reason'] = self.admit(ev, H, late)
            ok = self.db.execute("""INSERT INTO signal_events(signal_id,instrument,signal_type,source,timeframe,
                event_time_ms,produced_at_ms,payload) VALUES(%s,%s,'model',%s,'1h',%s,%s,%s)
                ON CONFLICT DO NOTHING RETURNING signal_id""",
                (ev['signal_id'], INSTRUMENT, self.s.source, H, ev['produced_at_ms'], Jsonb(ev))).fetchone()
            if ok:
                self.db.execute("""INSERT INTO outbox(topic,message_key,dedup_key,payload) VALUES(%s,%s,%s,%s)
                    ON CONFLICT DO NOTHING""", (MODEL_SIGNALS, INSTRUMENT, 'strategy:' + ev['signal_id'], Jsonb(ev)))
                if decision['good'] and decision['admitted']:
                    trade = sx.Trade(trade_id=ev['signal_id'], signal_id=ev['signal_id'], hour_open_ms=H - H1,
                                     side=1 if decision['side'] == 'LONG' else -1, b=ev['signal']['b_pct'])
                    trade.extra['decided_ms'] = now_ms()
                    save_trade(self.db, trade, self.s.source, now_ms(), self.params)
                    stt = get_state(self.db, self.s.source, self.s.start_equity)
                    set_state(self.db, self.s.source, stt['equity'], H - H1)
        log.info('strategy-h1 %s p_touch=%.3f p_up=%.3f signal=%s %s', pd.to_datetime(H, unit='ms', utc=True),
                 ev['signal']['p_touch'], ev['signal']['p_up'], decision['good'],
                 decision.get('skip_reason') or '')
        return ev

    def record_missing(self, H: int, reason: str) -> dict:
        sid = hashlib.sha256(f'{self.s.source}/{H}'.encode()).hexdigest()
        ev = {'schema_version': 3, 'signal_id': sid, 'instrument': INSTRUMENT, 'signal_type': 'model',
              'source': self.s.source, 'timeframe': '1h', 'event_time_ms': H, 'hour_open_ms': H - H1,
              'produced_at_ms': now_ms(), 'close': None, 'signal': None,
              'decision': {'good': False, 'side': None, 'admitted': False, 'skip_reason': 'no_data', 'error': reason}}
        self.db.execute("""INSERT INTO signal_events(signal_id,instrument,signal_type,source,timeframe,event_time_ms,
            produced_at_ms,payload) VALUES(%s,%s,'model',%s,'1h',%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (sid, INSTRUMENT, self.s.source, H, ev['produced_at_ms'], Jsonb(ev)))
        log.warning('strategy-h1 %s skipped: %s', pd.to_datetime(H, unit='ms', utc=True), reason)
        return ev

    def admit(self, ev: dict, H: int, late_ms: int) -> tuple[bool, str | None]:
        if late_ms > self.s.max_entry_delay_ms:
            return False, 'late'
        stt = get_state(self.db, self.s.source, self.s.start_equity)
        book = sx.Book(self.p, stt['equity'], load_trades(self.db, self.s.source, H - H1), stt['last_entry_hour_ms'])
        ok, why = book.admit(H - H1)
        return ok, (None if ok else why)

    # -- theo dõi lệnh ------------------------------------------------------- #
    def manage(self) -> dict:
        stt = get_state(self.db, self.s.source, self.s.start_equity)
        trades = [t for t in load_trades(self.db, self.s.source) if t.is_open]
        if not trades:
            return {'open': 0, 'filled': 0, 'closed': 0, 'equity': stt['equity']}
        book = sx.Book(self.p, stt['equity'], trades, stt['last_entry_hour_ms'])
        filled = closed = 0
        now = now_ms()
        for t in sorted(trades, key=lambda x: x.hour_open_ms):
            if t.status != 'PENDING':
                continue
            H = t.hour_open_ms + H1
            if self.s.entry == 'bar_open':
                f = db_candles(self.db, '5m', H, H + M5)
                if len(f):
                    book.fill(t, H, float(f['open'].iloc[0]), entry_ref=float(f['open'].iloc[0]))
            else:
                after = int(t.extra.get('decided_ms', now))
                first_min = after - after % M1 + (M1 if after % M1 else 0)
                f = db_candles(self.db, '1m', first_min, first_min + M1)
                ref = db_candles(self.db, '5m', H, H + M5)
                ref1 = db_candles(self.db, '1m', H, H + M1)
                entry_ref = (float(ref['open'].iloc[0]) if len(ref) else
                             float(ref1['open'].iloc[0]) if len(ref1) else None)
                if len(f):
                    book.fill(t, first_min, float(f['open'].iloc[0]), entry_ref=entry_ref)
                elif now - first_min > 30 * M1:                # không có nến 1m sau 30 phút -> huỷ
                    t.status = 'CANCELLED'
                    t.extra['cancel_reason'] = 'no 1m candle after decision'
            if t.status in ('OPEN', 'CANCELLED'):
                save_trade(self.db, t, self.s.source)
                filled += t.status == 'OPEN'
        live = [t for t in trades if t.status == 'OPEN']
        if live:
            def resume(t):
                lv = t.last_bar_ms + M1 if t.last_bar_ms is not None else t.entry_ms
                an = t.extra.get('last_anomaly_ms')
                an = an + M5 if an is not None else t.entry_ms - t.entry_ms % M5
                return min(lv, an)
            start = min(resume(t) for t in live)
            for bar in bars_for(self.db, start, now, self.rest):
                for t in book.on_bar(bar):
                    closed += 1
                    with self.db.transaction():
                        save_trade(self.db, t, self.s.source)
                        set_state(self.db, self.s.source, book.equity, book.last_entry_hour_ms)
                    log.info('strategy-h1 trade %s %s R=%.3f equity=%.4f', t.trade_id[:10], t.status, t.r_net,
                             book.equity)
            for t in live:
                if t.status == 'OPEN':
                    save_trade(self.db, t, self.s.source)      # lưu tiến độ (last_bar_ms)
        return {'open': sum(t.status in ('OPEN', 'PENDING') for t in trades), 'filled': filled, 'closed': closed,
                'equity': book.equity}


def strategy_worker(once: bool = False):
    settings = Settings()
    with connect() as db:
        strat = Strategy(db, settings)
        log.info('strategy-h1: bundle=%s fold=%s entry=%s params=%s', settings.bundle_path,
                 strat.bundle.get('trained_fold'), settings.entry, strat.params)
        while True:
            try:
                made = []
                for H in strat.pending_hours():
                    ev = strat.run_hour(H)
                    if ev is None:
                        break                                  # chờ dữ liệu giờ này, giữ thứ tự
                    made.append(ev)
                m = strat.manage()
                last = made[-1] if made else None
                sig = (last or {}).get('signal') or {}
                heartbeat(db, 'strategy-h1', {'state': 'ok', 'signals': len(made), **m,
                                              'last_hour': last['event_time_ms'] if last else None,
                                              'p_touch': sig.get('p_touch'), 'p_up': sig.get('p_up')})
                if once:
                    return
            except Exception as exc:  # noqa: BLE001
                heartbeat(db, 'strategy-h1', {'state': 'error', 'error': str(exc)[:500]})
                log.exception('strategy-h1 cycle failed')
                if once:
                    raise
            time.sleep(settings.poll)


# --------------------------------------------------------------------------- #
# Kiểm tra quy ước thời gian của dữ liệu vị thế REST
# --------------------------------------------------------------------------- #
def check_positioning(day: str | None = None):
    """So open interest REST (observations) với file ngày của data.binance.vision để tìm độ lệch timestamp.
    In ra giá trị POSITIONING_TS_SHIFT_MS nên dùng (kỳ vọng 0)."""
    import io
    import zipfile
    day = day or (pd.Timestamp.now(tz='UTC').normalize() - pd.Timedelta(days=2)).strftime('%Y-%m-%d')
    url = f'https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-{day}.zip'
    r = httpx.get(url, timeout=60, follow_redirects=True)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        vision = pd.read_csv(z.open(z.namelist()[0]))
    vision['create_ms'] = pd.to_datetime(vision['create_time'], utc=True).dt.as_unit('ms').astype('int64')
    lo = int(pd.Timestamp(day, tz='UTC').value // 1_000_000)
    with connect() as db:
        rows = db.execute("""SELECT data FROM observations WHERE instrument=%s AND timeframe='5m' AND kind='oi'
            AND period_start_ms >= %s AND period_start_ms < %s""", (INSTRUMENT, lo - H1, lo + 86_400_000 + H1)).fetchall()
    rest = pd.DataFrame([{'ts': int(x['data']['source_timestamp_ms']), 'oi': float(x['data']['sumOpenInterest'])}
                         for x in rows])
    if rest.empty:
        print(f'Không có open interest REST trong DB cho ngày {day} (collector chỉ giữ ~28 ngày).')
        return None
    v = vision.set_index('create_ms')['sum_open_interest']
    best = None
    for shift in (-2 * M5, -M5, 0, M5, 2 * M5):
        match = np.isclose(rest['oi'].to_numpy(), v.reindex(rest['ts'] + shift).to_numpy()).mean()
        print(f'shift {shift // 60000:+d} phút: khớp {match:.1%}')
        if best is None or match > best[1]:
            best = (shift, match)
    print(f'=> POSITIONING_TS_SHIFT_MS={best[0]} (khớp {best[1]:.1%})')
    return best
