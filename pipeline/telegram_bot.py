"""Worker `telegram`: gửi tín hiệu của model-signals qua Telegram bot, theo thời gian thực.

Luồng xử lý (poll bảng signal_events, không cần Kafka):
  1. Tín hiệu mới có decision.good = true (và chưa quá TELEGRAM_MAX_SIGNAL_AGE_MIN phút) -> gửi tin nhắn
     LONG/SHORT với giá tín hiệu, SL, TP 1R/2R, time stop, xác suất và phí ước tính (theo R).
     Mặc định các lệnh độc lập (giống bộ thực thi execution_m1): mỗi tín hiệu tốt là một lệnh riêng, kể cả khi
     lệnh trước còn mở. TELEGRAM_ONE_AT_A_TIME=true để chỉ theo dõi một lệnh tại một thời điểm.
  2. Lệnh đã gửi được theo dõi bằng nến 1m trong observations (thiếu nến 1m thì dùng 5m): chạm SL / TP
     (cùng nến chạm cả hai -> tính SL, bảo thủ) hoặc hết time stop -> trả lời ngay dưới tin gốc với R
     gộp và R sau phí.
  3. Cảnh báo khi model-signals ngừng heartbeat / báo lỗi / tín hiệu bị trễ, và báo khi hồi phục.
  4. Tổng kết hằng ngày lúc TELEGRAM_DAILY_HOUR (giờ địa phương).
  5. Lệnh trong chat: /status, /last, /help (chỉ trả lời chat đã cấu hình).

Mọi tin đã gửi được ghi vào bảng telegram_messages (khóa chống trùng), nên restart không gửi lặp.

Biến môi trường:
  TELEGRAM_BOT_TOKEN            token từ @BotFather (bắt buộc)
  TELEGRAM_CHAT_ID              chat / group / channel nhận tin (lấy bằng `python -m pipeline telegram-setup`)
  TELEGRAM_SOURCE               modelv3_event
  TELEGRAM_HEALTH_SERVICE       model-signals-v3
  TELEGRAM_TRACK_RR             1      (TP được theo dõi: 1 = TP 1R, 2 = TP 2R)
  TELEGRAM_FEE_PCT              0.001  (phí + trượt giá khứ hồi, tỉ lệ giá: 0.1%)
  TELEGRAM_ONE_AT_A_TIME        false
  TELEGRAM_MAX_SIGNAL_AGE_MIN   15     (không gửi tín hiệu cũ, ví dụ sau khi backfill)
  TELEGRAM_UTC_OFFSET_HOURS     7
  TELEGRAM_DAILY_HOUR           8      (-1 để tắt tổng kết hằng ngày)
  TELEGRAM_POLL_SECONDS         5
  TELEGRAM_STALE_MIN            15     (cảnh báo khi không có tín hiệu mới quá số phút này)
"""

from __future__ import annotations

import html
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import httpx
from psycopg.types.json import Jsonb

from .core import INSTRUMENT, now_ms
from .storage import connect, heartbeat

log = logging.getLogger(__name__)
API = 'https://api.telegram.org'
M1, M5 = 60_000, 300_000
SYMBOL = INSTRUMENT.split(':')[-1]          # binance:usdm:BTCUSDT -> BTCUSDT


def env(name, default, cast=str):
    value = os.getenv(name)
    return cast(default if value in (None, '') else value)


def env_bool(name, default):
    return env(name, default).strip().lower() in ('1', 'true', 'yes', 'on')


# --------------------------------------------------------------------------- #
# Telegram HTTP client
# --------------------------------------------------------------------------- #
class TelegramError(RuntimeError):
    pass


class Telegram:
    def __init__(self, token: str, client: httpx.Client | None = None, max_retries: int = 5):
        if not token:
            raise TelegramError('TELEGRAM_BOT_TOKEN is empty')
        self.base = f'{API}/bot{token}/'
        self.client = client or httpx.Client(timeout=30)
        self.max_retries = max_retries

    def call(self, method: str, **params):
        for attempt in range(self.max_retries + 1):
            try:
                r = self.client.post(self.base + method, json={k: v for k, v in params.items() if v is not None})
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise TelegramError(f'{method}: network error {exc}') from exc
                time.sleep(min(30, 2 ** attempt))
                continue
            try:
                body = r.json()
            except ValueError:
                body = {'ok': False, 'description': r.text[:200]}
            if body.get('ok'):
                return body['result']
            if r.status_code == 429 or r.status_code >= 500:          # flood limit / server error -> retry
                wait = (body.get('parameters') or {}).get('retry_after') or min(30, 2 ** attempt)
                if attempt < self.max_retries:
                    time.sleep(float(wait))
                    continue
            # never put the token in an error message
            raise TelegramError(f'{method}: HTTP {r.status_code} {body.get("description")}')
        raise TelegramError(f'{method}: retries exhausted')

    def send(self, chat_id, text: str, reply_to: int | None = None) -> int:
        params = {'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML',
                  'link_preview_options': {'is_disabled': True}}
        if reply_to:
            params['reply_parameters'] = {'message_id': int(reply_to), 'allow_sending_without_reply': True}
        return int(self.call('sendMessage', **params)['message_id'])


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def fmt_price(x) -> str:
    return '—' if x is None else f'{float(x):,.1f}'


def fmt_time(ms: int, offset_hours: float, with_date: bool = True) -> str:
    t = datetime.fromtimestamp(ms / 1000, tz=timezone(timedelta(hours=offset_hours)))
    return t.strftime('%H:%M %d/%m') if with_date else t.strftime('%H:%M')


def fmt_r(x) -> str:
    return '—' if x is None else f'{float(x):+.2f}R'


def trade_levels(ev: dict, rr: int) -> dict:
    """Entry, SL and tracked TP of a signal (levels are anchored to the signal close, as in the backtest)."""
    s, side = ev['signal'], ev['decision']['side']
    p = float(ev['close'])
    b = float(s['barrier_pct'])
    pre = 'long' if side == 'LONG' else 'short'
    tp = s[f'{pre}_tp_2r'] if rr == 2 and s.get(f'{pre}_tp_2r') is not None else s[f'{pre}_tp_1r']
    sl = s[f'{pre}_sl']
    return {'side': side, 'entry': p, 'sl': float(sl), 'tp': float(tp), 'barrier': b,
            'rr': abs(float(tp) - p) / (p * b), 'horizon_ms': int(s.get('time_stop_bars') or 48) * M5}


def prob_line(s: dict) -> str:
    """Xác suất của model: v3 (P(TP)/P(SL)) hoặc v2 (p_touch/p_up)."""
    if s.get('p_tp') is not None:
        ev_tags = [k for k in ('E1', 'E2', 'E3') if s.get(k)]
        names = {'E1': 'nến 5m bất thường', 'E2': '3 nến bùng nổ', 'E3': 'bất thường 1m'}
        tag = (' · ' + ', '.join(names[k] for k in ev_tags)) if ev_tags else ''
        return f'P(TP) {s["p_tp"]:.2f} · P(SL) {s["p_sl"]:.2f}{tag}'
    if s.get('p_touch') is not None:
        return f'p_touch {s["p_touch"]:.2f} · p_up {s["p_up_touch"]:.2f}'
    return ''


def format_signal(ev: dict, rr: int, fee_pct: float, offset_hours: float, source_label: str = '') -> str:
    s, d = ev['signal'], ev['decision']
    lv = trade_levels(ev, rr)
    long = lv['side'] == 'LONG'
    p, b = lv['entry'], lv['barrier']
    pre = 'long' if long else 'short'
    sign = 1 if long else -1
    tp1, tp2 = s.get(f'{pre}_tp_1r'), s.get(f'{pre}_tp_2r')
    mark1 = ' ← theo dõi' if rr == 1 else ''
    tp_label = 'TP' if tp2 is None else 'TP 1R'
    mark2 = ' ← theo dõi' if rr == 2 else ''
    e_key = f'e_{pre}_{rr}r'
    lines = [
        f'{"🟢" if long else "🔴"} <b>{lv["side"]} {SYMBOL}</b> · 5m{(" · " + html.escape(source_label)) if source_label else ""}',
        f'⏰ {fmt_time(ev["event_time_ms"], offset_hours)}',
        f'Giá tín hiệu: <b>{fmt_price(p)}</b>',
        f'SL: {fmt_price(lv["sl"])} ({-sign * b * 100:+.2f}%)',
        f'{tp_label}: {fmt_price(tp1)} ({(float(tp1) / p - 1) * 100:+.2f}%){mark1 if tp2 is not None else ""}',
    ]
    if tp2 is not None:
        lines.append(f'TP 2R: {fmt_price(tp2)} ({(float(tp2) / p - 1) * 100:+.2f}%){mark2}')
    lines += [
        f'Time stop: {lv["horizon_ms"] // 3_600_000}h (đến {fmt_time(ev["event_time_ms"] + lv["horizon_ms"], offset_hours, False)})',
        f'<i>Độ confident (R kỳ vọng sau phí) {fmt_r(d.get("score"))} · ngưỡng {fmt_r(d.get("threshold"))}</i>',
        f'<i>{prob_line(s)} · phí ≈ {fee_pct / b:.2f}R</i>',
    ]
    return '\n'.join(lines)


def evaluate_outcome(ev: dict, bars: list, step_ms: int, rr: int, fee_pct: float) -> dict:
    """Follow a trade on contiguous bars [(open_ms, high, low, close), ...] starting at the signal time.

    status: OPEN (still running, r_now = mark-to-market), GAP (missing bar -> caller may use another
    timeframe), TP / SL / TIMEOUT (closed). If one bar touches both SL and TP the result is SL."""
    lv = trade_levels(ev, rr)
    long = lv['side'] == 'LONG'
    p, b = lv['entry'], lv['barrier']
    t0 = int(ev['event_time_ms'])
    end = t0 + lv['horizon_ms']
    fee_r = fee_pct / b
    expected, last_close = t0, None

    def closed(status, price, at_ms, ambiguous=False):
        gross = (1 if long else -1) * (price - p) / (p * b)
        return {'status': status, 'exit_price': float(price), 'exit_ms': int(at_ms), 'r_gross': float(gross),
                'r_net': float(gross - fee_r), 'fee_r': float(fee_r), 'ambiguous': ambiguous,
                'minutes': int((at_ms - t0) // 60_000), 'bar_ms': step_ms}

    for start, high, low, close in bars:
        if start < t0:
            continue
        if start >= end:
            break
        if start != expected:
            return {'status': 'GAP', 'at_ms': expected}
        hit_sl = low <= lv['sl'] if long else high >= lv['sl']
        hit_tp = high >= lv['tp'] if long else low <= lv['tp']
        if hit_sl:
            return closed('SL', lv['sl'], start + step_ms, ambiguous=hit_tp)
        if hit_tp:
            return closed('TP', lv['tp'], start + step_ms)
        expected, last_close = start + step_ms, close
    if expected >= end:
        return closed('TIMEOUT', last_close, end)
    r_now = None if last_close is None else (1 if long else -1) * (last_close - p) / (p * b)
    return {'status': 'OPEN', 'covered_to_ms': expected, 'r_now': r_now, 'last_close': last_close}


def format_result(ev: dict, out: dict, rr: int, offset_hours: float) -> str:
    icon = {'TP': '✅', 'SL': '❌', 'TIMEOUT': '⏱', 'NO_DATA': '⚠️'}[out['status']]
    side = ev['decision']['side']
    if out['status'] == 'NO_DATA':
        return f'{icon} <b>{side}</b>: không đủ dữ liệu nến để chấm kết quả lệnh này.'
    name = {'TP': f'TP {rr}R', 'SL': 'SL', 'TIMEOUT': 'Hết time stop'}[out['status']]
    lines = [f'{icon} <b>{name}</b> · {side} · <b>{fmt_r(out["r_gross"])}</b> (sau phí {fmt_r(out["r_net"])})',
             f'Thoát {fmt_price(out["exit_price"])} lúc {fmt_time(out["exit_ms"], offset_hours, False)} '
             f'({out["minutes"]} phút)']
    if out.get('ambiguous'):
        lines.append('<i>Cùng một nến chạm cả SL và TP → tính SL.</i>')
    return '\n'.join(lines)


def summarize(results: list[dict]) -> dict:
    n = len(results)
    wins = sum(1 for r in results if r['r_net'] > 0)
    total = sum(r['r_net'] for r in results)
    return {'n': n, 'wins': wins, 'win_rate': wins / n if n else None, 'sum_r': total,
            'avg_r': total / n if n else None,
            'tp': sum(r['status'] == 'TP' for r in results), 'sl': sum(r['status'] == 'SL' for r in results),
            'timeout': sum(r['status'] == 'TIMEOUT' for r in results)}


def format_summary(title: str, st: dict) -> str:
    if not st['n']:
        return f'{title}: chưa có lệnh nào đóng.'
    return (f'{title}: {st["n"]} lệnh · thắng {st["win_rate"]:.0%} · TP {st["tp"]} / SL {st["sl"]} / '
            f'hết giờ {st["timeout"]} · tổng {st["sum_r"]:+.2f}R · TB {st["avg_r"]:+.3f}R/lệnh (sau phí)')


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
class Notifier:
    def __init__(self, db, bot: Telegram, chat_id: str):
        self.db, self.bot, self.chat = db, bot, str(chat_id)
        self.source = env('TELEGRAM_SOURCE', 'modelv3_event')
        self.health_service = env('TELEGRAM_HEALTH_SERVICE', 'model-signals-v3')
        self.rr = env('TELEGRAM_TRACK_RR', '1', int)
        if self.rr not in (1, 2):
            raise ValueError('TELEGRAM_TRACK_RR must be 1 or 2')
        self.fee = env('TELEGRAM_FEE_PCT', '0.001', float)
        self.one_at_a_time = env_bool('TELEGRAM_ONE_AT_A_TIME', 'false')
        self.max_age_ms = int(env('TELEGRAM_MAX_SIGNAL_AGE_MIN', '15', float) * 60_000)
        self.tz = env('TELEGRAM_UTC_OFFSET_HOURS', '7', float)
        self.daily_hour = env('TELEGRAM_DAILY_HOUR', '8', int)
        self.stale_ms = int(env('TELEGRAM_STALE_MIN', '15', float) * 60_000)
        self.update_offset = None
        self.alerts: dict[str, bool] = {}

    # -- bookkeeping -------------------------------------------------------- #
    def sent(self, key: str) -> bool:
        return self.db.execute('SELECT 1 FROM telegram_messages WHERE dedup_key=%s', (key,)).fetchone() is not None

    def record(self, key, kind, signal_id=None, message_id=None, payload=None):
        self.db.execute("""INSERT INTO telegram_messages(dedup_key,kind,chat_id,signal_id,telegram_message_id,payload)
            VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (key, kind, self.chat, signal_id, message_id, Jsonb(payload or {})))

    def send_once(self, key, kind, text, signal_id=None, reply_to=None, payload=None):
        if self.sent(key):
            return None
        mid = self.bot.send(self.chat, text, reply_to)
        self.record(key, kind, signal_id, mid, payload)
        return mid

    def open_trades(self):
        return self.db.execute("""SELECT m.signal_id, m.telegram_message_id, e.payload FROM telegram_messages m
            JOIN signal_events e ON e.signal_id = m.signal_id
            WHERE m.kind='signal' AND m.chat_id=%s AND NOT EXISTS (
                SELECT 1 FROM telegram_messages r WHERE r.dedup_key = 'result:' || m.signal_id)
            ORDER BY e.event_time_ms""", (self.chat,)).fetchall()

    def results(self, since_ms: int) -> list[dict]:
        rows = self.db.execute("""SELECT payload FROM telegram_messages WHERE kind='result' AND chat_id=%s
            AND (payload->>'event_time_ms')::bigint >= %s""", (self.chat, since_ms)).fetchall()
        return [r['payload']['outcome'] for r in rows if r['payload'].get('outcome', {}).get('status') in ('TP', 'SL', 'TIMEOUT')]

    # -- 1. new signals ------------------------------------------------------ #
    def push_signals(self) -> int:
        rows = self.db.execute("""SELECT signal_id, event_time_ms, payload FROM signal_events e
            WHERE source=%s AND event_time_ms >= %s AND (payload->'decision'->>'good')::boolean
              AND NOT EXISTS (SELECT 1 FROM telegram_messages m WHERE m.dedup_key IN
                    ('signal:' || e.signal_id, 'skipped:' || e.signal_id))
            ORDER BY event_time_ms""", (self.source, now_ms() - self.max_age_ms)).fetchall()
        n = 0
        for row in rows:
            ev = row['payload']
            if self.one_at_a_time and self.open_trades():
                self.record('skipped:' + row['signal_id'], 'skipped', row['signal_id'],
                            payload={'reason': 'trade_open', 'event_time_ms': row['event_time_ms']})
                log.info('signal %s skipped: a trade is still open', row['signal_id'][:10])
                continue
            text = format_signal(ev, self.rr, self.fee, self.tz, self.source)
            self.send_once('signal:' + row['signal_id'], 'signal', text, row['signal_id'],
                           payload={'event_time_ms': row['event_time_ms'], 'side': ev['decision']['side']})
            n += 1
        return n

    # -- 2. outcomes --------------------------------------------------------- #
    def bars(self, tf: str, start_ms: int, end_ms: int) -> list:
        rows = self.db.execute("""SELECT period_start_ms AS t, (data->>'high')::float AS h, (data->>'low')::float AS l,
            (data->>'close')::float AS c FROM observations WHERE instrument=%s AND timeframe=%s AND kind='candle'
            AND period_start_ms >= %s AND period_start_ms < %s ORDER BY period_start_ms""",
                               (INSTRUMENT, tf, start_ms, end_ms)).fetchall()
        return [(r['t'], r['h'], r['l'], r['c']) for r in rows]

    def outcome(self, ev: dict) -> dict:
        t0 = int(ev['event_time_ms'])
        end = t0 + trade_levels(ev, self.rr)['horizon_ms']
        out = evaluate_outcome(ev, self.bars('1m', t0, end), M1, self.rr, self.fee)
        if out['status'] in ('TP', 'SL', 'TIMEOUT'):
            return out
        # 1m missing / gapped / behind: 5m candles may already settle the trade
        out5 = evaluate_outcome(ev, self.bars('5m', t0, end), M5, self.rr, self.fee)
        if out5['status'] in ('TP', 'SL', 'TIMEOUT'):
            return out5
        if out['status'] == 'GAP' and out5['status'] != 'OPEN':
            return {'status': 'NO_DATA'} if now_ms() > end + 30 * 60_000 else out
        if now_ms() > end + 30 * 60_000:                 # horizon long over and still unsettled
            return {'status': 'NO_DATA'}
        return out if out['status'] == 'OPEN' and (out.get('covered_to_ms') or 0) >= (out5.get('covered_to_ms') or 0) else out5

    def push_results(self) -> int:
        n = 0
        for row in self.open_trades():
            ev = row['payload']
            out = self.outcome(ev)
            if out['status'] in ('OPEN', 'GAP'):
                continue
            self.send_once('result:' + row['signal_id'], 'result', format_result(ev, out, self.rr, self.tz),
                           row['signal_id'], reply_to=row['telegram_message_id'],
                           payload={'event_time_ms': ev['event_time_ms'], 'side': ev['decision']['side'],
                                    'outcome': out})
            n += 1
        return n

    # -- 3. health alerts ---------------------------------------------------- #
    def alert(self, name: str, bad: bool, text_bad: str, text_ok: str):
        if bad and not self.alerts.get(name):
            self.bot.send(self.chat, '⚠️ ' + text_bad)
        elif not bad and self.alerts.get(name):
            self.bot.send(self.chat, '✅ ' + text_ok)
        self.alerts[name] = bad

    def check_health(self):
        hb = self.db.execute("""SELECT details, extract(epoch FROM now()-updated_at) AS age
            FROM service_heartbeats WHERE service=%s""", (self.health_service,)).fetchone()
        last = self.db.execute('SELECT max(event_time_ms) AS t FROM signal_events WHERE source=%s',
                               (self.source,)).fetchone()['t']
        hb_bad = hb is None or hb['age'] > 600 or (hb['details'] or {}).get('state') == 'error'
        reason = 'không có heartbeat' if hb is None else (
            f'heartbeat cũ {hb["age"] / 60:.0f} phút' if hb['age'] > 600 else str(hb['details'].get('error'))[:300])
        self.alert('model', hb_bad, f'{self.health_service} có vấn đề: {html.escape(reason)}',
                   f'{self.health_service} hoạt động lại.')
        stale = last is not None and now_ms() - last > self.stale_ms
        self.alert('stale', stale,
                   f'Không có tín hiệu mới từ {fmt_time(last, self.tz) if last else "?"} — kiểm tra collector / dữ liệu nến.',
                   'Tín hiệu đã cập nhật trở lại.')

    # -- 4. daily summary ---------------------------------------------------- #
    def daily(self):
        if self.daily_hour < 0:
            return
        local = datetime.now(timezone(timedelta(hours=self.tz)))
        if local.hour < self.daily_hour:
            return
        key = 'daily:' + local.strftime('%Y-%m-%d')
        if self.sent(key):
            return
        self.send_once(key, 'daily', self.report_text(title='📊 <b>Tổng kết</b>'))

    def report_text(self, title='📊 <b>Trạng thái</b>') -> str:
        now = now_ms()
        lines = [title,
                 format_summary('24h', summarize(self.results(now - 86_400_000))),
                 format_summary('7 ngày', summarize(self.results(now - 7 * 86_400_000))),
                 format_summary('30 ngày', summarize(self.results(now - 30 * 86_400_000)))]
        opened = self.open_trades()
        for row in opened:
            ev = row['payload']
            out = self.outcome(ev)
            lines.append(f'Đang mở: {ev["decision"]["side"]} từ {fmt_price(ev["close"])} '
                         f'({fmt_time(ev["event_time_ms"], self.tz)}) · hiện {fmt_r(out.get("r_now"))}')
        if not opened:
            lines.append('Không có lệnh đang mở.')
        return '\n'.join(lines)

    def last_text(self) -> str:
        row = self.db.execute("""SELECT payload FROM signal_events WHERE source=%s
            ORDER BY event_time_ms DESC LIMIT 1""", (self.source,)).fetchone()
        if not row:
            return 'Chưa có tín hiệu nào.'
        ev = row['payload']
        s, d = ev['signal'], ev['decision']
        return (f'🕐 Nến gần nhất {fmt_time(ev["event_time_ms"], self.tz)} · giá {fmt_price(ev["close"])}\n'
                f'Hướng tốt hơn: {d["side"]} · R kỳ vọng sau phí {fmt_r(d.get("score"))} '
                f'(ngưỡng {fmt_r(d.get("threshold"))})\n'
                f'{prob_line(s)}\n'
                f'Vào lệnh: {"CÓ" if d["good"] else "không"}')

    # -- 5. chat commands ---------------------------------------------------- #
    def commands(self):
        if self.update_offset is None:                 # skip everything sent while the bot was down
            old = self.bot.call('getUpdates', offset=-1, timeout=0)
            self.update_offset = old[-1]['update_id'] + 1 if old else 0
        for upd in self.bot.call('getUpdates', offset=self.update_offset, timeout=0,
                                 allowed_updates=['message', 'channel_post']):
            self.update_offset = upd['update_id'] + 1
            msg = upd.get('message') or upd.get('channel_post') or {}
            if str(msg.get('chat', {}).get('id')) != self.chat:
                continue
            cmd = (msg.get('text') or '').split()[0].split('@')[0].lower() if msg.get('text') else ''
            if cmd == '/status':
                self.bot.send(self.chat, self.report_text())
            elif cmd == '/last':
                self.bot.send(self.chat, self.last_text())
            elif cmd in ('/help', '/start'):
                self.bot.send(self.chat, '/status – thống kê và lệnh đang mở\n/last – đầu ra model ở nến gần nhất\n'
                                         '/help – trợ giúp')

    def cycle(self, with_commands=True) -> dict:
        sent = self.push_signals()
        closed = self.push_results()
        self.check_health()
        self.daily()
        if with_commands:
            self.commands()
        return {'signals_sent': sent, 'results_sent': closed}


def telegram_worker(once: bool = False):
    token, chat = env('TELEGRAM_BOT_TOKEN', ''), env('TELEGRAM_CHAT_ID', '')
    poll = env('TELEGRAM_POLL_SECONDS', '5', float)
    with connect() as db:
        if not token or not chat:
            log.warning('telegram disabled: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env '
                        '(then run `docker compose run --rm telegram python -m pipeline telegram-setup`)')
            while not once:
                heartbeat(db, 'telegram', {'state': 'disabled'})
                time.sleep(300)
            return
        bot = Telegram(token)
        notifier = Notifier(db, bot, chat)
        if env_bool('TELEGRAM_STARTUP_MESSAGE', 'true') and not once:
            bot.send(chat, f'🤖 Bot tín hiệu {SYMBOL} 5m ({notifier.source}) đã khởi động.\n'
                           f'<i>Tín hiệu thử nghiệm, chưa phải khuyến nghị giao dịch.</i>')
        errors = 0
        while True:
            try:
                result = notifier.cycle()
                heartbeat(db, 'telegram', {'state': 'ok', **result})
                if any(result.values()):
                    log.info('telegram %s', result)
                errors = 0
            except Exception as exc:  # noqa: BLE001
                errors += 1
                heartbeat(db, 'telegram', {'state': 'error', 'error': str(exc)[:300]})
                log.exception('telegram cycle failed')
                if once:
                    raise
                time.sleep(min(300, poll * 2 ** min(errors, 6)))
                continue
            if once:
                return
            time.sleep(poll)


def telegram_setup():
    """Kiểm tra token, liệt kê các chat đã nhắn bot (để lấy TELEGRAM_CHAT_ID) và gửi tin thử."""
    token, chat = env('TELEGRAM_BOT_TOKEN', ''), env('TELEGRAM_CHAT_ID', '')
    if not token:
        print('Chưa có TELEGRAM_BOT_TOKEN trong .env. Tạo bot với @BotFather rồi dán token vào .env.')
        raise SystemExit(2)
    bot = Telegram(token)
    me = bot.call('getMe')
    print(f'Bot OK: @{me.get("username")} (id {me.get("id")})')
    if chat and str(chat) == str(me.get('id')):
        print(f'TELEGRAM_CHAT_ID={chat} là id của CHÍNH BOT, không phải chat của bạn. '
              'Hãy dùng một id trong danh sách bên dưới (sau khi nhắn /start cho bot).')
        chat = ''
    chats = {}
    for upd in bot.call('getUpdates', timeout=0):
        msg = upd.get('message') or upd.get('channel_post') or upd.get('my_chat_member') or {}
        c = msg.get('chat') or {}
        if c:
            chats[c['id']] = c.get('title') or c.get('username') or c.get('first_name') or ''
    if chats:
        print('Các chat đã nhắn bot gần đây (dùng id làm TELEGRAM_CHAT_ID):')
        for cid, name in chats.items():
            print(f'  {cid}\t{name}')
    else:
        print('Chưa thấy chat nào. Mở Telegram, nhắn /start cho bot (hoặc thêm bot vào group và nhắn một câu), '
              'rồi chạy lại lệnh này.')
    if chat:
        bot.send(chat, '✅ Kết nối thành công. Bot sẽ gửi tín hiệu vào chat này.')
        print(f'Đã gửi tin thử tới TELEGRAM_CHAT_ID={chat}')
