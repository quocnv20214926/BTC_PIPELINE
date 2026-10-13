"""Worker `telegram`: thông báo chiến lược H1 (strategy-h1) qua Telegram bot.

Luồng xử lý (poll PostgreSQL, không cần Kafka):
  1. Lệnh paper vừa khớp (strategy_trades: PENDING -> OPEN) -> tin LONG/SHORT: giá vào, TP, SL, b, xác suất
     của model, chênh lệch giá vào so với backtest, số lệnh đang mở, vốn paper.
  2. Lệnh vừa đóng (TP / SL / thoát nhanh 5m / hết thời gian giữ) -> trả lời ngay dưới tin gốc: R gộp, R sau phí,
     thời gian giữ, vốn sau lệnh.
  3. Tín hiệu bị bộ thực thi bỏ qua (đủ 5 lệnh, trễ) -> tin ngắn.
  4. Cảnh báo khi strategy-h1 ngừng heartbeat / báo lỗi / quá lâu không có tín hiệu giờ mới, và khi hồi phục.
  5. Tổng kết hằng ngày lúc TELEGRAM_DAILY_HOUR (giờ địa phương).
  6. Lệnh trong chat: /status, /last, /help (chỉ trả lời chat đã cấu hình).

Mọi tin đã gửi được ghi vào bảng telegram_messages (khóa chống trùng), nên restart không gửi lặp.

Biến môi trường:
  TELEGRAM_BOT_TOKEN            token từ @BotFather (bắt buộc)
  TELEGRAM_CHAT_ID              chat / group / channel nhận tin (lấy bằng `python -m pipeline telegram-setup`)
  TELEGRAM_SOURCE               strategy_h1
  TELEGRAM_HEALTH_SERVICE       strategy-h1
  TELEGRAM_MAX_SIGNAL_AGE_MIN   90     (không gửi thông báo cho lệnh / tín hiệu cũ hơn, ví dụ sau khi backfill)
  TELEGRAM_NOTIFY_SKIPPED       true
  TELEGRAM_UTC_OFFSET_HOURS     7
  TELEGRAM_DAILY_HOUR           8      (-1 để tắt tổng kết hằng ngày)
  TELEGRAM_POLL_SECONDS         5
  TELEGRAM_STALE_MIN            90     (cảnh báo khi không có tín hiệu giờ mới quá số phút này)
"""

from __future__ import annotations

import html
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone

import httpx
from psycopg.types.json import Jsonb

from .core import INSTRUMENT, STRATEGY_SOURCE, now_ms
from .storage import connect, heartbeat

log = logging.getLogger(__name__)
API = 'https://api.telegram.org'
M1, M5, H1 = 60_000, 300_000, 3_600_000
SYMBOL = INSTRUMENT.split(':')[-1]          # binance:usdm:BTCUSDT -> BTCUSDT
STATUS_NAME = {'TP': 'TP', 'SL': 'SL', 'ANOMALY': 'Thoát nhanh (nến 5m bất thường)', 'TIMEOUT': 'Hết thời gian giữ'}
STATUS_ICON = {'TP': '✅', 'SL': '❌', 'ANOMALY': '⚡', 'TIMEOUT': '⏱'}


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
# Định dạng tin nhắn (hàm thuần, có unit test)
# --------------------------------------------------------------------------- #
def fmt_price(x) -> str:
    return '—' if x is None else f'{float(x):,.1f}'


def fmt_time(ms: int, offset_hours: float, with_date: bool = True) -> str:
    t = datetime.fromtimestamp(ms / 1000, tz=timezone(timedelta(hours=offset_hours)))
    return t.strftime('%H:%M %d/%m') if with_date else t.strftime('%H:%M')


def fmt_r(x) -> str:
    return '—' if x is None else f'{float(x):+.2f}R'


def fmt_dur(ms: int) -> str:
    m = int(ms // 60_000)
    return f'{m // 1440}d {m % 1440 // 60}h' if m >= 1440 else (f'{m // 60}h {m % 60:02d}m' if m >= 60 else f'{m}m')


def format_open(tr: dict, ev: dict, offset_hours: float, open_count: int | None = None,
                equity: float | None = None) -> str:
    """tr: dòng strategy_trades (dict); ev: payload signal_events của tín hiệu."""
    long = int(tr['side']) > 0
    s, d = ev.get('signal') or {}, ev.get('decision') or {}
    e, b = float(tr['entry']), float(tr['b_pct'])
    p = tr.get('params') or {}
    fee = p.get('fee_pct', 0.001) / (p.get('sl_mult', 0.75) * b)
    slip = (tr.get('state') or {}).get('slippage_r')
    lines = [
        f'{"🟢" if long else "🔴"} <b>{"LONG" if long else "SHORT"} {SYMBOL}</b> · H1 · {html.escape(str(ev.get("source", "")))}',
        f'⏰ Nến tín hiệu {fmt_time(int(tr["hour_open_ms"]), offset_hours, False)}–'
        f'{fmt_time(int(tr["hour_open_ms"]) + H1, offset_hours)} · vào lúc {fmt_time(int(tr["entry_ms"]), offset_hours, False)}',
        f'Giá vào: <b>{fmt_price(e)}</b>',
        f'TP: {fmt_price(tr["tp"])} ({(float(tr["tp"]) / e - 1) * 100:+.2f}%)',
        f'SL: {fmt_price(tr["sl"])} ({(float(tr["sl"]) / e - 1) * 100:+.2f}%)',
        f'Thoát nhanh: nến 5m ngược chiều có biên độ ≥ {p.get("anomaly_range", 0.0075) * 100:.2f}% · '
        f'giữ tối đa {p.get("max_hold_h", 720) // 24} ngày',
        f'<i>p_touch {s.get("p_touch", float("nan")):.2f} · p_lên {s.get("p_up", float("nan")):.2f} · '
        f'b {b * 100:.2f}% · phí ≈ {fee:.2f}R</i>',
    ]
    extra = []
    if slip is not None:
        extra.append(f'lệch giá so với backtest {fmt_r(slip)}')
    if open_count is not None:
        extra.append(f'{open_count} lệnh đang mở')
    if equity is not None:
        extra.append(f'vốn paper ×{equity:.3f}')
    if extra:
        lines.append('<i>' + ' · '.join(extra) + '</i>')
    return '\n'.join(lines)


def format_close(tr: dict, offset_hours: float) -> str:
    st = tr['status']
    side = 'LONG' if int(tr['side']) > 0 else 'SHORT'
    lines = [f'{STATUS_ICON.get(st, "•")} <b>{STATUS_NAME.get(st, st)}</b> · {side} · '
             f'<b>{fmt_r(tr["r_gross"])}</b> (sau phí {fmt_r(tr["r_net"])})',
             f'Thoát {fmt_price(tr["exit_price"])} lúc {fmt_time(int(tr["exit_close_ms"]), offset_hours)} '
             f'(giữ {fmt_dur(int(tr["exit_close_ms"]) - int(tr["entry_ms"]))})']
    if tr.get('equity_after') is not None:
        lines.append(f'<i>Vốn paper ×{float(tr["equity_after"]):.3f}</i>')
    return '\n'.join(lines)


def format_skipped(ev: dict, offset_hours: float) -> str:
    d = ev['decision']
    why = {'max_positions': 'đã đủ số lệnh tối đa', 'min_gap': 'quá gần lệnh trước',
           'late': 'tín hiệu ra quá trễ (sau khi khởi động lại / thiếu dữ liệu)'}.get(d.get('skip_reason'),
                                                                                    d.get('skip_reason'))
    return (f'⏭ Tín hiệu {d["side"]} giờ {fmt_time(int(ev["hour_open_ms"]), offset_hours)} không vào lệnh: '
            f'{html.escape(str(why))}.')


def summarize(trades: list[dict]) -> dict:
    n = len(trades)
    total = sum(float(t['r_net']) for t in trades)
    return {'n': n, 'wins': sum(float(t['r_net']) > 0 for t in trades), 'sum_r': total,
            'avg_r': total / n if n else None,
            **{k.lower(): sum(t['status'] == k for t in trades) for k in ('TP', 'SL', 'ANOMALY', 'TIMEOUT')}}


def format_summary(title: str, st: dict) -> str:
    if not st['n']:
        return f'{title}: chưa có lệnh nào đóng.'
    return (f'{title}: {st["n"]} lệnh · thắng {st["wins"] / st["n"]:.0%} · TP {st["tp"]} / SL {st["sl"]} / '
            f'thoát nhanh {st["anomaly"]} / hết giờ {st["timeout"]} · tổng {st["sum_r"]:+.2f}R · '
            f'TB {st["avg_r"]:+.3f}R/lệnh (sau phí)')


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
class Notifier:
    def __init__(self, db, bot: Telegram, chat_id: str):
        self.db, self.bot, self.chat = db, bot, str(chat_id)
        self.source = env('TELEGRAM_SOURCE', STRATEGY_SOURCE)
        self.health_service = env('TELEGRAM_HEALTH_SERVICE', 'strategy-h1')
        self.max_age_ms = int(env('TELEGRAM_MAX_SIGNAL_AGE_MIN', '90', float) * 60_000)
        self.notify_skipped = env_bool('TELEGRAM_NOTIFY_SKIPPED', 'true')
        self.tz = env('TELEGRAM_UTC_OFFSET_HOURS', '7', float)
        self.daily_hour = env('TELEGRAM_DAILY_HOUR', '8', int)
        self.stale_ms = int(env('TELEGRAM_STALE_MIN', '90', float) * 60_000)
        self.update_offset = None
        self.alerts: dict[str, bool] = {}

    # -- bookkeeping -------------------------------------------------------- #
    def sent(self, key: str) -> bool:
        return self.db.execute('SELECT 1 FROM telegram_messages WHERE dedup_key=%s', (key,)).fetchone() is not None

    def message_id(self, key: str):
        row = self.db.execute('SELECT telegram_message_id FROM telegram_messages WHERE dedup_key=%s',
                              (key,)).fetchone()
        return row['telegram_message_id'] if row else None

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

    def equity(self):
        row = self.db.execute('SELECT equity FROM strategy_state WHERE source=%s', (self.source,)).fetchone()
        return float(row['equity']) if row else None

    def open_trades(self) -> list[dict]:
        return self.db.execute("""SELECT * FROM strategy_trades WHERE source=%s AND status IN ('PENDING','OPEN')
            ORDER BY hour_open_ms""", (self.source,)).fetchall()

    def closed(self, since_ms: int) -> list[dict]:
        return self.db.execute("""SELECT * FROM strategy_trades WHERE source=%s
            AND status IN ('TP','SL','ANOMALY','TIMEOUT') AND exit_close_ms >= %s ORDER BY exit_close_ms""",
                               (self.source, since_ms)).fetchall()

    # -- 1. lệnh vừa khớp ---------------------------------------------------- #
    def push_opened(self) -> int:
        rows = self.db.execute("""SELECT t.*, e.payload AS ev FROM strategy_trades t
            JOIN signal_events e ON e.signal_id = t.signal_id
            WHERE t.source=%s AND t.entry_ms IS NOT NULL AND t.entry_ms >= %s AND NOT EXISTS (
                SELECT 1 FROM telegram_messages m WHERE m.dedup_key = 'open:' || t.trade_id)
            ORDER BY t.entry_ms""", (self.source, now_ms() - self.max_age_ms)).fetchall()
        n = 0
        for r in rows:
            text = format_open(r, r['ev'], self.tz, len(self.open_trades()), self.equity())
            self.send_once('open:' + r['trade_id'], 'signal', text, r['signal_id'],
                           payload={'event_time_ms': r['hour_open_ms'] + H1, 'side': int(r['side']),
                                    'trade_id': r['trade_id']})
            n += 1
        return n

    # -- 2. lệnh vừa đóng ---------------------------------------------------- #
    def push_closed(self) -> int:
        rows = self.db.execute("""SELECT * FROM strategy_trades t WHERE source=%s
            AND status IN ('TP','SL','ANOMALY','TIMEOUT') AND exit_close_ms >= %s AND NOT EXISTS (
                SELECT 1 FROM telegram_messages m WHERE m.dedup_key = 'close:' || t.trade_id)
            ORDER BY exit_close_ms""", (self.source, now_ms() - self.max_age_ms)).fetchall()
        n = 0
        for r in rows:
            self.send_once('close:' + r['trade_id'], 'result', format_close(r, self.tz), r['signal_id'],
                           reply_to=self.message_id('open:' + r['trade_id']),
                           payload={'event_time_ms': r['hour_open_ms'] + H1, 'side': int(r['side']),
                                    'outcome': {'status': r['status'], 'r_net': r['r_net'], 'r_gross': r['r_gross'],
                                                'minutes': int((r['exit_close_ms'] - r['entry_ms']) // 60_000)}})
            n += 1
        return n

    # -- 3. tín hiệu bị bỏ qua ------------------------------------------------ #
    def push_skipped(self) -> int:
        if not self.notify_skipped:
            return 0
        rows = self.db.execute("""SELECT signal_id, payload FROM signal_events e WHERE source=%s
            AND event_time_ms >= %s AND (payload->'decision'->>'good')::boolean
            AND NOT COALESCE((payload->'decision'->>'admitted')::boolean, false)
            AND NOT EXISTS (SELECT 1 FROM telegram_messages m WHERE m.dedup_key = 'skipped:' || e.signal_id)
            ORDER BY event_time_ms""", (self.source, now_ms() - self.max_age_ms)).fetchall()
        for r in rows:
            self.send_once('skipped:' + r['signal_id'], 'skipped', format_skipped(r['payload'], self.tz),
                           r['signal_id'], payload={'reason': r['payload']['decision'].get('skip_reason')})
        return len(rows)

    # -- 4. sức khoẻ ---------------------------------------------------------- #
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
                   f'Không có tín hiệu giờ mới từ {fmt_time(last, self.tz) if last else "?"} — kiểm tra collector / '
                   f'dữ liệu nến và vị thế.', 'Tín hiệu giờ đã cập nhật trở lại.')

    # -- 5. tổng kết ---------------------------------------------------------- #
    def daily(self):
        if self.daily_hour < 0:
            return
        local = datetime.now(timezone(timedelta(hours=self.tz)))
        if local.hour < self.daily_hour:
            return
        key = 'daily:' + local.strftime('%Y-%m-%d')
        if not self.sent(key):
            self.send_once(key, 'daily', self.report_text(title='📊 <b>Tổng kết</b>'))

    def mark(self, tr: dict):
        row = self.db.execute("""SELECT (data->>'close')::float AS c FROM observations WHERE instrument=%s
            AND kind='candle' AND timeframe='1m' ORDER BY period_start_ms DESC LIMIT 1""", (INSTRUMENT,)).fetchone()
        if not row or tr['entry'] is None:
            return None
        sl_pct = float((tr['params'] or {}).get('sl_mult', 0.75)) * float(tr['b_pct'])
        return int(tr['side']) * math.log(row['c'] / float(tr['entry'])) / sl_pct

    def report_text(self, title='📊 <b>Trạng thái</b>') -> str:
        now = now_ms()
        eq = self.equity()
        lines = [title + (f' · vốn paper ×{eq:.3f}' if eq is not None else ''),
                 format_summary('24h', summarize(self.closed(now - 86_400_000))),
                 format_summary('7 ngày', summarize(self.closed(now - 7 * 86_400_000))),
                 format_summary('30 ngày', summarize(self.closed(now - 30 * 86_400_000))),
                 format_summary('Từ đầu', summarize(self.closed(0)))]
        opened = self.open_trades()
        for tr in opened:
            if tr['status'] == 'PENDING':
                lines.append(f'Chờ khớp: {"LONG" if tr["side"] > 0 else "SHORT"} (giờ {fmt_time(tr["hour_open_ms"], self.tz)})')
                continue
            lines.append(f'Đang mở: {"LONG" if tr["side"] > 0 else "SHORT"} từ {fmt_price(tr["entry"])} '
                         f'({fmt_time(tr["entry_ms"], self.tz)}) · hiện {fmt_r(self.mark(tr))}')
        if not opened:
            lines.append('Không có lệnh đang mở.')
        return '\n'.join(lines)

    def last_text(self) -> str:
        row = self.db.execute("""SELECT payload FROM signal_events WHERE source=%s
            ORDER BY event_time_ms DESC LIMIT 1""", (self.source,)).fetchone()
        if not row:
            return 'Chưa có tín hiệu nào.'
        ev = row['payload']
        s, d = ev.get('signal') or {}, ev['decision']
        if not s:
            return f'🕐 Giờ {fmt_time(ev["hour_open_ms"], self.tz)}: không tính được ({d.get("error", "thiếu dữ liệu")}).'
        return (f'🕐 Nến 1h {fmt_time(ev["hour_open_ms"], self.tz)} · giá đóng {fmt_price(ev["close"])}\n'
                f'p_touch {s["p_touch"]:.3f} (ngưỡng {s["thr_touch"]:.3f}) · p_lên {s["p_up"]:.3f} · '
                f'độ tự tin {s["conf"]:.3f} (ngưỡng {s["thr_conf"]:.3f})\n'
                f'Hướng: {d["side"]} · Tín hiệu: {"CÓ" if d["good"] else "không"}'
                + (f' · vào lệnh: {"có" if d.get("admitted") else "không (" + str(d.get("skip_reason")) + ")"}'
                   if d['good'] else ''))

    # -- 6. lệnh trong chat ---------------------------------------------------- #
    def commands(self):
        if self.update_offset is None:                 # bỏ qua tin gửi lúc bot tắt
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
                self.bot.send(self.chat, '/status – thống kê, vốn paper và lệnh đang mở\n'
                                         '/last – đầu ra model ở nến 1h gần nhất\n/help – trợ giúp')

    def cycle(self, with_commands=True) -> dict:
        out = {'opened': self.push_opened(), 'closed': self.push_closed(), 'skipped': self.push_skipped()}
        self.check_health()
        self.daily()
        if with_commands:
            self.commands()
        return out


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
            bot.send(chat, f'🤖 Bot chiến lược {SYMBOL} H1 ({notifier.source}) đã khởi động.\n'
                           f'<i>Lệnh paper (forward test), không phải khuyến nghị giao dịch.</i>')
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
