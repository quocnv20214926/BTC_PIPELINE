"""Bộ thực thi (paper) của chiến lược H1 - logic thuần, không phụ thuộc DB, dùng chung cho live và kiểm thử.

Khớp với model_training/strategy_h1.simulate + select_trades:
  * Tín hiệu của nến 1h mở lúc T (quyết định lúc H = T + 1h). Danh mục: bỏ tín hiệu nếu đang có >= max_positions
    lệnh (lệnh chờ khớp cũng tính; lệnh tính là đang mở nếu thời điểm thoát >= T) hoặc cách lần vào trước < min_gap.
  * Vào lệnh:
      next_open      : open của nến đầu tiên từ start_ms (backtest: nến 5m mở lúc H), phí taker.
      limit_pullback : limit = ref * exp(-side * pullback_mult * b), ref = open nến mở lúc H. Các nến mở trong
                       [start_ms, H + pullback_min): open đã qua limit -> khớp ở open; low / high xuyên limit >= fill_bp
                       -> khớp ở limit (nến khớp chỉ xét SL). Hết thời gian chờ -> market ở open nến kế tiếp (taker).
  * TP = entry * exp(side * tp_mult * b), SL = entry * exp(-side * sl_mult * b). Mỗi nến (kể cả nến vào lệnh): SL trước
    (khớp ở open nếu giá mở đã vượt SL), rồi TP. Hết max_hold_h -> thoát ở close nến cuối trước hạn.
  * R ròng = R - (phí vào + phí ra taker) / (sl_mult * b). Rủi ro mỗi lệnh = risk_per_trade * vốn lúc vào (lãi kép).

Nến được đưa vào theo thứ tự thời gian bằng Bar(open_ms, close_ms, o, h, l, c, levels). Chỉ nến levels=True được dùng
(live: nến 1m, thiếu thì nến 5m; backtest / kiểm thử: nến 5m).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

H_MS = 3_600_000
M1_MS = 60_000


@dataclass
class ExecParams:
    tp_mult: float = 1.0
    sl_mult: float = 1.0
    max_positions: int = 5
    min_gap_h: int = 1
    max_hold_h: int = 9
    entry_mode: str = "limit_pullback"
    pullback_mult: float = 0.05
    pullback_min: int = 10
    fill_bp: float = 0.0001
    fee_maker: float = 0.0002
    fee_taker: float = 0.0005
    risk_per_trade: float = 0.01

    @classmethod
    def from_strategy(cls, cfg, **over) -> "ExecParams":
        return cls(tp_mult=cfg.tp_mult, sl_mult=cfg.sl_mult, max_positions=cfg.max_positions, min_gap_h=cfg.min_gap_bars,
                   max_hold_h=cfg.max_hold_hours, entry_mode=cfg.entry_mode, pullback_mult=cfg.pullback_mult,
                   pullback_min=cfg.pullback_minutes, fill_bp=cfg.fill_bp, fee_maker=cfg.fee_maker,
                   fee_taker=cfg.fee_taker, risk_per_trade=cfg.risk_per_trade, **over)


@dataclass
class Bar:
    open_ms: int
    close_ms: int
    o: float
    h: float
    l: float
    c: float
    levels: bool = True


@dataclass
class Trade:
    trade_id: str
    signal_id: str
    hour_open_ms: int            # nến 1h tín hiệu mở lúc này
    side: int                    # +1 long, -1 short
    b: float                     # biên b (tỉ lệ, vd 0.012)
    status: str = "PENDING"      # PENDING -> OPEN -> TP | SL | TIMEOUT  (CANCELLED: không vào được)
    entry_ms: int | None = None  # open time của nến vào lệnh
    entry: float | None = None
    entry_ref: float | None = None   # giá vào của backtest (nến 5m), để đo trượt giá
    tp: float | None = None
    sl: float | None = None
    risk_frac: float | None = None   # phần vốn đặt rủi ro
    exit_ms: int | None = None       # open time của nến thoát lệnh
    exit_close_ms: int | None = None
    exit_price: float | None = None
    r_gross: float | None = None
    r_net: float | None = None
    equity_after: float | None = None
    last_bar_ms: int | None = None   # nến cuối cùng đã xử lý (để chạy tiếp sau khi khởi động lại)
    extra: dict = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.status in ("PENDING", "OPEN")

    @property
    def decision_ms(self) -> int:
        return self.hour_open_ms + H_MS

    def to_dict(self) -> dict:
        return asdict(self)


def sl_pct(p: ExecParams, b: float) -> float:
    return p.sl_mult * b


def fee_r(p: ExecParams, t: Trade) -> float:
    return ((p.fee_maker if t.extra.get("maker") else p.fee_taker) + p.fee_taker) / sl_pct(p, t.b)


def open_trade(t: Trade, entry_ms: int, entry: float, p: ExecParams, equity: float, maker: bool = False,
               intrabar: bool = False, entry_ref: float | None = None) -> Trade:
    """Đặt giá vào và mức TP / SL. intrabar: khớp limit giữa nến -> nến đó chỉ xét SL."""
    t.status, t.entry_ms, t.entry = "OPEN", int(entry_ms), float(entry)
    t.entry_ref = float(entry_ref) if entry_ref is not None else float(entry)
    t.tp = t.entry * math.exp(t.side * p.tp_mult * t.b)
    t.sl = t.entry * math.exp(-t.side * p.sl_mult * t.b)
    t.risk_frac = p.risk_per_trade * equity
    t.extra.update({"maker": bool(maker), "intrabar": bool(intrabar),
                    "slippage_r": t.side * math.log(t.entry / t.entry_ref) / sl_pct(p, t.b)})
    return t


def try_fill(t: Trade, bar: Bar, p: ExecParams, equity: float) -> bool:
    """Xử lý một nến cho lệnh PENDING. Trả về True nếu lệnh vừa vào. t.extra: start_ms (nến sớm nhất được dùng),
    ref (open nến mở lúc quyết định; mặc định = open nến đầu tiên được dùng)."""
    if t.status != "PENDING" or not bar.levels or bar.open_ms < int(t.extra.get("start_ms", t.decision_ms)):
        return False
    last = t.extra.get("last_pending_ms")
    if last is not None and bar.open_ms <= last:
        return False
    t.extra["last_pending_ms"] = bar.open_ms
    if p.entry_mode == "next_open":
        open_trade(t, bar.open_ms, bar.o, p, equity, entry_ref=t.extra.get("ref"))
        return True
    ref = t.extra.setdefault("ref", bar.o)
    s = t.side
    lim = ref * math.exp(-s * p.pullback_mult * t.b)
    if bar.open_ms >= t.decision_ms + p.pullback_min * M1_MS:                # hết thời gian chờ -> market
        open_trade(t, bar.open_ms, bar.o, p, equity, entry_ref=t.extra.get("ref"))
        return True
    if (s > 0 and bar.o <= lim) or (s < 0 and bar.o >= lim):
        open_trade(t, bar.open_ms, bar.o, p, equity, maker=True, entry_ref=t.extra.get("ref"))
        return True
    if (s > 0 and bar.l <= lim * (1 - p.fill_bp)) or (s < 0 and bar.h >= lim * (1 + p.fill_bp)):
        open_trade(t, bar.open_ms, lim, p, equity, maker=True, intrabar=True, entry_ref=t.extra.get("ref"))
        return True
    return False


def deadline_ms(t: Trade, p: ExecParams) -> int:
    """Nến có open < hạn này còn được xét (backtest: max_hold_h * 12 nến 5m kể từ nến vào lệnh)."""
    return int(t.entry_ms) + p.max_hold_h * H_MS


def step(t: Trade, bar: Bar, p: ExecParams) -> bool:
    """Xử lý một nến cho một lệnh đang mở (kể cả nến vào lệnh). Trả về True nếu lệnh vừa đóng.
    Mỗi nến chỉ được xử lý một lần (last_bar_ms) để chạy lại sau khi khởi động lại vẫn đúng."""
    if (t.status != "OPEN" or not bar.levels or bar.open_ms < t.entry_ms or bar.open_ms >= deadline_ms(t, p)
            or (t.last_bar_ms is not None and bar.open_ms <= t.last_bar_ms)):
        return False
    s, e, sp = t.side, t.entry, sl_pct(p, t.b)
    if (s > 0 and bar.l <= t.sl) or (s < 0 and bar.h >= t.sl):
        px = bar.o if ((s > 0 and bar.o < t.sl) or (s < 0 and bar.o > t.sl)) else t.sl
        return _close(t, "SL", bar, px, s * math.log(px / e) / sp, p)
    intrabar = t.extra.get("intrabar") and bar.open_ms == t.entry_ms
    if not intrabar and ((s > 0 and bar.h >= t.tp) or (s < 0 and bar.l <= t.tp)):
        return _close(t, "TP", bar, t.tp, p.tp_mult / p.sl_mult, p)
    t.last_bar_ms = bar.open_ms
    t.extra["last_close"] = bar.c
    if bar.close_ms >= deadline_ms(t, p):                # nến cuối cùng trước hạn giữ lệnh
        return _close(t, "TIMEOUT", bar, bar.c, s * math.log(bar.c / e) / sp, p)
    return False


def _close(t: Trade, status: str, bar: Bar, px: float, r_gross: float, p: ExecParams) -> bool:
    t.status, t.exit_ms, t.exit_close_ms, t.exit_price = status, int(bar.open_ms), int(bar.close_ms), float(px)
    t.r_gross = float(r_gross)
    t.r_net = float(r_gross - fee_r(p, t))
    t.last_bar_ms = bar.open_ms
    return True


def mark_r(t: Trade, price: float, p: ExecParams) -> float | None:
    if t.entry is None:
        return None
    return t.side * math.log(price / t.entry) / sl_pct(p, t.b)


class Book:
    """Danh mục paper: nhận / bỏ tín hiệu, vào lệnh, theo dõi, vốn lãi kép."""

    def __init__(self, p: ExecParams, equity: float = 1.0, trades: list[Trade] | None = None,
                 last_entry_hour_ms: int | None = None):
        self.p, self.equity = p, float(equity)
        self.trades = list(trades or [])
        self.active = [t for t in self.trades if t.is_open]          # lệnh chờ khớp / đang mở
        self.last_entry_hour_ms = last_entry_hour_ms

    def open_count(self, hour_open_ms: int) -> int:
        """Như backtest: lệnh tính là đang mở nếu chưa đóng hoặc nến thoát mở lúc >= giờ mở nến tín hiệu."""
        recent = self.trades[-4 * self.p.max_positions:]                  # lệnh cũ hơn đã thoát từ lâu
        return sum(1 for t in recent if t.status not in ("SKIPPED", "CANCELLED") and
                   (t.is_open or (t.exit_ms is not None and t.exit_ms >= hour_open_ms)))

    def admit(self, hour_open_ms: int) -> tuple[bool, str]:
        if self.open_count(hour_open_ms) >= self.p.max_positions:
            return False, "max_positions"
        if self.last_entry_hour_ms is not None and hour_open_ms - self.last_entry_hour_ms < self.p.min_gap_h * H_MS:
            return False, "min_gap"
        return True, "ok"

    def add_signal(self, trade: Trade) -> tuple[bool, str]:
        ok, why = self.admit(trade.hour_open_ms)
        if ok:
            self.trades.append(trade)
            self.active.append(trade)
            self.last_entry_hour_ms = trade.hour_open_ms
        return ok, why

    def on_bar(self, bar: Bar) -> tuple[list[Trade], list[Trade]]:
        """Một nến: lệnh đang mở trước (thoát trước khi vào lệnh mới), rồi lệnh chờ khớp (lệnh vừa vào được xét
        luôn trên nến đó). Trả về (lệnh vừa vào, lệnh vừa đóng)."""
        closed, filled = [], []
        for t in self.active:
            if t.status == "OPEN" and step(t, bar, self.p):
                self._book(t, closed)
        for t in self.active:
            if t.status == "PENDING" and try_fill(t, bar, self.p, self.equity):
                filled.append(t)
                if step(t, bar, self.p):
                    self._book(t, closed)
        if closed:
            self.active = [t for t in self.active if t.is_open]
        return filled, closed

    def _book(self, t: Trade, closed: list):
        self.equity += t.risk_frac * t.r_net
        t.equity_after = self.equity
        closed.append(t)


def replay(signals, bars, p: ExecParams, equity: float = 1.0) -> Book:
    """Chạy lại danh mục trên nến 5m liên tục theo đúng quy ước backtest.
    signals: iterable (signal_id, hour_open_ms, side, b) theo thời gian; bars: iterable Bar 5m."""
    book = Book(p, equity)
    pending = {}
    for sid, hour_open, side, b in signals:
        pending.setdefault(int(hour_open) + H_MS, []).append((sid, int(hour_open), int(side), float(b)))
    for bar in bars:
        for sid, hour_open, side, b in pending.pop(bar.open_ms, []):
            book.add_signal(Trade(trade_id=sid, signal_id=sid, hour_open_ms=hour_open, side=side, b=b))
        book.on_bar(bar)
    return book
