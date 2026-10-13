"""Bộ thực thi (paper) của chiến lược H1 – logic thuần, không phụ thuộc DB, dùng chung cho live và kiểm thử.

Khớp với model_training/strategy_h1.backtest_m5:
  * Tín hiệu của nến 1h mở lúc T (đóng lúc T + 1h). Danh mục: bỏ tín hiệu nếu đang có >= max_positions lệnh
    (một lệnh được tính là đang mở nếu thời điểm thoát >= T, đúng như backtest) hoặc cách lệnh trước < min_gap.
  * Giá vào: backtest = open nến 5m đầu tiên của giờ kế tiếp (entry_mode 'bar_open'); live mặc định
    'next_1m' = open của nến 1m đầu tiên SAU lúc ra quyết định (thực tế hơn; chênh lệch so với giá backtest
    được ghi lại ở slippage_r).
  * TP = entry * exp(side * tp_mult * b), SL = entry * exp(-side * sl_mult * b). Trên mỗi nến (kể cả nến vào lệnh):
    SL trước (khớp ở open nếu giá mở đã vượt SL), rồi TP (R = tp_mult / sl_mult), rồi THOÁT NHANH: nến 5m
    NGƯỢC chiều lệnh có (high - low) / open >= anomaly_range -> thoát ở close. Hết max_hold_h -> thoát ở close
    nến cuối cùng trước hạn.
  * R ròng = R - fee / (sl_mult * b). Rủi ro mỗi lệnh = risk_per_trade * vốn lúc vào lệnh (lãi kép).

Nến được đưa vào theo thứ tự thời gian bằng Bar(open_ms, close_ms, o, h, l, c, levels, anomaly):
  levels=True  -> dùng để kiểm tra TP / SL; anomaly=True -> dùng để kiểm tra thoát nhanh (nến 5m).
  Backtest / kiểm thử: chỉ nến 5m với cả hai cờ. Live: nến 1m (levels) + nến 5m (anomaly).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

H_MS = 3_600_000
M5_MS = 300_000


@dataclass
class ExecParams:
    tp_mult: float = 1.0
    sl_mult: float = 0.75
    anomaly_range: float = 0.0075
    max_positions: int = 5
    min_gap_h: int = 1
    max_hold_h: int = 720
    fee_pct: float = 0.001
    risk_per_trade: float = 0.01

    @classmethod
    def from_strategy(cls, cfg, **over) -> "ExecParams":
        return cls(tp_mult=cfg.tp_mult, sl_mult=cfg.sl_mult, anomaly_range=cfg.anomaly_range,
                   max_positions=cfg.max_positions, min_gap_h=cfg.min_gap_bars, max_hold_h=cfg.max_hold_bars,
                   fee_pct=cfg.fee_pct, risk_per_trade=cfg.risk_per_trade, **over)


@dataclass
class Bar:
    open_ms: int
    close_ms: int
    o: float
    h: float
    l: float
    c: float
    levels: bool = True
    anomaly: bool = True


@dataclass
class Trade:
    trade_id: str
    signal_id: str
    hour_open_ms: int            # nến 1h tín hiệu mở lúc này
    side: int                    # +1 long, -1 short
    b: float                     # biên b (tỉ lệ, vd 0.012)
    status: str = "PENDING"      # PENDING -> OPEN -> TP | SL | ANOMALY | TIMEOUT
    entry_ms: int | None = None  # open time của nến vào lệnh
    entry: float | None = None
    entry_ref: float | None = None   # giá vào của backtest (open nến 5m đầu giờ kế tiếp)
    tp: float | None = None
    sl: float | None = None
    risk_frac: float | None = None   # phần vốn đặt rủi ro
    exit_ms: int | None = None       # open time của nến thoát lệnh (giống t5[ex] của backtest)
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

    def to_dict(self) -> dict:
        return asdict(self)


def sl_pct(p: ExecParams, b: float) -> float:
    return p.sl_mult * b


def fee_r(p: ExecParams, b: float) -> float:
    return p.fee_pct / sl_pct(p, b)


def open_trade(t: Trade, entry_ms: int, entry: float, p: ExecParams, equity: float, entry_ref: float | None = None):
    """Đặt giá vào và mức TP / SL."""
    t.status, t.entry_ms, t.entry = "OPEN", int(entry_ms), float(entry)
    t.entry_ref = float(entry_ref) if entry_ref is not None else float(entry)
    t.tp = t.entry * math.exp(t.side * p.tp_mult * t.b)
    t.sl = t.entry * math.exp(-t.side * p.sl_mult * t.b)
    t.risk_frac = p.risk_per_trade * equity
    t.extra["slippage_r"] = t.side * math.log(t.entry / t.entry_ref) / sl_pct(p, t.b)
    return t


def deadline_ms(t: Trade, p: ExecParams) -> int:
    """Nến có open < hạn này còn được xét (backtest: max_hold_h * 12 nến 5m kể từ nến vào lệnh)."""
    return int(t.entry_ms) + p.max_hold_h * H_MS


def step(t: Trade, bar: Bar, p: ExecParams) -> bool:
    """Xử lý một nến cho một lệnh đang mở. Trả về True nếu lệnh vừa đóng.
    Nến levels: từ nến vào lệnh trở đi. Nến anomaly (5m): từ nến 5m chứa thời điểm vào lệnh trở đi.
    Mỗi nến chỉ được xử lý một lần (last_bar_ms / last_anomaly_ms) để chạy lại sau khi khởi động lại vẫn đúng."""
    if t.status != "OPEN" or bar.open_ms >= deadline_ms(t, p):
        return False
    s, e, sp = t.side, t.entry, sl_pct(p, t.b)
    use_levels = bar.levels and bar.open_ms >= t.entry_ms and (t.last_bar_ms is None or bar.open_ms > t.last_bar_ms)
    last_an = t.extra.get("last_anomaly_ms")
    use_anomaly = (bar.anomaly and p.anomaly_range > 0 and bar.open_ms >= t.entry_ms - t.entry_ms % M5_MS
                   and (last_an is None or bar.open_ms > last_an))
    if use_levels:
        hit_sl = bar.l <= t.sl if s > 0 else bar.h >= t.sl
        if hit_sl:
            px = t.sl
            if (s > 0 and bar.o < t.sl) or (s < 0 and bar.o > t.sl):
                px = bar.o
            return _close(t, "SL", bar, px, s * math.log(px / e) / sp, p)
        hit_tp = bar.h >= t.tp if s > 0 else bar.l <= t.tp
        if hit_tp:
            return _close(t, "TP", bar, t.tp, p.tp_mult / p.sl_mult, p)
    if use_anomaly:
        t.extra["last_anomaly_ms"] = bar.open_ms
        against = bar.c < bar.o if s > 0 else bar.c > bar.o
        if against and (bar.h - bar.l) / bar.o >= p.anomaly_range:
            return _close(t, "ANOMALY", bar, bar.c, s * math.log(bar.c / e) / sp, p)
    if use_levels:
        t.last_bar_ms = bar.open_ms
        t.extra["last_close"] = bar.c
        if bar.close_ms >= deadline_ms(t, p):            # nến cuối cùng trước hạn giữ lệnh
            return _close(t, "TIMEOUT", bar, bar.c, s * math.log(bar.c / e) / sp, p)
    return False


def _close(t: Trade, status: str, bar: Bar, px: float, r_gross: float, p: ExecParams) -> bool:
    t.status, t.exit_ms, t.exit_close_ms, t.exit_price = status, int(bar.open_ms), int(bar.close_ms), float(px)
    t.r_gross = float(r_gross)
    t.r_net = float(r_gross - fee_r(p, t.b))
    t.last_bar_ms = bar.open_ms
    return True


def mark_r(t: Trade, price: float, p: ExecParams) -> float | None:
    if t.entry is None:
        return None
    return t.side * math.log(price / t.entry) / sl_pct(p, t.b)


class Book:
    """Danh mục paper: quyết định nhận / bỏ tín hiệu, vốn lãi kép."""

    def __init__(self, p: ExecParams, equity: float = 1.0, trades: list[Trade] | None = None,
                 last_entry_hour_ms: int | None = None):
        self.p, self.equity = p, float(equity)
        self.trades = list(trades or [])
        self.last_entry_hour_ms = last_entry_hour_ms

    def open_count(self, hour_open_ms: int) -> int:
        """Như backtest: lệnh tính là đang mở nếu chưa đóng hoặc nến thoát mở lúc >= giờ mở nến tín hiệu."""
        return sum(1 for t in self.trades if t.status != "SKIPPED" and
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
            self.last_entry_hour_ms = trade.hour_open_ms
        return ok, why

    def fill(self, trade: Trade, entry_ms: int, entry: float, entry_ref: float | None = None):
        open_trade(trade, entry_ms, entry, self.p, self.equity, entry_ref)

    def on_bar(self, bar: Bar) -> list[Trade]:
        closed = []
        for t in self.trades:
            if t.status == "OPEN" and step(t, bar, self.p):
                self.equity += t.risk_frac * t.r_net
                t.equity_after = self.equity
                closed.append(t)
        return closed


def replay(signals, bars, p: ExecParams, equity: float = 1.0) -> Book:
    """Chạy lại danh mục trên nến 5m theo đúng quy ước backtest (giá vào = open nến 5m đầu giờ kế tiếp).
    signals: iterable (signal_id, hour_open_ms, side, b) đã sắp theo thời gian; bars: iterable Bar 5m liên tục."""
    book = Book(p, equity)
    pending = {}
    for sid, hour_open, side, b in signals:
        pending.setdefault(int(hour_open) + H_MS, []).append((sid, int(hour_open), int(side), float(b)))
    for bar in bars:
        book.on_bar(bar)                                   # lệnh cũ trước (thoát trước khi vào lệnh mới)
        for sid, hour_open, side, b in pending.pop(bar.open_ms, []):
            t = Trade(trade_id=sid, signal_id=sid, hour_open_ms=hour_open, side=side, b=b)
            ok, why = book.add_signal(t)
            if not ok:
                continue
            book.fill(t, bar.open_ms, bar.o, entry_ref=bar.o)
            if step(t, bar, p):
                book.equity += t.risk_frac * t.r_net
                t.equity_after = book.equity
    return book
