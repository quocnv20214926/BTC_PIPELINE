"""Paper trading tín hiệu chiến lược H1 trên tài khoản futures USDT-M (Binance BTCUSDT perpetual).

margin_mode = "cross" (mặc định): TP / SL đúng theo khoảng cách giá 1b như backtest, không thanh lý từng lệnh. Tài khoản
chỉ bị thanh lý khi vốn (gồm lỗ chưa chốt) < tổng maintenance margin; mỗi lần vào lệnh kiểm tra trường hợp xấu nhất:
mọi lệnh đang mở cùng chạm SL (cột cross_buffer_pct, phải luôn > 0).
margin_mode = "isolated": như mô tả dưới.

Khác backtest theo R (strategy_h1.simulate):
  * Khối lượng: margin = margin_frac * vốn (vốn đã chốt lúc vào lệnh nếu compound, không thì vốn ban đầu);
    notional = margin * leverage; qty BTC = notional / giá vào. Lãi / lỗ USDT = qty * (giá ra - giá vào) * hướng.
  * Thanh lý (isolated): giá thanh lý theo công thức Binance cho 1 vị thế, bậc maintenance margin theo notional. Nếu giá
    thanh lý gần hơn SL thì giá chạm thanh lý trước -> lệnh mất toàn bộ margin. Kiểm tra bằng high / low nến 5m (giá
    giao dịch; sàn dùng mark price, thường êm hơn, nên đây là giả định thận trọng).
  * Phí trên notional: vào lệnh maker (limit khớp) hoặc taker; TP là lệnh limit đặt sẵn -> maker (tp_maker);
    SL / hết giờ -> taker; thanh lý: không tính thêm phí (đã mất cả margin).
  * Funding (mỗi 8h): lệnh đang mở lúc funding trả (LONG khi rate > 0) / nhận qty * giá * rate.
  * Danh mục: đúng luật của backtest (tối đa max_positions lệnh mở kể cả lệnh chờ khớp, các lần vào cách nhau
    >= min_gap_bars giờ); thời điểm thoát theo kết quả futures (lệnh bị thanh lý thoát sớm hơn).
Vào lệnh / TP / SL / hết giờ dùng đúng logic của strategy_h1._simulate. Với liquidation=False, tp_maker=False,
funding=False, danh sách lệnh và R_gross trùng backtest (kiểm tra trong notebook).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import strategy_h1 as st

NS_MIN = 60 * 10**9

# Bậc maintenance margin BTCUSDT perpetual (Binance, gần đúng - kiểm tra lại trên sàn):
# (notional tối đa USDT, tỉ lệ MM, số tiền bù MM, đòn bẩy tối đa)
BTC_MM_TIERS = ((50_000, 0.004, 0.0, 125), (600_000, 0.005, 50.0, 100), (3_000_000, 0.0065, 950.0, 75),
                (12_000_000, 0.01, 11_450.0, 50), (70_000_000, 0.02, 131_450.0, 25), (100_000_000, 0.025, 481_450.0, 20))


@dataclass
class FuturesParams:
    capital: float = 100_000.0
    leverage: float = 50.0
    margin_frac: float = 0.01          # margin mỗi lệnh = 1% vốn
    compound: bool = True              # True: % vốn đã chốt lúc vào lệnh; False: % vốn ban đầu
    fee_maker: float = 0.0002
    fee_taker: float = 0.0005
    tp_maker: bool = True              # TP đặt sẵn bằng lệnh limit
    funding: bool = True
    margin_mode: str = "cross"         # cross | isolated
    liquidation: bool = True           # chỉ dùng với isolated
    tiers: tuple = BTC_MM_TIERS


def mm_tier(notional: float, tiers=BTC_MM_TIERS) -> tuple[float, float, float]:
    for cap, rate, amount, max_lev in tiers:
        if notional <= cap:
            return rate, amount, max_lev
    return tiers[-1][1], tiers[-1][2], tiers[-1][3]


def liquidation_price(side: int, entry: float, qty: float, margin: float, tiers=BTC_MM_TIERS) -> float:
    """Isolated, 1 vị thế, bỏ qua phí: long (qty*E - M - cum) / (qty*(1 - mmr)), short (qty*E + M + cum) / (qty*(1 + mmr))."""
    rate, cum, _ = mm_tier(qty * entry, tiers)
    if side > 0:
        return (qty * entry - margin - cum) / (qty * (1 - rate))
    return (qty * entry + margin + cum) / (qty * (1 + rate))


def _entry(E, i0, s, b, cfg):
    """(chỉ số nến vào, giá vào, maker, khớp giữa nến) hoặc None - giống strategy_h1._simulate."""
    o, h, l = E["o"], E["h"], E["l"]
    n5 = len(o)
    if cfg.entry_mode != "limit_pullback":
        return i0, o[i0], False, False
    pb_bars = max(1, cfg.pullback_minutes // 5)
    lim = o[i0] * math.exp(-s * cfg.pullback_mult * b)
    for k in range(i0, min(i0 + pb_bars, n5)):
        if (s > 0 and o[k] <= lim) or (s < 0 and o[k] >= lim):
            return k, o[k], True, False
        if (s > 0 and l[k] <= lim * (1 - cfg.fill_bp)) or (s < 0 and h[k] >= lim * (1 + cfg.fill_bp)):
            return k, lim, True, True
    ie = i0 + pb_bars
    return (ie, o[ie], False, False) if ie < n5 else None


def _exit(E, ie, s, tp, stop, stop_kind, intrabar, maxb):
    """(chỉ số nến thoát, giá thoát, loại) - SL / thanh lý trước, rồi TP; hết giờ ở close nến cuối."""
    o, h, l, c = E["o"], E["h"], E["l"], E["c"]
    n5 = len(o)
    k_end = min(n5, ie + maxb)
    for k in range(ie, k_end):
        if (s > 0 and l[k] <= stop) or (s < 0 and h[k] >= stop):
            gap = (s > 0 and o[k] < stop) or (s < 0 and o[k] > stop)
            return k, (o[k] if gap and stop_kind == "SL" else stop), stop_kind
        if intrabar and k == ie:
            continue
        if (s > 0 and h[k] >= tp) or (s < 0 and l[k] <= tp):
            return k, tp, "TP"
    return k_end - 1, c[k_end - 1], ("TIMEOUT" if ie + maxb <= n5 else "OPEN_END")


def run(D: st.Data, rows: pd.DataFrame, cfg: st.StrategyConfig, fp: FuturesParams,
        funding: pd.DataFrame | None = None) -> dict:
    """rows: các nến có tín hiệu (t_ns, side, b, p_touch, p_up), ví dụ evaluate(...)["signals"] lọc signal.
    Chạy tuần tự theo thời gian: luật danh mục -> vào lệnh -> khối lượng theo vốn đã chốt -> thanh lý / TP / SL."""
    E = st._exec_arrays(D)
    t5, o5 = E["t"], E["o"]
    rows = rows.sort_values("t_ns").reset_index(drop=True)
    fund_t = fund_r = np.array([])
    if fp.funding and funding is not None:
        f = pd.DataFrame({"t": funding.funding_time_ms.to_numpy(np.int64) * 10**6,
                          "r": pd.to_numeric(funding.funding_rate).to_numpy()}).sort_values("t")
        fund_t, fund_r = f.t.to_numpy(), f.r.to_numpy()
    maxb, gap_ns = cfg.max_hold_hours * 12, cfg.min_gap_bars * st.BAR_MIN * NS_MIN
    dec = rows.t_ns.to_numpy(np.int64) + st.BAR_MIN * NS_MIN
    start = np.searchsorted(t5, dec)
    done, last_sig = [], -(2**62)                 # done: lệnh đã nhận (exit_ns, ...)
    for j, r in enumerate(rows.itertuples(index=False)):
        t_sig = int(r.t_ns)
        if sum(1 for x in done if x["exit_ns"] >= t_sig) >= cfg.max_positions or t_sig - last_sig < gap_ns:
            continue
        i0 = int(start[j])
        if i0 >= len(t5) or t5[i0] - dec[j] > cfg.max_entry_delay_min * NS_MIN:
            continue
        s, b = int(r.side), float(r.b)
        en = _entry(E, i0, s, b, cfg)
        if en is None:
            continue
        ie, e, maker, intrabar = en
        entry_ns = int(t5[ie])
        realized = fp.capital + sum(x["net_usdt"] for x in done if x["exit_ns"] <= entry_ns)
        base = realized if fp.compound else fp.capital
        margin = fp.margin_frac * base
        notional = margin * fp.leverage
        qty = notional / e
        mmr, _, max_lev = mm_tier(notional, fp.tiers)
        tp, sl = e * math.exp(s * cfg.tp_mult * b), e * math.exp(-s * cfg.sl_mult * b)
        lq = liquidation_price(s, e, qty, margin, fp.tiers)
        isolated = fp.margin_mode == "isolated"
        liq_first = isolated and fp.liquidation and ((s > 0 and lq >= sl) or (s < 0 and lq <= sl))
        # cross: vốn còn lại nếu mọi lệnh đang mở (kể cả lệnh này) cùng chạm SL, trừ maintenance margin, % vốn
        opn = [x for x in done if x["exit_ns"] >= entry_ns] + [{"notional": notional, "sl_dist": abs(1 - sl / e), "mmr": mmr}]
        worst = realized - sum(x["notional"] * x["sl_dist"] for x in opn) - sum(x["notional"] * x["mmr"] for x in opn)
        used = sum(x["notional"] for x in opn) / fp.leverage
        xi, xpx, kind = _exit(E, ie, s, tp, lq if liq_first else sl, "LIQ" if liq_first else "SL", intrabar, maxb)
        if kind == "OPEN_END":
            break                                       # hết dữ liệu
        fee_in = notional * (fp.fee_maker if maker else fp.fee_taker)
        if kind == "LIQ":
            pnl, fee_out = -margin, 0.0
        else:
            pnl = qty * (xpx - e) * s
            fee_out = qty * xpx * (fp.fee_maker if (kind == "TP" and fp.tp_maker) else fp.fee_taker)
        a, z = np.searchsorted(fund_t, entry_ns, "right"), np.searchsorted(fund_t, int(t5[xi]) + 5 * NS_MIN, "left")
        fnd = float(sum(-s * qty * o5[min(np.searchsorted(t5, fund_t[k]), len(t5) - 1)] * fund_r[k] for k in range(a, z)))
        net = pnl - fee_in - fee_out + fnd
        done.append({"t_ns": t_sig, "side": s, "b": b, "p_touch": float(r.p_touch), "p_up": float(r.p_up),
                     "entry_ns": entry_ns, "entry": float(e), "maker_in": maker, "tp": tp, "sl": sl,
                     "liq_price": lq, "liq_before_sl": bool((s > 0 and lq >= sl) or (s < 0 and lq <= sl)), "exit_ns": int(t5[xi]), "exit": float(xpx),
                     "kind": kind, "R_gross": s * math.log(xpx / e) / (cfg.sl_mult * b),
                     "equity_base": base, "margin": margin, "notional": notional, "qty_btc": qty, "mmr": mmr,
                     "lev_ok": fp.leverage <= max_lev, "sl_dist": abs(1 - sl / e), "open_positions": len(opn),
                     "margin_used_pct": used / realized * 100, "cross_buffer_pct": worst / realized * 100,
                     "pnl_usdt": pnl, "fee_usdt": fee_in + fee_out,
                     "funding_usdt": fnd, "net_usdt": net, "net_pct_equity": net / base * 100,
                     "roe_pct": net / margin * 100})
        last_sig = t_sig
    T = pd.DataFrame(done)
    for a_, b_ in (("signal_time", "t_ns"), ("entry_time", "entry_ns"), ("exit_time", "exit_ns")):
        T[a_] = pd.to_datetime(T[b_], utc=True)
    C = T.sort_values(["exit_ns", "entry_ns"])
    curve = pd.DataFrame({"time": C.exit_time.to_numpy(), "equity": fp.capital + C.net_usdt.cumsum().to_numpy()})
    T.loc[C.index, "equity_after"] = curve.equity.to_numpy()             # vốn sau mỗi lần thoát (theo thứ tự thoát)
    return {"trades": T, "equity": curve, "summary": summarize(T, curve, fp), "capital": fp.capital}


def summarize(T: pd.DataFrame, curve: pd.DataFrame, fp: FuturesParams) -> dict:
    eq = np.r_[fp.capital, curve.equity.to_numpy()]
    peak = np.maximum.accumulate(eq)
    days = max((curve.time.iloc[-1] - T.entry_time.min()).days, 1)
    out = {"lệnh": len(T), "vốn đầu": fp.capital, "vốn cuối": float(eq[-1]), "lãi %": (eq[-1] / fp.capital - 1) * 100,
           "lãi %/năm (CAGR)": ((max(eq[-1], 0) / fp.capital) ** (365 / days) - 1) * 100,
           "DD max %": float(((peak - eq) / peak).max() * 100), "DD max USDT": float((peak - eq).max()),
           "thắng": float((T.net_usdt > 0).mean()), "TB lãi/lệnh % vốn": float(T.net_pct_equity.mean()),
           "TB ROE/lệnh %": float(T.roe_pct.mean()), "tổng PnL giá": float(T.pnl_usdt.sum()),
           "tổng phí": float(T.fee_usdt.sum()), "tổng funding": float(T.funding_usdt.sum()),
           "notional TB": float(T.notional.mean()), "lệnh có thanh lý isolated gần hơn SL %": float(T.liq_before_sl.mean() * 100),
           "margin dùng max % vốn": float(T.margin_used_pct.max()),
           "cross: vốn còn lại xấu nhất % (mọi lệnh mở cùng chạm SL)": float(T.cross_buffer_pct.min())}
    for k in ("TP", "SL", "LIQ", "TIMEOUT"):
        out[f"{k} %"] = float((T.kind == k).mean() * 100)
    out["lệnh vượt đòn bẩy tối đa của bậc"] = int((~T.lev_ok).sum())
    return out


def _count(kind):
    return lambda k: int((k == kind).sum())


def by_period(res: dict, freq: str = "Y") -> pd.DataFrame:
    T = res["trades"]
    t = pd.DatetimeIndex(T.exit_time).tz_localize(None)
    key = t.year if freq == "Y" else t.to_period("Q").astype(str)
    g = T.groupby(key)
    out = pd.DataFrame({"lệnh": g.size(), "thắng": g.net_usdt.apply(lambda x: (x > 0).mean()),
                        "net USDT": g.net_usdt.sum(), "phí USDT": g.fee_usdt.sum(), "funding USDT": g.funding_usdt.sum(),
                        "TP": g.kind.apply(_count("TP")), "SL": g.kind.apply(_count("SL")),
                        "LIQ": g.kind.apply(_count("LIQ")), "hết giờ": g.kind.apply(_count("TIMEOUT")),
                        "b TB %": g.b.mean() * 100})
    e = res["equity"]
    out["lãi % vốn"] = st.period_return(e.assign(equity=e.equity / res["capital"]), freq).reindex(out.index)
    out.index.name = "năm" if freq == "Y" else "quý"
    return out.round(2)


def by_b(res: dict, edges=(0.0, 0.012, 0.016, 0.022, 1.0)) -> pd.DataFrame:
    T = res["trades"]
    cut = pd.cut(T.b, list(edges), labels=[f"{a * 100:.1f}-{b * 100:.1f}%" for a, b in zip(edges[:-1], edges[1:])])
    g = T.groupby(cut, observed=True)
    return pd.DataFrame({"lệnh": g.size(), "thắng": g.net_usdt.apply(lambda x: (x > 0).mean()),
                         "TB % vốn/lệnh": g.net_pct_equity.mean(), "net USDT": g.net_usdt.sum(),
                         "LIQ %": g.kind.apply(lambda k: (k == "LIQ").mean() * 100),
                         "SL %": g.kind.apply(lambda k: (k == "SL").mean() * 100),
                         "TP %": g.kind.apply(lambda k: (k == "TP").mean() * 100)}).round(3)
