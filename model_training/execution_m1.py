"""Bộ thực thi đơn giản cho tín hiệu modelv2 (5m), khớp lệnh trên nến 1m.

Quy tắc (không ứng biến):
  1. Chỉ vào lệnh khi đủ confident. Độ confident = R kỳ vọng SAU PHÍ của model cho hướng tốt hơn:
         score = max(e_long_{rr}r, e_short_{rr}r) - fee_r,   fee_r = phí khứ hồi / barrier
     Ngưỡng: "rolling" = score nằm trong top `top_frac` của `window_days` ngày trước đó (nhân quả, chỉ dùng
     quá khứ) và score > min_score; hoặc "abs" = score >= min_score.
  2. Vào lệnh thị trường ở giá mở của nến 1m đầu tiên sau khi nến 5m tín hiệu đóng.
  3. SL = entry * (1 - side * sl_mult * b), TP = entry * (1 + side * tp_mult * b), b = barrier_pct của tín hiệu.
     Đặt một lần lúc vào lệnh, KHÔNG thay đổi giữa chừng.
  4. Giữ tối đa max_hold_min (240 phút = 4h), hết giờ đóng ở giá đóng cửa nến 1m cuối.
  5. Các lệnh độc lập: mỗi tín hiệu đủ điều kiện là một lệnh riêng, không bị tín hiệu sau ảnh hưởng (có thể
     chồng nhau; max_open giới hạn số lệnh mở cùng lúc nếu muốn, mặc định không giới hạn).
Nến 1m chạm cả SL và TP -> tính SL. Giá mở gap qua SL/TP -> khớp ở giá mở.
R = lợi nhuận ròng / (sl_mult * b): lỗ đủ SL = -1R - phí.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

M1, M5 = 60_000, 300_000


@dataclass
class ExecConfig:
    rr: int = 1                    # mẫu R kỳ vọng dùng để chấm điểm (1 -> e_*_1r, 2 -> e_*_2r)
    tp_mult: float = 1.0           # TP = tp_mult * barrier
    sl_mult: float = 1.0           # SL = sl_mult * barrier
    max_hold_min: int = 240
    mode: str = "rolling"          # "rolling" | "abs"
    top_frac: float = 0.02         # rolling: top 2% score của window_days ngày trước
    min_score: float = 0.0         # luôn đòi score (R kỳ vọng sau phí) > min_score
    window_days: float = 60.0
    min_history_days: float = 20.0
    fee_roundtrip: float = 0.001   # taker 0.04% x2 + trượt giá 0.01% x2
    max_open: int | None = None    # None = lệnh hoàn toàn độc lập

    @property
    def label(self) -> str:
        sel = f"top{self.top_frac:g}" if self.mode == "rolling" else "abs"
        return f"{self.rr}R-score {sel}>{self.min_score:g} TP{self.tp_mult:g}/SL{self.sl_mult:g}"


# --------------------------------------------------------------------------- #
# Tín hiệu
# --------------------------------------------------------------------------- #
SIG_COLS = ("timestamp_ns", "close", "barrier_pct", "p_touch", "p_up_touch", "e_long_1r", "e_short_1r",
            "e_long_2r", "e_short_2r", "label_r_long_1r", "label_r_short_1r")


def load_signals(pred_dir: str | Path) -> pd.DataFrame:
    """Dự báo OOS của modelv2 (predictions_<năm>.npz). decision_ms = giờ đóng nến 5m tín hiệu."""
    parts = []
    for f in sorted(Path(pred_dir).glob("predictions_*.npz")):
        z = np.load(f)
        parts.append(pd.DataFrame({k: z[k] for k in SIG_COLS if k in z.files}))
    sig = pd.concat(parts, ignore_index=True).drop_duplicates("timestamp_ns").sort_values("timestamp_ns")
    sig["decision_ms"] = sig["timestamp_ns"] // 1_000_000 + M5
    return sig.reset_index(drop=True)


def append_live(sig: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Nối các dòng của modelv2.predict_frame (nến sau file OOS) vào chuỗi tín hiệu."""
    f = frame.copy()
    f["timestamp_ns"] = pd.to_datetime(f["timestamp"], utc=True).astype("int64")
    f["decision_ms"] = f["timestamp_ns"] // 1_000_000 + M5
    f = f[f["decision_ms"] > sig["decision_ms"].max()]
    cols = [c for c in sig.columns if c in f.columns]
    return pd.concat([sig, f[cols]], ignore_index=True).sort_values("decision_ms").reset_index(drop=True)


def score_signals(sig: pd.DataFrame, cfg: ExecConfig) -> pd.DataFrame:
    """Thêm side, score (R kỳ vọng sau phí), threshold và cờ enter (nhân quả)."""
    out = sig.copy()
    el, es = out[f"e_long_{cfg.rr}r"], out[f"e_short_{cfg.rr}r"]
    out["side"] = np.where(el >= es, 1, -1)
    out["fee_r"] = cfg.fee_roundtrip / (cfg.sl_mult * out["barrier_pct"])
    out["score"] = np.maximum(el, es) - out["fee_r"]
    if cfg.mode == "rolling":
        step = float(np.median(np.diff(out["decision_ms"].to_numpy()[:5000])))
        rpd = 86_400_000 / step
        thr = out["score"].rolling(int(cfg.window_days * rpd), min_periods=int(cfg.min_history_days * rpd)) \
            .quantile(1 - cfg.top_frac).shift(1)
        out["threshold"] = np.maximum(thr, cfg.min_score)
    else:
        out["threshold"] = cfg.min_score
    out["enter"] = (out["score"] > out["threshold"]).fillna(False).astype(bool)
    out["time"] = pd.to_datetime(out["decision_ms"], unit="ms", utc=True)
    return out


def live_decision(history: pd.DataFrame, row: dict | pd.Series, cfg: ExecConfig) -> dict:
    """Quyết định cho MỘT nến mới (dùng trong pipeline): history = các tín hiệu trước đó (cột e_*_{rr}r,
    barrier_pct, decision_ms). Trả về side, score, threshold, enter."""
    el, es = float(row[f"e_long_{cfg.rr}r"]), float(row[f"e_short_{cfg.rr}r"])
    fee_r = cfg.fee_roundtrip / (cfg.sl_mult * float(row["barrier_pct"]))
    score = max(el, es) - fee_r
    thr = cfg.min_score
    if cfg.mode == "rolling":
        t = int(row["decision_ms"])
        h = history[(history["decision_ms"] < t) & (history["decision_ms"] >= t - int(cfg.window_days * 86_400_000))]
        if len(h) < cfg.min_history_days * 288:
            return {"side": 1 if el >= es else -1, "score": score, "threshold": None, "enter": False,
                    "reason": "warming_up"}
        hs = np.maximum(h[f"e_long_{cfg.rr}r"], h[f"e_short_{cfg.rr}r"]) - cfg.fee_roundtrip / (cfg.sl_mult * h["barrier_pct"])
        thr = max(float(hs.quantile(1 - cfg.top_frac)), cfg.min_score)
    return {"side": 1 if el >= es else -1, "score": score, "threshold": thr, "enter": bool(score > thr)}


# --------------------------------------------------------------------------- #
# Mô phỏng trên nến 1m
# --------------------------------------------------------------------------- #
def simulate(m1: dict, scored: pd.DataFrame, cfg: ExecConfig) -> pd.DataFrame:
    """m1: dict t (open ms, liên tục), o, h, l, c (numpy). Mỗi tín hiệu enter -> một lệnh độc lập."""
    t, o, h, l, c = m1["t"], m1["o"], m1["h"], m1["l"], m1["c"]
    sig = scored[scored["enter"]]
    idx = np.searchsorted(t, sig["decision_ms"].to_numpy(np.int64))
    valid = (idx < len(t)) & (t[np.minimum(idx, len(t) - 1)] == sig["decision_ms"].to_numpy())
    hold = int(cfg.max_hold_min * 60_000 // M1)
    rows, open_until = [], []
    for i0, r in zip(idx[valid], sig[valid].itertuples(index=False)):
        end = min(i0 + hold, len(t))
        if end - i0 < hold:                               # không đủ dữ liệu để theo dõi hết lệnh
            continue
        if cfg.max_open is not None:
            open_until = [x for x in open_until if x > i0]
            if len(open_until) >= cfg.max_open:
                continue
        s, b, entry = int(r.side), float(r.barrier_pct), float(o[i0])
        sl = entry * (1 - s * cfg.sl_mult * b)
        tp = entry * (1 + s * cfg.tp_mult * b)
        hh, ll, oo = h[i0:end], l[i0:end], o[i0:end]
        if s == 1:
            hit_sl, hit_tp = ll <= sl, hh >= tp
        else:
            hit_sl, hit_tp = hh >= sl, ll <= tp
        k_sl = int(np.argmax(hit_sl)) if hit_sl.any() else hold
        k_tp = int(np.argmax(hit_tp)) if hit_tp.any() else hold
        if k_sl == hold and k_tp == hold:
            k, reason, px = hold - 1, "timeout", float(c[end - 1])
        elif k_sl <= k_tp:                                # cùng nến -> SL (bảo thủ)
            k, reason = k_sl, "sl"
            px = float(oo[k]) if k > 0 and (oo[k] - sl) * s <= 0 else sl      # gap qua SL -> giá mở
        else:
            k, reason = k_tp, "tp"
            px = float(oo[k]) if k > 0 and (oo[k] - tp) * s >= 0 else tp
        if cfg.max_open is not None:
            open_until.append(i0 + k + 1)
        gross = s * (px / entry - 1)
        net = gross - cfg.fee_roundtrip
        risk = cfg.sl_mult * b
        rows.append({"signal_ms": int(r.decision_ms), "entry_ms": int(t[i0]), "exit_ms": int(t[i0 + k]) + M1,
                     "side": s, "entry": entry, "sl": sl, "tp": tp, "exit": px, "reason": reason,
                     "minutes": k + 1, "barrier_pct": b, "score": float(r.score), "threshold": float(r.threshold),
                     "p_touch": float(r.p_touch), "p_up_touch": float(r.p_up_touch),
                     "ret_pct": net * 100, "R": net / risk, "R_gross": gross / risk})
    tr = pd.DataFrame(rows)
    if len(tr):
        for k in ("signal_ms", "entry_ms", "exit_ms"):
            tr[k.replace("_ms", "_time")] = pd.to_datetime(tr[k], unit="ms", utc=True)
    return tr


def concurrency(tr: pd.DataFrame) -> int:
    """Số lệnh mở cùng lúc lớn nhất."""
    if not len(tr):
        return 0
    ev = np.concatenate([np.c_[tr["entry_ms"], np.ones(len(tr))], np.c_[tr["exit_ms"], -np.ones(len(tr))]])
    ev = ev[np.lexsort((ev[:, 1], ev[:, 0]))]
    return int(np.cumsum(ev[:, 1]).max())


# --------------------------------------------------------------------------- #
# Đánh giá
# --------------------------------------------------------------------------- #
def _dd(cum):
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))
    return float((peak[1:] - cum).max()) if len(cum) else 0.0


def block_ci(tr: pd.DataFrame, n_boot: int = 1000, seed: int = 0, block_days: int = 7) -> tuple[float, float]:
    """KTC 90% của R trung bình, bootstrap theo khối tuần (lệnh chồng nhau / gần nhau tương quan)."""
    if len(tr) < 10:
        return (float("nan"), float("nan"))
    blk = (tr["entry_ms"] // (block_days * 86_400_000)).to_numpy()
    keys, inv = np.unique(blk, return_inverse=True)
    sums = np.bincount(inv, weights=tr["R"].to_numpy())
    cnts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    means = sums[pick].sum(1) / cnts[pick].sum(1)
    return float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))


def performance(tr: pd.DataFrame) -> dict:
    if tr is None or not len(tr):
        return {"trades": 0}
    r = tr["R"].to_numpy()
    days = max((tr["exit_ms"].max() - tr["entry_ms"].min()) / 86_400_000, 1.0)
    lo, hi = block_ci(tr)
    win, loss = r[r > 0], r[r <= 0]
    rs = tr["reason"].value_counts(normalize=True)
    return {"trades": len(r), "per_month": len(r) / days * 30.4, "win_rate": float((r > 0).mean()),
            "avg_R": float(r.mean()), "ci90_lo": lo, "ci90_hi": hi, "avg_R_gross": float(tr["R_gross"].mean()),
            "total_R": float(r.sum()), "profit_factor": float(win.sum() / -loss.sum()) if loss.sum() < 0 else float("inf"),
            "max_dd_R": _dd(np.cumsum(r)), "max_open": concurrency(tr), "avg_minutes": float(tr["minutes"].mean()),
            "long_share": float((tr["side"] == 1).mean()),
            "tp": float(rs.get("tp", 0)), "sl": float(rs.get("sl", 0)), "timeout": float(rs.get("timeout", 0))}


def by_period(tr: pd.DataFrame, freq: str = "Y") -> pd.DataFrame:
    if not len(tr):
        return pd.DataFrame()
    key = tr["entry_time"].dt.tz_localize(None).dt.to_period(freq).astype(str)
    return pd.DataFrame([{"period": k, **performance(g)} for k, g in tr.groupby(key)]).set_index("period")


def calibration(scored: pd.DataFrame, tr: pd.DataFrame, bins=(0, 0.02, 0.05, 0.1, 0.15, 1)) -> pd.DataFrame:
    """R kỳ vọng của model (score) so với R thực tế theo nhóm score - model có quá tự tin không."""
    if not len(tr):
        return pd.DataFrame()
    g = pd.cut(tr["score"], list(bins))
    return tr.groupby(g, observed=True).agg(trades=("R", "size"), predicted=("score", "mean"), realized=("R", "mean"))


def sweep(m1: dict, sig: pd.DataFrame, configs: list[ExecConfig], split_ms: int) -> pd.DataFrame:
    """Chạy nhiều cấu hình; tách kết quả trước / sau split_ms (chọn cấu hình trên phần trước)."""
    rows = []
    for cfg in configs:
        tr = simulate(m1, score_signals(sig, cfg), cfg)
        a = performance(tr[tr["entry_ms"] < split_ms]) if len(tr) else {"trades": 0}
        b = performance(tr[tr["entry_ms"] >= split_ms]) if len(tr) else {"trades": 0}
        rows.append({"config": cfg.label, **{k: v for k, v in asdict(cfg).items() if k in ("rr", "top_frac", "min_score", "tp_mult")},
                     "sel_trades": a.get("trades", 0), "sel_avg_R": a.get("avg_R"), "sel_ci_lo": a.get("ci90_lo"),
                     "hold_trades": b.get("trades", 0), "hold_avg_R": b.get("avg_R"), "hold_ci_lo": b.get("ci90_lo"),
                     "hold_ci_hi": b.get("ci90_hi"), "hold_per_month": b.get("per_month")})
    return pd.DataFrame(rows)
