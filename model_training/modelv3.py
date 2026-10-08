"""modelv3 - model GIAO DỊCH theo sự kiện bất thường (BTCUSDT, quyết định ở nến 5m, khớp lệnh trên nến 1m).

Khác v2: không dự báo hướng trên mọi nến, mà chỉ học trên các nến SỰ KIỆN và học trực tiếp KẾT QUẢ GIAO DỊCH.

Sự kiện (tính tại lúc nến 5m đóng, chỉ dùng quá khứ):
  E1  nến 5m bất thường: range >= 0.5% giá và >= 3 lần trung vị range 48 nến trước.
  E2  3 nến 5m gần nhất biến động khác hẳn phần còn lại: range TB 3 nến / range TB 45 nến trước >= 2.5.
  E3  bất thường nến 1m trong nến 5m vừa đóng: một nến 1m có range VÀ volume >= 6 lần trung vị 240 phút trước.
  (Kiểm chứng 2020-2026: sau E1 giá đi >= 1% trong 4h ở 78% trường hợp, so với 50% ở nến thường.)

Giao dịch (nhãn), cho cả LONG và SHORT của mỗi sự kiện:
  vào lệnh market ở giá mở nến 1m đầu tiên sau khi nến 5m đóng;
  SL = d,  TP = tp_mult * d  (mặc định tp_mult = 1: TP = SL),  d = clip(sl_vol_mult * sigma_1h, sl_min, sl_max);
  giữ tối đa 4h; nến 1m chạm cả SL và TP -> SL.  Nhãn 3 lớp: 0 = SL, 1 = TP, 2 = hết giờ (+ R thực tế).

Model: XGBoost multi-class P(TP), P(SL), P(hết giờ) cho một cặp (sự kiện, phía); feature đối xứng long/short
(feature có hướng được đổi dấu / đổi cặp cho phía short) nên một model dùng chung cho hai phía.
R kỳ vọng sau phí  EV = P(TP)*tp_mult - P(SL) + P(hết giờ)*R_timeout - phí/d.  Vào lệnh khi EV >= ngưỡng; ngưỡng
chọn trên năm validation. Walk-forward theo năm: train < năm Y-1, validation = Y-1, test = Y.

Feature (tất cả tại lúc nến 5m đóng):
  sự kiện       cờ E1/E2/E3, độ lớn bất thường, thân/râu/vị trí đóng cửa, volume, taker của nến sự kiện
  nến 1m        bất thường lớn nhất trong nến, hướng nến 1m lớn nhất, taker 5 phút, biến động 1m 30' vs 4h
  tiền chuyển động  nén biến động trước sự kiện, tỉ lệ biến động 12/288, 48/2880, volume tích lũy
  xu hướng      lợi nhuận 1..288 nến chuẩn hóa theo sigma, EMA, RSI, vị trí trong range 48/288
  pivot         zigzag ĐÃ XÁC NHẬN ngưỡng 1% (giá đảo chiều >= 1% từ cực trị): sóng hiện tại, khoảng cách tới đỉnh/đáy pivot,
                cấu trúc đỉnh/đáy cao hơn - thấp hơn, phá đỉnh/đáy, tuổi pivot
  dòng tiền     taker imbalance 3/12/48 nến, số lệnh, cỡ lệnh, funding
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not a or not callable(a[0]) else a[0]

M1, M5 = 60_000, 300_000
STRATEGY = "event_trade_v3"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # sự kiện
    e1_min_range: float = 0.005
    e1_rel: float = 3.0
    e2_ratio: float = 2.5
    e3_rel: float = 6.0
    e3_vol_rel: float = 6.0
    events: tuple = ("E1", "E2", "E3")
    # pivot (zigzag đã xác nhận)
    pivot_thresholds: tuple = (0.01,)
    # giao dịch
    tp_mult: float = 1.0
    sl_vol_mult: float = 2.0
    sl_min: float = 0.006
    sl_max: float = 0.03
    hold_min: int = 240
    fee_roundtrip: float = 0.001
    # walk-forward
    eval_years: tuple = (2022, 2023, 2024, 2025, 2026)
    first_train_year: int = 2020
    val_es_frac: float = 0.6
    recency_halflife_days: float = 720.0
    # chọn ngưỡng trên validation
    thr_grid: tuple = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4)
    min_val_trades: int = 30
    # xgboost
    xgb: dict = field(default_factory=lambda: dict(n_estimators=2000, learning_rate=0.02, max_depth=3,
                                                   min_child_weight=30.0, subsample=0.7, colsample_bytree=0.6,
                                                   reg_lambda=10.0, reg_alpha=0.1, early_stopping_rounds=150))
    device: str = "cpu"
    seed: int = 42


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
KLINE_MAP = {"open_time_ms": "t", "open": "o", "high": "h", "low": "l", "close": "c", "volume": "v", "trades": "n",
             "taker_buy_base_asset_volume": "tb"}


def load_m1_csv(path, start=None) -> pd.DataFrame:
    """1m CSV của collect_historical (cần cột trades, taker_buy_base_asset_volume)."""
    df = pd.read_csv(path, usecols=lambda c: c in KLINE_MAP)
    df = df.rename(columns=KLINE_MAP)
    if start is not None:
        df = df[df["t"] >= pd.Timestamp(start, tz="UTC").value // 1_000_000]
    return standardize_m1(df)


def standardize_m1(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=KLINE_MAP)
    missing = {"t", "o", "h", "l", "c", "v"} - set(df.columns)
    if missing:
        raise ValueError(f"missing 1m columns: {missing}")
    if "n" not in df:
        df["n"] = np.nan
    if "tb" not in df:
        df["tb"] = np.nan
    df = df[["t", "o", "h", "l", "c", "v", "n", "tb"]].astype({"t": "int64"})
    for k in "ohlcvn":
        df[k] = df[k].astype(float)
    df["tb"] = df["tb"].astype(float)
    return df.drop_duplicates("t").sort_values("t").reset_index(drop=True)


def to_5m(m1: pd.DataFrame) -> pd.DataFrame:
    """Nến 5m dựng từ nến 1m (chỉ giữ nến đủ 5 phút)."""
    g = m1["t"] // M5
    b = m1.groupby(g).agg(t=("t", "first"), o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"),
                          v=("v", "sum"), n=("n", "sum"), tb=("tb", "sum"), cnt=("t", "size"))
    b = b[b["cnt"] == 5].drop(columns="cnt")
    b["t"] = (b.index * M5).astype("int64")
    return b.reset_index(drop=True)


def funding_table(path_or_df) -> pd.DataFrame | None:
    if path_or_df is None:
        return None
    f = pd.read_csv(path_or_df) if not isinstance(path_or_df, pd.DataFrame) else path_or_df.copy()
    if "fundingTime" in f:
        f = f.rename(columns={"fundingTime": "funding_time_ms", "fundingRate": "funding_rate"})
    f = f[["funding_time_ms", "funding_rate"]].astype({"funding_time_ms": "int64", "funding_rate": float})
    return f.drop_duplicates("funding_time_ms").sort_values("funding_time_ms").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Zigzag pivots (causal)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _zigzag(h, l, r):
    """Zigzag xác nhận theo thời gian thực. Trả về cho mỗi nến (sau khi nến đóng):
    leg_dir (+1 đang sóng lên từ đáy đã xác nhận, -1 ngược lại, 0 chưa có), giá pivot cuối, tuổi pivot cuối
    (số nến từ lúc pivot hình thành), 2 đỉnh và 2 đáy đã xác nhận gần nhất, cực trị của sóng hiện tại."""
    n = len(h)
    leg = np.zeros(n)
    last_p = np.full(n, np.nan)
    age = np.full(n, np.nan)
    ph1 = np.full(n, np.nan)
    ph2 = np.full(n, np.nan)
    pl1 = np.full(n, np.nan)
    pl2 = np.full(n, np.nan)
    ext = np.full(n, np.nan)
    d = 0
    ext_p = h[0]
    ext_i = 0
    lo_p = l[0]
    lo_i = 0
    hi_p = h[0]
    hi_i = 0
    cph1 = np.nan
    cph2 = np.nan
    cpl1 = np.nan
    cpl2 = np.nan
    cur_p = np.nan
    cur_i = -1
    for i in range(n):
        if d == 0:
            if h[i] > hi_p:
                hi_p, hi_i = h[i], i
            if l[i] < lo_p:
                lo_p, lo_i = l[i], i
            if h[i] >= lo_p * (1 + r) and lo_i < i:
                # first confirmed low
                cpl2, cpl1 = cpl1, lo_p
                cur_p, cur_i = lo_p, lo_i
                d = 1
                ext_p, ext_i = h[i], i
                for k in range(lo_i, i + 1):
                    if h[k] > ext_p:
                        ext_p, ext_i = h[k], k
            elif l[i] <= hi_p * (1 - r) and hi_i < i:
                cph2, cph1 = cph1, hi_p
                cur_p, cur_i = hi_p, hi_i
                d = -1
                ext_p, ext_i = l[i], i
                for k in range(hi_i, i + 1):
                    if l[k] < ext_p:
                        ext_p, ext_i = l[k], k
        elif d == 1:
            if h[i] > ext_p:
                ext_p, ext_i = h[i], i
            elif l[i] <= ext_p * (1 - r):
                cph2, cph1 = cph1, ext_p
                cur_p, cur_i = ext_p, ext_i
                d = -1
                ext_p, ext_i = l[i], i
                for k in range(cur_i + 1, i + 1):
                    if l[k] < ext_p:
                        ext_p, ext_i = l[k], k
        else:
            if l[i] < ext_p:
                ext_p, ext_i = l[i], i
            elif h[i] >= ext_p * (1 + r):
                cpl2, cpl1 = cpl1, ext_p
                cur_p, cur_i = ext_p, ext_i
                d = 1
                ext_p, ext_i = h[i], i
                for k in range(cur_i + 1, i + 1):
                    if h[k] > ext_p:
                        ext_p, ext_i = h[k], k
        leg[i] = d
        last_p[i] = cur_p
        age[i] = i - cur_i if cur_i >= 0 else np.nan
        ph1[i], ph2[i], pl1[i], pl2[i] = cph1, cph2, cpl1, cpl2
        ext[i] = ext_p if d != 0 else np.nan
    return leg, last_p, age, ph1, ph2, pl1, pl2, ext


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
@dataclass
class FeatureSpec:
    odd: list = field(default_factory=list)       # đổi dấu cho short
    even: list = field(default_factory=list)      # giữ nguyên
    swap: list = field(default_factory=list)      # cặp (a, b): short dùng (b, a)
    swapneg: list = field(default_factory=list)   # cặp (a, b): short dùng (-b, -a)

    @property
    def names(self):
        out = list(self.odd) + list(self.even)
        for a, b in list(self.swap) + list(self.swapneg):
            out += [a, b]
        return out


def _rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def compute_bars(m1: pd.DataFrame, funding: pd.DataFrame | None, cfg: Config) -> tuple[pd.DataFrame, FeatureSpec]:
    """Mọi feature + cờ sự kiện cho TẤT CẢ nến 5m (dòng i = trạng thái lúc nến i đóng)."""
    m1 = standardize_m1(m1)
    b = to_5m(m1)
    f = pd.DataFrame({"t": b["t"], "close": b["c"]})
    sp = FeatureSpec()
    c, o, h, l, v = b["c"], b["o"], b["h"], b["l"], b["v"]
    eps = 1e-12
    lr = np.log(c).diff()
    rng = (h - l) / o
    sig48 = lr.rolling(48).std()
    sig288 = lr.rolling(288).std()
    f["sigma_1h"] = sig48 * math.sqrt(12)

    # --- sự kiện ---------------------------------------------------------- #
    med48 = rng.shift(1).rolling(48).median()
    rel = rng / med48
    r3 = rng.rolling(3).mean()
    rest = rng.shift(3).rolling(45).mean()
    ratio3 = r3 / rest
    m = m1.copy()
    m["rng"] = (m["h"] - m["l"]) / m["o"]
    m["rel1"] = m["rng"] / m["rng"].shift(1).rolling(240).median()
    m["vrel1"] = m["v"] / m["v"].shift(1).rolling(240).median()
    m["dir1"] = np.sign(m["c"] - m["o"])
    m["imb1"] = 2 * m["tb"] - m["v"]
    m["lr1"] = np.log(m["c"]).diff()
    m["vol30"] = m["lr1"].rolling(30).std()
    m["vol240"] = m["lr1"].rolling(240).std()
    m["g"] = m["t"] // M5
    m["score1"] = np.minimum(m["rel1"], m["vrel1"])
    agg = pd.DataFrame({
        "m1_rel": m.groupby("g")["rel1"].max(),
        "m1_vrel": m.groupby("g")["vrel1"].max(),
        "m1_imb": m.groupby("g")["imb1"].sum() / m.groupby("g")["v"].sum().replace(0, np.nan),
        "m1_vol_ratio": m.groupby("g")["vol30"].last() / m.groupby("g")["vol240"].last(),
        "m1_last_ret": m.groupby("g")["lr1"].last(),
    })
    spike = m.dropna(subset=["score1"]).sort_values(["g", "score1"], kind="stable").groupby("g").tail(1).set_index("g")
    agg["m1_spike_dir"] = spike["dir1"]
    agg["m1_spike_min"] = (spike["t"] // M1 % 5).astype(float)
    agg = agg.reindex(b["t"] // M5)
    agg.index = b.index

    f["E1"] = ((rng >= cfg.e1_min_range) & (rel >= cfg.e1_rel)).astype(float)
    f["E2"] = (ratio3 >= cfg.e2_ratio).astype(float)
    f["E3"] = ((agg["m1_rel"] >= cfg.e3_rel) & (agg["m1_vrel"] >= cfg.e3_vol_rel)).astype(float)
    f["is_event"] = f[list(cfg.events)].max(axis=1) > 0
    f["E1_iso"] = ((f["E1"] > 0) & (rng.shift(1).rolling(12).max() < cfg.e1_min_range)).astype(float)
    sp.even += ["E1", "E2", "E3", "E1_iso"]

    # --- nến sự kiện ------------------------------------------------------ #
    hl = (h - l).replace(0, np.nan)
    f["ev_rng"] = rng
    f["ev_rel"] = np.log(rel.clip(lower=eps))
    f["ev_ratio3"] = np.log(ratio3.clip(lower=eps))
    f["ev_body"] = (c - o) / hl
    f["ev_clv"] = ((c - l) - (h - c)) / hl
    up_w = (h - np.maximum(o, c)) / hl
    lo_w = (np.minimum(o, c) - l) / hl
    f["ev_wick_tot"] = up_w + lo_w
    f["ev_wick_bias"] = lo_w - up_w
    vmed = v.shift(1).rolling(288).median()
    f["ev_vrel"] = np.log((v / vmed).clip(lower=eps))
    f["ev_imb"] = (2 * b["tb"] - v) / v.replace(0, np.nan)
    f["ev_body3"] = (c - o.shift(2)) / (h.rolling(3).max() - l.rolling(3).min()).replace(0, np.nan)
    sp.odd += ["ev_body", "ev_clv", "ev_wick_bias", "ev_imb", "ev_body3"]
    sp.even += ["ev_rng", "ev_rel", "ev_ratio3", "ev_wick_tot", "ev_vrel"]

    # --- 1m -------------------------------------------------------------- #
    f["m1_rel"] = np.log(agg["m1_rel"].clip(lower=eps))
    f["m1_vrel"] = np.log(agg["m1_vrel"].clip(lower=eps))
    f["m1_imb"] = agg["m1_imb"]
    f["m1_spike_dir"] = agg["m1_spike_dir"]
    f["m1_spike_min"] = agg["m1_spike_min"]
    f["m1_vol_ratio"] = np.log(agg["m1_vol_ratio"].clip(lower=eps))
    f["m1_last_ret"] = agg["m1_last_ret"] / (sig48 / math.sqrt(5))
    sp.odd += ["m1_imb", "m1_spike_dir", "m1_last_ret"]
    sp.even += ["m1_rel", "m1_vrel", "m1_spike_min", "m1_vol_ratio"]

    # --- tiền chuyển động: nén / bùng nổ --------------------------------- #
    f["vr_12_288"] = np.log((lr.rolling(12).std() / sig288).clip(lower=eps))
    f["vr_48_2880"] = np.log((sig48 / lr.rolling(2880, min_periods=1440).std()).clip(lower=eps))
    f["squeeze_24"] = np.log(ratio3.shift(3).rolling(24).min().clip(lower=eps))
    f["rng_trend"] = np.log((rng.rolling(12).mean() / rng.shift(12).rolling(36).mean()).clip(lower=eps))
    f["vol_build"] = np.log((v.rolling(12).sum() / v.shift(12).rolling(276).mean() / 12).clip(lower=eps))
    f["sigma_lvl"] = np.log(f["sigma_1h"].clip(lower=eps))
    sp.even += ["vr_12_288", "vr_48_2880", "squeeze_24", "rng_trend", "vol_build", "sigma_lvl"]

    # --- xu hướng -------------------------------------------------------- #
    for k in (1, 3, 6, 12, 48, 288):
        f[f"ret_{k}"] = np.log(c / c.shift(k)) / (sig48 * math.sqrt(k))
        sp.odd.append(f"ret_{k}")
    for a, bb in ((12, 48), (48, 288)):
        f[f"ema_{a}_{bb}"] = (c.ewm(span=a, adjust=False).mean() - c.ewm(span=bb, adjust=False).mean()) / c / sig48
        sp.odd.append(f"ema_{a}_{bb}")
    f["rsi14"] = (_rsi(c, 14) - 50) / 50
    f["rsi48"] = (_rsi(c, 48) - 50) / 50
    sp.odd += ["rsi14", "rsi48"]
    for w in (48, 288):
        hh, ll = h.rolling(w).max(), l.rolling(w).min()
        f[f"pos_{w}"] = (c - (hh + ll) / 2) / (hh - ll).replace(0, np.nan)
        f[f"room_up_{w}"] = np.log(hh / c) / sig48
        f[f"room_dn_{w}"] = np.log(c / ll) / sig48
        sp.odd.append(f"pos_{w}")
        sp.swap.append((f"room_up_{w}", f"room_dn_{w}"))

    # --- pivot zigzag (đã xác nhận) ------------------------------------- #
    hn, ln = h.to_numpy(float), l.to_numpy(float)
    cn = c.to_numpy(float)
    for r in cfg.pivot_thresholds:
        tag = f"zz{int(round(r * 1000))}"
        leg, last_p, age, ph1, ph2, pl1, pl2, ext = _zigzag(hn, ln, float(r))
        f[f"{tag}_leg"] = leg
        f[f"{tag}_prog"] = np.log(cn / last_p) / r                    # tiến trình sóng hiện tại (đơn vị r)
        f[f"{tag}_pull"] = np.log(cn / ext) / r                       # hồi về từ cực trị sóng hiện tại
        f[f"{tag}_age"] = np.log1p(age)
        f[f"{tag}_d_hi"] = np.log(ph1 / cn) / r                      # cách đỉnh pivot gần nhất
        f[f"{tag}_d_lo"] = np.log(cn / pl1) / r                      # cách đáy pivot gần nhất
        f[f"{tag}_hh"] = np.log(ph1 / ph2) / r                       # đỉnh sau cao hơn đỉnh trước
        f[f"{tag}_hl"] = np.log(pl1 / pl2) / r                       # đáy sau cao hơn đáy trước
        f[f"{tag}_brk"] = np.where(cn > ph1, 1.0, np.where(cn < pl1, -1.0, 0.0))
        sp.odd += [f"{tag}_leg", f"{tag}_prog", f"{tag}_pull", f"{tag}_brk"]
        sp.even.append(f"{tag}_age")
        sp.swap.append((f"{tag}_d_hi", f"{tag}_d_lo"))
        sp.swapneg.append((f"{tag}_hh", f"{tag}_hl"))

    # --- dòng tiền -------------------------------------------------------- #
    imb = 2 * b["tb"] - v
    for k in (3, 12, 48):
        f[f"imb_{k}"] = imb.rolling(k).sum() / v.rolling(k).sum().replace(0, np.nan)
        sp.odd.append(f"imb_{k}")
    n_ = b["n"].replace(0, np.nan)
    f["trades_rel"] = np.log((n_.rolling(3).sum() / n_.shift(3).rolling(288).median() / 3).clip(lower=eps))
    f["tsize_rel"] = np.log(((v / n_).rolling(3).mean() / (v / n_).shift(3).rolling(288).median()).clip(lower=eps))
    sp.even += ["trades_rel", "tsize_rel"]

    # --- sự kiện trước đó ------------------------------------------------- #
    e1 = f["E1"] > 0
    f["n_events_24h"] = f["is_event"].astype(float).rolling(288).sum()
    last_e1 = pd.Series(np.where(e1.shift(1, fill_value=False), np.arange(len(f)) - 1, np.nan)).ffill()
    f["since_e1"] = np.log1p(np.minimum(np.arange(len(f)) - last_e1.fillna(-1e9), 2880))
    f["last_e1_dir"] = pd.Series(np.where(e1, np.sign(c - o), np.nan)).shift(1).ffill()
    sp.even += ["n_events_24h", "since_e1"]
    sp.odd.append("last_e1_dir")

    # --- funding & thời gian --------------------------------------------- #
    fund = funding_table(funding) if funding is not None else None
    if fund is not None and len(fund):
        close_ms = (b["t"] + M5).to_numpy()
        pos = np.searchsorted(fund["funding_time_ms"].to_numpy(), close_ms, side="right") - 1
        fr_all = fund["funding_rate"]
        z_all = (fr_all - fr_all.rolling(90, min_periods=15).mean()) / fr_all.rolling(90, min_periods=15).std()  # 30 ngày
        valid = pos >= 0
        f["fund_rate"] = np.where(valid, fr_all.to_numpy()[np.clip(pos, 0, None)], np.nan) * 1e4
        f["fund_z"] = np.where(valid, z_all.to_numpy()[np.clip(pos, 0, None)], np.nan)
        sp.odd += ["fund_rate", "fund_z"]
    hour = pd.to_datetime(b["t"] + M5, unit="ms", utc=True).dt.hour + pd.to_datetime(b["t"] + M5, unit="ms", utc=True).dt.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    sp.even += ["hour_sin", "hour_cos"]

    f["sl_pct"] = np.clip(cfg.sl_vol_mult * f["sigma_1h"], cfg.sl_min, cfg.sl_max)
    f["fee_r"] = cfg.fee_roundtrip / f["sl_pct"]
    f["sl_log"] = np.log(f["sl_pct"])
    sp.even.append("sl_log")
    f = f.replace([np.inf, -np.inf], np.nan)
    f["warm"] = np.arange(len(f)) >= 2880
    return f, sp


def side_matrix(f: pd.DataFrame, sp: FeatureSpec, side: int) -> np.ndarray:
    """Ma trận feature cho phía `side` (+1 long, -1 short)."""
    cols = [f[k].to_numpy(float) * side for k in sp.odd] + [f[k].to_numpy(float) for k in sp.even]
    for a, b in sp.swap:
        cols += [f[a].to_numpy(float), f[b].to_numpy(float)] if side == 1 else [f[b].to_numpy(float), f[a].to_numpy(float)]
    for a, b in sp.swapneg:
        cols += [f[a].to_numpy(float), f[b].to_numpy(float)] if side == 1 else [-f[b].to_numpy(float), -f[a].to_numpy(float)]
    return np.column_stack(cols).astype(np.float32)


# --------------------------------------------------------------------------- #
# Labels: kết quả giao dịch trên nến 1m
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _trade_outcomes(o, h, l, c, i0s, d, tp_mult, hold):
    n = len(i0s)
    cls = np.full((n, 2), -1, np.int8)
    r = np.full((n, 2), np.nan)
    mins = np.full((n, 2), -1, np.int32)
    for j in range(n):
        i0 = i0s[j]
        if i0 < 0 or i0 + hold > len(o):
            continue
        e = o[i0]
        for si in range(2):
            s = 1 if si == 0 else -1
            sl = e * (1 - s * d[j])
            tp = e * (1 + s * tp_mult * d[j])
            res = 2
            k_end = hold - 1
            for k in range(hold):
                hh = h[i0 + k]
                ll = l[i0 + k]
                hit_sl = ll <= sl if s == 1 else hh >= sl
                hit_tp = hh >= tp if s == 1 else ll <= tp
                if hit_sl:
                    res = 0
                    k_end = k
                    break
                if hit_tp:
                    res = 1
                    k_end = k
                    break
            cls[j, si] = res
            mins[j, si] = k_end + 1
            if res == 0:
                op = o[i0 + k_end]
                px = op if k_end > 0 and (op - sl) * s <= 0 else sl
                r[j, si] = s * (px / e - 1) / d[j]
            elif res == 1:
                op = o[i0 + k_end]
                px = op if k_end > 0 and (op - tp) * s >= 0 else tp
                r[j, si] = s * (px / e - 1) / d[j]
            else:
                r[j, si] = s * (c[i0 + hold - 1] / e - 1) / d[j]
    return cls, r, mins


def label_events(m1: pd.DataFrame, ev: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Kết quả LONG và SHORT của mỗi sự kiện (R gộp, chưa trừ phí). ev cần cột t (open ms nến 5m), sl_pct."""
    t1 = m1["t"].to_numpy(np.int64)
    entry_ms = ev["t"].to_numpy(np.int64) + M5
    i0 = np.searchsorted(t1, entry_ms)
    ok = (i0 < len(t1)) & (t1[np.minimum(i0, len(t1) - 1)] == entry_ms)
    i0 = np.where(ok, i0, -1).astype(np.int64)
    cls, r, mins = _trade_outcomes(m1["o"].to_numpy(float), m1["h"].to_numpy(float), m1["l"].to_numpy(float),
                                   m1["c"].to_numpy(float), i0, ev["sl_pct"].to_numpy(float), float(cfg.tp_mult),
                                   int(cfg.hold_min))
    out = pd.DataFrame(index=ev.index)
    out["cls_long"], out["cls_short"] = cls[:, 0], cls[:, 1]
    out["r_long"], out["r_short"] = r[:, 0], r[:, 1]
    out["min_long"], out["min_short"] = mins[:, 0], mins[:, 1]
    out["entry_ms"] = entry_ms
    return out


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def build_dataset(m1: pd.DataFrame, funding, cfg: Config) -> dict:
    """Nến sự kiện + feature + nhãn giao dịch."""
    f, sp = compute_bars(m1, funding, cfg)
    ev = f[f["is_event"] & f["warm"]].copy()
    lab = label_events(standardize_m1(m1), ev, cfg)
    ev = ev.join(lab)
    ev = ev[(ev["cls_long"] >= 0) & (ev["cls_short"] >= 0)]
    ev["time"] = pd.to_datetime(ev["t"] + M5, unit="ms", utc=True)
    ev["year"] = ev["time"].dt.year
    return {"events": ev, "spec": sp, "bars": f}


def stack_sides(ev: pd.DataFrame, sp: FeatureSpec):
    """Mỗi sự kiện -> 2 dòng (long, short). y = 0 SL, 1 TP, 2 hết giờ."""
    xl, xs = side_matrix(ev, sp, 1), side_matrix(ev, sp, -1)
    x = np.vstack([xl, xs])
    y = np.concatenate([ev["cls_long"].to_numpy(), ev["cls_short"].to_numpy()]).astype(int)
    r = np.concatenate([ev["r_long"].to_numpy(), ev["r_short"].to_numpy()])
    side = np.concatenate([np.ones(len(ev)), -np.ones(len(ev))]).astype(int)
    t = np.concatenate([ev["t"].to_numpy(), ev["t"].to_numpy()])
    return x, y, r, side, t


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _xgb(cfg: Config):
    import xgboost as xgb
    p = dict(cfg.xgb)
    es = p.pop("early_stopping_rounds", None)
    return xgb.XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss", tree_method="hist",
                             device=cfg.device, random_state=cfg.seed, n_jobs=-1, early_stopping_rounds=es, **p)


def expected_r(proba: np.ndarray, r_timeout: float, cfg: Config, fee_r: np.ndarray) -> np.ndarray:
    return proba[:, 1] * cfg.tp_mult - proba[:, 0] + proba[:, 2] * r_timeout - fee_r


def fit_model(ev_tr: pd.DataFrame, ev_es: pd.DataFrame, sp: FeatureSpec, cfg: Config) -> dict:
    x, y, r, side, t = stack_sides(ev_tr, sp)
    xe, ye, _, _, _ = stack_sides(ev_es, sp)
    age_days = (t.max() - t) / 86_400_000
    w = 0.5 ** (age_days / cfg.recency_halflife_days) if cfg.recency_halflife_days else np.ones(len(t))
    model = _xgb(cfg)
    model.fit(x, y, sample_weight=w, eval_set=[(xe, ye)], verbose=False)
    r_to = float(np.nanmean(r[y == 2])) if (y == 2).any() else 0.0
    return {"model": model, "r_timeout": r_to, "best_iteration": int(getattr(model, "best_iteration", 0) or 0),
            "class_rate": np.bincount(y, minlength=3) / len(y)}


def predict_events(fit: dict, ev: pd.DataFrame, sp: FeatureSpec, cfg: Config) -> pd.DataFrame:
    """P(SL/TP/hết giờ) và EV sau phí cho long / short, phía được chọn."""
    out = pd.DataFrame(index=ev.index)
    fee = ev["fee_r"].to_numpy(float)
    for s, tag in ((1, "long"), (-1, "short")):
        p = fit["model"].predict_proba(side_matrix(ev, sp, s))
        out[f"p_sl_{tag}"], out[f"p_tp_{tag}"], out[f"p_to_{tag}"] = p[:, 0], p[:, 1], p[:, 2]
        out[f"ev_{tag}"] = expected_r(p, fit["r_timeout"], cfg, fee)
    out["side"] = np.where(out["ev_long"] >= out["ev_short"], 1, -1)
    out["ev"] = np.maximum(out["ev_long"], out["ev_short"])
    out["p_tp"] = np.where(out["side"] == 1, out["p_tp_long"], out["p_tp_short"])
    out["p_sl"] = np.where(out["side"] == 1, out["p_sl_long"], out["p_sl_short"])
    return out


def realized(ev: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Kết quả thực của phía được chọn (R sau phí, lý do thoát)."""
    long = pred["side"] == 1
    cls = np.where(long, ev["cls_long"], ev["cls_short"])
    r = np.where(long, ev["r_long"], ev["r_short"]) - ev["fee_r"]
    mins = np.where(long, ev["min_long"], ev["min_short"])
    return pd.DataFrame({"t": ev["t"], "time": ev["time"], "side": pred["side"], "ev": pred["ev"],
                         "p_tp": pred["p_tp"], "p_sl": pred["p_sl"], "reason": pd.Categorical.from_codes(cls, ["sl", "tp", "timeout"]),
                         "R": r, "minutes": mins, "sl_pct": ev["sl_pct"], "close": ev["close"],
                         "E1": ev["E1"], "E2": ev["E2"], "E3": ev["E3"]}, index=ev.index)


def choose_threshold(res_val: pd.DataFrame, cfg: Config) -> dict:
    """Ngưỡng EV trên phần validation dành cho chọn ngưỡng: tối đa (R trung bình - 1 sai số chuẩn), đủ số lệnh."""
    best = {"thr": None, "score": -np.inf, "n": 0, "avg_R": np.nan}
    rows = []
    for thr in cfg.thr_grid:
        s = res_val[res_val["ev"] >= thr]["R"]
        n = len(s)
        lb = s.mean() - s.std(ddof=1) / math.sqrt(n) if n >= 2 else np.nan
        rows.append({"thr": thr, "n": n, "avg_R": s.mean() if n else np.nan, "lower": lb})
        if n >= cfg.min_val_trades and lb > best["score"]:
            best = {"thr": thr, "score": lb, "n": n, "avg_R": float(s.mean())}
    best["table"] = pd.DataFrame(rows)
    return best


def _purge(ev, start_ms, end_ms, hold_ms):
    return ev[(ev["t"] >= start_ms) & (ev["t"] + M5 + hold_ms <= end_ms)]


def walk_forward(data: dict, cfg: Config, log=print) -> dict:
    """Walk-forward theo năm. Trả về dự báo OOS, ngưỡng và kết quả mỗi năm test."""
    ev, sp = data["events"], data["spec"]
    hold_ms = cfg.hold_min * M1
    ms = lambda y: int(pd.Timestamp(f"{y}-01-01", tz="UTC").value // 1_000_000)  # noqa: E731
    folds, preds = [], []
    for y in cfg.eval_years:
        tr = _purge(ev, ms(cfg.first_train_year), ms(y - 1), hold_ms)
        val = _purge(ev, ms(y - 1), ms(y), hold_ms)
        te = _purge(ev, ms(y), ms(y + 1), 0)
        if len(te) == 0 or len(val) < 100:
            continue
        cut = val["t"].quantile(cfg.val_es_frac)
        val_es, val_thr = val[val["t"] <= cut], val[val["t"] > cut + hold_ms]
        fit = fit_model(tr, val_es, sp, cfg)
        res_val = realized(val_thr, predict_events(fit, val_thr, sp, cfg))
        choice = choose_threshold(res_val, cfg)
        pred_te = predict_events(fit, te, sp, cfg)
        res_te = realized(te, pred_te)
        res_te["fold"] = y
        res_te["thr"] = choice["thr"] if choice["thr"] is not None else np.inf
        res_te["enter"] = res_te["ev"] >= res_te["thr"]
        preds.append(res_te.join(pred_te[[c for c in pred_te.columns if c not in res_te.columns]]))
        tk = res_te[res_te["enter"]]
        folds.append({"year": y, "train_events": len(tr), "val_events": len(val), "test_events": len(te),
                      "best_iter": fit["best_iteration"], "thr": choice["thr"], "val_trades": choice["n"],
                      "val_avg_R": choice["avg_R"], "test_trades": len(tk),
                      "test_avg_R": tk["R"].mean() if len(tk) else np.nan,
                      "test_tp": (tk["reason"] == "tp").mean() if len(tk) else np.nan,
                      "test_sl": (tk["reason"] == "sl").mean() if len(tk) else np.nan,
                      "all_events_avg_R_best_side": res_te["R"].mean()})
        log(f"[{y}] train {len(tr)} | val {len(val)} | thr {choice['thr']} (val {choice['n']} lệnh, "
            f"{choice['avg_R']:+.3f}R) | test {len(tk)} lệnh "
            f"{(tk['R'].mean() if len(tk) else float('nan')):+.3f}R")
    return {"folds": pd.DataFrame(folds), "oos": pd.concat(preds) if preds else pd.DataFrame()}


def train_final(data: dict, cfg: Config, val_days: float = 365.0) -> dict:
    """Model cuối: train trên mọi sự kiện trừ `val_days` cuối (dùng cho early stopping + chọn ngưỡng)."""
    ev, sp = data["events"], data["spec"]
    hold_ms = cfg.hold_min * M1
    end = int(ev["t"].max())
    cut = end - int(val_days * 86_400_000)
    tr = ev[ev["t"] + M5 + hold_ms <= cut]
    val = ev[ev["t"] > cut]
    vcut = val["t"].quantile(cfg.val_es_frac)
    fit = fit_model(tr, val[val["t"] <= vcut], sp, cfg)
    val_thr = val[val["t"] > vcut + hold_ms]
    choice = choose_threshold(realized(val_thr, predict_events(fit, val_thr, sp, cfg)), cfg)
    return {"strategy": STRATEGY, "config": asdict(cfg), "spec": asdict(sp), "fit": fit,
            "threshold": choice["thr"], "threshold_table": choice["table"], "trained_to_ms": cut}


def save_bundle(bundle: dict, path):
    import joblib
    joblib.dump(bundle, path)


def load_bundle(path) -> dict:
    import joblib
    return joblib.load(path)


def bundle_parts(bundle: dict):
    cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in bundle["config"].items()})
    sd = bundle["spec"]
    sp = FeatureSpec(odd=list(sd["odd"]), even=list(sd["even"]), swap=[tuple(x) for x in sd["swap"]],
                     swapneg=[tuple(x) for x in sd["swapneg"]])
    return cfg, sp


def predict_live(bundle: dict, m1: pd.DataFrame, funding=None, last_n: int = 1) -> pd.DataFrame:
    """Tính quyết định cho `last_n` nến 5m cuối (m1 cần >= 12 ngày nến 1m liên tục).
    Cột: t (open ms nến 5m), decision_ms, is_event, side, ev, p_tp, p_sl, enter, sl_pct, tp_pct, close."""
    cfg, sp = bundle_parts(bundle)
    f, _ = compute_bars(m1, funding, cfg)
    if len(f) < 2880 + last_n:
        raise ValueError(f"need >= {2880 + last_n} closed 5m bars of 1m history, got {len(f)}")
    tail = f.iloc[-last_n:].copy()
    pred = predict_events(bundle["fit"], tail, sp, cfg)
    out = tail[["t", "close", "is_event", "E1", "E2", "E3", "sl_pct", "fee_r"]].join(pred)
    thr = bundle["threshold"]
    out["threshold"] = np.inf if thr is None else thr
    out["enter"] = out["is_event"] & (out["ev"] >= out["threshold"])
    out["tp_pct"] = out["sl_pct"] * cfg.tp_mult
    out["decision_ms"] = out["t"] + M5
    return out


# --------------------------------------------------------------------------- #
# Đánh giá
# --------------------------------------------------------------------------- #
def block_ci(r: np.ndarray, t_ms: np.ndarray, n_boot=1000, seed=0, block_days=7):
    if len(r) < 10:
        return (np.nan, np.nan)
    keys, inv = np.unique(t_ms // (block_days * 86_400_000), return_inverse=True)
    sums, cnts = np.bincount(inv, weights=r), np.bincount(inv)
    pick = np.random.default_rng(seed).integers(0, len(keys), size=(n_boot, len(keys)))
    means = sums[pick].sum(1) / cnts[pick].sum(1)
    return float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))


def trade_stats(tr: pd.DataFrame) -> dict:
    if tr is None or not len(tr):
        return {"trades": 0}
    r = tr["R"].to_numpy()
    lo, hi = block_ci(r, tr["t"].to_numpy())
    days = max((tr["t"].max() - tr["t"].min()) / 86_400_000, 1)
    tp, sl = (tr["reason"] == "tp").mean(), (tr["reason"] == "sl").mean()
    cum = np.cumsum(r)
    dd = float((np.maximum.accumulate(np.concatenate([[0], cum]))[1:] - cum).max())
    return {"trades": len(r), "per_month": len(r) / days * 30.4, "TP%": 100 * tp, "SL%": 100 * sl,
            "timeout%": 100 * (1 - tp - sl), "win_of_resolved%": 100 * tp / (tp + sl) if tp + sl else np.nan,
            "avg_R": float(r.mean()), "ci90_lo": lo, "ci90_hi": hi, "total_R": float(r.sum()), "max_dd_R": dd,
            "avg_minutes": float(tr["minutes"].mean()), "long%": 100 * float((tr["side"] == 1).mean())}
