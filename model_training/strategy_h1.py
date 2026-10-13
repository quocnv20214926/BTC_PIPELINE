"""BTCUSDT H1 - chiến lược TOUCH + HƯỚNG (model B: giá + vị thế), cấu hình đồng thuận 2026-10-13.

Ý tưởng: touch dự báo biến động b, lệnh cũng chốt lời đúng b.
  1. TOUCH (modelv2 trên nến 1h): trong 8 nến tới giá có chạm +b hoặc -b tính từ open[t+1] không?
     b = clip(sigma_1h * sqrt(8), 1%, 5%).  Tín hiệu khi p_touch >= ngưỡng top 20% (ngưỡng lấy từ calibration).
  2. HƯỚNG (model B): XGBoost train CHỈ trên các mẫu thực sự chạm b trong 8h, nhãn = chạm +b trước.
     Feature = 101 feature touch v2 + 20 feature có hướng + 18 feature vị thế (OI, long/short, premium).
     Chỉ vào lệnh khi |p_lên - 0.5| >= ngưỡng top 10% (ngưỡng lấy từ các giờ touch cao của calibration).
  3. LỆNH: vào ở open nến kế tiếp theo hướng model B. TP = 1b, SL = 0.75b, mỗi lệnh có TP / SL riêng,
     giữ tới khi chạm (tối đa 30 ngày), nến chạm cả TP lẫn SL tính là SL. Tối đa 5 lệnh mở cùng lúc,
     được vào ở mỗi giờ có tín hiệu. Phí 0.1% (khứ hồi), tính theo khoảng SL.
  3b. THOÁT NHANH (nến 5m): nếu một nến 5m NGƯỢC chiều lệnh có range >= 0.75% thì đóng lệnh ở close nến đó
     (kiểm tra sau TP / SL). Kiểm định: R / lệnh 0.159 -> 0.188, sụt giảm 24% -> 20%.
  3c. THỜI ĐIỂM VÀO (kiểm định 2026-10-13): chờ 2-9 nến 5m sau khi H1 đóng cho R / lệnh ~0.20 so với 0.188 khi vào
     ngay, nhưng chênh lệch ghép cặp không có ý nghĩa (KTC 95% chứa 0) và không đơn điệu theo số nến chờ;
     vào bằng xác nhận nến 5m cùng chiều thì kém hơn. Mặc định giữ "next_open" (vào ngay).
  4. VỐN: rủi ro mỗi lệnh = risk_per_trade * vốn lúc vào lệnh (khuyến nghị 1%), lãi kép.

Walk-forward theo năm (như mọi model của dự án): train = nhãn kết thúc trước 1/1 của Y-1, Y-1 = 60% đầu dừng sớm
+ 40% sau calibration / ngưỡng, test = Y. Kết quả kiểm định (dữ liệu tới 08/10/2026, thực thi 5m + thoát nhanh,
rủi ro 1%): 796 lệnh, R / lệnh 0.188, R / năm 53.5 / 42.3 / 23.7 / 29.9 (2023-2026), vốn x3.91, sụt giảm 20%,
2 quý âm. (Backtest nến 1h không thoát nhanh: R / lệnh 0.15, x2.86, sụt giảm 24%.)

Dùng:
    import strategy_h1 as st
    cfg = st.StrategyConfig()
    st.make_1h_csv(cfg.csv_5m, cfg.csv_1h)            # gộp nến 5m -> 1h
    res = st.walk_forward(cfg)                          # train + đánh giá + backtest 5m 2023-2026
    st.anomaly_exit(side, candle_5m, cfg)               # live: True -> đóng lệnh ngay
    st.export_bundle(cfg, res)                          # lưu model cuối + kết quả vào artifacts/1h_strategy_v1
    sig = st.predict_live(bundle, candles_1h, positioning_5m, funding)   # tín hiệu cho nến 1h vừa đóng
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import modelv2 as mv2
from modelv2 import njit

STRATEGY = "touch_dirB_h1_v1"
BPD = 24


@dataclass
class StrategyConfig:
    csv_5m: str = "historical_data/BTCUSDT_5m_all.csv"
    csv_1h: str = "historical_data/BTCUSDT_1h_from5m.csv"
    positioning_csv: str = "historical_data/BTCUSDT_positioning_5m.csv"
    output_dir: str = "artifacts/1h_strategy_v1"
    # touch
    horizon: int = 8
    pivot_reversal: float = 0.01
    lookback_bars: int = 168
    vol_span: int = 168
    slow_vol_span: int = 720
    b_mult: float = 1.0
    min_barrier: float = 0.01
    max_barrier: float = 0.05
    touch_top: float = 0.20
    # hướng
    dir_top: float = 0.10
    # lệnh
    tp_mult: float = 1.0
    sl_mult: float = 0.75
    max_hold_bars: int = 720
    max_positions: int = 5
    min_gap_bars: int = 1
    fee_pct: float = 0.001
    risk_per_trade: float = 0.01
    # thực thi trên nến 5m (backtest_m5 / live)
    anomaly_range: float = 0.0075        # thoát khi 1 nến 5m NGƯỢC chiều lệnh có (high-low)/open >= ngưỡng (0 = tắt)
    entry_mode: str = "next_open"        # "next_open": open nến 1h kế tiếp | "delay": chờ entry_delay_m5 nến 5m
                                         # | "confirm": vào sau nến 5m đầu tiên đóng cùng chiều tín hiệu
    entry_delay_m5: int = 0
    confirm_window_m5: int = 6
    # giao thức
    eval_years: tuple[int, ...] = (2023, 2024, 2025, 2026)
    val_es_frac: float = 0.6
    seed: int = 42
    xgb_device: str = "cpu"
    n_boot: int = 200
    # model: "xgb" (XGBoost, mặc định) hoặc "tabm" (mạng nơ-ron TabM, nn_tabm.py, cần torch).
    # TabM đã kiểm định ngoài mẫu 2026 (chọn cấu hình trên 2023-2025) và KHÔNG đạt: R / lệnh -0.28 -> giữ xgb.
    model_type: str = "xgb"
    touch_xgb: dict = field(default_factory=dict)       # ghi đè mv2.XGBParams cho touch (vd {"max_depth": 3})
    dir_xgb: dict = field(default_factory=dict)         # ghi đè tham số XGBoost của model hướng B
    dir_features: str = "all"                           # "all" | "dir_pos" (bỏ 94 feature giá của touch)
    tabm_touch: dict = field(default_factory=dict)      # ghi đè TabMParams cho touch
    tabm_dir: dict = field(default_factory=dict)        # ghi đè TabMParams cho model hướng B

    def touch_cfg(self) -> mv2.Config:
        return mv2.Config(csv_path=self.csv_1h, output_dir=self.output_dir, bar_minutes=60,
                          pivot_reversal=self.pivot_reversal, lookback_bars=self.lookback_bars,
                          barrier_horizon=self.horizon, barrier_vol_mult=self.b_mult, min_barrier=self.min_barrier,
                          max_barrier=self.max_barrier, vol_span=self.vol_span, slow_vol_span=self.slow_vol_span,
                          sample_stride=1, eval_years=tuple(self.eval_years), val_es_frac=self.val_es_frac,
                          xgb_device=self.xgb_device, n_boot=self.n_boot, top_fractions=(1.0, 0.5, 0.3, 0.2, 0.1),
                          seed=self.seed, verbose=False, touch=mv2.XGBParams(**self.touch_xgb))


# --------------------------------------------------------------------------- #
# Dữ liệu
# --------------------------------------------------------------------------- #
def aggregate_1h(candles_5m: pd.DataFrame) -> pd.DataFrame:
    """Gộp nến 5m thành 1h (giữ taker_buy_volume, trades). Chỉ giữ giờ đủ 12 nến 5m.
    Dùng chung cho train (make_1h_csv) và live (pipeline/strategy_live.py) để hai bên giống hệt nhau."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = mv2.prepare_candles(candles_5m, 5).set_index("timestamp")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for c in ("taker_buy_volume", "trades"):
        if c in df:
            agg[c] = "sum"
    cnt = df["close"].resample("1h").count()
    h = df.resample("1h").agg(agg)
    return h[cnt == 12].dropna(subset=["open"]).reset_index()


def make_1h_csv(src_5m: str, dst: str) -> pd.DataFrame:
    """Gộp file nến 5m thành file nến 1h dùng để train."""
    h = aggregate_1h(pd.read_csv(src_5m))
    h.to_csv(dst, index=False)
    return h


def load_1h(cfg: StrategyConfig) -> pd.DataFrame:
    return mv2.load_candles(cfg.csv_1h, 60)


def positioning_features(candles_1h: pd.DataFrame, positioning_5m: pd.DataFrame) -> pd.DataFrame:
    """18 feature vị thế theo giờ, lấy giá trị đã biết lúc nến 1h đóng (dòng 5m cuối cùng của giờ).
    positioning_5m: bảng giống historical_data/BTCUSDT_positioning_5m.csv (open_time_ms, oi, top_acct_ls, top_pos_ls,
    acct_ls, prem_close, ...) - đã khớp quy ước thời gian (giá trị của dòng = đã biết lúc nến 5m đó đóng)."""
    P = positioning_5m.copy()
    P["t"] = pd.to_datetime(P["open_time_ms"], unit="ms", utc=True)
    P = P.set_index("t").sort_index()
    closes = pd.to_datetime(candles_1h["timestamp"], utc=True) + pd.Timedelta(minutes=55)
    Q = P.reindex(closes).reset_index(drop=True)
    lg = lambda s: np.log(s.where(s > 0))                                                # noqa: E731
    F = pd.DataFrame(index=candles_1h.index)
    oi = lg(Q["oi"])
    for k in (1, 4, 24):
        F[f"p_oi_{k}h"] = oi - oi.shift(k)
    F["p_oi4_x_ret4"] = F["p_oi_4h"] * np.sign(np.log(candles_1h["close"].reset_index(drop=True)).diff(4))
    for nm in ("top_acct_ls", "top_pos_ls", "acct_ls"):
        x = lg(Q[nm])
        F[f"p_{nm}"] = x
        F[f"p_{nm}_z7d"] = (x - x.rolling(168, min_periods=100).mean()) / x.rolling(168, min_periods=100).std()
        F[f"p_{nm}_d4"] = x - x.shift(4)
    F["p_top_minus_crowd"] = F["p_top_pos_ls_z7d"] - F["p_acct_ls_z7d"]
    pr = Q["prem_close"] * 1e4
    F["p_prem"] = pr
    F["p_prem_24"] = pr.rolling(24, min_periods=12).mean()
    F["p_prem_z7d"] = (pr - pr.rolling(168, min_periods=100).mean()) / pr.rolling(168, min_periods=100).std()
    return F


def directional_features(df: pd.DataFrame, ts: np.ndarray, pivots: pd.DataFrame, bf: dict) -> dict:
    high, low, close, vol = (df[c].to_numpy(np.float64) for c in ("high", "low", "close", "volume"))
    lc, sig = bf["logc"], bf["sigma"]
    f = {}

    def roll(x, w, how):
        return getattr(pd.Series(x).rolling(w, min_periods=w), how)().to_numpy()

    for k in (1, 3, 6, 12, 24):
        f[f"d_ret_{k}"] = (lc[ts] - lc[ts - k]) / (sig[ts] * math.sqrt(k))
    rng = np.maximum(high[ts] - low[ts], 1e-12)
    f["d_close_loc"] = ((close[ts] - low[ts]) / rng - 0.5) * 2
    if "taker_buy_volume" in df.columns:
        imb = 2 * df["taker_buy_volume"].to_numpy(np.float64) / np.maximum(vol, 1e-12) - 1
        for k in (3, 6, 12, 24):
            f[f"d_taker_{k}"] = roll(imb, k, "mean")[ts]
    if "funding_rate" in df.columns:
        f["d_funding"] = df["funding_rate"].to_numpy(np.float64)[ts] * 1e4
        f["d_funding_z"] = df["funding_z_90"].to_numpy(np.float64)[ts]
    for days in (1, 3, 7):
        w = BPD * days
        f[f"x{days}d_up"] = (np.log(roll(high, w, "max")[ts]) - lc[ts]) / sig[ts]
        f[f"x{days}d_dn"] = (lc[ts] - np.log(roll(low, w, "min")[ts])) / sig[ts]
    conf = pivots["confirm_idx"].to_numpy(np.int64)
    ke = np.searchsorted(conf, ts, side="right")
    kind = pivots["kind"].to_numpy(np.float64)
    price = pivots["price"].to_numpy(np.float64)
    last = np.maximum(ke - 1, 0)
    f["d_last_pivot_kind"] = -kind[last]
    f["d_pivot_hhll"] = np.log(price[last] / price[np.maximum(ke - 3, 0)]) / sig[ts]
    return {k: np.where(np.isfinite(v), v, np.nan).astype(np.float32) for k, v in f.items()}


@dataclass
class StrategyData:
    df: pd.DataFrame
    ds: mv2.Dataset
    x_dir: np.ndarray
    dir_names: tuple
    pivots: pd.DataFrame


def build_data(df: pd.DataFrame, positioning_5m: pd.DataFrame, cfg: StrategyConfig, live: bool = False,
               live_last: int | None = None) -> StrategyData:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        tc = cfg.touch_cfg()
        piv = mv2.find_confirmed_pivots(df, cfg.pivot_reversal)
        ds = mv2.build_dataset(df, piv, tc, live=live, live_last=live_last)
        bf = mv2.compute_bar_features(df, tc)
        ts = ds.decision_idx
        ex = directional_features(df, ts, piv, bf)
        PF = positioning_features(df, positioning_5m)
        names = tuple(ds.names) + tuple(ex) + tuple(PF.columns)
        X = np.hstack([ds.x] + [ex[k][:, None] for k in ex] + [PF.to_numpy(np.float32)[ts]]).astype(np.float32)
    return StrategyData(df=df, ds=ds, x_dir=X, dir_names=names, pivots=piv)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
DIR_XGB_DEFAULT = dict(n_estimators=1500, learning_rate=0.02, max_depth=3, min_child_weight=50, subsample=0.7,
                       colsample_bytree=0.6, reg_lambda=10, early_stopping_rounds=80)


def _dir_xgb(cfg: StrategyConfig):
    from xgboost import XGBClassifier
    return XGBClassifier(**{**DIR_XGB_DEFAULT, **cfg.dir_xgb}, n_jobs=-1, random_state=cfg.seed, tree_method="hist")


def _tabm(cfg: StrategyConfig, over: dict):
    from nn_tabm import TabMClassifier, TabMParams
    return TabMClassifier(TabMParams(**{"seed": cfg.seed, **over}))


def p_touch_of(touch, x: np.ndarray) -> np.ndarray:
    """P(touch) cho cả hai loại model (dict của modelv2 = XGBoost, còn lại = TabM)."""
    if isinstance(touch, dict):
        return mv2.predict_touch(touch, x)
    return touch.predict_proba(x)[:, 1]


def dir_cols(sd: StrategyData, cfg: StrategyConfig) -> np.ndarray:
    """Cột feature cho model B. 'dir_pos' = 19 feature có hướng + 18 feature vị thế."""
    n = len(sd.dir_names)
    if cfg.dir_features == "all":
        return np.arange(n)
    if cfg.dir_features == "dir_pos":
        return np.arange(sd.ds.x.shape[1], n)
    raise ValueError(cfg.dir_features)


def fit_fold(sd: StrategyData, year: int, cfg: StrategyConfig) -> dict:
    ds, tc = sd.ds, cfg.touch_cfg()
    tr, va, te = mv2.fold_ids(ds, year)
    es, cal = mv2.split_validation(ds, va, cfg.val_es_frac)
    y, first = ds.labels["y_touch"], ds.labels["first"]
    if cfg.model_type == "xgb":
        touch = mv2.fit_models(ds, tr, va, tc)["models"]
        touch_rounds = None
    else:
        w = mv2._time_weights(ds, tr, int(ds.timestamp_ns[tr].max()), tc)
        touch = _tabm(cfg, cfg.tabm_touch).fit(ds.x[tr], y[tr], eval_set=[(ds.x[es], y[es])], sample_weight=w)
        touch_rounds = touch.best_iteration + 1
    pc_touch = p_touch_of(touch, ds.x[cal])
    thr_touch = float(np.quantile(pc_touch, 1 - cfg.touch_top))
    ok = (y == 1) & (first != 0)
    up = (first == 1).astype(int)
    a, e = tr[ok[tr]], es[ok[es]]
    dm = _dir_xgb(cfg) if cfg.model_type == "xgb" else _tabm(cfg, cfg.tabm_dir)
    cols = dir_cols(sd, cfg)
    Xd = sd.x_dir[:, cols]
    dm.fit(Xd[a], up[a], eval_set=[(Xd[e], up[e])], verbose=False)
    pc_up = dm.predict_proba(Xd[cal])[:, 1]
    hi_cal = pc_touch >= thr_touch
    thr_conf = float(np.quantile(np.abs(pc_up - 0.5)[hi_cal], 1 - cfg.dir_top))
    return {"year": year, "touch": touch, "dir": dm, "dir_rounds": int(dm.best_iteration + 1),
            "touch_rounds": touch_rounds, "dir_cols": cols, "thr_touch": thr_touch, "thr_conf": thr_conf, "test_ids": te}


def predict(models: dict, sd: StrategyData, ids: np.ndarray) -> pd.DataFrame:
    p_touch = p_touch_of(models["touch"], sd.ds.x[ids])
    cols = models.get("dir_cols")
    Xd = sd.x_dir if cols is None else sd.x_dir[:, cols]
    p_up = models["dir"].predict_proba(Xd[ids])[:, 1]
    conf = np.abs(p_up - 0.5)
    sig = (p_touch >= models["thr_touch"]) & (conf >= models["thr_conf"])
    return pd.DataFrame({"sample": ids, "bar": sd.ds.decision_idx[ids],
                         "timestamp": pd.to_datetime(sd.ds.timestamp_ns[ids], utc=True), "close": sd.ds.close[ids],
                         "b": sd.ds.barrier_pct[ids], "p_touch": p_touch, "p_up": p_up, "conf": conf,
                         "signal": sig, "side": np.where(p_up > 0.5, 1, -1)})


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _first_passage(o, h, l, ts, side, tp, sl, maxh):
    m = len(ts)
    out = np.zeros(m, np.int64)
    ex = np.zeros(m, np.int64)
    n = len(o)
    for j in range(m):
        t = ts[j]
        e = o[t + 1]
        s = side[j]
        tpp = e * math.exp(s * tp[j])
        slp = e * math.exp(-s * sl[j])
        ex[j] = min(n - 1, t + maxh)
        for k in range(t + 1, min(n, t + 1 + maxh)):
            if s > 0:
                a, b = h[k] >= tpp, l[k] <= slp
            else:
                a, b = l[k] <= tpp, h[k] >= slp
            if a or b:
                out[j] = -1 if b else 1                     # chạm cả hai trong 1 nến -> SL
                ex[j] = k
                break
    return out, ex


def backtest(sd: StrategyData, sig: pd.DataFrame, cfg: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả về (bảng lệnh, đường vốn). Tín hiệu = các dòng sig.signal == True (đã sắp theo thời gian)."""
    df = sd.df
    o, h, l, c = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    s = sig[sig.signal].sort_values("bar")
    bars = s.bar.to_numpy(np.int64)
    s = s[bars + 1 < len(o)]
    bars = s.bar.to_numpy(np.int64)
    side = s.side.to_numpy(np.float64)
    b = s.b.to_numpy(np.float64)
    res, ex = _first_passage(o, h, l, bars, side, cfg.tp_mult * b, cfg.sl_mult * b, cfg.max_hold_bars)
    rows, opens, last = [], [], -10**9
    for k in range(len(bars)):
        t = bars[k]
        opens = [x for x in opens if x >= t]
        if len(opens) >= cfg.max_positions or t - last < cfg.min_gap_bars:
            continue
        sl_pct = cfg.sl_mult * b[k]
        if res[k] == 1:
            g, how = cfg.tp_mult / cfg.sl_mult, "TP"
        elif res[k] == -1:
            g, how = -1.0, "SL"
        else:
            g, how = side[k] * math.log(c[ex[k]] / o[t + 1]) / sl_pct, "hết giờ"
        e = o[t + 1]
        rows.append({"entry_time": df.timestamp.iloc[t + 1], "exit_time": df.timestamp.iloc[ex[k]],
                     "side": "long" if side[k] > 0 else "short", "entry": e,
                     "tp": e * math.exp(side[k] * cfg.tp_mult * b[k]), "sl": e * math.exp(-side[k] * sl_pct),
                     "b_pct": b[k] * 100, "sl_pct": sl_pct * 100, "p_touch": s.p_touch.iloc[k], "p_up": s.p_up.iloc[k],
                     "result": how, "R_gross": g, "R": g - cfg.fee_pct / sl_pct, "_t": t, "_ex": int(ex[k])})
        opens.append(int(ex[k]))
        last = t
    T = pd.DataFrame(rows)
    if T.empty:
        return T, pd.DataFrame(columns=["time", "equity"])
    ev = sorted([(r._t, 0, i) for i, r in T.iterrows()] + [(r._ex, 1, i) for i, r in T.iterrows()],
                key=lambda x: (x[0], -x[1]))
    eq, risk, curve = 1.0, {}, []
    for tt, kind, i in ev:
        if kind == 0:
            risk[i] = cfg.risk_per_trade * eq
        else:
            eq += risk.pop(i) * T.at[i, "R"]
            curve.append((df.timestamp.iloc[tt], eq))
    return T.drop(columns=["_t", "_ex"]), pd.DataFrame(curve, columns=["time", "equity"])


@njit(cache=True)
def _sim_m5(o, h, l, c, i0s, side, tp, sl, maxb, anomaly):
    """Mô phỏng trên nến 5m từ nến i0 (giá vào = open[i0]). kind: 1 TP, -1 SL, 2 thoát nhanh, 3 hết giờ."""
    m = len(i0s)
    R = np.empty(m)
    ex = np.empty(m, np.int64)
    kind = np.empty(m, np.int64)
    n = len(o)
    for j in range(m):
        s = side[j]
        i0 = i0s[j]
        e = o[i0]
        tpp = e * math.exp(s * tp[j])
        slp = e * math.exp(-s * sl[j])
        kind[j] = 3
        k_end = min(n, i0 + maxb)
        R[j] = s * math.log(c[k_end - 1] / e) / sl[j]
        ex[j] = k_end - 1
        for k in range(i0, k_end):
            if (s > 0 and l[k] <= slp) or (s < 0 and h[k] >= slp):
                px = slp
                if (s > 0 and o[k] < slp) or (s < 0 and o[k] > slp):
                    px = o[k]
                R[j], ex[j], kind[j] = s * math.log(px / e) / sl[j], k, -1
                break
            if (s > 0 and h[k] >= tpp) or (s < 0 and l[k] <= tpp):
                R[j], ex[j], kind[j] = tp[j] / sl[j], k, 1
                break
            if anomaly > 0:
                against = (c[k] < o[k]) if s > 0 else (c[k] > o[k])
                if against and (h[k] - l[k]) / o[k] >= anomaly:
                    R[j], ex[j], kind[j] = s * math.log(c[k] / e) / sl[j], k, 2
                    break
    return R, ex, kind


def _entry_index_m5(o5, c5, start, side, cfg: StrategyConfig) -> np.ndarray:
    """Chỉ số nến 5m vào lệnh (-1 = bỏ lệnh). start = nến 5m đầu tiên sau khi nến 1h tín hiệu đóng."""
    if cfg.entry_mode == "next_open":
        return start.copy()
    if cfg.entry_mode == "delay":
        return start + cfg.entry_delay_m5
    out = np.full(len(start), -1, np.int64)
    for j, i0 in enumerate(start):
        for k in range(i0, min(len(o5) - 1, i0 + cfg.confirm_window_m5)):
            if (c5[k] - o5[k]) * side[j] > 0:
                out[j] = k + 1
                break
    return out


def backtest_m5(sd: StrategyData, sig: pd.DataFrame, cfg: StrategyConfig, candles_5m: pd.DataFrame
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest thực thi trên nến 5m: vào lệnh theo cfg.entry_mode, TP / SL / thoát nhanh trên 5m,
    tối đa cfg.max_positions lệnh, lãi kép. Trả về (bảng lệnh, đường vốn)."""
    d5 = mv2.prepare_candles(candles_5m, 5)
    o, h, l, c = (d5[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    t5 = mv2._ns_int(d5["timestamp"])
    s = sig[sig.signal].sort_values("bar")
    sig_ns = mv2._ns_int(s.timestamp)
    start = np.searchsorted(t5, sig_ns + 3600 * 10**9)
    okk = start < len(t5) - 1
    s, sig_ns, start = s[okk], sig_ns[okk], start[okk]
    side = s.side.to_numpy(np.float64)
    b = s.b.to_numpy(np.float64)
    ent = _entry_index_m5(o, c, start, side, cfg)
    keep = (ent >= 0) & (ent < len(o) - 1)
    s, sig_ns, side, b, ent = s[keep], sig_ns[keep], side[keep], b[keep], ent[keep]
    R, ex, kind = _sim_m5(o, h, l, c, ent, side, cfg.tp_mult * b, cfg.sl_mult * b,
                          cfg.max_hold_bars * 12, cfg.anomaly_range)
    names = {1: "TP", -1: "SL", 2: "thoát nhanh 5m", 3: "hết giờ"}
    rows, opens, last = [], [], -(10**18)
    for k in range(len(ent)):
        t = int(sig_ns[k])
        opens = [x for x in opens if x >= t]
        if len(opens) >= cfg.max_positions or t - last < cfg.min_gap_bars * 3600 * 10**9:
            continue
        sl_pct = cfg.sl_mult * b[k]
        e = o[ent[k]]
        rows.append({"signal_time": s.timestamp.iloc[k], "entry_time": d5.timestamp.iloc[ent[k]],
                     "exit_time": d5.timestamp.iloc[ex[k]], "side": "long" if side[k] > 0 else "short", "entry": e,
                     "tp": e * math.exp(side[k] * cfg.tp_mult * b[k]), "sl": e * math.exp(-side[k] * sl_pct),
                     "exit": c[ex[k]] if kind[k] in (2, 3) else (e * math.exp(side[k] * cfg.tp_mult * b[k]) if kind[k] == 1 else e * math.exp(-side[k] * sl_pct)),
                     "b_pct": b[k] * 100, "sl_pct": sl_pct * 100, "p_touch": s.p_touch.iloc[k], "p_up": s.p_up.iloc[k],
                     "result": names[int(kind[k])], "R_gross": R[k], "R": R[k] - cfg.fee_pct / sl_pct,
                     "_t": int(t5[ent[k]]), "_ex": int(t5[ex[k]])})
        opens.append(int(t5[ex[k]]))
        last = t
    T = pd.DataFrame(rows)
    if T.empty:
        return T, pd.DataFrame(columns=["time", "equity"])
    ev = sorted([(r._t, 0, i) for i, r in T.iterrows()] + [(r._ex, 1, i) for i, r in T.iterrows()],
                key=lambda x: (x[0], -x[1]))
    eq, risk, curve = 1.0, {}, []
    for tt, kd, i in ev:
        if kd == 0:
            risk[i] = cfg.risk_per_trade * eq
        else:
            # lệnh thoát ngay trong nến vào lệnh: rủi ro tính trên vốn hiện tại
            eq += risk.pop(i, cfg.risk_per_trade * eq) * T.at[i, "R"]
            curve.append((pd.to_datetime(tt, utc=True), eq))
    return T.drop(columns=["_t", "_ex"]), pd.DataFrame(curve, columns=["time", "equity"])


def anomaly_exit(side: int, candle_5m: dict, cfg: StrategyConfig) -> bool:
    """Live: True nếu nến 5m vừa đóng NGƯỢC chiều lệnh và có range >= cfg.anomaly_range -> đóng lệnh."""
    o_, h_, l_, c_ = (float(candle_5m[k]) for k in ("open", "high", "low", "close"))
    against = c_ < o_ if side > 0 else c_ > o_
    return bool(cfg.anomaly_range > 0 and against and (h_ - l_) / o_ >= cfg.anomaly_range)


def summarize(T: pd.DataFrame, curve: pd.DataFrame) -> dict:
    if T.empty:
        return {}
    T = T.sort_values("exit_time")
    cum = np.r_[0, np.cumsum(T.R.to_numpy())]
    eq = np.r_[1.0, curve.equity.to_numpy()]
    pk = np.maximum.accumulate(eq)
    yr = pd.to_datetime(T.exit_time, utc=True).dt.year
    q = pd.to_datetime(T.exit_time, utc=True).dt.tz_localize(None).dt.to_period("Q").astype(str)
    qs = T.groupby(q).R.sum()
    return {"trades": int(len(T)), "win_rate": float((T.R > 0).mean()), "R_per_trade": float(T.R.mean()),
            "total_R": float(T.R.sum()), "max_dd_R": float((np.maximum.accumulate(cum) - cum).max()),
            "max_dd_pct": float(((pk - eq) / pk).max() * 100), "equity_multiple": float(eq[-1]),
            "R_by_year": {int(k): float(v) for k, v in T.groupby(yr).R.sum().items()},
            "trades_by_year": {int(k): int(v) for k, v in T.groupby(yr).size().items()},
            "R_by_quarter": {k: float(v) for k, v in qs.items()}, "negative_quarters": int((qs < 0).sum())}


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def walk_forward(cfg: StrategyConfig, df: pd.DataFrame | None = None, positioning: pd.DataFrame | None = None,
                 candles_5m: pd.DataFrame | None = None, log=print) -> dict:
    """Train walk-forward, tạo tín hiệu và backtest. Backtest chính chạy trên nến 5m (cfg.csv_5m) với
    luật thoát nhanh anomaly; backtest nến 1h giữ lại trong res['summary_1h'] để tham chiếu."""
    df = load_1h(cfg) if df is None else df
    positioning = pd.read_csv(cfg.positioning_csv) if positioning is None else positioning
    sd = build_data(df, positioning, cfg)
    ds = sd.ds
    first, ytouch = ds.labels["first"], ds.labels["y_touch"]
    folds, sigs = [], []
    for Y in cfg.eval_years:
        m = fit_fold(sd, Y, cfg)
        te = m["test_ids"]
        P = predict(m, sd, te)
        ok = (ytouch[te] == 1) & (first[te] != 0)
        sel = P.signal.to_numpy()
        hit = ((P.side.to_numpy() == first[te]) & ok)[sel].sum() / max((ok & sel).sum(), 1)
        rep = {"year": Y, "touch_auc": float(roc_auc_score(ytouch[te], P.p_touch)),
               "dir_auc_touched": float(roc_auc_score(first[te][ok] == 1, P.p_up[ok])),
               "signal_hours": int(sel.sum()), "dir_hit_on_touched_signals": float(hit),
               "thr_touch": m["thr_touch"], "thr_conf": m["thr_conf"], "dir_rounds": m["dir_rounds"]}
        folds.append(rep)
        sigs.append(P.assign(year=Y))
        log(f"{Y}: touch AUC {rep['touch_auc']:.3f} | AUC hướng {rep['dir_auc_touched']:.3f} | "
            f"{rep['signal_hours']} giờ có tín hiệu | đúng hướng {hit:.3f}")
        last = m
    S = pd.concat(sigs, ignore_index=True)
    T1, c1 = backtest(sd, S, cfg)
    summ_1h = summarize(T1, c1)
    log(f"Backtest nến 1h (tham chiếu, không thoát nhanh): {summ_1h['trades']} lệnh | R/lệnh {summ_1h['R_per_trade']:+.3f} | "
        f"vốn x{summ_1h['equity_multiple']:.2f} | sụt giảm {summ_1h['max_dd_pct']:.1f}%")
    if candles_5m is None and Path(cfg.csv_5m).exists():
        candles_5m = pd.read_csv(cfg.csv_5m)
    if candles_5m is not None:
        T, curve = backtest_m5(sd, S, cfg, candles_5m)
    else:
        T, curve = T1, c1
    summ = summarize(T, curve)
    log(f"Backtest {'5m (thực thi chính)' if candles_5m is not None else '1h'}: {summ['trades']} lệnh | "
        f"R/lệnh {summ['R_per_trade']:+.3f} | vốn x{summ['equity_multiple']:.2f} | "
        f"sụt giảm {summ['max_dd_pct']:.1f}% | R theo năm {summ['R_by_year']}")
    return {"data": sd, "folds": folds, "signals": S, "trades": T, "equity": curve, "summary": summ,
            "summary_1h": summ_1h, "candles_5m": candles_5m, "last_models": last}


def export_bundle(cfg: StrategyConfig, res: dict) -> Path:
    import joblib
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = res["last_models"]
    joblib.dump(portable_bundle({"strategy": STRATEGY, "config": cfg, "touch": m["touch"], "dir": m["dir"],
                                 "thr_touch": m["thr_touch"], "thr_conf": m["thr_conf"], "dir_cols": m.get("dir_cols"),
                                 "touch_names": res["data"].ds.names, "dir_names": res["data"].dir_names,
                                 "trained_fold": m["year"]}), out / "models_final.joblib")
    res["trades"].to_csv(out / "backtest_trades.csv", index=False)
    res["equity"].to_csv(out / "backtest_equity.csv", index=False)
    res["signals"].drop(columns=["sample"]).to_csv(out / "walkforward_signals.csv", index=False)
    meta = {"strategy": STRATEGY, "config": asdict(cfg), "folds": res["folds"], "backtest": res["summary"],
            "backtest_1h_reference": res.get("summary_1h")}
    if "entry_timing" in res:
        meta["entry_timing"] = res["entry_timing"]
    if "risk_sensitivity" in res:
        meta["risk_sensitivity"] = res["risk_sensitivity"]
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=mv2._json_default),
                                       encoding="utf-8")
    return out


def upgrade_config(old) -> StrategyConfig:
    """Config pickled in an older bundle -> current StrategyConfig (new fields get their defaults)."""
    from dataclasses import fields
    return StrategyConfig(**{f.name: getattr(old, f.name) for f in fields(StrategyConfig) if hasattr(old, f.name)})


def _xgb_to_bytes(model) -> bytes:
    import os
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        model.save_model(tmp)
        return Path(tmp).read_bytes()
    finally:
        os.remove(tmp)


def _xgb_from_bytes(raw: bytes):
    from xgboost import XGBClassifier
    m = XGBClassifier()
    m.load_model(bytearray(raw))
    return m


def portable_bundle(b: dict) -> dict:
    """Bundle không chứa đối tượng sklearn / xgboost bị pickle: model lưu dạng JSON của XGBoost (đọc được ở
    mọi bản XGBoost 2.x / 3.x), config lưu dạng dict -> chạy được trên máy / container khác phiên bản thư viện."""
    out = {k: v for k, v in b.items() if k not in ("touch", "dir", "config")}
    out["format"] = "portable-v1"
    out["config"] = asdict(b["config"]) if not isinstance(b["config"], dict) else b["config"]
    t = b["touch"]
    if isinstance(t, dict):                            # XGBoost của modelv2
        model, best = t["touch"]
        out["touch_xgb_json"] = _xgb_to_bytes(model)
        out["touch_meta"] = {"best": int(best), **{k: v for k, v in t.items() if k != "touch"}}
    else:
        raise TypeError("portable_bundle chỉ hỗ trợ touch XGBoost")
    out["dir_xgb_json"] = _xgb_to_bytes(b["dir"])
    out["dir_best_iteration"] = int(getattr(b["dir"], "best_iteration", -1))
    return out


def load_bundle(path: str | Path) -> dict:
    import joblib
    b = joblib.load(path)
    if b.get("format") == "portable-v1":
        cfg = b["config"]
        if isinstance(cfg, dict):
            cfg = StrategyConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items()
                                    if k in StrategyConfig.__dataclass_fields__})
        meta = dict(b["touch_meta"])
        best = meta.pop("best")
        touch = {"touch": (_xgb_from_bytes(b["touch_xgb_json"]), best), **meta}
        d = _xgb_from_bytes(b["dir_xgb_json"])
        return {**{k: v for k, v in b.items() if not k.endswith("_json")}, "config": cfg, "touch": touch, "dir": d}
    b["config"] = upgrade_config(b["config"])
    return b


def predict_live(bundle: dict, candles_1h: pd.DataFrame, positioning_5m: pd.DataFrame,
                 funding: pd.DataFrame | None = None, last_n: int = 1) -> pd.DataFrame:
    """Tín hiệu cho các nến 1h ĐÃ ĐÓNG gần nhất (cần >= khoảng 65 ngày nến 1h và bảng vị thế 5m tương ứng).
    Trả về: p_touch, p_up, signal, side, mức vào dự kiến (close), TP / SL theo giá close (lệnh thật vào ở open nến sau)."""
    cfg: StrategyConfig = bundle["config"]
    df = mv2.prepare_candles(candles_1h, 60)
    if funding is not None:
        df = mv2.attach_funding(df, funding, 60)
    sd = build_data(df, positioning_5m, cfg, live=True, live_last=last_n)
    if tuple(sd.dir_names) != tuple(bundle["dir_names"]):
        raise ValueError("feature khác bundle (thiếu taker / funding / vị thế?)")
    models = {"touch": bundle["touch"], "dir": bundle["dir"], "thr_touch": bundle["thr_touch"],
              "thr_conf": bundle["thr_conf"], "dir_cols": bundle.get("dir_cols")}
    P = predict(models, sd, np.arange(len(sd.ds.decision_idx)))
    P["tp_price"] = P.close * np.exp(P.side * cfg.tp_mult * P.b)
    P["sl_price"] = P.close * np.exp(-P.side * cfg.sl_mult * P.b)
    return P.drop(columns=["sample"])
