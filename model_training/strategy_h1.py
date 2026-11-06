"""BTCUSDT H1 - chiến lược TOUCH + DIRECTION (cấu hình chọn ngày 2026-11-03, xem report.md).

Mỗi giờ (nến 1h t vừa đóng):
  1. TOUCH (modelv2): P(giá chạm +b hoặc -b trong 8 nến 1h tới, tính từ open[t+1]),
     b = clip(sigma_EWM168 * sqrt(8), 1%, 5%). 92 feature giá, pivot ZigZag 0.75%.
  2. DIRECTION: XGBoost 3 lớp trên MỌI giờ: chạm +b trước / chạm -b trước / không chạm (hoặc chạm cả 2 mức trong
     cùng 1 nến). Điểm hướng = P(lên trước) - P(xuống trước) = kỳ vọng R của lệnh 1:1 trong 8h; lưu dưới dạng
     p_up = 0.5 + 0.5 * điểm. 37 feature: 20 có hướng (giá, dòng lệnh taker, funding, pivot 1%) + 17 vị thế
     (open interest, tỉ lệ long / short, premium).
  3. TÍN HIỆU, ngưỡng TRƯỢT (không nhìn tương lai):
       touch:     p_touch >= phân vị (1 - touch_top) của p_touch 14 ngày trước;
       direction: trong các nến 90 ngày trước cùng qua cổng touch, LONG nếu (p_up - 0.5) >= phân vị (1 - dir_top/2)
                  của (p_up - 0.5), SHORT tương tự với (0.5 - p_up) -> mỗi phía khoảng dir_top/2 (cân bằng 2 phía).
  4. LỆNH (thực thi trên nến 5m): limit lùi 0.05b so với open, chờ 10 phút, không khớp thì market; TP = SL = 1b
     (khớp nhãn), giữ tối đa 9h; tối đa 5 lệnh, các lần vào cách >= 1 giờ. Phí maker 0.02% (limit) / taker 0.05%.

Huấn luyện / kiểm định: walk-forward theo QUÝ từ 2022Q1, ensemble 5 seed. Quý Q: train = nhãn kết thúc trước
(Q - 2 quý); 2 quý trước Q = validation (nửa đầu dừng sớm, nửa sau temperature scaling của touch); dự báo cho
90 ngày trước Q + Q (để tính ngưỡng trượt). Chọn tham số giao dịch chỉ trên dev 2022-2024 (0.5 TB năm + 0.5 năm kém,
>= min_trades_per_year lệnh / năm, điểm vùng lân cận), khoá, rồi test 2025-2026.

Dùng (từ thư mục gốc BTC_PIPELINE, xem model_training/train_strategy_h1.ipynb):
    import strategy_h1 as st
    cfg = st.StrategyConfig(); D = st.build(cfg)
    P_dev, R_dev = st.walk_forward(D, cfg, 'dev'); G = st.grid_search(D, P_dev, cfg)
    locked = st.load_locked(st.lock_config(cfg, st.select_best(G), G).locked_from)
    P_test, R_test = st.walk_forward(D, locked, 'test'); res = st.evaluate(D, P_test, locked, 'test')
    st.export_bundle(D, locked)                                   # bundle live (quý mới nhất)
    st.predict_live(st.load_bundle(path), candles_1h, positioning_5m, funding, history)
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

import modelv2 as mv2
from modelv2 import njit

STRATEGY = "touch_direction_h1_v2"
BAR_MIN, BPD = 60, 24
NS_MIN, NS_DAY = 60 * 10**9, 86_400 * 10**9
DIRECTION_XGB_DEFAULT = dict(n_estimators=1500, learning_rate=0.02, max_depth=3, min_child_weight=50, subsample=0.7,
                             colsample_bytree=0.6, reg_lambda=10, early_stopping_rounds=80)
GRID = {"touch_top": (0.2, 0.3, 0.5, 1.0),          # 1.0 = không lọc touch
        "dir_top": (0.05, 0.10, 0.20),
        "max_positions": (3, 5),
        "min_gap_bars": (1, 2),
        "entry_mode": ("next_open", "limit_pullback")}
NUMERIC_AXES = ("touch_top", "dir_top", "max_positions", "min_gap_bars")
KIND_NAMES = {1: "TP", -1: "SL", 3: "hết giờ", 4: "cắt cuối kỳ"}
DIRECTION_LABELS = ("3class", "ret8", "tb_sign")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class StrategyConfig:
    csv_5m: str = "historical_data/BTCUSDT_5m_all.csv"
    positioning_csv: str = "historical_data/BTCUSDT_positioning_5m.csv"
    funding_csv: str = "historical_data/BTCUSDT_funding.csv"
    output_dir: str = "artifacts/1h_strategy_v2"
    data_start: str = "2020-01-01"
    # chia dữ liệu (walk-forward theo quý)
    dev_end: str = "2025-01-01"                 # mọi lựa chọn tham số chỉ dùng dữ liệu trước mốc này
    first_dev_quarter: str = "2022Q1"
    first_test_quarter: str = "2025Q1"
    last_test_quarter: str | None = None        # None = tới hết dữ liệu
    val_quarters: int = 2
    val_es_frac: float = 0.5
    seeds: tuple[int, ...] = (1000, 1001, 1002, 1003, 1004)
    # touch
    horizon_hours: int = 8
    lookback_bars: int = 168
    vol_span: int = 168
    slow_vol_span: int = 720
    b_mult: float = 1.0
    min_barrier: float = 0.01
    max_barrier: float = 0.05
    touch_pivot_reversal: float = 0.0075
    time_decay_halflife_days: float = 540.0
    touch_xgb: dict = field(default_factory=dict)        # ghi đè modelv2.XGBParams
    # direction
    pivot_reversal: float = 0.01                         # pivot cho d_last_pivot_kind / d_pivot_hhll
    direction_xgb: dict = field(default_factory=dict)    # ghi đè DIRECTION_XGB_DEFAULT
    direction_label: str = "3class"                      # nhãn Direction, xem DIRECTION_LABELS
    xgb_device: str = "cpu"
    # model: xgb (production) | tabm | lstm (nn_models.py, chỉ để so sánh; bundle live chỉ hỗ trợ xgb)
    model_kind: str = "xgb"
    nn_params: dict = field(default_factory=dict)        # ghi đè nn_models.TabMParams / LSTMParams
    nn_device: str = "auto"                              # auto = cuda nếu có
    nn_threads: int = 4
    # Direction gộp nhiều model: p_up = 0.5 + sum w_k * (p_up_k - 0.5), trộn theo từng seed. {} = chỉ model_kind.
    # Production: {"xgb": 0.5, "lstm": 0.5} (exp_direction_blend.ipynb). Touch luôn là model_kind (xgb).
    direction_blend: dict = field(default_factory=lambda: {"xgb": 0.5, "lstm": 0.5})
    # tín hiệu (ngưỡng trượt)
    touch_gate_days: int = 14
    dir_window_days: int = 90                            # cũng là lịch sử dự báo trước mỗi quý
    thr_min_frac: float = 0.5                            # cần >= 50% số nến của cửa sổ mới tính ngưỡng
    dir_balance: bool = True
    gate_order: str = "touch_first"                      # touch_first | dir_first (xem rolling_thresholds)
    touch_top: float = 0.2
    dir_top: float = 0.10
    # lệnh (thực thi trên nến 5m)
    tp_mult: float = 1.0
    sl_mult: float = 1.0
    max_hold_hours: int = 9
    max_positions: int = 5
    min_gap_bars: int = 1
    entry_mode: str = "limit_pullback"                   # "next_open" | "limit_pullback"
    pullback_mult: float = 0.05
    pullback_minutes: int = 10
    fill_bp: float = 0.0001                              # giá phải xuyên qua limit >= 1 bp mới tính là khớp
    max_entry_delay_min: int = 15
    fee_maker: float = 0.0002
    fee_taker: float = 0.0005
    risk_per_trade: float = 0.01
    # chọn tham số
    min_trades_per_year: int = 150
    select_weight_worst: float = 0.5
    locked_from: str = ""

    def __post_init__(self):
        if self.entry_mode not in ("next_open", "limit_pullback"):
            raise ValueError("entry_mode: next_open | limit_pullback")
        if self.direction_label not in DIRECTION_LABELS:
            raise ValueError(f"direction_label: {' | '.join(DIRECTION_LABELS)}")
        if self.model_kind not in ("xgb", "tabm", "lstm"):
            raise ValueError("model_kind: xgb | tabm | lstm")
        if any(k not in ("xgb", "tabm", "lstm") for k in self.direction_blend):
            raise ValueError("direction_blend: khoá xgb | tabm | lstm")
        if self.gate_order not in ("touch_first", "dir_first"):
            raise ValueError("gate_order: touch_first | dir_first")
        if not 1 <= self.touch_gate_days <= self.dir_window_days:
            raise ValueError("cần 1 <= touch_gate_days <= dir_window_days")
        self.seeds = tuple(self.seeds)

    @property
    def horizon(self) -> int:
        return self.horizon_hours * 60 // BAR_MIN

    def touch_cfg(self) -> mv2.Config:
        return mv2.Config(bar_minutes=BAR_MIN, lookback_bars=self.lookback_bars, barrier_horizon=self.horizon,
                          barrier_vol_mult=self.b_mult, min_barrier=self.min_barrier, max_barrier=self.max_barrier,
                          vol_span=self.vol_span, slow_vol_span=self.slow_vol_span, xgb_device=self.xgb_device,
                          time_decay_halflife_days=self.time_decay_halflife_days, touch=mv2.XGBParams(**self.touch_xgb))


MODEL_FIELDS = ("data_start", "horizon_hours", "lookback_bars", "vol_span", "slow_vol_span", "b_mult", "min_barrier",
                "max_barrier", "touch_pivot_reversal", "time_decay_halflife_days", "touch_xgb", "pivot_reversal",
                "direction_xgb", "direction_label", "val_quarters", "val_es_frac", "dir_window_days")


def _utc(s) -> pd.Timestamp:
    t = pd.Timestamp(s)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _ns(s) -> int:
    return int(_utc(s).value)


def _dti(t_ns) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(t_ns, dtype=np.int64), unit="ns"))


# --------------------------------------------------------------------------- #
# Dữ liệu và feature
# --------------------------------------------------------------------------- #
def aggregate_1h(candles_5m: pd.DataFrame) -> pd.DataFrame:
    """Gộp nến 5m thành 1h (giữ taker_buy_volume, trades), chỉ giữ giờ đủ 12 nến 5m. Dùng chung train và live."""
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


def positioning_features(candles_1h: pd.DataFrame, positioning_5m: pd.DataFrame) -> pd.DataFrame:
    """17 feature vị thế theo giờ: giá trị đã biết lúc nến 1h đóng (dòng 5m cuối giờ). Bảng vị thế: positioning_data."""
    P = positioning_5m.copy()
    P["t"] = pd.to_datetime(P["open_time_ms"], unit="ms", utc=True)
    P = P.set_index("t").sort_index()
    closes = pd.to_datetime(candles_1h["timestamp"], utc=True) + pd.Timedelta(minutes=55)
    Q = P.reindex(closes).reset_index(drop=True)
    lg = lambda s: np.log(s.where(s > 0))                                                # noqa: E731
    F = pd.DataFrame(index=candles_1h.index)
    oi = lg(Q["oi"])
    for k in (1, 4, 24):
        F[f"p_oi_{k}h"] = oi - oi.shift(k)                                   # tiền mới vào / đóng vị thế
    F["p_oi4_x_ret4"] = F["p_oi_4h"] * np.sign(np.log(candles_1h["close"].reset_index(drop=True)).diff(4))
    for nm in ("top_acct_ls", "top_pos_ls", "acct_ls"):                      # tỉ lệ long / short (log)
        x = lg(Q[nm])
        F[f"p_{nm}"] = x
        F[f"p_{nm}_z7d"] = (x - x.rolling(168, min_periods=100).mean()) / x.rolling(168, min_periods=100).std()
        F[f"p_{nm}_d4"] = x - x.shift(4)
    F["p_top_minus_crowd"] = F["p_top_pos_ls_z7d"] - F["p_acct_ls_z7d"]       # top trader ngược đám đông
    pr = Q["prem_close"] * 1e4                                                # premium futures / spot (bp)
    F["p_prem"] = pr
    F["p_prem_24"] = pr.rolling(24, min_periods=12).mean()
    F["p_prem_z7d"] = (pr - pr.rolling(168, min_periods=100).mean()) / pr.rolling(168, min_periods=100).std()
    return F


def directional_features(df: pd.DataFrame, ts: np.ndarray, pivots: pd.DataFrame, bf: dict) -> dict:
    """20 feature có hướng tại các nến quyết định ts (đơn vị sigma 1h)."""
    high, low, close, vol = (df[c].to_numpy(np.float64) for c in ("high", "low", "close", "volume"))
    lc, sig = bf["logc"], bf["sigma"]
    f = {}

    def roll(x, w, how):
        return getattr(pd.Series(x).rolling(w, min_periods=w), how)().to_numpy()

    for k in (1, 3, 6, 12, 24):                                              # đà giá k giờ
        f[f"d_ret_{k}"] = (lc[ts] - lc[ts - k]) / (sig[ts] * math.sqrt(k))
    rng = np.maximum(high[ts] - low[ts], 1e-12)
    f["d_close_loc"] = ((close[ts] - low[ts]) / rng - 0.5) * 2                # đóng sát đỉnh (+1) / đáy (-1)
    if "taker_buy_volume" in df.columns:                                     # mua - bán chủ động
        imb = 2 * df["taker_buy_volume"].to_numpy(np.float64) / np.maximum(vol, 1e-12) - 1
        for k in (3, 6, 12, 24):
            f[f"d_taker_{k}"] = roll(imb, k, "mean")[ts]
    if "funding_rate" in df.columns:
        f["d_funding"] = df["funding_rate"].to_numpy(np.float64)[ts] * 1e4
        f["d_funding_z"] = df["funding_z_90"].to_numpy(np.float64)[ts]
    for days in (1, 3, 7):                                                   # khoảng cách tới đỉnh / đáy gần đây
        w = BPD * days
        f[f"x{days}d_up"] = (np.log(roll(high, w, "max")[ts]) - lc[ts]) / sig[ts]
        f[f"x{days}d_dn"] = (lc[ts] - np.log(roll(low, w, "min")[ts])) / sig[ts]
    conf = pivots["confirm_idx"].to_numpy(np.int64)
    ke = np.searchsorted(conf, ts, side="right")
    kind = pivots["kind"].to_numpy(np.float64)
    price = pivots["price"].to_numpy(np.float64)
    last = np.maximum(ke - 1, 0)
    f["d_last_pivot_kind"] = -kind[last]                                      # +1: pivot cuối là đáy (đang sóng lên)
    f["d_pivot_hhll"] = np.log(price[last] / price[np.maximum(ke - 3, 0)]) / sig[ts]   # HH / LL
    return {k: np.where(np.isfinite(v), v, np.nan).astype(np.float32) for k, v in f.items()}


@dataclass
class Data:
    cfg: StrategyConfig
    df: pd.DataFrame            # nến 1h (+ funding)
    d5: pd.DataFrame | None     # nến 5m để thực thi lệnh (None ở live)
    ds: mv2.Dataset             # touch: feature + nhãn
    xd: np.ndarray              # direction: 37 feature
    dir_names: tuple


def build_features(df: pd.DataFrame, positioning_5m: pd.DataFrame, cfg: StrategyConfig, live: bool = False,
                   live_last: int | None = None) -> tuple:
    """(ds touch, ma trận direction, tên feature direction) cho nến 1h df (đã gắn funding)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tc = cfg.touch_cfg()
        ds = mv2.build_dataset(df, mv2.find_confirmed_pivots(df, cfg.touch_pivot_reversal), tc, live=live,
                               live_last=live_last)
        bf = mv2.compute_bar_features(df, tc)
        ex = directional_features(df, ds.decision_idx, mv2.find_confirmed_pivots(df, cfg.pivot_reversal), bf)
        PF = positioning_features(df, positioning_5m)
        X = np.hstack([ex[k][:, None] for k in ex] + [PF.to_numpy(np.float32)[ds.decision_idx]]).astype(np.float32)
    return ds, X, tuple(ex) + tuple(PF.columns)


def build(cfg: StrategyConfig | None = None, log=print) -> Data:
    cfg = StrategyConfig() if cfg is None else cfg
    t0 = time.time()
    c5 = pd.read_csv(cfg.csv_5m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d5 = mv2.prepare_candles(c5, 5)
        df = aggregate_1h(c5)
    del c5
    start = _utc(cfg.data_start)
    d5 = d5[d5["timestamp"] >= start].reset_index(drop=True)
    df = df[df["timestamp"] >= start].reset_index(drop=True)
    df = mv2.attach_funding(df, pd.read_csv(cfg.funding_csv), BAR_MIN)
    ds, X, names = build_features(df, pd.read_csv(cfg.positioning_csv), cfg)
    log(f"Dữ liệu: {len(df):,} nến 1h, {len(ds.decision_idx):,} mẫu "
        f"({pd.to_datetime(ds.timestamp_ns[0], utc=True):%Y-%m-%d} -> {pd.to_datetime(ds.timestamp_ns[-1], utc=True):%Y-%m-%d}), "
        f"touch {ds.x.shape[1]} feature, direction {X.shape[1]} feature, {time.time() - t0:.0f}s")
    return Data(cfg=cfg, df=df, d5=d5, ds=ds, xd=X, dir_names=names)


def data_report(D: Data) -> pd.DataFrame:
    """b, tỉ lệ chạm, tỉ lệ 'chạm +b trước' theo năm."""
    lab = D.ds.labels
    T = pd.DataFrame({"năm": pd.to_datetime(D.ds.timestamp_ns, utc=True).year, "b": D.ds.barrier_pct * 100,
                      "touch": lab["y_touch"], "up": lab["first"] == 1, "tie": (lab["y_touch"] == 1) & (lab["first"] == 0)})
    g = T.groupby("năm")
    return pd.DataFrame({"mẫu": g.size(), "b TB %": g.b.mean(), "tỉ lệ chạm": g.touch.mean(),
                         "chạm +b trước / đã chạm": g.apply(lambda x: x.up.sum() / max(x.touch.sum() - x.tie.sum(), 1)),
                         "chạm 2 mức cùng nến": g.tie.mean()}).round(4)


# --------------------------------------------------------------------------- #
# Walk-forward theo quý
# --------------------------------------------------------------------------- #
def quarters(cfg: StrategyConfig, phase: str, D: Data | None = None) -> list[pd.Period]:
    """dev: first_dev_quarter .. quý cuối trước dev_end | test: first_test_quarter .. hết dữ liệu |
    frozen: [quý train model, quý test cuối]."""
    if phase == "dev":
        return list(pd.period_range(pd.Period(cfg.first_dev_quarter, "Q"),
                                    pd.Period(_utc(cfg.dev_end).tz_localize(None) - pd.Timedelta(days=1), "Q"), freq="Q"))
    if phase not in ("test", "frozen"):
        raise ValueError(phase)
    a = pd.Period(cfg.first_test_quarter, "Q")
    b = (pd.Period(cfg.last_test_quarter, "Q") if cfg.last_test_quarter
         else pd.Period(pd.to_datetime(D.ds.timestamp_ns[-1], utc=True).tz_localize(None), "Q"))
    return [a, b] if phase == "frozen" else list(pd.period_range(a, b, freq="Q"))


def fold_split(D: Data, q: pd.Period, cfg: StrategyConfig, test_end_ns: int | None = None) -> dict:
    ts, end = D.ds.timestamp_ns, D.ds.label_end_ns
    qs = _ns(q.start_time)
    qe = _ns((q + 1).start_time) if test_end_ns is None else test_end_ns
    vs = _ns((q - cfg.val_quarters).start_time)
    ids = np.arange(len(ts))
    train = ids[(end < vs) & (ts >= _ns(cfg.data_start))]
    val = ids[(ts >= vs) & (end < qs)]
    test = ids[(ts >= qs) & (ts < qe)]
    window = ids[(ts >= qs - cfg.dir_window_days * NS_DAY) & (ts < qe)]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(f"{q}: thiếu dữ liệu (train {len(train)}, val {len(val)}, test {len(test)})")
    c = int(len(val) * cfg.val_es_frac)
    es, cal = val[:c], val[c:]
    es = es[end[es] < ts[cal[0]]]
    return {"q": q, "train": train, "es": es, "cal": cal, "test": test, "window": window,
            "test_start_ns": qs, "test_end_ns": qe}


def direction_labels(ds: mv2.Dataset, kind: str = "3class") -> np.ndarray:
    """Nhãn Direction trên MỌI nến (H nến kể từ open[t+1]):
    3class : 0 = không chạm (hoặc chạm cả 2 mức trong cùng nến), 1 = chạm +b trước, 2 = chạm -b trước.
    ret8   : 1 = close[t+H] > open[t+1] (giá đi lên sau H nến), 0 = không.
    tb_sign: triple barrier 2 lớp: đã chạm -> mức chạm trước; không chạm / chạm cùng nến -> dấu lợi suất cuối H nến."""
    lab = ds.labels
    touched = (lab["y_touch"] == 1) & (lab["first"] != 0)
    up_end = lab["r_end"] > 0
    if kind == "3class":
        return np.where(touched & (lab["first"] == 1), 1, np.where(touched & (lab["first"] == -1), 2, 0)).astype(np.int64)
    if kind == "ret8":
        return up_end.astype(np.int64)
    if kind == "tb_sign":
        return np.where(touched, lab["first"] == 1, up_end).astype(np.int64)
    raise ValueError(kind)


def direction_model(cfg: StrategyConfig, seed: int):
    from xgboost import XGBClassifier
    kw = {"device": cfg.xgb_device} if cfg.xgb_device != "cpu" else {}
    obj = (dict(objective="multi:softprob", eval_metric="mlogloss") if cfg.direction_label == "3class"
           else dict(objective="binary:logistic", eval_metric="logloss"))
    return XGBClassifier(**{**DIRECTION_XGB_DEFAULT, **cfg.direction_xgb}, **obj, n_jobs=-1, random_state=seed,
                         tree_method="hist", **kw)


def direction_proba(model, best: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(p_up, p_dtouch). 3 lớp: p_up = 0.5 + 0.5 * (P(lên trước) - P(xuống trước)), p_dtouch = 1 - P(không chạm).
    2 lớp: p_up = P(lên), p_dtouch = NaN."""
    pr = model.predict_proba(x, iteration_range=(0, best))
    if pr.shape[1] == 3:
        return 0.5 + 0.5 * (pr[:, 1] - pr[:, 2]), 1.0 - pr[:, 0]
    return pr[:, 1], np.full(len(pr), np.nan)


def direction_score(model, best: int, x: np.ndarray) -> np.ndarray:
    return direction_proba(model, best, x)[0]


def fit_direction(D: Data, sp: dict, cfg: StrategyConfig, seed: int) -> tuple:
    y = direction_labels(D.ds, cfg.direction_label)
    tr, es = sp["train"], sp["es"]
    dm = direction_model(cfg, seed)
    dm.fit(D.xd[tr], y[tr], eval_set=[(D.xd[es], y[es])], verbose=False)
    return dm, int(dm.best_iteration) + 1


def fit_fold(D: Data, sp: dict, cfg: StrategyConfig, seed: int) -> dict:
    """Train touch + direction cho 1 quý và 1 seed; dự báo trên cửa sổ [test_start - dir_window, test_end)."""
    ds, tc = D.ds, cfg.touch_cfg()
    y = ds.labels["y_touch"]
    tr, es, cal, win = sp["train"], sp["es"], sp["cal"], sp["window"]
    w = mv2._time_weights(ds, tr, int(ds.timestamp_ns[tr].max()), tc)
    tm = mv2.make_xgb(tc.touch, tc, seed)
    tm.fit(ds.x[tr], y[tr], sample_weight=w, eval_set=[(ds.x[es], y[es])], verbose=False)
    touch = {"touch": (tm, int(getattr(tm, "best_iteration", tc.touch.n_estimators - 1)) + 1), "T_touch": 1.0}
    touch["T_touch"] = mv2.fit_temperature(mv2.logit(mv2.raw_proba(touch, ds.x[cal])), y[cal])
    dm, dbest = fit_direction(D, sp, cfg, seed)
    pu, pdt = direction_proba(dm, dbest, D.xd[win])
    return {"p_touch": mv2.predict_touch(touch, ds.x[win]).astype(np.float32),
            "p_up": pu.astype(np.float32),
            "p_dtouch": pdt.astype(np.float32),        # P(chạm) theo chính Direction (chỉ bản 3 lớp)
            "meta": {"touch_rounds": touch["touch"][1], "dir_rounds": dbest, "T_touch": float(touch["T_touch"]),
                     "n_train": int(len(tr))},
            "models": {"touch": touch, "direction": (dm, dbest)}}


def _cache_key(D: Data, cfg: StrategyConfig, sp: dict, seed: int) -> str:
    d = {"c": {k: getattr(cfg, k) for k in MODEL_FIELDS}, "seed": seed, "q": str(sp["q"]),
         "touch": list(D.ds.names), "dir": list(D.dir_names), "n": len(D.ds.timestamp_ns),
         "tr": [int(sp["train"][0]), int(sp["train"][-1]), len(sp["train"])],
         "win": [int(sp["window"][0]), int(sp["window"][-1])]}
    if cfg.model_kind != "xgb":                          # khoá cache của xgb giữ nguyên như trước
        import nn_models
        d["nn"] = nn_models.params_dict(cfg)
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()


def walk_forward(D: Data, cfg: StrategyConfig, phase: str = "dev", seeds=None, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train lại mỗi quý ('dev' / 'test') hoặc 1 bộ model cho toàn bộ giai đoạn test ('frozen').
    Trả về (P: dự báo trên cửa sổ của mọi quý, R: báo cáo theo quý). Kết quả quý x seed cache ở output_dir/folds."""
    from sklearn.metrics import roc_auc_score
    if phase != "dev" and not cfg.locked_from:
        raise RuntimeError("phase test / frozen chỉ chạy với config đã khoá (lock_config / load_locked)")
    seeds = tuple(cfg.seeds if seeds is None else seeds)
    out = Path(cfg.output_dir) / "folds"
    out.mkdir(parents=True, exist_ok=True)
    ds, lab = D.ds, D.ds.labels
    dev_end = _ns(cfg.dev_end)
    metric_end = dev_end if phase == "dev" else int(ds.label_end_ns.max()) + 1
    qlist = quarters(cfg, phase, D)
    test_end = None
    if phase == "frozen":
        test_end = min(_ns((qlist[1] + 1).start_time), int(ds.timestamp_ns.max()) + 1)
        qlist = qlist[:1]
    frames, reps = [], []
    for q in qlist:
        sp = fold_split(D, q, cfg, test_end)
        if phase == "dev":
            assert ds.label_end_ns[np.r_[sp["train"], sp["es"], sp["cal"]]].max() < dev_end
            assert ds.timestamp_ns[sp["test"]].max() < dev_end
        win = sp["window"]
        pt, pu, pd_, metas = [], [], [], []
        t0 = time.time()
        for s in seeds:
            f = out / (f"{phase}_{q}_s{s}.npz" if cfg.model_kind == "xgb" else f"{cfg.model_kind}_{phase}_{q}_s{s}.npz")
            key = _cache_key(D, cfg, sp, s)
            if f.exists():
                z = np.load(f, allow_pickle=False)
                if str(z["key"]) == key and "p_dtouch" in z.files:
                    pt.append(z["p_touch"]); pu.append(z["p_up"]); pd_.append(z["p_dtouch"])
                    metas.append(json.loads(str(z["meta"])))
                    continue
            if cfg.model_kind == "xgb":
                r = fit_fold(D, sp, cfg, s)
            else:
                import nn_models
                r = nn_models.fit_fold_nn(D, sp, cfg, s)
            np.savez_compressed(f, key=key, p_touch=r["p_touch"], p_up=r["p_up"], p_dtouch=r["p_dtouch"],
                                meta=json.dumps(r["meta"]))
            pt.append(r["p_touch"]); pu.append(r["p_up"]); pd_.append(r["p_dtouch"]); metas.append(r["meta"])
        F = pd.DataFrame({"fold": str(q), "sample": win, "t_ns": ds.timestamp_ns[win], "label_end_ns": ds.label_end_ns[win],
                          "close": ds.close[win], "b": ds.barrier_pct[win], "y_touch": lab["y_touch"][win],
                          "first": lab["first"][win],
                          "is_test": (ds.timestamp_ns[win] >= sp["test_start_ns"]) & (ds.timestamp_ns[win] < sp["test_end_ns"])})
        for s, a, b, c in zip(seeds, pt, pu, pd_):
            F[f"p_touch_s{s}"], F[f"p_up_s{s}"], F[f"p_dtouch_s{s}"] = a, b, c
        F["p_touch"], F["p_up"], F["p_dtouch"] = np.mean(pt, axis=0), np.mean(pu, axis=0), np.mean(pd_, axis=0)
        frames.append(F)
        T = F[F.is_test & (F.label_end_ns < metric_end)]
        ok = (T.y_touch == 1) & (T["first"] != 0)
        rep = {"quý": str(q), "mẫu test": int(len(T)), "tỉ lệ chạm": float(T.y_touch.mean()), "b TB %": float(T.b.mean() * 100),
               "AUC touch": float(roc_auc_score(T.y_touch, T.p_touch)) if T.y_touch.nunique() == 2 else np.nan,
               "AUC touch của Direction": (float(roc_auc_score(T.y_touch, T.p_dtouch))
                                           if T.y_touch.nunique() == 2 and T.p_dtouch.notna().all() else np.nan),
               "AUC direction (đã chạm)": float(roc_auc_score(T["first"][ok] == 1, T.p_up[ok])) if ok.sum() > 50 else np.nan,
               "n train": metas[0]["n_train"], "cây touch TB": float(np.mean([m["touch_rounds"] for m in metas])),
               "cây direction TB": float(np.mean([m["dir_rounds"] for m in metas])), "giây": round(time.time() - t0)}
        reps.append(rep)
        log(f"{phase} {q}: AUC touch {rep['AUC touch']:.3f} | AUC direction {rep['AUC direction (đã chạm)']:.3f} | "
            f"train {rep['n train']:,} | {rep['giây']}s")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(reps)


def blend_direction(parts: dict, weights: dict) -> pd.DataFrame:
    """parts: {model: P của walk_forward cùng phase}. Lấy touch (p_touch*) của parts[touch] với touch = model đầu tiên
    của parts; p_up (ensemble + từng seed) = 0.5 + sum w_k * (p_up_k - 0.5)."""
    keys = list(parts)
    base = parts[keys[0]]
    for k in keys[1:]:
        if not np.array_equal(parts[k].t_ns.to_numpy(), base.t_ns.to_numpy()):
            raise ValueError(f"{k}: dự báo không cùng hàng với {keys[0]}")
    Q = base.copy()
    for c in [c for c in base.columns if c == "p_up" or c.startswith("p_up_s")]:
        Q[c] = (0.5 + sum(w * (parts[k][c].to_numpy(np.float64) - 0.5) for k, w in weights.items())).astype(np.float32)
    return Q


def walk_forward_blend(D: Data, cfg: StrategyConfig, phase: str = "dev", seeds=None, log=print) -> tuple:
    """walk_forward cho touch (cfg.model_kind) và mọi model trong cfg.direction_blend, rồi gộp Direction.
    Không có direction_blend -> như walk_forward. Trả về (P gộp, R: báo cáo theo quý của từng model)."""
    if not cfg.direction_blend:
        return walk_forward(D, cfg, phase, seeds, log)
    kinds = list(dict.fromkeys([cfg.model_kind, *cfg.direction_blend]))
    parts, reps = {}, []
    for k in kinds:
        Pk, Rk = walk_forward(D, replace(cfg, model_kind=k), phase, seeds, log)
        parts[k] = Pk
        reps.append(Rk.assign(model=k))
    return blend_direction(parts, cfg.direction_blend), pd.concat(reps, ignore_index=True)


def walk_forward_direction(D: Data, cfg: StrategyConfig, P_base: pd.DataFrame, phase: str, seeds=None,
                           log=print) -> pd.DataFrame:
    """Chỉ train lại Direction (nhãn cfg.direction_label, model cfg.model_kind) trên đúng các fold của P_base, giữ nguyên
    p_touch. Trả về P_base với p_up (ensemble + từng seed) và p_dtouch thay bằng của Direction mới.
    Cache ở output_dir/folds/dir_{label}_{phase}_{quý}_s{seed}.npz."""
    seeds = tuple(cfg.seeds if seeds is None else seeds)
    out = Path(cfg.output_dir) / "folds"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for qs, F in P_base.groupby("fold", sort=False):
        q = pd.Period(qs, "Q")
        sp = fold_split(D, q, cfg, int(F.t_ns.max()) + 1 if phase == "frozen" else None)
        win = sp["window"]
        if not np.array_equal(win, F["sample"].to_numpy()):
            raise ValueError(f"{q}: cửa sổ khác P_base")
        F = F.copy()
        pu, pdt = [], []
        t0 = time.time()
        for s in seeds:
            pre = "" if cfg.model_kind == "xgb" else \
                f"{cfg.model_kind}{hashlib.sha1(json.dumps(cfg.nn_params, sort_keys=True).encode()).hexdigest()[:6]}_"
            f = out / f"dir_{pre}{cfg.direction_label}_{phase}_{q}_s{s}.npz"
            key = _cache_key(D, cfg, sp, s)
            if f.exists():
                z = np.load(f, allow_pickle=False)
                if str(z["key"]) == key:
                    pu.append(z["p_up"]); pdt.append(z["p_dtouch"])
                    continue
            if cfg.model_kind == "xgb":
                dm, dbest = fit_direction(D, sp, cfg, s)
                a, c = direction_proba(dm, dbest, D.xd[win])
            else:
                import nn_models
                a, c = nn_models.scores(nn_models.fit_direction_nn(D, sp, cfg, s).predict_proba(win))
            np.savez_compressed(f, key=key, p_up=a.astype(np.float32), p_dtouch=c.astype(np.float32))
            pu.append(a.astype(np.float32)); pdt.append(c.astype(np.float32))
        for s, a, c in zip(seeds, pu, pdt):
            F[f"p_up_s{s}"], F[f"p_dtouch_s{s}"] = a, c
        F["p_up"], F["p_dtouch"] = np.mean(pu, axis=0), np.mean(pdt, axis=0)
        frames.append(F)
        log(f"direction {cfg.model_kind} {cfg.direction_label} {phase} {q}: {time.time() - t0:.0f}s")
    return pd.concat(frames, ignore_index=True)


def direction_auc(D: Data, P: pd.DataFrame) -> pd.DataFrame:
    """AUC của p_up theo quý: hướng chạm trước (giờ đã chạm) và hướng 8h (mọi giờ: close[t+H] > open[t+1])."""
    from sklearn.metrics import roc_auc_score
    T = P[P.is_test]
    r_end = D.ds.labels["r_end"][T["sample"].to_numpy()]
    T = T.assign(up_end=r_end > 0)
    rows = []
    for q, g in T.groupby("fold", sort=False):
        ok = (g.y_touch == 1) & (g["first"] != 0)
        rows.append({"quý": q, "AUC chạm trước (đã chạm)": roc_auc_score(g["first"][ok] == 1, g.p_up[ok]),
                     "AUC hướng 8h (mọi giờ)": roc_auc_score(g.up_end, g.p_up)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Ngưỡng trượt -> tín hiệu
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _qtile(a, q):
    pos = q * (len(a) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(a) - 1)
    return a[lo] + (a[hi] - a[lo]) * (pos - lo)


@njit(cache=True)
def _touch_thresholds(t_ns, p, out_idx, W_ns, min_n, tts):
    """Ngưỡng touch của nến out_idx[k] = phân vị (1 - tt) của p_touch trong [t - W, t)."""
    thr = np.full((len(out_idx), len(tts)), np.nan)
    lo = 0
    for k in range(len(out_idx)):
        i = out_idx[k]
        while lo < i and t_ns[lo] < t_ns[i] - W_ns:
            lo += 1
        if i - lo < min_n:
            continue
        ps = np.sort(p[lo:i])
        for a in range(len(tts)):
            thr[k, a] = -1e300 if tts[a] >= 1.0 else _qtile(ps, 1.0 - tts[a])      # top 100% = không lọc
    return thr


@njit(cache=True)
def _dir_thresholds(t_ns, p, pu, out_idx, thr_tch, W_ns, min_n, dts, balance):
    """Ngưỡng direction của nến out_idx[k] trên các nến trong [t - W, t) có p_touch >= thr_tch[k]. d = p_up - 0.5.
    balance: LONG nếu d >= phân vị (1 - dt/2) của d, SHORT nếu -d >= phân vị (1 - dt/2) của -d.
    Không balance: |d| >= phân vị (1 - dt) của |d| (cùng ngưỡng 2 phía). Trả về (ngưỡng LONG, ngưỡng SHORT)."""
    n_out, nd = len(out_idx), len(dts)
    thL = np.full((n_out, nd), np.nan)
    thS = np.full((n_out, nd), np.nan)
    lo = 0
    for k in range(n_out):
        i = out_idx[k]
        while lo < i and t_ns[lo] < t_ns[i] - W_ns:
            lo += 1
        m = i - lo
        if m < min_n or not np.isfinite(thr_tch[k]):
            continue
        bufL, bufS = np.empty(m), np.empty(m)
        cnt = 0
        for u in range(lo, i):
            if p[u] >= thr_tch[k]:
                d = pu[u] - 0.5
                bufL[cnt] = d if balance else abs(d)
                bufS[cnt] = -d
                cnt += 1
        if cnt < 20:
            continue
        sL = np.sort(bufL[:cnt])
        sS = np.sort(bufS[:cnt])
        for b in range(nd):
            if balance:
                thL[k, b] = _qtile(sL, 1.0 - dts[b] / 2)
                thS[k, b] = _qtile(sS, 1.0 - dts[b] / 2)
            else:
                thL[k, b] = _qtile(sL, 1.0 - dts[b])
                thS[k, b] = thL[k, b]
    return thL, thS


@njit(cache=True)
def _touch_thresholds_masked(t_ns, p, mask, out_idx, W_ns, min_n, tt):
    """Ngưỡng touch của nến out_idx[k] = phân vị (1 - tt) của p_touch trên các nến trong [t - W, t) có mask."""
    thr = np.full(len(out_idx), np.nan)
    lo = 0
    for k in range(len(out_idx)):
        i = out_idx[k]
        while lo < i and t_ns[lo] < t_ns[i] - W_ns:
            lo += 1
        if i - lo < min_n:
            continue
        buf = np.empty(i - lo)
        cnt = 0
        for u in range(lo, i):
            if mask[u]:
                buf[cnt] = p[u]
                cnt += 1
        if cnt < 5:
            continue
        thr[k] = -1e300 if tt >= 1.0 else _qtile(np.sort(buf[:cnt]), 1.0 - tt)
    return thr


def rolling_thresholds(t_ns, p_touch, p_up, out_idx, cfg: StrategyConfig, touch_tops, dir_tops) -> tuple:
    """(ngưỡng touch [n, A, D], ngưỡng LONG [n, A, D], ngưỡng SHORT [n, A, D]) cho các nến out_idx (chỉ nhìn quá khứ).
    touch_first: ngưỡng touch trên mọi nến 14 ngày trước; ngưỡng Direction trên các nến qua touch của 90 ngày trước.
    dir_first  : ngưỡng Direction trên mọi nến 90 ngày trước; ngưỡng touch trên các nến 14 ngày trước đã qua Direction
                 (mỗi nến với ngưỡng của chính nó)."""
    t_ns, p_touch, p_up = (np.asarray(x) for x in (t_ns, p_touch, p_up))
    t_ns, p_touch, p_up = t_ns.astype(np.int64), p_touch.astype(np.float64), p_up.astype(np.float64)
    out_idx = np.asarray(out_idx, np.int64)
    tts, dts = np.asarray(touch_tops, np.float64), np.asarray(dir_tops, np.float64)
    W_t, n_t = cfg.touch_gate_days * NS_DAY, int(cfg.thr_min_frac * cfg.touch_gate_days * BPD)
    W_d, n_d = cfg.dir_window_days * NS_DAY, int(cfg.thr_min_frac * cfg.dir_window_days * BPD)
    if cfg.gate_order == "dir_first":
        allb = np.arange(len(t_ns), dtype=np.int64)
        L, S = _dir_thresholds(t_ns, p_touch, p_up, allb, np.full(len(t_ns), -1e300), W_d, n_d, dts, cfg.dir_balance)
        d = p_up - 0.5
        TT = np.full((len(out_idx), len(tts), len(dts)), np.nan)
        for b in range(len(dts)):
            ok = _side_mask(d, L[:, b], S[:, b])
            for a in range(len(tts)):
                TT[:, a, b] = _touch_thresholds_masked(t_ns, p_touch, ok, out_idx, W_t, n_t, float(tts[a]))
        TL = np.repeat(L[out_idx][:, None, :], len(tts), axis=1)
        TS = np.repeat(S[out_idx][:, None, :], len(tts), axis=1)
        return TT, TL, TS
    TT1 = _touch_thresholds(t_ns, p_touch, out_idx, W_t, n_t, tts)
    TT = np.repeat(TT1[:, :, None], len(dts), axis=2)
    TL = np.full((len(out_idx), len(tts), len(dts)), np.nan)
    TS = TL.copy()
    for a in range(len(tts)):
        TL[:, a, :], TS[:, a, :] = _dir_thresholds(t_ns, p_touch, p_up, out_idx, TT1[:, a].copy(), W_d, n_d, dts,
                                                   cfg.dir_balance)
    return TT, TL, TS


def _side_mask(d, thL, thS):
    with np.errstate(invalid="ignore"):
        return ((d > 0) & (d >= thL)) | ((d < 0) & (-d >= thS))


def make_signals(P: pd.DataFrame, cfg: StrategyConfig, touch_tops=None, dir_tops=None, suffix: str = "") -> dict:
    """Tín hiệu trên các nến test cho từng cặp (touch_top, dir_top). suffix '_s1000' = dùng riêng 1 seed.
    Trả về {"rows": nến test (side, conf, thr_touch / thr_long / thr_short của cfg), (tt, dt): mask}."""
    tts = list(touch_tops if touch_tops is not None else [cfg.touch_top])
    dts = list(dir_tops if dir_tops is not None else [cfg.dir_top])
    rows, TT, TL, TS = [], [], [], []
    for _, F in P.groupby("fold", sort=False):
        F = F.sort_values("t_ns")
        p, pu = F[f"p_touch{suffix}"].to_numpy(np.float64), F[f"p_up{suffix}"].to_numpy(np.float64)
        idx = np.flatnonzero(F.is_test.to_numpy())
        tt, tl, tsh = rolling_thresholds(F.t_ns.to_numpy(), p, pu, idx, cfg, tts, dts)
        R = F.iloc[idx][["fold", "sample", "t_ns", "close", "b", "y_touch", "first"]].copy()
        R["p_touch"], R["p_up"] = p[idx], pu[idx]
        rows.append(R); TT.append(tt); TL.append(tl); TS.append(tsh)
    R = pd.concat(rows, ignore_index=True)
    TT, TL, TS = np.vstack(TT), np.concatenate(TL), np.concatenate(TS)
    d = R.p_up.to_numpy() - 0.5
    R["conf"], R["side"] = np.abs(d), np.where(d > 0, 1, -1)
    out = {"rows": R}
    with np.errstate(invalid="ignore"):
        for a, t in enumerate(tts):
            for b, dd in enumerate(dts):
                out[(float(t), float(dd))] = (R.p_touch.to_numpy() >= TT[:, a, b]) & _side_mask(d, TL[:, a, b], TS[:, a, b])
    a, b = tts.index(cfg.touch_top) if cfg.touch_top in tts else 0, dts.index(cfg.dir_top) if cfg.dir_top in dts else 0
    R["thr_touch"], R["thr_long"], R["thr_short"] = TT[:, a, b], TL[:, a, b], TS[:, a, b]
    return out


# --------------------------------------------------------------------------- #
# Backtest trên nến 5m
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _simulate(o, h, l, c, start, side, b, tp_mult, sl_mult, maxb, mode, pb_mult, pb_bars, fill_bp, end_idx):
    """mode 0: vào ở open nến 5m start. mode 1: limit lùi pb_mult*b so với open[start], chờ pb_bars nến 5m, không khớp
    thì vào market ở open nến kế tiếp. Nến khớp limit giữa nến: chỉ xét SL ở nến đó (không biết thứ tự trong nến).
    Mỗi nến: SL trước (gap qua SL -> khớp ở open), rồi TP. kind: 1 TP, -1 SL, 3 hết giờ, 4 cắt ở end_idx."""
    m = len(start)
    ent = np.full(m, -1, np.int64)
    epx = np.full(m, np.nan)
    maker = np.zeros(m, np.bool_)
    ex = np.full(m, -1, np.int64)
    xpx = np.full(m, np.nan)
    kind = np.zeros(m, np.int64)
    R = np.full(m, np.nan)
    for j in range(m):
        s, i0 = side[j], start[j]
        sl, tp = sl_mult * b[j], tp_mult * b[j]
        intrabar = False
        if i0 > end_idx:
            continue
        if mode == 1:
            lim = o[i0] * math.exp(-s * pb_mult * b[j])
            filled, e = -1, 0.0
            for k in range(i0, min(i0 + pb_bars, end_idx + 1)):
                if (s > 0 and o[k] <= lim) or (s < 0 and o[k] >= lim):
                    filled, e = k, o[k]
                    break
                if (s > 0 and l[k] <= lim * (1 - fill_bp)) or (s < 0 and h[k] >= lim * (1 + fill_bp)):
                    filled, e, intrabar = k, lim, True
                    break
            if filled >= 0:
                ie = filled
                maker[j] = True
            else:
                ie = i0 + pb_bars
                if ie > end_idx:
                    continue
                e = o[ie]
        else:
            ie, e = i0, o[i0]
        ent[j], epx[j] = ie, e
        tpp, slp = e * math.exp(s * tp), e * math.exp(-s * sl)
        k_end = min(end_idx + 1, ie + maxb)
        done = False
        for k in range(ie, k_end):
            if (s > 0 and l[k] <= slp) or (s < 0 and h[k] >= slp):
                px = slp
                if (s > 0 and o[k] < slp) or (s < 0 and o[k] > slp):
                    px = o[k]
                ex[j], xpx[j], kind[j] = k, px, -1
                done = True
                break
            if intrabar and k == ie:
                continue
            if (s > 0 and h[k] >= tpp) or (s < 0 and l[k] <= tpp):
                ex[j], xpx[j], kind[j] = k, tpp, 1
                done = True
                break
        if not done:
            ex[j], xpx[j] = k_end - 1, c[k_end - 1]
            kind[j] = 4 if (ie + maxb > end_idx + 1) else 3
        R[j] = s * math.log(xpx[j] / e) / sl
    return ent, epx, maker, ex, xpx, kind, R


def _exec_arrays(D: Data) -> dict:
    if not hasattr(D, "_ex"):
        d5 = D.d5
        D._ex = {k: d5[c].to_numpy(np.float64) for k, c in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"))}
        D._ex["t"] = mv2._ns_int(d5["timestamp"])
    return D._ex


def simulate(D: Data, rows: pd.DataFrame, cfg: StrategyConfig, end_ns: int | None = None) -> pd.DataFrame:
    """Mô phỏng MỌI tín hiệu (chưa giới hạn số lệnh). rows: các nến có tín hiệu (t_ns, b, side, ...)."""
    E = _exec_arrays(D)
    t5 = E["t"]
    dec = rows.t_ns.to_numpy(np.int64) + BAR_MIN * NS_MIN                    # lúc nến 1h tín hiệu đóng
    start = np.searchsorted(t5, dec)
    end_idx = len(t5) - 1 if end_ns is None else int(np.searchsorted(t5, end_ns) - 1)
    ok = (start <= end_idx) & (start < len(t5))
    ok[ok] &= (t5[start[ok]] - dec[ok]) <= cfg.max_entry_delay_min * NS_MIN
    S = rows[ok].copy()
    ent, epx, maker, ex, xpx, kind, R = _simulate(
        E["o"], E["h"], E["l"], E["c"], start[ok].astype(np.int64), S.side.to_numpy(np.float64), S.b.to_numpy(np.float64),
        cfg.tp_mult, cfg.sl_mult, cfg.max_hold_hours * 12, 1 if cfg.entry_mode == "limit_pullback" else 0,
        cfg.pullback_mult, max(1, cfg.pullback_minutes // 5), cfg.fill_bp, end_idx)
    S["entry_idx"], S["entry"], S["maker"], S["exit_idx"], S["exit"], S["kind"], S["R_gross"] = ent, epx, maker, ex, xpx, kind, R
    S = S[S.entry_idx >= 0].copy()
    S["sl_pct"] = cfg.sl_mult * S.b.to_numpy()
    fe = np.where(S.maker.to_numpy(), cfg.fee_maker, cfg.fee_taker)
    S["R"] = S.R_gross - (fe + cfg.fee_taker) / S.sl_pct
    S["entry_ns"], S["exit_ns"] = t5[S.entry_idx.to_numpy()], t5[S.exit_idx.to_numpy()]
    return S.sort_values("t_ns").reset_index(drop=True)


def select_trades(S: pd.DataFrame, max_positions: int, min_gap_bars: int) -> np.ndarray:
    """Tối đa max_positions lệnh mở (lệnh đang chờ limit cũng tính), các lần vào cách nhau >= min_gap_bars giờ."""
    t, xe = S.t_ns.to_numpy(np.int64).tolist(), S.exit_ns.to_numpy(np.int64).tolist()
    keep = np.zeros(len(S), bool)
    opens, last, gap = [], -(2**62), min_gap_bars * BAR_MIN * NS_MIN
    for k in range(len(S)):
        opens = [x for x in opens if x >= t[k]]
        if len(opens) >= max_positions or t[k] - last < gap:
            continue
        keep[k] = True
        opens.append(xe[k])
        last = t[k]
    return keep


def trades_table(S: pd.DataFrame) -> pd.DataFrame:
    T = S.copy()
    for a, b in (("signal_time", "t_ns"), ("entry_time", "entry_ns"), ("exit_time", "exit_ns")):
        T[a] = pd.to_datetime(T[b], utc=True)
    T["side_name"] = np.where(T.side > 0, "long", "short")
    T["result"] = T.kind.map(KIND_NAMES)
    return T


def equity_curve(T: pd.DataFrame, risk: float) -> pd.DataFrame:
    if T.empty:
        return pd.DataFrame(columns=["time", "equity"])
    ev = sorted([(int(r.entry_ns), 0, i) for i, r in T.iterrows()] + [(int(r.exit_ns), 1, i) for i, r in T.iterrows()],
                key=lambda x: (x[0], -x[1]))
    eq, at_risk, curve = 1.0, {}, []
    for tt, kd, i in ev:
        if kd == 0:
            at_risk[i] = risk * eq
        else:
            eq += at_risk.pop(i, risk * eq) * T.at[i, "R"]
            curve.append((pd.to_datetime(tt, utc=True), eq))
    return pd.DataFrame(curve, columns=["time", "equity"])


def summarize(T: pd.DataFrame, curve: pd.DataFrame | None = None) -> dict:
    if T.empty:
        return {"trades": 0}
    T = T.sort_values("exit_ns")
    cum = np.r_[0, np.cumsum(T.R.to_numpy())]
    yr, q = _dti(T.t_ns).year, _dti(T.t_ns).to_period("Q").astype(str)
    qs = T.groupby(q).R.sum()
    out = {"trades": int(len(T)), "win_rate": float((T.R > 0).mean()), "R_per_trade": float(T.R.mean()),
           "total_R": float(T.R.sum()), "max_dd_R": float((np.maximum.accumulate(cum) - cum).max()),
           "long_share": float((T.side > 0).mean()), "maker_share": float(T.maker.mean()),
           "timeout_share": float((T.kind == 3).mean()),
           # % giá = R * khoảng SL (lợi nhuận ròng theo % giá trị lệnh, nếu mỗi lệnh dùng cùng 1 khối lượng)
           "b_mean_pct": float(T.b.mean() * 100), "pct_per_trade": float((T.R * T.sl_pct).mean() * 100),
           "total_pct": float((T.R * T.sl_pct).sum() * 100),
           "R_by_year": {int(k): float(v) for k, v in T.groupby(yr).R.sum().items()},
           "pct_by_year": {int(k): float(v) for k, v in (T.R * T.sl_pct * 100).groupby(yr).sum().items()},
           "trades_by_year": {int(k): int(v) for k, v in T.groupby(yr).size().items()},
           "negative_quarters": int((qs < 0).sum()), "quarters": int(len(qs))}
    if curve is not None and len(curve):
        eq = np.r_[1.0, curve.equity.to_numpy()]
        out["equity_multiple"] = float(eq[-1])
        out["max_dd_pct"] = float(((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100)
        out["account_pct_by_year"] = {int(k): float(v) for k, v in period_return(curve, "Y").items()}
    return out


def period_return(curve: pd.DataFrame, freq: str = "Y") -> pd.Series:
    """Lãi % vốn (lãi kép, rủi ro cố định % vốn mỗi lệnh) theo năm / quý, tính theo thời điểm thoát lệnh."""
    if curve is None or not len(curve):
        return pd.Series(dtype=float)
    t = pd.DatetimeIndex(curve.time)
    t = t.tz_localize(None) if t.tz is not None else t
    key = t.year if freq == "Y" else t.to_period("Q").astype(str)
    eq = curve.equity.to_numpy()
    prev = np.r_[1.0, eq[:-1]]
    g = pd.DataFrame({"k": key, "start": prev, "end": eq}).groupby("k", sort=True)
    return (g.end.last() / g.start.first() - 1) * 100


def by_period(T: pd.DataFrame, freq: str = "Y", curve: pd.DataFrame | None = None) -> pd.DataFrame:
    t = _dti(T.t_ns)
    key = t.year if freq == "Y" else t.to_period("Q").astype(str)
    g = T.groupby(key)
    gp = (T.R * T.sl_pct * 100).groupby(key)
    res = pd.DataFrame({"lệnh": g.size(), "R/lệnh": g.R.mean(), "tổng R": g.R.sum(), "thắng": g.R.apply(lambda r: (r > 0).mean()),
                        "tỉ lệ LONG": g.side.apply(lambda s: (s > 0).mean()), "TP": g.kind.apply(lambda k: (k == 1).mean()),
                        "SL": g.kind.apply(lambda k: (k == -1).mean()), "hết giờ": g.kind.apply(lambda k: (k == 3).mean()),
                        "khớp limit": g.maker.mean(), "b TB %": g.b.mean() * 100,
                        "% giá/lệnh": gp.mean(), "tổng % giá": gp.sum()})
    if curve is not None and len(curve):
        res["lãi % vốn"] = period_return(curve, freq).reindex(res.index)
    res.index.name = "năm" if freq == "Y" else "quý"
    return res.round(3)


def evaluate(D: Data, P: pd.DataFrame, cfg: StrategyConfig, phase: str = "dev", suffix: str = "") -> dict:
    """Tín hiệu + mô phỏng + giới hạn lệnh với đúng tham số của cfg. 'dev' cắt lệnh tại dev_end."""
    if phase != "dev" and not cfg.locked_from:
        raise RuntimeError("đánh giá test cần config đã khoá")
    sig = make_signals(P, cfg, suffix=suffix)
    sig["rows"]["signal"] = sig[(float(cfg.touch_top), float(cfg.dir_top))]
    S = simulate(D, sig["rows"][sig["rows"].signal], cfg, _ns(cfg.dev_end) if phase == "dev" else None)
    T = trades_table(S[select_trades(S, cfg.max_positions, cfg.min_gap_bars)].reset_index(drop=True))
    curve = equity_curve(T, cfg.risk_per_trade)
    return {"signals": sig["rows"], "all_signal_trades": S, "trades": T, "equity": curve, "summary": summarize(T, curve),
            "by_year": by_period(T, "Y", curve) if len(T) else pd.DataFrame(),
            "by_quarter": by_period(T, "Q", curve) if len(T) else pd.DataFrame()}


def seed_table(D: Data, P: pd.DataFrame, cfg: StrategyConfig, phase: str = "dev") -> pd.DataFrame:
    """Cùng tham số: từng seed riêng và ensemble; TB / std / seed tệ nhất."""
    seeds = [c[len("p_touch_s"):] for c in P.columns if c.startswith("p_touch_s")]
    rows = []
    for s in seeds + ["ens"]:
        ev = evaluate(D, P, cfg, phase, suffix="" if s == "ens" else f"_s{s}")
        T, sm = ev["trades"], ev["summary"]
        row = {"seed": s, "lệnh": len(T), "R/lệnh": float(T.R.mean()) if len(T) else np.nan, "tổng R": float(T.R.sum()),
               "% giá/lệnh": sm.get("pct_per_trade", np.nan), "tổng % giá": sm.get("total_pct", np.nan),
               "lãi % vốn": (sm.get("equity_multiple", np.nan) - 1) * 100}
        if len(T):
            row.update({f"R/lệnh {y}": float(g.R.mean()) for y, g in T.groupby(_dti(T.t_ns).year)})
        rows.append(row)
    out = pd.DataFrame(rows).set_index("seed")
    single = out.drop(index="ens")
    out.loc["TB seed"], out.loc["std seed"], out.loc["seed tệ nhất"] = single.mean(), single.std(), single.min()
    return out.round(4)


# --------------------------------------------------------------------------- #
# Chọn tham số trên dev, khoá config
# --------------------------------------------------------------------------- #
def _year_stats(T: pd.DataFrame, years) -> dict:
    yr = _dti(T.t_ns).year.to_numpy() if len(T) else np.array([], int)
    R = T.R.to_numpy() if len(T) else np.array([])
    side = T.side.to_numpy() if len(T) else np.array([])
    out = {}
    for y in years:
        m = yr == y
        out.update({f"n_{y}": int(m.sum()), f"R_{y}": float(R[m].mean()) if m.any() else np.nan,
                    f"totR_{y}": float(R[m].sum()), f"long_{y}": float((side[m] > 0).mean()) if m.any() else np.nan})
    return out


def _score_cols(G: pd.DataFrame, years, cfg: StrategyConfig) -> pd.DataFrame:
    w = cfg.select_weight_worst
    R = G[[f"R_{y}" for y in years]].to_numpy()
    n = G[[f"n_{y}" for y in years]].to_numpy()
    mean, worst = np.nanmean(R, axis=1), np.nanmin(R, axis=1)
    valid = (n >= cfg.min_trades_per_year).all(axis=1) & np.isfinite(R).all(axis=1)
    return pd.DataFrame({"TB năm": mean, "năm kém": worst, "điểm": (1 - w) * mean + w * worst, "hợp lệ": valid}, index=G.index)


def _region(G: pd.DataFrame, score: pd.Series, valid: pd.Series, grid: dict) -> pd.Series:
    """Điểm vùng = TB điểm của ô và các ô lệch 1 nấc trên đúng 1 trục số (ô không hợp lệ bị bỏ)."""
    axes = list(grid)
    pos = {a: {v: i for i, v in enumerate(grid[a])} for a in axes}
    key = [tuple(pos[a][r[a]] for a in axes) for _, r in G[axes].iterrows()]
    lut = {k: i for i, k in enumerate(key)}
    sc, va = score.to_numpy(), valid.to_numpy()
    out = np.full(len(G), np.nan)
    for i, k in enumerate(key):
        if not va[i]:
            continue
        vals = [sc[i]]
        for ai, a in enumerate(axes):
            if a not in NUMERIC_AXES:
                continue
            for d in (-1, 1):
                kk = list(k)
                kk[ai] += d
                j = lut.get(tuple(kk))
                if j is not None and va[j]:
                    vals.append(sc[j])
        out[i] = float(np.mean(vals))
    return pd.Series(out, index=G.index)


def grid_search(D: Data, P_dev: pd.DataFrame, cfg: StrategyConfig, grid: dict | None = None, log=print) -> pd.DataFrame:
    """Lưới tham số giao dịch trên tín hiệu dev (lệnh còn mở bị cắt tại dev_end)."""
    grid = dict(GRID if grid is None else grid)
    years = tuple(sorted(set(_dti(P_dev.t_ns[P_dev.is_test]).year)))
    dev_end = _ns(cfg.dev_end)
    if P_dev.t_ns.max() >= dev_end:
        raise ValueError("P_dev có dữ liệu sau dev_end")
    t0 = time.time()
    sig = make_signals(P_dev, cfg, grid["touch_top"], grid["dir_top"])
    out = []
    for tt in grid["touch_top"]:
        for dt in grid["dir_top"]:
            for em in grid["entry_mode"]:
                S = simulate(D, sig["rows"][sig[(float(tt), float(dt))]], replace(cfg, entry_mode=em), dev_end)
                for mp in grid["max_positions"]:
                    for gp in grid["min_gap_bars"]:
                        T = S[select_trades(S, mp, gp)]
                        out.append({"touch_top": tt, "dir_top": dt, "entry_mode": em, "max_positions": mp, "min_gap_bars": gp,
                                    "lệnh": len(T), "R/lệnh": float(T.R.mean()) if len(T) else np.nan,
                                    "tổng R": float(T.R.sum()), **_year_stats(T, years)})
    G = pd.DataFrame(out)
    G = pd.concat([G, _score_cols(G, years, cfg)], axis=1)
    G["điểm vùng"] = _region(G, G["điểm"], G["hợp lệ"], grid)
    G.attrs.update({"grid": grid, "years": years})
    log(f"Lưới {len(G)} ô, {int(G['hợp lệ'].sum())} ô hợp lệ (>= {cfg.min_trades_per_year} lệnh/năm), {time.time() - t0:.0f}s")
    return G.sort_values("điểm vùng", ascending=False, na_position="last").reset_index(drop=True)


def select_best(G: pd.DataFrame) -> pd.Series:
    V = G[G["hợp lệ"] & G["điểm vùng"].notna()]
    if V.empty:
        raise ValueError("không có ô nào đủ số lệnh / năm")
    return V.sort_values(["điểm vùng", "điểm"], ascending=False).iloc[0]


def nested_selection(G: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Kiểm tra quy trình chọn: tham số cho năm Y chỉ chọn bằng các năm trước Y, rồi đo trên Y."""
    grid, years = G.attrs["grid"], G.attrs["years"]
    rows = []
    for k, Y in enumerate(years[1:], start=1):
        sc = _score_cols(G, years[:k], cfg)
        reg = _region(G, sc["điểm"], sc["hợp lệ"], grid)
        if reg.notna().sum() == 0:
            continue
        r = G.loc[reg.idxmax()]
        rows.append({"năm test": Y, "chọn bằng": "-".join(map(str, years[:k])), **{a: r[a] for a in grid},
                     "R/lệnh năm test": float(r[f"R_{Y}"]), "lệnh năm test": int(r[f"n_{Y}"]),
                     "TB toàn lưới năm test": float(G[f"R_{Y}"].mean())})
    return pd.DataFrame(rows)


def lock_config(cfg: StrategyConfig, best: pd.Series, G: pd.DataFrame | None = None, path: str | None = None) -> StrategyConfig:
    """Khoá tham số (chọn trên dev) vào file json. Phase test chỉ chạy với config đã khoá."""
    path = str(Path(cfg.output_dir) / "config_locked.json") if path is None else path
    upd = {a: (best[a].item() if hasattr(best[a], "item") else best[a]) for a in GRID if a in best.index}
    c = replace(cfg, **upd, locked_from=path)
    meta = {"strategy": STRATEGY, "locked_at": pd.Timestamp.now(tz="UTC").isoformat(), "config": asdict(c),
            "selected_on": f"dev (walk-forward quý {cfg.first_dev_quarter} -> trước {cfg.dev_end})",
            "best_cell": {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}}
    if G is not None:
        meta["grid"] = {k: list(v) for k, v in G.attrs.get("grid", {}).items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return c


def config_from_dict(d: dict, locked_from: str = "") -> StrategyConfig:
    kw = {f.name: (tuple(d[f.name]) if isinstance(d[f.name], list) else d[f.name]) for f in fields(StrategyConfig) if f.name in d}
    if locked_from:
        kw["locked_from"] = locked_from
    return StrategyConfig(**kw)


def load_locked(path: str) -> StrategyConfig:
    return config_from_dict(json.loads(Path(path).read_text(encoding="utf-8"))["config"], path)


def save_result(res: dict, cfg: StrategyConfig, name: str) -> Path:
    out = Path(cfg.output_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    T = res["trades"]
    cols = ["signal_time", "entry_time", "exit_time", "side_name", "entry", "exit", "b", "sl_pct", "p_touch", "p_up",
            "maker", "result", "R_gross", "R", "fold"]
    T[[c for c in cols if c in T]].to_csv(out / "trades.csv", index=False)
    res["equity"].to_csv(out / "equity.csv", index=False)
    for k in ("by_year", "by_quarter"):
        if len(res.get(k, [])):
            res[k].to_csv(out / f"{k}.csv")
    (out / "summary.json").write_text(json.dumps({"config": asdict(cfg), "summary": res["summary"]}, indent=2,
                                                  ensure_ascii=False, default=str), encoding="utf-8")
    return out


def save_live_artifacts(res: dict, cfg: StrategyConfig) -> Path:
    """File pipeline dùng: walkforward_signals.csv (mọi giờ test: xác suất, ngưỡng, tín hiệu) cho kiểm thử live và
    backtest_trades.csv (lệnh backtest test) cho trang public."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    W = res["signals"].copy()
    W.insert(0, "timestamp", pd.to_datetime(W.t_ns, utc=True))
    W.to_csv(out / "walkforward_signals.csv", index=False)
    T = res["trades"]
    cols = ["signal_time", "entry_time", "exit_time", "side_name", "side", "entry", "exit", "b", "p_touch", "p_up",
            "maker", "result", "R_gross", "R", "fold"]
    T[[c for c in cols if c in T]].to_csv(out / "backtest_trades.csv", index=False)
    return out


# --------------------------------------------------------------------------- #
# Bundle live
# --------------------------------------------------------------------------- #
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


def export_bundle(D: Data, cfg: StrategyConfig, path: str | None = None, log=print) -> Path:
    """Bundle live = model của quý chứa nến cuối cùng (đúng như fold walk-forward của quý đó: train tới Q - 2 quý),
    5 seed. Lưu kèm lịch sử dự báo của cửa sổ (dir_window_days ngày trước quý + phần đã qua của quý) để live tính
    ngưỡng trượt giống hệt walk-forward. Định dạng portable: XGBoost JSON + config dict."""
    import joblib
    if not cfg.locked_from:
        raise RuntimeError("bundle chỉ xuất từ config đã khoá")
    if cfg.model_kind != "xgb":
        raise RuntimeError("bundle live: touch phải là xgb (model_kind = xgb)")
    import nn_models
    blend = dict(cfg.direction_blend) or {"xgb": 1.0}
    ds = D.ds
    q = pd.Period(pd.to_datetime(ds.timestamp_ns[-1], utc=True).tz_localize(None), "Q")
    sp = fold_split(D, q, cfg, int(ds.timestamp_ns[-1]) + 1)
    win = sp["window"]
    models, pt, pu = [], [], []
    for s in cfg.seeds:
        r = fit_fold(D, sp, cfg, s)
        (tm, tb), (dm, dbest) = r["models"]["touch"]["touch"], r["models"]["direction"]
        m = {"seed": s, "touch_json": _xgb_to_bytes(tm), "touch_best": int(tb),
             "T_touch": float(r["models"]["touch"]["T_touch"]), "dir_json": _xgb_to_bytes(dm), "dir_best": int(dbest), "nn": {}}
        d = blend.get("xgb", 0.0) * (r["p_up"].astype(np.float64) - 0.5)
        msg = f"bundle {q} seed {s}: cây touch {tb}, cây direction {dbest}"
        for k, w in blend.items():
            if k == "xgb":
                continue
            nm = nn_models.fit_direction_nn(D, sp, cfg, s, kind=k)
            m["nn"][k] = nn_models.export_model(nm)
            p_t = nn_models.scores(nm.predict_proba(win))[0]                       # torch
            p_n = nn_models.scores(nn_models.predict_model(m["nn"][k], D.xd, ds.timestamp_ns, win))[0]   # onnx (live)
            err = float(np.abs(p_t - p_n).max())
            if err > 1e-4:
                raise RuntimeError(f"{k} seed {s}: onnx khác torch {err:.2e}")
            d = d + w * (p_n - 0.5)
            msg += f", {k} epoch {nm.best_epoch} (onnx = torch, sai số {err:.1e})"
        models.append(m)
        pt.append(r["p_touch"]); pu.append(0.5 + d)
        log(msg)
    hist = pd.DataFrame({"t_ns": ds.timestamp_ns[win], "p_touch": np.mean(pt, axis=0), "p_up": np.mean(pu, axis=0)})
    out = Path(path or Path(cfg.output_dir) / "models_final.joblib")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"format": "portable-v2", "strategy": STRATEGY, "config": asdict(cfg), "models": models, "direction_blend": blend,
                 "touch_names": list(ds.names), "dir_names": list(D.dir_names), "trained_quarter": str(q),
                 "history": hist, "data_end_ns": int(ds.timestamp_ns[-1])}, out)
    return out


def load_bundle(path: str | Path) -> dict:
    import joblib
    b = joblib.load(path)
    if b.get("format") != "portable-v2":
        raise ValueError(f"{path}: không phải bundle {STRATEGY} (portable-v2)")
    b["config"] = config_from_dict(b["config"], b["config"].get("locked_from", ""))
    b["models"] = [{**m, "touch": {"touch": (_xgb_from_bytes(m["touch_json"]), m["touch_best"]), "T_touch": m["T_touch"]},
                    "direction": (_xgb_from_bytes(m["dir_json"]), m["dir_best"])} for m in b["models"]]
    return b


def predict_bundle(bundle: dict, x_touch: np.ndarray, x_dir: np.ndarray, ts_ns: np.ndarray | None = None) -> tuple:
    """(p_touch, p_up) ensemble các seed; Direction gộp theo bundle['direction_blend'] (model NN dự báo bằng onnxruntime).
    x_dir theo thời gian, ts_ns = thời điểm các hàng của x_dir; LSTM chỉ đúng cho hàng có đủ seq_len hàng trước."""
    import nn_models
    blend = bundle.get("direction_blend") or {"xgb": 1.0}
    pt = np.mean([mv2.predict_touch(m["touch"], x_touch) for m in bundle["models"]], axis=0)
    pus = []
    for m in bundle["models"]:
        d = blend.get("xgb", 0.0) * (direction_score(*m["direction"], x_dir) - 0.5)
        for k, w in blend.items():
            if k != "xgb":
                d = d + w * (nn_models.scores(nn_models.predict_model(m["nn"][k], x_dir, ts_ns))[0] - 0.5)
        pus.append(0.5 + d)
    return pt, np.mean(pus, axis=0)


def predict_live(bundle: dict, candles_1h: pd.DataFrame, positioning_5m: pd.DataFrame, funding: pd.DataFrame | None,
                 history: pd.DataFrame | None = None) -> dict:
    """Tín hiệu cho nến 1h ĐÃ ĐÓNG cuối cùng của candles_1h (cần >= khoảng 180 ngày nến 1h, >= 8 ngày vị thế 5m).
    history: dự báo live các giờ trước (t_ns, p_touch, p_up) cùng bundle; ghép với bundle['history'] để tính ngưỡng
    trượt (14 ngày touch, 90 ngày direction) đúng như walk-forward."""
    cfg: StrategyConfig = bundle["config"]
    df = mv2.prepare_candles(candles_1h, BAR_MIN)
    if funding is not None:
        df = mv2.attach_funding(df, funding, BAR_MIN)
    import nn_models
    ds, X, names = build_features(df, positioning_5m, cfg, live=True, live_last=nn_models.history_rows(bundle["models"]))
    if list(ds.names) != bundle["touch_names"] or list(names) != bundle["dir_names"]:
        raise ValueError("feature khác bundle (thiếu taker / funding / vị thế?)")
    pt, pu = predict_bundle(bundle, ds.x[-1:], X, ds.timestamp_ns)
    t_now = int(ds.timestamp_ns[-1])
    H = pd.concat([bundle["history"], history if history is not None else None], ignore_index=True)
    H = H[H.t_ns < t_now].drop_duplicates("t_ns", keep="last").sort_values("t_ns")
    t = np.r_[H.t_ns.to_numpy(np.int64), t_now]
    TT, TL, TS = rolling_thresholds(t, np.r_[H.p_touch.to_numpy(), pt[-1]], np.r_[H.p_up.to_numpy(), pu[-1]],
                                    [len(t) - 1], cfg, [cfg.touch_top], [cfg.dir_top])
    d = float(pu[-1]) - 0.5
    thr_t, thL, thS = float(TT[0, 0, 0]), float(TL[0, 0, 0]), float(TS[0, 0, 0])
    touch_ok = bool(pt[-1] >= thr_t)
    dir_ok = bool(_side_mask(np.array([d]), np.array([thL]), np.array([thS]))[0])
    return {"timestamp": pd.to_datetime(t_now, utc=True), "t_ns": t_now, "close": float(ds.close[-1]),
            "b": float(ds.barrier_pct[-1]), "p_touch": float(pt[-1]), "p_up": float(pu[-1]), "conf": abs(d),
            "side": 1 if d > 0 else -1, "thr_touch": thr_t, "thr_long": thL, "thr_short": thS,
            "touch_ok": touch_ok, "dir_ok": dir_ok, "signal": touch_ok and dir_ok,
            "history_hours": int(len(H))}
