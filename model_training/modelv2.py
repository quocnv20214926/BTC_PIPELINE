"""BTCUSDT 5m - model TOUCH (v2): xác suất giá chạy đủ xa trong 4 giờ tới.

Câu hỏi duy nhất của model: tại lúc nến 5m t đóng, trong H nến tiếp theo (mặc định 48 = 4h) giá có chạm
+b HOẶC -b (tính từ giá tham chiếu open[t+1]) không?  b = clip(k * sigma * sqrt(H), min_barrier, max_barrier).
Không dự báo hướng (các thử nghiệm dự báo hướng đã bị loại: xem lịch sử dự án).

Feature (nhân quả, tại lúc nến t đóng): cửa sổ biến động, các sóng pivot gần nhất, quan hệ giữa các sóng, vị
trí pivot, kênh giá, mức giá gần, tỉ lệ chạm trong quá khứ (chỉ nhãn đã đóng), bối cảnh (biến động chậm,
xu hướng 1/5 ngày, giờ / thứ, số lệnh, funding).

Đánh giá: walk-forward theo năm, ranh giới đã purge. Năm Y: train = nhãn kết thúc trước 1/1 của Y-1,
validation = Y-1 (phần đầu: dừng sớm | phần sau: temperature scaling + ngưỡng), test = Y.
Tín hiệu đầu ra mỗi nến: p_touch, barrier_pct, mức giá trên / dưới (close * (1 +- b)), time stop.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:  # numba: các vòng lặp pivot từ vài phút xuống vài giây
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f


STRATEGY = "touch_signal_v2"
TAKER_COLUMNS = ("taker_buy_base_volume", "taker_buy_base_asset_volume", "taker_buy_volume", "taker_buy_base")
TOUCH_GROUPS = ("window", "legs", "relations", "positions", "channel", "levels", "past", "context")
FUNDING_FILE = "BTCUSDT_funding.csv"
FUNDING_COLUMNS = ("funding_rate", "funding_mean_3", "funding_z_90", "funding_dev_90")
WINDOWS = (6, 12, 24, 48, 96, 288)
PAST_WINDOWS = (288, 576, 1440)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class XGBParams:
    n_estimators: int = 1000
    learning_rate: float = 0.03
    max_depth: int = 4
    min_child_weight: float = 40.0
    subsample: float = 0.7
    colsample_bytree: float = 0.6
    reg_alpha: float = 0.1
    reg_lambda: float = 10.0
    gamma: float = 0.0
    early_stopping_rounds: int = 60


@dataclass
class Config:
    csv_path: str = "historical_data/BTCUSDT_5m_all.csv"
    output_dir: str = "artifacts/5m_xgboost_signal_v2"
    bar_minutes: int = 5
    # ---- barrier / nhãn
    pivot_reversal: float = 0.005
    pivot_count: int = 8
    lookback_bars: int = 288
    barrier_horizon: int = 48
    barrier_vol_mult: float = 1.0
    min_barrier: float = 0.007
    max_barrier: float = 0.03
    vol_span: int = 288
    slow_vol_span: int = 2880
    sample_stride: int = 1
    # ---- feature
    touch_groups: tuple[str, ...] = TOUCH_GROUPS
    # ---- giao thức
    eval_years: tuple[int, ...] = (2022, 2023, 2024, 2025, 2026)
    val_es_frac: float = 0.6
    # ---- huấn luyện
    xgb_device: str = "auto"                  # "auto" | "cpu" | "cuda:N"
    xgb_n_jobs: int = -1
    time_decay_halflife_days: float = 540.0
    time_decay_floor: float = 0.2
    calibrate: bool = True                    # temperature scaling trên phần calibration
    touch: XGBParams = field(default_factory=XGBParams)
    # ---- đánh giá
    n_boot: int = 200
    top_fractions: tuple[float, ...] = (1.0, 0.5, 0.3, 0.2, 0.1, 0.05)
    seed: int = 42
    verbose: bool = True

    def __post_init__(self) -> None:
        positive = {"bar_minutes": self.bar_minutes, "barrier_horizon": self.barrier_horizon,
                    "vol_span": self.vol_span, "slow_vol_span": self.slow_vol_span, "pivot_count": self.pivot_count,
                    "lookback_bars": self.lookback_bars, "sample_stride": self.sample_stride}
        bad = [k for k, v in positive.items() if v < 1]
        if bad:
            raise ValueError(f"must be >= 1: {', '.join(bad)}")
        if 1440 % self.bar_minutes:
            raise ValueError("bar_minutes must divide one day (1440 minutes)")
        if self.pivot_count < 4:
            raise ValueError("pivot_count must be >= 4")
        if self.slow_vol_span < self.trend_long_bars:
            raise ValueError(f"slow_vol_span must be >= {self.trend_long_bars}")
        if not 0 < self.min_barrier <= self.max_barrier < 1:
            raise ValueError("need 0 < min_barrier <= max_barrier < 1")
        if not 0 < self.val_es_frac < 1:
            raise ValueError("val_es_frac must be in (0, 1)")
        if not self.eval_years or tuple(sorted(set(self.eval_years))) != tuple(self.eval_years):
            raise ValueError("eval_years must be non-empty, sorted and unique")
        if not self.touch_groups or any(g not in TOUCH_GROUPS for g in self.touch_groups):
            raise ValueError(f"touch_groups must be a non-empty subset of {TOUCH_GROUPS}")

    @property
    def bars_per_day(self) -> int:
        return 1440 // self.bar_minutes

    @property
    def trend_short_bars(self) -> int:
        return self.bars_per_day

    @property
    def trend_long_bars(self) -> int:
        return 5 * self.bars_per_day

    @property
    def gap_lookback(self) -> int:
        """Đoạn nến dài nhất mà feature nhìn lại (có nến thiếu trong đoạn này thì bỏ mẫu)."""
        return max(self.lookback_bars, max(PAST_WINDOWS) + self.barrier_horizon, self.trend_long_bars)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_device(device: str) -> str:
    """'auto' -> 'cuda:0' only when XGBoost really trains on a GPU (silent CPU fallback counts as no GPU)."""
    if device != "auto":
        return device
    try:
        import xgboost as xgb
        if not xgb.build_info().get("USE_CUDA", False):
            return "cpu"
        rng = np.random.RandomState(0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            xgb.XGBClassifier(n_estimators=2, tree_method="hist", device="cuda:0").fit(
                rng.rand(64, 3), np.arange(64) % 2)
        if any(("GPU" in str(w.message)) or ("Device is changed" in str(w.message)) for w in caught):
            return "cpu"
        return "cuda:0"
    except Exception:  # noqa: BLE001
        return "cpu"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _ns_int(ts) -> np.ndarray:
    s = pd.Series(pd.to_datetime(ts, utc=True))
    try:
        s = s.dt.as_unit("ns")
    except AttributeError:
        pass
    return s.astype("int64").to_numpy()


def gap_flags(df: pd.DataFrame, bar_minutes: int) -> np.ndarray:
    ns = _ns_int(df["timestamp"])
    flags = np.zeros(len(ns), dtype=np.int64)
    flags[1:] = np.diff(ns) != bar_minutes * 60 * 10**9
    return flags


def load_candles(path: str | Path, bar_minutes: int, funding_csv: str | Path | None = "auto") -> pd.DataFrame:
    """Load a kline CSV. funding_csv="auto" attaches historical_data/BTCUSDT_funding.csv when it exists
    (next to the kline file); None disables funding features."""
    df = pd.read_csv(path)
    missing = {"open", "high", "low", "close", "volume"}.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    out = prepare_candles(df, bar_minutes)
    if funding_csv == "auto":
        funding_csv = Path(path).parent / FUNDING_FILE
        funding_csv = funding_csv if funding_csv.exists() else None
    if funding_csv is not None:
        out = attach_funding(out, pd.read_csv(funding_csv), bar_minutes)
    return out


def funding_table(funding: pd.DataFrame) -> pd.DataFrame:
    """Settled funding rates (rows of BTCUSDT_funding.csv or /fapi/v1/fundingRate) + causal rolling stats."""
    f = pd.DataFrame({
        "funding_time": pd.to_datetime(funding["funding_time_ms"] if "funding_time_ms" in funding
                                       else funding["fundingTime"], unit="ms", utc=True),
        "rate": pd.to_numeric(funding["funding_rate"] if "funding_rate" in funding else funding["fundingRate"]),
    }).sort_values("funding_time").drop_duplicates("funding_time").reset_index(drop=True)
    r = f["rate"]
    f["funding_rate"] = r
    f["funding_mean_3"] = r.rolling(3, min_periods=1).mean()                       # last 24h
    mean90, std90 = r.rolling(90, min_periods=30).mean(), r.rolling(90, min_periods=30).std()  # last 30 days
    f["funding_z_90"] = (r - mean90) / std90.where(std90 > 0)
    f["funding_dev_90"] = r - mean90
    return f.drop(columns="rate")


def attach_funding(candles: pd.DataFrame, funding: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """Per candle: the last funding settlement known at the CLOSE of the candle (no look-ahead)."""
    f = funding_table(funding)
    out = candles.copy()
    close_time = out["timestamp"] + pd.Timedelta(minutes=bar_minutes)
    try:
        close_time = close_time.dt.as_unit("ns")
        f["funding_time"] = f["funding_time"].dt.as_unit("ns")
    except AttributeError:
        pass
    left = pd.DataFrame({"close_time": close_time, "_row": np.arange(len(out))})
    m = pd.merge_asof(left.sort_values("close_time"), f, left_on="close_time", right_on="funding_time",
                      direction="backward").sort_values("_row")
    for c in FUNDING_COLUMNS:
        out[c] = m[c].to_numpy(np.float64)
    out["funding_age_h"] = ((m["close_time"] - m["funding_time"]).dt.total_seconds() / 3600).to_numpy()
    return out


def prepare_candles(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """Normalise a raw kline frame (CSV or live feed) to timestamp/open/high/low/close/volume[/taker]."""
    if "timestamp" in df:
        ts = pd.to_datetime(df["timestamp"], utc=True)
    elif "open_time_utc" in df:
        ts = pd.to_datetime(df["open_time_utc"], utc=True, errors="raise")
    elif "open_time_ms" in df:
        ts = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True, errors="raise")
    else:
        raise ValueError("candles need timestamp, open_time_utc or open_time_ms")
    try:
        ts = ts.dt.as_unit("ns")
    except AttributeError:
        pass
    out = df.loc[:, ["open", "high", "low", "close", "volume"]].astype("float64").reset_index(drop=True)
    taker = next((c for c in TAKER_COLUMNS if c in df.columns), None)
    if taker is not None:
        out["taker_buy_volume"] = df[taker].astype("float64").to_numpy()
    if "trades" in df.columns:
        out["trades"] = df["trades"].astype("float64").to_numpy()
    for c in FUNDING_COLUMNS + ("funding_age_h",):       # live frames may carry funding already attached
        if c in df.columns:
            out[c] = df[c].astype("float64").to_numpy()
    out.insert(0, "timestamp", ts.to_numpy())
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    gaps = int(gap_flags(out, bar_minutes).sum())
    if gaps:
        warnings.warn(f"{gaps} timestamp gaps/irregular steps found; samples whose feature window or "
                      "label horizon spans a gap are dropped", stacklevel=2)
    return out


# --------------------------------------------------------------------------- #
# ZigZag pivots (nhân quả)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _pivot_kernel(high, low, reversal):
    n = len(high)
    pidx = np.empty(n, np.int64)
    cidx = np.empty(n, np.int64)
    price = np.empty(n, np.float64)
    kind = np.empty(n, np.int64)
    cnt = 0
    direction = 0
    run_hi, hi_idx = high[0], 0
    run_lo, lo_idx = low[0], 0
    for i in range(1, n):
        if direction == 0:
            if high[i] > run_hi:
                run_hi, hi_idx = high[i], i
            if low[i] < run_lo:
                run_lo, lo_idx = low[i], i
            up = high[i] / run_lo - 1.0
            down = 1.0 - low[i] / run_hi
            if up >= reversal and up >= down and lo_idx < i:
                pidx[cnt], cidx[cnt], price[cnt], kind[cnt] = lo_idx, i, run_lo, -1
                cnt += 1
                direction, run_hi, hi_idx = 1, high[i], i
            elif down >= reversal and hi_idx < i:
                pidx[cnt], cidx[cnt], price[cnt], kind[cnt] = hi_idx, i, run_hi, 1
                cnt += 1
                direction, run_lo, lo_idx = -1, low[i], i
        elif direction == 1:
            if high[i] >= run_hi:
                run_hi, hi_idx = high[i], i
                continue
            if low[i] <= run_hi * (1.0 - reversal):
                pidx[cnt], cidx[cnt], price[cnt], kind[cnt] = hi_idx, i, run_hi, 1
                cnt += 1
                direction, run_lo, lo_idx = -1, low[i], i
        else:
            if low[i] <= run_lo:
                run_lo, lo_idx = low[i], i
                continue
            if high[i] >= run_lo * (1.0 + reversal):
                pidx[cnt], cidx[cnt], price[cnt], kind[cnt] = lo_idx, i, run_lo, -1
                cnt += 1
                direction, run_hi, hi_idx = 1, high[i], i
    return pidx[:cnt], cidx[:cnt], price[:cnt], kind[:cnt]


def find_confirmed_pivots(df: pd.DataFrame, reversal: float = 0.005) -> pd.DataFrame:
    if not 0 < reversal < 1:
        raise ValueError("reversal must be between 0 and 1")
    if len(df) < 2:
        return pd.DataFrame(columns=["pivot_idx", "confirm_idx", "price", "kind"])
    p, c, pr, k = _pivot_kernel(df["high"].to_numpy(np.float64), df["low"].to_numpy(np.float64), float(reversal))
    pivots = pd.DataFrame({"pivot_idx": p, "confirm_idx": c, "price": pr, "kind": k})
    if not pivots.empty:
        pivots["confirmation_lag"] = c - p
    return pivots


@njit(cache=True)
def _developing_kernel(high, low, confirm, kinds):
    n = len(high)
    dev_idx = np.full(n, -1, np.int64)
    dev_price = np.full(n, np.nan, np.float64)
    p = 0
    kind = 0
    idx = -1
    price = np.nan
    for i in range(n):
        if p < len(confirm) and confirm[p] == i:
            kind = -kinds[p]
            idx = i
            price = high[i] if kind == 1 else low[i]
            p += 1
        elif kind == 1 and high[i] >= price:
            idx, price = i, high[i]
        elif kind == -1 and low[i] <= price:
            idx, price = i, low[i]
        dev_idx[i] = idx
        dev_price[i] = price
    return dev_idx, dev_price


# --------------------------------------------------------------------------- #
# Nhãn: tỉ lệ chạm từng nến (cho feature 'past')
# --------------------------------------------------------------------------- #
def touch_label_series(open_, high, low, bar_barrier, horizon) -> np.ndarray:
    """Touch label of EVERY bar t (used for the causal 'past touch rate' features)."""
    n = len(open_)
    out = np.full(n, np.nan, dtype=np.float32)
    if n <= horizon + 1:
        return out
    fmax = pd.Series(high).rolling(horizon).max().to_numpy()
    fmin = pd.Series(low).rolling(horizon).min().to_numpy()
    t = np.arange(0, n - horizon)
    ref = open_[t + 1]
    b = bar_barrier[t]
    hit = (fmax[t + horizon] >= ref * (1.0 + b)) | (fmin[t + horizon] <= ref * (1.0 - b))
    out[t] = np.where(np.isfinite(b), hit.astype(np.float32), np.nan)
    return out


def make_labels(open_, high, low, close, ts, barrier, horizon, chunk: int | None = None) -> dict:
    """Từ open[t+1], trên các nến t+1..t+H.

    y_touch    1 nếu chạm +b hoặc -b
    first      +1 chạm +b trước, -1 chạm -b trước, 0 cả hai trong cùng một nến, 0 nếu không chạm (xem y_touch)
    t_first    số nến tới lần chạm đầu (H+1 nếu không chạm)
    r_end      close[t+H] / open[t+1] - 1, đơn vị barrier
    """
    m = len(ts)
    chunk = chunk or max(1_000, 2_400_000 // horizon)
    out = {k: np.empty(m, dtype=np.int64) for k in ("y_touch", "first", "t_first")}
    out["r_end"] = np.empty(m, dtype=np.float64)
    steps = np.arange(horizon)[None, :]
    for s0 in range(0, m, chunk):
        sl = slice(s0, min(m, s0 + chunk))
        t = ts[sl]
        b = barrier[sl][:, None]
        idx = t[:, None] + 1 + steps
        c0 = open_[t + 1][:, None]
        up, dn = high[idx] >= c0 * (1.0 + b), low[idx] <= c0 * (1.0 - b)
        fu = np.where(up.any(1), up.argmax(1), horizon)
        fd = np.where(dn.any(1), dn.argmax(1), horizon)
        first = np.minimum(fu, fd)
        touched = first < horizon
        out["y_touch"][sl] = touched
        out["first"][sl] = np.where(fu < fd, 1, np.where(fd < fu, -1, 0))
        out["t_first"][sl] = np.where(touched, first + 1, horizon + 1)
        out["r_end"][sl] = (close[t + horizon] / open_[t + 1] - 1.0) / barrier[sl]
    return out


# --------------------------------------------------------------------------- #
# Feature theo nến và thống kê đường giá
# --------------------------------------------------------------------------- #
def compute_bar_features(df: pd.DataFrame, cfg: Config) -> dict:
    n = len(df)
    c, v = df["close"].to_numpy(np.float64), df["volume"].to_numpy(np.float64)
    logc = np.log(c)
    r = np.zeros(n)
    r[1:] = np.diff(logc)
    r2 = pd.Series(r * r)
    sigma = np.sqrt(r2.ewm(span=cfg.vol_span, adjust=False, min_periods=cfg.vol_span).mean().to_numpy())
    sigma_slow = np.sqrt(r2.ewm(span=cfg.slow_vol_span, adjust=False, min_periods=cfg.slow_vol_span).mean().to_numpy())
    sigma = np.maximum(sigma, 1e-6)
    sigma_slow = np.maximum(sigma_slow, 1e-6)

    logv = np.log1p(v)
    win = cfg.bars_per_day
    vol_med = pd.Series(logv).rolling(win, min_periods=1).median().to_numpy()
    vol_mad = pd.Series(np.abs(logv - vol_med)).rolling(win, min_periods=1).median().to_numpy()
    rvol = np.clip((logv - vol_med) / np.maximum(vol_mad * 1.4826, 1e-4), -5.0, 5.0)

    def trend(k: int) -> np.ndarray:
        out = np.full(n, np.nan)
        out[k:] = logc[k:] - logc[:-k]
        return out / (sigma * math.sqrt(k))

    ts = df["timestamp"]
    hour = (ts.dt.hour + ts.dt.minute / 60.0).to_numpy(np.float64)
    dow = ts.dt.dayofweek.to_numpy(np.float64)
    return {
        "sigma": sigma, "logc": logc, "r": r, "absd": np.abs(r), "rvol": rvol,
        "sigma_ratio": np.log(sigma / sigma_slow),
        "trend_1d_z": trend(cfg.trend_short_bars), "trend_5d_z": trend(cfg.trend_long_bars),
        "hour_sin": np.sin(2 * np.pi * hour / 24.0), "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "dow_sin": np.sin(2 * np.pi * dow / 7.0), "dow_cos": np.cos(2 * np.pi * dow / 7.0),
    }


def _roll(x: np.ndarray, k: int, how: str = "mean", min_periods: int | None = None) -> np.ndarray:
    return getattr(pd.Series(x).rolling(k, min_periods=k if min_periods is None else min_periods), how)().to_numpy()


def path_prefix(bf: dict) -> dict:
    return {"logc": bf["logc"], "cS": np.cumsum(bf["absd"]), "cR2": np.cumsum(bf["r"] ** 2)}


def path_stats(a, b, pre: dict) -> dict:
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    valid = b > a
    bb = np.where(valid, b, np.minimum(a + 1, len(pre["logc"]) - 1))   # invalid spans -> NaN below
    dur = np.maximum(bb - a, 1)
    lc = pre["logc"]
    disp = lc[bb] - lc[a]
    plen = pre["cS"][bb] - pre["cS"][a]
    out = {
        "dur": dur.astype(np.float64), "disp_c": disp, "plen": plen,
        "er": np.abs(disp) / np.maximum(plen, 1e-12),
        "rv": np.sqrt(np.maximum(pre["cR2"][bb] - pre["cR2"][a], 0.0) / dur),
    }
    return {k: np.where(valid, v, np.nan) for k, v in out.items()}


def completed_legs(pivots: pd.DataFrame, bf: dict, pre: dict) -> dict:
    pidx = pivots["pivot_idx"].to_numpy(np.int64)
    pprice = pivots["price"].to_numpy(np.float64)
    st = path_stats(pidx[:-1], pidx[1:], pre)
    lo, hi = int(pidx[0]), int(pidx[-1])
    starts = pidx[:-1] - lo
    st["jump"] = np.clip(np.maximum.reduceat(bf["absd"][lo:hi], starts) / np.maximum(st["plen"], 1e-12), 0.0, 1.0)
    st["disp_price"] = np.log(pprice[1:] / pprice[:-1])
    return st


# --------------------------------------------------------------------------- #
# TOUCH features (không có hướng)
# --------------------------------------------------------------------------- #
def touch_features(df, pivots, bf, pre, bar_barrier, touch_s, ts, known_end, cfg: Config):
    P, H, L = cfg.pivot_count, cfg.barrier_horizon, cfg.lookback_bars
    sigma, logc = bf["sigma"], bf["logc"]
    high, low, close = (df[k].to_numpy(np.float64) for k in ("high", "low", "close"))
    pivot_idx = pivots["pivot_idx"].to_numpy(np.int64)
    pivot_price = pivots["price"].to_numpy(np.float64)
    pivot_kind = pivots["kind"].to_numpy(np.float64)
    dev_idx, dev_price = _developing_kernel(high, low, pivots["confirm_idx"].to_numpy(np.int64),
                                            pivots["kind"].to_numpy(np.int64))
    close_t, sig_t = close[ts], sigma[ts]
    s = sig_t * math.sqrt(H)
    barrier = bar_barrier[ts]
    rb = barrier / s
    f: dict[str, np.ndarray] = {}
    reg: list[tuple[str, str]] = []

    def add(name, arr, group):
        arr = np.asarray(arr, dtype=np.float64)
        f[name] = np.where(np.isfinite(arr), arr, np.nan).astype(np.float32)
        reg.append((name, group))

    def emit_path(prefix, group, st, disp, with_duration):
        add(f"{prefix}_adisp", np.abs(disp / s), group)
        add(f"{prefix}_er", st["er"], group)
        add(f"{prefix}_rv", st["rv"] / sig_t, group)
        if with_duration:
            add(f"{prefix}_dur", st["dur"] / H, group)

    # ---- context
    for name in ("sigma_ratio", "trend_1d_z", "trend_5d_z", "hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        add(name, bf[name][ts], "context")
    add("log_sigma", np.log(sig_t), "context")
    add("barrier_pct", barrier, "context")
    add("barrier_ratio", rb, "context")
    add("abs_trend_1d_z", np.abs(f["trend_1d_z"]), "context")
    add("abs_trend_5d_z", np.abs(f["trend_5d_z"]), "context")
    if "trades" in df.columns:                        # trading intensity (number of trades per bar)
        lt = np.log1p(df["trades"].to_numpy(np.float64))
        med = pd.Series(lt).rolling(cfg.bars_per_day, min_periods=cfg.bars_per_day // 4).median().to_numpy()
        add("trades_rz_6", _roll(lt - med, 6, "mean")[ts], "context")
        add("trades_rz_48", _roll(lt - med, 48, "mean")[ts], "context")
        size = np.log1p(df["volume"].to_numpy(np.float64)) - lt                  # log average trade size
        smed = pd.Series(size).rolling(cfg.bars_per_day, min_periods=cfg.bars_per_day // 4).median().to_numpy()
        add("trade_size_rz_12", _roll(size - smed, 12, "mean")[ts], "context")
    if "funding_rate" in df.columns:
        add("funding_abs", np.abs(df["funding_rate"].to_numpy(np.float64))[ts] * 1e4, "context")

    # ---- window
    for k in WINDOWS:
        if k > L:
            continue
        st = path_stats(ts - k, ts, pre)
        emit_path(f"w{k}", "window", st, st["disp_c"], with_duration=False)
        add(f"w{k}_jump", _roll(bf["absd"], k, "max")[ts] / np.maximum(st["plen"], 1e-12), "window")
        up = (np.log(_roll(high, k, "max")[ts]) - logc[ts]) / s
        dn = (logc[ts] - np.log(_roll(low, k, "min")[ts])) / s
        add(f"w{k}_up", up, "window")
        add(f"w{k}_dn", dn, "window")
        add(f"w{k}_rng", up + dn, "window")

    # ---- positions
    sel = known_end[:, None] - P + np.arange(P)[None, :]
    pi, pp, pk = pivot_idx[sel], pivot_price[sel], pivot_kind[sel]
    X = (pi - ts[:, None]) / H
    Y = np.log(pp / close_t[:, None]) / s[:, None]
    last = known_end - 1
    di, dp = dev_idx[ts], dev_price[ts]
    xd, yd = (di - ts) / H, np.log(dp / close_t) / s
    for j in range(1, 5):
        add(f"pos{j}_r", np.hypot(X[:, P - j], Y[:, P - j]), "positions")
    add("posd_r", np.hypot(xd, yd), "positions")

    # ---- legs
    legs = completed_legs(pivots, bf, pre)
    leg_info: dict[str, dict] = {}
    for j in (1, 2, 3):
        li = known_end - 1 - j
        st = {k: v[li] for k, v in legs.items() if k not in ("jump", "disp_price")}
        emit_path(f"l{j}", "legs", st, legs["disp_price"][li], with_duration=True)
        add(f"l{j}_jump", legs["jump"][li], "legs")
        leg_info[f"l{j}"] = {"disp": legs["disp_price"][li], "dur": st["dur"]}
    st1 = path_stats(pivot_idx[last], di, pre)
    disp1 = np.log(dp / pivot_price[last])
    emit_path("c1", "legs", st1, disp1, with_duration=True)
    st2 = path_stats(di, ts, pre)
    disp2 = np.log(close_t / dp)
    emit_path("c2", "legs", st2, disp2, with_duration=True)
    add("retrace_now", -disp2 / disp1, "legs")
    leg_info["c1"] = {"disp": disp1, "dur": st1["dur"]}

    # ---- relations
    for x_leg, y_leg in (("c1", "l1"), ("l1", "l2"), ("l2", "l3")):
        a_, b_ = leg_info[x_leg], leg_info[y_leg]
        add(f"rel_{x_leg}_{y_leg}_size",
            np.log(np.maximum(np.abs(a_["disp"]), 1e-9) / np.maximum(np.abs(b_["disp"]), 1e-9)), "relations")
        add(f"rel_{x_leg}_{y_leg}_time", np.log(a_["dur"] / b_["dur"]), "relations")

    # ---- channel
    hi_last = pk[:, -1] == 1
    hi_prev = pk[:, -3] == 1
    x_h1, y_h1 = np.where(hi_last, X[:, -1], X[:, -2]), np.where(hi_last, Y[:, -1], Y[:, -2])
    x_l1, y_l1 = np.where(hi_last, X[:, -2], X[:, -1]), np.where(hi_last, Y[:, -2], Y[:, -1])
    x_h2, y_h2 = np.where(hi_prev, X[:, -3], X[:, -4]), np.where(hi_prev, Y[:, -3], Y[:, -4])
    x_l2 = np.where(hi_prev, X[:, -4], X[:, -3])
    y_l2 = np.where(hi_prev, Y[:, -4], Y[:, -3])
    slope_u = (y_h1 - y_h2) / np.maximum(x_h1 - x_h2, 1e-6)
    slope_l = (y_l1 - y_l2) / np.maximum(x_l1 - x_l2, 1e-6)
    u0, l0 = y_h1 - slope_u * x_h1, y_l1 - slope_l * x_l1
    x_ref = np.maximum(x_h2, x_l2)
    w_ref = (y_h1 + slope_u * (x_ref - x_h1)) - (y_l1 + slope_l * (x_ref - x_l1))
    width0 = u0 - l0
    conv = (width0 - w_ref) / np.maximum(-x_ref, 1e-6)
    add("ch_width", np.clip(width0 / rb, -20.0, 20.0), "channel")
    add("ch_conv", np.clip(conv, -20.0, 20.0), "channel")

    # ---- levels
    ys = np.column_stack([Y, yd])
    above = np.where(ys > 0, ys, np.inf).min(axis=1) / rb
    below = np.where(ys < 0, -ys, np.inf).min(axis=1) / rb
    add("lvl_above", np.where(np.isfinite(above), above, np.nan), "levels")
    add("lvl_below", np.where(np.isfinite(below), below, np.nan), "levels")
    add("lvl_count", (np.abs(ys) <= rb[:, None]).sum(axis=1), "levels")

    # ---- past touch rate (labels closed by t)
    for k in PAST_WINDOWS:
        add(f"past_touch_{k}", _roll(touch_s, k, "mean", max(10, k // 2))[ts - H], "past")

    groups = set(cfg.touch_groups)
    reg = [r for r in reg if r[1] in groups]
    return {nm: f[nm] for nm, _ in reg}, reg


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #
def bce_np(prob, y) -> float:
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def bce_logits_np(logits, y) -> float:
    z = np.asarray(logits, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, z) - np.asarray(y, dtype=np.float64) * z))


def safe_auc(y, score) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, score)) if len(y) and len(np.unique(y)) == 2 else float("nan")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def fit_temperature(logits, y) -> float:
    """Scalar T > 0 minimising log loss of sigmoid(logit / T)."""
    if len(y) < 50 or len(np.unique(y)) < 2:
        return 1.0
    grid = np.exp(np.linspace(math.log(0.25), math.log(100.0), 241))
    return float(grid[int(np.argmin([bce_logits_np(logits / t, y) for t in grid]))])


def block_bootstrap_ci(fn, blocks: np.ndarray, *arrays, n_boot: int = 200, seed: int = 0,
                       alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of fn(*arrays) resampling whole blocks (UTC days) - robust to overlapping labels."""
    if n_boot <= 0 or len(blocks) == 0:
        return float("nan"), float("nan")
    _, inv = np.unique(blocks, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv)
    groups = np.split(order, np.cumsum(counts)[:-1])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        vals.append(fn(*(a[idx] for a in arrays)))
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def binary_report(y, p, prior: float) -> dict:
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if len(y) == 0:
        return {"samples": 0}
    ll, prior_ll = bce_np(p, y), bce_np(np.full(len(y), prior), y)
    pred = (p >= 0.5).astype(np.int64)
    tn, fp = int(np.sum((y == 0) & (pred == 0))), int(np.sum((y == 0) & (pred == 1)))
    fn, tp = int(np.sum((y == 1) & (pred == 0))), int(np.sum((y == 1) & (pred == 1)))
    div = lambda a, b: float(a / b) if b else None  # noqa: E731
    recall, specificity = div(tp, tp + fn), div(tn, tn + fp)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "samples": int(len(y)), "positive_rate": float(y.mean()), "train_prior": float(prior),
        "log_loss": ll, "prior_log_loss": prior_ll,
        "log_loss_skill": float(1.0 - ll / prior_ll) if prior_ll > 0 else float("nan"),
        "auc": safe_auc(y, p), "accuracy": float(np.mean(pred == y)),
        "majority_accuracy": float(max(y.mean(), 1.0 - y.mean())),
        "balanced_accuracy": (recall + specificity) / 2.0 if recall is not None and specificity is not None else None,
        "precision": div(tp, tp + fp), "recall": recall, "specificity": specificity,
        "mcc": float((tp * tn - fp * fn) / mcc_den) if mcc_den else None,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def reliability_bins(pred, real, k: int = 10) -> list[dict]:
    pred, real = np.asarray(pred, dtype=np.float64), np.asarray(real, dtype=np.float64)
    if len(pred) == 0:
        return []
    edges = np.unique(np.quantile(pred, np.linspace(0, 1, k + 1)))
    which = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, max(len(edges) - 2, 0))
    return [{"bin": int(b), "samples": int((which == b).sum()), "mean_pred": float(pred[which == b].mean()),
             "mean_realised": float(real[which == b].mean())} for b in range(max(len(edges) - 1, 1)) if (which == b).any()]


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #
def _utc_ns(year: int) -> int:
    return pd.Timestamp(f"{year}-01-01", tz="UTC").value


def fold_ids(ds: Dataset, year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """train: labels end before Jan-1 of year-1 | val: all of year-1 | test: year. Boundary labels purged."""
    val_start, test_start, test_end = _utc_ns(year - 1), _utc_ns(year), _utc_ns(year + 1)
    ts, end = ds.timestamp_ns, ds.label_end_ns
    ids = np.arange(len(ts))
    train = ids[end < val_start]
    val = ids[(ts >= val_start) & (end < test_start)]
    test = ids[(ts >= test_start) & (end < test_end)]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(f"Incomplete walk-forward split for {year}")
    return train, val, test


def split_validation(ds: Dataset, val_ids: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
    c = int(len(val_ids) * frac)
    es, cal = val_ids[:c], val_ids[c:]
    if len(es) == 0 or len(cal) == 0:
        raise ValueError("Validation year too small to split")
    es = es[ds.label_end_ns[es] < ds.timestamp_ns[cal[0]]]
    if len(es) == 0:
        raise ValueError("Purging removed the whole early-stopping set")
    return es, cal


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    decision_idx: np.ndarray
    timestamp_ns: np.ndarray
    label_end_ns: np.ndarray | None
    close: np.ndarray
    barrier_pct: np.ndarray
    x: np.ndarray
    names: tuple[str, ...]
    feature_groups: dict
    labels: dict | None            # None ở chế độ live


def build_dataset(df: pd.DataFrame, pivots: pd.DataFrame, cfg: Config, live: bool = False,
                  live_last: int | None = None) -> Dataset:
    """Feature nhân quả tại lúc nến quyết định đóng (+ nhãn nếu không phải live)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return _build_dataset(df, pivots, cfg, live, live_last)


def _build_dataset(df, pivots, cfg: Config, live: bool, live_last: int | None) -> Dataset:
    n = len(df)
    P, H, L = cfg.pivot_count, cfg.barrier_horizon, cfg.lookback_bars
    if len(pivots) <= P:
        raise ValueError(f"Only {len(pivots)} pivots found; need more than pivot_count={P}")
    bf = compute_bar_features(df, cfg)
    pre = path_prefix(bf)
    sigma = bf["sigma"]
    open_, high, low, close = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    ns = _ns_int(df["timestamp"])
    confirm_idx = pivots["confirm_idx"].to_numpy(np.int64)
    gap_cum = np.cumsum(gap_flags(df, cfg.bar_minutes))
    bar_barrier = np.clip(cfg.barrier_vol_mult * sigma * math.sqrt(H), cfg.min_barrier, cfg.max_barrier)

    W = cfg.gap_lookback
    first = max(cfg.slow_vol_span + L, int(confirm_idx[P - 1]), W + 1)
    if live:
        ts = np.arange(first, n, dtype=np.int64)
        ok = gap_cum[ts] - gap_cum[ts - W] == 0
    else:
        ts = np.arange(first, n - H, cfg.sample_stride, dtype=np.int64)
        ok = (gap_cum[ts] - gap_cum[ts - W] == 0) & (gap_cum[ts + H] - gap_cum[ts] == 0)
    ts = ts[ok]
    known_end = np.searchsorted(confirm_idx, ts, side="right")
    dev_idx, _ = _developing_kernel(high, low, confirm_idx, pivots["kind"].to_numpy(np.int64))
    keep = (known_end >= P) & (dev_idx[ts] >= 0) & np.isfinite(sigma[ts]) & np.isfinite(bar_barrier[ts])
    ts, known_end = ts[keep], known_end[keep]
    if live and live_last:
        ts, known_end = ts[-live_last:], known_end[-live_last:]
    if len(ts) == 0:
        raise ValueError("No decision bars available; check data length / gaps")

    touch_s = touch_label_series(open_, high, low, bar_barrier, H)
    feats, reg = touch_features(df, pivots, bf, pre, bar_barrier, touch_s, ts, known_end, cfg)
    names = tuple(nm for nm, _ in reg)
    labels, end_ns = None, None
    if not live:
        labels = make_labels(open_, high, low, close, ts, bar_barrier[ts], H)
        end_ns = ns[ts + H]
    return Dataset(decision_idx=ts, timestamp_ns=ns[ts], label_end_ns=end_ns, close=close[ts],
                   barrier_pct=bar_barrier[ts], x=np.column_stack([feats[nm] for nm in names]).astype(np.float32),
                   names=names, feature_groups=dict(reg), labels=labels)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def make_xgb(p: XGBParams, cfg: Config, seed: int):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Install XGBoost first: pip install xgboost") from exc
    params = dict(n_estimators=p.n_estimators, learning_rate=p.learning_rate, max_depth=p.max_depth,
                  min_child_weight=p.min_child_weight, subsample=p.subsample, colsample_bytree=p.colsample_bytree,
                  reg_alpha=p.reg_alpha, reg_lambda=p.reg_lambda, gamma=p.gamma,
                  early_stopping_rounds=p.early_stopping_rounds, tree_method="hist", n_jobs=cfg.xgb_n_jobs,
                  random_state=seed)
    if cfg.xgb_device != "cpu":
        params["device"] = cfg.xgb_device
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **params)


def _time_weights(ds: Dataset, ids: np.ndarray, ref_ns: int, cfg: Config) -> np.ndarray | None:
    if cfg.time_decay_halflife_days <= 0:
        return None
    age_days = np.maximum(ref_ns - ds.timestamp_ns[ids], 0) / 86_400e9
    return np.maximum(0.5 ** (age_days / cfg.time_decay_halflife_days), cfg.time_decay_floor).astype(np.float32)


def raw_proba(models: dict, x: np.ndarray) -> np.ndarray:
    model, best = models["touch"]
    return np.asarray(model.predict_proba(x, iteration_range=(0, best))[:, 1], dtype=np.float64)


def predict_touch(models: dict, x: np.ndarray) -> np.ndarray:
    """P(touch) đã hiệu chỉnh (temperature scaling)."""
    return sigmoid(logit(raw_proba(models, x)) / max(models["T_touch"], 1e-4))


def fit_models(ds: Dataset, train_ids: np.ndarray, val_ids: np.ndarray, cfg: Config) -> dict:
    """Dừng sớm trên phần đầu năm validation; temperature + ngưỡng trên phần còn lại."""
    y = ds.labels["y_touch"]
    es_ids, cal_ids = split_validation(ds, val_ids, cfg.val_es_frac)
    w = _time_weights(ds, train_ids, int(ds.timestamp_ns[train_ids].max()), cfg)
    model = make_xgb(cfg.touch, cfg, cfg.seed)
    model.fit(ds.x[train_ids], y[train_ids], sample_weight=w, eval_set=[(ds.x[es_ids], y[es_ids])], verbose=False)
    best = int(getattr(model, "best_iteration", model.n_estimators - 1)) + 1
    if cfg.xgb_device != "cpu":
        model.set_params(device="cpu")
        model.get_booster().set_param({"device": "cpu"})
    models = {"touch": (model, best), "T_touch": 1.0, "train_rate": float(y[train_ids].mean())}
    if cfg.calibrate:
        models["T_touch"] = fit_temperature(logit(raw_proba(models, ds.x[cal_ids])), y[cal_ids])
    p_cal = predict_touch(models, ds.x[cal_ids])
    models["thresholds"] = {q: float(np.quantile(p_cal, 1.0 - q)) for q in cfg.top_fractions}
    return {"models": models, "calib_ids": cal_ids, "best_rounds": best,
            "samples": {"train": int(len(train_ids)), "val_early_stop": int(len(es_ids)),
                        "val_calibration": int(len(cal_ids))}}


# --------------------------------------------------------------------------- #
# Đánh giá
# --------------------------------------------------------------------------- #
def top_table(p: np.ndarray, y: np.ndarray, thresholds: dict, decision_idx: np.ndarray, horizon: int) -> list[dict]:
    """Tỉ lệ chạm thực tế của các nến có p_touch >= ngưỡng (ngưỡng lấy từ phần calibration)."""
    rows = []
    for q, thr in thresholds.items():
        sel = p >= thr
        idx = np.flatnonzero(sel)
        keep, nxt = [], -1
        for i in idx:                                   # tín hiệu không chồng nhau (cách nhau >= H nến)
            if decision_idx[i] >= nxt:
                keep.append(i)
                nxt = decision_idx[i] + horizon
        rows.append({"top_fraction": q, "threshold": thr, "selected_share": float(sel.mean()),
                     "samples": int(sel.sum()), "mean_pred": float(p[sel].mean()) if sel.any() else np.nan,
                     "touch_rate": float(y[sel].mean()) if sel.any() else np.nan,
                     "independent": len(keep), "independent_touch_rate": float(y[keep].mean()) if keep else np.nan})
    return rows


def evaluate(ds: Dataset, p: np.ndarray, ids: np.ndarray, train_rate: float, thresholds: dict, cfg: Config) -> dict:
    y = ds.labels["y_touch"][ids]
    day = ds.timestamp_ns[ids] // 86_400_000_000_000
    rep = binary_report(y, p, train_rate)
    rep["auc_ci"] = block_bootstrap_ci(safe_auc, day, y, p, n_boot=cfg.n_boot, seed=cfg.seed)
    rep["reliability"] = reliability_bins(p, y, 10)
    rep["top"] = top_table(p, y, thresholds, ds.decision_idx[ids], cfg.barrier_horizon)
    month = np.asarray(pd.DatetimeIndex(pd.to_datetime(ds.timestamp_ns[ids], utc=True)).tz_localize(None)
                       .to_period("M").astype(str))
    rep["monthly"] = [{"month": m, "samples": int((month == m).sum()), "touch_rate": float(y[month == m].mean()),
                       "mean_pred": float(p[month == m].mean()), "auc": safe_auc(y[month == m], p[month == m])}
                      for m in np.unique(month)]
    return rep


def feature_importance(models: dict, ds: Dataset, top: int = 25) -> dict:
    imp = np.asarray(models["touch"][0].feature_importances_, dtype=np.float64)
    order = np.argsort(imp)[::-1][:top]
    groups = pd.Series(imp, index=list(ds.names)).groupby(lambda n: ds.feature_groups[n]).sum()
    return {"top": [{"feature": ds.names[i], "gain_share": float(imp[i])} for i in order],
            "groups": {g: float(v) for g, v in groups.sort_values(ascending=False).items()}}


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_predictions(path: Path, ds: Dataset, ids: np.ndarray, p: np.ndarray) -> None:
    np.savez_compressed(path, sample_id=ids, timestamp_ns=ds.timestamp_ns[ids], decision_idx=ds.decision_idx[ids],
                        close=ds.close[ids], barrier_pct=ds.barrier_pct[ids], p_touch=p,
                        **{f"label_{k}": v[ids] for k, v in ds.labels.items()})


def _fold_key(cfg: Config, ds: Dataset) -> str:
    c = {k: v for k, v in asdict(cfg).items() if k not in ("eval_years", "verbose", "output_dir", "csv_path",
                                                              "xgb_device", "xgb_n_jobs", "n_boot")}
    return json.dumps({"cfg": c, "features": list(ds.names), "samples": int(len(ds.decision_idx)),
                       "last_ts": int(ds.timestamp_ns[-1])}, sort_keys=True, default=str)


def run_fold(ds: Dataset, year: int, cfg: Config, out: Path | None = None) -> tuple[dict, dict]:
    """Train + đánh giá một năm. Có `out`: lưu predictions_<năm>.npz + fold_<năm>.joblib và dùng lại nếu
    config / feature / dữ liệu không đổi (chạy tiếp được khi bị ngắt)."""
    import joblib
    path = out / f"fold_{year}.joblib" if out is not None else None
    key = _fold_key(cfg, ds)
    if path is not None and path.exists():
        saved = joblib.load(path)
        if saved.get("key") == key:
            return saved["report"], saved["fitted"]
    train_ids, val_ids, test_ids = fold_ids(ds, year)
    fitted = fit_models(ds, train_ids, val_ids, cfg)
    m = fitted["models"]
    p_cal = predict_touch(m, ds.x[fitted["calib_ids"]])
    p_test = predict_touch(m, ds.x[test_ids])
    rep = {"fold": str(year), "samples": fitted["samples"], "best_rounds": fitted["best_rounds"],
           "temperature": m["T_touch"], "thresholds_from_calibration": m["thresholds"],
           "validation_metrics": evaluate(ds, p_cal, fitted["calib_ids"], m["train_rate"], m["thresholds"], cfg),
           "heldout_metrics": evaluate(ds, p_test, test_ids, m["train_rate"], m["thresholds"], cfg),
           "importance": feature_importance(m, ds)}
    if path is not None:
        save_predictions(out / f"predictions_{year}.npz", ds, test_ids, p_test)
        joblib.dump({"key": key, "report": rep, "fitted": fitted}, path)
    return rep, fitted


def summary_line(rep: dict) -> str:
    h = rep["heldout_metrics"]
    top = next((r for r in h["top"] if r["top_fraction"] == 0.1), {})
    return (f"{rep['fold']:>6} | AUC {h['auc']:.3f} [{h['auc_ci'][0]:.3f},{h['auc_ci'][1]:.3f}] "
            f"| log-loss skill {h['log_loss_skill']:+.3f} | touch rate {h['positive_rate']:.3f} "
            f"| top10%: {top.get('touch_rate', float('nan')):.3f}")


def export_bundle(path: Path, models: dict, ds: Dataset, cfg: Config) -> None:
    import joblib
    joblib.dump({"strategy": STRATEGY, "config": cfg, "models": models, "names": ds.names}, path)


def run_pipeline(cfg: Config) -> dict:
    """Walk-forward theo năm + export model của năm đánh giá cuối."""
    cfg = replace(cfg, xgb_device=resolve_device(cfg.xgb_device))
    seed_everything(cfg.seed)
    df = load_candles(cfg.csv_path, cfg.bar_minutes)
    pivots = find_confirmed_pivots(df, cfg.pivot_reversal)
    ds = build_dataset(df, pivots, cfg)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports, fitted = [], None
    for year in cfg.eval_years:
        rep, fitted = run_fold(ds, year, cfg, out)
        reports.append(rep)
        if cfg.verbose:
            print(summary_line(rep), flush=True)
    export_bundle(out / "models_final.joblib", fitted["models"], ds, cfg)
    result = {"candles": len(df), "pivots": len(pivots), "samples": len(ds.decision_idx), "features": len(ds.names),
              "folds": reports, "final_fold": reports[-1]["fold"], "output_dir": str(out)}
    (out / "metadata.json").write_text(json.dumps({"strategy": STRATEGY, "config": asdict(cfg), **result},
                                                  indent=2, default=_json_default), encoding="utf-8")
    return result


# --------------------------------------------------------------------------- #
# Live / batch inference
# --------------------------------------------------------------------------- #
def load_bundle(path: str | Path) -> dict:
    import joblib
    return joblib.load(path)


def predict_frame(bundle: dict, candles: pd.DataFrame, last_n: int | None = 1,
                  funding: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tín hiệu cho các nến ĐÃ ĐÓNG gần nhất của một bảng kline (cần >= ~20 ngày nến 5m).

    funding: các lần funding đã chốt (bảng BTCUSDT_funding.csv hoặc /fapi/v1/fundingRate) - bắt buộc nếu bundle
    được train có funding. Cột trả về: timestamp, close, barrier_pct, p_touch, up_level, down_level, time_stop_bars."""
    cfg: Config = bundle["config"]
    df = prepare_candles(candles, cfg.bar_minutes)
    if funding is not None:
        df = attach_funding(df, funding, cfg.bar_minutes)
    pivots = find_confirmed_pivots(df, cfg.pivot_reversal)
    ds = build_dataset(df, pivots, cfg, live=True, live_last=last_n)
    if ds.names != tuple(bundle["names"]):
        missing = sorted(set(bundle["names"]) - set(ds.names))
        raise ValueError(f"feature set differs from the exported bundle (trades / funding columns?) - missing: {missing[:10]}")
    p = predict_touch(bundle["models"], ds.x)
    frame = pd.DataFrame({"timestamp": pd.to_datetime(ds.timestamp_ns, utc=True), "close": ds.close,
                          "barrier_pct": ds.barrier_pct, "p_touch": p})
    frame["up_level"] = frame["close"] * (1 + frame["barrier_pct"])
    frame["down_level"] = frame["close"] * (1 - frame["barrier_pct"])
    frame["time_stop_bars"] = cfg.barrier_horizon
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the v2 touch model")
    d = Config()
    parser.add_argument("--csv", default=d.csv_path)
    parser.add_argument("--output-dir", default=d.output_dir)
    parser.add_argument("--xgb-device", default=d.xgb_device, help="auto | cpu | cuda:0")
    parser.add_argument("--sample-stride", type=int, default=d.sample_stride)
    parser.add_argument("--horizon", type=int, default=d.barrier_horizon)
    parser.add_argument("--years", type=int, nargs="+", default=list(d.eval_years))
    parser.add_argument("--n-boot", type=int, default=d.n_boot)
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--quiet", action="store_true")
    a = parser.parse_args()
    cfg = Config(csv_path=a.csv, output_dir=a.output_dir, xgb_device=a.xgb_device, sample_stride=a.sample_stride,
                 barrier_horizon=a.horizon, eval_years=tuple(sorted(set(a.years))), n_boot=a.n_boot, seed=a.seed,
                 verbose=not a.quiet)
    result = run_pipeline(cfg)
    print(json.dumps({"output_dir": result["output_dir"], "folds": [r["fold"] for r in result["folds"]]}, indent=2))


if __name__ == "__main__":
    main()
