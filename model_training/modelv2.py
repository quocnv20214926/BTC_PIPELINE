"""BTCUSDT - model TOUCH: xác suất giá chạm +b HOẶC -b trong H nến tới (tính từ open[t+1]).

Dùng cho chiến lược H1 (strategy_h1.py): nến 1h, H = 8, b = clip(sigma_EWM168 * sqrt(8), 1%, 5%).
Phần này chỉ chứa: chuẩn hoá nến, funding, pivot ZigZag (nhân quả), nhãn, feature touch, dataset, XGBoost touch
(trọng số thời gian + temperature scaling). Walk-forward, tín hiệu, backtest: strategy_h1.py.
Feature touch: xem touch_features; tất cả tính tại lúc nến quyết định đóng, chỉ dùng dữ liệu đã biết.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:  # numba: các vòng lặp pivot từ vài phút xuống vài giây
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

TAKER_COLUMNS = ("taker_buy_base_volume", "taker_buy_base_asset_volume", "taker_buy_volume", "taker_buy_base")
TOUCH_GROUPS = ("window", "legs", "relations", "positions", "channel", "levels", "context")
FUNDING_FILE = "BTCUSDT_funding.csv"
FUNDING_COLUMNS = ("funding_rate", "funding_mean_3", "funding_z_90", "funding_dev_90")
WINDOWS = (4, 8, 12, 24)          # cửa sổ (nến)
POSITIONS = 2                     # khoảng cách tới 2 pivot đã xác nhận gần nhất
LEGS = 5                          # 5 sóng đã hoàn tất gần nhất
LEVEL_PIVOTS = 5                  # levels: 5 pivot gần nhất + pivot đang hình thành
FEATURE_HISTORY_BARS = 1440       # mẫu cần 1440 + H nến liên tục phía trước (giữ đúng tập mẫu của các thử nghiệm)


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
    bar_minutes: int = 60
    pivot_count: int = 8
    lookback_bars: int = 168
    barrier_horizon: int = 8
    barrier_vol_mult: float = 1.0
    min_barrier: float = 0.01
    max_barrier: float = 0.05
    vol_span: int = 168
    slow_vol_span: int = 720
    sample_stride: int = 1
    touch_groups: tuple[str, ...] = TOUCH_GROUPS
    xgb_device: str = "cpu"
    xgb_n_jobs: int = -1
    time_decay_halflife_days: float = 540.0
    time_decay_floor: float = 0.2
    touch: XGBParams = field(default_factory=XGBParams)

    def __post_init__(self) -> None:
        bad = [k for k in ("bar_minutes", "barrier_horizon", "vol_span", "slow_vol_span", "lookback_bars", "sample_stride")
               if getattr(self, k) < 1]
        if bad:
            raise ValueError(f"must be >= 1: {', '.join(bad)}")
        if 1440 % self.bar_minutes:
            raise ValueError("bar_minutes must divide one day (1440 minutes)")
        if self.pivot_count < max(LEGS + 1, LEVEL_PIVOTS, 4):
            raise ValueError(f"pivot_count must be >= {max(LEGS + 1, LEVEL_PIVOTS, 4)}")
        if self.slow_vol_span < self.trend_long_bars:
            raise ValueError(f"slow_vol_span must be >= {self.trend_long_bars}")
        if not 0 < self.min_barrier <= self.max_barrier < 1:
            raise ValueError("need 0 < min_barrier <= max_barrier < 1")
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
        """Đoạn nến dài nhất phía trước mẫu phải liền mạch (có nến thiếu trong đoạn này thì bỏ mẫu)."""
        return max(self.lookback_bars, FEATURE_HISTORY_BARS + self.barrier_horizon, self.trend_long_bars)


# --------------------------------------------------------------------------- #
# Nến, funding
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
# Pivot ZigZag (nhân quả) và nhãn
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
# Feature touch
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


def touch_features(df, pivots, bf, pre, bar_barrier, ts, known_end, cfg: Config):
    """Feature touch (không có hướng) tại các nến quyết định ts. known_end[i] = số pivot đã xác nhận tới ts[i].
      context    sigma 1h / 30 ngày, xu hướng 1 / 5 ngày (có dấu và trị tuyệt đối), giờ / thứ, mức sigma, b, b / s,
                 rvol (z-score bền của log volume so với 1 ngày), cường độ / cỡ lệnh, |funding|
      window     cửa sổ WINDOWS nến: độ dời, efficiency ratio, biến động thực, spike, khoảng cách tới đỉnh / đáy, biên độ
      positions  khoảng cách (thời gian, giá) tới POSITIONS pivot đã xác nhận gần nhất
      legs       LEGS sóng đã hoàn tất + sóng hiện tại (c1) + đoạn hồi (c2) + tỉ lệ hồi
      relations  tỉ lệ độ dài / thời gian giữa c1-l1, l1-l2, l2-l3
      channel    độ rộng / độ hội tụ kênh 2 đỉnh + 2 đáy gần nhất
      levels     khoảng cách tới mức pivot gần nhất trên / dưới (LEVEL_PIVOTS pivot + pivot đang hình thành), mật độ mức
    Đơn vị: s = sigma * sqrt(H) (biên độ 8h kỳ vọng), rb = b / s."""
    H, L = cfg.barrier_horizon, cfg.lookback_bars
    NP = max(LEVEL_PIVOTS, 4, POSITIONS)          # số pivot đã xác nhận cần cho positions / channel / levels
    sigma, logc = bf["sigma"], bf["logc"]
    high, low, close = (df[k].to_numpy(np.float64) for k in ("high", "low", "close"))
    pivot_idx = pivots["pivot_idx"].to_numpy(np.int64)
    pivot_price = pivots["price"].to_numpy(np.float64)
    pivot_kind = pivots["kind"].to_numpy(np.float64)
    dev_idx, dev_price = _developing_kernel(high, low, pivots["confirm_idx"].to_numpy(np.int64),
                                            pivots["kind"].to_numpy(np.int64))
    if int(known_end.min()) < max(NP, LEGS + 1):
        raise ValueError("touch_features: chưa đủ pivot đã xác nhận ở mẫu đầu (tăng pivot_count)")
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

    # ---- context (như v2 + rvol)
    for name in ("sigma_ratio", "trend_1d_z", "trend_5d_z", "hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        add(name, bf[name][ts], "context")
    add("log_sigma", np.log(sig_t), "context")
    add("barrier_pct", barrier, "context")
    add("barrier_ratio", rb, "context")
    add("abs_trend_1d_z", np.abs(f["trend_1d_z"]), "context")
    add("abs_trend_5d_z", np.abs(f["trend_5d_z"]), "context")
    add("rvol", bf["rvol"][ts], "context")
    if "trades" in df.columns:
        lt = np.log1p(df["trades"].to_numpy(np.float64))
        med = pd.Series(lt).rolling(cfg.bars_per_day, min_periods=cfg.bars_per_day // 4).median().to_numpy()
        add("trades_rz_6", _roll(lt - med, 6, "mean")[ts], "context")
        add("trades_rz_48", _roll(lt - med, 48, "mean")[ts], "context")
        size = np.log1p(df["volume"].to_numpy(np.float64)) - lt
        smed = pd.Series(size).rolling(cfg.bars_per_day, min_periods=cfg.bars_per_day // 4).median().to_numpy()
        add("trade_size_rz_12", _roll(size - smed, 12, "mean")[ts], "context")
    if "funding_rate" in df.columns:
        add("funding_abs", np.abs(df["funding_rate"].to_numpy(np.float64))[ts] * 1e4, "context")

    # ---- window: 4 / 8 / 12 / 24 nến
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

    # ---- positions: 2 pivot đã xác nhận gần nhất
    sel = known_end[:, None] - NP + np.arange(NP)[None, :]
    pi, pp, pk = pivot_idx[sel], pivot_price[sel], pivot_kind[sel]
    X = (pi - ts[:, None]) / H
    Y = np.log(pp / close_t[:, None]) / s[:, None]
    last = known_end - 1
    di, dp = dev_idx[ts], dev_price[ts]
    yd = np.log(dp / close_t) / s
    for j in range(1, POSITIONS + 1):
        add(f"pos{j}_r", np.hypot(X[:, NP - j], Y[:, NP - j]), "positions")

    # ---- legs: 5 sóng đã hoàn tất + sóng hiện tại
    legs = completed_legs(pivots, bf, pre)
    leg_info: dict[str, dict] = {}
    for j in range(1, LEGS + 1):
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

    # ---- relations (giữ nguyên)
    for x_leg, y_leg in (("c1", "l1"), ("l1", "l2"), ("l2", "l3")):
        a_, b_ = leg_info[x_leg], leg_info[y_leg]
        add(f"rel_{x_leg}_{y_leg}_size",
            np.log(np.maximum(np.abs(a_["disp"]), 1e-9) / np.maximum(np.abs(b_["disp"]), 1e-9)), "relations")
        add(f"rel_{x_leg}_{y_leg}_time", np.log(a_["dur"] / b_["dur"]), "relations")

    # ---- channel (giữ nguyên: 4 pivot cuối)
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

    # ---- levels: 5 pivot gần nhất + pivot đang hình thành
    ys = np.column_stack([Y[:, NP - LEVEL_PIVOTS:], yd])
    above = np.where(ys > 0, ys, np.inf).min(axis=1) / rb
    below = np.where(ys < 0, -ys, np.inf).min(axis=1) / rb
    add("lvl_above", np.where(np.isfinite(above), above, np.nan), "levels")
    add("lvl_below", np.where(np.isfinite(below), below, np.nan), "levels")
    add("lvl_count", (np.abs(ys) <= rb[:, None]).sum(axis=1), "levels")

    groups = set(cfg.touch_groups)
    reg = [r for r in reg if r[1] in groups]
    return {nm: f[nm] for nm, _ in reg}, reg


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

    feats, reg = touch_features(df, pivots, bf, pre, bar_barrier, ts, known_end, cfg)
    names = tuple(nm for nm, _ in reg)
    labels, end_ns = None, None
    if not live:
        labels = make_labels(open_, high, low, close, ts, bar_barrier[ts], H)
        end_ns = ns[ts + H]
    return Dataset(decision_idx=ts, timestamp_ns=ns[ts], label_end_ns=end_ns, close=close[ts],
                   barrier_pct=bar_barrier[ts], x=np.column_stack([feats[nm] for nm in names]).astype(np.float32),
                   names=names, feature_groups=dict(reg), labels=labels)


# --------------------------------------------------------------------------- #
# Model touch: XGBoost + temperature scaling
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


def bce_logits_np(logits, y) -> float:
    z = np.asarray(logits, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, z) - np.asarray(y, dtype=np.float64) * z))


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
