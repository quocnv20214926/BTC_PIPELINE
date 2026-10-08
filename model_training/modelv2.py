"""BTCUSDT 5m signal model (v2) - probabilistic, trade-design oriented.

Why v2 (findings from the v1 evaluation)
---------------------------------------
* Touch XGBoost is the strong, well-calibrated part -> kept unchanged.
* The v1 five-class shape head mostly re-learned volatility (CHOP vs non-CHOP) and diluted
  the small directional signal; its blend with the R-regression head was dominated by the
  shape head. -> replaced by a factorised set of small, single-purpose models.
* Directional information sits in large-scale structure (big ZigZag swings, multi-day VWAP,
  1-5 day momentum). -> more large-scale features (ZigZag 4x barrier, 10-day lags/VWAP,
  previous day / week high-low anchors).

Models (all features causal at the close of decision bar t; outcomes measured from the
executable reference open[t+1] over candles t+1..t+H, barrier b = clip(k * sigma * sqrt(H)))
  1. touch   : P(touch)            = P(+b or -b is reached within H)
  2. dir     : P(up | touch)       = P(+b reached before -b | a barrier is reached)
  3. ext     : P(ext | up first)   = P(+m*b reached before -b | +b reached first)   (m = ext_mult)
  4. timeout : E[R_end | no touch] = expected close-out return (barrier units) when nothing is hit
dir / ext / timeout are trained with mirror augmentation (price -> -price) and their predictions
are exactly mirror-symmetric, so the model has no structural long/short bias.

Signals produced for every decision bar (columns of `predict_signals`) - the building blocks
for execution / backtest design:
  barrier_pct                    b, as a fraction of price (TP/SL distance for a 1R trade)
  p_touch                        probability that price travels at least b within H bars
  p_up_touch                     direction given movement (0.5 = no view)
  p_up_first, p_dn_first         probability that +b / -b is the FIRST barrier hit
  p_tie, p_none                  both barriers in the same candle / no barrier within H
  p_ext_long, p_ext_short        P(reach m*b before -b | +b first) and its mirror for shorts
  mu_timeout                     expected R (long) at the horizon close when nothing is hit
  e_long_1r, e_short_1r          expected R of TP = b, SL = b, time stop H  (gross, R units)
  e_long_2r, e_short_2r          expected R of TP = m*b, SL = b, time stop H (gross, R units)
  best_side, best_rr, best_e     the best of the four trade templates and its expected R
Fees are NOT included in e_*: a round-trip fee costs fee_pct / barrier_pct R.

Evaluation: pure yearly walk-forward with purged boundaries (as v1). For eval year Y:
  train = labels ending before Jan-1 of Y-1, validation = Y-1 (early-stop part | calibration part),
  test = Y. Metrics come with day-block bootstrap confidence intervals, the realised R of the
  signals is reported gross and net of fees, and top-signal tables use thresholds taken from the
  calibration part (no look-ahead) and count non-overlapping signals. Still no order simulation.
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

try:  # optional: ZigZag / pivot loops go from minutes to seconds
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f


STRATEGY = "touch_dir_ext_signal_v2"
TAKER_COLUMNS = ("taker_buy_base_volume", "taker_buy_base_asset_volume", "taker_buy_volume", "taker_buy_base")
TOUCH_GROUPS = ("window", "legs", "relations", "positions", "channel", "levels", "past", "context")
DIR_GROUPS = ("move", "range", "flow", "candle", "vwap", "structure", "anchors", "funding", "regime")
FUNDING_FILE = "BTCUSDT_funding.csv"
FUNDING_COLUMNS = ("funding_rate", "funding_mean_3", "funding_z_90", "funding_dev_90")
TRADE_TEMPLATES = ("long_1r", "short_1r", "long_2r", "short_2r")
RR_SETS = {"1r": ("long_1r", "short_1r"), "2r": ("long_2r", "short_2r"), "best": TRADE_TEMPLATES}

WINDOWS = (6, 12, 24, 48, 96, 288)
PAST_WINDOWS = (288, 576, 1440)
DIR_MOVE_LAGS = (1, 3, 6, 12, 24, 48, 96, 288, 1440, 2880)
DIR_ER_WINDOWS = (12, 48, 288, 1440)
DIR_RANGE_WINDOWS = (12, 48, 288, 1440)
DIR_FLOW_WINDOWS = (1, 6, 24, 96, 288)
DIR_VUD_WINDOWS = (6, 24, 96, 288)
DIR_CANDLE_WINDOWS = (1, 3, 12)
DIR_VWAP_WINDOWS = (48, 288, 1440, 2880)
FLOW_BASELINE = 2880


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


def _touch_params() -> XGBParams:
    return XGBParams()


def _dir_params() -> XGBParams:
    return XGBParams(n_estimators=2000, learning_rate=0.02, max_depth=3, min_child_weight=200.0,
                     subsample=0.6, colsample_bytree=0.5, reg_lambda=30.0, early_stopping_rounds=200)


def _ext_params() -> XGBParams:
    return XGBParams(n_estimators=1500, learning_rate=0.02, max_depth=3, min_child_weight=100.0,
                     subsample=0.6, colsample_bytree=0.5, reg_lambda=30.0, early_stopping_rounds=150)


def _timeout_params() -> XGBParams:
    return XGBParams(n_estimators=1500, learning_rate=0.02, max_depth=3, min_child_weight=300.0,
                     subsample=0.6, colsample_bytree=0.5, reg_lambda=30.0, early_stopping_rounds=150)


@dataclass
class Config:
    csv_path: str = "historical_data/BTCUSDT_5m_all.csv"
    output_dir: str = "artifacts/5m_xgboost_signal_v2"
    bar_minutes: int = 5
    # ---- barrier / labels
    pivot_reversal: float = 0.005
    pivot_count: int = 8
    lookback_bars: int = 288
    barrier_horizon: int = 48
    barrier_vol_mult: float = 1.0
    min_barrier: float = 0.007
    max_barrier: float = 0.03
    ext_mult: float = 2.0                     # TP multiple of the "2r" template
    vol_span: int = 288
    slow_vol_span: int = 2880
    sample_stride: int = 1
    # ---- features
    touch_groups: tuple[str, ...] = TOUCH_GROUPS
    dir_groups: tuple[str, ...] = DIR_GROUPS
    dir_zz_scales: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    # ---- protocol
    eval_years: tuple[int, ...] = (2022, 2023, 2024, 2025, 2026)
    val_es_frac: float = 0.6
    # ---- training
    xgb_device: str = "auto"                  # "auto" | "cpu" | "cuda:N"
    xgb_n_jobs: int = -1
    time_decay_halflife_days: float = 540.0
    time_decay_floor: float = 0.2
    calibrate: bool = True                    # temperature scaling of touch / dir / ext
    dir_train_stride: int = 2                 # subsampling of the (mirrored) direction-type training sets
    touch: XGBParams = field(default_factory=_touch_params)
    dir: XGBParams = field(default_factory=_dir_params)
    ext: XGBParams = field(default_factory=_ext_params)
    timeout: XGBParams = field(default_factory=_timeout_params)
    # ---- evaluation only
    fee_pct_roundtrip: float = 0.0008         # 0.08 % round trip (taker 0.04 % x 2)
    n_boot: int = 200
    top_fractions: tuple[float, ...] = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02)
    seed: int = 42
    verbose: bool = True

    def __post_init__(self) -> None:
        positive = {
            "bar_minutes": self.bar_minutes, "barrier_horizon": self.barrier_horizon, "vol_span": self.vol_span,
            "slow_vol_span": self.slow_vol_span, "pivot_count": self.pivot_count,
            "lookback_bars": self.lookback_bars, "sample_stride": self.sample_stride,
            "dir_train_stride": self.dir_train_stride,
        }
        bad = [k for k, v in positive.items() if v < 1]
        if bad:
            raise ValueError(f"must be >= 1: {', '.join(bad)}")
        if 1440 % self.bar_minutes:
            raise ValueError("bar_minutes must divide one day (1440 minutes)")
        if self.pivot_count < 4:
            raise ValueError("pivot_count must be >= 4")
        need = max(self.trend_long_bars, max(DIR_MOVE_LAGS), max(DIR_VWAP_WINDOWS))
        if self.slow_vol_span < need:
            raise ValueError(f"slow_vol_span must be >= {need}")
        if not 0 < self.min_barrier <= self.max_barrier < 1:
            raise ValueError("need 0 < min_barrier <= max_barrier < 1")
        if self.ext_mult <= 1.0:
            raise ValueError("ext_mult must be > 1")
        if not 0 < self.val_es_frac < 1:
            raise ValueError("val_es_frac must be in (0, 1)")
        if not self.eval_years or tuple(sorted(set(self.eval_years))) != tuple(self.eval_years):
            raise ValueError("eval_years must be non-empty, sorted and unique")
        if not self.touch_groups or any(g not in TOUCH_GROUPS for g in self.touch_groups):
            raise ValueError(f"touch_groups must be a non-empty subset of {TOUCH_GROUPS}")
        if not self.dir_groups or any(g not in DIR_GROUPS for g in self.dir_groups):
            raise ValueError(f"dir_groups must be a non-empty subset of {DIR_GROUPS}")
        if not self.dir_zz_scales or any(s <= 0 for s in self.dir_zz_scales):
            raise ValueError("dir_zz_scales must be positive")

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
        """Longest backward span of candles any feature touches (a gap inside it drops the sample)."""
        return max(self.lookback_bars, max(PAST_WINDOWS) + self.barrier_horizon, max(DIR_MOVE_LAGS),
                   max(DIR_VWAP_WINDOWS), max(DIR_RANGE_WINDOWS), max(DIR_ER_WINDOWS))


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
# ZigZag kernels (causal)
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


@njit(cache=True)
def _zigzag_kernel(lh, ll, thr):
    """Per-bar ZigZag state on LOG high/low with per-bar threshold (mirror-symmetric).

    Columns: 0 state 1 piv 2 piv_i 3 dev 4 dev_i 5 h1 6 h2 7 l1 8 l2 9 up_leg 10 dn_leg
    """
    n = len(lh)
    out = np.full((n, 11), np.nan)
    state = 0
    run_hi, hi_i, run_lo, lo_i = lh[0], 0, ll[0], 0
    piv_p, piv_i = np.nan, -1
    h1, h2, l1, l2 = np.nan, np.nan, np.nan, np.nan
    up_leg, dn_leg = np.nan, np.nan
    for i in range(1, n):
        hi, lo, th = lh[i], ll[i], thr[i]
        if state == 0:
            if hi > run_hi:
                run_hi, hi_i = hi, i
            if lo < run_lo:
                run_lo, lo_i = lo, i
            if hi - run_lo >= th and lo_i < i:
                l2, l1 = l1, run_lo
                piv_p, piv_i = run_lo, lo_i
                state, run_hi, hi_i = 1, hi, i
            elif run_hi - lo >= th and hi_i < i:
                h2, h1 = h1, run_hi
                piv_p, piv_i = run_hi, hi_i
                state, run_lo, lo_i = -1, lo, i
        elif state == 1:
            if hi >= run_hi:
                run_hi, hi_i = hi, i
            elif run_hi - lo >= th:
                up_leg = run_hi - piv_p
                h2, h1 = h1, run_hi
                piv_p, piv_i = run_hi, hi_i
                state, run_lo, lo_i = -1, lo, i
        else:
            if lo <= run_lo:
                run_lo, lo_i = lo, i
            elif hi - run_lo >= th:
                dn_leg = piv_p - run_lo
                l2, l1 = l1, run_lo
                piv_p, piv_i = run_lo, lo_i
                state, run_hi, hi_i = 1, hi, i
        out[i, 0] = state
        out[i, 1] = piv_p
        out[i, 2] = piv_i if piv_i >= 0 else np.nan
        if state == 1:
            out[i, 3], out[i, 4] = run_hi, hi_i
        elif state == -1:
            out[i, 3], out[i, 4] = run_lo, lo_i
        out[i, 5], out[i, 6], out[i, 7], out[i, 8] = h1, h2, l1, l2
        out[i, 9], out[i, 10] = up_leg, dn_leg
    return out


_ZZ_COLS = ("state", "piv", "piv_i", "dev", "dev_i", "h1", "h2", "l1", "l2", "up_leg", "dn_leg")


def zigzag_state_at(lh, ll, thr, ts) -> dict:
    full = _zigzag_kernel(lh, ll, thr)[ts]
    return {name: full[:, j] for j, name in enumerate(_ZZ_COLS)}


# --------------------------------------------------------------------------- #
# Labels
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


def _first_hit(mask: np.ndarray, horizon: int) -> np.ndarray:
    return np.where(mask.any(axis=1), mask.argmax(axis=1), horizon)


def make_labels(open_, close, high, low, ts, barrier, horizon, ext_mult, chunk: int | None = None) -> dict:
    """Outcomes from open[t+1] over candles t+1..t+H. A candle touching both levels is a stop (conservative).

    y_touch       1 if +b or -b is reached
    y_up          1 up first / 0 down first / -1 no touch or both in the same candle
    y_ext_long    for y_up == 1: +m*b reached before -b (else -1)
    y_ext_short   for y_up == 0: -m*b reached before +b (else -1)
    r_end         close[t+H] / open[t+1] - 1 in barrier units
    r_<template>  realised R of the four trade templates (TP/SL, time stop at close[t+H])
    t_first       bars until the first barrier hit (H+1 if none)
    """
    m = len(ts)
    chunk = chunk or max(1_000, 2_400_000 // horizon)   # bounded memory for long horizons
    keys_i = ("y_touch", "y_up", "y_ext_long", "y_ext_short", "t_first")
    keys_f = ("r_end",) + tuple(f"r_{k}" for k in TRADE_TEMPLATES)
    out = {k: np.empty(m, dtype=np.int64) for k in keys_i}
    out.update({k: np.empty(m, dtype=np.float64) for k in keys_f})
    steps = np.arange(horizon)[None, :]
    for s0 in range(0, m, chunk):
        sl = slice(s0, min(m, s0 + chunk))
        t = ts[sl]
        bvec = barrier[sl]
        b = bvec[:, None]
        idx = t[:, None] + 1 + steps
        c0 = open_[t + 1][:, None]
        hi, lo = high[idx], low[idx]
        fu = _first_hit(hi >= c0 * (1.0 + b), horizon)
        fd = _first_hit(lo <= c0 * (1.0 - b), horizon)
        feu = _first_hit(hi >= c0 * (1.0 + ext_mult * b), horizon)
        fed = _first_hit(lo <= c0 * (1.0 - ext_mult * b), horizon)
        first = np.minimum(fu, fd)
        touched = first < horizon
        y_up = np.where(fu < fd, 1, np.where(fd < fu, 0, -1))
        end_r = (close[t + horizon] / open_[t + 1] - 1.0) / bvec

        out["y_touch"][sl] = touched
        out["y_up"][sl] = y_up
        out["y_ext_long"][sl] = np.where(y_up == 1, (feu < fd).astype(np.int64), -1)
        out["y_ext_short"][sl] = np.where(y_up == 0, (fed < fu).astype(np.int64), -1)
        out["t_first"][sl] = np.where(touched, first + 1, horizon + 1)
        out["r_end"][sl] = end_r
        out["r_long_1r"][sl] = np.where((fd <= fu) & (fd < horizon), -1.0,
                                        np.where(fu < horizon, 1.0, np.clip(end_r, -1.0, 1.0)))
        out["r_short_1r"][sl] = np.where((fu <= fd) & (fu < horizon), -1.0,
                                         np.where(fd < horizon, 1.0, np.clip(-end_r, -1.0, 1.0)))
        out["r_long_2r"][sl] = np.where((fd <= feu) & (fd < horizon), -1.0,
                                        np.where(feu < horizon, ext_mult, np.clip(end_r, -1.0, ext_mult)))
        out["r_short_2r"][sl] = np.where((fu <= fed) & (fu < horizon), -1.0,
                                         np.where(fed < horizon, ext_mult, np.clip(-end_r, -1.0, ext_mult)))
    return out


# --------------------------------------------------------------------------- #
# Per-bar series and path statistics
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
# TOUCH features (non-directional; same design as v1)
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
# DIRECTION features (origin close_t, mirror-aware)
# --------------------------------------------------------------------------- #
def _prev_period_levels(df: pd.DataFrame, key: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per bar: previous completed period's high / low and the current period's open (all causal)."""
    agg = pd.DataFrame({"h": df["high"].to_numpy(), "l": df["low"].to_numpy(), "o": df["open"].to_numpy(),
                        "k": key.to_numpy()}).groupby("k").agg(h=("h", "max"), l=("l", "min"), o=("o", "first"))
    prev = agg.shift(1)
    k = key.to_numpy()
    return (np.log(prev["h"].reindex(k).to_numpy(np.float64)), np.log(prev["l"].reindex(k).to_numpy(np.float64)),
            np.log(agg["o"].reindex(k).to_numpy(np.float64)))


def direction_features(df, bf, bar_barrier, ts, cfg: Config):
    """Registry entries are (name, group, parity): "odd" (sign flips under the mirror),
    "even" (unchanged) or ("swap", partner, sign)."""
    H = cfg.barrier_horizon
    o = np.log(df["open"].to_numpy(np.float64))
    lh = np.log(df["high"].to_numpy(np.float64))
    ll = np.log(df["low"].to_numpy(np.float64))
    v = df["volume"].to_numpy(np.float64)
    logc, sigma = bf["logc"], bf["sigma"]
    lc = logc[ts]
    sig_t = sigma[ts]
    s = sig_t * math.sqrt(H)
    b = bar_barrier[ts]
    groups = set(cfg.dir_groups)
    feats: dict[str, np.ndarray] = {}
    reg: list[tuple[str, str, object]] = []

    def add(name, arr, group, parity="odd"):
        arr = np.asarray(arr, dtype=np.float64)
        feats[name] = np.where(np.isfinite(arr), arr, np.nan).astype(np.float32)
        reg.append((name, group, parity))

    def swap_pair(a, xa, c, xc, group, sign):
        add(a, xa, group, ("swap", c, sign))
        add(c, xc, group, ("swap", a, sign))

    def rsum(x, k):
        return _roll(x, k, "sum")

    # ---- move
    for k in DIR_MOVE_LAGS:
        add(f"mv_{k}", (lc - logc[ts - k]) / (sig_t * math.sqrt(k)), "move")
    for k in DIR_ER_WINDOWS:
        plen = rsum(bf["absd"], k)[ts]
        add(f"er_{k}", np.abs(lc - logc[ts - k]) / np.maximum(plen, 1e-12), "move", "even")

    # ---- range (barrier units)
    for k in DIR_RANGE_WINDOWS:
        up = (_roll(lh, k, "max")[ts] - lc) / b
        dn = (lc - _roll(ll, k, "min")[ts]) / b
        swap_pair(f"rg{k}_up", up, f"rg{k}_dn", dn, "range", 1.0)
        add(f"rg{k}_clv", (dn - up) / np.maximum(up + dn, 1e-12), "range")
        add(f"rg{k}_width", up + dn, "range", "even")

    # ---- flow
    if "taker_buy_volume" in df.columns:
        net = 2.0 * df["taker_buy_volume"].to_numpy(np.float64) - v
        base = _roll(net, FLOW_BASELINE, "sum", 288) / np.maximum(_roll(v, FLOW_BASELINE, "sum", 288), 1e-12)
        for k in DIR_FLOW_WINDOWS:
            imb = rsum(net, k) / np.maximum(rsum(v, k), 1e-12)
            add(f"tk_{k}", (imb - base)[ts], "flow")
    sv = v * np.sign(bf["r"])
    for k in DIR_VUD_WINDOWS:
        add(f"vud_{k}", (rsum(sv, k) / np.maximum(rsum(v, k), 1e-12))[ts], "flow")

    # ---- candle
    body = (logc - o) / sigma
    rng = np.maximum(lh - ll, 1e-12)
    wick = ((np.minimum(o, logc) - ll) - (lh - np.maximum(o, logc))) / rng
    for k in DIR_CANDLE_WINDOWS:
        add(f"body_{k}", _roll(body, k, "mean")[ts], "candle")
        add(f"wick_{k}", _roll(wick, k, "mean")[ts], "candle")

    # ---- vwap
    tv = ((lh + ll + logc) / 3.0) * v
    for k in DIR_VWAP_WINDOWS:
        sv_k = rsum(v, k)
        vw = rsum(tv, k) / np.where(sv_k > 0, sv_k, np.nan)
        add(f"vwap_{k}", (lc - vw[ts]) / s, "vwap")

    # ---- structure: volatility-adaptive ZigZag at several multiples of the barrier
    if "structure" in groups:
        base_thr = np.where(np.isfinite(bar_barrier), bar_barrier, cfg.min_barrier)
        for scale in cfg.dir_zz_scales:
            pfx = f"zz{int(round(scale * 10)):02d}"
            st = zigzag_state_at(lh, ll, scale * base_thr, ts)
            add(f"{pfx}_state", st["state"], "structure")
            add(f"{pfx}_leg", (lc - st["piv"]) / b, "structure")
            add(f"{pfx}_pb", (lc - st["dev"]) / b, "structure")
            add(f"{pfx}_age", (ts - st["piv_i"]) / H, "structure", "even")
            add(f"{pfx}_devage", (ts - st["dev_i"]) / H, "structure", "even")
            swap_pair(f"{pfx}_hh", (st["h1"] - st["h2"]) / b, f"{pfx}_hl", (st["l1"] - st["l2"]) / b,
                      "structure", -1.0)
            add(f"{pfx}_struct", np.sign(st["h1"] - st["h2"]) + np.sign(st["l1"] - st["l2"]), "structure")
            swap_pair(f"{pfx}_res", (st["h1"] - lc) / b, f"{pfx}_sup", (lc - st["l1"]) / b, "structure", 1.0)
            add(f"{pfx}_legasym", np.log(np.maximum(st["up_leg"], 1e-9) / np.maximum(st["dn_leg"], 1e-9)),
                "structure")

    # ---- anchors: previous UTC day / week high-low, distance to current day / week open
    if "anchors" in groups:
        day = df["timestamp"].dt.floor("D")
        week = day - pd.to_timedelta(df["timestamp"].dt.dayofweek, unit="D")
        for pfx, key in (("d1", day), ("w1", week)):
            ph, pl, op = _prev_period_levels(df, key)
            ph, pl, op = ph[ts], pl[ts], op[ts]
            res, sup = (ph - lc) / b, (lc - pl) / b
            swap_pair(f"{pfx}_res", res, f"{pfx}_sup", sup, "anchors", 1.0)
            add(f"{pfx}_pos", (sup - res) / np.maximum(res + sup, 1e-12), "anchors")
            add(f"{pfx}_open", (lc - op) / s, "anchors")

    # ---- funding: crowd positioning. Positive funding = longs pay shorts; in the mirrored chart longs and
    #      shorts swap, so every signed funding feature is odd and its magnitude is even.
    if "funding_rate" in df.columns:
        fr = df["funding_rate"].to_numpy(np.float64)[ts] * 1e4                      # basis points
        add("fund_rate", fr, "funding")
        add("fund_mean_3", df["funding_mean_3"].to_numpy(np.float64)[ts] * 1e4, "funding")
        add("fund_z_90", df["funding_z_90"].to_numpy(np.float64)[ts], "funding")
        add("fund_dev_90", df["funding_dev_90"].to_numpy(np.float64)[ts] * 1e4, "funding")
        add("fund_abs", np.abs(fr), "funding", "even")
        add("fund_x_mv288", fr * (lc - logc[ts - 288]) / (sig_t * math.sqrt(288)), "funding", "even")

    # ---- regime: even conditioners only
    add("rg_sigma_ratio", bf["sigma_ratio"][ts], "regime", "even")
    add("rg_barrier_ratio", b / s, "regime", "even")
    add("rg_rvol_6", _roll(bf["rvol"], 6, "mean")[ts], "regime", "even")
    for name in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        add(f"rg_{name}", bf[name][ts], "regime", "even")

    reg = [r for r in reg if r[1] in groups]
    return {r[0]: feats[r[0]] for r in reg}, reg


def mirror_matrix(x: np.ndarray, names: tuple[str, ...], parity: dict) -> np.ndarray:
    """Features of the mirrored chart (log price -> -log price, high <-> low, taker buy <-> sell)."""
    idx = {n: i for i, n in enumerate(names)}
    m = x.copy()
    for n, i in idx.items():
        p = parity[n]
        if p == "odd":
            m[:, i] = -x[:, i]
        elif p != "even":
            _, partner, sign = p
            m[:, i] = sign * x[:, idx[partner]]
    return m


def check_parity(names: tuple[str, ...], parity: dict) -> None:
    present = set(names)
    for n in names:
        p = parity[n]
        if p in ("odd", "even"):
            continue
        _, partner, sign = p
        back = parity.get(partner)
        if partner not in present or not isinstance(back, tuple) or back[1] != n or back[2] != sign:
            raise ValueError(f"inconsistent mirror parity for {n} <-> {partner}")


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
    x_touch: np.ndarray
    x_dir: np.ndarray
    touch_names: tuple[str, ...]
    dir_names: tuple[str, ...]
    dir_parity: dict
    feature_groups: dict
    labels: dict | None            # None in live mode

    @property
    def x_dir_mirror(self) -> np.ndarray:
        if not hasattr(self, "_xm"):
            object.__setattr__(self, "_xm", mirror_matrix(self.x_dir, self.dir_names, self.dir_parity))
        return self._xm


def build_dataset(df: pd.DataFrame, pivots: pd.DataFrame, cfg: Config,
                  live: bool = False, live_last: int | None = None) -> Dataset:
    """Causal features at the close of each decision bar (+ labels unless live).

    live=True keeps the most recent bars (no label horizon needed); live_last keeps only the last N.
    """
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
    tf, treg = touch_features(df, pivots, bf, pre, bar_barrier, touch_s, ts, known_end, cfg)
    df_feats, dreg = direction_features(df, bf, bar_barrier, ts, cfg)
    touch_names = tuple(nm for nm, _ in treg)
    dir_names = tuple(nm for nm, _, _ in dreg)
    if not touch_names or not dir_names:
        raise ValueError("feature group selection leaves a model without features")
    dir_parity = {nm: p for nm, _, p in dreg}
    check_parity(dir_names, dir_parity)
    groups_map = {nm: g for nm, g in treg}
    groups_map.update({nm: f"dir:{g}" for nm, g, _ in dreg})

    labels, end_ns = None, None
    if not live:
        labels = make_labels(open_, close, high, low, ts, bar_barrier[ts], H, cfg.ext_mult)
        end_ns = ns[ts + H]
    return Dataset(
        decision_idx=ts, timestamp_ns=ns[ts], label_end_ns=end_ns, close=close[ts], barrier_pct=bar_barrier[ts],
        x_touch=np.column_stack([tf[nm] for nm in touch_names]).astype(np.float32),
        x_dir=np.column_stack([df_feats[nm] for nm in dir_names]).astype(np.float32),
        touch_names=touch_names, dir_names=dir_names, dir_parity=dir_parity, feature_groups=groups_map,
        labels=labels,
    )


# --------------------------------------------------------------------------- #
# Metric helpers
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


def spearman(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


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
# XGBoost models
# --------------------------------------------------------------------------- #
def make_xgb(p: XGBParams, kind: str, cfg: Config, seed: int):
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as exc:
        raise ImportError("Install XGBoost first: pip install xgboost") from exc
    params = dict(n_estimators=p.n_estimators, learning_rate=p.learning_rate, max_depth=p.max_depth,
                  min_child_weight=p.min_child_weight, subsample=p.subsample, colsample_bytree=p.colsample_bytree,
                  reg_alpha=p.reg_alpha, reg_lambda=p.reg_lambda, gamma=p.gamma,
                  early_stopping_rounds=p.early_stopping_rounds, tree_method="hist", n_jobs=cfg.xgb_n_jobs,
                  random_state=seed)
    if cfg.xgb_device != "cpu":
        params["device"] = cfg.xgb_device
    if kind == "binary":
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **params)
    return XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **params)


def _fit(model, x_tr, y_tr, x_va, y_va, w_tr, cfg: Config) -> tuple[object, int]:
    model.fit(x_tr, y_tr, sample_weight=w_tr, eval_set=[(x_va, y_va)], verbose=False)
    best = int(getattr(model, "best_iteration", model.n_estimators - 1)) + 1
    if cfg.xgb_device != "cpu":
        model.set_params(device="cpu")
        model.get_booster().set_param({"device": "cpu"})
    return model, best


def _p1(mb, x) -> np.ndarray:
    model, best = mb
    return np.asarray(model.predict_proba(x, iteration_range=(0, best))[:, 1], dtype=np.float64)


def _reg(mb, x) -> np.ndarray:
    model, best = mb
    return np.asarray(model.predict(x, iteration_range=(0, best)), dtype=np.float64)


def _time_weights(ds: Dataset, ids: np.ndarray, ref_ns: int, cfg: Config) -> np.ndarray | None:
    if cfg.time_decay_halflife_days <= 0:
        return None
    age_days = np.maximum(ref_ns - ds.timestamp_ns[ids], 0) / 86_400e9
    return np.maximum(0.5 ** (age_days / cfg.time_decay_halflife_days), cfg.time_decay_floor).astype(np.float32)


# design matrices for the mirror-augmented models ----------------------------
def _design_dir(ds: Dataset, ids: np.ndarray):
    """Touched, non-tie samples + mirror images. Target: up first (mirror: 1 - y)."""
    ids = ids[ds.labels["y_up"][ids] >= 0]
    y = ds.labels["y_up"][ids]
    return ids, np.vstack([ds.x_dir[ids], ds.x_dir_mirror[ids]]), np.concatenate([y, 1 - y])


def _design_ext(ds: Dataset, ids: np.ndarray):
    """Long orientation only: up-first rows as is, down-first rows mirrored. Target: extension reached."""
    up = ids[ds.labels["y_up"][ids] == 1]
    dn = ids[ds.labels["y_up"][ids] == 0]
    x = np.vstack([ds.x_dir[up], ds.x_dir_mirror[dn]])
    y = np.concatenate([ds.labels["y_ext_long"][up], ds.labels["y_ext_short"][dn]])
    return np.concatenate([up, dn]), x, y


def _design_timeout(ds: Dataset, ids: np.ndarray):
    """No-touch samples + mirror images. Target: close-out return in barrier units (mirror: negated)."""
    ids = ids[ds.labels["y_touch"][ids] == 0]
    r = ds.labels["r_end"][ids]
    return ids, np.vstack([ds.x_dir[ids], ds.x_dir_mirror[ids]]), np.concatenate([r, -r])


def _dup(w):
    return None if w is None else np.concatenate([w, w])


def fit_models(ds: Dataset, train_ids: np.ndarray, val_ids: np.ndarray, cfg: Config) -> dict:
    """Early stopping on the first part of the validation year; temperatures on the rest."""
    L = ds.labels
    es_ids, cal_ids = split_validation(ds, val_ids, cfg.val_es_frac)
    ref_ns = int(ds.timestamp_ns[train_ids].max())

    # 1. touch
    w = _time_weights(ds, train_ids, ref_ns, cfg)
    touch = _fit(make_xgb(cfg.touch, "binary", cfg, cfg.seed), ds.x_touch[train_ids], L["y_touch"][train_ids],
                 ds.x_touch[es_ids], L["y_touch"][es_ids], w, cfg)

    tr = train_ids[:: cfg.dir_train_stride]
    # 2. direction | touch
    ids_d, x_tr, y_tr = _design_dir(ds, tr)
    _, x_va, y_va = _design_dir(ds, es_ids)
    direction = _fit(make_xgb(cfg.dir, "binary", cfg, cfg.seed + 1), x_tr, y_tr, x_va, y_va,
                     _dup(_time_weights(ds, ids_d, ref_ns, cfg)), cfg)
    # 3. extension | first side
    ids_e, x_tr, y_tr = _design_ext(ds, tr)
    _, x_va, y_va = _design_ext(ds, es_ids)
    ext = _fit(make_xgb(cfg.ext, "binary", cfg, cfg.seed + 2), x_tr, y_tr, x_va, y_va,
               _time_weights(ds, ids_e, ref_ns, cfg), cfg)
    # 4. timeout drift | no touch
    ids_t, x_tr, y_tr = _design_timeout(ds, tr)
    _, x_va, y_va = _design_timeout(ds, es_ids)
    timeout = _fit(make_xgb(cfg.timeout, "reg", cfg, cfg.seed + 3), x_tr, y_tr, x_va, y_va,
                   _dup(_time_weights(ds, ids_t, ref_ns, cfg)), cfg)

    models = {"touch": touch, "dir": direction, "ext": ext, "timeout": timeout,
              "T_touch": 1.0, "T_dir": 1.0, "T_ext": 1.0}
    # ---- empirical constants (train, mirror-pooled)
    touched = train_ids[L["y_touch"][train_ids] == 1]
    up = train_ids[(L["y_up"][train_ids] == 1) & (L["y_ext_long"][train_ids] == 0)]
    dn = train_ids[(L["y_up"][train_ids] == 0) & (L["y_ext_short"][train_ids] == 0)]
    nonext = np.concatenate([L["r_long_2r"][up], L["r_short_2r"][dn]])
    models["tie_rate"] = float((L["y_up"][touched] == -1).mean()) if len(touched) else 0.0
    models["nonext_R"] = float(nonext.mean()) if len(nonext) else -1.0
    models["ext_mult"] = cfg.ext_mult

    # ---- temperatures on the calibration part
    if cfg.calibrate:
        models["T_touch"] = fit_temperature(logit(_p1(touch, ds.x_touch[cal_ids])), L["y_touch"][cal_ids])
        cal_d = cal_ids[L["y_up"][cal_ids] >= 0]
        raw = _raw_dir(models, ds.x_dir[cal_d], ds.x_dir_mirror[cal_d])
        models["T_dir"] = fit_temperature(logit(raw), L["y_up"][cal_d])
        _, x_c, y_c = _design_ext(ds, cal_ids)
        models["T_ext"] = fit_temperature(logit(_p1(ext, x_c)), y_c)

    return {
        "models": models, "calib_ids": cal_ids,
        "best_rounds": {"touch": touch[1], "dir": direction[1], "ext": ext[1], "timeout": timeout[1]},
        "samples": {"train": int(len(train_ids)), "val_early_stop": int(len(es_ids)),
                    "val_calibration": int(len(cal_ids)), "dir_train_rows": int(2 * len(ids_d)),
                    "ext_train_rows": int(len(ids_e)), "timeout_train_rows": int(2 * len(ids_t))},
    }


def _raw_dir(models: dict, x: np.ndarray, xm: np.ndarray) -> np.ndarray:
    """Mirror-symmetrised P(up | touch) before calibration."""
    return 0.5 * (_p1(models["dir"], x) + 1.0 - _p1(models["dir"], xm))


def signals_from_features(models: dict, x_touch: np.ndarray, x_dir: np.ndarray, x_dir_mirror: np.ndarray,
                          barrier_pct: np.ndarray) -> dict:
    """All trade-design signals for a block of decision bars (see module docstring)."""
    m = models["ext_mult"]
    p_touch = sigmoid(logit(_p1(models["touch"], x_touch)) / max(models["T_touch"], 1e-4))
    p_up = sigmoid(logit(_raw_dir(models, x_dir, x_dir_mirror)) / max(models["T_dir"], 1e-4))
    t_ext = max(models["T_ext"], 1e-4)
    p_ext_long = sigmoid(logit(_p1(models["ext"], x_dir)) / t_ext)
    p_ext_short = sigmoid(logit(_p1(models["ext"], x_dir_mirror)) / t_ext)
    mu = np.clip(0.5 * (_reg(models["timeout"], x_dir) - _reg(models["timeout"], x_dir_mirror)), -1.0, 1.0)

    tie = models["tie_rate"]
    p_tie = p_touch * tie
    p_up_first = p_touch * (1.0 - tie) * p_up
    p_dn_first = p_touch * (1.0 - tie) * (1.0 - p_up)
    p_none = 1.0 - p_touch
    q = models["nonext_R"]
    out = {
        "barrier_pct": np.asarray(barrier_pct, dtype=np.float64),
        "p_touch": p_touch, "p_up_touch": p_up,
        "p_up_first": p_up_first, "p_dn_first": p_dn_first, "p_tie": p_tie, "p_none": p_none,
        "p_ext_long": p_ext_long, "p_ext_short": p_ext_short, "mu_timeout": mu,
        "e_long_1r": p_up_first - p_dn_first - p_tie + p_none * mu,
        "e_short_1r": p_dn_first - p_up_first - p_tie - p_none * mu,
        "e_long_2r": p_up_first * (m * p_ext_long + (1.0 - p_ext_long) * q) - p_dn_first - p_tie + p_none * mu,
        "e_short_2r": p_dn_first * (m * p_ext_short + (1.0 - p_ext_short) * q) - p_up_first - p_tie - p_none * mu,
    }
    e = np.column_stack([out[f"e_{k}"] for k in TRADE_TEMPLATES])
    k = e.argmax(axis=1)
    out["best_side"] = np.where(np.isin(k, (0, 2)), 1, -1)
    out["best_rr"] = np.where(k >= 2, m, 1.0)
    out["best_e"] = e[np.arange(len(k)), k]
    return out


def predict_signals(models: dict, ds: Dataset, ids: np.ndarray) -> dict:
    out = signals_from_features(models, ds.x_touch[ids], ds.x_dir[ids], ds.x_dir_mirror[ids], ds.barrier_pct[ids])
    out["ids"] = ids
    return out


# --------------------------------------------------------------------------- #
# Evaluation (signal quality, no order simulation)
# --------------------------------------------------------------------------- #
def choose(sig: dict, labels: dict, ids: np.ndarray, rr: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Side (+1/-1), predicted edge and realised R of the best template inside an RR set."""
    names = RR_SETS[rr]
    e = np.column_stack([sig[f"e_{k}"] for k in names])
    r = np.column_stack([labels[f"r_{k}"][ids] for k in names])
    k = e.argmax(axis=1)
    row = np.arange(len(k))
    side = np.array([1 if n.startswith("long") else -1 for n in names])[k]
    return side, e[row, k], r[row, k]


def non_overlapping(decision_idx: np.ndarray, horizon: int) -> np.ndarray:
    """Greedy chronological selection so that consecutive signals' holding windows do not overlap."""
    keep, last = [], -10**18
    for i, t in enumerate(decision_idx):
        if t >= last + horizon:
            keep.append(i)
            last = t
    return np.asarray(keep, dtype=np.int64)


def selection_table(edge, realised, fee_r, decision_idx, side, horizon, fractions, thresholds=None) -> list[dict]:
    rows = []
    for q in fractions:
        thr = thresholds[q] if thresholds is not None else float(np.quantile(edge, 1.0 - q))
        sel = edge >= thr
        if not sel.any():
            rows.append({"top_fraction": q, "threshold": thr, "samples": 0})
            continue
        net = realised[sel] - fee_r[sel]
        ind = non_overlapping(decision_idx[sel], horizon)
        rows.append({
            "top_fraction": q, "threshold": float(thr), "selected_share": float(sel.mean()), "samples": int(sel.sum()),
            "mean_pred_R": float(edge[sel].mean()), "mean_R": float(realised[sel].mean()), "mean_net_R": float(net.mean()),
            "hit_rate": float((realised[sel] > 0).mean()), "long_share": float((side[sel] == 1).mean()),
            "independent_signals": int(len(ind)), "independent_mean_net_R": float(net[ind].mean()),
            "independent_total_net_R": float(net[ind].sum()),
        })
    return rows


def evaluate_signals(ds: Dataset, sig: dict, train_ids: np.ndarray, cfg: Config, thresholds: dict | None = None) -> dict:
    ids = sig["ids"]
    L = {k: v[ids] for k, v in ds.labels.items()}
    day = ds.timestamp_ns[ids] // 86_400_000_000_000
    nb, seed = cfg.n_boot, cfg.seed
    out: dict = {"samples": int(len(ids))}

    # ---- touch
    y_t = L["y_touch"]
    out["touch"] = binary_report(y_t, sig["p_touch"], float(ds.labels["y_touch"][train_ids].mean()))
    out["touch"]["auc_ci"] = block_bootstrap_ci(safe_auc, day, y_t, sig["p_touch"], n_boot=nb, seed=seed)
    out["touch"]["reliability"] = reliability_bins(sig["p_touch"], y_t, 5)

    # ---- direction | touch
    md = L["y_up"] >= 0
    y_d, p_d = L["y_up"][md], sig["p_up_touch"][md]
    out["direction"] = binary_report(y_d, p_d, 0.5)
    out["direction"]["auc_ci"] = block_bootstrap_ci(safe_auc, day[md], y_d, p_d, n_boot=nb, seed=seed)
    out["direction"]["reliability"] = reliability_bins(p_d, y_d, 10)
    hi = md & (sig["p_touch"] >= np.quantile(sig["p_touch"], 0.6))
    out["direction"]["auc_top40_p_touch"] = safe_auc(L["y_up"][hi], sig["p_up_touch"][hi])

    # ---- extension | first side (long orientation pooled)
    up, dn = L["y_up"] == 1, L["y_up"] == 0
    y_e = np.concatenate([L["y_ext_long"][up], L["y_ext_short"][dn]])
    p_e = np.concatenate([sig["p_ext_long"][up], sig["p_ext_short"][dn]])
    d_e = np.concatenate([day[up], day[dn]])
    tr = ds.labels
    prior_e = np.concatenate([tr["y_ext_long"][train_ids][tr["y_up"][train_ids] == 1],
                              tr["y_ext_short"][train_ids][tr["y_up"][train_ids] == 0]])
    out["extension"] = binary_report(y_e, p_e, float(prior_e.mean()) if len(prior_e) else 0.5)
    out["extension"]["auc_ci"] = block_bootstrap_ci(safe_auc, d_e, y_e, p_e, n_boot=nb, seed=seed)

    # ---- timeout drift | no touch
    mt = L["y_touch"] == 0
    out["timeout"] = {"samples": int(mt.sum()), "ic": spearman(sig["mu_timeout"][mt], L["r_end"][mt]),
                      "ic_ci": block_bootstrap_ci(spearman, day[mt], sig["mu_timeout"][mt], L["r_end"][mt],
                                                  n_boot=nb, seed=seed),
                      "mean_pred": float(sig["mu_timeout"][mt].mean()) if mt.any() else float("nan"),
                      "mean_realised": float(L["r_end"][mt].mean()) if mt.any() else float("nan")}

    # ---- composite expected R
    fee_r = cfg.fee_pct_roundtrip / ds.barrier_pct[ids]
    out["mean_fee_R"] = float(fee_r.mean())
    comp = {}
    for rr in ("1r", "2r"):
        a, b = RR_SETS[rr]
        spread_p = sig[f"e_{a}"] - sig[f"e_{b}"]
        spread_r = L[f"r_{a}"] - L[f"r_{b}"]
        comp[rr] = {"ic_spread": spearman(spread_p, spread_r),
                    "ic_spread_ci": block_bootstrap_ci(spearman, day, spread_p, spread_r, n_boot=nb, seed=seed),
                    "calibration": {k: reliability_bins(sig[f"e_{k}"], L[f"r_{k}"], 10) for k in (a, b)}}
    for rr in ("1r", "2r", "best"):
        side, edge, real = choose(sig, ds.labels, ids, rr)
        comp.setdefault(rr, {})
        comp[rr].update({
            "ic_edge": spearman(edge, real),
            "ic_edge_ci": block_bootstrap_ci(spearman, day, edge, real, n_boot=nb, seed=seed),
            "edge_calibration": reliability_bins(edge, real, 10),
            "top": selection_table(edge, real, fee_r, ds.decision_idx[ids], side, cfg.barrier_horizon,
                                   cfg.top_fractions),
        })
        if thresholds is not None:
            comp[rr]["top_cal_threshold"] = selection_table(
                edge, real, fee_r, ds.decision_idx[ids], side, cfg.barrier_horizon, cfg.top_fractions,
                thresholds[rr])
    out["expected_R"] = comp
    return out


def edge_thresholds(sig: dict, ds: Dataset, cfg: Config) -> dict:
    """Edge thresholds per RR set and top fraction, measured on a (calibration) prediction block."""
    out = {}
    for rr in ("1r", "2r", "best"):
        _, edge, _ = choose(sig, ds.labels, sig["ids"], rr)
        out[rr] = {q: float(np.quantile(edge, 1.0 - q)) for q in cfg.top_fractions}
    return out


def monthly_report(ds: Dataset, sig: dict) -> list[dict]:
    ids = sig["ids"]
    L = {k: v[ids] for k, v in ds.labels.items()}
    month = pd.DatetimeIndex(pd.to_datetime(ds.timestamp_ns[ids], utc=True)).tz_localize(None).to_period("M").astype(str)
    month = np.asarray(month)
    rows = []
    for mth in np.unique(month):
        k = month == mth
        md = k & (L["y_up"] >= 0)
        rows.append({"month": str(mth), "samples": int(k.sum()),
                     "touch_auc": safe_auc(L["y_touch"][k], sig["p_touch"][k]),
                     "dir_auc": safe_auc(L["y_up"][md], sig["p_up_touch"][md]),
                     "ic_spread_1r": spearman(sig["e_long_1r"][k] - sig["e_short_1r"][k],
                                              L["r_long_1r"][k] - L["r_short_1r"][k])})
    return rows


def feature_importance(models: dict, ds: Dataset, top: int = 25) -> dict:
    out = {}
    for key, names in (("touch", ds.touch_names), ("dir", ds.dir_names), ("ext", ds.dir_names),
                       ("timeout", ds.dir_names)):
        imp = np.asarray(models[key][0].feature_importances_, dtype=np.float64)
        order = np.argsort(imp)[::-1][:top]
        groups = pd.Series(imp, index=list(names)).groupby(lambda n: ds.feature_groups[n]).sum()
        out[key] = {"top": [{"feature": names[i], "gain_share": float(imp[i])} for i in order],
                    "groups": {g: float(v) for g, v in groups.sort_values(ascending=False).items()}}
    return out


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


def save_predictions(path: Path, ds: Dataset, sig: dict) -> None:
    ids = sig["ids"]
    np.savez_compressed(
        path, sample_id=ids, timestamp_ns=ds.timestamp_ns[ids], decision_idx=ds.decision_idx[ids],
        close=ds.close[ids], **{f"label_{k}": v[ids] for k, v in ds.labels.items()},
        **{k: v for k, v in sig.items() if k != "ids"})


def evaluate_fold(name: str, fitted: dict, train_ids, test_ids, ds: Dataset, cfg: Config, out: Path | None) -> dict:
    models = fitted["models"]
    cal_sig = predict_signals(models, ds, fitted["calib_ids"])
    test_sig = predict_signals(models, ds, test_ids)
    thr = edge_thresholds(cal_sig, ds, cfg)
    if out is not None:
        save_predictions(out / f"predictions_{name}.npz", ds, test_sig)
    return {
        "fold": name, "samples": fitted["samples"], "best_rounds": fitted["best_rounds"],
        "temperatures": {k: models[k] for k in ("T_touch", "T_dir", "T_ext")},
        "constants": {k: models[k] for k in ("tie_rate", "nonext_R", "ext_mult")},
        "edge_thresholds_from_calibration": thr,
        "validation_metrics": evaluate_signals(ds, cal_sig, train_ids, cfg),
        "heldout_metrics": evaluate_signals(ds, test_sig, train_ids, cfg, thresholds=thr),
        "monthly": monthly_report(ds, test_sig),
        "importance": feature_importance(models, ds),
    }


def summary_line(rep: dict) -> str:
    h = rep["heldout_metrics"]
    t, d, e, er = h["touch"], h["direction"], h["extension"], h["expected_R"]
    top5 = next((r for r in er["best"]["top_cal_threshold"] if r["top_fraction"] == 0.05), {})
    ci = lambda x: f"[{x[0]:+.3f},{x[1]:+.3f}]"  # noqa: E731
    return (f"{rep['fold']:>6} | touch AUC {t['auc']:.3f} | dir AUC {d['auc']:.3f} {ci(d['auc_ci'])} "
            f"| ext AUC {e['auc']:.3f} | IC1r {er['1r']['ic_spread']:+.3f} {ci(er['1r']['ic_spread_ci'])} "
            f"| top5% best: net {top5.get('independent_mean_net_R', float('nan')):+.3f}R "
            f"x{top5.get('independent_signals', 0)} indep")


def export_bundle(path: Path, models: dict, ds: Dataset, cfg: Config) -> None:
    import joblib
    joblib.dump({"strategy": STRATEGY, "config": cfg, "models": models, "touch_names": ds.touch_names,
                 "dir_names": ds.dir_names, "dir_parity": ds.dir_parity}, path)


def run_pipeline(cfg: Config) -> dict:
    """Yearly walk-forward training + signal evaluation; exports the model of the last eval year."""
    cfg = replace(cfg, xgb_device=resolve_device(cfg.xgb_device))
    seed_everything(cfg.seed)
    if cfg.verbose:
        print(f"device={cfg.xgb_device} numba={HAVE_NUMBA}", flush=True)
    df = load_candles(cfg.csv_path, cfg.bar_minutes)
    pivots = find_confirmed_pivots(df, cfg.pivot_reversal)
    ds = build_dataset(df, pivots, cfg)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    reports, fitted = [], None
    for year in cfg.eval_years:
        train_ids, val_ids, test_ids = fold_ids(ds, year)
        fitted = fit_models(ds, train_ids, val_ids, cfg)
        rep = evaluate_fold(str(year), fitted, train_ids, test_ids, ds, cfg, out)
        reports.append(rep)
        if cfg.verbose:
            print(summary_line(rep), flush=True)

    export_bundle(out / "models_final.joblib", fitted["models"], ds, cfg)
    result = {"candles": len(df), "pivots": len(pivots), "samples": len(ds.decision_idx),
              "touch_features": len(ds.touch_names), "direction_features": len(ds.dir_names),
              "has_taker_volume": "taker_buy_volume" in df.columns,
              "folds": reports, "final_fold": reports[-1]["fold"], "output_dir": str(out)}
    (out / "metadata.json").write_text(json.dumps(
        {"strategy": STRATEGY, "config": asdict(cfg), **result,
         "protocol_note": "Yearly walk-forward, one fresh model set per eval year; early stopping on the first part "
                          "of the prior year, temperatures + validation metrics on its last part; boundary labels "
                          "purged. Signal evaluation only (gross / net-of-fee realised R of trade templates); "
                          "no order simulation."},
        indent=2, default=_json_default), encoding="utf-8")
    return result


# --------------------------------------------------------------------------- #
# Live / batch inference from an exported bundle
# --------------------------------------------------------------------------- #
def load_bundle(path: str | Path) -> dict:
    import joblib
    return joblib.load(path)


def predict_frame(bundle: dict, candles: pd.DataFrame, last_n: int | None = 1,
                  funding: pd.DataFrame | None = None) -> pd.DataFrame:
    """Signals for the most recent closed candles of a raw kline frame.

    `candles` must contain enough history (>= ~20 days of 5m bars) and only CLOSED candles.
    `funding`: settled funding rows (BTCUSDT_funding.csv layout or the raw /fapi/v1/fundingRate response,
    >= 30 days) - required when the bundle was trained with funding features.
    Price levels use close_t as the reference (the executable reference is the next open).
    """
    cfg: Config = bundle["config"]
    df = prepare_candles(candles, cfg.bar_minutes)
    if funding is not None:
        df = attach_funding(df, funding, cfg.bar_minutes)
    pivots = find_confirmed_pivots(df, cfg.pivot_reversal)
    ds = build_dataset(df, pivots, cfg, live=True, live_last=last_n)
    if ds.touch_names != tuple(bundle["touch_names"]) or ds.dir_names != tuple(bundle["dir_names"]):
        missing = sorted(set(bundle["touch_names"]) - set(ds.touch_names) | set(bundle["dir_names"]) - set(ds.dir_names))
        raise ValueError("feature set differs from the exported bundle (taker / trades / funding columns?) "
                         f"- missing: {missing[:10]}")
    sig = signals_from_features(bundle["models"], ds.x_touch, ds.x_dir, ds.x_dir_mirror, ds.barrier_pct)
    frame = pd.DataFrame({"timestamp": pd.to_datetime(ds.timestamp_ns, utc=True), "close": ds.close, **sig})
    b, c, m = frame["barrier_pct"], frame["close"], bundle["models"]["ext_mult"]
    frame["long_sl"], frame["long_tp_1r"], frame["long_tp_2r"] = c * (1 - b), c * (1 + b), c * (1 + m * b)
    frame["short_sl"], frame["short_tp_1r"], frame["short_tp_2r"] = c * (1 + b), c * (1 - b), c * (1 - m * b)
    frame["time_stop_bars"] = cfg.barrier_horizon
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the v2 signal models (touch/dir/ext/timeout)")
    d = Config()
    parser.add_argument("--csv", default=d.csv_path)
    parser.add_argument("--output-dir", default=d.output_dir)
    parser.add_argument("--xgb-device", default=d.xgb_device, help="auto | cpu | cuda:0")
    parser.add_argument("--sample-stride", type=int, default=d.sample_stride)
    parser.add_argument("--horizon", type=int, default=d.barrier_horizon)
    parser.add_argument("--years", type=int, nargs="+", default=list(d.eval_years))
    parser.add_argument("--fee-pct", type=float, default=d.fee_pct_roundtrip, help="round-trip fee as a fraction")
    parser.add_argument("--n-boot", type=int, default=d.n_boot)
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--quiet", action="store_true")
    a = parser.parse_args()
    cfg = Config(csv_path=a.csv, output_dir=a.output_dir, xgb_device=a.xgb_device, sample_stride=a.sample_stride,
                 barrier_horizon=a.horizon, eval_years=tuple(sorted(set(a.years))), fee_pct_roundtrip=a.fee_pct,
                 n_boot=a.n_boot, seed=a.seed, verbose=not a.quiet)
    result = run_pipeline(cfg)
    print(json.dumps({"output_dir": result["output_dir"], "folds": [r["fold"] for r in result["folds"]],
                      "final_fold": result["final_fold"]}, indent=2))


if __name__ == "__main__":
    main()
