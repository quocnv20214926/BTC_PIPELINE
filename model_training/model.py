

from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ("SHORT", "HOLD", "LONG")
TOKEN_FEATURES = (
    "age",                # (t - pivot_idx) / 96            how old the pivot is (days)
    "dist_z",             # log(pivot_price / close_t) / (sigma*sqrt(H))   distance in vol units
    "kind",               # +1 high, -1 low
    "swing_abs_z",        # |log(price / previous pivot price)| / (sigma*sqrt(H))
    "swing_bars",         # bars between previous pivot and this pivot / 96
    "confirm_bars",       # bars needed to confirm the reversal / 96 (0 for developing token)
    "reversal_progress",  # developing token only: |log(close/extreme)| / reversal threshold
    "is_developing",      # 1 for the unconfirmed extreme token
)
CONTEXT_FEATURES = (
    "sigma_ratio",   # log(fast vol / slow vol): volatility regime
    "trend_1d_z",    # 1-day log return as a random-walk z-score
    "trend_5d_z",    # 5-day log return as a random-walk z-score
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
)
PIVOT_FEATURES = TOKEN_FEATURES + CONTEXT_FEATURES
CANDLE_BASE_FEATURES = (
    "ret_z",           # log return / previous-bar volatility
    "range_z",         # log(high/low) / previous-bar volatility
    "body_frac",       # (close-open)/(high-low), candle shape in [-1, 1]
    "wick_imbalance",  # (lower wick - upper wick)/(high-low) in [-1, 1]
    "path_vs_close",   # (log close_i - log close_t) / (sigma*sqrt(H)): path in vol units
    "rel_log_volume",  # (log1p(vol) - rolling_median) / rolling_mad (robust scale)
    "sigma_ratio", "trend_1d_z", "trend_5d_z",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
)
TAKER_FEATURE = "taker_imbalance"  # added only when the CSV has taker-buy volume
TAKER_COLUMNS = ("taker_buy_base_volume", "taker_buy_base_asset_volume", "taker_buy_volume", "taker_buy_base")
STANDARDIZE_CLIP = 5.0
SCALE_BARS = 96.0  # one day of 15-minute candles
TREND_SHORT, TREND_LONG = 96, 480


@dataclass
class Config:
    csv_path: str = "historical_data/BTCUSDT_15m_all.csv"
    output_dir: str = "artifacts/15m_dual_models"
    bar_minutes: int = 15
    pivot_reversal: float = 0.005
    barrier_horizon: int = 24
    barrier_vol_mult: float = 1.0
    min_barrier: float = 0.007
    max_barrier: float = 0.03
    vol_span: int = 96
    slow_vol_span: int = 960
    pivot_count: int = 8
    candle_count: int = 48
    fold_years: tuple[int, ...] = (2023, 2024, 2025)
    backtest_year: int = 2026
    batch_size: int = 512
    epochs: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    hidden_size: int = 64
    dropout: float = 0.15
    class_weighted: bool = False   # plain CE keeps probabilities calibrated for the edge rule
    early_stop_fraction: float = 0.10
    sample_stride: int = 1
    seed: int = 42
    num_workers: int = 0
    fee_bps: float = 4.0
    slippage_bps: float = 2.0
    funding_bps_per_8h: float = 0.0  # optional funding cost per 8 hours (set e.g. 1.0 for 0.01%)
    calibrate_temperature: bool = True  # post-hoc temperature scaling on validation set
    ensemble_weight: float = 0.5   # weight of the pivot transformer in the ensemble
    min_edge: float = 0.0
    signal_mode: str = "average"   # "average" or "agree" (both models must favour the same side)
    tune_thresholds: bool = True
    min_edge_grid: tuple[float, ...] = (0.0, 0.02, 0.04, 0.06)
    signal_mode_grid: tuple[str, ...] = ("average", "agree")
    min_tune_trades: int = 30
    run_baselines: bool = True
    baseline_max_samples: int = 50_000
    random_baseline_sims: int = 200
    bootstrap_samples: int = 2000
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.signal_mode not in ("average", "agree"):
            raise ValueError("signal_mode must be 'average' or 'agree'")
        if not 0.0 <= self.ensemble_weight <= 1.0:
            raise ValueError("ensemble_weight must be in [0, 1]")
        if not 0 < self.min_barrier <= self.max_barrier < 1:
            raise ValueError("need 0 < min_barrier <= max_barrier < 1")
        if self.slow_vol_span < TREND_LONG:
            raise ValueError(f"slow_vol_span must be >= {TREND_LONG} (longest trend lookback)")

    @property
    def cost(self) -> float:
        """Round-trip cost as a fraction of price (fees + slippage, both sides)."""
        return 2.0 * (self.fee_bps + self.slippage_bps) / 10_000.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _ns_int(ts) -> np.ndarray:
    """UTC timestamps as int64 nanoseconds, independent of pandas' datetime unit."""
    s = pd.Series(pd.to_datetime(ts, utc=True))
    try:  # pandas >= 2.0; pandas 3 may infer us/ms resolution
        s = s.dt.as_unit("ns")
    except AttributeError:
        pass
    return s.astype("int64").to_numpy()


def gap_flags(df: pd.DataFrame, bar_minutes: int) -> np.ndarray:
    """flags[i] = 1 when candle i is not exactly one bar after candle i-1."""
    ns = _ns_int(df["timestamp"])
    flags = np.zeros(len(ns), dtype=np.int64)
    flags[1:] = np.diff(ns) != bar_minutes * 60 * 10**9
    return flags


def load_candles(path: str | Path, bar_minutes: int = 15) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    if "open_time_utc" in df:
        ts = pd.to_datetime(df["open_time_utc"], utc=True, errors="raise")
    elif "open_time_ms" in df:
        ts = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True, errors="raise")
    else:
        raise ValueError("CSV needs open_time_utc or open_time_ms")
    try:
        ts = ts.dt.as_unit("ns")
    except AttributeError:
        pass
    out = df.loc[:, ["open", "high", "low", "close", "volume"]].astype("float64")
    taker = next((c for c in TAKER_COLUMNS if c in df.columns), None)
    if taker is not None:
        out["taker_buy_volume"] = df[taker].astype("float64")
    out.insert(0, "timestamp", ts)
    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if not out["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps are not monotonic")
    gaps = int(gap_flags(out, bar_minutes).sum())
    if gaps:
        warnings.warn(
            f"{gaps} timestamp gaps/irregular steps found; samples whose candle window or "
            "label horizon spans a gap are dropped",
            stacklevel=2,
        )
    return out


# --------------------------------------------------------------------------- #
# Pivots (ZigZag, causal)
# --------------------------------------------------------------------------- #
def find_confirmed_pivots(df: pd.DataFrame, reversal: float = 0.005) -> pd.DataFrame:
    """Return causal ZigZag pivots with occurrence and confirmation indices.

    A high is confirmed only after price falls ``reversal`` from the running
    high; a low is confirmed only after price rises by that amount. Therefore
    ``confirm_idx`` is always later than ``pivot_idx`` and is the only timestamp
    at which the pivot may become an input feature.
    """
    if not 0 < reversal < 1:
        raise ValueError("reversal must be between 0 and 1")
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    if n < 2:
        return pd.DataFrame(columns=["pivot_idx", "confirm_idx", "price", "kind"])

    direction = 0  # +1 tracks a high, -1 tracks a low
    run_hi, hi_idx = high[0], 0
    run_lo, lo_idx = low[0], 0
    records: list[tuple[int, int, float, int]] = []
    for i in range(1, n):
        if direction == 0:
            if high[i] > run_hi:
                run_hi, hi_idx = high[i], i
            if low[i] < run_lo:
                run_lo, lo_idx = low[i], i
            up = high[i] / run_lo - 1.0
            down = 1.0 - low[i] / run_hi
            if up >= reversal and up >= down and lo_idx < i:
                records.append((lo_idx, i, run_lo, -1))
                direction, run_hi, hi_idx = 1, high[i], i
            elif down >= reversal and hi_idx < i:
                records.append((hi_idx, i, run_hi, 1))
                direction, run_lo, lo_idx = -1, low[i], i
        elif direction == 1:
            if high[i] >= run_hi:
                run_hi, hi_idx = high[i], i
                # OHLC does not reveal whether this bar's high or low came first.
                # Never confirm an extreme on the same bar on which it was set.
                continue
            if low[i] <= run_hi * (1.0 - reversal):
                records.append((hi_idx, i, run_hi, 1))
                direction, run_lo, lo_idx = -1, low[i], i
        else:
            if low[i] <= run_lo:
                run_lo, lo_idx = low[i], i
                continue
            if high[i] >= run_lo * (1.0 + reversal):
                records.append((lo_idx, i, run_lo, -1))
                direction, run_hi, hi_idx = 1, high[i], i

    pivots = pd.DataFrame(records, columns=["pivot_idx", "confirm_idx", "price", "kind"])
    if pivots.empty:
        return pivots
    pivots["pivot_time"] = df.loc[pivots["pivot_idx"], "timestamp"].to_numpy()
    pivots["confirmation_time"] = df.loc[pivots["confirm_idx"], "timestamp"].to_numpy()
    pivots["confirmation_lag"] = pivots["confirm_idx"] - pivots["pivot_idx"]
    return pivots


def developing_extremes(
    high: np.ndarray, low: np.ndarray, pivots: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bar state of the ZigZag *unconfirmed* extreme, known at the close of bar i.

    After a high is confirmed the ZigZag tracks a running low (and vice versa),
    starting at the confirmation bar. The tracked extreme only uses data up to
    bar ``i`` so it is causal. Returns (index, price, kind) with kind=+1 for a
    developing high and -1 for a developing low (idx=-1 before the first pivot).
    """
    n = len(high)
    dev_idx = np.full(n, -1, dtype=np.int64)
    dev_price = np.full(n, np.nan, dtype=np.float64)
    dev_kind = np.zeros(n, dtype=np.int64)
    confirm = pivots["confirm_idx"].to_numpy(np.int64)
    kinds = pivots["kind"].to_numpy(np.int64)
    p, kind, idx, price = 0, 0, -1, math.nan
    for i in range(n):
        if p < len(confirm) and confirm[p] == i:
            kind = -int(kinds[p])
            idx = i
            price = float(high[i] if kind == 1 else low[i])
            p += 1
        elif kind == 1 and high[i] >= price:
            idx, price = i, float(high[i])
        elif kind == -1 and low[i] <= price:
            idx, price = i, float(low[i])
        dev_idx[i], dev_price[i], dev_kind[i] = idx, price, kind
    return dev_idx, dev_price, dev_kind


# --------------------------------------------------------------------------- #
# Bar features (all causal: bar i uses data up to and including bar i)
# --------------------------------------------------------------------------- #
def candle_feature_names(df: pd.DataFrame) -> tuple[str, ...]:
    names = CANDLE_BASE_FEATURES
    return names + (TAKER_FEATURE,) if "taker_buy_volume" in df.columns else names


def compute_bar_features(df: pd.DataFrame, cfg: Config) -> dict:
    """Per-bar features and volatility. ``path_vs_close`` depends on the decision
    candle and is therefore filled per sample (placeholder column here)."""
    n = len(df)
    o, h, l, c, v = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close", "volume"))
    logc = np.log(c)
    r = np.zeros(n)
    r[1:] = np.diff(logc)
    r2 = pd.Series(r * r)
    sigma = np.sqrt(r2.ewm(span=cfg.vol_span, adjust=False, min_periods=cfg.vol_span).mean().to_numpy())
    sigma_slow = np.sqrt(
        r2.ewm(span=cfg.slow_vol_span, adjust=False, min_periods=cfg.slow_vol_span).mean().to_numpy())
    sigma = np.maximum(sigma, 1e-6)
    sigma_slow = np.maximum(sigma_slow, 1e-6)
    sigma_prev = np.r_[np.nan, sigma[:-1]]

    span = h - l
    safe = np.where(span > 0, span, np.nan)
    lower_wick = np.minimum(o, c) - l
    upper_wick = h - np.maximum(o, c)
    logv = np.log1p(v)

    # Robust Volume Normalization: Center by rolling median and scale by rolling MAD
    # to maintain invariant scale across changing crypto market regimes (bull vs bear).
    s_logv = pd.Series(logv)
    vol_med = s_logv.rolling(96, min_periods=1).median().to_numpy()
    vol_mad = (pd.Series(np.abs(logv - vol_med)).rolling(96, min_periods=1).median().to_numpy())
    vol_scale = np.maximum(vol_mad * 1.4826, 1e-4)  # 1.4826 * MAD approximates std for normal dist
    rel_log_volume = np.clip((logv - vol_med) / vol_scale, -STANDARDIZE_CLIP, STANDARDIZE_CLIP)

    def trend(k: int) -> np.ndarray:
        out = np.full(n, np.nan)
        out[k:] = logc[k:] - logc[:-k]
        return out / (sigma * math.sqrt(k))

    ts = df["timestamp"]
    hour = (ts.dt.hour + ts.dt.minute / 60.0).to_numpy(np.float64)
    dow = ts.dt.dayofweek.to_numpy(np.float64)
    cols = {
        "ret_z": r / sigma_prev,
        "range_z": np.log(h / l) / sigma_prev,
        "body_frac": np.nan_to_num((c - o) / safe),
        "wick_imbalance": np.nan_to_num((lower_wick - upper_wick) / safe),
        "path_vs_close": np.zeros(n),
        "rel_log_volume": rel_log_volume,
        "sigma_ratio": np.log(sigma / sigma_slow),
        "trend_1d_z": trend(TREND_SHORT),
        "trend_5d_z": trend(TREND_LONG),
        "hour_sin": np.sin(2 * np.pi * hour / 24.0), "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "dow_sin": np.sin(2 * np.pi * dow / 7.0), "dow_cos": np.cos(2 * np.pi * dow / 7.0),
    }
    if "taker_buy_volume" in df.columns:
        vol = np.where(v > 0, v, np.nan)
        cols[TAKER_FEATURE] = np.clip(np.nan_to_num(2.0 * df["taker_buy_volume"].to_numpy(np.float64) / vol - 1.0),
                                      -1.0, 1.0)
    return {"sigma": sigma, "logc": logc, "cols": cols, "names": candle_feature_names(df)}


# --------------------------------------------------------------------------- #
# Unified label
# --------------------------------------------------------------------------- #
def triple_barrier_labels(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, decision_idx: np.ndarray,
    barrier: np.ndarray, horizon: int,
) -> np.ndarray:
    """SHORT=0 / HOLD=1 / LONG=2 for a trade entered at open[t+1] with +-barrier.

    Candles t+1 .. t+horizon are scanned. If both barriers are touched inside the
    same candle (order unknown) the label is HOLD, matching the backtest where such a
    trade would be stopped out.
    """
    idx = decision_idx[:, None] + 1 + np.arange(horizon)[None, :]
    ref = open_[decision_idx + 1]
    hit_up = high[idx] >= (ref * (1.0 + barrier))[:, None]
    hit_dn = low[idx] <= (ref * (1.0 - barrier))[:, None]
    first_up = np.where(hit_up.any(axis=1), hit_up.argmax(axis=1), horizon)
    first_dn = np.where(hit_dn.any(axis=1), hit_dn.argmax(axis=1), horizon)
    y = np.ones(len(decision_idx), dtype=np.int64)
    y[first_up < first_dn] = 2
    y[first_dn < first_up] = 0
    return y


@dataclass
class PreparedData:
    decision_idx: np.ndarray
    label_end_idx: np.ndarray
    label_end_ns: np.ndarray
    timestamp_ns: np.ndarray
    pivot_x: np.ndarray
    candle_x: np.ndarray
    y: np.ndarray
    barrier_pct: np.ndarray      # per-sample barrier b (same b the trade uses)
    bar_barrier: np.ndarray      # per-bar barrier (NaN during warm-up)
    candle_features: tuple[str, ...]


def prepare_examples(df: pd.DataFrame, pivots: pd.DataFrame, cfg: Config) -> PreparedData:
    """Build aligned samples without using information unavailable at decision time."""
    n = len(df)
    P, C, H = cfg.pivot_count, cfg.candle_count, cfg.barrier_horizon
    if len(pivots) <= P:
        raise ValueError(
            f"Only {len(pivots)} pivots found; need more than pivot_count={P}. "
            "Check the data length or lower pivot_reversal."
        )
    feats = compute_bar_features(df, cfg)
    sigma, logc, names = feats["sigma"], feats["logc"], feats["names"]
    open_ = df["open"].to_numpy(np.float64)
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)
    ns = _ns_int(df["timestamp"])
    pivot_idx = pivots["pivot_idx"].to_numpy(np.int64)
    confirm_idx = pivots["confirm_idx"].to_numpy(np.int64)
    pivot_price = pivots["price"].to_numpy(np.float64)
    pivot_kind = pivots["kind"].to_numpy(np.float64)
    prev_idx = np.r_[pivot_idx[:1], pivot_idx[:-1]]
    prev_price = np.r_[pivot_price[:1], pivot_price[:-1]]
    dev_idx, dev_price, dev_kind = developing_extremes(high, low, pivots)
    gap_cum = np.cumsum(gap_flags(df, cfg.bar_minutes))
    bar_barrier = np.clip(cfg.barrier_vol_mult * sigma * math.sqrt(H), cfg.min_barrier, cfg.max_barrier)

    first = max(cfg.slow_vol_span + C, int(confirm_idx[P - 1]))
    ts = np.arange(first, n - H, cfg.sample_stride, dtype=np.int64)   # t + H <= n - 1
    if len(ts) == 0:
        raise ValueError("No examples were produced; data too short")
    ok = (gap_cum[ts] - gap_cum[ts - C + 1] == 0) & (gap_cum[ts + H] - gap_cum[ts] == 0)
    ts = ts[ok]
    known_end = np.searchsorted(confirm_idx, ts, side="right")
    keep = (known_end >= P) & (dev_idx[ts] >= 0) & np.isfinite(sigma[ts])
    ts, known_end = ts[keep], known_end[keep]
    m = len(ts)
    if m == 0:
        raise ValueError("No examples were produced; check pivot threshold and data length")

    # ---- pivot tokens: P confirmed pivots + 1 developing extreme + broadcast context
    sel = known_end[:, None] - P + np.arange(P)[None, :]
    pi, pp = pivot_idx[sel], pivot_price[sel]
    close_t = close[ts]
    sig_h = sigma[ts] * math.sqrt(H)
    F, T = len(PIVOT_FEATURES), len(TOKEN_FEATURES)
    pivot_x = np.zeros((m, P + 1, F), dtype=np.float32)
    tk = pivot_x[:, :P]
    tk[:, :, 0] = (ts[:, None] - pi) / SCALE_BARS
    tk[:, :, 1] = np.log(pp / close_t[:, None]) / sig_h[:, None]
    tk[:, :, 2] = pivot_kind[sel]
    tk[:, :, 3] = np.abs(np.log(pp / prev_price[sel])) / sig_h[:, None]
    tk[:, :, 4] = (pi - prev_idx[sel]) / SCALE_BARS
    tk[:, :, 5] = (confirm_idx[sel] - pi) / SCALE_BARS
    last = known_end - 1
    di, dp = dev_idx[ts], dev_price[ts]
    dv = pivot_x[:, P]
    dv[:, 0] = (ts - di) / SCALE_BARS
    dv[:, 1] = np.log(dp / close_t) / sig_h
    dv[:, 2] = dev_kind[ts]
    dv[:, 3] = np.abs(np.log(dp / pivot_price[last])) / sig_h
    dv[:, 4] = (di - pivot_idx[last]) / SCALE_BARS
    dv[:, 6] = np.abs(np.log(close_t / dp)) / cfg.pivot_reversal
    dv[:, 7] = 1.0
    ctx = np.column_stack([feats["cols"][name][ts] for name in CONTEXT_FEATURES])
    pivot_x[:, :, T:] = ctx[:, None, :]

    # ---- candle windows
    raw = np.column_stack([feats["cols"][nm] for nm in names]).astype(np.float32)
    win = ts[:, None] - C + 1 + np.arange(C)[None, :]
    candle_x = raw[win]
    candle_x[:, :, names.index("path_vs_close")] = ((logc[win] - logc[ts][:, None]) / sig_h[:, None]).astype(np.float32)

    # ---- unified label
    barrier_pct = bar_barrier[ts]
    y = triple_barrier_labels(open_, high, low, ts, barrier_pct, H)
    ends = ts + H
    return PreparedData(
        decision_idx=ts, label_end_idx=ends, label_end_ns=ns[ends], timestamp_ns=ns[ts],
        pivot_x=pivot_x, candle_x=candle_x, y=y, barrier_pct=barrier_pct,
        bar_barrier=bar_barrier, candle_features=tuple(names),
    )


# --------------------------------------------------------------------------- #
# Models & Temperature Calibration
# --------------------------------------------------------------------------- #
class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


class PivotTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        input_size: int = len(PIVOT_FEATURES),
        hidden: int = 64,
        dropout: float = 0.15,
        heads: int = 4,
        classes: int = 3,
    ):
        super().__init__()
        if hidden % heads:
            raise ValueError(f"hidden_size={hidden} must be divisible by heads={heads}")
        self.seq_len = seq_len
        self.project = nn.Linear(input_size, hidden)
        self.position = nn.Parameter(torch.zeros(1, seq_len, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 3,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.seq_len:
            raise ValueError(f"sequence length {x.shape[1]} exceeds seq_len={self.seq_len}")
        z = self.project(x) + self.position[:, : x.shape[1]]
        return self.head(self.encoder(z)[:, -1])


class CandleLSTM(nn.Module):
    def __init__(self, input_size: int = len(CANDLE_BASE_FEATURES), hidden: int = 64,
                 dropout: float = 0.15, classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=2, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axes = tuple(range(x.ndim - 1))
    mean = x.mean(axis=axes, dtype=np.float64).astype(np.float32)
    std = x.std(axis=axes, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def standardize(
    x: np.ndarray, stats: tuple[np.ndarray, np.ndarray], ids: np.ndarray | None = None,
    clip: float = STANDARDIZE_CLIP,
) -> np.ndarray:
    """Select ``ids`` (one copy), z-score in place and clip fat tails."""
    mean, std = stats
    z = x[ids] if ids is not None else x.copy()
    z = z.astype(np.float32, copy=False)
    z -= mean
    z /= std
    np.clip(z, -clip, clip, out=z)
    return z


def _class_weights(y: np.ndarray, classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=classes).astype(np.float64)
    weights = len(y) / (classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def nll(y: np.ndarray, prob: np.ndarray) -> float:
    """Mean negative log-likelihood (log loss) of the true class."""
    p = np.clip(prob[np.arange(len(y)), y], 1e-7, 1.0)
    return float(-np.mean(np.log(p)))


def fit_temperature_scaling(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    """Find scalar temperature T > 0 on validation logits using L-BFGS to calibrate probabilities."""
    model.eval()
    logits_list, y_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits_list.append(model(xb.to(device)))
            y_list.append(yb.to(device))
    if not logits_list:
        return 1.0
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(y_list, dim=0)

    temperature = nn.Parameter(torch.ones(1, device=device))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval_loss():
        optimizer.zero_grad()
        loss = criterion(logits / temperature.clamp(min=0.05), labels)
        loss.backward()
        return loss

    try:
        optimizer.step(eval_loss)
        t_val = float(temperature.clamp(min=0.05, max=5.0).item())
        return t_val
    except Exception:
        return 1.0


def train_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    stop_x: np.ndarray,
    stop_y: np.ndarray,
    cfg: Config,
    classes: int,
) -> tuple[nn.Module, list[dict], float]:
    """Train with early stopping on the log loss of a purged slice of the TRAIN window."""
    device = resolve_device(cfg.device)
    model = model.to(device)
    train_loader = DataLoader(
        ArrayDataset(train_x, train_y), batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
    )
    stop_loader = DataLoader(ArrayDataset(stop_x, stop_y), batch_size=cfg.batch_size * 2)
    weight = _class_weights(train_y, classes).to(device) if cfg.class_weighted else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    history, best_state, best_loss = [], None, math.inf
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
        pred, prob = predict(model, stop_loader, device, temperature=1.0)
        val_ll = nll(stop_y, prob)
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / len(train_y),
            "early_stop_log_loss": val_ll,
            "early_stop_balanced_accuracy": float(balanced_accuracy_score(stop_y, pred)),
        })
        if val_ll < best_loss:
            best_loss = val_ll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise ValueError("epochs must be >= 1")
    model.load_state_dict(best_state)

    temperature = 1.0
    if cfg.calibrate_temperature:
        temperature = fit_temperature_scaling(model, stop_loader, device)

    return model, history, temperature


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device, temperature: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, probabilities = [], []
    inv_t = 1.0 / max(temperature, 1e-4)
    for xb, _ in loader:
        logits = model(xb.to(device)) * inv_t
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        probabilities.append(prob)
        labels.append(prob.argmax(axis=1))
    p = np.concatenate(probabilities)
    return np.concatenate(labels), p


def predict_array(
    model: nn.Module, x: np.ndarray, cfg: Config, temperature: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    dummy = np.zeros(len(x), dtype=np.int64)
    loader = DataLoader(ArrayDataset(x, dummy), batch_size=cfg.batch_size * 2)
    return predict(model, loader, resolve_device(cfg.device), temperature=temperature)


def predict_ids(fitted: dict, data: PreparedData, ids: np.ndarray, cfg: Config) -> dict:
    _, pprob = predict_array(
        fitted["pivot_model"],
        standardize(data.pivot_x, fitted["pivot_stats"], ids),
        cfg,
        temperature=fitted.get("pivot_temperature", 1.0),
    )
    _, cprob = predict_array(
        fitted["lstm_model"],
        standardize(data.candle_x, fitted["candle_stats"], ids),
        cfg,
        temperature=fitted.get("lstm_temperature", 1.0),
    )
    return {"ids": ids, "pprob": pprob, "cprob": cprob}


def ensemble_probs(preds: dict, cfg: Config) -> np.ndarray:
    return cfg.ensemble_weight * preds["pprob"] + (1.0 - cfg.ensemble_weight) * preds["cprob"]


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def year_bounds(year: int) -> tuple[int, int]:
    start = pd.Timestamp(f"{year}-01-01", tz="UTC").as_unit("ns").value
    end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC").as_unit("ns").value
    return int(start), int(end)


def year_masks(data: PreparedData, year: int) -> tuple[np.ndarray, np.ndarray]:
    start, end = year_bounds(year)
    ts, label_end = data.timestamp_ns, data.label_end_ns
    # Purge training observations whose future label overlaps validation.
    train = (ts < start) & (label_end < start)
    val = (ts >= start) & (ts < end) & (label_end < end)
    return train, val


def purged_early_stop_split(
    ids: np.ndarray, data: PreparedData, fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Chronological tail of the training window for early stopping, with purging."""
    cut = int(len(ids) * (1.0 - fraction))
    if cut < 1 or cut >= len(ids):
        raise ValueError("Not enough training samples for an early-stopping split")
    stop_ids = ids[cut:]
    fit_ids = ids[:cut]
    fit_ids = fit_ids[data.label_end_ns[fit_ids] < data.timestamp_ns[stop_ids[0]]]
    if len(fit_ids) == 0:
        raise ValueError("Purging left no fit samples")
    return fit_ids, stop_ids


def train_pair(
    data: PreparedData, fit_ids: np.ndarray, stop_ids: np.ndarray, cfg: Config, seed: int
) -> dict:
    p_stats = fit_standardizer(data.pivot_x[fit_ids])
    c_stats = fit_standardizer(data.candle_x[fit_ids])
    seed_everything(seed)
    p_model, p_hist, p_temp = train_model(
        PivotTransformer(seq_len=data.pivot_x.shape[1], input_size=data.pivot_x.shape[2],
                         hidden=cfg.hidden_size, dropout=cfg.dropout),
        standardize(data.pivot_x, p_stats, fit_ids), data.y[fit_ids],
        standardize(data.pivot_x, p_stats, stop_ids), data.y[stop_ids], cfg, 3,
    )
    seed_everything(seed + 1)
    c_model, c_hist, c_temp = train_model(
        CandleLSTM(input_size=data.candle_x.shape[2], hidden=cfg.hidden_size, dropout=cfg.dropout),
        standardize(data.candle_x, c_stats, fit_ids), data.y[fit_ids],
        standardize(data.candle_x, c_stats, stop_ids), data.y[stop_ids], cfg, 3,
    )
    return {
        "pivot_model": p_model, "lstm_model": c_model, "pivot_stats": p_stats,
        "candle_stats": c_stats, "pivot_history": p_hist, "lstm_history": c_hist,
        "pivot_temperature": p_temp, "lstm_temperature": c_temp,
        "fit_samples": int(len(fit_ids)), "early_stop_samples": int(len(stop_ids)),
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_probs(y: np.ndarray, prob: np.ndarray, prior: np.ndarray) -> dict:
    """Log loss vs the class prior, accuracy, and directional accuracy on LONG/SHORT cases."""
    pred = prob.argmax(axis=1)
    ll = nll(y, prob)
    prior_ll = float(-np.mean(np.log(np.clip(prior[y], 1e-7, 1.0))))
    directional = y != 1
    dir_acc = (float(np.mean((prob[directional, 2] > prob[directional, 0]) == (y[directional] == 2)))
               if directional.any() else float("nan"))
    return {
        "log_loss": ll, "prior_log_loss": prior_ll, "log_loss_skill": float(1.0 - ll / prior_ll),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "dir_accuracy": dir_acc,
        "report": classification_report(y, pred, labels=[0, 1, 2], target_names=list(CLASS_NAMES),
                                        output_dict=True, zero_division=0),
    }


def logreg_baseline(
    x_all: np.ndarray, y_all: np.ndarray, train_ids: np.ndarray, val_ids: np.ndarray,
    cfg: Config, prior: np.ndarray,
) -> dict:
    train_ids = train_ids[-cfg.baseline_max_samples:]
    stats = fit_standardizer(x_all[train_ids])
    xtr = standardize(x_all, stats, train_ids).reshape(len(train_ids), -1)
    xva = standardize(x_all, stats, val_ids).reshape(len(val_ids), -1)
    clf = LogisticRegression(max_iter=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(xtr, y_all[train_ids])
    prob = np.full((len(val_ids), 3), 1e-7)
    prob[:, clf.classes_] = clf.predict_proba(xva)
    return evaluate_probs(y_all[val_ids], prob, prior)


def edge_calibration(data: PreparedData, preds: dict, cfg: Config, bins: int = 5) -> pd.DataFrame:
    """Check predicted directional edge vs realised empirical edge in quantile bins."""
    ids = preds["ids"]
    ens = ensemble_probs(preds, cfg)
    s = ens[:, 2] - ens[:, 0]
    y = data.y[ids]
    edges = np.unique(np.quantile(s, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = which == b
        if m.any():
            rows.append({
                "bin": b, "samples": int(m.sum()), "predicted_edge": float(s[m].mean()),
                "realised_edge": float(np.mean(y[m] == 2) - np.mean(y[m] == 0)),
                "hold_rate": float(np.mean(y[m] == 1)),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
_ROW_KEYS = ("log_loss", "prior_log_loss", "log_loss_skill", "accuracy", "balanced_accuracy", "dir_accuracy")


def run_walk_forward(data: PreparedData, cfg: Config) -> tuple[pd.DataFrame, dict, dict]:
    rows, details, fold_preds = [], {}, {}
    for year in cfg.fold_years:
        train_mask, val_mask = year_masks(data, year)
        if not train_mask.any() or not val_mask.any():
            raise ValueError(f"Fold {year} has no training or validation samples")
        train_ids = np.flatnonzero(train_mask)
        fit_ids, stop_ids = purged_early_stop_split(train_ids, data, cfg.early_stop_fraction)
        fitted = train_pair(data, fit_ids, stop_ids, cfg, cfg.seed + year)

        start, end = year_bounds(year)
        year_ids = np.flatnonzero((data.timestamp_ns >= start) & (data.timestamp_ns < end))
        preds = predict_ids(fitted, data, year_ids, cfg)
        in_val = val_mask[year_ids]
        val_ids = year_ids[in_val]
        y_val = data.y[val_ids]
        prior = np.bincount(data.y[fit_ids], minlength=3) / len(fit_ids)
        majority_acc = float(np.mean(y_val == int(prior.argmax())))

        probs = {
            "pivot_transformer": preds["pprob"][in_val],
            "candle_lstm": preds["cprob"][in_val],
            "ensemble": ensemble_probs(preds, cfg)[in_val],
        }
        base: dict[str, dict] = {}
        if cfg.run_baselines:
            base["pivot_transformer"] = logreg_baseline(data.pivot_x, data.y, fit_ids, val_ids, cfg, prior)
            base["candle_lstm"] = logreg_baseline(data.candle_x, data.y, fit_ids, val_ids, cfg, prior)
        fold_details = {}
        for name, prob in probs.items():
            m = evaluate_probs(y_val, prob, prior)
            b = base.get(name)
            rows.append({
                "fold": year, "model": name, "samples": int(len(val_ids)),
                **{k: m[k] for k in _ROW_KEYS}, "majority_accuracy": majority_acc,
                "logreg_log_loss": b["log_loss"] if b else np.nan,
                "logreg_dir_accuracy": b["dir_accuracy"] if b else np.nan,
            })
            fold_details[name] = m
        details[str(year)] = {
            "models": fold_details, "pivot_history": fitted["pivot_history"],
            "lstm_history": fitted["lstm_history"], "pivot_temperature": fitted["pivot_temperature"],
            "lstm_temperature": fitted["lstm_temperature"], "fit_samples": fitted["fit_samples"],
            "early_stop_samples": fitted["early_stop_samples"], "prior": prior,
        }
        fold_preds[year] = preds
    return pd.DataFrame(rows), details, fold_preds


def fit_final_models(data: PreparedData, cfg: Config) -> dict:
    cutoff, _ = year_bounds(cfg.backtest_year)
    train = (data.timestamp_ns < cutoff) & (data.label_end_ns < cutoff)
    if not train.any():
        raise ValueError("No final training samples before backtest year")
    fit_ids, stop_ids = purged_early_stop_split(np.flatnonzero(train), data, cfg.early_stop_fraction)
    return train_pair(data, fit_ids, stop_ids, cfg, cfg.seed + cfg.backtest_year)


# --------------------------------------------------------------------------- #
# Backtest: the trade is exactly the labelled experiment
# --------------------------------------------------------------------------- #
@dataclass
class Market:
    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    year: np.ndarray


def make_market(df: pd.DataFrame) -> Market:
    ts = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    return Market(
        ts=ts,
        open=df["open"].to_numpy(np.float64), high=df["high"].to_numpy(np.float64),
        low=df["low"].to_numpy(np.float64), close=df["close"].to_numpy(np.float64),
        year=df["timestamp"].dt.year.to_numpy(),
    )


def build_signals(data: PreparedData, preds: dict, n_bars: int, cfg: Config, min_edge: float, mode: str):
    """Cost-aware edge rule on the ensemble probability.

    Expected return of a 1:1 barrier trade is about b*(P_dir - P_opp) - cost, so the
    break-even probability gap is cost / b. A trade needs gap - cost/b >= min_edge.
    """
    ids, pp, cp = preds["ids"], preds["pprob"], preds["cprob"]
    ens = ensemble_probs(preds, cfg)
    b = data.barrier_pct[ids]
    breakeven = cfg.cost / b
    edge_long = ens[:, 2] - ens[:, 0] - breakeven
    edge_short = ens[:, 0] - ens[:, 2] - breakeven
    side = np.where(edge_long >= edge_short, 1, -1)
    edge = np.maximum(edge_long, edge_short)
    ok = edge >= min_edge
    if mode == "agree":
        side_p = np.where(pp[:, 2] > pp[:, 0], 1, -1)
        side_c = np.where(cp[:, 2] > cp[:, 0], 1, -1)
        ok &= (side_p == side) & (side_c == side)
    bars = data.decision_idx[ids]
    sig = np.zeros(n_bars, dtype=np.int64)
    edge_bar = np.zeros(n_bars)
    barrier_bar = np.zeros(n_bars)
    sig[bars], edge_bar[bars], barrier_bar[bars] = np.where(ok, side, 0), edge, b
    return sig, edge_bar, barrier_bar


def _empty_summary(year: int, n_pred: int, min_edge: float, mode: str) -> dict:
    return {
        "strategy": "unified_triple_barrier_v4_next_open", "year": year,
        "prediction_candles": n_pred, "trades": 0, "total_return": 0.0,
        "win_rate": None, "average_trade": None, "profit_factor": None,
        "max_drawdown": 0.0, "sharpe": None, "exposure": 0.0, "avg_barrier": None,
        "min_edge": min_edge, "signal_mode": mode,
    }


def run_backtest(
    mkt: Market, data: PreparedData, preds: dict, year: int, cfg: Config,
    min_edge: float | None = None, mode: str | None = None,
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Fixed-barrier backtest identical to the label definition.

    Decision at the close of candle ``t``; order filled at open of ``t+1`` plus slippage.
    Stop and take-profit sit at +-b from open of ``t+1``.
    - If market gaps beyond stop/take at open, fills at open price (gap-aware).
    - If both touched in same candle, stop wins.
    - Closed at close of candle ``t+H`` at the latest.
    """
    min_edge = cfg.min_edge if min_edge is None else min_edge
    mode = cfg.signal_mode if mode is None else mode
    n_bars = len(mkt.close)
    year_rows = np.flatnonzero(mkt.year == year)
    if len(year_rows) == 0 or len(preds["ids"]) == 0:
        raise ValueError(f"No model samples in {year}")
    first_bar, last_bar = int(year_rows[0]), int(year_rows[-1])
    sig, edge_bar, barrier_bar = build_signals(data, preds, n_bars, cfg, min_edge, mode)

    op, hi_a, lo_a, cl = mkt.open, mkt.high, mkt.low, mkt.close
    slip, fee = cfg.slippage_bps / 10_000.0, cfg.fee_bps / 10_000.0
    funding_per_bar = (cfg.funding_bps_per_8h / 10_000.0) / (8.0 * 60.0 / cfg.bar_minutes)
    trades: list[dict] = []
    curve = np.empty(last_bar - first_bar + 1)
    equity, position = 1.0, None
    pending: tuple[int, float, float] | None = None
    in_position_bars = 0

    def close_position(t: int, raw_exit: float, reason: str) -> None:
        nonlocal position, equity
        side = position["side"]
        exit_price = raw_exit * (1.0 - slip * side)
        bars_held = t - position["entry_idx"] + 1
        funding_cost = bars_held * funding_per_bar
        net = side * (exit_price / position["entry"] - 1.0) - 2.0 * fee - funding_cost
        equity *= 1.0 + net
        trades.append({
            "entry_time": mkt.ts[position["entry_idx"]], "exit_time": mkt.ts[t],
            "side": "LONG" if side == 1 else "SHORT", "barrier": position["barrier"],
            "entry": position["entry"], "exit": exit_price, "return": net,
            "bars_held": bars_held, "exit_reason": reason,
            "edge": position["edge"], "equity": equity,
        })
        position = None

    def open_position(t: int, side: int, barrier: float, edge: float) -> None:
        nonlocal position
        reference = float(op[t])
        position = {
            "side": side, "entry_idx": t, "entry": reference * (1.0 + slip * side),
            "stop": reference * (1.0 - barrier * side), "take": reference * (1.0 + barrier * side),
            "natural_expiry": t + cfg.barrier_horizon - 1, "barrier": barrier, "edge": edge,
        }

    for t in range(first_bar, last_bar + 1):
        # 1) fill the order created at the previous close at this candle's open
        if pending is not None and position is None:
            open_position(t, *pending)
        pending = None

        # 2) intrabar stop / take-profit (entry candle included: we entered at its open)
        if position is not None:
            side, stop, take = position["side"], position["stop"], position["take"]
            hi, lo, o = float(hi_a[t]), float(lo_a[t]), float(op[t])
            hit_stop = lo <= stop if side == 1 else hi >= stop
            hit_take = hi >= take if side == 1 else lo <= take
            if hit_stop:
                # Stop wins ties; gap-aware fill at open if opened past stop
                raw = min(stop, o) if side == 1 else max(stop, o)
                close_position(t, raw, "SL_ambiguous" if hit_take else "SL")
            elif hit_take:
                # Gap-aware TP fill: if opened above TP for LONG or below TP for SHORT
                raw = max(take, o) if side == 1 else min(take, o)
                close_position(t, raw, "TP")

        # 3) time-out (same horizon as the label) / end of backtest at the close
        if position is not None:
            if t == position["natural_expiry"]:
                close_position(t, float(cl[t]), "timeout")
            elif t == last_bar:
                close_position(t, float(cl[t]), "end_of_backtest")

        # 4) decision at the close of t -> filled at the open of t+1
        if position is None and t < last_bar and sig[t] != 0:
            pending = (int(sig[t]), float(barrier_bar[t]), float(edge_bar[t]))

        # 5) mark-to-market equity at the close
        if position is not None:
            in_position_bars += 1
            s = position["side"]
            bars_so_far = t - position["entry_idx"] + 1
            f_cost = bars_so_far * funding_per_bar
            unreal = s * (float(cl[t]) * (1.0 - slip * s) / position["entry"] - 1.0) - 2.0 * fee - f_cost
            curve[t - first_bar] = equity * (1.0 + unreal)
        else:
            curve[t - first_bar] = equity

    trade_df = pd.DataFrame(trades)
    if trade_df.empty:
        return trade_df, _empty_summary(year, int(len(preds["ids"])), min_edge, mode), curve
    returns = trade_df["return"].to_numpy()
    prev = np.r_[1.0, curve[:-1]]
    bar_ret = curve / prev - 1.0
    peak = np.maximum.accumulate(np.r_[1.0, curve])[1:]
    bars_per_year = 365.0 * 24.0 * 60.0 / cfg.bar_minutes
    std = float(bar_ret.std())
    losses = float(-returns[returns < 0].sum())
    summary = {
        "strategy": "unified_triple_barrier_v4_next_open", "year": year,
        "prediction_candles": int(len(preds["ids"])), "trades": int(len(trade_df)),
        "total_return": float(curve[-1] - 1.0),
        "win_rate": float((returns > 0).mean()), "average_trade": float(returns.mean()),
        "profit_factor": float(returns[returns > 0].sum() / losses) if losses > 0 else None,
        "max_drawdown": float((curve / peak - 1.0).min()),
        "sharpe": float(bar_ret.mean() / std * math.sqrt(bars_per_year)) if std > 0 else None,
        "exposure": float(in_position_bars / len(curve)),
        "avg_barrier": float(trade_df["barrier"].mean()),
        "min_edge": min_edge, "signal_mode": mode,
    }
    return trade_df, summary, curve


def tune_thresholds(
    mkt: Market, data: PreparedData, fold_preds: dict, cfg: Config
) -> tuple[float, str, pd.DataFrame]:
    """Choose min_edge and signal mode on the walk-forward folds ONLY (never the test year)."""
    rows = []
    for edge, mode in product(cfg.min_edge_grid, cfg.signal_mode_grid):
        per = [run_backtest(mkt, data, p, y, cfg, edge, mode)[1] for y, p in fold_preds.items()]
        rets = [s["total_return"] for s in per]
        rows.append({
            "min_edge": edge, "signal_mode": mode,
            "mean_fold_return": float(np.mean(rets)), "min_fold_return": float(np.min(rets)),
            "min_fold_trades": int(min(s["trades"] for s in per)),
        })
    table = pd.DataFrame(rows)
    ok = table[table["min_fold_trades"] >= cfg.min_tune_trades]
    if ok.empty:
        warnings.warn("No threshold combination reached min_tune_trades on every fold; using config defaults",
                      stacklevel=2)
        return cfg.min_edge, cfg.signal_mode, table
    best = ok.sort_values(["mean_fold_return", "min_fold_return"], ascending=False).iloc[0]
    return float(best["min_edge"]), str(best["signal_mode"]), table


def _simulate_fixed_trade(
    mkt: Market, entry_bar: int, side: int, barrier: float, cfg: Config, last_bar: int
) -> float:
    """Same entry/SL/TP/time-out rules as the backtest, without any model."""
    slip, fee = cfg.slippage_bps / 10_000.0, cfg.fee_bps / 10_000.0
    funding_per_bar = (cfg.funding_bps_per_8h / 10_000.0) / (8.0 * 60.0 / cfg.bar_minutes)
    ref = mkt.open[entry_bar]
    entry = ref * (1.0 + slip * side)
    stop, take = ref * (1.0 - barrier * side), ref * (1.0 + barrier * side)
    end = min(entry_bar + cfg.barrier_horizon - 1, last_bar)
    raw = mkt.close[end]
    bars_held = end - entry_bar + 1
    for i in range(entry_bar, end + 1):
        hi, lo = mkt.high[i], mkt.low[i]
        hit_stop = lo <= stop if side == 1 else hi >= stop
        hit_take = hi >= take if side == 1 else lo <= take
        if hit_stop:
            raw = min(stop, mkt.open[i]) if side == 1 else max(stop, mkt.open[i])
            bars_held = i - entry_bar + 1
            break
        if hit_take:
            raw = max(take, mkt.open[i]) if side == 1 else min(take, mkt.open[i])
            bars_held = i - entry_bar + 1
            break
    exit_price = raw * (1.0 - slip * side)
    funding_cost = bars_held * funding_per_bar
    return side * (exit_price / entry - 1.0) - 2.0 * fee - funding_cost


def random_entry_baseline(
    mkt: Market, data: PreparedData, trades: pd.DataFrame, year: int, cfg: Config, model_return: float
) -> dict | None:
    """Random entries with the same trade count and long share under the same trade rules."""
    if trades.empty or cfg.random_baseline_sims <= 0:
        return None
    rng = np.random.default_rng(cfg.seed)
    year_rows = np.flatnonzero(mkt.year == year)
    first_bar, last_bar = int(year_rows[0]), int(year_rows[-1])
    candidates = np.arange(first_bar, last_bar)
    candidates = candidates[np.isfinite(data.bar_barrier[candidates - 1])]
    n, p_long = len(trades), float((trades["side"] == "LONG").mean())
    totals = np.empty(cfg.random_baseline_sims)
    for s in range(cfg.random_baseline_sims):
        entries = rng.choice(candidates, size=n)
        sides = np.where(rng.random(n) < p_long, 1, -1)
        equity = 1.0
        for e, sd in zip(entries, sides):
            equity *= 1.0 + _simulate_fixed_trade(mkt, int(e), int(sd), float(data.bar_barrier[e - 1]),
                                                  cfg, last_bar)
        totals[s] = equity - 1.0
    return {
        "simulations": int(cfg.random_baseline_sims), "mean_total_return": float(totals.mean()),
        "p05_total_return": float(np.percentile(totals, 5)),
        "p95_total_return": float(np.percentile(totals, 95)),
        "model_percentile": float((totals < model_return).mean()),
        "note": "random entries obey the same barriers/horizon; compares trade selection only",
    }


def bootstrap_mean_ci(returns: np.ndarray, cfg: Config) -> tuple[float, float]:
    rng = np.random.default_rng(cfg.seed)
    idx = rng.integers(0, len(returns), size=(cfg.bootstrap_samples, len(returns)))
    means = returns[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --------------------------------------------------------------------------- #
# Artifacts / pipeline
# --------------------------------------------------------------------------- #
def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_artifacts(
    fitted: dict, cfg: Config, signal: tuple[float, str], data: PreparedData,
    fold_table: pd.DataFrame, details: dict, fold_backtests: pd.DataFrame,
    tuning: pd.DataFrame, trades: pd.DataFrame, backtest_summary: dict,
) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(fitted["pivot_model"].state_dict(), out / "pivot_transformer.pt")
    torch.save(fitted["lstm_model"].state_dict(), out / "candle_lstm.pt")
    np.savez(
        out / "normalization.npz",
        pivot_mean=fitted["pivot_stats"][0],
        pivot_std=fitted["pivot_stats"][1],
        candle_mean=fitted["candle_stats"][0],
        candle_std=fitted["candle_stats"][1],
    )
    fold_table.to_csv(out / "walk_forward_metrics.csv", index=False)
    fold_backtests.to_csv(out / "walk_forward_backtests.csv", index=False)
    tuning.to_csv(out / "threshold_tuning.csv", index=False)
    trades.to_csv(out / f"backtest_{cfg.backtest_year}_trades.csv", index=False)
    metadata = {
        "config": asdict(cfg),
        "pivot_features": PIVOT_FEATURES,
        "candle_features": data.candle_features,
        "pivot_sequence_length": cfg.pivot_count + 1,
        "standardize_clip": STANDARDIZE_CLIP,
        "class_mapping": {str(i): n for i, n in enumerate(CLASS_NAMES)},
        "temperatures": {
            "pivot_transformer": fitted.get("pivot_temperature", 1.0),
            "candle_lstm": fitted.get("lstm_temperature", 1.0),
        },
        "label": (
            "triple barrier, entry at next open, barrier b = clip(mult*sigma*sqrt(H)), "
            "same b/H/entry/ties as the backtest"
        ),
        "signal": {
            "min_edge": signal[0], "mode": signal[1], "ensemble_weight": cfg.ensemble_weight,
            "rule": "P(dir)-P(opp) - cost/b >= min_edge", "selected_on": "walk-forward folds only",
        },
        "walk_forward": details,
        "backtest": backtest_summary,
        "causality": (
            "Pivot tokens become visible only at confirmation_time; the developing extreme and all "
            "bar features use data up to the decision candle; early stopping uses a purged tail of "
            "the training window; thresholds are tuned on walk-forward folds."
        ),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
    return out


def run_pipeline(cfg: Config) -> dict:
    seed_everything(cfg.seed)
    df = load_candles(cfg.csv_path, cfg.bar_minutes)
    pivots = find_confirmed_pivots(df, cfg.pivot_reversal)
    data = prepare_examples(df, pivots, cfg)
    mkt = make_market(df)

    fold_table, fold_details, fold_preds = run_walk_forward(data, cfg)
    if cfg.tune_thresholds:
        min_edge, mode, tuning = tune_thresholds(mkt, data, fold_preds, cfg)
    else:
        min_edge, mode, tuning = cfg.min_edge, cfg.signal_mode, pd.DataFrame()
    fold_backtests = pd.DataFrame(
        [run_backtest(mkt, data, p, y, cfg, min_edge, mode)[1] for y, p in fold_preds.items()]
    )

    fitted = fit_final_models(data, cfg)
    start, end = year_bounds(cfg.backtest_year)
    test_ids = np.flatnonzero((data.timestamp_ns >= start) & (data.timestamp_ns < end))
    preds = predict_ids(fitted, data, test_ids, cfg)
    trades, summary, _ = run_backtest(mkt, data, preds, cfg.backtest_year, cfg, min_edge, mode)
    if not trades.empty:
        lo, hi = bootstrap_mean_ci(trades["return"].to_numpy(), cfg)
        summary["mean_trade_ci95"] = [lo, hi]
        summary["random_entry_baseline"] = random_entry_baseline(
            mkt, data, trades, cfg.backtest_year, cfg, summary["total_return"]
        )
    output = save_artifacts(
        fitted, cfg, (min_edge, mode), data, fold_table, fold_details,
        fold_backtests, tuning, trades, summary,
    )
    return {
        "candles": len(df), "pivots": len(pivots), "samples": len(data.decision_idx),
        "folds": fold_table, "fold_backtests": fold_backtests, "tuning": tuning,
        "backtest": summary, "output_dir": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=Config.csv_path)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--pivot-reversal", type=float, default=Config.pivot_reversal)
    parser.add_argument("--barrier-mult", type=float, default=Config.barrier_vol_mult)
    parser.add_argument("--sample-stride", type=int, default=Config.sample_stride)
    parser.add_argument("--device", default=Config.device)
    parser.add_argument("--backtest-year", type=int, default=Config.backtest_year)
    parser.add_argument("--fold-years", type=int, nargs="+", default=list(Config.fold_years))
    parser.add_argument("--min-edge", type=float, default=Config.min_edge)
    parser.add_argument("--signal-mode", choices=["average", "agree"], default=Config.signal_mode)
    parser.add_argument("--class-weighted", action="store_true")
    parser.add_argument("--funding-bps", type=float, default=0.0, help="Funding rate bps per 8h (e.g. 1.0 for 0.01%%)")
    parser.add_argument("--no-calibration", action="store_true", help="skip temperature scaling calibration")
    parser.add_argument("--no-tune", action="store_true", help="skip threshold tuning on the folds")
    parser.add_argument("--no-baselines", action="store_true", help="skip logistic-regression baselines")
    args = parser.parse_args()
    cfg = Config(
        csv_path=args.csv, output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size,
        pivot_reversal=args.pivot_reversal, barrier_vol_mult=args.barrier_mult,
        sample_stride=args.sample_stride, device=args.device, backtest_year=args.backtest_year,
        fold_years=tuple(args.fold_years), min_edge=args.min_edge, signal_mode=args.signal_mode,
        class_weighted=args.class_weighted, funding_bps_per_8h=args.funding_bps,
        calibrate_temperature=not args.no_calibration, tune_thresholds=not args.no_tune,
        run_baselines=not args.no_baselines,
    )
    result = run_pipeline(cfg)
    print(result["folds"].round(4).to_string(index=False))
    print(result["fold_backtests"].to_string(index=False))
    print(json.dumps(result["backtest"], indent=2, default=_json_default))
    print(f"Artifacts: {result['output_dir']}")


if __name__ == "__main__":
    main()
