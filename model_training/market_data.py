"""Binance USD-M BTCUSDT market data for research notebooks (standard library HTTP only).

  load_klines(csv, interval)   historical CSV (any column layout) + REST top-up to the last closed candle
  fetch_klines(...)            raw klines in a time range (all fields)
  fetch_metric(kind, ...)      last ~30 days of futures statistics (Binance keeps no older data):
                                 ratio     global long/short account ratio
                                 top_ratio top-trader long/short POSITION ratio
                                 taker     taker buy/sell volume ratio
                                 oi        open interest (contracts and USD value)
  fetch_funding(...)           settled funding rates
  metrics_frame(...)           all statistics joined on the 5m grid, aligned to the time they became KNOWN

Every statistic is stamped with `known_ms` = the end of its period (ratio / top_ratio / oi timestamps are
period ends, taker timestamps are period starts). A decision taken at the close of a 5m bar may only use
rows with known_ms <= that close; `metrics_frame` adds one extra period of lag for safety by default.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://fapi.binance.com"
STEPS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}
KLINE_COLS = ["open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms", "quote_volume", "trades",
              "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
METRIC_PATHS = {"ratio": "/futures/data/globalLongShortAccountRatio",
                "top_ratio": "/futures/data/topLongShortPositionRatio",
                "taker": "/futures/data/takerlongshortRatio",
                "oi": "/futures/data/openInterestHist"}
END_STAMPED = ("ratio", "top_ratio", "oi")


def get(path: str, params: dict, retries: int = 6):
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            if exc.code not in (418, 429, 500, 502, 503, 504):
                raise RuntimeError(f"Binance {exc.code}: {exc.read()[:200]!r} for {url}") from exc
            delay = max(float(exc.headers.get("Retry-After", "0") or 0), min(60, 2 ** attempt))
        except (urllib.error.URLError, TimeoutError):
            delay = min(60, 2 ** attempt)
        time.sleep(delay)
    raise RuntimeError(f"retry budget exhausted for {url}")


def server_time_ms() -> int:
    return int(get("/fapi/v1/time", {})["serverTime"])


# --------------------------------------------------------------------------- #
# Klines
# --------------------------------------------------------------------------- #
def fetch_klines(interval: str, start_ms: int, end_ms: int, pause: float = 0.15) -> pd.DataFrame:
    """Closed klines with open time in [start_ms, end_ms)."""
    step, rows, cursor = STEPS[interval], [], start_ms
    while cursor < end_ms:
        batch = get("/fapi/v1/klines", {"symbol": "BTCUSDT", "interval": interval, "startTime": cursor,
                                        "endTime": end_ms - 1, "limit": 1500})
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + step
        time.sleep(pause)
    df = pd.DataFrame(rows, columns=KLINE_COLS).drop(columns=["ignore", "close_time_ms"])
    df = df.apply(pd.to_numeric)
    return df[(df["open_time_ms"] >= start_ms) & (df["open_time_ms"] < end_ms)].reset_index(drop=True)


def load_klines(csv_path: str | Path, interval: str, top_up: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Historical CSV + REST top-up until the last CLOSED candle. Missing columns of old CSV layouts are kept
    absent (only OHLCV is required). Returns columns open_time_ms, open, high, low, close, volume[, ...]."""
    df = pd.read_csv(csv_path)
    if "open_time_ms" not in df:
        df["open_time_ms"] = pd.to_datetime(df["open_time_utc"], utc=True).astype("int64") // 1_000_000
    df = df.drop(columns=[c for c in ("open_time_utc",) if c in df])
    if top_up:
        step = STEPS[interval]
        end = server_time_ms() // step * step                  # first still-open candle
        start = int(df["open_time_ms"].max()) + step
        if start < end:
            new = fetch_klines(interval, start, end)
            new = new[[c for c in new.columns if c in df.columns or c == "open_time_ms"]]
            df = pd.concat([df, new], ignore_index=True)
            if verbose:
                print(f"{interval}: +{len(new)} candles from REST (to {pd.to_datetime(end, unit='ms', utc=True)})")
    return df.drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Futures statistics (last ~30 days only) and funding
# --------------------------------------------------------------------------- #
def fetch_metric(kind: str, period: str = "5m", days: float = 29.5, end_ms: int | None = None,
                 pause: float = 0.15) -> pd.DataFrame:
    """Rows of one statistic over the last `days` (Binance limit: 30 days). Adds `known_ms`."""
    step = STEPS[period]
    end_ms = end_ms or server_time_ms()
    start = end_ms - int(days * 86_400_000)
    rows, cursor = [], start
    while cursor < end_ms:
        page_end = min(end_ms, cursor + 500 * step)
        batch = get(METRIC_PATHS[kind], {"symbol": "BTCUSDT", "period": period, "startTime": cursor,
                                         "endTime": page_end - 1, "limit": 500})
        rows.extend(batch or [])
        cursor = page_end
        time.sleep(pause)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop(columns=["symbol"], errors="ignore").drop_duplicates("timestamp")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["known_ms"] = df["timestamp"] + (0 if kind in END_STAMPED else step)
    return df.sort_values("known_ms").reset_index(drop=True)


def fetch_funding(start_ms: int, end_ms: int | None = None) -> pd.DataFrame:
    end_ms = end_ms or server_time_ms()
    rows, cursor = [], start_ms
    while cursor < end_ms:
        batch = get("/fapi/v1/fundingRate", {"symbol": "BTCUSDT", "startTime": cursor, "endTime": end_ms - 1,
                                              "limit": 1000})
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1]["fundingTime"]) + 1
        time.sleep(0.15)
    if not rows:                                         # nothing settled since start_ms
        return pd.DataFrame({"funding_time_ms": pd.Series(dtype="int64"), "funding_rate": pd.Series(dtype="float64")})
    df = pd.DataFrame(rows)
    return pd.DataFrame({"funding_time_ms": df["fundingTime"].astype("int64"),
                         "funding_rate": pd.to_numeric(df["fundingRate"])})


def metrics_frame(grid_close_ms, period: str = "5m", days: float = 29.5, safety_lag_periods: int = 1,
                  frames: dict | None = None) -> pd.DataFrame:
    """Join every statistic onto decision times (5m bar CLOSE times, ms) as of `known_ms` <= time - lag.

    frames: pre-fetched {kind: DataFrame} (e.g. from a cache); missing kinds are fetched."""
    frames = dict(frames or {})
    for kind in METRIC_PATHS:
        if kind not in frames:
            frames[kind] = fetch_metric(kind, period, days)
    lag = safety_lag_periods * STEPS[period]
    out = pd.DataFrame({"t": pd.Series(grid_close_ms, dtype="int64")}).sort_values("t")
    out["asof"] = out["t"] - lag
    rename = {"ratio": {"longShortRatio": "acct_ls_ratio", "longAccount": "acct_long"},
              "top_ratio": {"longShortRatio": "top_ls_ratio", "longAccount": "top_long"},
              "taker": {"buySellRatio": "taker_bs_ratio", "buyVol": "taker_buy", "sellVol": "taker_sell"},
              "oi": {"sumOpenInterest": "oi", "sumOpenInterestValue": "oi_usd"}}
    for kind, cols in rename.items():
        f = frames[kind]
        if f is None or not len(f):
            continue
        f = f[["known_ms", *cols]].rename(columns=cols).sort_values("known_ms")
        out = pd.merge_asof(out, f, left_on="asof", right_on="known_ms", direction="backward").drop(columns="known_ms")
    return out.drop(columns="asof").set_index("t")
