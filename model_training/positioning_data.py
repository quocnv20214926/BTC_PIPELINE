"""Bảng vị thế 5m (open interest, tỉ lệ long/short, premium index) khớp 1-1 với lưới nến 5m.

Dùng chung cho TRAIN (dựng lại historical_data/BTCUSDT_positioning_5m.csv từ dữ liệu data.binance.vision)
và LIVE (dựng cùng bảng từ REST Binance trong pipeline), để model nhìn thấy đúng một quy ước thời gian.

Ánh xạ cột (đã kiểm định: khớp 100% với bảng train):
    oi           = sum_open_interest                 (REST /futures/data/openInterestHist: sumOpenInterest)
    oi_usd       = sum_open_interest_value           (REST: sumOpenInterestValue)
    top_acct_ls  = count_toptrader_long_short_ratio  (REST /futures/data/topLongShortAccountRatio: longShortRatio)
    top_pos_ls   = sum_toptrader_long_short_ratio    (REST /futures/data/topLongShortPositionRatio: longShortRatio)
    acct_ls      = count_long_short_ratio            (REST /futures/data/globalLongShortAccountRatio: longShortRatio)
    prem_*       = /fapi/v1/premiumIndexKlines 5m (open, high, low, close, count)

Quy ước thời gian (kiểm định bằng tương quan taker ratio / OI với klines):
    - trước 2024-03-04 00:00 UTC: metric create_time T mô tả [T-5m, T)  -> gán cho nến 5m mở lúc T-5m
    - từ 2024-03-04 00:00 UTC:    metric create_time T mô tả [T, T+5m)  -> gán cho nến 5m mở lúc T
    - premium kline mở lúc t                                         -> gán cho nến 5m mở lúc t
  Giá trị của dòng t được biết lúc nến 5m đó ĐÓNG (t + 5m). Thiếu <= 3 nến: giữ giá trị gần nhất (chỉ nhìn
  về quá khứ), số nến đã giữ ghi ở *_age_bars; thiếu lâu hơn -> NaN.

LIVE: timestamp REST được coi là create_time của data.binance.vision (POSITIONING_TS_SHIFT_MS = 0). Lệnh
`python -m pipeline check-positioning` so REST với file ngày của data.binance.vision để xác nhận độ lệch này.

Dựng lại bảng train:
    python model_training/positioning_data.py --metrics historical_data/BTCUSDT_metrics_5m.csv \
        --premium historical_data/BTCUSDT_premium_5m.csv --klines historical_data/BTCUSDT_5m_all.csv \
        --out historical_data/BTCUSDT_positioning_5m.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

STEP_MS = 300_000
CUTOVER_MS = 1_709_510_400_000            # 2024-03-04 00:00:00 UTC
FFILL_LIMIT = 3
METRIC_COLUMNS = ("oi", "oi_usd", "top_acct_ls", "top_pos_ls", "acct_ls")
VISION_MAP = {"sum_open_interest": "oi", "sum_open_interest_value": "oi_usd",
              "count_toptrader_long_short_ratio": "top_acct_ls", "sum_toptrader_long_short_ratio": "top_pos_ls",
              "count_long_short_ratio": "acct_ls"}
OUT_COLUMNS = ("open_time_ms", "oi", "oi_usd", "top_acct_ls", "top_pos_ls", "acct_ls", "prem_open", "prem_high",
               "prem_low", "prem_close", "prem_samples", "m_age_bars", "p_age_bars", "m_ok", "p_ok", "top_ok", "acct_ok")


def metric_owner_open_ms(create_ms: np.ndarray, cutover_ms: int = CUTOVER_MS) -> np.ndarray:
    """Nến 5m (open time) mà một metric có create_time này thuộc về."""
    create_ms = np.asarray(create_ms, np.int64)
    return np.where(create_ms < cutover_ms, create_ms - STEP_MS, create_ms)


def clean_metrics(m: pd.DataFrame, ref_close: pd.Series | None = None) -> pd.DataFrame:
    """Giá trị không hợp lệ -> NaN: OI / tỉ lệ <= 0; oi_usd lệch > 2% so với oi x giá (khi có giá tham chiếu)."""
    m = m.copy()
    for c in METRIC_COLUMNS:
        m[c] = pd.to_numeric(m[c], errors="coerce")
        m.loc[~(m[c] > 0), c] = np.nan
    if ref_close is not None:
        implied = m["oi"] * ref_close.reindex(m.index).to_numpy()
        bad = (m["oi_usd"] / implied - 1).abs() > 0.02
        m.loc[bad.fillna(False), "oi_usd"] = np.nan
    return m


def align(metrics: pd.DataFrame, premium: pd.DataFrame, grid_open_ms, cutover_ms: int = CUTOVER_MS,
          ref_close: pd.Series | None = None, ffill_limit: int = FFILL_LIMIT) -> pd.DataFrame:
    """metrics: create_ms + METRIC_COLUMNS (quy ước data.binance.vision); premium: open_ms, open, high, low, close,
    count. grid_open_ms: open time các nến 5m. Trả về bảng giống BTCUSDT_positioning_5m.csv."""
    grid = pd.Index(np.asarray(grid_open_ms, np.int64), name="open_time_ms")
    # 1) dòng THIẾU (không có bản ghi nguồn) được giữ giá trị gần nhất tối đa ffill_limit nến;
    # 2) sau đó giá trị không hợp lệ -> NaN (không lấp bằng giá trị cũ).
    m = metrics.copy()
    m["open_time_ms"] = metric_owner_open_ms(m["create_ms"].to_numpy(np.int64), cutover_ms)
    m = m.drop_duplicates("open_time_ms", keep="last").set_index("open_time_ms")[list(METRIC_COLUMNS)]
    have_m = pd.Series(grid.isin(m.index), index=grid)
    m = m.reindex(grid)
    m_age = _age(have_m, ffill_limit)
    m = _ffill_rows(m, have_m, ffill_limit)
    m = clean_metrics(m, ref_close.reindex(m.index) if ref_close is not None else None)

    p = premium.copy()
    p = p.drop_duplicates("open_ms", keep="last").set_index("open_ms")
    pr = pd.DataFrame({"prem_open": pd.to_numeric(p["open"]), "prem_high": pd.to_numeric(p["high"]),
                       "prem_low": pd.to_numeric(p["low"]), "prem_close": pd.to_numeric(p["close"]),
                       "prem_samples": pd.to_numeric(p["count"])}, index=p.index)
    have_p = pd.Series(grid.isin(pr.index), index=grid)
    pr = pr.reindex(grid)
    p_age = _age(have_p, ffill_limit)
    pr = _ffill_rows(pr, have_p, ffill_limit)
    for c in ("prem_open", "prem_high", "prem_low", "prem_close"):
        pr.loc[pr[c] == 0, c] = np.nan                 # đúng bằng 0 = mẫu thiếu -> giá trị gần nhất (<= 3 nến)
        pr[c] = pr[c].ffill(limit=ffill_limit).where(p_age <= ffill_limit)

    out = pd.concat([m, pr], axis=1)
    out["m_age_bars"] = m_age.where(m_age <= ffill_limit)
    out["p_age_bars"] = p_age.where(p_age <= ffill_limit)
    out["m_ok"] = out["oi"].notna().astype(int)
    out["p_ok"] = out["prem_close"].notna().astype(int)
    out["top_ok"] = out["top_pos_ls"].notna().astype(int)
    out["acct_ok"] = out["acct_ls"].notna().astype(int)
    return out.reset_index()[list(OUT_COLUMNS)]


def _ffill_rows(df: pd.DataFrame, have: pd.Series, limit: int) -> pd.DataFrame:
    """Dòng không có bản ghi nguồn lấy nguyên dòng có bản ghi gần nhất trước đó (tối đa `limit` nến)."""
    pos = pd.Series(np.where(have.to_numpy(), np.arange(len(have)), np.nan)).ffill()
    gap = np.arange(len(have)) - pos.to_numpy()
    src = np.where(np.isfinite(pos) & (gap <= limit), pos, -1).astype(np.int64)
    vals = df.to_numpy(np.float64)
    out = np.full_like(vals, np.nan)
    ok = src >= 0
    out[ok] = vals[src[ok]]
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def _age(have: pd.Series, limit: int) -> pd.Series:
    """Số nến kể từ lần có giá trị gần nhất (0 = có ngay tại nến này)."""
    idx = pd.Series(np.where(have.to_numpy(), np.arange(len(have)), np.nan), index=have.index).ffill()
    return pd.Series(np.arange(len(have)), index=have.index) - idx


# --------------------------------------------------------------------------- #
# Nguồn dữ liệu
# --------------------------------------------------------------------------- #
def vision_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """CSV metrics của data.binance.vision -> create_ms + METRIC_COLUMNS."""
    t = pd.to_datetime(df["create_time"], utc=True).dt.as_unit("ms").astype("int64")
    out = df.rename(columns=VISION_MAP)[list(METRIC_COLUMNS)].copy()
    out.insert(0, "create_ms", t.to_numpy())
    return out


def vision_premium(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"open_ms": df["open_time"].astype("int64"), "open": df["open"], "high": df["high"],
                         "low": df["low"], "close": df["close"], "count": df["count"]})


def rest_metrics(oi: list[dict], top_acct: list[dict], top_pos: list[dict], acct: list[dict],
                 ts_shift_ms: int = 0) -> pd.DataFrame:
    """Dòng REST (timestamp, ...) của bốn endpoint -> create_ms + METRIC_COLUMNS (quy ước data.binance.vision).
    ts_shift_ms: create_time = timestamp REST + ts_shift_ms (mặc định 0, xác nhận bằng check-positioning)."""
    def frame(rows, cols):
        if not rows:
            return pd.DataFrame(columns=["create_ms", *cols.values()]).set_index("create_ms")
        d = pd.DataFrame(rows)
        d["create_ms"] = pd.to_numeric(d["timestamp"]).astype("int64") + ts_shift_ms
        return d.drop_duplicates("create_ms", keep="last").set_index("create_ms")[list(cols)].rename(columns=cols)
    parts = [frame(oi, {"sumOpenInterest": "oi", "sumOpenInterestValue": "oi_usd"}),
             frame(top_acct, {"longShortRatio": "top_acct_ls"}),
             frame(top_pos, {"longShortRatio": "top_pos_ls"}),
             frame(acct, {"longShortRatio": "acct_ls"})]
    out = pd.concat(parts, axis=1).sort_index()
    for c in METRIC_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce") if c in out else np.nan
    return out.reset_index()[["create_ms", *METRIC_COLUMNS]]


def rest_premium(rows: list) -> pd.DataFrame:
    """Kline premium index REST ([open_time, open, high, low, close, ...]). Không có count -> 1 (mẫu hợp lệ)."""
    if not rows:
        return pd.DataFrame(columns=["open_ms", "open", "high", "low", "close", "count"])
    d = pd.DataFrame([r[:5] for r in rows], columns=["open_ms", "open", "high", "low", "close"])
    d["open_ms"] = d["open_ms"].astype("int64")
    for c in ("open", "high", "low", "close"):
        d[c] = pd.to_numeric(d[c])
    d["count"] = 1
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="historical_data/BTCUSDT_metrics_5m.csv")
    ap.add_argument("--premium", default="historical_data/BTCUSDT_premium_5m.csv")
    ap.add_argument("--klines", default="historical_data/BTCUSDT_5m_all.csv")
    ap.add_argument("--out", default="historical_data/BTCUSDT_positioning_5m.csv")
    a = ap.parse_args()
    k = pd.read_csv(a.klines, usecols=lambda c: c in ("open_time_ms", "close"))
    out = align(vision_metrics(pd.read_csv(a.metrics)), vision_premium(pd.read_csv(a.premium)), k["open_time_ms"],
                ref_close=k.set_index("open_time_ms")["close"])
    out.to_csv(a.out, index=False)
    print(f"{a.out}: {len(out)} dòng, OI có ở {out['m_ok'].mean():.1%}, premium {out['p_ok'].mean():.1%}")


if __name__ == "__main__":
    main()
