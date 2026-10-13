"""Đánh giá đầy đủ cho chiến lược H1 (touch + hướng B, XGBoost): overfit, model có học thật không, tối ưu tham số
(chỉ dùng validation), lưới cách giao dịch / ngưỡng chọn theo vùng ổn định, và so với giao dịch ngẫu nhiên.

Giao thức (giống strategy_h1): fold năm Y: train = nhãn kết thúc trước 1/1 của Y-1; Y-1 = 60% đầu dừng sớm (es)
+ 40% sau calibration (cal); test = Y.
  - Tối ưu tham số model: chọn theo AUC trên cal (năm Y-1), KHÔNG nhìn test.
  - Chọn cách giao dịch / ngưỡng: chỉ dùng test 2023, 2024, 2025 (mỗi năm là ngoài mẫu với model của fold đó).
  - 2026 chỉ báo cáo sau cùng (lưu ý: các phiên trước đã từng xem 2026, nên không còn là dữ liệu chưa nhìn).

Dùng (xem evaluate_strategy_h1.ipynb):
    import evaluate_h1 as ev
    sd = st.build_data(st.load_1h(cfg), positioning, cfg)
    ev.overfit_table(sd, cfg)              # AUC / logloss train - es - cal - test
    ev.permutation_null(sd, cfg)           # train lại với nhãn xáo trộn
    ...
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

import modelv2 as mv2
import strategy_h1 as st
from modelv2 import njit

SEL_YEARS = (2023, 2024, 2025)


# --------------------------------------------------------------------------- #
# Fit (cho phép thay nhãn, tập cột, tham số)
# --------------------------------------------------------------------------- #
def split(sd: st.StrategyData, year: int, cfg: st.StrategyConfig) -> dict:
    tr, va, te = mv2.fold_ids(sd.ds, year)
    es, cal = mv2.split_validation(sd.ds, va, cfg.val_es_frac)
    return {"tr": tr, "es": es, "cal": cal, "te": te}


def labels(sd: st.StrategyData):
    y = sd.ds.labels["y_touch"].astype(int)
    first = sd.ds.labels["first"]
    touched = (y == 1) & (first != 0)
    up = (first == 1).astype(int)
    return y, touched, up


def fit_touch(sd, S, cfg, y=None, cols=None):
    """XGBoost touch như mv2.fit_models (trọng số thời gian, dừng sớm trên es). Trả về (model, số cây)."""
    tc = cfg.touch_cfg()
    y = labels(sd)[0] if y is None else y
    X = sd.ds.x if cols is None else sd.ds.x[:, cols]
    w = mv2._time_weights(sd.ds, S["tr"], int(sd.ds.timestamp_ns[S["tr"]].max()), tc)
    m = mv2.make_xgb(tc.touch, tc, cfg.seed)
    m.fit(X[S["tr"]], y[S["tr"]], sample_weight=w, eval_set=[(X[S["es"]], y[S["es"]])], verbose=False)
    return m, int(m.best_iteration) + 1


def fit_dir(sd, S, cfg, up=None, cols=None):
    """Model hướng B trên các mẫu đã chạm (train / es). Trả về (model, số cây)."""
    _, touched, up0 = labels(sd)
    up = up0 if up is None else up
    cols = st.dir_cols(sd, cfg) if cols is None else np.asarray(cols)
    X = sd.x_dir[:, cols]
    a, e = S["tr"][touched[S["tr"]]], S["es"][touched[S["es"]]]
    m = st._dir_xgb(cfg)
    m.fit(X[a], up[a], eval_set=[(X[e], up[e])], verbose=False)
    m._cols = cols
    return m, int(m.best_iteration) + 1


def _pdir(m, X_full, rounds=None):
    """Dự báo model B từ ma trận x_dir đầy đủ (tự chọn cột của model)."""
    return _p(m, X_full[:, m._cols], rounds)


def _p(m, X, rounds=None):
    if rounds is None:
        return m.predict_proba(X)[:, 1]
    return m.predict_proba(X, iteration_range=(0, rounds))[:, 1]


def _auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


# --------------------------------------------------------------------------- #
# 1. Overfit: train - es - cal - test
# --------------------------------------------------------------------------- #
def overfit_table(sd, cfg, years=(2023, 2024, 2025, 2026)) -> pd.DataFrame:
    y, touched, up = labels(sd)
    rows = []
    for Y in years:
        S = split(sd, Y, cfg)
        mt, nt = fit_touch(sd, S, cfg)
        md, nd = fit_dir(sd, S, cfg)
        for part in ("tr", "es", "cal", "te"):
            i = S[part]
            pt = _p(mt, sd.ds.x[i])
            j = i[touched[i]]
            pd_ = _pdir(md, sd.x_dir[j])
            rows.append({"năm test": Y, "tập": {"tr": "train", "es": "dừng sớm", "cal": "calibration", "te": "TEST"}[part],
                         "mẫu": len(i), "touch AUC": _auc(y[i], pt), "touch logloss": log_loss(y[i], pt, labels=[0, 1]),
                         "mẫu đã chạm": len(j), "B AUC": _auc(up[j], pd_),
                         "B logloss": log_loss(up[j], pd_, labels=[0, 1]), "cây touch": nt, "cây B": nd})
    return pd.DataFrame(rows)


def learning_curve(sd, cfg, year: int = 2025, step: int = 20) -> pd.DataFrame:
    """AUC theo số cây (không dừng sớm, tối đa n_estimators) trên train / es / test cho touch và B."""
    y, touched, up = labels(sd)
    S = split(sd, year, cfg)
    big_t = replace(cfg, touch_xgb={**cfg.touch_xgb, "early_stopping_rounds": 10**6, "n_estimators": 600})
    big_d = replace(cfg, dir_xgb={**cfg.dir_xgb, "early_stopping_rounds": 10**6, "n_estimators": 800})
    mt, _ = fit_touch(sd, S, big_t)
    md, _ = fit_dir(sd, S, big_d)
    rows = []
    for r in range(step, 801, step):
        row = {"số cây": r}
        for part in ("tr", "es", "te"):
            i = S[part]
            if r <= 600:
                row[f"touch {part}"] = _auc(y[i], _p(mt, sd.ds.x[i], r))
            j = i[touched[i]]
            row[f"B {part}"] = _auc(up[j], _pdir(md, sd.x_dir[j], r))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Model có học thật không
# --------------------------------------------------------------------------- #
def _permute(y, up, ids, ts_ns, mode, rng):
    """Xáo nhãn trên tập ids. mode 'iid': xáo từng mẫu. mode 'ngày': hoán đổi nguyên ngày lịch, giữ đúng giờ trong
    ngày (giữ mùa vụ theo giờ và tự tương quan, chỉ phá liên hệ giữa feature và tương lai)."""
    yp, upp = y.copy(), up.copy()
    if mode == "iid":
        src = rng.permutation(ids)
        yp[ids], upp[ids] = y[src], up[src]
        return yp, upp
    day = ts_ns[ids] // 86_400_000_000_000
    hour = (ts_ns[ids] // 3_600_000_000_000) % 24
    lut = {(d, h): i for d, h, i in zip(day, hour, ids)}
    ud = np.unique(day)
    mp = dict(zip(ud, rng.permutation(ud)))
    for d, h, i in zip(day, hour, ids):
        j = lut.get((mp[d], h))
        if j is not None:
            yp[i], upp[i] = y[j], up[j]
    return yp, upp


def permutation_null(sd, cfg, years=SEL_YEARS, n_perm: int = 30, mode: str = "ngày", seed: int = 0) -> pd.DataFrame:
    """Train lại với nhãn train / es bị xáo trộn (mode 'iid' hoặc 'ngày', xem _permute), đo AUC test.
    p-value = (1 + số lần AUC xáo >= AUC thật) / (1 + n_perm)."""
    rng = np.random.default_rng(seed)
    y, touched, up = labels(sd)
    rows = []
    for Y in years:
        S = split(sd, Y, cfg)
        te = S["te"]
        jt = te[touched[te]]
        real_t = _auc(y[te], _p(fit_touch(sd, S, cfg)[0], sd.ds.x[te]))
        real_d = _auc(up[jt], _pdir(fit_dir(sd, S, cfg)[0], sd.x_dir[jt]))
        nt, nd = [], []
        fit_ids = np.r_[S["tr"], S["es"]]
        for _ in range(n_perm):
            yp, upp = _permute(y, up, fit_ids, sd.ds.timestamp_ns, mode, rng)
            nt.append(_auc(y[te], _p(fit_touch(sd, S, cfg, y=yp)[0], sd.ds.x[te])))
            nd.append(_auc(up[jt], _pdir(fit_dir(sd, S, cfg, up=upp)[0], sd.x_dir[jt])))
        nt, nd = np.array(nt), np.array(nd)
        rows.append({"năm": Y, "kiểu xáo": mode, "touch AUC thật": real_t, "touch AUC xáo (TB)": nt.mean(),
                     "touch AUC xáo (max)": nt.max(), "touch p": (1 + (nt >= real_t).sum()) / (1 + n_perm),
                     "B AUC thật": real_d, "B AUC xáo (TB)": nd.mean(), "B AUC xáo (max)": nd.max(),
                     "B p": (1 + (nd >= real_d).sum()) / (1 + n_perm)})
    return pd.DataFrame(rows)


def _impute(X, med):
    X = np.where(np.isfinite(X), X, np.nan)
    return np.where(np.isnan(X), med, X)


def baselines(sd, cfg, years=SEL_YEARS) -> pd.DataFrame:
    """So XGBoost với: (a) hồi quy logistic trên cùng feature, (b) feature đơn tốt nhất (chọn trên train)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    y, touched, up = labels(sd)
    rows = []
    for Y in years:
        S = split(sd, Y, cfg)
        for task in ("touch", "B"):
            X = sd.ds.x if task == "touch" else sd.x_dir
            names = sd.ds.names if task == "touch" else sd.dir_names
            lab = y if task == "touch" else up
            tr = S["tr"] if task == "touch" else S["tr"][touched[S["tr"]]]
            te = S["te"] if task == "touch" else S["te"][touched[S["te"]]]
            m, _ = (fit_touch if task == "touch" else fit_dir)(sd, S, cfg)
            med = np.nan_to_num(np.nanmedian(np.where(np.isfinite(X[tr]), X[tr], np.nan), 0))
            sc = StandardScaler().fit(_impute(X[tr], med))
            lr = LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(_impute(X[tr], med)), lab[tr])
            p_lr = lr.predict_proba(sc.transform(_impute(X[te], med)))[:, 1]
            best, bj, bs = 0.5, 0, 1
            for k in range(X.shape[1]):
                a = _auc(lab[tr], np.nan_to_num(_impute(X[tr, k:k + 1], med[k:k + 1])[:, 0]))
                if np.isfinite(a) and abs(a - 0.5) > abs(best - 0.5):
                    best, bj, bs = a, k, (1 if a >= 0.5 else -1)
            p_one = bs * _impute(X[te, bj:bj + 1], med[bj:bj + 1])[:, 0]
            p_x = _p(m, X[te]) if task == "touch" else _pdir(m, X[te])
            rows.append({"năm": Y, "model": task, "XGBoost": _auc(lab[te], p_x),
                         "logistic": _auc(lab[te], p_lr), "feature đơn tốt nhất": _auc(lab[te], p_one),
                         "tên feature đơn": names[bj]})
    return pd.DataFrame(rows)


def dir_feature_groups(sd) -> dict:
    n_t = sd.ds.x.shape[1]
    names = sd.dir_names
    pos = [i for i, n in enumerate(names) if n.startswith("p_")]
    dire = [i for i in range(n_t, len(names)) if i not in pos]
    return {"giá (feature touch)": list(range(n_t)), "có hướng": dire, "vị thế": pos}


def dir_ablation(sd, cfg, years=SEL_YEARS) -> pd.DataFrame:
    """AUC test của model B khi chỉ dùng một phần nhóm feature."""
    _, touched, up = labels(sd)
    G = dir_feature_groups(sd)
    sets = {"tất cả": None, "chỉ giá": G["giá (feature touch)"],
            "giá + có hướng (không vị thế)": G["giá (feature touch)"] + G["có hướng"],
            "chỉ vị thế": G["vị thế"], "có hướng + vị thế": G["có hướng"] + G["vị thế"]}
    rows = []
    for Y in years:
        S = split(sd, Y, cfg)
        jt = S["te"][touched[S["te"]]]
        row = {"năm": Y}
        for nm, cols in sets.items():
            m, _ = fit_dir(sd, S, cfg, cols=np.arange(sd.x_dir.shape[1]) if cols is None else cols)
            row[nm] = _auc(up[jt], _pdir(m, sd.x_dir[jt]))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"năm": "TB", **df.drop(columns="năm").mean().to_dict()}
    return df


def group_permutation_importance(sd, cfg, years=SEL_YEARS, n_rep: int = 5, seed: int = 0) -> pd.DataFrame:
    """Giảm AUC test khi xáo cả một nhóm feature (touch: nhóm của modelv2; B: giá / có hướng / vị thế)."""
    rng = np.random.default_rng(seed)
    y, touched, up = labels(sd)
    tg = {}
    for i, n in enumerate(sd.ds.names):
        tg.setdefault(sd.ds.feature_groups[n], []).append(i)
    rows = []
    for Y in years:
        S = split(sd, Y, cfg)
        te = S["te"]
        jt = te[touched[te]]
        mt, _ = fit_touch(sd, S, cfg)
        md, _ = fit_dir(sd, S, cfg)
        for task, m, X, lab, ids, groups in (("touch", mt, sd.ds.x, y, te, tg),
                                             ("B", md, sd.x_dir, up, jt, dir_feature_groups(sd))):
            f = (lambda mm, XX: _p(mm, XX)) if task == "touch" else _pdir
            base = _auc(lab[ids], f(m, X[ids]))
            for g, cols in groups.items():
                d = []
                for _ in range(n_rep):
                    Xp = X[ids].copy()
                    Xp[:, cols] = Xp[rng.permutation(len(ids))][:, cols]
                    d.append(base - _auc(lab[ids], f(m, Xp)))
                rows.append({"năm": Y, "model": task, "nhóm": g, "số feature": len(cols), "AUC giảm": float(np.mean(d))})
    out = pd.DataFrame(rows)
    return out.pivot_table(index=["model", "nhóm", "số feature"], columns="năm", values="AUC giảm").assign(
        TB=lambda d: d.mean(1)).sort_values(["model", "TB"], ascending=[True, False])


def decile_tables(sd, cfg, years=SEL_YEARS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Touch: tỉ lệ chạm thực tế theo nhóm p_touch (10 nhóm, test gộp các năm).
    B: tỉ lệ đúng hướng theo nhóm độ tự tin |p_lên - 0.5|, trên các giờ touch top 20% đã chạm."""
    y, touched, up = labels(sd)
    T, B = [], []
    for Y in years:
        S = split(sd, Y, cfg)
        te, cal = S["te"], S["cal"]
        mt, _ = fit_touch(sd, S, cfg)
        md, _ = fit_dir(sd, S, cfg)
        pt, pt_cal = _p(mt, sd.ds.x[te]), _p(mt, sd.ds.x[cal])
        thr = np.quantile(pt_cal, 1 - cfg.touch_top)
        T.append(pd.DataFrame({"năm": Y, "p": pt, "chạm": y[te], "rank": pd.Series(pt).rank(pct=True).to_numpy()}))
        hi = (pt >= thr) & touched[te]
        pu = _pdir(md, sd.x_dir[te[hi]])
        B.append(pd.DataFrame({"năm": Y, "conf": np.abs(pu - 0.5), "đúng": ((pu > 0.5) == (up[te[hi]] == 1)).astype(int)}))
    T, B = pd.concat(T), pd.concat(B)
    T["nhóm"] = T.groupby("năm")["rank"].transform(lambda r: np.minimum((r * 10).astype(int), 9) + 1)
    tt = T.groupby("nhóm").agg(p_TB=("p", "mean"), tỉ_lệ_chạm=("chạm", "mean"), mẫu=("chạm", "size"))
    B["nhóm"] = B.groupby("năm")["conf"].transform(lambda c: np.minimum((c.rank(pct=True) * 10).astype(int), 9) + 1)
    bb = B.groupby("nhóm").agg(conf_TB=("conf", "mean"), tỉ_lệ_đúng=("đúng", "mean"), mẫu=("đúng", "size"))
    by_year = B.assign(top10=B["nhóm"] == 10).groupby(["năm", "top10"])["đúng"].mean().unstack()
    bb.attrs["theo năm (top 10% vs còn lại)"] = by_year
    return tt, bb


# --------------------------------------------------------------------------- #
# 3. Tối ưu tham số (chỉ theo validation)
# --------------------------------------------------------------------------- #
TOUCH_GRID = {"max_depth": [3, 4, 6], "min_child_weight": [20, 40, 100], "colsample_bytree": [0.4, 0.6, 0.8]}
DIR_GRID = {"dir_features": ["all", "dir_pos"], "max_depth": [2, 3, 4], "min_child_weight": [20, 50, 150],
            "colsample_bytree": [0.4, 0.8], "reg_lambda": [1, 10]}


def tune(sd, cfg, task: str, grid: dict, years=SEL_YEARS) -> pd.DataFrame:
    """Quét lưới tham số; tiêu chí = AUC trên calibration (40% cuối năm Y-1) trung bình 3 fold.
    Cột test chỉ để xem, KHÔNG dùng để chọn."""
    y, touched, up = labels(sd)
    keys = list(grid)
    splits = {Y: split(sd, Y, cfg) for Y in years}
    rows = []
    for vals in itertools.product(*grid.values()):
        over = dict(zip(keys, vals))
        row = dict(over)
        for Y, S in splits.items():
            if task == "touch":
                c = replace(cfg, touch_xgb={**cfg.touch_xgb, **over})
                m, n = fit_touch(sd, S, c)
                row[f"cal {Y}"] = _auc(y[S["cal"]], _p(m, sd.ds.x[S["cal"]]))
                row[f"test {Y}"] = _auc(y[S["te"]], _p(m, sd.ds.x[S["te"]]))
            else:
                o2 = {k: v for k, v in over.items() if k != "dir_features"}
                c = replace(cfg, dir_xgb={**cfg.dir_xgb, **o2}, dir_features=over.get("dir_features", cfg.dir_features))
                m, n = fit_dir(sd, S, c)
                jc, jt = S["cal"][touched[S["cal"]]], S["te"][touched[S["te"]]]
                row[f"cal {Y}"] = _auc(up[jc], _pdir(m, sd.x_dir[jc]))
                row[f"test {Y}"] = _auc(up[jt], _pdir(m, sd.x_dir[jt]))
            row[f"cây {Y}"] = n
        rows.append(row)
    df = pd.DataFrame(rows)
    df["cal TB"] = df[[f"cal {Y}" for Y in years]].mean(1)
    df["test TB (chỉ xem)"] = df[[f"test {Y}" for Y in years]].mean(1)
    return df.sort_values("cal TB", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. Lưới cách giao dịch / ngưỡng
# --------------------------------------------------------------------------- #
def fold_predictions(sd, cfg, years=(2023, 2024, 2025, 2026)) -> dict:
    """Fit mỗi fold một lần; lưu dự báo test (mọi giờ) + dự báo cal để tính ngưỡng cho bất kỳ top% nào."""
    out = {}
    for Y in years:
        m = st.fit_fold(sd, Y, cfg)
        S = split(sd, Y, cfg)
        P = st.predict(m, sd, S["te"]).assign(year=Y)
        out[Y] = {"P": P, "pt_cal": st.p_touch_of(m["touch"], sd.ds.x[S["cal"]]),
                  "pu_cal": m["dir"].predict_proba(sd.x_dir[S["cal"]][:, m["dir_cols"]])[:, 1], "models": m}
    return out


def signal_mask(fp: dict, Y: int, touch_top: float, dir_top: float) -> np.ndarray:
    f = fp[Y]
    thr_t = np.quantile(f["pt_cal"], 1 - touch_top)
    hi = f["pt_cal"] >= thr_t
    thr_c = np.quantile(np.abs(f["pu_cal"] - 0.5)[hi], 1 - dir_top)
    P = f["P"]
    return ((P.p_touch >= thr_t) & (P.conf >= thr_c)).to_numpy()


class Sim5m:
    """Mô phỏng nhanh trên nến 5m (giống st.backtest_m5, entry_mode next_open)."""

    def __init__(self, candles_5m: pd.DataFrame):
        d5 = mv2.prepare_candles(candles_5m, 5)
        self.o, self.h, self.l, self.c = (d5[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
        self.t5 = mv2._ns_int(d5["timestamp"])

    def run(self, sig_ns, side, b, tp_mult, sl_mult, anomaly, fee=0.001, max_hold_h=720):
        start = np.searchsorted(self.t5, sig_ns + 3600 * 10**9)
        ok = start < len(self.t5) - 1
        R = np.full(len(sig_ns), np.nan)
        ex = np.full(len(sig_ns), -1, np.int64)
        kind = np.zeros(len(sig_ns), np.int64)
        if ok.any():
            r, e, k = st._sim_m5(self.o, self.h, self.l, self.c, start[ok], side[ok].astype(np.float64),
                                 tp_mult * b[ok], sl_mult * b[ok], max_hold_h * 12, anomaly)
            R[ok] = r - fee / (sl_mult * b[ok])
            ex[ok] = self.t5[e]
            kind[ok] = k
        return R, ex, kind


@njit(cache=True)
def _portfolio(sig_ns, ex_ns, max_pos, gap_ns):
    n = len(sig_ns)
    keep = np.zeros(n, np.bool_)
    opens = np.empty(max_pos + 1, np.int64)
    k_open = 0
    last = -(10**18)
    for j in range(n):
        if ex_ns[j] < 0:
            continue
        t = sig_ns[j]
        m = 0
        for q in range(k_open):
            if opens[q] >= t:
                opens[m] = opens[q]
                m += 1
        k_open = m
        if k_open >= max_pos or t - last < gap_ns:
            continue
        keep[j] = True
        opens[k_open] = ex_ns[j]
        k_open += 1
        last = t
    return keep


def metrics(R, ex_ns, years=SEL_YEARS) -> dict:
    o = np.argsort(ex_ns)
    R, ex_ns = R[o], ex_ns[o]
    cum = np.r_[0, np.cumsum(R)]
    t = pd.to_datetime(ex_ns, utc=True)
    yr = pd.Series(R).groupby(t.year).sum()
    q = pd.Series(R).groupby(t.to_period("Q").astype(str)).sum()
    ry = {Y: float(yr.get(Y, 0.0)) for Y in years}
    dd = float((np.maximum.accumulate(cum) - cum).max())
    tot = float(R.sum())
    return {"lệnh": len(R), "thắng": float((R > 0).mean()) if len(R) else np.nan,
            "R/lệnh": float(R.mean()) if len(R) else np.nan, "tổng R": tot, "sụt giảm R": dd,
            "tổng R / sụt giảm": tot / max(dd, 1.0), "năm kém nhất R": min(ry.values()),
            "quý âm": int((q < 0).sum()), **{f"R {Y}": v for Y, v in ry.items()}}


GRID = {"touch_top": [0.1, 0.2, 0.3, 0.5, 1.0], "dir_top": [0.03, 0.05, 0.1, 0.15, 0.2, 0.3],
        "tp_mult": [0.75, 1.0, 1.25], "sl_mult": [0.5, 0.6, 0.75, 1.0], "anomaly": [0.0, 0.0075, 0.01],
        "max_positions": [1, 3, 5, 8]}


def trade_grid(sd, fp: dict, sim: Sim5m, grid: dict = GRID, years=SEL_YEARS, fee: float = 0.001) -> pd.DataFrame:
    P = pd.concat([fp[Y]["P"] for Y in years], ignore_index=True)
    ns_all = mv2._ns_int(P.timestamp)
    order = np.argsort(ns_all, kind="stable")
    P, ns_all = P.iloc[order].reset_index(drop=True), ns_all[order]
    masks = {}
    for tt, dt in itertools.product(grid["touch_top"], grid["dir_top"]):
        masks[(tt, dt)] = np.concatenate([signal_mask(fp, Y, tt, dt) for Y in years])[order]
    union = np.any(np.stack(list(masks.values())), 0)
    idx = np.where(union)[0]
    side, b, ns = P.side.to_numpy()[idx], P.b.to_numpy()[idx], ns_all[idx]
    rows = []
    for tp, sl, an in itertools.product(grid["tp_mult"], grid["sl_mult"], grid["anomaly"]):
        R, ex, _ = sim.run(ns, side, b, tp, sl, an, fee)
        for (tt, dt), mk in masks.items():
            sel = mk[idx]
            for mp in grid["max_positions"]:
                kk = _portfolio(ns[sel], ex[sel], mp, 3600 * 10**9)
                rows.append({"touch_top": tt, "dir_top": dt, "tp_mult": tp, "sl_mult": sl, "anomaly": an,
                             "max_positions": mp, **metrics(R[sel][kk], ex[sel][kk], years)})
    return pd.DataFrame(rows)


def plateau(df: pd.DataFrame, axes=tuple(GRID), score: str = "tổng R / sụt giảm") -> pd.DataFrame:
    """Điểm vùng ổn định = trung bình 'score' trên ô đó và mọi ô lân cận (lệch tối đa 1 bậc ở mỗi trục),
    cộng thêm tỉ lệ ô lân cận có mọi năm dương. Chọn ô có điểm vùng cao nhất trong các ô đủ điều kiện."""
    d = df.copy()
    lev = {a: {v: i for i, v in enumerate(sorted(d[a].unique()))} for a in axes}
    I = np.stack([d[a].map(lev[a]).to_numpy() for a in axes], 1)
    sc = d[score].to_numpy()
    pos = (d["năm kém nhất R"] > 0).to_numpy()
    key = {tuple(r): i for i, r in enumerate(I)}
    nb_mean, nb_pos, nb_n = np.empty(len(d)), np.empty(len(d)), np.empty(len(d), int)
    for i, r in enumerate(I):
        js = [key[tuple(r + np.array(off))] for off in itertools.product(*[(-1, 0, 1)] * len(axes))
              if tuple(r + np.array(off)) in key]
        nb_mean[i], nb_pos[i], nb_n[i] = sc[js].mean(), pos[js].mean(), len(js)
    d["điểm vùng"] = nb_mean
    d["tỉ lệ lân cận mọi năm dương"] = nb_pos
    d["số ô lân cận"] = nb_n
    d["đủ điều kiện"] = pos & (nb_pos >= 0.8)
    return d.sort_values(["đủ điều kiện", "điểm vùng"], ascending=False).reset_index(drop=True)


def rank_stability(G: pd.DataFrame, years, fixed: dict | None = None) -> pd.DataFrame:
    """Tương quan thứ hạng (Spearman) của R từng năm giữa các cấu hình: thấp = thứ hạng cấu hình không bền,
    tức chọn 'cấu hình tốt nhất' phần lớn là chọn nhiễu."""
    d = G
    for k, v in (fixed or {}).items():
        d = d[d[k] == v]
    return d[[f"R {Y}" for Y in years]].corr("spearman")


def selection_rules(G: pd.DataFrame, axes, scores=("tổng R / sụt giảm", "tổng R", "năm kém nhất R", "R/lệnh")) -> pd.DataFrame:
    """Ô được chọn theo từng tiêu chí (cùng quy tắc vùng ổn định)."""
    key = list(GRID)
    rows = []
    for sc in scores:
        w = plateau(G, axes=axes, score=sc).iloc[0]
        rows.append({"tiêu chí": sc, **{k: w[k] for k in key}, **{k: w[k] for k in G.columns if k not in key},
                     "điểm vùng": w["điểm vùng"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. So với giao dịch ngẫu nhiên
# --------------------------------------------------------------------------- #
def random_nulls(sd, fp: dict, sim: Sim5m, conf: dict, years=SEL_YEARS, n: int = 500, seed: int = 0,
                 fee: float = 0.001) -> dict:
    """(a) cùng giờ tín hiệu, hướng ngẫu nhiên -> kiểm định model B;
    (b) cùng số tín hiệu, giờ ngẫu nhiên trong các giờ touch top%, hướng của model B -> kiểm định độ tự tin B;
    (c) giờ ngẫu nhiên trong MỌI giờ, hướng ngẫu nhiên -> kiểm định cả hệ thống.
    Các mô phỏng cùng TP / SL / thoát nhanh / số lệnh tối đa như cấu hình 'conf'."""
    rng = np.random.default_rng(seed)
    P = pd.concat([fp[Y]["P"] for Y in years], ignore_index=True)
    ns = mv2._ns_int(P.timestamp)
    o = np.argsort(ns, kind="stable")
    P, ns = P.iloc[o].reset_index(drop=True), ns[o]
    sig = np.concatenate([signal_mask(fp, Y, conf["touch_top"], conf["dir_top"]) for Y in years])[o]
    hi = np.concatenate([signal_mask(fp, Y, conf["touch_top"], 1.0) for Y in years])[o]
    b, side = P.b.to_numpy(), P.side.to_numpy().astype(np.float64)
    tp, sl, an, mp = conf["tp_mult"], conf["sl_mult"], conf["anomaly"], conf["max_positions"]
    Rp, exp_, _ = sim.run(ns, np.ones(len(ns)), b, tp, sl, an, fee)
    Rm, exm, _ = sim.run(ns, -np.ones(len(ns)), b, tp, sl, an, fee)

    def run(mask, s):
        R = np.where(s > 0, Rp, Rm)[mask]
        ex = np.where(s > 0, exp_, exm)[mask]
        kk = _portfolio(ns[mask], ex, mp, 3600 * 10**9)
        return metrics(R[kk], ex[kk], years)

    real = run(sig, side)
    out = {"thật": real}
    k = int(sig.sum())
    allowed = np.isfinite(Rp) & np.isfinite(Rm)
    for nm in ("a", "b", "c"):
        res = []
        for _ in range(n):
            if nm == "a":
                m, s = sig, rng.choice([-1.0, 1.0], len(ns))
            elif nm == "b":
                m = np.zeros(len(ns), bool)
                m[rng.choice(np.where(hi & allowed)[0], k, replace=False)] = True
                s = side
            else:
                m = np.zeros(len(ns), bool)
                m[rng.choice(np.where(allowed)[0], k, replace=False)] = True
                s = rng.choice([-1.0, 1.0], len(ns))
            res.append(run(m, s))
        out[nm] = pd.DataFrame(res)
    names = {"a": "(a) cùng giờ, hướng ngẫu nhiên", "b": "(b) giờ touch cao ngẫu nhiên, hướng model B",
             "c": "(c) giờ ngẫu nhiên, hướng ngẫu nhiên"}
    summ = []
    for nm in ("a", "b", "c"):
        d = out[nm]
        summ.append({"null": names[nm], "tổng R TB": d["tổng R"].mean(), "tổng R 95%": d["tổng R"].quantile(0.95),
                     "R/lệnh TB": d["R/lệnh"].mean(), "p (tổng R)": (1 + (d["tổng R"] >= real["tổng R"]).sum()) / (1 + n),
                     "p (R/lệnh)": (1 + (d["R/lệnh"] >= real["R/lệnh"]).sum()) / (1 + n)})
    out["tóm tắt"] = pd.DataFrame(summ)
    return out
