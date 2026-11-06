"""Hai model deep learning thay cho XGBoost trong chiến lược H1 (touch + Direction): TabM và LSTM.

Cả hai dùng ĐÚNG dữ liệu, feature, nhãn, fold walk-forward, trọng số thời gian (touch), hiệu chỉnh nhiệt độ (touch) và
dừng sớm trên tập es như XGBoost (strategy_h1.fit_fold), để so sánh công bằng. Chọn bằng StrategyConfig.model_kind.

TabM (Gorishniy và cộng sự, ICLR 2025, "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"):
  MLP + BatchEnsemble. Mỗi lớp tuyến tính có W dùng chung và vector r_i, s_i, b_i riêng của thành viên i:
  y_i = ((x_i * r_i) @ W) * s_i + b_i. k thành viên train đồng thời (loss = TB cross-entropy từng thành viên),
  dự báo = TB xác suất. Đầu vào: feature của nến t (giống XGBoost).
LSTM: đầu vào là chuỗi feature của seq_len nến gần nhất (t - seq_len + 1 .. t), chỉ dùng các nến quá khứ đã biết lúc t;
  nến thiếu (đầu dữ liệu / chỗ gap) = 0 và kênh cờ = 0. LSTM nhiều lớp -> trạng thái cuối -> lớp tuyến tính.

Tiền xử lý (chỉ học trên tập train của fold): NaN / inf -> trung vị + cột cờ thiếu, QuantileTransformer về phân phối
chuẩn, cắt ±5. Loss cross-entropy (touch: 2 lớp, Direction: số lớp của nhãn), có trọng số mẫu.
Dừng sớm theo log-loss của dự báo trên tập es, giữ trọng số của epoch tốt nhất.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

import modelv2 as mv2

H_NS = 3600 * 10**9


@dataclass
class TabMParams:
    k: int = 16                 # số thành viên BatchEnsemble
    width: int = 256
    depth: int = 3
    dropout: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 80
    patience: int = 8


@dataclass
class LSTMParams:
    seq_len: int = 24           # số nến H1 của chuỗi đầu vào
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    in_proj: int = 64           # lớp tuyến tính nén feature trước LSTM (0 = không)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 60
    patience: int = 8            # tính theo epoch
    # --- chống dừng sớm ở epoch 1-2 / giảm phương sai (mặc định = tắt, giữ đúng hành vi cũ) ---
    eval_per_epoch: int = 1      # số lần đo es mỗi epoch (dừng sớm mịn hơn)
    lr_schedule: str = "none"    # none | cosine (warmup tuyến tính rồi cosine về 5% lr)
    warmup_frac: float = 0.05    # phần tổng số bước dùng để warmup (cosine)
    label_smoothing: float = 0.0
    ema: float = 0.0             # trung bình động trọng số (vd 0.999); đo es và lưu bằng trọng số EMA
    input_noise: float = 0.0     # nhiễu Gauss thêm vào feature (đã chuẩn hoá) khi train
    input_dropout: float = 0.0   # xác suất đặt 1 feature = 0 (= trung vị) khi train
    n_models: int = 1            # bagging: số model / seed (seed + 7919 * j), dự báo = TB


# các trường mới: bỏ khỏi khoá cache khi để mặc định -> cache cũ vẫn dùng được
_NEW_DEFAULTS = {"eval_per_epoch": 1, "lr_schedule": "none", "warmup_frac": 0.05, "label_smoothing": 0.0, "ema": 0.0,
                 "input_noise": 0.0, "input_dropout": 0.0, "n_models": 1}


def make_params(kind: str, over: dict | None = None):
    P = {"tabm": TabMParams, "lstm": LSTMParams}[kind]
    return P(**(over or {}))


class Prep:
    """NaN -> trung vị + cờ thiếu, QuantileTransformer (chuẩn), cắt ±5. Học trên các hàng train."""

    def fit(self, X: np.ndarray):
        from sklearn.preprocessing import QuantileTransformer
        X = np.where(np.isfinite(X), X, np.nan).astype(np.float64)
        self.med = np.nan_to_num(np.nanmedian(X, 0))
        self.nan_cols = np.where(np.isnan(X).any(0))[0]
        Z = np.where(np.isnan(X), self.med, X)
        self.qt = QuantileTransformer(n_quantiles=min(1000, len(Z)), output_distribution="normal",
                                      subsample=200_000, random_state=0).fit(Z)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.where(np.isfinite(X), X, np.nan).astype(np.float64)
        m = np.isnan(X)
        Z = np.clip(self.qt.transform(np.where(m, self.med, X)), -5, 5)
        return np.hstack([Z, m[:, self.nan_cols]]).astype(np.float32)


def _torch():
    import torch
    from torch import nn
    return torch, nn


def _build_net(kind: str, d_in: int, n_out: int, p):
    torch, nn = _torch()

    class LinearBE(nn.Module):
        def __init__(self, a, b, k, random_r):
            super().__init__()
            self.W = nn.Parameter(torch.empty(a, b))
            nn.init.kaiming_uniform_(self.W.T, a=5 ** 0.5)
            r = torch.where(torch.rand(k, a) < 0.5, -1.0, 1.0) if random_r else torch.ones(k, a)
            self.r = nn.Parameter(r)
            self.s = nn.Parameter(torch.ones(k, b))
            bound = 1 / a ** 0.5
            self.b = nn.Parameter(torch.empty(k, b).uniform_(-bound, bound))

        def forward(self, x):                              # (B, k, a) -> (B, k, b)
            return ((x * self.r) @ self.W) * self.s + self.b

    class TabM(nn.Module):
        def __init__(self):
            super().__init__()
            self.k = p.k
            layers, a = [], d_in
            for i in range(p.depth):
                layers += [LinearBE(a, p.width, p.k, i == 0), nn.ReLU(), nn.Dropout(p.dropout)]
                a = p.width
            self.body = nn.Sequential(*layers)
            self.head = LinearBE(a, n_out, p.k, False)

        def forward(self, x):                              # (B, d) -> logits (B, k, C)
            return self.head(self.body(x.unsqueeze(1).expand(-1, self.k, -1)))

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            d = d_in + 1                                   # + kênh cờ "nến có dữ liệu"
            self.proj = nn.Sequential(nn.Linear(d, p.in_proj), nn.GELU()) if p.in_proj else nn.Identity()
            self.lstm = nn.LSTM(p.in_proj or d, p.hidden, num_layers=p.layers, batch_first=True,
                                dropout=p.dropout if p.layers > 1 else 0.0)
            self.head = nn.Sequential(nn.Dropout(p.dropout), nn.Linear(p.hidden, n_out))

        def forward(self, x):                              # (B, L, d+1) -> logits (B, 1, C)
            out, _ = self.lstm(self.proj(x))
            return self.head(out[:, -1]).unsqueeze(1)

    return TabM() if kind == "tabm" else LSTMNet()


class NNClassifier:
    """fit(Z, ts_ns, tr, y_tr, es, y_es, w_tr) trên ma trận đã tiền xử lý Z của MỌI hàng (theo thời gian);
    predict_proba(idx) -> (n, C). TabM dùng hàng idx; LSTM dùng chuỗi seq_len hàng kết thúc ở idx."""

    def __init__(self, kind: str, params, n_classes: int, seed: int, device: str = "auto", threads: int = 4):
        self.kind, self.p, self.C, self.seed = kind, params, n_classes, seed
        self.device, self.threads = device, threads

    def _dev(self):
        torch, _ = _torch()
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _batch(self, idx, keep_last: int | None = None):
        torch, _ = _torch()
        if self.kind == "tabm":
            return self.Zt[torch.as_tensor(idx, device=self.dev)]
        L = self.p.seq_len
        J = idx[:, None] + np.arange(-L + 1, 1)[None, :]
        Jc = np.clip(J, 0, None)
        ok = (J >= 0) & (self.ts[idx][:, None] - self.ts[Jc] <= (L - 1) * H_NS)
        if keep_last is not None and keep_last < L:                   # chẩn đoán: che các nến cũ hơn keep_last nến cuối
            ok[:, :L - keep_last] = False
        Jt = torch.as_tensor(Jc, device=self.dev)
        okt = torch.as_tensor(ok, device=self.dev, dtype=torch.float32).unsqueeze(-1)
        return torch.cat([self.Zt[Jt] * okt, okt], dim=-1)

    def fit(self, Z, ts_ns, tr, y_tr, es, y_es, w_tr=None, verbose=False):
        import copy
        import math
        torch, nn = _torch()
        p = self.p
        g = lambda k, d: getattr(p, k, d)                                  # noqa: E731  (TabMParams không có trường mới)
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        rng = np.random.default_rng(self.seed)
        self.dev = self._dev()
        self.Zt = torch.as_tensor(Z, device=self.dev)
        self.ts = np.asarray(ts_ns, np.int64)
        tr, es = np.asarray(tr, np.int64), np.asarray(es, np.int64)
        yt = torch.as_tensor(np.asarray(y_tr, np.int64), device=self.dev)
        w = np.ones(len(tr), np.float32) if w_tr is None else np.asarray(w_tr, np.float32) / np.mean(w_tr)
        wt = torch.as_tensor(w, device=self.dev)
        y_es = np.asarray(y_es, np.int64)
        self.net = _build_net(self.kind, Z.shape[1], self.C, p).to(self.dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=p.lr, weight_decay=p.weight_decay)
        ce = nn.CrossEntropyLoss(reduction="none", label_smoothing=g("label_smoothing", 0.0))
        steps_ep = math.ceil(len(tr) / p.batch_size)
        n_eval = max(1, int(g("eval_per_epoch", 1)))
        eval_at = {round(steps_ep * (e + 1) / n_eval) for e in range(n_eval)}   # bước (trong epoch) đo es
        sched = None
        if g("lr_schedule", "none") == "cosine":
            total, warm = steps_ep * p.max_epochs, max(1, int(g("warmup_frac", 0.05) * steps_ep * p.max_epochs))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / warm) if t < warm else
                                                      0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, total - warm))))
        ema_d = float(g("ema", 0.0))
        ema_net = copy.deepcopy(self.net) if ema_d > 0 else None
        noise, idrop = float(g("input_noise", 0.0)), float(g("input_dropout", 0.0))
        patience = p.patience * n_eval
        best, best_state, bad, self.history, self.train_history = np.inf, None, 0, [], []
        stop = False
        for ep in range(p.max_epochs):
            self.net.train()
            perm = rng.permutation(len(tr))
            tot, seen = torch.zeros((), device=self.dev), 0
            for step, i in enumerate(range(0, len(tr), p.batch_size), start=1):
                b = perm[i:i + p.batch_size]
                x = self._batch(tr[b])
                if noise > 0 or idrop > 0:                                  # tăng cường dữ liệu: chỉ trên feature
                    f = x[..., :-1] if self.kind == "lstm" else x
                    if noise > 0:
                        f = f + noise * torch.randn_like(f) * (x[..., -1:] if self.kind == "lstm" else 1.0)
                    if idrop > 0:
                        f = f * (torch.rand_like(f) >= idrop)
                    x = torch.cat([f, x[..., -1:]], dim=-1) if self.kind == "lstm" else f
                logits = self.net(x)                                        # (B, k, C)
                B, k, C = logits.shape
                yb = yt[torch.as_tensor(b, device=self.dev)]
                loss = ce(logits.reshape(B * k, C), yb.repeat_interleave(k)).reshape(B, k).mean(1)
                loss = (loss * wt[torch.as_tensor(b, device=self.dev)]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                if sched is not None:
                    sched.step()
                if ema_net is not None:
                    with torch.no_grad():
                        for pe_, pn in zip(ema_net.parameters(), self.net.parameters()):
                            pe_.mul_(ema_d).add_(pn.detach(), alpha=1 - ema_d)
                tot, seen = tot + loss.detach() * len(b), seen + len(b)
                if step not in eval_at:
                    continue
                self.train_history.append(float(tot) / max(seen, 1))     # loss train TB từ lần đo trước (có dropout)
                tot, seen = torch.zeros((), device=self.dev), 0
                raw = self.net
                if ema_net is not None:
                    self.net = ema_net
                pe = np.clip(self.predict_proba(es)[np.arange(len(es)), y_es], 1e-7, 1)
                cur = self.net
                self.net = raw
                raw.train()
                ll = float(-np.mean(np.log(pe)))
                self.history.append(ll)
                if verbose:
                    print(self.kind, ep, step, round(ll, 5))
                if ll < best - 1e-5:
                    best, bad = ll, 0
                    best_state = {k_: v.detach().clone() for k_, v in cur.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        stop = True
                        break
            if stop:
                break
        self.net.load_state_dict(best_state)
        self.best_epoch = (int(np.argmin(self.history)) + 1) / n_eval        # số epoch (có thể lẻ) tại điểm tốt nhất
        if n_eval == 1:
            self.best_epoch = int(self.best_epoch)
        self.best_logloss = best
        return self

    def predict_proba(self, idx, Z: np.ndarray | None = None, keep_last: int | None = None) -> np.ndarray:
        """Z: ma trận đã tiền xử lý thay cho ma trận lúc fit (chẩn đoán, ví dụ hoán vị feature);
        keep_last: LSTM chỉ thấy keep_last nến cuối của chuỗi."""
        torch, _ = _torch()
        idx = np.asarray(idx, np.int64)
        old = self.Zt
        if Z is not None:
            self.Zt = torch.as_tensor(np.asarray(Z, np.float32), device=self.dev)
        self.net.eval()
        out = []
        try:
            with torch.no_grad():
                for i in range(0, len(idx), 4096):
                    pr = torch.softmax(self.net(self._batch(idx[i:i + 4096], keep_last)), dim=-1).mean(1)
                    out.append(pr.float().cpu().numpy())
        finally:
            self.Zt = old
        return np.concatenate(out).astype(np.float64)


def fit_fold_nn(D, sp: dict, cfg, seed: int) -> dict:
    """Như strategy_h1.fit_fold nhưng model = cfg.model_kind (tabm | lstm). Trả về p_touch / p_up / p_dtouch trên cửa sổ."""
    kind = cfg.model_kind
    p = make_params(kind, cfg.nn_params)
    ds, tc = D.ds, cfg.touch_cfg()
    tr, es, cal, win = sp["train"], sp["es"], sp["cal"], sp["window"]
    ts = ds.timestamp_ns
    kw = dict(device=cfg.nn_device, threads=cfg.nn_threads)
    # touch: nhị phân, trọng số thời gian, hiệu chỉnh nhiệt độ trên cal (giống XGBoost)
    y = ds.labels["y_touch"].astype(np.int64)
    Zt = Prep().fit(ds.x[tr]).transform(ds.x)
    w = mv2._time_weights(ds, tr, int(ts[tr].max()), tc)
    tm = NNClassifier(kind, p, 2, seed, **kw).fit(Zt, ts, tr, y[tr], es, y[es], w)
    T = mv2.fit_temperature(mv2.logit(tm.predict_proba(cal)[:, 1]), y[cal])
    p_touch = mv2.sigmoid(mv2.logit(tm.predict_proba(win)[:, 1]) / max(T, 1e-4))
    dm = fit_direction_nn(D, sp, cfg, seed)
    p_up, p_dt = scores(dm.predict_proba(win))
    return {"p_touch": p_touch.astype(np.float32), "p_up": p_up.astype(np.float32), "p_dtouch": p_dt.astype(np.float32),
            "meta": {"touch_rounds": tm.best_epoch, "dir_rounds": dm.best_epoch, "T_touch": float(T),
                     "n_train": int(len(tr)), "touch_es_logloss": tm.best_logloss, "dir_es_logloss": dm.best_logloss},
            "models": {"touch": tm, "direction": dm}}


def scores(pr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(p_up, p_dtouch) từ xác suất lớp, giống strategy_h1.direction_proba."""
    if pr.shape[1] == 3:
        return 0.5 + 0.5 * (pr[:, 1] - pr[:, 2]), 1.0 - pr[:, 0]
    return pr[:, 1], np.full(len(pr), np.nan)


def fit_direction_nn(D, sp: dict, cfg, seed: int, kind: str | None = None):
    """Direction (nhãn cfg.direction_label, không trọng số - giống XGBoost) bằng model kind (mặc định cfg.model_kind)."""
    import strategy_h1 as st
    kind = kind or cfg.model_kind
    p = make_params(kind, cfg.nn_params)
    ds, tr, es = D.ds, sp["train"], sp["es"]
    y3 = st.direction_labels(ds, cfg.direction_label)
    prep = Prep().fit(D.xd[tr])
    Z = prep.transform(D.xd)
    members = []
    for j in range(max(1, int(getattr(p, "n_models", 1)))):              # j = 0: đúng model 1-seed như trước
        dm = NNClassifier(kind, p, int(y3.max()) + 1, seed + 7919 * j, device=cfg.nn_device, threads=cfg.nn_threads)
        dm.fit(Z, ds.timestamp_ns, tr, y3[tr], es, y3[es])
        dm.prep = prep
        members.append(dm)
    return members[0] if len(members) == 1 else Bag(members)


class Bag:
    """Bagging nhiều NNClassifier cùng dữ liệu, khác seed: xác suất = TB các thành viên. API như NNClassifier."""

    def __init__(self, members: list):
        self.members, self.kind, self.prep = members, members[0].kind, members[0].prep
        self.best_epoch = float(np.mean([m.best_epoch for m in members]))
        self.best_logloss = float(np.mean([m.best_logloss for m in members]))

    def predict_proba(self, idx, Z=None, keep_last=None) -> np.ndarray:
        return np.mean([m.predict_proba(idx, Z=Z, keep_last=keep_last) for m in self.members], axis=0)


# --------------------------------------------------------------------------- #
# Xuất ONNX / dự báo bằng onnxruntime (container live không cần torch)
# --------------------------------------------------------------------------- #
ONNX_OPSET = 17
_SESSIONS: dict = {}


def export_model(m) -> dict:
    """Model đã train -> dict {kind, params, n_classes, prep (sklearn), onnx (bytes)}; dùng với predict_model.
    Đồ thị ONNX nhận đầu vào đã tiền xử lý (TabM: (B, d); LSTM: (B, L, d+1)) và trả về logits (B, k, C).
    Bag -> {"kind": "bag", "members": [...]}."""
    import io
    if isinstance(m, Bag):
        return {"kind": "bag", "members": [export_model(x) for x in m.members]}
    torch, _ = _torch()
    net = m.net.to("cpu").eval()
    d = len(m.prep.med) + len(m.prep.nan_cols)
    dummy = torch.zeros((2, d) if m.kind == "tabm" else (2, m.p.seq_len, d + 1), dtype=torch.float32)
    buf = io.BytesIO()
    torch.onnx.export(net, dummy, buf, input_names=["x"], output_names=["logits"], opset_version=ONNX_OPSET,
                      dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    net.to(m.dev)
    return {"kind": m.kind, "params": asdict(m.p), "n_classes": m.C, "prep": m.prep, "onnx": buf.getvalue()}


def _session(m: dict):
    key = id(m["onnx"])
    if key not in _SESSIONS:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        _SESSIONS[key] = ort.InferenceSession(m["onnx"], so, providers=["CPUExecutionProvider"])
    return _SESSIONS[key]


def sequences(Z: np.ndarray, ts_ns: np.ndarray, idx: np.ndarray, L: int) -> np.ndarray:
    """Chuỗi (n, L, d+1) kết thúc ở các hàng idx, giống NNClassifier._batch."""
    ts_ns = np.asarray(ts_ns, np.int64)
    J = idx[:, None] + np.arange(-L + 1, 1)[None, :]
    Jc = np.clip(J, 0, None)
    ok = ((J >= 0) & (ts_ns[idx][:, None] - ts_ns[Jc] <= (L - 1) * H_NS)).astype(np.float32)[..., None]
    return np.concatenate([Z[Jc] * ok, ok], axis=-1).astype(np.float32)


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def predict_model(m: dict, X: np.ndarray, ts_ns: np.ndarray, idx=None) -> np.ndarray:
    """Xác suất lớp (onnxruntime) cho các hàng idx (mặc định mọi hàng) của ma trận feature thô X (theo thời gian).
    LSTM: hàng nào có ít hơn seq_len hàng trước trong X sẽ thiếu đầu chuỗi -> chỉ dùng cho hàng cuối của X."""
    if m["kind"] == "bag":
        return np.mean([predict_model(x, X, ts_ns, idx) for x in m["members"]], axis=0)
    Z = m["prep"].transform(X)
    idx = np.arange(len(Z)) if idx is None else np.asarray(idx, np.int64)
    sess, out = _session(m), []
    for a in range(0, len(idx), 2048):
        b = idx[a:a + 2048]
        x = sequences(Z, ts_ns, b, m["params"]["seq_len"]) if m["kind"] == "lstm" else Z[b].astype(np.float32)
        logits = sess.run(None, {"x": x})[0].astype(np.float64)
        out.append(_softmax(logits).mean(1))
    return np.concatenate(out)


def history_rows(bundle_models: list) -> int:
    """Số hàng feature live cần để dự báo hàng cuối (seq_len lớn nhất của các model LSTM)."""
    L = 1
    for m in bundle_models:
        for nm in m.get("nn", {}).values():
            for x in (nm["members"] if nm["kind"] == "bag" else [nm]):
                if x["kind"] == "lstm":
                    L = max(L, int(x["params"]["seq_len"]))
    return L


def params_dict(cfg) -> dict:
    d = asdict(make_params(cfg.model_kind, cfg.nn_params))
    return {"kind": cfg.model_kind, **{k: v for k, v in d.items() if not (k in _NEW_DEFAULTS and v == _NEW_DEFAULTS[k])}}
