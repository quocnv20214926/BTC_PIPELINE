"""TabM: MLP + BatchEnsemble (k thành viên chia sẻ trọng số), Gorishniy và cộng sự, ICLR 2025
"TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling".

Dùng cho bảng feature (touch H1 và model hướng B). Mỗi lớp tuyến tính có trọng số W dùng chung và các vector
r_i, s_i, b_i riêng cho thành viên i:  y_i = ((x_i * r_i) @ W) * s_i + b_i.  k thành viên được train đồng thời
(loss = trung bình BCE của từng thành viên), dự báo = trung bình xác suất. Cho chất lượng gần một ensemble k MLP
với chi phí của một MLP rộng.

Tiền xử lý (chỉ học trên tập train): NaN -> trung vị + cột cờ thiếu (cho các cột có NaN), QuantileTransformer
về phân phối chuẩn, cắt ±5. Dừng sớm theo log-loss của dự báo trung bình trên tập dừng sớm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TabMParams:
    k: int = 16                 # số thành viên ensemble
    width: int = 256
    depth: int = 3
    dropout: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 80
    patience: int = 8
    seed: int = 42
    threads: int = 2


class _Prep:
    def fit(self, X: np.ndarray):
        from sklearn.preprocessing import QuantileTransformer
        X = np.asarray(X, np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        self.med = np.nan_to_num(np.nanmedian(X, 0))
        self.nan_cols = np.where(np.isnan(X).any(0))[0]
        Z = np.where(np.isnan(X), self.med, X)
        self.qt = QuantileTransformer(n_quantiles=min(1000, len(Z)), output_distribution="normal",
                                      subsample=200_000, random_state=0).fit(Z)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        m = np.isnan(X)
        Z = np.clip(self.qt.transform(np.where(m, self.med, X)), -5, 5)
        return np.hstack([Z, m[:, self.nan_cols].astype(np.float64)]).astype(np.float32)


import torch
from torch import nn


class LinearBE(nn.Module):
    """Lớp tuyến tính BatchEnsemble: W dùng chung, r / s / b riêng cho từng thành viên."""

    def __init__(self, a: int, b: int, k: int, random_r: bool):
        super().__init__()
        self.W = nn.Parameter(torch.empty(a, b))
        nn.init.kaiming_uniform_(self.W.T, a=5 ** 0.5)
        r = torch.ones(k, a)
        if random_r:                                       # đa dạng hoá thành viên từ lớp đầu
            r = torch.where(torch.rand(k, a) < 0.5, -1.0, 1.0)
        self.r = nn.Parameter(r)
        self.s = nn.Parameter(torch.ones(k, b))
        bound = 1 / a ** 0.5
        self.b = nn.Parameter(torch.empty(k, b).uniform_(-bound, bound))

    def forward(self, x):                                  # x: (B, k, a)
        return ((x * self.r) @ self.W) * self.s + self.b


class TabM(nn.Module):
    def __init__(self, d_in: int, p: TabMParams):
        super().__init__()
        self.k = p.k
        layers, a = [], d_in
        for i in range(p.depth):
            layers += [LinearBE(a, p.width, p.k, i == 0), nn.ReLU(), nn.Dropout(p.dropout)]
            a = p.width
        self.body = nn.Sequential(*layers)
        self.head = LinearBE(a, 1, p.k, False)

    def forward(self, x):                                  # x: (B, d) -> logit (B, k)
        x = x.unsqueeze(1).expand(-1, self.k, -1)
        return self.head(self.body(x)).squeeze(-1)


class TabMClassifier:
    """Phân loại nhị phân. API giống sklearn: fit(X, y, eval_set=[(Xe, ye)], sample_weight), predict_proba."""

    def __init__(self, params: TabMParams | None = None):
        self.p = params or TabMParams()

    def fit(self, X, y, eval_set, sample_weight=None, verbose=False):
        torch.manual_seed(self.p.seed)
        torch.set_num_threads(self.p.threads)
        rng = np.random.default_rng(self.p.seed)
        self.prep = _Prep().fit(X)
        Xt = torch.from_numpy(self.prep.transform(X))
        yt = torch.from_numpy(np.asarray(y, np.float32))
        wt = torch.from_numpy(np.ones(len(yt), np.float32) if sample_weight is None
                              else np.asarray(sample_weight, np.float32) / np.mean(sample_weight))
        Xe_np, ye = eval_set[0]
        Xe = torch.from_numpy(self.prep.transform(Xe_np))
        ye = np.asarray(ye, np.float64)
        self.net = TabM(Xt.shape[1], self.p)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.p.lr, weight_decay=self.p.weight_decay)
        bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        best, best_state, bad, self.history = np.inf, None, 0, []
        n = len(yt)
        for ep in range(self.p.max_epochs):
            self.net.train()
            perm = torch.from_numpy(rng.permutation(n))
            for i in range(0, n, self.p.batch_size):
                idx = perm[i:i + self.p.batch_size]
                logit = self.net(Xt[idx])
                loss = (bce(logit, yt[idx, None].expand_as(logit)).mean(1) * wt[idx]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            pe = np.clip(self._proba_t(Xe), 1e-6, 1 - 1e-6)
            ll = float(-np.mean(ye * np.log(pe) + (1 - ye) * np.log(1 - pe)))
            self.history.append(ll)
            if verbose:
                print(ep, round(ll, 5))
            if ll < best - 1e-5:
                best, bad = ll, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.p.patience:
                    break
        self.net.load_state_dict(best_state)
        self.best_iteration = int(np.argmin(self.history))          # số epoch tốt nhất (0-based)
        self.best_logloss = best
        return self

    def _proba_t(self, Xt):
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Xt), 8192):
                out.append(torch.sigmoid(self.net(Xt[i:i + 8192])).mean(1).numpy())
        return np.concatenate(out).astype(np.float64)

    def predict_proba(self, X):
        p = self._proba_t(torch.from_numpy(self.prep.transform(X)))
        return np.column_stack([1 - p, p])
