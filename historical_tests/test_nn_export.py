"""Model Direction NN (TabM / LSTM) xuất sang ONNX (nn_models.export_model / predict_model) phải dự báo giống torch.
Container live không có torch, chỉ có onnxruntime. Cần torch + onnxruntime (môi trường train) - không có thì bỏ qua."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model_training'))

HAS_TORCH = all(importlib.util.find_spec(m) is not None for m in ('torch', 'onnxruntime'))


@unittest.skipUnless(HAS_TORCH, 'cần torch + onnxruntime')
class NNExportTests(unittest.TestCase):
    def _data(self, n=1500, d=12):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n, d))
        X[rng.random((n, d)) < 0.03] = np.nan
        ts = (np.arange(n) * 3600 * 10**9).astype(np.int64)
        ts[800:] += 30 * 3600 * 10**9                       # gap: chuỗi LSTM phải cắt ở đây
        y = (np.nan_to_num(X[:, 0]) + rng.normal(size=n) > 0).astype(int) + (rng.random(n) < 0.3)
        return X, ts, y

    def test_onnx_equals_torch(self):
        import nn_models as nm
        X, ts, y = self._data()
        tr, es, te = np.arange(0, 1000), np.arange(1000, 1200), np.arange(1200, 1500)
        for kind, over in (('lstm', {'seq_len': 8, 'max_epochs': 2}), ('tabm', {'max_epochs': 2, 'k': 4, 'width': 32})):
            with self.subTest(kind=kind):
                prep = nm.Prep().fit(X[tr])
                m = nm.NNClassifier(kind, nm.make_params(kind, over), 3, seed=1, device='cpu', threads=1)
                m.fit(prep.transform(X), ts, tr, y[tr], es, y[es])
                m.prep = prep
                p_t = m.predict_proba(te)
                ex = nm.export_model(m)
                p_n = nm.predict_model(ex, X, ts, te)
                self.assertLess(float(np.abs(p_t - p_n).max()), 1e-5)
                if kind == 'lstm':                           # chỉ đưa vào seq_len hàng cuối như live
                    i = te[-1]
                    p_live = nm.predict_model(ex, X[i - 7:i + 1], ts[i - 7:i + 1])[-1]
                    self.assertLess(float(np.abs(p_live - p_t[-1]).max()), 1e-5)


if __name__ == '__main__':
    unittest.main()
