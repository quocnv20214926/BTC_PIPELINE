"""Nối thêm lịch sử dự báo vào bundle live tới dữ liệu mới nhất, không train lại.

Live tính ngưỡng trượt (touch 14 ngày, Direction 90 ngày) từ lịch sử dự báo lưu trong bundle cộng với dự báo live đã
ghi. Nếu bundle dừng ở ngày X mà pipeline bắt đầu chạy muộn hơn nhiều ngày, cửa sổ 14 ngày sẽ không đủ 50% số giờ, nên
pipeline không ra tín hiệu cho tới khi live tự tích luỹ đủ (khoảng 7 ngày). Script này dùng CHÍNH các model trong bundle
(cùng quý) để dự báo các giờ từ data_end_ns tới hết dữ liệu lịch sử mới, rồi nối vào bundle['history'].

Model không đổi: fold của quý hiện tại train trên nhãn tới trước Q - 2 quý, dữ liệu mới chỉ nằm trong cửa sổ dự báo.
Sang quý mới thì phải train lại bằng train_strategy_h1.ipynb (worker sẽ cảnh báo bundle cũ).

Các bước:
    python collect_historical.py --timeframes 5m 1h      # nến 5m / 1h + funding tới nến đóng gần nhất
    python collect_positioning.py                        # vị thế (data.binance.vision, trễ khoảng 1 ngày) + dựng bảng
    python model_training/refresh_bundle.py              # bundle mặc định artifacts/1h_strategy_v2/models_final.joblib
Bản cũ được lưu lại thành models_final.<data_end cũ>.joblib.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model_training"))

import modelv2 as mv2  # noqa: E402
import nn_models  # noqa: E402
import strategy_h1 as st  # noqa: E402


def extend(bundle_path: Path, log=print) -> Path:
    import joblib
    raw = joblib.load(bundle_path)
    b = st.load_bundle(bundle_path)
    cfg: st.StrategyConfig = b["config"]
    D = st.build(cfg, log=log)
    ds = D.ds
    pos = pd.read_csv(cfg.positioning_csv, usecols=["open_time_ms", "m_ok"])
    pos_end_ns = int(pos.open_time_ms[pos.m_ok == 1].max()) * 1_000_000          # nến 5m cuối có dữ liệu metrics thật
    end_old = int(b["data_end_ns"])
    # chỉ các giờ đã có đủ dữ liệu vị thế (nến 1h [t, t + 1h) cần vị thế tới t + 55 phút)
    new = np.flatnonzero((ds.timestamp_ns > end_old) & (ds.timestamp_ns + 3_600_000_000_000 <= pos_end_ns + 300_000_000_000))
    if not len(new):
        log(f"không có giờ mới sau {pd.to_datetime(end_old, utc=True)} (vị thế tới {pd.to_datetime(pos_end_ns, utc=True)})")
        return bundle_path
    if list(ds.names) != b["touch_names"] or list(D.dir_names) != b["dir_names"]:
        raise ValueError("feature khác bundle")
    blend = b.get("direction_blend") or {"xgb": 1.0}
    pt = np.mean([mv2.predict_touch(m["touch"], ds.x[new]) for m in b["models"]], axis=0)
    pus = []
    for m in b["models"]:
        d = blend.get("xgb", 0.0) * (st.direction_score(*m["direction"], D.xd[new]) - 0.5)
        for k, w in blend.items():
            if k != "xgb":
                d = d + w * (nn_models.scores(nn_models.predict_model(m["nn"][k], D.xd, ds.timestamp_ns, new))[0] - 0.5)
        pus.append(0.5 + d)
    add = pd.DataFrame({"t_ns": ds.timestamp_ns[new], "p_touch": pt, "p_up": np.mean(pus, axis=0)})
    # kiểm tra: các giờ trùng với lịch sử cũ phải cho cùng dự báo (cùng model, cùng feature)
    chk = np.flatnonzero((ds.timestamp_ns > end_old - 48 * 3_600_000_000_000) & (ds.timestamp_ns <= end_old))
    if len(chk):
        old = raw["history"].set_index("t_ns")
        ptc = np.mean([mv2.predict_touch(m["touch"], ds.x[chk]) for m in b["models"]], axis=0)
        dt = float(np.nanmax(np.abs(ptc - old.reindex(ds.timestamp_ns[chk]).p_touch.to_numpy())))
        log(f"kiểm tra 48 giờ cuối của lịch sử cũ: max |Δp_touch| = {dt:.2e}")
        if dt > 1e-4:
            raise RuntimeError("dự báo khác lịch sử cũ - dữ liệu lịch sử đã đổi? Hãy train lại bằng notebook")
    backup = bundle_path.with_name(f"{bundle_path.stem}.{pd.to_datetime(end_old, utc=True):%Y%m%dT%H%M}.joblib")
    shutil.copy2(bundle_path, backup)
    raw["history"] = pd.concat([raw["history"], add], ignore_index=True).drop_duplicates("t_ns", keep="last")
    raw["data_end_ns"] = int(add.t_ns.iloc[-1])
    joblib.dump(raw, bundle_path)
    q_now = pd.Period(pd.Timestamp.now(tz="UTC").tz_localize(None), "Q")
    log(f"nối {len(add)} giờ: {pd.to_datetime(add.t_ns.iloc[0], utc=True)} -> {pd.to_datetime(add.t_ns.iloc[-1], utc=True)}; "
        f"lịch sử {len(raw['history'])} giờ; bản cũ: {backup.name}")
    if str(q_now) != raw["trained_quarter"]:
        log(f"CẢNH BÁO: bundle train cho {raw['trained_quarter']}, quý hiện tại {q_now} -> chạy lại train_strategy_h1.ipynb")
    return bundle_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="artifacts/1h_strategy_v2/models_final.joblib")
    a = ap.parse_args()
    import os
    os.chdir(ROOT)
    extend(Path(a.bundle))
