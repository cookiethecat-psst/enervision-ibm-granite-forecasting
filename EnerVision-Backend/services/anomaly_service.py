# services/anomaly_service.py
from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Sequence

import numpy as np

# from repositories.meter_repo import get_timeseries  # <- your real repo

def _to_float(x):
    import numpy as np  # local import prevents global type issues
    if isinstance(x, (np.floating, np.float32, np.float64)): return float(x)
    if isinstance(x, (np.integer,)): return int(x)
    return float(x)

async def detect_anomalies_for_window(
    building_id: str, start: datetime, end: datetime
) -> List[Dict]:
    # 1) Load your time series: [{"ts": datetime, "value": float}, ...]
    # rows = await get_timeseries(building_id, start, end)
    rows: Sequence[Dict] = []  # TODO: wire your real repo

    if not rows:
        return []

    rows = sorted(rows, key=lambda r: r["ts"])
    ts = [r["ts"] for r in rows]
    vals = np.array([_to_float(r["value"]) for r in rows], dtype=float)

    # Remove NaNs/infs
    mask = np.isfinite(vals)
    if mask.sum() < 5:
        return []  # not enough points to fit a model
    idx = np.where(mask)[0]
    ts = [ts[i] for i in idx]
    vals = vals[mask].reshape(-1, 1)

    # 2) IsolationForest (lightweight, no scaling needed for 1D)
    try:
        from sklearn.ensemble import IsolationForest
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for IsolationForest. Install with: "
            "pip install scikit-learn"
        ) from e

    # contamination='auto' adapts to the data; set 0.01–0.05 to be stricter/looser
    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(vals)
    preds = iso.predict(vals)          # -1 = anomaly, 1 = normal
    scores = -iso.score_samples(vals)  # higher = more anomalous

    anomalies: List[Dict] = []
    for i, (p, s) in enumerate(zip(preds, scores)):
        if p == -1:
            anomalies.append({
                "ts": ts[i],
                "value": float(vals[i][0]),
                "score": float(s),
                "reason": "IsolationForest",
            })
    return anomalies
