from __future__ import annotations
import numpy as np, json, time

def delta_encode(points: np.ndarray, scale: float = 100.0):
    """Delta-encode polyline vertices (float->int), return dict."""
    if points is None or len(points) == 0:
        return {"k": 0, "d": []}
    q = np.round(points * scale).astype(int)
    d = []
    prev = q[0].tolist()
    d.extend(prev)
    for i in range(1, len(q)):
        delta = (q[i] - q[i-1]).tolist()
        d.extend(delta)
    return {"k": len(q), "d": d, "scale": scale}

def build_message(poly: np.ndarray, gsd: float, eps: float, pose=(0,0,0)):
    payload = {
        "ts": time.time(),
        "gsd": gsd,
        "eps": eps,
        "pose": {"x": pose[0], "y": pose[1], "yaw": pose[2]},
        "poly": delta_encode(poly if poly is not None else np.zeros((0,2))),
        "crc": 0  # placeholder
    }
    return json.dumps(payload, ensure_ascii=False)
