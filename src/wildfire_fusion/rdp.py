from __future__ import annotations
import numpy as np

def _perp_dist(point, start, end):
    # Distance from 'point' to line segment (start, end)
    if np.allclose(start, end):
        return np.linalg.norm(point - start)
    v = end - start
    w = point - start
    c1 = np.dot(w, v)
    if c1 <= 0: return np.linalg.norm(point - start)
    c2 = np.dot(v, v)
    if c2 <= c1: return np.linalg.norm(point - end)
    b = c1 / c2
    pb = start + b * v
    return np.linalg.norm(point - pb)

def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer–Douglas–Peucker simplification.
    points: Nx2 array
    epsilon: tolerance (same unit as points)
    """
    if len(points) < 3:
        return points
    # Find point with max dist
    start, end = points[0], points[-1]
    dmax, idx = 0.0, 0
    for i in range(1, len(points) - 1):
        d = _perp_dist(points[i], start, end)
        if d > dmax:
            idx, dmax = i, d
    if dmax > epsilon:
        left = rdp(points[:idx+1], epsilon)
        right = rdp(points[idx:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.vstack([start, end])
