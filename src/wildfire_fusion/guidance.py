from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float  # radians

def tangent_follow(poly: np.ndarray, pose: Pose2D, v0: float = 5.0):
    """Simple tangent-follow controller stub.
    Returns target heading (rad) and speed.
    """
    if poly is None or len(poly) < 2:
        return pose.yaw, 0.0
    # Find closest segment
    pts = poly[:, :2]
    diffs = pts - np.array([pose.x, pose.y])
    d2 = np.sum(diffs**2, axis=1)
    idx = int(np.argmin(d2))
    idx2 = min(idx + 1, len(pts)-1)
    seg = pts[idx2] - pts[idx]
    yaw_target = np.arctan2(seg[1], seg[0])
    return yaw_target, v0
