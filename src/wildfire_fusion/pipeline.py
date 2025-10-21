from __future__ import annotations
import os, json, math
from dataclasses import dataclass
import numpy as np
import cv2

from .utils import EMA, to_gray, ensure_uint8
from .rdp import rdp
from .messaging import build_message

@dataclass
class Params:
    # gsd is meters/pixel and is used for reasoning about scale,
    # yet core thresholds below are defined directly in pixels.
    gsd: float = 0.2       # meters per pixel (approx.)
    rdp_k: float = 1.0     # RDP epsilon in pixels
    band_c: float = 3.0    # band half-width in pixels (distance transform threshold)
    ema_alpha: float = 0.2

class FusionPipeline:
    def __init__(self, params: Params):
        self.p = params
        self._ema_thresh = None

    # ---- Thermal mask (Otsu + morphology) ----
    def thermal_mask(self, thermal_u8: np.ndarray) -> np.ndarray:
        gray = ensure_uint8(thermal_u8)
        # Otsu threshold
        t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # EMA on threshold to stabilize flicker
        if self._ema_thresh is None:
            self._ema_thresh = t
        self._ema_thresh = (1 - self.p.ema_alpha) * self._ema_thresh + self.p.ema_alpha * t
        t_ema = int(round(self._ema_thresh))
        _, mask = cv2.threshold(gray, t_ema, 255, cv2.THRESH_BINARY)

        # Morphology: radius tied to band width (pixels)
        r_px = max(1, int(round(max(1.0, self.p.band_c))))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r_px+1, 2*r_px+1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)  # fill holes-ish
        return mask

    # ---- Edge map (Canny) restricted to boundary band ----
    def rgb_edges_in_band(self, rgb: np.ndarray, mask: np.ndarray):
        gray = to_gray(rgb)
        gray = ensure_uint8(gray)

        # Distance transform on the inverse mask; band in PIXELS
        dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
        band_px = max(1.0, float(self.p.band_c))
        band = (dist <= band_px).astype(np.uint8) * 255

        # Canny edges then gate by band
        edges = cv2.Canny(gray, 50, 150)
        gated = cv2.bitwise_and(edges, band)
        return gated, band

    def fuse_and_polygonize(self, mask: np.ndarray, gated_edges: np.ndarray):
        # Combine band-gated edges with rim of the thermal mask
        rim = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
        fused = cv2.bitwise_and(gated_edges, rim)

        # Contours -> select the longest perimeter contour
        cnts, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return None, fused
        cnt = max(cnts, key=lambda c: cv2.arcLength(c, True))
        poly_raw = cnt.reshape(-1, 2).astype(np.float32)

        # RDP with pixel epsilon
        eps_px = float(self.p.rdp_k)
        poly = rdp(poly_raw, epsilon=eps_px)
        return poly, fused

    def step(self, thermal_u8: np.ndarray, rgb_bgr: np.ndarray):
        mask = self.thermal_mask(thermal_u8)
        edges, band = self.rgb_edges_in_band(rgb_bgr, mask)
        poly, fused = self.fuse_and_polygonize(mask, edges)
        return {
            "mask": mask,
            "edges": edges,
            "band": band,
            "fused": fused,
            "poly": poly
        }

def save_overlay(out_dir: str, rgb_bgr: np.ndarray, mask: np.ndarray, edges: np.ndarray, band: np.ndarray, poly: np.ndarray):
    os.makedirs(out_dir, exist_ok=True)
    overlay = rgb_bgr.copy()
    # colorize mask and edges
    mcol = cv2.applyColorMap((mask>0).astype(np.uint8)*255, cv2.COLORMAP_JET)
    e3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    b3 = cv2.cvtColor(band, cv2.COLOR_GRAY2BGR)
    # blend
    overlay = cv2.addWeighted(overlay, 1.0, mcol, 0.4, 0)
    overlay = cv2.addWeighted(overlay, 1.0, e3, 0.8, 0)
    overlay = cv2.addWeighted(overlay, 1.0, b3, 0.3, 0)
    if poly is not None and len(poly) >= 2:
        for i in range(len(poly)-1):
            p1 = tuple(map(int, poly[i]))
            p2 = tuple(map(int, poly[i+1]))
            cv2.line(overlay, p1, p2, (0,255,0), 2)
    return overlay

def export_poly_json(path: str, poly: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if poly is None:
        data = {"k": 0, "points": []}
    else:
        data = {"k": int(len(poly)), "points": poly.tolist()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def export_message_json(path: str, poly: np.ndarray, params: Params, pose=(0,0,0)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Note: eps is in pixels in this design
    msg = build_message(poly, gsd=params.gsd, eps=params.rdp_k, pose=pose)
    with open(path, "w", encoding="utf-8") as f:
        f.write(msg)
