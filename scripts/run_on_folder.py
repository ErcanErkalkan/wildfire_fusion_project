#!/usr/bin/env python
from __future__ import annotations
import argparse, os, glob, cv2
import numpy as np
from wildfire_fusion.pipeline import FusionPipeline, Params, save_overlay, export_poly_json, export_message_json

def imread8(p):
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(p)
    if im.ndim == 2:
        # Mono: scale to uint8 if needed
        if im.dtype == np.uint16:
            return cv2.convertScaleAbs(im, alpha=255.0/65535.0)
        if im.dtype != np.uint8:
            im = np.clip(im, 0, 255).astype(np.uint8)
        return im
    else:
        # Color image (BGR); ensure uint8
        if im.dtype != np.uint8:
            im = np.clip(im, 0, 255).astype(np.uint8)
        return im

def main():
    ap = argparse.ArgumentParser(description="Run fusion on paired Thermal+RGB frame folders.")
    ap.add_argument("--thermal_dir", required=True, help="Folder of thermal frames (8-bit or 16-bit).")
    ap.add_argument("--rgb_dir", required=True, help="Folder of RGB frames (BGR).")
    ap.add_argument("--out_dir", default="results", help="Output folder.")
    ap.add_argument("--gsd", type=float, default=0.2, help="Meters per pixel (approx).")
    ap.add_argument("--rdp_k", type=float, default=1.0, help="RDP epsilon in pixels.")
    ap.add_argument("--band_c", type=float, default=3.0, help="Band half-width in pixels.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ov_dir = os.path.join(args.out_dir, "overlays")
    poly_dir = os.path.join(args.out_dir, "polylines")
    msg_dir = os.path.join(args.out_dir, "messages")
    for d in [ov_dir, poly_dir, msg_dir]:
        os.makedirs(d, exist_ok=True)

    params = Params(gsd=args.gsd, rdp_k=args.rdp_k, band_c=args.band_c)
    pipe = FusionPipeline(params)

    thermals = sorted(glob.glob(os.path.join(args.thermal_dir, "*.*")))
    rgbs = sorted(glob.glob(os.path.join(args.rgb_dir, "*.*")))
    n = min(len(thermals), len(rgbs))
    for i in range(n):
        t = imread8(thermals[i])     # mono
        r = cv2.imread(rgbs[i], cv2.IMREAD_COLOR)
        out = pipe.step(t, r)
        overlay = save_overlay(ov_dir, r, out["mask"], out["edges"], out["band"], out["poly"])
        cv2.imwrite(os.path.join(ov_dir, f"{i:06d}.png"), overlay)
        export_poly_json(os.path.join(poly_dir, f"{i:06d}.json"), out["poly"])
        export_message_json(os.path.join(msg_dir, f"{i:06d}.json"), out["poly"], params=params)

    print(f"Done. Wrote {n} frames to {args.out_dir}")

if __name__ == "__main__":
    main()
