#!/usr/bin/env python
from __future__ import annotations
import argparse, os, cv2, numpy as np
from wildfire_fusion.pipeline import FusionPipeline, Params, save_overlay, export_poly_json, export_message_json

def main():
    ap = argparse.ArgumentParser(description="Run fusion on a video (thermal proxy from V channel).")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--gsd", type=float, default=0.2)
    ap.add_argument("--rdp_k", type=float, default=1.0)
    ap.add_argument("--band_c", type=float, default=3.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ov_dir = os.path.join(args.out_dir, "overlays")
    poly_dir = os.path.join(args.out_dir, "polylines")
    msg_dir = os.path.join(args.out_dir, "messages")
    for d in [ov_dir, poly_dir, msg_dir]:
        os.makedirs(d, exist_ok=True)

    params = Params(gsd=args.gsd, rdp_k=args.rdp_k, band_c=args.band_c)
    pipe = FusionPipeline(params)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Mock thermal as the V channel or use a grayscale proxy if no separate thermal is provided.
        rgb = frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        thermal_proxy = hsv[:,:,2]  # proxy
        out = pipe.step(thermal_proxy, rgb)
        overlay = save_overlay(ov_dir, rgb, out["mask"], out["edges"], out["band"], out["poly"])
        cv2.imwrite(os.path.join(ov_dir, f"{i:06d}.png"), overlay)
        export_poly_json(os.path.join(poly_dir, f"{i:06d}.json"), out["poly"])
        export_message_json(os.path.join(msg_dir, f"{i:06d}.json"), out["poly"], params=params)
        i += 1

    print(f"Done. Processed {i} frames -> {args.out_dir}")

if __name__ == "__main__":
    main()
