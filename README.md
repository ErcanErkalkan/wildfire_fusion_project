# wildfire-fusion

**Thermal–RGB Fusion for Micro‑UAV Wildfire Perimeter Tracking (Minimal Comms)**

This repository provides a **reference Python implementation** of a lightweight perimeter‑tracking pipeline that fuses **Thermal + RGB** frames for **micro‑UAV teams** operating under **minimal communications**. The goals are: approach **sub‑50 ms** loop times on embedded SoCs, keep parameters **GSD‑aware**, and broadcast compact **delta‑coded polylines** over **single‑hop** links.

**Method overview:** build a coarse **hot‑region mask** from the thermal frame (Otsu threshold + morphology); extract **edges** from RGB (Canny); **gate** edges inside a narrow band around the thermal rim; simplify the resulting contour using **Ramer–Douglas–Peucker (RDP)**; output **tangent‑follow** guidance hints and **delta‑coded** messages.

---

## Installation

```bash
# 1) (Recommended) virtualenv
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install
pip install -r requirements.txt
# or as an editable package:
pip install -e .
```

---

## Quick Start

### 1) Run on paired frame folders
```bash
python scripts/run_on_folder.py   --thermal_dir data/thermal_frames   --rgb_dir data/rgb_frames   --out_dir results   --gsd 0.20   --rdp_k 1.0   --band_c 3.0
```

### 2) Run on a video (demo with thermal proxy from V channel)
```bash
python scripts/run_on_video.py --video path/to/input.mp4 --out_dir results --gsd 0.20
```

**Parameters**
- `--gsd` — **Ground Sample Distance** (meters/pixel). Used to reason about scales, but the core thresholds below are in pixels.
- `--rdp_k` — RDP tolerance in **pixels** (ε = `rdp_k`).
- `--band_c` — Band half‑width around the thermal rim in **pixels** used to **gate edges** (distance transform ≤ `band_c`).

**Outputs**
- `results/overlays/` — composite visualizations (thermal mask, edges, band, simplified polyline).
- `results/polylines/*.json` — time‑stamped polyline(s).
- `results/messages/*.json` — **delta‑coded** compact messages for single‑hop broadcast.

---

## Project Structure

```
wildfire_fusion/
  ├─ src/wildfire_fusion/
  │   ├─ __init__.py
  │   ├─ pipeline.py
  │   ├─ rdp.py
  │   ├─ guidance.py
  │   ├─ messaging.py
  │   └─ utils.py
  ├─ scripts/
  │   ├─ run_on_folder.py
  │   └─ run_on_video.py
  ├─ examples/config.yaml
  ├─ tests/
  ├─ LICENSE
  ├─ CITATION.cff
  ├─ pyproject.toml
  ├─ requirements.txt
  └─ README.md
```

---

## Notes & Design Decisions

- **16‑bit thermal input** is auto‑scaled to 8‑bit (`0..65535 → 0..255`) in `run_on_folder.py`.
- The **gating band** is defined in **pixels** using a distance transform (≤ `band_c`). This makes field tuning straightforward and portable.
- **RDP ε** is also in pixels (`rdp_k`), which is intuitive when inspecting image overlays.
- Guidance contains a minimal **tangent‑follow** stub; extend as needed (e.g., GNSS‑denied odometry, multi‑UAV tasking).
- Messages are **delta‑coded** JSON with a `scale` field; a CRC placeholder is included for later link‑layer integration.

---

## Citation

If you use this software or method, please cite the following (also see `CITATION.cff`):

> E. Erkalkan, V. Topuz, and A. Ak, “Thermal–RGB Fusion for Micro-UAV Wildfire Perimeter Tracking with Minimal Communications,” in Proc. 17th Int. Scientific Studies Congress (UBCAK 2025)—Full Text Book, H. Göker, Ed. Elazığ, Türkiye: Asos Yayınevi, 2025, pp. 278–285. ISBN 978-625-5909-68-8.

---

## License

MIT License (see `LICENSE`). Algorithm names/terms belong to their original publications.

**Key references:** Otsu (1979), Canny (1986), Ramer (1972), Douglas–Peucker (1973).
