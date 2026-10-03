"""Robust (outlier-resistant) 3D fitting on vs off, both matches, label-free quality measures.

    PYTHONPATH=. python3 training/compare_robust_arcs.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

import pickleball.ball3d as B
from pickleball.analysis import load_detections
from pickleball.ball import track_ball
from pickleball.camera3d import calibrate_from_run
from pickleball.speed import match_speeds

ROOT = Path(__file__).resolve().parents[1]
rows = []
for run in ("full_match", "lv_fahey_yoshitomi"):
    d = ROOT / "outputs" / run
    meta, recs = load_detections(d)
    A = json.loads((d / "analysis.json").read_text())
    P, ball = calibrate_from_run(d).P, track_ball(recs)
    for robust in (False, True):
        B.ROBUST = robust
        arcs = [a for v in B.match_arcs(A, ball, P).values() for a in v]
        sp = match_speeds(A, ball, P)
        n_shots = len([s for s in A["shots"] if not s.get("inferred")])
        conf = Counter(v["conf"] for v in sp.values())
        rows.append((run, "robust" if robust else "plain", len(arcs), 100 * np.mean([a["confident"] for a in arcs]),
                     float(np.median([a["err_px"] for a in arcs if a["err_px"] is not None])),
                     100 * len(sp) / n_shots, 100 * conf["high"] / max(len(sp), 1)))
print(f"{'match':20s} {'fit':7s} {'arcs':>5s} {'trusted':>8s} {'err px':>7s} {'speed':>7s} {'both agree':>11s}")
for r in rows:
    print(f"{r[0]:20s} {r[1]:7s} {r[2]:>5d} {r[3]:>7.1f}% {r[4]:>7.1f} {r[5]:>6.1f}% {r[6]:>10.1f}%")
