"""Game-rule depth check: where do servers stand, per court end?

    PYTHONPATH=. python3 training/depth_check.py outputs/<run> [...]

Rule: a server stands just behind the baseline. Near baseline y=44, far baseline y=0.
Compares ground points from (a) the ankle keypoint on the ground (current), (b) the ankle ray
meeting a plane at ankle height, (c) the box bottom, using the 3D camera.
"""
import json, sys
from pathlib import Path

import cv2
import numpy as np

from pickleball.analysis import load_detections
from pickleball.camera3d import calibrate_from_run
from pickleball.players import select_players


def ray_to_height(cam, uv, h):
    """Pixel -> court (x, y) where its ray meets the plane at height h (ft)."""
    R, _ = cv2.Rodrigues(cam.rvec)
    C = (-R.T @ cam.tvec.reshape(3, 1)).ravel()                 # OpenCV world (z = -h)
    d = R.T @ np.linalg.inv(cam.K) @ np.array([uv[0], uv[1], 1.0])
    s = (-h - C[2]) / d[2]
    X = C + s * d
    return X[0], X[1]


def behind(side, y):
    return y - 44 if side == "near" else -y


for run in map(Path, sys.argv[1:]):
    meta, recs = load_detections(run)
    R = {r["f"]: r for r in recs}
    A = json.loads((run / "analysis.json").read_text())
    cam = calibrate_from_run(run)
    print(f"\n== {run.name}: camera at {np.round(cam.position, 1)} ft, err {cam.err_px:.1f}px")
    rows = {"near": [], "far": []}
    for s in A["shots"]:
        if s.get("shot_type") != "serve" or s.get("inferred"):
            continue
        # stance just before the swing (feet planted), median over a few frames
        vals = []
        for f in range(s["frame"] - 12, s["frame"] - 2, 2):
            r = R.get(f)
            if not r or not r.get("court"):
                continue
            p = {q["side"]: q for q in select_players(r.get("people", []), 1)}.get(s["side"])
            if not p:
                continue
            kp = np.array(p["kp"]); box = p["box"]
            ank = kp[[15, 16]]; vis = ank[:, 2] > 0.5
            if not vis.any():
                continue
            u, v = ank[vis, 0].mean(), ank[vis, 1].max()
            vals.append([p["court"][1], ray_to_height(cam, (u, v), 0)[1], ray_to_height(cam, (u, v), 0.25)[1],
                         ray_to_height(cam, ((box[0] + box[2]) / 2, box[3]), 0)[1], int(bool(p.get("zoom")))])
        if vals:
            rows[s["side"]].append(np.median(np.array(vals), 0))
    for side, v in rows.items():
        v = np.array(v)
        if not len(v):
            continue
        b = lambda k: np.median([behind(side, y) for y in v[:, k]])
        print(f"  {side:4s} servers n={len(v):2d}  ft behind baseline: current {b(0):+.1f}  ankle@0 {b(1):+.1f}  "
              f"ankle@0.25ft {b(2):+.1f}  box-bottom {b(3):+.1f}  zoomed {v[:, 4].mean():.0%}")
