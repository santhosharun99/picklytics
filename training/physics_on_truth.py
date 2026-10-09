"""Physics reconstruction (pickleball/rally3d.py) on the HAND-CLICKED ball positions (labels/balltruth_*.json):
tests the physics layer alone, without detector errors. Every 2nd frame is labelled; 1280x720 -> source pixels.

    PYTHONPATH=. python3 training/physics_on_truth.py
Reports per clip: fit rms (px, at source resolution), share of clicked positions explained (< 6 px), junctions found
(contacts / bounces) vs the pipeline's detected hits, and whether every junction is physically plausible.
"""
import json
from pathlib import Path

import numpy as np

from pickleball.camera3d import calibrate_from_run
from pickleball.rally3d import reconstruct

ROOT = Path(__file__).resolve().parents[1]
Q = json.loads((ROOT / "labels" / "balltruth" / "queue.json").read_text())["clips"]
rows = []
for c in Q:
    run = ROOT / "outputs" / c["run"]
    A = json.loads((run / "analysis.json").read_text())
    fps = A["meta"]["fps"]
    sx = A["meta"]["width"] / c["size"][0]
    sy = A["meta"]["height"] / c["size"][1]
    T = json.loads((ROOT / "labels" / f"balltruth_{c['clip']}.json").read_text())["frames"]
    ball = {int(f): [v["x"] * sx, v["y"] * sy, False] for f, v in T.items() if v.get("x") is not None}
    if len(ball) < 10:
        continue
    cam = calibrate_from_run(run)
    pos = {int(round(p["t"] * fps)): [tuple(v) for v in p["p"].values()] for p in A.get("positions", []) if p["p"]}
    players = {f + d: v for f, v in pos.items() for d in range(-2, 3)}
    f0, f1 = min(ball), max(ball)
    hits = [s["frame"] for s in A["shots"] if f0 <= s["frame"] <= f1 and not s.get("inferred")]
    fit = reconstruct(cam, ball, f0, f1, fps, players)
    if fit is None:
        continue
    J = fit.junctions
    nc = sum(j.kind == "contact" for j in J)
    nb = sum(j.kind == "bounce" for j in J)
    rows.append((c["clip"], round(fit.rms_px, 1), round(float(fit.inliers.mean()), 2), nc, nb, len(hits)))
    print(f"{c['clip']:26s} rms {fit.rms_px:5.1f} px | explained {fit.inliers.mean():4.0%} | contacts {nc:2d} bounces {nb:2d} "
          f"| pipeline hits {len(hits):2d} | camera err {cam.err_px:.1f} px", flush=True)
r = np.array([x[1:] for x in rows], float)
print(f"\nmedian rms {np.median(r[:, 0]):.1f} px, explained {np.mean(r[:, 1]):.0%}; contacts {int(r[:, 2].sum())} vs pipeline hits {int(r[:, 4].sum())}; bounces {int(r[:, 3].sum())}")
