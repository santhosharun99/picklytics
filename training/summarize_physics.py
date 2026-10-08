"""Summary of the physics pass per match (outputs/<run>/rally3d.json) and comparison with what reports use now.

    PYTHONPATH=. python3 training/summarize_physics.py [runs ...]
Per match: rallies / shots reconstructed, share physically plausible ('ok'), speeds (serves vs other shots),
effective gravity (spin) by shot, landing spots found, how points ended; and against analysis.json: speed vs the
speed module (ball_mph), landing spot vs the 3D lifter (land_3d), for the same shots.
"""
import json, sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
runs = sys.argv[1:] or sorted(p.parent.name for p in (ROOT / "outputs").glob("*/rally3d.json"))
print(f"{'match':22s} {'rallies':>7s} {'shots':>6s} {'ok':>5s} {'serve mph':>9s} {'other mph':>9s} {'g serve':>7s} "
      f"{'g other':>7s} {'landed':>6s} | {'mph vs now':>10s} {'land vs lifter':>14s} | endings")
for run in runs:
    rd = ROOT / "outputs" / run
    P = json.loads((rd / "rally3d.json").read_text())
    A = json.loads((rd / "analysis.json").read_text())
    shots = {s["frame"]: s for s in A["shots"]}
    first = {}
    for s in sorted(A["shots"], key=lambda s: s["frame"]):
        if not s.get("inferred"):
            first.setdefault(s["rally"], s["frame"])
    serve_frames = set(first.values())
    ok = {int(f): r for f, r in P["shots"].items() if r.get("ok")}
    sv = [r for f, r in ok.items() if f in serve_frames]
    ot = [r for f, r in ok.items() if f not in serve_frames]
    med = lambda xs: f"{np.median(xs):.0f}" if xs else "-"
    landed = sum(1 for r in ok.values() if r.get("land"))
    d_mph, d_land = [], []
    for f, r in ok.items():
        s = shots.get(f)
        if not s:
            continue
        if s.get("ball_mph") and s.get("speed_conf") in ("high", "medium"):
            d_mph.append(r["speed_mph"] - s["ball_mph"])
        if r.get("land") and s.get("land_3d"):
            d_land.append(float(np.hypot(r["land"][0] - s["land_3d"][0], r["land"][1] - s["land_3d"][1])))
    end = Counter(x["ending"] for x in P["rallies"].values())
    print(f"{run:22s} {len(P['rallies']):7d} {len(P['shots']):6d} {len(ok) / max(len(P['shots']), 1):5.0%} "
          f"{med([r['speed_mph'] for r in sv]):>9s} {med([r['speed_mph'] for r in ot]):>9s} "
          f"{med([r['g_eff'] for r in sv]):>7s} {med([r['g_eff'] for r in ot]):>7s} {landed:6d} | "
          f"{(f'{np.median(d_mph):+.0f} (n{len(d_mph)})' if d_mph else '-'):>10s} "
          f"{(f'{np.median(d_land):.1f} ft (n{len(d_land)})' if d_land else '-'):>14s} | {dict(end)}")
