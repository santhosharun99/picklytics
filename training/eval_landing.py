"""Landing spots vs hand tags: physics pass (rally3d.json 'land') and the 3D lifter (analysis.json 'land_3d').

    PYTHONPATH=. python3 training/eval_landing.py [--labels labels/Mens_Singles_GOLD_2026_US_Open.json --run full_match]
Truth: each tagged hit's next tagged bounce, whose pixel (on the floor) maps exactly to the court through the
camera's ground homography. Compared on the same shots (tagged hit within +-3 frames of the detector's hit).
"""
import argparse, json
from pathlib import Path

import numpy as np

from pickleball.camera3d import calibrate_from_run

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--labels", default="labels/Mens_Singles_GOLD_2026_US_Open.json")
ap.add_argument("--run", default="full_match")
a = ap.parse_args()
L = json.loads((ROOT / a.labels).read_text())
rd = ROOT / "outputs" / a.run
A = json.loads((rd / "analysis.json").read_text())
P = json.loads((rd / "rally3d.json").read_text())["shots"]
cam = calibrate_from_run(rd)
sw, sh = L.get("source_size", [A["meta"]["width"], A["meta"]["height"]])
sx, sy = A["meta"]["width"] / sw, A["meta"]["height"] / sh
Pm = cam.P[:, [0, 1, 3]]  # ground plane (h = 0): 3x3 homography court -> image
Hinv = np.linalg.inv(Pm)


def to_court(px):
    v = Hinv @ np.array([px[0] * sx, px[1] * sy, 1.0])
    return v[:2] / v[2]


shots = {s["frame"]: s for s in A["shots"] if not s.get("inferred")}
rows = []
for r in L["rallies"]:
    if r.get("mode") != "hits+bounces":
        continue
    ev = sorted(r["events"], key=lambda e: e["frame"])
    for i, e in enumerate(ev):
        if e["kind"] != "hit":
            continue
        nxt = next((x for x in ev[i + 1:] if x["kind"] in ("hit", "bounce")), None)
        if not nxt or nxt["kind"] != "bounce" or "xy" not in nxt:
            continue
        truth = to_court(nxt["xy"])
        f = min(shots, key=lambda g: abs(g - e["frame"])) if shots else None
        if f is None or abs(f - e["frame"]) > 3:
            continue
        ph, s = P.get(str(f)), shots[f]
        rows.append({"truth": truth,
                     "phys": ph["land"] if ph and ph.get("land") else None, "phys_ok": bool(ph and ph.get("ok")),
                     "lifter": s.get("land_3d"), "old2d": s.get("landing_court")})


def err(key, only_ok=False):
    d = [float(np.hypot(*(np.array(x[key]) - x["truth"]))) for x in rows if x.get(key) is not None and (not only_ok or x["phys_ok"])]
    return f"n {len(d):3d} | median {np.median(d):.2f} ft, 80% within {np.percentile(d, 80):.2f} ft" if d else "n 0"


print(f"{len(rows)} tagged shots with a tagged landing")
print("physics (all)     :", err("phys"))
print("physics (ok only) :", err("phys", True))
print("3D lifter         :", err("lifter"))
print("2D landing (old)  :", err("old2d"))
both = [x for x in rows if x["phys"] is not None and x["lifter"] is not None]
dp = [float(np.hypot(*(np.array(x["phys"]) - x["truth"]))) for x in both]
dl = [float(np.hypot(*(np.array(x["lifter"]) - x["truth"]))) for x in both]
if both:
    print(f"same {len(both)} shots: physics median {np.median(dp):.2f} ft vs lifter {np.median(dl):.2f} ft; physics closer on {np.mean(np.array(dp) < np.array(dl)):.0%}")
