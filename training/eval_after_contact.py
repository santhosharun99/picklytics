"""Check the after-contact 3D follower (pickleball/after_contact.py).

    PYTHONPATH=. python3 training/eval_after_contact.py [run ...]
1. Without labels: a contact followed by a contact from the OTHER side within 3 s must have cleared the net ->
   how often the fitted flight says 'crossed'.
2. With labels (labels/endings.json): for the last live contact of each labelled rally, net vs crossed, and for
   'in' / 'out' endings the landing point's distance to the line.
Runs one match at a time (laptop overheats with parallel video jobs).
"""
import json, sys
from collections import Counter
from pathlib import Path

import numpy as np

from pickleball.after_contact import follow
from pickleball.camera3d import calibrate_from_run

ROOT = Path(__file__).resolve().parents[1]
ANS = json.loads((ROOT / "labels" / "endings.json").read_text())["answers"]
runs = sys.argv[1:] or sorted({a["run"] for a in ANS.values()})
mid = Counter()
lab = []
for rid in runs:
    run = ROOT / "outputs" / rid
    A = json.loads((run / "analysis.json").read_text())
    fps = A["meta"]["fps"]
    ball = {int(k): v for k, v in json.loads((run / "render_data.json").read_text())["ball"].items()}
    cam = calibrate_from_run(run)
    by = {}
    for s in A["shots"]:
        if not s.get("inferred") and s.get("hitter_court"):
            by.setdefault(s["rally"], []).append(s)
    m = Counter()
    for rally, sh in by.items():
        sh.sort(key=lambda s: s["frame"])
        for a, b in zip(sh, sh[1:]):
            if a["side"] != b["side"] and (b["frame"] - a["frame"]) / fps < 3.0:
                o = follow(cam, ball, a["frame"], b["frame"], a["hitter_court"], a["side"], fps)
                if o["crossed"] is not None and o["fit_err_px"] is not None and o["fit_err_px"] < 25:
                    m["checked"] += 1
                    m["crossed"] += o["crossed"]
    mid += m
    print(f"{rid}: camera error {cam.err_px:.1f} px | mid-rally shots answered by the other side: fitted flight clears the net {m['crossed']}/{m['checked']}")
    for key, ans in ANS.items():
        if ans["run"] != rid or ans.get("ending") not in ("net", "out", "in"):
            continue
        sh = [s for s in by.get(ans["rally"], []) if s["t"] <= ans["end_t"] + 0.2 and
              (ans.get("serve_t") is None or s["t"] >= ans["serve_t"] - 0.6)]
        if not sh:
            continue
        last = sh[-1]
        o = follow(cam, ball, last["frame"], None, last["hitter_court"], last["side"], fps)
        lab.append((ans["ending"], o))
print(f"\nALL mid-rally: clears the net {mid['crossed']}/{mid['checked']} ({mid['crossed'] / max(mid['checked'], 1):.0%})")
ok = [(e, o) for e, o in lab if o["crossed"] is not None]
print(f"labelled endings with a fitted flight: {len(ok)}/{len(lab)}")
print("  truth vs fitted 'crossed':", Counter((e, o["crossed"]) for e, o in ok))
io = [(e, o["land_line_ft"]) for e, o in ok if e in ("in", "out") and o["land_line_ft"] is not None]
print("  in/out from the fitted landing point:", Counter((e, "in" if d >= 0 else "out") for e, d in io))
print("  distances (truth, ft from line):", sorted((e, d) for e, d in io))
