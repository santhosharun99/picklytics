"""Events from the physics reconstruction (pickleball/rally3d.py) vs hand-tagged hits / bounces.

    PYTHONPATH=. python3 training/eval_rally3d.py [--tol 3]
Tagged rallies (mode hits+bounces) of labels/Mens_Singles_GOLD_2026_US_Open.json (run outputs/full_match). Two set-ups:
  A. contacts = our detected hits, the fit decides which other breaks are bounces -> bounce F1 vs the tags, compared
     with the current bounce detector;
  B. nothing given: every junction typed by the fit -> hit F1 and bounce F1;
  C. initial junctions = our detected hits + bounces, then rally3d.refine() (time shifts, insertions, removals);
  D. as C on the live window only (rally-end model: serve .. last live touch) + rally3d.end_of_play (ball dead after
     the first bounce that follows the last contact).
--ball: a ball track {frame: [x, y, conf]} from training/balltrack_run.py instead of the run's render_data.json.
"""
import argparse, json
from pathlib import Path

import numpy as np

from pickleball.camera3d import calibrate_from_run
from pickleball.rally3d import end_of_play, reconstruct, refine
from pickleball import rally_end

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--tol", type=int, default=3)
ap.add_argument("--labels", default="labels/Mens_Singles_GOLD_2026_US_Open.json")
ap.add_argument("--run", default="outputs/full_match")
ap.add_argument("--lam", nargs=2, type=float, default=[30.0, 20.0], help="price per contact, per bounce")
ap.add_argument("--only-b", action="store_true", help="only set-up B (no hits given)")
ap.add_argument("--ball", help="ball track json {frame: [x, y, conf]} (default: run's render_data.json)")
ap.add_argument("--thr", type=float, default=0.3, help="min confidence for --ball detections")
ap.add_argument("--refine-lam", type=float, default=60.0)
ap.add_argument("--setups", default="ABCD", help="which set-ups to run (A, B: reconstruct; C: refine)")
ap.add_argument("--max-frames", type=int, default=1500, help="skip longer tagged stretches (dead time between points)")
ap.add_argument("--out", help="per-rally results json (written after every rally)")
a = ap.parse_args()
L = json.loads((ROOT / a.labels).read_text())
run = ROOT / a.run
A = json.loads((run / "analysis.json").read_text())
fps = A["meta"]["fps"]
if a.ball:
    ball = {int(k): [v[0], v[1], False] for k, v in json.loads((ROOT / a.ball).read_text()).items() if v[2] >= a.thr}
else:
    ball = {int(k): v for k, v in json.loads((run / "render_data.json").read_text())["ball"].items()}
cam = calibrate_from_run(run)
pos = {}
for p in A.get("positions", []):
    pos[int(round(p["t"] * fps))] = [tuple(v) for v in p["p"].values()]
players = {}
for f0 in pos:
    for d in range(-2, 3):
        players[f0 + d] = pos[f0]


def match(pred, true, tol):
    true, used, tp = list(true), set(), 0
    for p in pred:
        c = [i for i, t in enumerate(true) if i not in used and abs(t - p) <= tol]
        if c:
            used.add(min(c, key=lambda i: abs(true[i] - p))); tp += 1
    return tp, len(pred) - tp, len(true) - tp


def f1(c):
    tp, fp, fn = c
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return f"P {p:.0%} R {r:.0%} F1 {2 * p * r / max(p + r, 1e-9):.0%} (tp {tp}, fp {fp}, fn {fn})"


tot = {k: np.zeros(3, int) for k in ("A_bounce", "old_bounce", "B_hit", "B_bounce", "old_hit", "C_hit", "C_bounce", "D_hit", "D_bounce")}
dead = rally_end.dead_touches(A, run) if "D" in a.setups else {}
live_frames = sorted(x["frame"] for x in A["shots"] if (x["rally"], x["index"]) not in dead and not x.get("inferred"))
n, recs = 0, []
for r in L["rallies"]:
    if r.get("mode") != "hits+bounces" or not r.get("reviewed", True):
        continue
    s, e = int(r["start"]), int(r["end"])
    if e <= s:
        continue
    hits_t = [ev["frame"] for ev in r["events"] if ev["kind"] == "hit"]
    bnc_t = [ev["frame"] for ev in r["events"] if ev["kind"] == "bounce"]
    our_hits = [x["frame"] for x in A["shots"] if s <= x["frame"] <= e and not x.get("inferred")]
    old_b = [b["frame"] for b in A["bounces"] if s <= b["frame"] <= e]
    if e - s > a.max_frames:
        print(f"rally {r['id']}: {e - s} frames - skipped (> --max-frames)", flush=True)
        continue
    fr = [f for f in range(s, e + 1) if f in ball and not ball[f][2]]
    if len(fr) < 8:
        continue
    n += 1
    tot["old_bounce"] += match(old_b, bnc_t, a.tol)
    tot["old_hit"] += match(our_hits, hits_t, a.tol)
    rec = {"id": r["id"], "hits": hits_t, "bounces": bnc_t}
    kw = dict(lam_contact=a.lam[0], lam_bounce=a.lam[1])
    if "B" in a.setups:
        fb = reconstruct(cam, ball, s, e, fps, players, **kw)
        tot["B_hit"] += match([j.frame for j in fb.junctions if j.kind == "contact"], hits_t, a.tol)
        tot["B_bounce"] += match([j.frame for j in fb.junctions if j.kind == "bounce"], bnc_t, a.tol)
        rec["B"] = [(j.kind, j.frame) for j in fb.junctions]
    if "A" in a.setups:
        fa = reconstruct(cam, ball, s, e, fps, players, contact_frames=our_hits, **kw)
        tot["A_bounce"] += match([j.frame for j in fa.junctions if j.kind == "bounce"], bnc_t, a.tol)
    if "C" in a.setups:
        init = sorted({(f, "contact") for f in our_hits if fr[0] < f < fr[-1]} | {(f, "bounce") for f in old_b if fr[0] < f < fr[-1]})
        init = [x for i, x in enumerate(init) if i == 0 or x[0] != init[i - 1][0]]  # one junction per frame
        jf = [fr[0]] + [f for f, _ in init] + [fr[-1]]
        fc = refine(cam, fr, np.array([ball[f][:2] for f in fr], float), fps, jf, ["start"] + [k for _, k in init] + ["end"],
                    players, lam=a.refine_lam)
        tot["C_hit"] += match([j.frame for j in fc.junctions if j.kind == "contact"], hits_t, a.tol)
        tot["C_bounce"] += match([j.frame for j in fc.junctions if j.kind == "bounce"], bnc_t, a.tol)
        rec["C"] = [(j.kind, j.frame, [round(float(v), 2) for v in j.xyh]) for j in fc.junctions]
        print(f"rally {r['id']}: fit rms {fc.rms_px:.1f} px, inliers {fc.inliers.mean():.0%}", flush=True)
    if "D" in a.setups:
        lv = [f for f in live_frames if s <= f <= e]
        if lv:
            w0, w1 = lv[0], min(e, lv[-1] + int(2.0 * fps))  # serve .. last live touch + 2 s (end_of_play cuts)
            frd = [f for f in fr if w0 <= f <= w1]
            pxd = np.array([ball[f][:2] for f in frd], float)
            init = sorted({(f, "contact") for f in lv if frd[0] < f < frd[-1]} | {(f, "bounce") for f in old_b if frd[0] < f < frd[-1]})
            init = [x for i, x in enumerate(init) if i == 0 or x[0] != init[i - 1][0]]
            fd = refine(cam, frd, pxd, fps, [frd[0]] + [f for f, _ in init] + [frd[-1]], ["start"] + [k for _, k in init] + ["end"],
                        players, lam=a.refine_lam, anchors=lv)
            fd = end_of_play(cam, frd, pxd, fps, fd, players, last_contact=lv[-1])
            # the window starts at the serve: the 'start' junction there is the serve contact
            dj = [("contact" if (j.kind == "start" and abs(j.frame - lv[0]) <= 3) else j.kind, j.frame) for j in fd.junctions]
            tot["D_hit"] += match([f for k, f in dj if k == "contact"], hits_t, a.tol)
            tot["D_bounce"] += match([f for k, f in dj if k == "bounce"], bnc_t, a.tol)
            rec["D"] = [(k, f) for k, f in dj]
            rec["D_g"] = fd.g
    recs.append(rec)
    if a.out:
        (ROOT / a.out).write_text(json.dumps(recs))
print(f"\n{n} tagged rallies, tolerance +-{a.tol} frames")
print("bounces - current detector        :", f1(tot["old_bounce"]))
print("bounces - physics fit (A, hits given):", f1(tot["A_bounce"]))
print("hits    - current detector        :", f1(tot["old_hit"]))
print("hits    - physics fit alone (B)   :", f1(tot["B_hit"]))
print("bounces - physics fit alone (B)   :", f1(tot["B_bounce"]))
print("hits    - detected + refine (C)   :", f1(tot["C_hit"]))
print("bounces - detected + refine (C)   :", f1(tot["C_bounce"]))
print("hits    - live window + end (D)   :", f1(tot["D_hit"]))
print("bounces - live window + end (D)   :", f1(tot["D_bounce"]))
