"""How did the point end? Physics line calls vs the hand-labelled endings (labels/endings.json: net / out / in).

    PYTHONPATH=. python3 training/eval_linecalls.py [--runs tnppl_1_MS ...] [--oracle-window]
Per labelled rally of a run that has the new ball track (outputs/<run>/ball_v3.json from training/balltrack_run.py):
live window from the rally-end model (serve .. last live touch + 2 s; --oracle-window: the labelled serve / end
times instead), rally3d.refine (anchors = live hits) + end_of_play, then the final bounce after the last contact:
  - on the last hitter's side of the net (side of the tracked player who made the last live shot) -> 'net';
  - else rally3d.line_call on its position +- the fit's 1-sigma -> 'in' / 'out' / 'close' (too close to call);
  - a 'net' junction (ball stopped by the net, rally3d) first -> 'net';
  - no bounce after the last contact -> 'none' (ball lost).
Main-camera gate: frames with a main-camera score < 0.6 (<out>_main.json) are left out.
"""
import argparse, json
from collections import Counter
from pathlib import Path

import numpy as np

from pickleball import rally_end
from pickleball.camera3d import calibrate_from_run
from pickleball.court import NET_Y
from pickleball.rally3d import end_of_play, line_call, refine

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="*")
ap.add_argument("--thr", type=float, default=0.3)
ap.add_argument("--main", type=float, default=0.6)
ap.add_argument("--oracle-window", action="store_true")
ap.add_argument("--out", default="outputs/balltrack/linecalls.json")
a = ap.parse_args()
ans = json.loads((ROOT / "labels/endings.json").read_text())["answers"]
runs = sorted({v["run"] for v in ans.values()})
if a.runs:
    runs = [r for r in runs if r in a.runs]
conf, recs = Counter(), []
for run in runs:
    rd = ROOT / "outputs" / run
    bf = rd / "ball_v3.json"
    lab = {v["rally"]: v for v in ans.values() if v["run"] == run and v.get("ending") in ("net", "out", "in")}
    if not bf.exists() or not lab:
        continue
    A = json.loads((rd / "analysis.json").read_text())
    fps = A["meta"]["fps"]
    mf = rd / "ball_v3_main.json"
    main = {int(k): v for k, v in json.loads(mf.read_text()).items()} if mf.exists() else {}
    ball = {int(k): v for k, v in json.loads(bf.read_text()).items() if v[2] >= a.thr and main.get(int(k), 1.0) >= a.main}
    cam = calibrate_from_run(rd)
    players = {}
    for p in A.get("positions", []):
        f0 = int(round(p["t"] * fps))
        for d in range(-2, 3):
            players[f0 + d] = [tuple(v) for v in p["p"].values()]
    dead = rally_end.dead_touches(A, rd)
    rallies = {r["index"]: r for r in A["rallies"]}
    for idx, L in sorted(lab.items()):
        r = rallies.get(idx)
        if r is None:
            continue
        live = {s["frame"]: s for s in A["shots"] if s["rally"] == idx and not s.get("inferred") and (idx, s["index"]) not in dead}
        lv = sorted(live)
        if a.oracle_window:
            w0, w1 = int(L["serve_t"] * fps), int((L["end_t"] + 0.3) * fps)
            lv = [f for f in lv if w0 - 3 <= f <= w1]
        elif lv:
            w0, w1 = lv[0], min(int((r["end_t"] + 1.0) * fps), lv[-1] + int(2.0 * fps))
        if not lv:
            conf[(L["ending"], "no live hits")] += 1
            continue
        fr = [f for f in range(w0, w1 + 1) if f in ball]
        if len(fr) < 10:
            conf[(L["ending"], "no track")] += 1
            continue
        px = np.array([ball[f][:2] for f in fr], float)
        b = [x["frame"] for x in A.get("bounces", []) if fr[0] < x["frame"] < fr[-1]]
        init = sorted({(f, "contact") for f in lv if fr[0] < f < fr[-1]} | {(f, "bounce") for f in b})
        init = [x for i, x in enumerate(init) if i == 0 or x[0] != init[i - 1][0]]
        fit_ = refine(cam, fr, px, fps, [fr[0]] + [f for f, _ in init] + [fr[-1]], ["start"] + [k for _, k in init] + ["end"],
                      players, anchors=lv)
        hitter_far = live[lv[-1]]["side"] == "far" if live[lv[-1]].get("side") else None
        fit_ = end_of_play(cam, fr, px, fps, fit_, players, last_contact=lv[-1], hitter_far=hitter_far)
        J = fit_.junctions
        lc = [i for i, j in enumerate(J) if j.kind == "contact" or (j.kind == "start" and abs(j.frame - lv[0]) <= 3)]
        lc = [i for i in lc if J[i].frame <= lv[-1] + 3]
        own_ok = lambda i: hitter_far is None or not ((J[i].xyh[1] < NET_Y) == hitter_far and J[i].frame - J[lc[-1]].frame < 0.3 * fps)
        nb = next((i for i in range(lc[-1] + 1, len(J)) if J[i].kind == "net" or (J[i].kind == "bounce" and own_ok(i))), None) if lc else None
        if nb is None:
            pred, extra = "none", {}
        elif J[nb].kind == "net":
            pred, extra = "net", {"net_hit": [round(float(J[nb].xyh[0]), 2), round(float(J[nb].xyh[2]), 2)]}
        else:
            hitter_far = J[lc[-1]].xyh[1] < NET_Y if hitter_far is None else hitter_far
            bounce_far = J[nb].xyh[1] < NET_Y
            if hitter_far == bounce_far:
                pred, extra = "net", {}
            else:
                call, d_in, sd = line_call(J[nb].xyh, J[nb].sd)
                pred, extra = call, {"d_in_ft": round(float(d_in), 2), "sd_ft": round(float(sd), 2)}
            if J[nb].kind == "bounce":
                extra["bounce"] = [round(float(v), 2) for v in J[nb].xyh[:2]]
        conf[(L["ending"], pred)] += 1
        recs.append({"run": run, "rally": idx, "label": L["ending"], "pred": pred, **extra,
                     "junctions": [(j.kind, j.frame) for j in J], "rms_px": round(fit_.rms_px, 2)})
        print(f"{run}:{idx} label {L['ending']:4s} pred {pred:5s} {extra}", flush=True)
(ROOT / a.out).write_text(json.dumps(recs, indent=1))
print("\nlabel -> prediction counts:")
for lab_ in ("in", "out", "net"):
    row = {p: c for (l_, p), c in conf.items() if l_ == lab_}
    print(f"  {lab_:4s}", dict(sorted(row.items(), key=lambda x: -x[1])))
called = [r for r in recs if r["pred"] in ("in", "out", "net")]
ok = sum(r["pred"] == r["label"] for r in called)
print(f"called (in/out/net): {len(called)} of {len(recs)}, correct {ok} ({ok / max(len(called), 1):.0%})")
