"""Ball detection accuracy on the human-verified ground truth (dashboard/balltruth.html -> labels/balltruth_<clip>.json).

    PYTHONPATH=. python3 training/eval_balltruth.py [--pred name=path.json ...] [--px 4]
Always scores the current system (the tracked ball shown as the proposal). --pred adds models: a JSON
{clip: {frame: [x, y] | null}} in 1280x720 pixels.
Metric (TrackNet / WASB convention): true positive = prediction within `px` of the ball; a prediction too far away
counts as a false positive AND a miss; a prediction on a 'no ball visible' frame is a false positive. Precision,
recall, F1 per clip group and overall; 95% bootstrap intervals by resampling clips.
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--pred", nargs="*", default=[])
ap.add_argument("--px", type=float, default=4.0)
ap.add_argument("--thr", type=float, nargs="*", default=[None], help="confidence thresholds for --pred files with confidences")
a = ap.parse_args()
Q = json.loads((ROOT / "labels" / "balltruth" / "queue.json").read_text())["clips"]
GROUP = lambda c: "TNPPL (indoor, Chennai)" if c.startswith("tnppl") else ("PPA Europe" if c.startswith("europe") else
                  ("US Open" if c.startswith("full_match") else "PPA Las Vegas"))
truth = {}
for c in Q:
    p = ROOT / "labels" / f"balltruth_{c['clip']}.json"
    if p.exists():
        truth[c["clip"]] = {int(f): v for f, v in json.loads(p.read_text())["frames"].items()}
systems = {"current system (YOLO11 + path search)": {c["clip"]: {f["frame"]: f["proposal"] for f in c["frames"]} for c in Q}}
for spec in a.pred:
    name, path = spec.split("=", 1)
    raw = json.loads(Path(path).read_text())
    for thr in a.thr:
        keep = lambda v: v is not None and (thr is None or len(v) < 3 or v[2] >= thr)
        systems[name + ("" if thr is None else f" @ thr {thr}")] = {k: {int(f): (v if keep(v) else None) for f, v in d.items()} for k, d in raw.items()}


def counts(pred, gt):
    tp = fp = fn = 0
    for f, g in gt.items():
        p = pred.get(f)
        has = g.get("x") is not None
        if has and p is not None and math.hypot(p[0] - g["x"], p[1] - g["y"]) <= a.px:
            tp += 1
        else:
            fp += p is not None
            fn += has
    return np.array([tp, fp, fn])


def prf(c):
    tp, fp, fn = c
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


truth = {c: v for c, v in truth.items() if v}
if not truth:
    raise SystemExit("no ground-truth labels yet (dashboard/balltruth.html)")
n_frames = sum(len(v) for v in truth.values())
print(f"ground truth: {len(truth)} clips, {n_frames} frames, ball visible in {sum(1 for v in truth.values() for g in v.values() if g.get('x') is not None)}")
rng = np.random.default_rng(0)
for name, P in systems.items():
    per = {c: counts(P.get(c, {}), g) for c, g in truth.items()}
    groups = defaultdict(lambda: np.zeros(3, int))
    for c, v in per.items():
        groups[GROUP(c)] += v
    tot = sum(per.values(), np.zeros(3, int))
    clips = list(per)
    boot = [prf(sum((per[clips[i]] for i in rng.integers(0, len(clips), len(clips))), np.zeros(3, int)))[2] for _ in range(1000)]
    p, r, f = prf(tot)
    print(f"\n{name}: precision {p:.1%} recall {r:.1%} F1 {f:.1%} (95% CI {np.percentile(boot, 2.5):.1%}-{np.percentile(boot, 97.5):.1%})")
    for g, v in sorted(groups.items()):
        p, r, f = prf(v)
        print(f"   {g:26s} P {p:.1%}  R {r:.1%}  F1 {f:.1%}  (tp {v[0]}, fp {v[1]}, fn {v[2]})")
