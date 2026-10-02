"""Learned bounce detector: leave-one-rally-out comparison on rallies with analyst-marked bounces.

    PYTHONPATH=. python3 training/bounce_model_cv.py

Candidates = flight breaks + 3D-fit contacts + the pipeline's own bounces + image-y turning
points of the ball. Clues = the hit clues (swing, 2D ball motion, ball-to-player, 3D, paddle)
plus where the ball meets the ground (court x/y, inside the court). Swing clues come from
swing models trained without the test rally. Scored against the current pipeline bounces.
"""
import json, time
from pathlib import Path

import numpy as np

import pickleball.hitmodel as hm
from training.evaluate import match
from training.hit_model_cv import MODELS, RUNS, TOL, Stretch, fit, pick_threshold, train_swing

ROOT = Path(__file__).resolve().parents[1]


def checked(r):
    return r.get("mode") == "hits+bounces" or (r.get("mode") is None and any(e["kind"] == "bounce" for e in r["events"]))


def bounce_candidates(S, r, sp, pipeline_bounces):
    return hm.bounce_candidates(S.ctx, sp, r["start"] - 5, r["end"] + 5, pipeline_bounces)


def bounce_row(S, f, sp):
    return hm.bounce_row(S.ctx, f, sp)


def main():
    t0 = time.time()
    S = [Stretch(r) for r in RUNS]
    items = [(s, r) for s in S for r in s.rallies]
    swing_rows = [s.swing_xy(r) for s, r in items]
    test = [k for k, (_, r) in enumerate(items) if checked(r)]
    print(f"{len(test)} rallies with marked bounces, {sum(sum(e['kind'] == 'bounce' for e in items[k][1]['events']) for k in test)} bounces ({time.time() - t0:.0f}s)", flush=True)
    rows, truth_by, base_pred = [], {}, {}
    cover = n_truth = 0
    for k in test:
        s, r = items[k]
        m = train_swing([swing_rows[j] for j in range(len(items)) if j != k])
        sp = {}
        for side in ("near", "far"):
            fs = [f for f in s.sf[side] if r["start"] - 15 <= f <= r["end"] + 15]
            sp[side] = dict(zip(fs, m.predict_proba(np.array([s.sf[side][f] for f in fs]))[:, 1])) if fs else {}
        A = json.loads((s.run / "analysis.json").read_text())
        pb = [b["frame"] for b in A["bounces"]]
        base_pred[k] = [f for f in pb if r["start"] <= f <= r["end"]]
        cs = bounce_candidates(s, r, sp, pb)
        truth = [e["frame"] for e in r["events"] if e["kind"] == "bounce"]
        truth_by[k] = truth
        pos = {cs[j] for _, j in match(truth, cs, TOL)}
        cover, n_truth = cover + len(pos), n_truth + len(truth)
        for f in cs:
            rows.append((k, f, bounce_row(s, f, sp), int(f in pos)))
    print(f"{len(rows)} candidates cover {cover}/{n_truth} marked bounces (the ceiling) ({time.time() - t0:.0f}s)", flush=True)
    g = np.array([r[0] for r in rows]); fr = np.array([r[1] for r in rows])
    X = np.array([r[2] for r in rows], float); y = np.array([r[3] for r in rows])

    def score(pred_by):
        tp = fp = fn = 0
        for k in test:
            m = len(match(truth_by[k], pred_by[k], TOL))
            tp, fp, fn = tp + m, fp + len(pred_by[k]) - m, fn + len(truth_by[k]) - m
        return tp, fp, fn

    res = {"Current system": score(base_pred)}
    for name in MODELS:
        pred = {}
        for k in test:
            tr, te = g != k, g == k
            thr = pick_threshold(name, X[tr], y[tr], g[tr], fr[tr])
            pred[k] = hm.nms_pick(fr[te], fit(name, X[tr], y[tr]).predict_proba(X[te])[:, 1], thr)
        res[name] = score(pred)
        tp, fp, fn = res[name]
        print(f"  {name:40s} found {tp}/{tp + fn}  false {fp}  F1 {2 * tp / max(2 * tp + fp + fn, 1):.0%} ({time.time() - t0:.0f}s)", flush=True)
    print(f"\n{'bounce model':40s} {'found':>10s} {'real':>6s} {'F1':>5s}")
    out = []
    for name, (tp, fp, fn) in sorted(res.items(), key=lambda kv: -2 * kv[1][0] / max(2 * kv[1][0] + kv[1][1] + kv[1][2], 1)):
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        print(f"{name:40s} {tp:>4d}/{tp + fn:<4d} {tp / max(tp + fp, 1):>6.0%} {f1:>5.0%}")
        out.append({"model": name, "found": tp, "truth": tp + fn, "false": fp, "f1": f1})
    (ROOT / "outputs" / "bounce_model_cv.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
