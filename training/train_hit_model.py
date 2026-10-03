"""Train the hit decider on all labelled rallies -> models/hit_clf.joblib.

    PYTHONPATH=. python3 training/train_hit_model.py [--model "Random Forest"] [--clues +3D+paddle]

Swing clues for each rally come from a swing model trained WITHOUT that rally (as in
hit_model_cv.py), so the decider learns from the same kind of swing scores it gets on new video.
Needs outputs/<run>/paddles.json for the labelled rallies (training/detect_paddles.py).
"""
import argparse, json

import joblib
import numpy as np

import pickleball.hitmodel as hm
from training.evaluate import match
from training.hit_model_cv import MODELS, Stretch, TOL, fit, pick_threshold, train_swing
from training.cross_match_cv import SOURCES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Random Forest", choices=list(MODELS))
    ap.add_argument("--clues", default="+3D+paddle", choices=["base", "+3D", "+paddle", "+3D+paddle"])
    a = ap.parse_args()
    S = [Stretch(run, json.loads(lab.read_text())) for _, run, lab in SOURCES]  # all tagged matches
    items = [(s, r) for s in S for r in s.rallies]
    swing_rows = [s.swing_xy(r) for s, r in items]
    X, y, g, fr = [], [], [], []
    for k, (s, r) in enumerate(items):
        m = train_swing([swing_rows[j] for j in range(len(items)) if j != k])
        sp = {}
        for side in ("near", "far"):
            fs = [f for f in s.sf[side] if r["start"] - 15 <= f <= r["end"] + 15]
            sp[side] = dict(zip(fs, m.predict_proba(np.array([s.sf[side][f] for f in fs]))[:, 1])) if fs else {}
        cs = hm.candidates(s.ctx, sp, r["start"] - 5, r["end"] + 5)
        truth = [e["frame"] for e in r["events"] if e["kind"] == "hit"]
        pos = {cs[j] for _, j in match(truth, cs, TOL)}
        for f in cs:
            X.append(hm.row(s.ctx, f, sp, a.clues))
            y.append(int(f in pos))
            g.append(k)
            fr.append(f)
    X, y, g, fr = np.array(X, float), np.array(y), np.array(g), np.array(fr)
    thr = pick_threshold(a.model, X, y, g, fr)
    model = fit(a.model, X, y)
    joblib.dump({"model": model, "threshold": float(thr), "clues": a.clues, "name": a.model,
                 "trained_on": f"{len(items)} rallies, {int(y.sum())} hits"}, hm.MODEL_PATH)
    print(f"saved {hm.MODEL_PATH} ({a.model}, {a.clues}, threshold {thr:.2f}, {len(y)} candidates, {int(y.sum())} hits)")


if __name__ == "__main__":
    main()
