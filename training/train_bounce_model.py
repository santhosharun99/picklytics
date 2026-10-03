"""Train the bounce detector on all rallies with marked bounces -> models/bounce_clf.joblib.

    PYTHONPATH=. python3 training/train_bounce_model.py [--model "Logistic Regression"]
Same out-of-rally swing clues and candidates as bounce_model_cv.py.
"""
import argparse, json

import joblib
import numpy as np

import pickleball.hitmodel as hm
from training.bounce_model_cv import checked
from training.evaluate import match
from training.hit_model_cv import MODELS, TOL, Stretch, fit, pick_threshold, train_swing
from training.cross_match_cv import SOURCES

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Logistic Regression", choices=list(MODELS))
a = ap.parse_args()
S = [Stretch(run, json.loads(lab.read_text())) for _, run, lab in SOURCES]  # all tagged matches
items = [(s, r) for s in S for r in s.rallies]
swing_rows = [s.swing_xy(r) for s, r in items]
X, y, g, fr = [], [], [], []
for k, (s, r) in enumerate(items):
    if not checked(r):
        continue
    m = train_swing([swing_rows[j] for j in range(len(items)) if j != k])
    sp = {}
    for side in ("near", "far"):
        fs = [f for f in s.sf[side] if r["start"] - 15 <= f <= r["end"] + 15]
        sp[side] = dict(zip(fs, m.predict_proba(np.array([s.sf[side][f] for f in fs]))[:, 1])) if fs else {}
    pb = [b["frame"] for b in json.loads((s.run / "analysis.json").read_text())["bounces"]]
    cs = hm.bounce_candidates(s.ctx, sp, r["start"] - 5, r["end"] + 5, pb)
    truth = [e["frame"] for e in r["events"] if e["kind"] == "bounce"]
    pos = {cs[j] for _, j in match(truth, cs, TOL)}
    for f in cs:
        X.append(hm.bounce_row(s.ctx, f, sp)); y.append(int(f in pos)); g.append(k); fr.append(f)
X, y, g, fr = np.array(X, float), np.array(y), np.array(g), np.array(fr)
thr = pick_threshold(a.model, X, y, g, fr)
joblib.dump({"model": fit(a.model, X, y), "threshold": float(thr), "name": a.model,
             "trained_on": f"{len(set(g))} rallies, {int(y.sum())} bounces"}, hm.BOUNCE_PATH)
print(f"saved {hm.BOUNCE_PATH} ({a.model}, threshold {thr:.2f}, {len(set(g))} rallies, {int(y.sum())} bounces)")
