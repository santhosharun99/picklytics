"""Train + test the rally-end model (pickleball/rally_end.py) on labels/endings.json.

    PYTHONPATH=. python3 training/train_rally_end.py
Leave-one-match-out cross-validation (the model never sees the match it is tested on), compared with the old
rules; then trains on everything -> models/rally_end.joblib.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pickleball import rally_end as RE

ROOT = Path(__file__).resolve().parents[1]
ANS = json.loads((ROOT / "labels" / "endings.json").read_text())["answers"]
runs = sorted({a["run"] for a in ANS.values()})
live_X, live_y, live_g, end_X, end_y, end_g, base_end, inout = [], [], [], [], [], [], [], []
live_r, end_gap = [], []
for rid in runs:
    A = json.loads((ROOT / "outputs" / rid / "analysis.json").read_text())
    rows = RE.touch_features(A, ROOT / "outputs" / rid)
    by = defaultdict(list)
    for r in rows:
        by[r["rally"]].append(r)
    for key, ans in ANS.items():
        if ans["run"] != rid or (not ans.get("no_rally") and ans.get("ending") == "unclear" and ans.get("end_t") is None):
            continue
        rr = sorted(by.get(ans["rally"], []), key=lambda r: r["t"])
        if not rr:
            continue
        ys = RE.labels_for(rr, ans)
        for r, y in zip(rr, ys):
            live_X.append([r[f] for f in RE.LIVE_FEATS]); live_y.append(y); live_g.append(rid); live_r.append(key)
        if ans.get("ending") in ("net", "out", "in"):
            lv = [r for r, y in zip(rr, ys) if y == 1]
            if not lv:
                continue
            last = lv[-1]
            shot = next(x for x in A["shots"] if x["rally"] == last["rally"] and x["index"] == last["index"])
            last = {**last, **RE.follow_feats(ROOT / "outputs" / rid, A, shot)}
            end_gap.append(ans["end_t"] - last["t"])
            end_X.append([last[f] for f in RE.END_FEATS]); end_y.append(ans["ending"]); end_g.append(rid)
            # old rules (outcome.rally_outcome): first bounce after the last hit decides
            base = "net" if last["b_own"] and not (last["b_opp_in"] or last["b_opp_out"]) else \
                ("out" if last["b_opp_out"] and not last["b_opp_in"] else "in")
            base_end.append(base)
            if ans["ending"] in ("in", "out") and (last["b_opp_in"] or last["b_opp_out"]):
                inout.append((ans["ending"], "out" if last["b_dist_line"] < 0 else "in", last["b_dist_line"]))
live_X, live_y, live_g = np.array(live_X), np.array(live_y), np.array(live_g)
end_X, end_y, end_g = np.array(end_X), np.array(end_y), np.array(end_g)
print(f"touches {len(live_y)} (live {live_y.sum()}, dead {(live_y == 0).sum()}); rally endings {Counter(end_y)}; matches {len(runs)}")


def lomo(make, X, y, g):
    pred = np.empty(len(y), dtype=object)
    prob = np.zeros(len(y))
    for m in sorted(set(g)):
        te = g == m
        if len(set(y[~te])) < 2:
            continue
        clf = make().fit(X[~te], y[~te])
        pred[te] = clf.predict(X[te])
        if len(clf.classes_) == 2:
            prob[te] = clf.predict_proba(X[te])[:, 1]
    return pred, prob


models = {"boosted trees": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=150, l2_regularization=1.0),
          "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))}
print("\nLIVE vs DEAD touch (leave one match out)")
dead_true = live_y == 0
print(f"  old pipeline (every touch counted): accuracy {live_y.mean():.0%}, dead touches caught 0/{dead_true.sum()}")
best = None
for name, mk in models.items():
    p, _ = lomo(mk, live_X, live_y, live_g)
    p = p.astype(int)
    acc = (p == live_y).mean()
    caught = ((p == 0) & dead_true).sum()
    wrong = ((p == 0) & ~dead_true).sum()
    print(f"  {name:13s} accuracy {acc:.0%} | dead touches caught {caught}/{dead_true.sum()} | live touches wrongly dropped {wrong}/{(~dead_true).sum()}")
    if best is None or acc > best[1]:
        best = (name, acc)
live_name = best[0]

# structured decoding: live play is ONE stretch per rally (serve .. last shot) -> best contiguous block
p_cv = lomo(models[live_name], live_X, live_y, live_g)[1]
dec = np.zeros(len(live_y), int)
live_r = np.array(live_r)
for key in dict.fromkeys(live_r):
    idx = np.where(live_r == key)[0]
    lp, ln = np.log(np.clip(p_cv[idx], 1e-4, 1)), np.log(np.clip(1 - p_cv[idx], 1e-4, 1))
    best_s, best_ab = ln.sum(), None  # no live stretch at all = only ball handling
    for a_ in range(len(idx)):
        for b_ in range(a_, len(idx)):
            sc = ln[:a_].sum() + lp[a_:b_ + 1].sum() + ln[b_ + 1:].sum()
            if sc > best_s:
                best_s, best_ab = sc, (a_, b_)
    if best_ab:
        dec[idx[best_ab[0]:best_ab[1] + 1]] = 1
caught = ((dec == 0) & dead_true).sum(); wrong = ((dec == 0) & ~dead_true).sum()
print(f"  {live_name} + one live stretch per rally: accuracy {(dec == live_y).mean():.0%} | dead touches caught {caught}/{dead_true.sum()} | live touches wrongly dropped {wrong}/{(~dead_true).sum()}")
g = np.array(end_gap)
print(f"  time from the last detected live touch to 'point over': median {np.median(g):.1f} s; > 2.5 s (real last shot probably NOT detected): {(g > 2.5).sum()}/{len(g)}")

print("\nHOW THE POINT ENDED (last live touch; leave one match out)")
base_acc = np.mean(np.array(base_end) == end_y)
print(f"  old rules: {base_acc:.0%}  confusion {Counter(zip(end_y, base_end))}")
best_e = None
for name, mk in models.items():
    p, _ = lomo(mk, end_X, end_y, end_g)
    ok = p == end_y
    print(f"  {name:13s} {ok.mean():.0%}  confusion (truth, predicted) {Counter(zip(end_y, p))}")
    if best_e is None or ok.mean() > best_e[1]:
        best_e = (name, ok.mean())
if inout:
    ok = sum(t == c for t, c, _ in inout)
    close = [(t, c, round(d, 2)) for t, c, d in inout if t != c]
    print(f"\nIN / OUT of the final bounce (old call from the bounce position): {ok}/{len(inout)} right; wrong ones (truth, call, ft from line): {close}")

final = {"live": models[live_name]().fit(live_X, live_y), "ending": models[best_e[0]]().fit(end_X, end_y),
         "live_model": live_name, "ending_model": best_e[0], "n_touches": int(len(live_y)), "n_endings": int(len(end_y)),
         "cv": {"live_acc": round(float(best[1]), 3), "ending_acc": round(float(best_e[1]), 3), "ending_rules_acc": round(float(base_acc), 3)}}
joblib.dump(final, RE.MODEL)
print(f"\nsaved {RE.MODEL.name}: live = {live_name}, ending = {best_e[0]}")

# where are the mistakes? (position of the touch inside the detected rally, after decoding)
pos = np.array([("first" if x[RE.LIVE_FEATS.index("k_from_start")] == 0 else "last" if x[RE.LIVE_FEATS.index("k_from_end")] == 0 else "middle")
                for x in live_X])
for kind, m in (("dead touch kept", (dec == 1) & dead_true), ("live touch dropped", (dec == 0) & ~dead_true)):
    print(f"  {kind}: {dict(Counter(pos[m]))} | by match {dict(Counter(live_g[m]))}")
