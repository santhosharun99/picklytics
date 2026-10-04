"""Learned shot-type classifier from 3D ball + position features, vs the current rules.

    PYTHONPATH=. python3 training/shot_type_cv.py [--save]

Answer key: hit tags whose shot type a person chose (not auto-imported): label events with shot_tagged, or
hand-made events (no "auto" flag). Serves and returns are excluded (rules get them right); third shots are
scored as their base type (drive / drop / groundstroke). Leave-one-rally-out CV. --save writes
models/shot_clf.joblib trained on all tagged shots.
"""
import argparse, json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from pickleball.shottype import FEATURES, MODEL_PATH, features

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [("full_match", "Mens_Singles_GOLD_2026_US_Open"), ("lv_fahey_yoshitomi", "Fahey_Yoshitomi_LV2026"),
           ("lv_waters_johns", "Waters_Johns_Black_Tardio_LV2026"), ("lv_bright_patriquin", "Bright_Patriquin_Fahey_Staksrud_LV2026")]
BASE = {"third-shot drive": "drive", "third-shot drop": "drop", "third-shot groundstroke": "groundstroke"}
SKIP = {"serve", "return", None}


def tagged():
    """Human shot-type answers from (a) the labeller (shot_tagged or hand-made hits) and (b) the review queue
    (labels/<video>.shots.json: {frame: {"shot", "volley", "unclear"}}). "unclear" answers are left out."""
    X, y, g, rule = [], [], [], []
    for run, video in SOURCES:
        run_dir = ROOT / "outputs" / run
        if not (run_dir / "analysis.json").exists():
            continue
        A = json.loads((run_dir / "analysis.json").read_text())
        F = features(A, json.loads((run_dir / "render_data.json").read_text())["players"])
        shots = {s["frame"]: s for s in A["shots"]}
        answers = {}  # frame -> (shot, group)
        lab = ROOT / "labels" / f"{video}.json"
        if lab.exists():
            for r in json.loads(lab.read_text())["rallies"]:
                for e in r["events"]:
                    if e["kind"] == "hit" and (e.get("shot_tagged") or not e.get("auto")) and e.get("shot") not in SKIP:
                        answers[e["frame"]] = (e["shot"], f"{run}:{r['id']}")
        rq = ROOT / "labels" / f"{video}.shots.json"
        if rq.exists():
            for f, a in json.loads(rq.read_text()).get("answers", {}).items():
                if a.get("shot") and not a.get("unclear") and a["shot"] not in SKIP:
                    answers[int(f)] = (a["shot"], f"{run}:q{a.get('rally', f)}")
        for f, (shot, grp) in answers.items():
            m = min(shots, key=lambda x: abs(x - f))
            if abs(m - f) > 4 or shots[m].get("shot_type") in ("serve", "return"):
                continue
            X.append(F[m]); y.append(BASE.get(shot, shot)); g.append(grp)
            rt = shots[m].get("shot_type_rules", shots[m].get("shot_type"))  # compare with the rules, not the model
            rule.append(BASE.get(rt, rt))
    return np.array(X, float), np.array(y), np.array(g), np.array(rule)


def report(name, y, pred):
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    labels = sorted(set(y))
    p, r, _, n = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    print(f"{name}: accuracy {np.mean(pred == y):.0%}, macro-F1 {f1_score(y, pred, labels=labels, average='macro', zero_division=0):.2f}  | " +
          ", ".join(f"{c} P{pp:.0%}/R{rr:.0%} (n={nn})" for c, pp, rr, nn in zip(labels, p, r, n)))


def make(name):
    return (HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=150, class_weight="balanced")
            if name == "gb" else RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--groups", action="store_true", help="merge drop / dink / reset into 'soft'")
    ap.add_argument("--no-pose", action="store_true")
    a = ap.parse_args()
    X, y, g, rule = tagged()
    if a.groups:  # drop / dink / reset -> one "soft" class (singles has few of each)
        soft = {"drop": "soft", "dink": "soft", "reset": "soft"}
        y = np.array([soft.get(v, v) for v in y]); rule = np.array([soft.get(v, v) for v in rule])
    if a.no_pose:
        X = X[:, :len(FEATURES) - 3]
    match = np.array([x.split(":")[0] for x in g])
    print(f"{len(y)} tagged rally shots (no serves/returns) from {len(set(match))} matches: {dict(Counter(y).most_common())}")
    print(f"always 'drive': {np.mean(y == 'drive'):.0%}")
    report("rules", y, rule)
    if len(set(y)) < 2:
        return
    # the honest test: train on the other matches, test on a match the model never saw
    for name in ("rf", "gb"):
        pred = np.empty(len(y), object)
        for mt in sorted(set(match)):
            te = match == mt
            if len(set(y[~te])) < 2:
                pred[te] = Counter(y[~te]).most_common(1)[0][0] if (~te).any() else "drive"; continue
            mdl = make(name).fit(np.nan_to_num(X[~te], nan=-1), y[~te])
            pred[te] = mdl.predict(np.nan_to_num(X[te], nan=-1))
        report(f"learned {name}, unseen match", y, pred)
    if a.save:
        m = make("rf").fit(np.nan_to_num(X, nan=-1), y)
        joblib.dump({"model": m, "features": FEATURES, "classes": sorted(set(y)), "n": len(y)}, MODEL_PATH)
        print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()
