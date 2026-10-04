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
SOURCES = [("full_match", "Mens_Singles_GOLD_2026_US_Open"), ("lv_fahey_yoshitomi", "Fahey_Yoshitomi_LV2026")]
BASE = {"third-shot drive": "drive", "third-shot drop": "drop", "third-shot groundstroke": "groundstroke"}
SKIP = {"serve", "return", None}


def tagged():
    X, y, g, rule = [], [], [], []
    for run, video in SOURCES:
        A = json.loads((ROOT / "outputs" / run / "analysis.json").read_text())
        F = features(A, json.loads((ROOT / "outputs" / run / "render_data.json").read_text())["players"])
        shots = {s["frame"]: s for s in A["shots"]}
        L = json.loads((ROOT / "labels" / f"{video}.json").read_text())
        for r in L["rallies"]:
            for e in r["events"]:
                human = e.get("shot_tagged") or not e.get("auto")
                if e["kind"] != "hit" or not human or e.get("shot") in SKIP:
                    continue
                m = min(shots, key=lambda f: abs(f - e["frame"]))
                if abs(m - e["frame"]) > 4 or shots[m].get("shot_type") in ("serve", "return"):
                    continue
                X.append(F[m]); y.append(BASE.get(e["shot"], e["shot"])); g.append(f"{run}:{r['id']}")
                rt = shots[m].get("shot_type_rules", shots[m].get("shot_type"))  # compare with the rules, not the model
                rule.append(BASE.get(rt, rt))
    return np.array(X, float), np.array(y), np.array(g), np.array(rule)


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
    print(f"{len(y)} tagged rally shots (no serves/returns): {dict(Counter(y).most_common())}")
    print(f"rules:          {np.mean(rule == y):.0%}")
    if len(set(y)) < 2:
        return
    for name in ("rf", "gb"):
        pred = np.empty(len(y), object)
        for grp in np.unique(g):
            te = g == grp
            if len(set(y[~te])) < 2:
                pred[te] = Counter(y[~te]).most_common(1)[0][0]; continue
            m = make(name).fit(np.nan_to_num(X[~te], nan=-1), y[~te])
            pred[te] = m.predict(np.nan_to_num(X[te], nan=-1))
        print(f"learned ({name}):  {np.mean(pred == y):.0%}   per type: " +
              ", ".join(f"{c} {np.mean(pred[y == c] == c):.0%} (n={np.sum(y == c)})" for c in sorted(set(y))))
    if a.save:
        m = make("rf").fit(np.nan_to_num(X, nan=-1), y)
        joblib.dump({"model": m, "features": FEATURES, "classes": sorted(set(y)), "n": len(y)}, MODEL_PATH)
        print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()
