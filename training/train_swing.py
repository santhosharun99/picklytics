"""Train the swing detector on labelled hits.

    python training/train_swing.py labels/<video>.json outputs/<run1> [outputs/<run2> ...] --out models/swing_clf.joblib
"""
import argparse, json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from pickleball.players import select_players
from pickleball.swing import frame_features


def build_xy(labels, run):
    K = {"near": {}, "far": {}}
    for line in open(Path(run) / "detections.jsonl"):
        r = json.loads(line)
        if r.get("court"):
            for p in select_players(r.get("people", []), 1):
                K[p["side"]][r["f"]] = (np.asarray(p["kp"]), p["box"][3] - p["box"][1])
    X, y = [], []
    for r in labels["rallies"]:
        if r.get("reviewed") is False or not r.get("end"):
            continue
        hs = [e for e in r["events"] if e["kind"] == "hit"]
        for side in ("near", "far"):
            for f in range(r["start"], r["end"] + 1):
                ft = frame_features(K, side, f)
                if ft is None:
                    continue
                d = min((abs(e["frame"] - f) for e in hs if e["side"] == side), default=99)
                if 3 <= d <= 6:
                    continue  # ambiguous around contact
                X.append(ft)
                y.append(int(d <= 2))
    return X, y


def train(labels, runs):
    X, y = [], []
    for run in runs:
        a, b = build_xy(labels, run)
        X += a
        y += b
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                          random_state=0).fit(np.array(X), np.array(y))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = train(json.loads(Path(a.labels).read_text()), a.runs)
    joblib.dump(m, a.out)
    print("saved", a.out)
