"""Which ML model decides hits best? Leave-one-rally-out comparison on the labelled rallies.

    PYTHONPATH=. python3 training/hit_model_cv.py

Every candidate moment (ball-path break or arm-swing peak) gets a row of clues; a classifier
decides hit / not hit. For each of the 15 labelled rallies: train on the other 14, test on it.
The arm-swing model that feeds the swing clues is also retrained without the test rally, and
the current pipeline is scored the same way, so all numbers are directly comparable.

Clue sets:  base = swing + 2D ball + ball-to-player;  +3D = 3D camera/arc clues;  +paddle.
"""
import contextlib, io, json, time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import pickleball.events as ev
from pickleball.analysis import load_detections, run_analysis
from pickleball.ball import track_ball
from pickleball.camera3d import calibrate_from_run
from pickleball.players import PlayerIdentifier, select_players
from pickleball.swing import frame_features, pose_index
import pickleball.hitmodel as hm
from training.evaluate import match

ROOT = Path(__file__).resolve().parents[1]
LABELS = json.loads((ROOT / "labels" / "Mens_Singles_GOLD_2026_US_Open.json").read_text())
RUNS = sorted((ROOT / "outputs").glob("cmp_ball_detector_v3_kaggle_*"))
NAMES = {"P1": "Devilliers", "P2": "Boyer"}
TOL, NMS = 4, 8


def features(S, f, sp):
    return hm.features(S.ctx, f, sp)


def candidates(S, rally, sp):
    return hm.candidates(S.ctx, sp, rally["start"] - 5, rally["end"] + 5)


nms_pick = hm.nms_pick


class Stretch:
    def __init__(self, run: Path, labels: dict | None = None, lazy_paddles: bool = False):
        self.run = run
        self.meta, recs = load_detections(run)
        self.fps = self.meta["fps"]
        cam = calibrate_from_run(run)
        self.P = cam.P
        ident, self.pl = PlayerIdentifier(2), {}
        for r in recs:
            if r.get("court"):
                self.pl[r["f"]] = ident.assign(select_players(r.get("people", []), 1))
        self.K = pose_index(self.pl)
        ball = track_ball(recs)
        self.ball = {f: v for f, v in ball.items() if f in self.pl}
        lo, hi = self.meta["first_frame"], self.meta["last_frame"]
        labels = labels or LABELS
        fps_l = labels.get("fps", 29.97)
        self.rallies = [r for r in labels["rallies"] if r["end"] and lo <= r["start"] <= r["end"] < hi
                        and r.get("reviewed", True) and (r["end"] - r["start"]) / fps_l <= 60]  # >60 s = mis-set end
        if lazy_paddles:
            from pickleball.paddles import PaddleCache
            paddles = PaddleCache(run, self.meta["video"])
        else:
            pf = run / "paddles.json"
            paddles = {int(k): v for k, v in json.loads(pf.read_text()).items()} if pf.exists() else {}
        self.ctx = hm.Context.build(self.ball, self.pl, self.P, self.fps, paddles)
        # per-frame swing features (model applied per fold)
        self.sf = {s: {} for s in ("near", "far")}
        for r in self.rallies:
            for s in ("near", "far"):
                for f in range(r["start"] - 15, r["end"] + 16):
                    x = frame_features(self.K, s, f)
                    if x is not None:
                        self.sf[s][f] = x

    def swing_xy(self, rally):
        """Training rows for the swing model from one rally (same rule as train_swing.build_xy)."""
        hs = [e for e in rally["events"] if e["kind"] == "hit"]
        X, y = [], []
        for s in ("near", "far"):
            for f in range(rally["start"], rally["end"] + 1):
                if f not in self.sf[s]:
                    continue
                d = min((abs(e["frame"] - f) for e in hs if e["side"] == s), default=99)
                if 3 <= d <= 6:
                    continue
                X.append(self.sf[s][f])
                y.append(int(d <= 2))
        return X, y


def train_swing(rows):
    X = np.array([x for r in rows for x in r[0]])
    y = np.array([v for r in rows for v in r[1]])
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0).fit(X, y)














MODELS = {
    "Current model type (gradient boosting)": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=0),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=-1),
    "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=4),
    "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)),
    "Neural net (MLP)": lambda: make_pipeline(StandardScaler(), MLPClassifier((32, 16), alpha=1e-2, max_iter=1500, random_state=0)),
}


def fit(name, X, y):
    m = MODELS[name]()
    if name == "XGBoost":
        m.set_params(scale_pos_weight=max((y == 0).sum() / max((y == 1).sum(), 1), 1.0))
    return m.fit(X, y)





def pick_threshold(name, X, y, groups, frames_by_row):
    """Threshold chosen on the TRAINING rallies only (inner grouped CV)."""
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(5, len(set(groups)))).split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        oof[te] = fit(name, X[tr], y[tr]).predict_proba(X[te])[:, 1]
    best = (0.5, -1)
    for thr in np.arange(0.2, 0.85, 0.05):
        tp = fp = fn = 0
        for g in set(groups):
            idx = np.where(groups == g)[0]
            pred = nms_pick(frames_by_row[idx], oof[idx], thr)
            truth = [frames_by_row[i] for i in idx if y[i] == 1]
            m = len(match(truth, pred, TOL))
            tp, fp, fn = tp + m, fp + len(pred) - m, fn + len(truth) - m
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[1]:
            best = (thr, f1)
    return best[0]


def main():
    t0 = time.time()
    S = [Stretch(r) for r in RUNS]
    items = [(s, r) for s in S for r in s.rallies]  # (stretch, rally)
    print(f"loaded {len(items)} rallies, {sum(sum(e['kind'] == 'hit' for e in r['events']) for _, r in items)} hits ({time.time() - t0:.0f}s)", flush=True)
    swing_rows = [s.swing_xy(r) for s, r in items]

    # out-of-rally swing probabilities: rally k scored by a swing model trained without rally k
    swing_models, sp = [], []
    for k, (s, r) in enumerate(items):
        m = train_swing([swing_rows[j] for j in range(len(items)) if j != k])
        swing_models.append(m)
        d = {}
        for side in ("near", "far"):
            fr = [f for f in s.sf[side] if r["start"] - 15 <= f <= r["end"] + 15]
            d[side] = dict(zip(fr, m.predict_proba(np.array([s.sf[side][f] for f in fr]))[:, 1])) if fr else {}
        sp.append(d)
    print(f"swing models done ({time.time() - t0:.0f}s)", flush=True)

    # rows
    rows = []  # (rally idx, frame, base, d3, pad, label)
    n_truth, covered = 0, 0
    for k, (s, r) in enumerate(items):
        cs = candidates(s, r, sp[k])
        truth = [e["frame"] for e in r["events"] if e["kind"] == "hit"]
        pos = {cs[j] for _, j in match(truth, cs, TOL)}
        n_truth, covered = n_truth + len(truth), covered + len(pos)
        for f in cs:
            b, d3, pad = features(s, f, sp[k])
            rows.append((k, f, b, d3, pad, int(f in pos)))
    print(f"{len(rows)} candidates; they cover {covered}/{n_truth} labelled hits (the ceiling) ({time.time() - t0:.0f}s)", flush=True)
    g = np.array([r[0] for r in rows])
    fr = np.array([r[1] for r in rows])
    y = np.array([r[5] for r in rows])
    sets = {"base": np.array([r[2] for r in rows], float),
            "+3D": np.array([r[2] + r[3] for r in rows], float),
            "+3D+paddle": np.array([r[2] + r[3] + r[4] for r in rows], float),
            "+paddle": np.array([r[2] + r[4] for r in rows], float)}
    truth_by = {k: [e["frame"] for e in r["events"] if e["kind"] == "hit"] for k, (_, r) in enumerate(items)}

    results = {}
    # current pipeline, same leave-one-rally-out swing models
    tp = fp = fn = 0
    for k, (s, r) in enumerate(items):
        ev.SWING_MODEL["model"] = swing_models[k]
        with contextlib.redirect_stdout(io.StringIO()):
            A = run_analysis(s.run, 2, NAMES)
        pred = [x["frame"] for x in A["shots"] if r["start"] <= x["frame"] <= r["end"]]
        m = len(match(truth_by[k], pred, TOL))
        tp, fp, fn = tp + m, fp + len(pred) - m, fn + len(truth_by[k]) - m
    results[("Current system (as running now)", "-")] = (tp, fp, fn)
    print(f"  {'Current system (as running now)':40s} {'-':11s} found {tp}/{tp + fn}  false {fp}  F1 {2 * tp / max(2 * tp + fp + fn, 1):.0%}  ({time.time() - t0:.0f}s)", flush=True)

    for set_name, X in sets.items():
        for name in MODELS:
            tp = fp = fn = 0
            for k in range(len(items)):
                tr, te = g != k, g == k
                thr = pick_threshold(name, X[tr], y[tr], g[tr], fr[tr])
                m = fit(name, X[tr], y[tr])
                pred = nms_pick(fr[te], m.predict_proba(X[te])[:, 1], thr)
                mm = len(match(truth_by[k], pred, TOL))
                tp, fp, fn = tp + mm, fp + len(pred) - mm, fn + len(truth_by[k]) - mm
            results[(name, set_name)] = (tp, fp, fn)
            print(f"  {name:40s} {set_name:11s} found {tp}/{tp + fn}  false {fp}  F1 {2 * tp / max(2 * tp + fp + fn, 1):.0%}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n{'model':40s} {'clues':11s} {'hits found':>11s} {'real':>6s} {'F1':>5s}")
    out = []
    for (name, set_name), (tp, fp, fn) in sorted(results.items(), key=lambda kv: -2 * kv[1][0] / max(2 * kv[1][0] + kv[1][1] + kv[1][2], 1)):
        rec, prec = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        print(f"{name:40s} {set_name:11s} {rec:>5.0%} ({tp}/{tp + fn}) {prec:>5.0%} {f1:>5.0%}")
        out.append({"model": name, "clues": set_name, "found": tp, "truth": tp + fn, "false": fp, "recall": rec, "precision": prec, "f1": f1})
    (ROOT / "outputs" / "hit_model_cv.json").write_text(json.dumps(out, indent=1))



def strict(model_name="Random Forest", set_name="+3D+paddle"):
    """Fully nested check: for test rally k, EVERY swing model (also those scoring the training
    rallies) is trained without rally k, so no label of k can reach the hit model in any way."""
    t0 = time.time()
    S = [Stretch(r) for r in RUNS]
    items = [(s, r) for s in S for r in s.rallies]
    rows_sw = [s.swing_xy(r) for s, r in items]
    n = len(items)
    truth_by = {k: [e["frame"] for e in r["events"] if e["kind"] == "hit"] for k, (_, r) in enumerate(items)}

    def probs(m, k):
        s, r = items[k]
        d = {}
        for side in ("near", "far"):
            fr = [f for f in s.sf[side] if r["start"] - 15 <= f <= r["end"] + 15]
            d[side] = dict(zip(fr, m.predict_proba(np.array([s.sf[side][f] for f in fr]))[:, 1])) if fr else {}
        return d

    def rows_for(k, sp_k):
        s, r = items[k]
        cs = candidates(s, r, sp_k)
        pos = {cs[j] for _, j in match(truth_by[k], cs, TOL)}
        out = []
        for f in cs:
            b, d3, pad = features(s, f, sp_k)
            x = {"base": b, "+3D": b + d3, "+3D+paddle": b + d3 + pad, "+paddle": b + pad}[set_name]
            out.append((k, f, x, int(f in pos)))
        return out

    tp = fp = fn = 0
    for k in range(n):
        rows = []
        for j in range(n):
            excl = {k} if j == k else {k, j}
            m = train_swing([rows_sw[i] for i in range(n) if i not in excl])
            rows += rows_for(j, probs(m, j))
        g = np.array([r[0] for r in rows]); fr = np.array([r[1] for r in rows])
        X = np.array([r[2] for r in rows], float); y = np.array([r[3] for r in rows])
        tr, te = g != k, g == k
        thr = pick_threshold(model_name, X[tr], y[tr], g[tr], fr[tr])
        pred = nms_pick(fr[te], fit(model_name, X[tr], y[tr]).predict_proba(X[te])[:, 1], thr)
        mm = len(match(truth_by[k], pred, TOL))
        tp, fp, fn = tp + mm, fp + len(pred) - mm, fn + len(truth_by[k]) - mm
        print(f"  rally {k + 1}/{n}: found {mm}/{len(truth_by[k])}, false {len(pred) - mm}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"STRICT {model_name} {set_name}: found {tp}/{tp + fn}  false {fp}  F1 {2 * tp / max(2 * tp + fp + fn, 1):.0%}")


if __name__ == "__main__":
    import sys
    strict() if "--strict" in sys.argv else main()
