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
from pickleball.flights import flight_boundaries
from pickleball.players import PlayerIdentifier, select_players
from pickleball.swing import frame_features, pose_index
from pickleball.trajectory3d import anchored_track, ground_point, plane_y_point, track_flights
from training.evaluate import match

ROOT = Path(__file__).resolve().parents[1]
LABELS = json.loads((ROOT / "labels" / "Mens_Singles_GOLD_2026_US_Open.json").read_text())
RUNS = sorted((ROOT / "outputs").glob("cmp_ball_detector_v3_kaggle_*"))
NAMES = {"P1": "Devilliers", "P2": "Boyer"}
TOL, NMS = 4, 8


class Stretch:
    def __init__(self, run: Path):
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
        self.rallies = [r for r in LABELS["rallies"] if r["end"] and lo <= r["start"] and r["end"] < hi and r.get("reviewed", True)]
        self.fb = {f: (vi, vo) for f, _, vi, vo in flight_boundaries(self.ball, penalty=400)}
        self.vit, _ = anchored_track(self.P, self.ball, sorted(self.fb), self.pl, self.fps, beta=0.5, penalty=2000)
        _, self.free = track_flights(self.P, self.ball, self.fps, beta=0.5, penalty=900, joint=False)
        pf = run / "paddles.json"
        self.paddles = {int(k): v for k, v in json.loads(pf.read_text()).items()} if pf.exists() else {}
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


def ball_at(S, f):
    for d in (0, -1, 1, -2, 2, -3, 3):
        if f + d in S.ball:
            return np.array(S.ball[f + d][:2], float)
    return None


def player_at(S, f, side):
    for d in (0, -1, 1, -2, 2, -3, 3):
        for p in S.pl.get(f + d, []):
            if p["side"] == side:
                return p
    return None


def features(S, f, sp):
    """sp: side -> {frame: swing prob} for this rally. Returns (base, 3d, paddle) feature lists."""
    base = []
    for s in ("near", "far"):
        w = [(sp[s].get(g, 0.0), g - f) for g in range(f - 6, f + 7)]
        m = max(w)
        base += [m[0], m[1], sp[s].get(f, 0.0)]
    meas = [g for g in range(f - 5, f + 6) if g in S.ball and not S.ball[g][3]]
    base.append(len(meas))
    def vel(a, b):
        pts = [np.array(S.ball[g][:2]) for g in range(a, b + 1) if g in S.ball]
        return (pts[-1] - pts[0]) / max(len(pts) - 1, 1) if len(pts) >= 2 else np.zeros(2)
    vb, va = vel(f - 5, f), vel(f, f + 5)
    nb, na = np.linalg.norm(vb), np.linalg.norm(va)
    cos = float(vb @ va / (nb * na)) if nb > 0 and na > 0 else 0.0
    base += [*vb, *va, nb, na, cos, float(vb[1] * va[1] < 0)]
    base.append(min((abs(g - f) for g in S.fb), default=99))
    xy = ball_at(S, f)
    for s in ("near", "far"):
        p = player_at(S, f, s)
        if p is None or xy is None:
            base += [9.0, 9.0]
            continue
        x1, y1, x2, y2 = p["box"]
        h = max(y2 - y1, 1)
        dx, dy = max(x1 - xy[0], 0, xy[0] - x2), max(y1 - xy[1], 0, xy[1] - y2)
        kp = np.asarray(p["kp"])
        wr = min(np.hypot(*(kp[k, :2] - xy)) for k in (9, 10))
        base += [np.hypot(dx, dy) / h, wr / h]
    d3 = []
    for s in ("near", "far"):
        p = player_at(S, f, s)
        if p is None or xy is None:
            d3 += [-9, 9, 99]
            continue
        X = plane_y_point(S.P, xy, p["court"][1])[0]
        G = ground_point(S.P, xy)[0]
        d3 += [float(np.clip(X[2], -10, 20)), abs(X[0] - p["court"][0]), float(np.hypot(G[0] - p["court"][0], G[1] - p["court"][1]))]
    v = min(S.vit, key=lambda e: abs(e["frame"] - f), default=None)
    near_v = v if v and abs(v["frame"] - f) <= 3 else None
    d3 += [float(bool(near_v) and near_v["kind"] == "hit"), float(bool(near_v) and near_v["kind"] == "bounce"),
           near_v["pos"][2] if near_v else -1.0]
    c = min(S.free, key=lambda e: abs(e["frame"] - f), default=None)
    if c and abs(c["frame"] - f) <= 4:
        d3 += [1.0, float(np.clip(c["pos"][2], -5, 20)), float(c["v_in"][1] * c["v_out"][1] < 0),
               min(c["speed_in_mph"], 120), min(c["speed_out_mph"], 120)]
    else:
        d3 += [0.0, -1.0, 0.0, 0.0, 0.0]
    pad = []
    best, cnt, off = 999.0, 0, 0
    for g in range(f - 3, f + 4):
        bx = ball_at(S, g)
        if bx is None:
            continue
        for b in S.paddles.get(g, []):
            d = float(np.hypot((b[0] + b[2]) / 2 - bx[0], (b[1] + b[3]) / 2 - bx[1]))
            cnt += d < 80
            if d < best:
                best, off = d, g - f
    pad += [min(best, 999.0) / 100, cnt, off]
    return base, d3, pad


def candidates(S, rally, sp):
    lo, hi = rally["start"] - 5, rally["end"] + 5
    c = {f for f in S.fb if lo <= f <= hi}
    for s in ("near", "far"):
        for f, p in sp[s].items():
            if lo <= f <= hi and p >= 0.15 and p >= max(sp[s].get(g, 0) for g in range(f - 5, f + 6)):
                c.add(f)
    out = []
    for f in sorted(c):
        if not out or f - out[-1] > 2:
            out.append(f)
    return out


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


def nms_pick(frames, probs, thr):
    keep = []
    for i in np.argsort(-probs):
        if probs[i] >= thr and all(abs(frames[i] - frames[j]) > NMS for j in keep):
            keep.append(i)
    return [frames[i] for i in keep]


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
