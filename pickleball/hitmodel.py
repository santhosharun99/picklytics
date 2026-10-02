"""Learned hit decider.

Every candidate moment (a break in the ball's flight, or a peak of the arm-swing model) gets
a row of clues - arm swing, 2D ball motion, ball-to-player distance, 3D camera/arc clues and
ball-to-paddle distance - and a Random Forest decides hit / no hit. Chosen with
training/hit_model_cv.py: leave-one-rally-out F1 94% (fully nested) vs 81% for the
flight + swing-confirm rules. Trained by training/train_hit_model.py -> models/hit_clf.joblib.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .flights import flight_boundaries
from .trajectory3d import anchored_track, ground_point, plane_y_point, track_flights

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hit_clf.joblib"
NMS = 8  # frames: at most one hit in any +-8 frame window


@dataclass
class Context:
    """Everything the clues need for one analysed stretch of video."""
    ball: dict                # {frame: (x, y, conf, interpolated)} on main-view frames
    pl: dict                  # {frame: [player dicts with side, kp, box, court, id]}
    P: np.ndarray             # 3x4 camera projection (camera3d)
    fps: float
    paddles: object = None    # anything with .get(frame) -> boxes (PaddleCache) or a dict
    fb: dict = field(default_factory=dict)
    vit: list = field(default_factory=list)
    free: list = field(default_factory=list)

    @classmethod
    def build(cls, ball, pl, P, fps, paddles=None):
        c = cls(ball, pl, P, fps, paddles)
        c.fb = {f: (vi, vo) for f, _, vi, vo in flight_boundaries(ball, penalty=400)}
        c.vit, _ = anchored_track(P, ball, sorted(c.fb), pl, fps, beta=0.5, penalty=2000)
        _, c.free = track_flights(P, ball, fps, beta=0.5, penalty=900, joint=False)
        return c

    def paddle_boxes(self, f):
        if self.paddles is None:
            return []
        return self.paddles.get(f, []) if isinstance(self.paddles, dict) else self.paddles.get(f)


def ball_at(c: Context, f):
    for d in (0, -1, 1, -2, 2, -3, 3):
        if f + d in c.ball:
            return np.array(c.ball[f + d][:2], float)
    return None


def player_at(c: Context, f, side):
    for d in (0, -1, 1, -2, 2, -3, 3):
        for p in c.pl.get(f + d, []):
            if p["side"] == side:
                return p
    return None


def candidates(c: Context, sp, lo, hi):
    """Flight breaks + swing peaks (p >= 0.15) in [lo, hi], merged within 2 frames."""
    cs = {f for f in c.fb if lo <= f <= hi}
    for s in ("near", "far"):
        for f, p in sp[s].items():
            if lo <= f <= hi and p >= 0.15 and p >= max(sp[s].get(g, 0) for g in range(f - 5, f + 6)):
                cs.add(f)
    out = []
    for f in sorted(cs):
        if not out or f - out[-1] > 2:
            out.append(f)
    return out


def features(c: Context, f, sp):
    """Clues for candidate frame f. sp: side -> {frame: swing prob}. Returns (base, d3, paddle)."""
    base = []
    for s in ("near", "far"):
        w = [(sp[s].get(g, 0.0), g - f) for g in range(f - 6, f + 7)]
        m = max(w)
        base += [m[0], m[1], sp[s].get(f, 0.0)]
    meas = [g for g in range(f - 5, f + 6) if g in c.ball and not c.ball[g][3]]
    base.append(len(meas))

    def vel(a, b):
        pts = [np.array(c.ball[g][:2]) for g in range(a, b + 1) if g in c.ball]
        return (pts[-1] - pts[0]) / max(len(pts) - 1, 1) if len(pts) >= 2 else np.zeros(2)
    vb, va = vel(f - 5, f), vel(f, f + 5)
    nb, na = np.linalg.norm(vb), np.linalg.norm(va)
    cos = float(vb @ va / (nb * na)) if nb > 0 and na > 0 else 0.0
    base += [*vb, *va, nb, na, cos, float(vb[1] * va[1] < 0)]
    base.append(min((abs(g - f) for g in c.fb), default=99))
    xy = ball_at(c, f)
    for s in ("near", "far"):
        p = player_at(c, f, s)
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
        p = player_at(c, f, s)
        if p is None or xy is None:
            d3 += [-9, 9, 99]
            continue
        X = plane_y_point(c.P, xy, p["court"][1])[0]
        G = ground_point(c.P, xy)[0]
        d3 += [float(np.clip(X[2], -10, 20)), abs(X[0] - p["court"][0]),
               float(np.hypot(G[0] - p["court"][0], G[1] - p["court"][1]))]
    v = min(c.vit, key=lambda e: abs(e["frame"] - f), default=None)
    nv = v if v and abs(v["frame"] - f) <= 3 else None
    d3 += [float(bool(nv) and nv["kind"] == "hit"), float(bool(nv) and nv["kind"] == "bounce"), nv["pos"][2] if nv else -1.0]
    fc = min(c.free, key=lambda e: abs(e["frame"] - f), default=None)
    if fc and abs(fc["frame"] - f) <= 4:
        d3 += [1.0, float(np.clip(fc["pos"][2], -5, 20)), float(fc["v_in"][1] * fc["v_out"][1] < 0),
               min(fc["speed_in_mph"], 120), min(fc["speed_out_mph"], 120)]
    else:
        d3 += [0.0, -1.0, 0.0, 0.0, 0.0]
    best, cnt, off = 999.0, 0, 0
    for g in range(f - 3, f + 4):
        bx = ball_at(c, g)
        if bx is None:
            continue
        for b in c.paddle_boxes(g):
            d = float(np.hypot((b[0] + b[2]) / 2 - bx[0], (b[1] + b[3]) / 2 - bx[1]))
            cnt += d < 80
            if d < best:
                best, off = d, g - f
    return base, d3, [min(best, 999.0) / 100, cnt, off]


def row(c: Context, f, sp, clues="+3D+paddle"):
    b, d3, pad = features(c, f, sp)
    return {"base": b, "+3D": b + d3, "+paddle": b + pad, "+3D+paddle": b + d3 + pad}[clues]


def nms_pick(frames, probs, thr):
    keep = []
    for i in np.argsort(-np.asarray(probs)):
        if probs[i] >= thr and all(abs(frames[i] - frames[j]) > NMS for j in keep):
            keep.append(i)
    return [frames[i] for i in keep]


def hitter_side(c: Context, f, sp):
    """Side whose player is closest to the ball (wrist distance), else the stronger swing."""
    xy = ball_at(c, f)
    best = None
    for s in ("near", "far"):
        p = player_at(c, f, s)
        if p is None or xy is None:
            continue
        x1, y1, x2, y2 = p["box"]
        kp = np.asarray(p["kp"])
        d = min(np.hypot(*(kp[k, :2] - xy)) for k in (9, 10)) / max(y2 - y1, 1)
        if best is None or d < best[0]:
            best = (d, s)
    if best and best[0] < 1.5:
        return best[1]
    sw = {s: max((sp[s].get(g, 0) for g in range(f - 6, f + 7)), default=0) for s in ("near", "far")}
    return max(sw, key=sw.get)


def load_bundle():
    if not MODEL_PATH.exists():
        return None
    import joblib
    return joblib.load(MODEL_PATH)


def predict_hits(c: Context, sp, bundle, lo, hi):
    """[(frame, prob, side)] hits in [lo, hi]."""
    cs = candidates(c, sp, lo, hi)
    if not cs:
        return []
    if hasattr(c.paddles, "ensure"):
        c.paddles.ensure({g for f in cs for g in range(f - 3, f + 4)})
    X = np.array([row(c, f, sp, bundle["clues"]) for f in cs], float)
    pr = bundle["model"].predict_proba(X)[:, 1]
    picked = nms_pick(cs, pr, bundle["threshold"])
    p_of = dict(zip(cs, pr))
    return [(f, float(p_of[f]), hitter_side(c, f, sp)) for f in sorted(picked)]


# ---------------------------------------------------------------------------------------------
# Bounces: same candidates + clues, plus where the ball meets the ground. Chosen with
# training/bounce_model_cv.py (leave-one-rally-out F1 ~83% vs 57% for the trajectory rules).
# ---------------------------------------------------------------------------------------------
BOUNCE_PATH = MODEL_PATH.parent / "bounce_clf.joblib"


def bounce_candidates(c: Context, sp, lo, hi, extra=()):
    cs = set(candidates(c, sp, lo, hi))
    cs |= {e["frame"] for e in c.free if lo <= e["frame"] <= hi}
    cs |= {f for f in extra if lo <= f <= hi}
    fr = [f for f in range(lo, hi + 1) if f in c.ball]
    for a, f, b in zip(fr, fr[1:], fr[2:]):  # image-y turning points
        if (c.ball[f][1] - c.ball[a][1]) * (c.ball[b][1] - c.ball[f][1]) < 0:
            cs.add(f)
    out = []
    for f in sorted(cs):
        if not out or f - out[-1] > 2:
            out.append(f)
    return out


def bounce_row(c: Context, f, sp):
    b, d3, pad = features(c, f, sp)
    xy = ball_at(c, f)
    if xy is None:
        g = [-99.0, -99.0, 0.0]
    else:
        G = ground_point(c.P, xy)[0]
        g = [float(np.clip(G[0], -40, 60)), float(np.clip(G[1], -40, 90)), float(-1 <= G[0] <= 21 and -1 <= G[1] <= 45)]
    return b + d3 + pad + g


def load_bounce_bundle():
    if not BOUNCE_PATH.exists():
        return None
    import joblib
    return joblib.load(BOUNCE_PATH)


def predict_bounces(c: Context, sp, bundle, lo, hi, extra=()):
    """[(frame, prob, ground (x, y) in court feet)] bounces in [lo, hi]."""
    cs = bounce_candidates(c, sp, lo, hi, extra)
    if not cs:
        return []
    if hasattr(c.paddles, "ensure"):
        c.paddles.ensure({g for f in cs for g in range(f - 3, f + 4)})
    X = np.array([bounce_row(c, f, sp) for f in cs], float)
    pr = bundle["model"].predict_proba(X)[:, 1]
    p_of = dict(zip(cs, pr))
    out = []
    for f in sorted(nms_pick(cs, pr, bundle["threshold"])):
        xy = ball_at(c, f)
        G = ground_point(c.P, xy)[0] if xy is not None else None
        out.append((f, float(p_of[f]), (float(G[0]), float(G[1])) if G is not None else None))
    return out
