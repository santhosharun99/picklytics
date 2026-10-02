"""3D ball path for every rally, pinned to the events we already detect.

One camera cannot tell how far away a flying ball is. Contacts can:
  - bounce : the ball is ON THE GROUND -> its court spot (x, y, 0) is known;
  - hit    : the ball is AT THE HITTER  -> follow the camera ray through the ball pixel to the hitter's
             court depth (feet y): that gives the ball's side position and height at contact.
Between two pinned points the ball flies under gravity + air drag, so the arc is fixed by its two ends
and the flight time; the drag strength is picked per arc to best match the ball seen in the video.
After the last contact the arc is fitted from that contact and the remaining sightings (out, net, winner).

Quality checks per arc: pixel error against the ball seen in the video, and net clearance.
"""
import numpy as np

from .court import NET_Y
from .trajectory3d import _arc_between, _arc_free_end, plane_y_point, positions, reproject, velocities

BETAS = (0.0, 0.4, 0.8, 1.2)   # air drag strengths tried per arc (1/s)
MAX_FLIGHT_S = 2.6
NET_TOP_FT = 2.9               # 34 in at the centre, 36 in at the posts
BAD_PX = 40.0                  # an arc further than this from the ball seen in the video is not trusted


def _ball_px(ball, f, win=3):
    for d in [0] + [s * k for k in range(1, win + 1) for s in (-1, 1)]:
        v = ball.get(f + d)
        if v is not None and not v[3]:
            return np.array(v[:2], float)
    return None


def _obs(ball, f0, f1):
    fr = [f for f in range(f0 + 1, f1) if f in ball and not ball[f][3]]
    return np.array(fr, float), (np.array([ball[f][:2] for f in fr], float) if fr else np.zeros((0, 2)))


def rally_anchors(shots, bounces, ball, P, start_f, end_f):
    """Pinned 3D points (frame, xyh, kind) for one rally, time order."""
    out = []
    for s in shots:
        if s.get("inferred") or not s.get("hitter_court"):
            continue
        uv = _ball_px(ball, s["frame"])
        hx, hy = s["hitter_court"]
        if uv is not None:
            X = plane_y_point(P, uv, hy)[0]
            if not (0.3 <= X[2] <= 9.0 and abs(X[0] - hx) <= 6.0):
                X = np.array([hx, hy, 3.0])  # ray misses the hitter: assume waist height at the hitter
        else:
            X = np.array([hx, hy, 3.0])
        out.append((s["frame"], X, "hit"))
    for b in bounces:
        if start_f - 5 <= b["frame"] <= end_f + 45:
            out.append((b["frame"], np.array([b["court"][0], b["court"][1], 0.0]), "bounce"))
    out.sort(key=lambda a: a[0])
    dedup = []
    for a in out:
        if dedup and a[0] - dedup[-1][0] < 3:
            if a[2] == "hit":  # keep the hit when a bounce and a hit coincide
                dedup[-1] = a
            continue
        dedup.append(a)
    return dedup


def _px_err(P, theta, beta, t, uv):
    if len(t) == 0:
        return None
    d = np.linalg.norm(reproject(P, positions(theta, t, beta)) - uv, axis=1)
    return float(np.median(d))


def rally_arcs(anchors, ball, P, fps, tail_s=1.2):
    """Fitted arcs between consecutive anchors (+ the final flight). Each arc:
    {f0, f1, kind0, kind1, beta, theta (local time from f0), err_px, speed_mph, net_clear_ft}."""
    arcs = []
    for (f0, X0, k0), (f1, X1, k1) in zip(anchors, anchors[1:]):
        T = (f1 - f0) / fps
        if T <= 0 or T > MAX_FLIGHT_S:
            continue
        fr, uv = _obs(ball, f0, f1)
        t = (fr - f0) / fps
        best = None
        for beta in BETAS:
            th = _arc_between(X0, X1, 0.0, T, beta)
            e = _px_err(P, th, beta, t, uv)
            key = e if e is not None else 0.0
            if best is None or key < best[0]:
                best = (key, beta, th, e)
        _, beta, th, e = best
        if e is not None and e > BAD_PX and len(t) >= 4:
            # the two ends don't explain the flight (usually a missed contact in between): pin only the
            # start and let the sightings decide the velocity; keep it if it matches the video better
            for b2 in BETAS:
                th2 = _arc_free_end(P, uv, t, X0, 0.0, b2)
                e2 = _px_err(P, th2, b2, t, uv)
                if e2 < e:
                    beta, th, e = b2, th2, e2
        arc = _describe(f0, f1, k0, k1, beta, th, e, fps)
        arc["confident"] = (e is None or e <= BAD_PX) and arc["sane"]
        arcs.append(arc)
    if anchors:  # final flight after the last contact, fitted from the sightings that follow
        f0, X0, k0 = anchors[-1]
        fr, uv = _obs(ball, f0, f0 + int(tail_s * fps))
        if len(fr) >= 4:
            # stop at the first long gap (ball gone)
            gaps = np.diff(np.r_[f0, fr])
            cut = np.where(gaps > 6)[0]
            if len(cut):
                fr, uv = fr[:cut[0]], uv[:cut[0]]
        if len(fr) >= 4:
            t = (fr - f0) / fps
            best = None
            for beta in BETAS:
                th = _arc_free_end(P, uv, t, X0, 0.0, beta)
                e = _px_err(P, th, beta, t, uv)
                if best is None or e < best[0]:
                    best = (e, beta, th)
            e, beta, th = best
            arc = _describe(f0, int(fr[-1]), k0, "end", beta, th, e, fps)
            arc["confident"] = e <= BAD_PX and arc["sane"]
            arcs.append(arc)
    return arcs


def _describe(f0, f1, k0, k1, beta, th, err, fps):
    T = (f1 - f0) / fps
    ts = np.linspace(0, T, max(int(T * fps) + 1, 2))
    X = positions(th, ts, beta)
    v0 = velocities(th, [0.0], beta)[0]
    clear = None
    side = np.sign(X[:, 1] - NET_Y)
    cross = np.where(np.diff(side) != 0)[0]
    if len(cross):
        i = cross[0]
        a = (NET_Y - X[i, 1]) / (X[i + 1, 1] - X[i, 1] + 1e-9)
        clear = float(X[i, 2] + a * (X[i + 1, 2] - X[i, 2]) - NET_TOP_FT)
    # physics sanity: never under the court; a ball that is played on must have cleared the net
    sane = X[:, 2].min() >= -0.5 and X[:, 2].max() <= 25 and (clear is None or clear >= -0.5 or k1 == "end")
    return {"f0": int(f0), "f1": int(f1), "kind0": k0, "kind1": k1, "beta": beta, "theta": th.tolist(),
            "err_px": err, "speed_mph": float(np.linalg.norm(v0) * 0.6818), "net_clear_ft": clear,
            "apex_ft": float(X[:, 2].max()), "sane": bool(sane)}


def sample(arcs, fps):
    """Per-frame ball positions {frame: (x, y, h)} from the arcs."""
    out = {}
    for a in arcs:
        th = np.asarray(a["theta"])
        fr = np.arange(a["f0"], a["f1"] + 1)
        X = positions(th, (fr - a["f0"]) / fps, a["beta"])
        for f, x in zip(fr, X):
            out[int(f)] = [round(float(v), 2) for v in x]
    return out


def match_arcs(A: dict, ball: dict, P) -> dict:
    """{rally index: arcs} for an analysis.json dict."""
    fps = A["meta"]["fps"]
    out = {}
    for r in A["rallies"]:
        shots = sorted([s for s in A["shots"] if s["rally"] == r["index"]], key=lambda s: s["frame"])
        f0, f1 = int(r["start_t"] * fps), int(r["end_t"] * fps)
        anchors = rally_anchors(shots, A["bounces"], ball, P, f0, f1)
        out[r["index"]] = rally_arcs(anchors, ball, P, fps)
    return out
