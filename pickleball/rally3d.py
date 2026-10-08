"""Physics-based 3D reconstruction of a rally from one broadcast camera (docs/BALL_RESEARCH_PLAN.md, section 3).

The ball is a hybrid system: ballistic flight (gravity + linear air drag) between JUNCTIONS - paddle contacts and
bounces. One camera cannot see depth, but the game's physics pins it: a bounce happens at height 0, a contact
within reach of a player, and between two junctions the flight is fully determined by their 3D positions and the
time between them (closed-form boundary-value solution, `_velocity`). So a rally is parameterised only by its
junction points; they are found by least squares on the reprojection error of every observed ball pixel, with
bounces constrained to h = 0 and contacts to player reach. The type of each junction (contact / bounce) is chosen by
the fit (lower robust cost + a small complexity prior), not by hand rules. The covariance of the fit gives the
uncertainty of every bounce point -> line calls 'in' / 'out' / 'too close to call'.

Candidate junction frames come from the 2D track (flights.segment_flights: piecewise-quadratic breaks).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .court import COURT_L, COURT_W, NET_Y  # noqa: F401 (NET_Y: end_of_play)
from .flights import segment_flights

G, DRAG = 32.2, 0.6
# Spin: a pickleball's topspin / backspin bends a flight down / up (Magnus) - modelled as a per-flight effective gravity
# g_eff with a weak prior around G. US Open tagged flights (2026-10-08): fixed g leaves serves at ~13 px median error,
# per-flight g_eff fits them to ~1 px; g_eff medians serves 47, drives 36, long flights 31 ft/s^2 (pure gravity 32.2).
# DRAG 0.6/s ~ terminal speed 54 ft/s (quadratic-drag fit of a serve gave 46) - 0.12 was far too little.
G_LO, G_HI, G_SD = 0.5 * G, 2.2 * G, 12.0
MIN_GAP = 4  # frames between two junctions (a bounce and the next contact are never closer at 30-60 fps)


def _e(t):
    return (1 - np.exp(-DRAG * t)) / DRAG


def _velocity(p0, p1, T, g=G):
    """Launch velocity (vx, vy, vh) that carries the ball from p0 to p1 in T seconds (effective gravity g)."""
    e = _e(T)
    vx = (p1[0] - p0[0]) / e
    vy = (p1[1] - p0[1]) / e
    vh = (p1[2] - p0[2] + g / DRAG * T) / e - g / DRAG
    return np.array([vx, vy, vh])


def _flight(p0, v, t, g=G):
    e = _e(t)
    return np.c_[p0[0] + v[0] * e, p0[1] + v[1] * e, p0[2] + (v[2] + g / DRAG) * e - g / DRAG * t]


@dataclass
class Junction:
    frame: float
    kind: str                     # 'start' | 'contact' | 'bounce' | 'net' | 'end'
    xyh: np.ndarray = field(default=None)
    sd: np.ndarray = field(default=None)   # 1-sigma of (x, y, h) from the fit


@dataclass
class RallyFit:
    junctions: list
    cost: float
    rms_px: float
    n_obs: int
    inliers: np.ndarray
    g: list = None   # effective gravity per flight (ft/s^2): > G topspin, < G backspin


def _side_y(side):
    return 33.0 if side == "near" else 11.0


def fit(cam, frames, px, fps, kinds, jframes, players=None, init=None, inlier_px=6.0, init_g=None):
    """Least-squares junction positions for given junction frames and kinds.
    frames, px: observed ball (N,), (N, 2); jframes: sorted junction frames (first = start, last = end);
    kinds: list of kinds per junction; players: {frame: [(x, y), ...]} court positions (contact reach prior)."""
    frames = np.asarray(frames, float)
    jf = np.asarray(jframes, float)
    nJ = len(jf)
    free = []  # parameter layout: per junction x, y, (h unless bounce)
    for k in kinds:
        free.append(2 if k in ("bounce", "net") else 3)  # bounce: (x, y) at h 0; net: (x, h) at y NET_Y
    idx = np.cumsum([0] + free)
    nq = idx[-1]  # then one effective gravity per flight

    def unpack(q):
        P = []
        for j in range(nJ):
            a = q[idx[j]:idx[j + 1]]
            if kinds[j] == "net":
                P.append(np.array([a[0], NET_Y, a[1]]))
            else:
                P.append(np.array([a[0], a[1], 0.0 if kinds[j] == "bounce" else a[2]]))
        return P, q[nq:]

    seg = np.clip(np.searchsorted(jf, frames, side="right") - 1, 0, nJ - 2)
    Pm = cam.P

    def project(X):
        Xh = np.c_[X[:, 0], X[:, 1], -X[:, 2], np.ones(len(X))] @ Pm.T  # same as cv2.projectPoints, no distortion
        return Xh[:, :2] / Xh[:, 2:3]

    def predict(P, gs):
        out = np.zeros((len(frames), 3))
        for s in range(nJ - 1):
            m = seg == s
            if not m.any():
                continue
            T = (jf[s + 1] - jf[s]) / fps
            v = _velocity(P[s], P[s + 1], T, gs[s])
            out[m] = _flight(P[s], v, (frames[m] - jf[s]) / fps, gs[s])
        return out

    def residuals(q):
        P, gs = unpack(q)
        r = (project(predict(P, gs)) - px).ravel()
        pri = list((np.asarray(gs) - G) / G_SD)
        for j, k in enumerate(kinds):
            if k == "contact" and players is not None:
                cand = players.get(int(round(jf[j])), [])
                if cand:
                    d = min(np.hypot(P[j][0] - c[0], P[j][1] - c[1]) for c in cand)
                    pri.append(max(0.0, d - 3.0) * 4.0)   # within ~3 ft of a player: free; beyond: penalised
            if k in ("contact", "start"):
                pri.append(max(0.0, 0.3 - P[j][2]) * 20 + max(0.0, P[j][2] - 9.0) * 20)
        return np.r_[r, pri]

    if init is None:
        init = init_from_pieces(cam, frames, px, fps, jf, kinds)
    elif any(g is None for g in init):  # e.g. a newly inserted junction: fill from the piece fits
        pi = init_from_pieces(cam, frames, px, fps, jf, kinds)
        init = [g if g is not None else pi[j] for j, g in enumerate(init)]
    q0 = []
    for j in range(nJ):
        g = init[j] if init is not None else None
        if g is None:
            # back-project the observed pixel nearest in time onto a plausible height
            i = int(np.argmin(np.abs(frames - jf[j])))
            g = _backproject(cam, px[i], 0.0 if kinds[j] == "bounce" else 3.0)
        if kinds[j] == "net":
            q0 += [g[0], g[2] if len(g) > 2 else 2.0]
        else:
            q0 += [g[0], g[1]] + ([] if kinds[j] == "bounce" else [g[2] if len(g) > 2 else 3.0])
    # physical limits (game rules): bounces on or near the court; contacts 0.3-8 ft high within the playing area;
    # the clip's first / last point is free in height (ball in flight) but still above the floor
    q0 += list(np.clip(init_g if init_g is not None else [G] * (nJ - 1), G_LO + 1e-3, G_HI - 1e-3))
    lo, hi = [], []
    for k in kinds:
        if k == "net":  # the ball hits the net (tape at ~3 ft) and drops
            lo += [-1, 0.0]; hi += [21, 3.0]
        elif k == "bounce":
            lo += [-6, -8]; hi += [26, 52]
        elif k == "contact":
            lo += [-8, -12, 0.3]; hi += [28, 56, 8.0]
        else:
            lo += [-12, -20, 0.0]; hi += [32, 64, 15.0]
    lo += [G_LO] * (nJ - 1); hi += [G_HI] * (nJ - 1)
    q0 = np.clip(q0, np.array(lo) + 1e-3, np.array(hi) - 1e-3)
    # sparsity: a pixel depends only on the two junctions around its flight; a prior only on its junction
    from scipy.sparse import lil_matrix
    n_pri = len(residuals(q0)) - 2 * len(frames)
    S = lil_matrix((2 * len(frames) + n_pri, nq + nJ - 1), dtype=int)
    for i, sg in enumerate(seg):
        for j in (sg, sg + 1):
            S[2 * i:2 * i + 2, idx[j]:idx[j + 1]] = 1
        S[2 * i:2 * i + 2, nq + sg] = 1
    row = 2 * len(frames)
    for s_ in range(nJ - 1):
        S[row, nq + s_] = 1; row += 1
    for j, k in enumerate(kinds):  # same order as in residuals()
        if k == "contact" and players is not None and players.get(int(round(jf[j]))):
            S[row, idx[j]:idx[j + 1]] = 1; row += 1
        if k in ("contact", "start"):
            S[row, idx[j]:idx[j + 1]] = 1; row += 1
    r = least_squares(residuals, q0, bounds=(lo, hi), loss="soft_l1", f_scale=inlier_px / 2, jac_sparsity=S,
                      x_scale="jac", max_nfev=200)
    P, gs = unpack(r.x)
    res = np.linalg.norm((project(predict(P, gs)) - px), axis=1)
    inl = res < inlier_px
    # 1-sigma of every junction coordinate: Gauss-Newton covariance scaled by the inlier residual variance
    J = r.jac[: 2 * len(frames)]
    J = J.toarray() if hasattr(J, "toarray") else J
    s2 = float(np.mean(res[inl] ** 2)) / 2 if inl.any() else 25.0
    try:
        cov = np.linalg.pinv(J.T @ J) * s2
        sd_all = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        sd_all = np.full(len(r.x), np.nan)
    js = []
    for j in range(nJ):
        a = sd_all[idx[j]:idx[j + 1]]
        sd_j = np.array([a[0], 0.0, a[1]]) if kinds[j] == "net" else np.array([a[0], a[1], 0.0 if kinds[j] == "bounce" else a[2]])
        js.append(Junction(float(jf[j]), kinds[j], P[j], sd_j))
    return RallyFit(js, float(r.cost), float(np.sqrt(np.mean(res[inl] ** 2))) if inl.any() else float("inf"), len(frames), inl,
                    [float(g) for g in gs])


def _piece_fit(cam, ts, px, n_try: int = 24, seed: int = 0):
    """Free 3D flight (start + velocity) through one piece's pixels (RANSAC over 4-point samples, then refit on
    inliers). Returns (p0, v) or None."""
    if len(ts) < 4:
        return None
    rng = np.random.default_rng(seed)
    lo, hi = [-12, -20, 0.0, -60, -130, -60], [32, 64, 15, 60, 130, 60]
    best = None
    for k in range(n_try):
        idx = np.sort(rng.choice(len(ts), min(4, len(ts)), replace=False))

        def res(q, idx=idx):
            return (cam.project(_flight(q[:3], q[3:], ts[idx])) - px[idx]).ravel()
        q0 = [10.0, (12.0, 32.0)[k % 2], 3.0, 0.0, (30.0, -30.0)[k % 2], (5.0, 15.0)[(k // 2) % 2]]
        r = least_squares(res, q0, bounds=(lo, hi))
        d = np.linalg.norm(cam.project(_flight(r.x[:3], r.x[3:], ts)) - px, axis=1)
        n = int((d < 6).sum())
        if best is None or n > best[0]:
            best = (n, r.x, d < 6)
    if best is None or best[0] < 3:
        return None
    m = best[2]
    r = least_squares(lambda q: (cam.project(_flight(q[:3], q[3:], ts[m])) - px[m]).ravel(), best[1], bounds=(lo, hi))
    return r.x[:3], r.x[3:]


def init_from_pieces(cam, frames, px, fps, jf, kinds):
    """Starting junction positions: fit every piece between junctions on its own, then place each junction where
    its neighbouring pieces meet (mean of the two ends; bounces on the floor)."""
    frames = np.asarray(frames, float)
    ends = [[] for _ in jf]
    for s in range(len(jf) - 1):
        m = (frames >= jf[s]) & (frames <= jf[s + 1])
        ts = (frames[m] - jf[s]) / fps
        pf = _piece_fit(cam, ts, px[m], seed=s)
        if pf is None:
            continue
        p0, v = pf
        T = (jf[s + 1] - jf[s]) / fps
        ends[s].append(_flight(p0, v, np.array([0.0]))[0])
        ends[s + 1].append(_flight(p0, v, np.array([T]))[0])
    init = []
    for j, e in enumerate(ends):
        if e:
            g = np.mean(e, 0)
            if kinds[j] == "bounce":
                g[2] = 0.0
            elif kinds[j] == "net":
                g[1], g[2] = NET_Y, float(np.clip(g[2], 0.0, 3.0))
            init.append(g)
        else:
            init.append(None)
    return init


def _backproject(cam, uv, h):
    """Court point at height h whose projection is pixel uv."""
    def res(q):
        return cam.project([[q[0], q[1], h]])[0] - uv
    r = least_squares(res, [10.0, 22.0], bounds=([-15, -25], [35, 70]))
    return np.array([r.x[0], r.x[1], h])


def reconstruct(cam, ball: dict, f_start: int, f_end: int, fps: float, players=None, contact_frames=(),
                lam_contact: float = 30.0, lam_bounce: float = 20.0, penalty_2d: float = 150.0, passes: int = 2):
    """3D reconstruction of one rally stretch. ball: {frame: [x, y, interpolated]}; contact_frames: frames known to
    be contacts (they stay contacts). Candidate junctions = 2D flight breaks (deliberately generous). Model
    selection (MDL-style): every candidate is tried as bounce, contact or no junction at all; the option with the
    lowest robust reprojection cost + lam per junction wins. Moves are evaluated on a local window (the two flights
    on each side, outer junctions fixed), then the whole rally is refitted. Returns RallyFit or None."""
    fr = [f for f in range(f_start, f_end + 1) if f in ball and not ball[f][2]]
    if len(fr) < 8:
        return None
    fr_a = np.array(fr, float)
    px = np.array([ball[f][:2] for f in fr], float)
    segs = segment_flights(fr_a, px, penalty=penalty_2d)
    forced = {int(c) for c in contact_frames if fr[0] < c < fr[-1]}
    jf = sorted({fr[0], fr[-1]} | {fr[a] for a, b in segs[1:]} | forced)
    kinds = ["start"] + ["contact" if f in forced else "bounce" for f in jf[1:-1]] + ["end"]
    best = fit(cam, fr, px, fps, kinds, jf, players)
    lam = {"contact": lam_contact, "bounce": lam_bounce}
    for _ in range(passes):
        changed = False
        j = 1
        while j < len(jf) - 1:
            if int(jf[j]) in forced:
                j += 1
                continue
            lo, hi = max(0, j - 2), min(len(jf) - 1, j + 2)
            m = (fr_a >= jf[lo]) & (fr_a <= jf[hi])
            options = []
            for opt in ("bounce", "contact", "none"):
                wj = [jf[x] for x in range(lo, hi + 1) if not (x == j and opt == "none")]
                wk = ["start"] + [opt if x == j else kinds[x] for x in range(lo + 1, hi) if not (x == j and opt == "none")] + ["end"]
                init = [best.junctions[x].xyh for x in range(lo, hi + 1) if not (x == j and opt == "none")]
                f_ = fit(cam, fr_a[m], px[m], fps, wk, wj, players, init=init)
                pen = sum(lam.get(k, 0.0) for k in wk[1:-1])
                options.append((f_.cost + pen, opt))
            cost, opt = min(options)
            if opt != kinds[j]:
                changed = True
                if opt == "none":
                    del jf[j], kinds[j]
                    best = fit(cam, fr, px, fps, kinds, jf, players, init=[jj.xyh for x, jj in enumerate(best.junctions) if x != j])
                    continue
                kinds[j] = opt
                best = fit(cam, fr, px, fps, kinds, jf, players, init=[jj.xyh for jj in best.junctions])
            j += 1
        if not changed:
            break
    return best


def line_call(xy, sd, k: float = 1.96):
    """'in' / 'out' / 'close' for a bounce point with 1-sigma sd (x, y): out of bounds by more than k sigma = out."""
    d_in = min(xy[0], COURT_W - xy[0], xy[1], COURT_L - xy[1])
    s = float(np.hypot(sd[0], sd[1])) if sd is not None and np.all(np.isfinite(sd[:2])) else 1.0
    if d_in > k * s:
        return "in", d_in, s
    if d_in < -k * s:
        return "out", d_in, s
    return "close", d_in, s


def _seg_residuals(cam, frames, px, fps, rf):
    """Per-observation pixel residual of a RallyFit (closed-form flights between its junctions)."""
    P = [j.xyh for j in rf.junctions]
    jf = np.array([j.frame for j in rf.junctions], float)
    seg = np.clip(np.searchsorted(jf, frames, side="right") - 1, 0, len(jf) - 2)
    gs = rf.g or [G] * (len(jf) - 1)
    pred = np.zeros((len(frames), 3))
    for s in range(len(jf) - 1):
        m = seg == s
        if m.any():
            v = _velocity(P[s], P[s + 1], (jf[s + 1] - jf[s]) / fps, gs[s])
            pred[m] = _flight(P[s], v, (frames[m] - jf[s]) / fps, gs[s])
    return np.linalg.norm(cam.project(pred) - px, axis=1), seg


def refine(cam, frames, px, fps, jf, kinds, players=None, lam: float = 60.0, passes: int = 4, bad_px: float = 10.0,
           anchors=()):
    """Hybrid-system estimate from initial junctions (e.g. detected hits + bounces):
      1. time shifts: every interior junction may move +-2 frames (event detectors are off by a frame or two);
      2. insertion: where a flight leaves >= 3 consecutive observations above bad_px, a junction is inserted at the
         first of them (tried as bounce and as contact) - missed bounces / contacts;
      3. removal: a junction whose removal costs less than lam is dropped.
    A change is kept only if the robust cost (+ lam per junction) goes down. Each move is judged on a local window
    (two junctions either side, window ends free) - a pixel only depends on the junctions around its flight - and
    the whole rally is refitted once per pass (~10x faster than refitting it per move).
    anchors: frames of trusted contacts (e.g. the hit detector's live touches): they stay contacts, are never removed
    and move at most +-2 frames from where they started - near the paddle the ball is often hidden and a free fit
    drifts the contact 5-12 frames late (US Open tags). Returns the best RallyFit."""
    frames = np.asarray(frames, float)
    jf, kinds = list(jf), list(kinds)
    keep = [0]  # initial junctions closer than MIN_GAP: keep one (a contact over a bounce)
    for j in range(1, len(jf)):
        if jf[j] - jf[keep[-1]] >= MIN_GAP or j == len(jf) - 1:
            keep.append(j)
        elif kinds[j] == "contact" and kinds[keep[-1]] == "bounce" and keep[-1] != 0:
            keep[-1] = j
    jf, kinds = [jf[j] for j in keep], [kinds[j] for j in keep]
    anchors = {int(round(a)) for a in anchors}
    orig = [f if (k == "contact" and int(round(f)) in anchors) else None for f, k in zip(jf, kinds)]  # anchor origin
    best = fit(cam, frames, px, fps, kinds, jf, players)
    X = [x.xyh for x in best.junctions]

    def local(cj, ck, cX, lo, hi):
        """Robust cost + lam per interior junction on junctions lo..hi of (cj, ck) (window ends free)."""
        lo, hi = max(0, lo), min(len(cj) - 1, hi)
        m = (frames >= cj[lo]) & (frames <= cj[hi])
        if m.sum() < 4:
            return np.inf, None
        wk = ["start"] + ck[lo + 1:hi] + ["end"]
        f_ = fit(cam, frames[m], px[m], fps, wk, cj[lo:hi + 1], players, init=cX[lo:hi + 1])
        return f_.cost + lam * (hi - lo - 1), (lo, f_)

    def take(cX, lohf):
        lo, f_ = lohf
        cX = list(cX)
        for i, jn in enumerate(f_.junctions[1:-1], start=lo + 1):
            cX[i] = jn.xyh
        return cX

    for _ in range(passes):
        improved = False
        # 1. time shifts
        for j in range(1, len(jf) - 1):
            c0, _w = local(jf, kinds, X, j - 2, j + 2)
            for d in (-2, -1, 1, 2):
                t = jf[j] + d
                if not (jf[j - 1] + MIN_GAP <= t <= jf[j + 1] - MIN_GAP) or (orig[j] is not None and abs(t - orig[j]) > 2):
                    continue
                cj = jf[:j] + [t] + jf[j + 1:]
                c1, w1 = local(cj, kinds, X, j - 2, j + 2)
                if c1 < c0 - 1e-6:
                    jf, X, c0, improved = cj, take(X, w1), c1, True
        # 1b. retype: a 'bounce' close to the net may really be the ball stopped by the net, and vice versa
        for j in range(1, len(jf) - 1):
            if orig[j] is not None or kinds[j] not in ("bounce", "net"):
                continue
            other = "net" if kinds[j] == "bounce" else "bounce"
            if other == "net" and X[j] is not None and abs(X[j][1] - NET_Y) > 8:
                continue  # far from the net: not worth a fit
            c0, _w = local(jf, kinds, X, j - 2, j + 2)
            ck = kinds[:j] + [other] + kinds[j + 1:]
            cX = X[:j] + [None] + X[j + 1:]
            c1, w1 = local(jf, ck, cX, j - 2, j + 2)
            if c1 < c0 - 1e-6:
                kinds, X, improved = ck, take(cX, w1), True
        best = fit(cam, frames, px, fps, kinds, jf, players, init=X)
        X = [x.xyh for x in best.junctions]
        # 2. insertion where a flight breaks
        res, _seg = _seg_residuals(cam, frames, px, fps, best)
        bad = res > bad_px
        i = 0
        while i < len(frames) - 2:
            if not (bad[i] and bad[i + 1] and bad[i + 2]) or any(abs(frames[i] - x) < MIN_GAP for x in jf):
                i += 1
                continue
            t = frames[i]
            k = int(np.searchsorted(jf, t))
            c0, _w = local(jf, kinds, X, k - 2, k + 1)  # same frames as the new junction's window below
            for kind in ("bounce", "contact", "net"):
                cj, ck, cX = jf[:k] + [t] + jf[k:], kinds[:k] + [kind] + kinds[k:], X[:k] + [None] + X[k:]
                c1, w1 = local(cj, ck, cX, k - 2, k + 2)
                if c1 < c0:
                    jf, kinds, X, improved = cj, ck, take(cX, w1), True
                    orig = orig[:k] + [None] + orig[k:]
                    break
            while i < len(frames) and bad[i]:  # one try per run of bad observations
                i += 1
        if improved:
            best = fit(cam, frames, px, fps, kinds, jf, players, init=X)
            X = [x.xyh for x in best.junctions]
        # 3. removal
        j = 1
        while j < len(jf) - 1:
            if orig[j] is not None:
                j += 1
                continue
            c0, _w = local(jf, kinds, X, j - 2, j + 2)
            cj, ck, cX = jf[:j] + jf[j + 1:], kinds[:j] + kinds[j + 1:], X[:j] + X[j + 1:]
            c1, w1 = local(cj, ck, cX, j - 2, j + 1)  # same frames, one junction fewer
            if c1 < c0:
                jf, kinds, X, improved = cj, ck, take(cX, w1), True
                orig = orig[:j] + orig[j + 1:]
                continue
            j += 1
        best = fit(cam, frames, px, fps, kinds, jf, players, init=X)
        X = [x.xyh for x in best.junctions]
        if not improved:
            break
    return best


def end_of_play(cam, frames, px, fps, rf, players=None, last_contact=None, hitter_far=None, own_side_s: float = 0.3):
    """'Ball dead' end state: after the rally's last contact the point is over at the ball's first bounce or net hit
    (a ball that bounces and is not hit again - in or out - ends the point; a net junction = the ball stopped by the
    net, which ballistic flight alone cannot model: it 'flew through' and was called in on the other side). Everything
    tracked after that is ball handling (picked up, rolled back, bounced before the next serve), which otherwise
    makes the fit invent bounces and contacts. last_contact: frame of the last live touch (rally-end model), else
    the fit's own last contact. hitter_far: side of the last hitter (tracked player) - a bounce on the hitter's own
    side sooner than own_side_s after the contact is impossible (even a net ball needs ~0.3 s to drop) and is skipped:
    near the paddle the track is noisy (TNPPL: fake bounce 4 frames after the last hit ended the point there).
    Returns the refitted RallyFit (ending at that bounce + MIN_GAP) or rf unchanged."""
    frames = np.asarray(frames, float)
    jf = [j.frame for j in rf.junctions]
    kinds = [j.kind for j in rf.junctions]
    lc = [i for i, k in enumerate(kinds) if k == "contact" and (last_contact is None or jf[i] <= last_contact + 3)]
    if not lc:
        return rf
    def plausible(i):
        if hitter_far is None:
            return True
        own = (rf.junctions[i].xyh[1] < NET_Y) == hitter_far
        return not (own and jf[i] - jf[lc[-1]] < own_side_s * fps)
    nb = next((i for i in range(lc[-1] + 1, len(jf) - 1) if kinds[i] == "net" or (kinds[i] == "bounce" and plausible(i))), None)
    if nb is None or nb >= len(jf) - 2:
        return rf
    cut = jf[nb] + MIN_GAP
    m = frames <= cut
    if m.sum() < 8:
        return rf
    end = float(frames[m][-1])
    if end <= jf[nb]:
        return rf
    cj, ck = jf[:nb + 1] + [end], kinds[:nb + 1] + ["end"]
    return fit(cam, frames[m], px[m], fps, ck, cj, players, init=[j.xyh for j in rf.junctions[:nb + 1]] + [None])
