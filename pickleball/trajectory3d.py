"""3D ball flights from one fixed camera.

Between two contacts the ball moves under gravity and air drag only. With a calibrated
camera (camera3d), each flight is a 3D curve with 6 unknowns (start position + velocity);
gravity fixes the scale, so one camera is enough. We use linear drag (dv/dt = -beta v + g),
which keeps every pixel observation a LINEAR equation in the unknowns:

    X(t) = A + B * exp(-beta t) + (0, 0, -(g / beta) t)        (beta > 0)
    X(t) = A + B * t + (0, 0, -g t^2 / 2)                       (beta = 0, no drag)

so fits are plain least squares and the best split of a rally into flights can be found with
dynamic programming over all breakpoints (like flights.py, but in 3D instead of image space).
At each split we know the contact height (0 = bounce on the ground, > 1 ft = in the air: a
hit), and the 3D speed / direction before and after.
"""
import numpy as np

G = 32.17  # ft/s^2
S = np.array([1.0, 1.0, -1.0])  # (x, y, h) -> OpenCV (x, y, z)


def _basis(t, beta):
    if beta <= 0:
        return t, -0.5 * G * t ** 2
    return np.exp(-beta * t), -(G / beta) * t


def _rows(P, uv, t, beta, w):
    """Two weighted linear equations per observation: rows (N*2, 6), rhs (N*2,)."""
    phi, force = _basis(t, beta)
    rows, rhs = [], []
    for r in (0, 1):
        q = uv[:, r:r + 1] * P[2] - P[r]           # (N, 4): q . (X, 1) = 0
        q3 = q[:, :3] * S
        rows.append(np.c_[q3, q3 * phi[:, None]] * w[:, None])
        rhs.append(-(q3[:, 2] * force + q[:, 3]) * w)
    R = np.stack(rows, 1).reshape(-1, 6)
    b = np.stack(rhs, 1).reshape(-1)
    return R, b


def positions(theta, t, beta):
    phi, force = _basis(np.asarray(t, float), beta)
    A, B = theta[:3], theta[3:]
    X = A[None] + B[None] * phi[:, None]
    X[:, 2] += force
    return X


def velocities(theta, t, beta):
    t = np.asarray(t, float)
    B = theta[3:]
    if beta <= 0:
        V = np.repeat(B[None], len(t), 0)
        V[:, 2] -= G * t
    else:
        V = B[None] * (-beta * np.exp(-beta * t))[:, None]
        V[:, 2] -= G / beta
    return V


def fit_flight(P, uv, t, beta, depth, iters=2, ridge=1e-6):
    """Least-squares 3D flight through pixel observations uv (N, 2) at times t (s, local)."""
    w = np.full(len(t), 1.0 / depth)
    theta = np.zeros(6)
    for _ in range(iters):
        R, b = _rows(P, uv, t, beta, w)
        M = R.T @ R
        theta = np.linalg.solve(M + ridge * np.trace(M) / 6 * np.eye(6), R.T @ b)
        X = positions(theta, t, beta) * S
        d = X @ P[2, :3] + P[2, 3]               # depth of each fitted point
        w = 1.0 / np.clip(d, 5.0, None)
    return theta


def reproject(P, X):
    Xh = np.c_[X * S, np.ones(len(X))]
    p = Xh @ P.T
    return p[:, :2] / p[:, 2:3]


def segment_3d(P, frames, uv, fps, beta=0.5, penalty=900.0, min_len=4, max_len=60, depth=60.0,
               h_range=(-1.5, 25.0)):
    """Optimal split of one ball track chunk into 3D flights.

    Cost of a flight = sum of squared (approx.) pixel residuals, +inf when the fitted curve
    leaves the plausible height range. Returns list of (i, j) inclusive index ranges."""
    t_all = (np.asarray(frames, float) - frames[0]) / fps
    n = len(t_all)
    W = 1.0 / depth
    cost = np.full((n, max_len + 1), np.inf)  # cost[i, L] = cost of flight i .. i+L-1
    for i in range(n):
        L = min(max_len, n - i)
        if L < min_len:
            continue
        t = t_all[i:i + L] - t_all[i]
        R, b = _rows(P, uv[i:i + L], t, beta, np.full(L, W))
        R2 = R.reshape(L, 2, 6)
        b2 = b.reshape(L, 2)
        M = np.cumsum(np.einsum("nri,nrj->nij", R2, R2), 0)        # (L, 6, 6)
        v = np.cumsum(np.einsum("nri,nr->ni", R2, b2), 0)          # (L, 6)
        bb = np.cumsum((b2 ** 2).sum(1))
        ks = np.arange(min_len - 1, L)
        Mk = M[ks] + 1e-6 * np.trace(M[ks], axis1=1, axis2=2)[:, None, None] / 6 * np.eye(6)
        th = np.linalg.solve(Mk, v[ks][..., None])[..., 0]
        c = np.maximum(0.0, bb[ks] - np.einsum("ki,ki->k", th, v[ks]))
        # physically implausible flights (underground / sky-high) are not allowed
        for kk, k in enumerate(ks):
            if not np.isfinite(c[kk]):
                continue
            h = positions(th[kk], t[[0, k // 2, k]], beta)[:, 2]
            if h.min() < h_range[0] or h.max() > h_range[1]:
                c[kk] = np.inf
        cost[i, ks + 1] = c
    best = np.full(n + 1, np.inf)
    best[0] = 0.0
    prev = np.zeros(n + 1, int)
    for j in range(min_len, n + 1):
        for L in range(min_len, min(max_len, j) + 1):
            i = j - L
            c = best[i] + cost[i, L] + penalty
            if c < best[j]:
                best[j], prev[j] = c, i
    if not np.isfinite(best[n]):
        return []
    segs, j = [], n
    while j > 0:
        segs.append((prev[j], j - 1))
        j = prev[j]
    return segs[::-1]


def fit_chunk(P, t, uv, segs, fps, beta, lam=150.0, depth=60.0, iters=3, ground=()):
    """Joint fit of consecutive flights that must MEET at each contact (position continuous,
    velocity free). Independent fits let each flight drift along the camera ray; chaining them
    (and pinning bounces to h=0 via `ground` = indices of boundaries known to be bounces)
    fixes the depth. Returns (thetas (K, 6), contact frames (K-1,))."""
    K = len(segs)
    t0 = np.array([t[i] for i, _ in segs], float)
    # contact time: middle of the gap between the last point of one flight and the first of the next
    tc = np.array([(t[segs[k][1]] + t[segs[k + 1][0]]) / 2 for k in range(K - 1)])
    w_all = np.full(len(t), 1.0 / depth)
    thetas = np.zeros((K, 6))
    for _ in range(iters):
        blocks, rhs = [], []
        for k, (i, j) in enumerate(segs):
            R, b = _rows(P, uv[i:j + 1], (t[i:j + 1] - t0[k]) / fps, beta, w_all[i:j + 1])
            Z = np.zeros((len(R), 6 * K))
            Z[:, 6 * k:6 * k + 6] = R
            blocks.append(Z)
            rhs.append(b)
        for k in range(K - 1):  # X_k(tc) == X_{k+1}(tc)
            ta, tb = (tc[k] - t0[k]) / fps, (tc[k] - t0[k + 1]) / fps
            pa, fa = _basis(np.array([ta]), beta)
            pb, fb = _basis(np.array([tb]), beta)
            Z = np.zeros((3, 6 * K))
            Z[:, 6 * k:6 * k + 3] = np.eye(3)
            Z[:, 6 * k + 3:6 * k + 6] = np.eye(3) * pa[0]
            Z[:, 6 * k + 6:6 * k + 9] = -np.eye(3)
            Z[:, 6 * k + 9:6 * k + 12] = -np.eye(3) * pb[0]
            r = np.zeros(3)
            r[2] = -(fa[0] - fb[0])
            blocks.append(Z * lam)
            rhs.append(r * lam)
            if k in ground:  # bounce: height 0 at contact
                Zg = np.zeros((1, 6 * K))
                Zg[0, 6 * k + 2], Zg[0, 6 * k + 5] = 1.0, pa[0]
                blocks.append(Zg * lam)
                rhs.append(np.array([-fa[0]]) * lam)
        A = np.vstack(blocks)
        b = np.concatenate(rhs)
        M = A.T @ A
        x = np.linalg.solve(M + 1e-6 * np.trace(M) / len(M) * np.eye(len(M)), A.T @ b)
        thetas = x.reshape(K, 6)
        for k, (i, j) in enumerate(segs):  # per-point depth for the next pass
            X = positions(thetas[k], (t[i:j + 1] - t0[k]) / fps, beta) * S
            w_all[i:j + 1] = 1.0 / np.clip(X @ P[2, :3] + P[2, 3], 5.0, None)
    return thetas, tc


def classify_contact(v_in, v_out, h, ground_h=1.0, min_vy=3.0):
    """'hit' when the ball turns back along the court (every hit crosses the net),
    'bounce' when it keeps going but its height is ~0, else None (spurious split)."""
    if v_in[1] * v_out[1] < 0 and max(abs(v_in[1]), abs(v_out[1])) > min_vy:
        return "hit"
    if h < ground_h:
        return "bounce"
    return None


def track_flights(P, ball: dict, fps: float, beta=0.5, penalty=900.0, max_gap=12, joint=True, **kw):
    """3D flights for a whole ball track {frame: (x, y, conf, interpolated)}.

    Returns (flights, contacts):
      flights:  [{"f0", "f1", "theta", "beta", "rms_px"}]  (theta in local time from f0)
      contacts: [{"frame", "pos": (x, y, h), "v_in", "v_out", "speed_in_mph", "speed_out_mph"}]
    """
    from .flights import _remove_outliers
    fr = sorted(f for f, v in ball.items() if not v[3])
    chunks, cur = [], fr[:1]
    for a, b in zip(fr, fr[1:]):
        if b - a > max_gap:
            chunks.append(cur)
            cur = []
        cur.append(b)
    chunks.append(cur)
    flights, contacts = [], []
    for ch in chunks:
        if len(ch) < 10:
            continue
        t, uv = _remove_outliers(np.array(ch, float), np.array([ball[f][:2] for f in ch], float))
        if len(t) < 10:
            continue
        segs = segment_3d(P, t, uv, fps, beta, penalty, **kw)
        if not segs:
            continue
        if joint:
            fl, cs = _joint(P, t, uv, segs, fps, beta)
            flights += fl
            contacts += cs
            continue
        fl = []
        for i, j in segs:
            ts = (t[i:j + 1] - t[i]) / fps
            th = fit_flight(P, uv[i:j + 1], ts, beta, 60.0)
            res = reproject(P, positions(th, ts, beta)) - uv[i:j + 1]
            fl.append({"f0": int(t[i]), "f1": int(t[j]), "theta": th, "beta": beta,
                       "rms_px": float(np.sqrt((res ** 2).sum(1).mean()))})
        flights += fl
        for a, b in zip(fl, fl[1:]):
            # contact = frame between the flights where the two 3D curves come closest
            grid = np.arange(a["f1"], b["f0"] + 1)
            pa = positions(a["theta"], (grid - a["f0"]) / fps, beta)
            pb = positions(b["theta"], (grid - b["f0"]) / fps, beta)
            k = int(np.argmin(np.linalg.norm(pa - pb, axis=1)))
            f = int(grid[k])
            vi = velocities(a["theta"], [(f - a["f0"]) / fps], beta)[0]
            vo = velocities(b["theta"], [(f - b["f0"]) / fps], beta)[0]
            contacts.append(_contact(f, (pa[k] + pb[k]) / 2, vi, vo, float(np.linalg.norm(pa[k] - pb[k]))))
    return flights, contacts


def _contact(f, pos, vi, vo, gap=0.0):
    return {"frame": int(f), "pos": np.asarray(pos).tolist(), "gap_ft": gap, "v_in": vi.tolist(), "v_out": vo.tolist(),
            "speed_in_mph": float(np.linalg.norm(vi) * 0.6818), "speed_out_mph": float(np.linalg.norm(vo) * 0.6818),
            "kind": classify_contact(vi, vo, float(pos[2]))}


def _joint(P, t, uv, segs, fps, beta, rounds=4):
    """Chained fit; then pin bounces to the ground and drop spurious splits, and refit."""
    ground = set()
    for _ in range(rounds):
        th, tc = fit_chunk(P, t, uv, segs, fps, beta, ground=ground)
        kinds = []
        for k in range(len(segs) - 1):
            pa = positions(th[k], [(tc[k] - t[segs[k][0]]) / fps], beta)[0]
            vi = velocities(th[k], [(tc[k] - t[segs[k][0]]) / fps], beta)[0]
            vo = velocities(th[k + 1], [(tc[k] - t[segs[k + 1][0]]) / fps], beta)[0]
            kinds.append(classify_contact(vi, vo, pa[2]))
        drop = [k for k, kd in enumerate(kinds) if kd is None]
        new_ground = {k for k, kd in enumerate(kinds) if kd == "bounce"}
        if not drop and new_ground == ground:
            break
        if drop:  # merge flights split without a real contact
            keep = [k for k in range(len(segs) - 1) if k not in drop]
            bounds = [segs[0][0]] + [segs[k + 1][0] for k in keep] + [len(t)]
            segs = [(a, b - 1) for a, b in zip(bounds, bounds[1:])]
            ground = set()
        else:
            ground = new_ground
    th, tc = fit_chunk(P, t, uv, segs, fps, beta, ground=ground)
    flights, contacts = [], []
    for k, (i, j) in enumerate(segs):
        ts = (t[i:j + 1] - t[i]) / fps
        res = reproject(P, positions(th[k], ts, beta)) - uv[i:j + 1]
        flights.append({"f0": int(t[i]), "f1": int(t[j]), "theta": th[k], "beta": beta,
                        "rms_px": float(np.sqrt((res ** 2).sum(1).mean()))})
    for k in range(len(segs) - 1):
        ta, tb = (tc[k] - t[segs[k][0]]) / fps, (tc[k] - t[segs[k + 1][0]]) / fps
        pos = positions(th[k], [ta], beta)[0]
        contacts.append(_contact(round(tc[k]), pos, velocities(th[k], [ta], beta)[0], velocities(th[k + 1], [tb], beta)[0]))
    return flights, contacts


# ---------------------------------------------------------------------------------------------
# Anchored arcs. A free 3D fit from one camera slides along the camera ray (depth is weakly
# observed). Contacts fix it: a bounce is ON THE GROUND (ray meets h=0), a hit is AT THE HITTER
# (ray meets the plane y = hitter's court y). Between two anchored contacts gravity + drag fix
# the whole arc, so every hypothesis sequence (bounce / hit by player p / no contact at each
# candidate) predicts the image track exactly; Viterbi picks the sequence that matches best.
# ---------------------------------------------------------------------------------------------

def ray(P, uv):
    """Camera centre C and unit direction D (OpenCV world coords) through pixels uv (N, 2)."""
    M, p4 = P[:, :3], P[:, 3]
    C = -np.linalg.solve(M, p4)
    D = np.linalg.solve(M, np.c_[np.asarray(uv, float).reshape(-1, 2), np.ones(len(np.atleast_2d(uv)))].T).T
    return C, D / np.linalg.norm(D, axis=1, keepdims=True)


def ground_point(P, uv):
    """(x, y, 0) where the camera ray through uv meets the ground."""
    C, D = ray(P, uv)
    s = -C[2] / D[:, 2]
    X = C + s[:, None] * D
    return X * S


def plane_y_point(P, uv, y):
    """(x, y, h) where the camera ray through uv crosses the vertical plane at court y."""
    C, D = ray(P, uv)
    s = (y - C[1]) / D[:, 1]
    X = C + s[:, None] * D
    return X * S


def _arc_between(Sp, Ep, ts, te, beta):
    """theta of the drag+gravity arc through Sp at ts and Ep at te (times relative to theta)."""
    pa, fa = _basis(np.array([ts, te]), beta)
    B = (Sp - Ep) / (pa[0] - pa[1])
    B[2] = (Sp[2] - Ep[2] - (fa[0] - fa[1])) / (pa[0] - pa[1])
    A = Sp - B * pa[0]
    A[2] -= fa[0]
    return np.r_[A, B]


def _arc_free_end(P, uv, t, anchor, ta, beta, depth=60.0):
    """Arc through a fixed anchor at time ta, velocity fitted to observations (open-ended)."""
    pa, fa = _basis(np.array([ta]), beta)
    phi, force = _basis(t, beta)
    # X(t) = anchor + B (phi - pa) + (force - fa) e_h   -> 3 unknowns B
    rows, rhs = [], []
    for r in (0, 1):
        q = uv[:, r:r + 1] * P[2] - P[r]
        q3 = q[:, :3] * S
        rows.append(q3 * (phi - pa[0])[:, None] / depth)
        rhs.append(-(q3 @ anchor + q3[:, 2] * (force - fa[0]) + q[:, 3]) / depth)
    R, b = np.vstack(rows), np.concatenate(rhs)
    B = np.linalg.lstsq(R, b, rcond=None)[0]
    A = anchor - B * pa[0]
    A[2] -= fa[0]
    return np.r_[A, B]


def _arc_cost(P, theta, uv, t, beta, clip=400.0):
    if len(t) == 0:
        return 0.0
    res = reproject(P, positions(theta, t, beta)) - uv
    return float(np.minimum((res ** 2).sum(1), clip).sum())


def anchored_track(P, ball: dict, candidates: list[int], players: dict, fps: float, max_gap=12, **kw):
    """anchored_events on every continuous stretch of the ball track (split at long gaps)."""
    fr = sorted(f for f, v in ball.items() if not v[3])
    chunks, cur = [], fr[:1]
    for a, b in zip(fr, fr[1:]):
        if b - a > max_gap:
            chunks.append(cur)
            cur = []
        cur.append(b)
    chunks.append(cur)
    events, arcs = [], []
    for ch in chunks:
        if len(ch) < 10:
            continue
        cs = [c for c in candidates if ch[0] - 3 <= c <= ch[-1] + 3]
        # candidates too far apart to be one flight start a new group (ball held, between points)
        span = kw.get("max_flight_s", 2.5) * fps
        groups = []
        for c in cs:
            if groups and c - groups[-1][-1] <= span:
                groups[-1].append(c)
            else:
                groups.append([c])
        for g in groups:
            sub = {f: ball[f] for f in ch if g[0] - span <= f <= g[-1] + span}
            if len(sub) < 5:
                continue
            e, a = anchored_events(P, sub, g, players, fps, max_gap=max_gap, **kw)
            events += e
            arcs += a
    return events, arcs


def anchored_events(P, ball: dict, candidates: list[int], players: dict, fps: float, beta=0.5,
                    penalty=300.0, max_flight_s=2.5, reach_x=6.0, h_range=(0.3, 10.0), max_gap=12,
                    bounce_bias=0.0, open_s=0.5):
    """Label candidate contact frames as 'hit' (with hitter id), 'bounce' or nothing.

    players: {frame: [{"id", "court": (x, y)}...]}. Returns (events, arcs) where events are
    {"frame", "kind", "player", "pos": (x, y, h)} and arcs are the fitted 3D flights."""
    fr = np.array(sorted(f for f, v in ball.items() if not v[3]))
    if len(fr) < 10:
        return [], []
    uv_all = np.array([ball[f][:2] for f in fr], float)

    def obs(f0, f1):  # observations strictly inside (f0, f1)
        i, j = np.searchsorted(fr, f0, "right"), np.searchsorted(fr, f1, "left")
        return fr[i:j], uv_all[i:j]

    def ball_uv(f):  # image ball position at f (nearest measured)
        k = int(np.clip(np.searchsorted(fr, f), 1, len(fr) - 1))
        k = k if abs(fr[k] - f) < abs(fr[k - 1] - f) else k - 1
        return uv_all[k]

    def plist(f):
        for d in range(0, 8):
            for g in (f - d, f + d):
                if players.get(g):
                    return players[g]
        return []

    # hypotheses per candidate: (kind, player id, 3D anchor)
    hyps = []
    for f in candidates:
        u = ball_uv(f)
        hs = [("bounce", None, ground_point(P, u)[0])]
        for p in plist(f):
            X = plane_y_point(P, u, p["court"][1])[0]
            if h_range[0] <= X[2] <= h_range[1] and abs(X[0] - p["court"][0]) <= reach_x:
                hs.append(("hit", p.get("id"), X))
        hyps.append(hs)

    # Viterbi over (candidate, hypothesis); start/end arcs are open-ended
    n = len(candidates)
    best = [[np.inf] * len(h) for h in hyps]
    back = [[None] * len(h) for h in hyps]
    for c in range(n):
        f = candidates[c]
        for a, (kind, pid, X) in enumerate(hyps[c]):
            # start: open arc from the first observation (within the same chunk) to this anchor
            t_o, uv_o = obs(fr[0] - 1, f)
            far = t_o < f - max_flight_s * fps
            cost0 = penalty + 400.0 * far.sum()
            t_o, uv_o = t_o[~far], uv_o[~far]
            if len(t_o) >= 3:
                th = _arc_free_end(P, uv_o, (t_o - f) / fps, X, 0.0, beta)
                cost0 += _arc_cost(P, th, uv_o, (t_o - f) / fps, beta)
            # a chain must start near the beginning of the stretch (no skipping by open arcs)
            best[c][a] = cost0 + (bounce_bias if kind == "bounce" else 0.0) if f - candidates[0] <= open_s * fps else np.inf
            for c0 in range(c - 1, -1, -1):
                f0 = candidates[c0]
                if f - f0 > max_flight_s * fps:
                    break
                t_o, uv_o = obs(f0, f)
                if len(t_o) and np.diff(np.r_[f0, t_o, f]).max() > max_gap:
                    continue
                for a0, (k0, p0, X0) in enumerate(hyps[c0]):
                    if not np.isfinite(best[c0][a0]):
                        continue
                    if kind == "hit" and k0 == "hit" and pid == p0:
                        continue  # same player cannot hit twice in a row
                    if kind == "bounce" and k0 == "bounce":
                        continue
                    th = _arc_between(X0, X, 0.0, (f - f0) / fps, beta)
                    hmid = positions(th, [(f - f0) / fps / 2], beta)[0, 2]
                    if hmid < -0.5 or hmid > 30:
                        continue
                    cst = best[c0][a0] + penalty + _arc_cost(P, th, uv_o, (t_o - f0) / fps, beta) \
                        + (bounce_bias if kind == "bounce" else 0.0)
                    if cst < best[c][a]:
                        best[c][a], back[c][a] = cst, (c0, a0)
    # end: open arc after the last anchor
    fin = []
    for c in range(n):
        f = candidates[c]
        for a, (kind, pid, X) in enumerate(hyps[c]):
            if not np.isfinite(best[c][a]) or candidates[-1] - f > open_s * fps:
                continue
            t_o, uv_o = obs(f, fr[-1] + 1)
            far = t_o > f + max_flight_s * fps
            cost1 = 400.0 * far.sum()
            t_o, uv_o = t_o[~far], uv_o[~far]
            if len(t_o) >= 3:
                th = _arc_free_end(P, uv_o, (t_o - f) / fps, X, 0.0, beta)
                cost1 = _arc_cost(P, th, uv_o, (t_o - f) / fps, beta)
            fin.append((best[c][a] + cost1, c, a))
    if not fin:
        return [], []
    # several rallies/chunks: take the best chain ending at each "last" candidate greedily
    events, used = [], set()
    _, c, a = min(fin)
    chain = []
    while (c, a) is not None and c is not None:
        chain.append((c, a))
        nxt = back[c][a]
        if nxt is None:
            break
        c, a = nxt
    chain.reverse()
    arcs = []
    for (c0, a0), (c1, a1) in zip(chain, chain[1:]):
        X0, X1 = hyps[c0][a0][2], hyps[c1][a1][2]
        f0, f1 = candidates[c0], candidates[c1]
        arcs.append({"f0": f0, "f1": f1, "theta": _arc_between(X0, X1, 0.0, (f1 - f0) / fps, beta), "beta": beta})
    for c, a in chain:
        kind, pid, X = hyps[c][a]
        events.append({"frame": int(candidates[c]), "kind": kind, "player": pid, "pos": np.round(X, 2).tolist()})
    return events, arcs
