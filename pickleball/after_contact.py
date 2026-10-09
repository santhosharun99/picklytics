"""Follow the ball after every paddle contact, frame by frame, with real 3D physics.

For a contact at frame f: every tracked ball pixel from f+1 until the next contact (or 2 s) is collected. The first
smooth piece of that track (flights.segment_flights: until the ball bounces, hits the net or is hit again) is fitted
with a 3D ballistic flight seen through the calibrated camera (camera3d): start point near the paddle, launch
velocity, gravity (32.2 ft/s^2) and air drag, chosen so the projected flight lands on the tracked pixels
(least squares). From the fitted flight:
  - height when it reaches the net plane (y = 22 ft) vs the net top (36 in at the posts, 34 in in the middle):
    'net' if it reaches the net plane below the tape, else 'crossed';
  - landing point (height 0) and its signed distance to the nearest line (+ inside, - outside), unless the
    ball is hit again first (volley).
Works whether or not the ball bounces - in doubles most shots are volleyed.
"""
import numpy as np
from scipy.optimize import least_squares

from .court import COURT_L, COURT_W, NET_Y
from .flights import segment_flights

G = 32.2          # ft/s^2
DRAG = 0.12       # 1/s, light linear drag (pickleball: holes, ~0.9 oz) - keeps long drives from overshooting


def _net_h(x: float) -> float:
    return 3.0 - (2 / 12) * (1 - min(abs(x - 10) / 11, 1.0))


def _pos(p, t):
    """Ballistic flight with linear drag. p = x0, y0, h0, vx, vy, vh; t in seconds -> (N, 3) x, y, h."""
    x0, y0, h0, vx, vy, vh = p
    k = DRAG
    e = (1 - np.exp(-k * t)) / k
    x = x0 + vx * e
    y = y0 + vy * e
    h = h0 + (vh + G / k) * e - G / k * t
    return np.c_[x, y, h]


def fit_flight_ransac(cam, ts, px, start_xy, side, trials: int = 40, inlier_px: float = 8.0, seed: int = 0):
    """Robust 3D flight: the flight leaving the contact that explains the most tracked pixels (RANSAC). Pixels it
    does not explain - another ball, a ball kid, a logo, the ball after a bounce or net touch - are outliers.
    Returns (params, rms error on inliers, inlier mask) or None."""
    n = len(ts)
    if n < 5:
        return None
    rng = np.random.default_rng(seed)
    toward = -1.0 if side == "near" else 1.0
    x0, y0 = start_xy
    vlo, vhi = (-120.0, -3.0) if toward < 0 else (3.0, 120.0)
    lo, hi = [x0 - 3, y0 - 3, 0.3, -40, vlo, -60], [x0 + 3, y0 + 3, 9.0, 40, vhi, 60]
    best = None
    early = np.where(ts <= min(1.0, ts[-1]))[0]
    for k in range(trials):
        if len(early) < 4:
            break
        idx = np.sort(rng.choice(early, 4, replace=False))

        def res(p, idx=idx):
            return (cam.project(_pos(p, ts[idx])) - px[idx]).ravel()
        p0 = [x0, y0, 3.0, 0.0, toward * (20.0 + 15.0 * (k % 3)), (-4.0, 8.0, 18.0)[(k // 3) % 3]]
        r = least_squares(res, p0, bounds=(lo, hi))
        d = np.linalg.norm(cam.project(_pos(r.x, ts)) - px, axis=1)
        inl = d < inlier_px
        # the flight must explain the pixels right after the contact (it starts at the paddle)
        score = inl.sum() + 2 * inl[:3].sum()
        if best is None or score > best[0]:
            best = (score, r.x, inl)
    if best is None or best[2].sum() < 5:
        return None
    inl = best[2]

    def res_all(p):
        return (cam.project(_pos(p, ts[inl])) - px[inl]).ravel()
    r = least_squares(res_all, best[1], bounds=(lo, hi))
    err = float(np.sqrt(np.mean(r.fun ** 2)))
    return r.x, err, inl


def fit_flight(cam, ts, px, start_xy, side):
    """3D flight through the tracked pixels px (N, 2) at times ts (s after contact). start_xy: hitter's court
    position. Returns (params, rms pixel error)."""
    toward = -1.0 if side == "near" else 1.0  # the ball must head to the other side for a normal shot
    x0, y0 = start_xy

    def res(p):
        return (cam.project(_pos(p, ts)) - px).ravel()

    best = None
    # the shot heads to the other side: velocity along the court is bounded to that direction
    vlo, vhi = (-120.0, -3.0) if toward < 0 else (3.0, 120.0)
    for vy in (15.0, 35.0, 60.0):
        for vh in (-5.0, 8.0, 20.0):
            p0 = [x0, y0, 3.0, 0.0, toward * vy, vh]
            lo = [x0 - 5, y0 - 5, 0.3, -40, vlo, -60]
            hi = [x0 + 5, y0 + 5, 9.0, 40, vhi, 60]
            r = least_squares(res, p0, bounds=(lo, hi), loss="soft_l1", f_scale=6.0)
            err = float(np.sqrt(np.mean(r.fun ** 2)))
            if best is None or err < best[1]:
                best = (r.x, err)
    return best


def _contact_point(cam, px, side):
    """Court (x, y) under the ball at the contact: the point at 3 ft height (typical contact) whose projection is
    the ball's pixel - taken from the ball itself, not from the player credited with the hit (often the partner)."""
    def res(q):
        return cam.project([[q[0], q[1], 3.0]])[0] - px
    y0 = 33.0 if side == "near" else 11.0
    r = least_squares(res, [10.0, y0], bounds=([-10, 22.0 if side == "near" else -15], [30, 60 if side == "near" else 22.0]))
    return r.x


def _moving(fr, xy, min_px: float = 3.0, run: int = 4):
    """Drop stretches where the 'ball' stands still (a ball lying by the fence, a logo) - a ball in flight always
    moves. Keeps index lists aligned."""
    keep = np.ones(len(fr), bool)
    i = 0
    while i < len(fr):
        j = i
        while j + 1 < len(fr) and np.hypot(*(xy[j + 1] - xy[j])) < min_px:
            j += 1
        if j - i + 1 >= run:
            keep[i:j + 1] = False
        i = j + 1
    return [f for f, k in zip(fr, keep) if k], xy[keep]


def follow(cam, ball: dict, f0: int, f_next: int | None, start_xy, side: str, fps: float, max_s: float = 2.0) -> dict:
    """Outcome of the flight after a contact. ball: {frame: [x, y, interpolated]} (render_data); f_next: frame
    of the next contact (None = no further touch)."""
    end = min(f_next - 1 if f_next else 10 ** 9, f0 + int(max_s * fps))
    fr = [f for f in range(f0 + 1, end + 1) if f in ball and not ball[f][2]]
    out = {"n_px": len(fr), "fit_err_px": None, "inliers": 0, "inlier_share": None, "h_at_net": None, "crossed": None, "land": None, "land_line_ft": None,
           "volleyed_before_landing": None, "piece_s": None}
    if len(fr) < 5:
        return out
    xy = np.array([ball[f][:2] for f in fr], float)
    fr, xy = _moving(fr, xy)
    if len(fr) < 6:
        return out
    if f0 in ball:  # start from the ball itself at the contact
        start_xy = _contact_point(cam, np.array(ball[f0][:2], float), side)
    ts = (np.array(fr, float) - f0) / fps
    fit = fit_flight_ransac(cam, ts, xy, start_xy, side)
    if fit is None:
        return out
    p, err, inl = fit
    out["fit_err_px"] = round(err, 1)
    out["inliers"] = int(inl.sum())
    out["inlier_share"] = round(float(inl.mean()), 2)
    out["piece_s"] = round(float(ts[inl][-1]), 2)
    tt = np.linspace(0, 3.0, 601)
    P = _pos(p, tt)
    toward = (NET_Y - P[:, 1]) if side == "near" else (P[:, 1] - NET_Y)
    land_i = np.argmax(P[:, 2] <= 0) if (P[:, 2] <= 0).any() else None
    ci = np.argmax(toward >= 0) if (toward >= 0).any() else None
    if ci is not None and (land_i is None or ci < land_i):
        out["h_at_net"] = round(float(P[ci, 2]), 2)
        out["crossed"] = bool(P[ci, 2] > _net_h(P[ci, 0]))
    else:
        out["crossed"] = False  # lands on the hitter's own side before reaching the net
    if land_i is not None and (out["crossed"] or ci is None):
        lx, ly = P[land_i, :2]
        out["land"] = [round(float(lx), 2), round(float(ly), 2)]
        out["land_line_ft"] = round(float(min(lx, COURT_W - lx, ly, COURT_L - ly)), 2)
        if f_next:
            out["volleyed_before_landing"] = bool((f_next - f0) / fps < tt[land_i])
    return out
