"""Flight segmentation: split the ball's image track into smooth ballistic pieces.

Between two contacts (hit or bounce) the ball follows one smooth curve, so x(t) and y(t)
are each well described by a quadratic. A contact is where one quadratic stops fitting
and the next begins - even when the ball itself is hidden at that moment (behind the
hitter's body or paddle), because the two curves extrapolate into the gap.

The split points are found by optimal segmentation (dynamic programming over all
possible breakpoints) with a fixed penalty per extra segment.
"""
import numpy as np


def _remove_outliers(t, xy, k=3, max_dev=25.0):
    keep = np.ones(len(t), bool)
    for i in range(len(t)):
        lo, hi = max(0, i - k), min(len(t), i + k + 1)
        nb = np.delete(np.arange(lo, hi), i - lo)
        if len(nb) < 2:
            continue
        med = np.median(xy[nb], axis=0)
        if np.hypot(*(xy[i] - med)) > max_dev * max(1.0, abs(np.median(t[nb]) - t[i]) / 2):
            keep[i] = False
    return t[keep], xy[keep]


class _QuadCost:
    """O(1) sum of squared residuals of a least-squares quadratic fit on any [i, j] range."""

    def __init__(self, t, v):
        T = np.stack([np.ones_like(t), t, t ** 2, t ** 3, t ** 4], 1)
        self.S = np.vstack([np.zeros(5), np.cumsum(T, 0)])               # sums of t^k
        self.B = np.vstack([np.zeros(3), np.cumsum(T[:, :3] * v[:, None], 0)])  # sums of t^k v
        self.V = np.concatenate([[0.0], np.cumsum(v * v)])

    def __call__(self, i, j):  # inclusive range
        s = self.S[j + 1] - self.S[i]
        A = np.array([[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]])
        b = self.B[j + 1] - self.B[i]
        try:
            coef = np.linalg.solve(A + 1e-9 * np.eye(3), b)
        except np.linalg.LinAlgError:
            return np.inf
        return max(0.0, (self.V[j + 1] - self.V[i]) - coef @ b)


def segment_flights(frames, xy, penalty: float = 400.0, min_len: int = 4, max_len: int = 90):
    """Optimal piecewise-quadratic segmentation. Returns list of (start_idx, end_idx)."""
    t = np.asarray(frames, float)
    t0 = t[0]
    t = (t - t0) / 30.0  # seconds-ish scale keeps the normal equations well conditioned
    cx, cy = _QuadCost(t, xy[:, 0]), _QuadCost(t, xy[:, 1])
    n = len(t)
    best = np.full(n + 1, np.inf)
    best[0] = 0.0
    prev = np.zeros(n + 1, int)
    for j in range(min_len, n + 1):
        for i in range(max(0, j - max_len), j - min_len + 1):
            if not np.isfinite(best[i]):
                continue
            c = best[i] + cx(i, j - 1) + cy(i, j - 1) + penalty
            if c < best[j]:
                best[j], prev[j] = c, i
    if not np.isfinite(best[n]):
        return [(0, n - 1)]
    segs, j = [], n
    while j > 0:
        i = prev[j]
        segs.append((i, j - 1))
        j = i
    return segs[::-1]


def _fit(t, v):
    return np.polyfit(t, v, 2) if len(t) >= 3 else np.polyfit(t, v, 1)


def flight_boundaries(ball: dict, max_gap: int = 12, penalty: float = 400.0):
    """Yield (frame, (x, y), seg_before, seg_after) for every flight-to-flight boundary.

    The boundary frame is where the two fitted curves come closest, searched between the
    last point of one flight and the first point of the next."""
    fr = sorted(f for f, v in ball.items() if not v[3])  # measured points only
    if len(fr) < 10:
        return []
    # split into chunks at long gaps (camera cuts, ball out of view)
    chunks, cur = [], [fr[0]]
    for a, b in zip(fr, fr[1:]):
        if b - a > max_gap:
            chunks.append(cur)
            cur = []
        cur.append(b)
    chunks.append(cur)
    out = []
    for ch in chunks:
        if len(ch) < 10:
            continue
        t = np.array(ch, float)
        xy = np.array([ball[f][:2] for f in ch], float)
        t, xy = _remove_outliers(t, xy)
        if len(t) < 10:
            continue
        segs = segment_flights(t, xy, penalty)
        fits = []
        for i, j in segs:
            ts = t[i:j + 1]
            fits.append((ts, _fit(ts, xy[i:j + 1, 0]), _fit(ts, xy[i:j + 1, 1])))
        for (ta, ax, ay), (tb, bx, by) in zip(fits, fits[1:]):
            grid = np.arange(ta[-1], tb[0] + 1)
            pa = np.stack([np.polyval(ax, grid), np.polyval(ay, grid)], 1)
            pb = np.stack([np.polyval(bx, grid), np.polyval(by, grid)], 1)
            k = int(np.argmin(np.linalg.norm(pa - pb, axis=1)))
            f = int(grid[k])
            pos = tuple(((pa[k] + pb[k]) / 2).tolist())
            v_in = (np.polyval(np.polyder(ax), f), np.polyval(np.polyder(ay), f))
            v_out = (np.polyval(np.polyder(bx), f), np.polyval(np.polyder(by), f))
            out.append((f, pos, v_in, v_out))
    return out
