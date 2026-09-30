"""Build a clean ball trajectory from per-frame detection candidates.

- Picks at most one candidate per frame: the one closest to where the ball should be
  (constant-velocity prediction), else a confident fresh detection.
- Rejects static false positives (e.g. a ball lying by the fence, logos).
- Fills only short gaps (never across rallies or camera cuts).
"""
import numpy as np


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def track_ball(frames: list[dict], max_jump: float = 120.0, fresh_conf: float = 0.35,
               max_gap: int = 6, static_frames: int = 20, static_px: float = 3.0):
    """Return {frame: (x, y, conf, interpolated)} for main-view frames."""
    chosen: dict[int, tuple[float, float, float]] = {}
    prev, prev_f, vel = None, None, (0.0, 0.0)
    for rec in frames:
        f = rec["f"]
        if rec.get("court") is None or rec.get("cut"):
            prev, vel = None, (0.0, 0.0)
            continue
        cands = [(*_center(b), b[4]) for b in rec.get("balls", [])]
        if not cands:
            continue
        pick = None
        if prev is not None and f - prev_f <= max_gap:
            dt = f - prev_f
            px, py = prev[0] + vel[0] * dt, prev[1] + vel[1] * dt
            d = [np.hypot(c[0] - px, c[1] - py) for c in cands]
            i = int(np.argmin(d))
            if d[i] < max_jump * dt ** 0.5:
                pick = cands[i]
        if pick is None:
            best = max(cands, key=lambda c: c[2])
            if best[2] >= fresh_conf:
                pick = best
        if pick is None:
            continue
        if prev is not None and f - prev_f <= max_gap:
            dt = f - prev_f
            vel = ((pick[0] - prev[0]) / dt, (pick[1] - prev[1]) / dt)
        else:
            vel = (0.0, 0.0)
        chosen[f] = pick
        prev, prev_f = pick, f

    _drop_static(chosen, static_frames, static_px)
    return _fill_gaps(chosen, max_gap)


def _drop_static(chosen, n, px):
    fs = sorted(chosen)
    run = [fs[0]] if fs else []
    for a, b in zip(fs, fs[1:]):
        if b - a <= 2 and np.hypot(chosen[b][0] - chosen[a][0], chosen[b][1] - chosen[a][1]) < px:
            run.append(b)
            continue
        if len(run) >= n:
            for f in run:
                chosen.pop(f, None)
        run = [b]
    if len(run) >= n:
        for f in run:
            chosen.pop(f, None)


def _fill_gaps(chosen, max_gap):
    out = {f: (x, y, c, False) for f, (x, y, c) in chosen.items()}
    fs = sorted(chosen)
    for a, b in zip(fs, fs[1:]):
        if 1 < b - a <= max_gap:
            for f in range(a + 1, b):
                t = (f - a) / (b - a)
                out[f] = (chosen[a][0] + t * (chosen[b][0] - chosen[a][0]),
                          chosen[a][1] + t * (chosen[b][1] - chosen[a][1]), 0.0, True)
    return dict(sorted(out.items()))
