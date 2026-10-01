"""Build a clean ball trajectory from per-frame detection candidates.

- Picks at most one candidate per frame: the one closest to where the ball should be
  (constant-velocity prediction), else a confident fresh detection.
- Rejects static false positives (e.g. a ball lying by the fence, logos).
- Fills only short gaps (never across rallies or camera cuts).
"""
import numpy as np


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def static_cells(frames: list[dict], cell: int = 12, min_share: float = 0.05) -> set:
    """Grid cells where 'balls' are detected in many frames: logos, wall graphics, a spare
    ball by the fence. A real ball never stays put."""
    from collections import Counter
    counts, n = Counter(), 0
    for rec in frames:
        if rec.get("court") is None:
            continue
        n += 1
        seen = set()
        for b in rec.get("balls", []):
            cx, cy = _center(b)
            seen.add((int(cx // cell), int(cy // cell)))
        counts.update(seen)
    limit = max(15, min_share * n)
    hot = {c for c, k in counts.items() if k >= limit}
    return {(x + dx, y + dy) for x, y in hot for dx in (-1, 0, 1) for dy in (-1, 0, 1)}


def track_ball_global(frames: list[dict], min_conf: float = 0.12, max_skip: int = 6, free_speed: float = 45.0,
               restart_cost: float = 0.6, max_gap: int = 6, cell: int = 12):
    """Global best ball path (segmental Viterbi) over all candidates.

    Each candidate earns (conf - min_conf). Linking two candidates k frames apart costs
    nothing up to `free_speed` px/frame (so sharp reversals at contacts are fine), then
    grows with speed; skipping frames costs a little; starting a new track costs
    `restart_cost`. Unlike greedy gating, the path can use the future to pick the right
    candidate when the ball turns around at a player.
    Returns {frame: (x, y, conf, interpolated)}.
    """
    static = static_cells(frames, cell)
    seq = []  # (frame, [(x, y, conf)])
    for rec in frames:
        if rec.get("court") is None:
            continue
        cands = [(*_center(b), b[4]) for b in rec.get("balls", []) if b[4] >= min_conf]
        cands = [c for c in cands if (int(c[0] // cell), int(c[1] // cell)) not in static]
        seq.append((rec["f"], cands, bool(rec.get("cut"))))

    best, back = [], []  # per seq index: list of scores / back-pointers per candidate
    run_best, run_arg = [], []  # best score of any node up to index i (for restarts)
    for i, (f, cands, cut) in enumerate(seq):
        scores, ptrs = [], []
        g_prev = run_best[i - 1] if i else 0.0
        for c in cands:
            node = c[2] - min_conf
            top, ptr = g_prev - restart_cost, (run_arg[i - 1] if i else None, "restart")
            for k in range(1, max_skip + 1):
                j = i - k
                if j < 0 or seq[i][0] - seq[j][0] > max_skip:
                    break
                if any(seq[m][2] for m in range(j + 1, i + 1)):  # never link across a camera cut
                    break
                fj = seq[j][0]
                for bi, b in enumerate(seq[j][1]):
                    dt = f - fj
                    speed = np.hypot(c[0] - b[0], c[1] - b[1]) / dt
                    cost = max(0.0, speed - free_speed) * 0.05 + 0.08 * (dt - 1)
                    v = best[j][bi] - cost
                    if v > top:
                        top, ptr = v, (j, bi)
            scores.append(node + top)
            ptrs.append(ptr)
        best.append(scores)
        back.append(ptrs)
        cand_best = max(scores) if scores else -np.inf
        if i and run_best[i - 1] >= cand_best:
            run_best.append(run_best[i - 1]); run_arg.append(run_arg[i - 1])
        else:
            run_best.append(max(cand_best, 0.0) if scores else 0.0)
            run_arg.append((i, int(np.argmax(scores))) if scores else None)

    chosen: dict[int, tuple[float, float, float]] = {}
    node = run_arg[-1] if run_arg else None
    while node is not None:
        if isinstance(node, tuple) and len(node) == 2 and node[1] == "restart":
            node = node[0]
            continue
        i, ci = node
        f, cands, _ = seq[i]
        chosen[f] = cands[ci]
        node = back[i][ci]
    _drop_static(chosen, 20, 3.0)
    return _fill_gaps(chosen, max_gap)


def track_ball(frames: list[dict], max_jump: float = 120.0, fresh_conf: float = 0.3,
               max_gap: int = 6, static_frames: int = 20, static_px: float = 3.0, cell: int = 12):
    """Return {frame: (x, y, conf, interpolated)} for main-view frames."""
    static = static_cells(frames, cell)
    chosen: dict[int, tuple[float, float, float]] = {}
    prev, prev_f, vel = None, None, (0.0, 0.0)
    for rec in frames:
        f = rec["f"]
        if rec.get("court") is None or rec.get("cut"):
            prev, vel = None, (0.0, 0.0)
            continue
        cands = [(*_center(b), b[4]) for b in rec.get("balls", [])]
        cands = [c for c in cands if (int(c[0] // cell), int(c[1] // cell)) not in static]
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
