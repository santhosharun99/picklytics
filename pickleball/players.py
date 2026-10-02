"""Pick the on-court players each frame and give them stable identities.

Selection: per court half, keep the `per_side` people closest to the court's centre
line (line judges and ball kids stand at the edges, players are inside).
Identity: players swap ends during a match, so they are named by shirt colour, not
by side. Prototypes are learned from the first confident main-view frames and
updated slowly.
"""
from itertools import permutations

import numpy as np

from .court import COURT_W, NET_Y


def select_players(people: list[dict], per_side: int = 1) -> list[dict]:
    chosen = []
    for far in (True, False):
        side = [p for p in people if (p["court"][1] < NET_Y) == far]
        # score: distance from the centre line across the court, plus how far outside the sidelines
        def score(p):
            x = p["court"][0]
            outside = max(0.0, -x, x - COURT_W)
            return abs(x - COURT_W / 2) + 3.0 * outside - 2.0 * p["conf"]
        for p in sorted(side, key=score)[:per_side]:
            chosen.append({**p, "side": "far" if far else "near"})
    return chosen


def _sim(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    # Bhattacharyya coefficient on the hue part, plus value part
    return float(np.sqrt(a[:16] * b[:16]).sum() + 0.5 * np.sqrt(a[16:] * b[16:]).sum())


class PlayerIdentifier:
    """Assigns ids P1..Pn by shirt colour signature."""

    def __init__(self, n_players: int, lr: float = 0.02):
        self.n = n_players
        self.lr = lr
        self.protos: list[np.ndarray] = []

    def assign(self, players: list[dict]) -> list[dict]:
        if not players:
            return players
        sigs = [np.asarray(p["sig"]) for p in players]
        if len(self.protos) < self.n:
            # bootstrap: first frame with the full set of players defines the prototypes,
            # ordered far side first so P1 starts at the far end
            if len(players) == self.n:
                self.protos = [s.copy() for s in sigs]
            else:
                return [{**p, "id": None} for p in players]

        best, best_perm = -1.0, None
        for perm in permutations(range(self.n), len(players)):
            s = sum(_sim(sigs[i], self.protos[j]) for i, j in enumerate(perm))
            if s > best:
                best, best_perm = s, perm
        out = []
        for p, j, s in zip(players, best_perm, sigs):
            self.protos[j] = (1 - self.lr) * self.protos[j] + self.lr * s
            out.append({**p, "id": f"P{j + 1}"})
        return out


def smooth_positions(track: list[tuple[int, float, float]], window: int = 5):
    """Moving-average smoothing of a player's (frame, x, y) court track."""
    if len(track) < window:
        return track
    arr = np.array(track, dtype=float)
    k = np.ones(window) / window
    xs = np.convolve(arr[:, 1], k, mode="same")
    ys = np.convolve(arr[:, 2], k, mode="same")
    half = window // 2
    xs[:half], xs[-half:] = arr[:half, 1], arr[-half:, 1]
    ys[:half], ys[-half:] = arr[:half, 2], arr[-half:, 2]
    return [(int(f), float(x), float(y)) for f, x, y in zip(arr[:, 0], xs, ys)]


def identify_players(frames: dict[int, list[dict]], n_players: int = 2, seg_gap: int = 15) -> dict[int, list[dict]]:
    """Global player identity for singles: {frame: selected players} -> same with "id" set.

    The online PlayerIdentifier adapts its shirt prototypes every frame and can drift onto the
    wrong player (seen when players switch ends). Here: (1) two shirt-colour clusters from the
    whole video (k-means on Hellinger-mapped signatures), (2) one side->player mapping per
    continuous camera segment, decided by majority vote over all its frames. Players only change
    sides between points, and every point sits inside one segment, so a single bad frame can't
    flip identities. P1 = the far-side player of the first segment (old convention).
    """
    if n_players != 2:
        ident = PlayerIdentifier(n_players)
        return {f: ident.assign(ps) for f, ps in sorted(frames.items())}
    rows = [(f, p["side"], np.sqrt(np.asarray(p["sig"], float))) for f, ps in frames.items() for p in ps if p.get("sig") is not None]
    if len(rows) < 20:
        ident = PlayerIdentifier(n_players)
        return {f: ident.assign(ps) for f, ps in sorted(frames.items())}
    X = np.stack([r[2] for r in rows])
    # k-means (k=2), initialised with the mean far / near signature of the first frames
    first = sorted({r[0] for r in rows})[: max(50, len(rows) // 50)]
    init = [X[[i for i, r in enumerate(rows) if r[0] in first and r[1] == s]].mean(0) for s in ("far", "near")]
    C = np.stack([c if np.isfinite(c).all() else X[k] for k, c in enumerate(init)])
    for _ in range(30):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(2), 1)
        newC = np.stack([X[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(2)])
        if np.allclose(newC, C):
            break
        C = newC
    # segments of consecutive main-view frames
    fs = sorted(frames)
    seg_of, seg = {}, 0
    for a, b in zip([None] + fs, fs):
        if a is not None and b - a > seg_gap:
            seg += 1
        seg_of[b] = seg
    # vote per segment: score of mapping "far = cluster 0" vs "far = cluster 1"
    vote = {}
    for f, side, x in rows:
        d0, d1 = ((x - C[0]) ** 2).sum(), ((x - C[1]) ** 2).sum()
        v = (d1 - d0) if side == "far" else (d0 - d1)  # > 0 supports far = cluster 0
        vote[seg_of[f]] = vote.get(seg_of[f], 0.0) + v
    out = {}
    for f, ps in frames.items():
        far_is_0 = vote.get(seg_of[f], 0.0) >= 0
        out[f] = [{**p, "id": ("P1" if (p["side"] == "far") == far_is_0 else "P2")} for p in ps]
    return out
