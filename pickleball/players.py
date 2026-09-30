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
