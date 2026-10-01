"""Shot classification from rally events.

Each hit is described by: its place in the rally, whether it was a volley (no bounce
since the previous hit), where the hitter stood, where the ball landed, how fast it
travelled (court distance / flight time), and the hitter's pose (overhead, side).

Thresholds live in SHOT_RULES so they can be tuned against labelled clips.
"""
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from .court import COURT_L, COURT_W, NET_Y, court_zone
from .events import Event, Rally

FT_PER_S_TO_MPH = 0.6818

SHOT_RULES = {
    "kitchen_depth": 7.0,        # ft from net: lands in the kitchen
    "soft_mph": 22.0,            # at or below: soft shot (dink / drop)
    "drive_mph": 27.0,           # at or above: drive (tuned on labelled rallies: drives 28-69 mph)
    "drive_flight_s": 0.65,      # or: from the back and this quick to its next event
    "reset_depth": 15.0,         # soft shot hit from within this many ft of the net (transition) = reset
    "lob_flight_s": 1.3,         # long hang time...
    "lob_depth": 12.0,           # ...landing deep
    "overhead_mph": 25.0,
    "near_net": 10.0,            # hitter within this many ft of the net: "at the net"
}


@dataclass
class Shot:
    rally: int
    index: int                   # 0 = serve
    frame: int
    t: float
    player: str | None
    side: str | None
    shot_type: str
    stroke: str | None           # forehand / backhand
    volley: bool
    speed_mph: float | None
    flight_s: float | None
    hitter_court: tuple | None
    hitter_zone: str | None
    landing_court: tuple | None
    landing_zone: str | None
    out: bool | None


def infer_handedness(rallies: list[Rally]) -> dict[str, str]:
    """Dominant hand per player = the wrist most often closest to the ball at contact."""
    votes: dict[str, Counter] = {}
    for r in rallies:
        for h in r.hits:
            if h.player and h.wrist:
                votes.setdefault(h.player, Counter())[h.wrist] += 1
    return {p: c.most_common(1)[0][0] for p, c in votes.items()}


def _stroke(hit: Event, hand: str) -> str | None:
    if hit.body_x is None:
        return None
    ball_right_in_image = hit.img[0] > hit.body_x
    # near player faces away from the camera (their right = image right); far player faces it
    ball_on_players_right = ball_right_in_image if hit.side == "near" else not ball_right_in_image
    return "forehand" if ball_on_players_right == (hand == "right") else "backhand"


def classify_rally(rally: Rally, handedness: dict[str, str], rules=SHOT_RULES) -> list[Shot]:
    shots = []
    evs = rally.events
    hit_idx = [i for i, e in enumerate(evs) if e.kind == "hit"]
    for n, i in enumerate(hit_idx):
        h = evs[i]
        prev_i = hit_idx[n - 1] if n > 0 else None
        next_i = hit_idx[n + 1] if n + 1 < len(hit_idx) else None

        # two-bounce rule: the return and the third shot always bounce first
        volley = n > 2 and not any(evs[j].kind == "bounce" for j in range(prev_i + 1, i))
        end = next_i if next_i is not None else len(evs)
        bounce = next((evs[j] for j in range(i + 1, end) if evs[j].kind == "bounce"), None)

        target = bounce or (evs[next_i] if next_i is not None else None)
        flight = (target.t - h.t) if target else None
        speed = None
        if target and h.court and target.court and flight and flight > 0:
            dist = float(np.hypot(target.court[0] - h.court[0], target.court[1] - h.court[1]))
            speed = dist / flight * FT_PER_S_TO_MPH

        d_hit = abs(h.court[1] - NET_Y) if h.court else None
        land = bounce.court if bounce else None
        d_land = abs(land[1] - NET_Y) if land else None
        out = None
        if land:
            out = not (0 <= land[0] <= COURT_W and 0 <= land[1] <= COURT_L)

        stype = _shot_type(n, volley, h, speed, flight, d_hit, d_land, rules)
        hand = handedness.get(h.player or "", "right")
        shots.append(Shot(
            rally=rally.index, index=n, frame=h.frame, t=round(h.t, 2), player=h.player, side=h.side,
            shot_type=stype, stroke=_stroke(h, hand), volley=volley,
            speed_mph=round(speed, 1) if speed else None, flight_s=round(flight, 2) if flight else None,
            hitter_court=tuple(round(v, 1) for v in h.court) if h.court else None,
            hitter_zone=court_zone(h.court[1]) if h.court else None,
            landing_court=tuple(round(v, 1) for v in land) if land else None,
            landing_zone=court_zone(land[1]) if land else None, out=out,
        ))
    return shots


def _shot_type(n, volley, h, speed, flight, d_hit, d_land, r) -> str:
    if n == 0:
        return "serve"
    if n == 1:
        return "return"
    third = "third-shot " if n == 2 else ""
    if h.above_head and (speed is None or speed >= r["overhead_mph"]):
        return "overhead"
    if flight and d_land is not None and flight >= r["lob_flight_s"] and d_land >= r["lob_depth"]:
        return "lob"
    soft = speed is not None and speed <= r["soft_mph"]
    if d_land is not None and d_land <= r["kitchen_depth"] and (soft or speed is None):
        if d_hit is not None and d_hit <= r["near_net"] and not third:
            return "dink"
        return f"{third}drop"
    if speed is not None and speed >= r["drive_mph"]:
        return f"{third}drive"
    if flight is not None and flight <= r["drive_flight_s"] and d_hit is not None and d_hit > r["near_net"]:
        return f"{third}drive"
    if volley and d_hit is not None and d_hit <= r["near_net"]:
        return "volley"
    if d_hit is not None and d_hit <= r["reset_depth"]:
        return "reset"
    return f"{third}groundstroke" if third else "groundstroke"


def shots_to_dicts(shots: list[Shot]) -> list[dict]:
    return [asdict(s) for s in shots]
