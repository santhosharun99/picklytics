"""Court zones for placement analysis - the ONE place for orientation, bands and boundary rules (docs/PLACEMENT.md).

Court coordinates (pickleball/court.py): x across the 20 ft width, 0 = left sideline as seen from the NEAR baseline;
y along the 44 ft length, 0 = far baseline, 44 = near baseline, net at y = 22.

Shot frame: every location of a shot (origin, bounce, interception, projected landing) is described from the HITTER's
viewpoint, looking over the net toward the opponent:
  - lateral u (ft): 0 = the hitter's left sideline, 20 = right sideline. Near-side hitter (y > 22) faces the far end:
    u = x. Far-side hitter faces the near end: u = 20 - x. Ends change between games, so the side is decided per
    shot from the hitter's own position (never per match).
  - depth d (ft from the net) on each half: 0 at the net, 22 at that half's baseline; > 22 behind it.
Bands (half-open, lower bound included, so a position exactly on 7 ft is midcourt and exactly on 20/3 is centre):
  depth  kitchen [0, 7)   midcourt [7, 14.5)   deep [14.5, 22]   behind > 22 (outside, kept as such)
  width  left [0, 20/3)   centre [20/3, 40/3)  right [40/3, 20]  wide-left < 0 / wide-right > 20 (outside)
  (the baseline 22 and the sideline 20 belong to the court: lines are in.)
Nothing outside the court is clamped into an inside zone: an origin behind the baseline is 'behind ...', an
estimated landing past a line is 'out ...' (an estimate, not a line call).
Uncertainty: with a 1-sigma position sd, every zone within +-K_SIGMA*sd is a candidate; more than one candidate ->
the zone is reported as 'boundary' with the candidates listed (never a forced precise zone).
"""
from dataclasses import dataclass

COURT_W, COURT_L, NET_Y = 20.0, 44.0, 22.0
DEPTH_BANDS = (("kitchen", 0.0, 7.0), ("midcourt", 7.0, 14.5), ("deep", 14.5, 22.0))
WIDTH_EDGES = (0.0, 20.0 / 3, 40.0 / 3, 20.0)
WIDTH_NAMES = ("left", "centre", "right")
K_SIGMA = 1.0  # zone candidates within +-1 sigma (documented; no calibrated error bars are claimed)


def hitter_is_near(hitter_y: float) -> bool:
    """Which end the hitter is at, from their own position (court y)."""
    return hitter_y > NET_Y


def to_shot_frame(x: float, y: float, hitter_near: bool):
    """(u, d, own_half): lateral from the hitter's left, depth from the net, and whether the point is on the
    hitter's own half."""
    u = x if hitter_near else COURT_W - x
    own = (y > NET_Y) == hitter_near if y != NET_Y else False
    return u, abs(y - NET_Y), own


def _depth_band(d: float) -> str:
    for name, lo, hi in DEPTH_BANDS:
        if lo <= d < hi or (name == "deep" and d == hi):
            return name
    return "behind"


def _width_band(u: float) -> str:
    if u < 0:
        return "wide-left"
    if u > COURT_W:
        return "wide-right"
    for i, name in enumerate(WIDTH_NAMES):
        if WIDTH_EDGES[i] <= u < WIDTH_EDGES[i + 1] or (i == 2 and u == COURT_W):
            return name
    return "right"


def zone_name(u: float, d: float) -> str:
    """'kitchen left' ... 'deep right', or an outside name: 'behind centre', 'deep wide-left', 'behind wide-right'."""
    return f"{_depth_band(d)} {_width_band(u)}"


def is_inside(u: float, d: float) -> bool:
    return 0 <= u <= COURT_W and 0 <= d <= COURT_L / 2


@dataclass
class Zone:
    name: str            # zone, or 'boundary' when the uncertainty spans several zones
    half: str            # 'own' (hitter's side) | 'opponent'
    inside: bool
    candidates: list     # zones within +-K_SIGMA*sd (one entry when certain)
    u: float
    d: float

    def to_json(self) -> dict:
        return {"zone": self.name, "half": self.half, "inside": self.inside, "candidates": self.candidates,
                "u": round(self.u, 2), "d": round(self.d, 2)}


def locate(x: float, y: float, hitter_near: bool, sd: float | None = None) -> Zone:
    """Zone of a court point (x, y) for a shot whose hitter stands at the near end (hitter_near) or the far end -
    taken from the tracked side of the hitter, not from a position near the net. sd: 1-sigma position
    uncertainty (ft) if known; None = no estimate (one zone; the caller's evidence type says how good it is)."""
    near = hitter_near
    u, d, own = to_shot_frame(x, y, near)
    z = zone_name(u, d)
    cands = [z]
    if sd:
        k = K_SIGMA * sd
        cands = sorted({zone_name(u + du, max(0.0, d + dd)) for du in (-k, 0, k) for dd in (-k, 0, k)},
                       key=lambda c: (c != z, c))
    return Zone("boundary" if len(cands) > 1 else z, "own" if own else "opponent", is_inside(u, d), cands, u, d)


ALL_ZONES = [f"{d} {w}" for d, _, _ in DEPTH_BANDS for w in WIDTH_NAMES]
