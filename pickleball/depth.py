"""Player depth correction from the game rules.

A player's court position comes from the ankle keypoint mapped onto the ground. The ankle is
above the floor, so its ray meets the ground behind the player; from a low broadcast camera
(~12-14 ft up, ~70-80 ft from the far baseline) this pushes far-end players 3-5 ft too deep and
near-end players slightly too shallow (training/depth_check.py).

Fix: treat the foot point as sitting at an effective height h above the ground and move it along
the camera ray down to h (a shrink toward the camera's ground point by 1 - h / camera height).
h is calibrated per match with a rule: servers stand just behind the baseline, and the same two
players serve from both ends, so the median server stance must be equal at both ends. One
physical parameter, no tagging needed. Used for every position-based metric (not for hit
detection, which was trained on uncorrected positions).
"""
import numpy as np

from .court import NET_Y

DEFAULT_H = 0.3          # ankle joint height (ft) when there are too few serves to calibrate
H_RANGE = (0.1, 0.9)
MIN_SERVES = 8           # per end


def camera_ground(cam) -> list[float]:
    """Camera position (x, y, height) in court feet."""
    return [round(float(v), 3) for v in cam.position]


def shrink(xy, cam_xyh, h):
    cx, cy, ch = cam_xyh
    k = 1 - h / ch
    return cx + (xy[0] - cx) * k, cy + (xy[1] - cy) * k


def behind_baseline(xy) -> float:
    """Feet behind the baseline of the player's own end (negative = inside the court)."""
    return xy[1] - 44 if xy[1] >= NET_Y else -xy[1]


def calibrate(stances, cam_xyh) -> dict:
    """stances: raw (x, y) of servers at the serve hit. -> depth model for correct()."""
    near = [s for s in stances if s[1] >= NET_Y]
    far = [s for s in stances if s[1] < NET_Y]
    gap = lambda h: (np.median([behind_baseline(shrink(s, cam_xyh, h)) for s in near])
                     - np.median([behind_baseline(shrink(s, cam_xyh, h)) for s in far]))
    if len(near) >= MIN_SERVES and len(far) >= MIN_SERVES:
        lo, hi = H_RANGE   # near stance grows and far stance shrinks with h: bisect for equal stances
        if gap(lo) >= 0:
            h = lo
        elif gap(hi) <= 0:
            h = hi
        else:
            for _ in range(40):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
            h = (lo + hi) / 2
        how = "rules"
    else:
        h, how = DEFAULT_H, "default"
    st = lambda ss, hh: round(float(np.median([behind_baseline(shrink(s, cam_xyh, hh)) for s in ss])), 2) if ss else None
    return {"camera": list(cam_xyh), "h_ft": round(float(h), 3), "method": how,
            "serves": {"near": len(near), "far": len(far)},
            "server_stance_ft": {"raw": {"near": st(near, 0), "far": st(far, 0)},
                                 "corrected": {"near": st(near, h), "far": st(far, h)}}}


def correct(xy, D):
    """Raw ground position -> corrected (x, y). D from calibrate(); None = no correction."""
    if xy is None or not D:
        return xy
    x, y = shrink(xy, D["camera"], D["h_ft"])
    return round(float(x), 2), round(float(y), 2)
