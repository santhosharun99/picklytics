"""Shot speed (ball speed off the paddle), measured two independent ways and cross-checked.

1. Stopwatch: flight time from the hit frame to the next contact (bounce, or the opponent's hit for a
   volley) and the 3D distance between the contact point (camera ray at the hitter's depth) and the
   landing spot -> average speed. A pickleball slows quickly (air drag), so the speed off the paddle
   is higher than the average: v0 = v_avg * bT / (1 - exp(-bT)) with drag b ~ 0.6 /s (linear drag).
   Uncertainty: +-1 frame at each end and +-1.5 ft at each end, propagated.
2. 3D arc: initial speed of the fitted flight (pickleball/ball3d.py), when that arc is trusted.

Reported speed = the mean of both when they agree within 20% ("high"), else the stopwatch estimate
("medium" if its error is <= 25%, else "low"). Serves and returns must bounce first, so a flight that
ends at a hit (bounce not seen) is not measured.
"""
import numpy as np

from .ball3d import positions, rally_anchors, rally_arcs

DRAG_B = 0.6        # 1/s, typical for a perforated pickleball at rally speeds
POS_ERR_FT = 1.5
FT_S_TO_MPH = 0.6818
AGREE = 0.20        # the two measurements "agree" within 20%


def _launch_from_avg(v_avg, T, b=DRAG_B):
    return v_avg * (b * T) / (1 - np.exp(-b * T)) if T > 0 else v_avg


def shot_speeds(shots: list[dict], bounces: list[dict], ball: dict, P, fps: float) -> dict:
    """{shot frame: {"mph", "avg_mph", "err_mph", "arc_mph", "flight_s", "dist_ft", "conf"}} for one rally."""
    shots = sorted(shots, key=lambda s: s["frame"])
    if not shots:
        return {}
    f0, f1 = shots[0]["frame"], shots[-1]["frame"]
    anchors = rally_anchors(shots, bounces, ball, P, f0, f1)
    arcs = {a["f0"]: a for a in rally_arcs(anchors, ball, P, fps)}
    pos_at = {f: X for f, X, _ in anchors}
    out = {}
    for k, s in enumerate(shots):
        if s.get("inferred") or s["frame"] not in pos_at:
            continue
        # next contact after this hit: first anchor after it (bounce or the next hit)
        later = [(f, X, kind) for f, X, kind in anchors if f > s["frame"]]
        if not later:
            continue
        f_next, X_next, kind_next = later[0]
        if kind_next == "hit" and k < 2:
            continue  # serve and return must bounce first: the bounce was missed, this spans two flights
        T = (f_next - s["frame"]) / fps
        if not (0.12 <= T <= 2.5):
            continue
        X0 = pos_at[s["frame"]]
        dist = float(np.linalg.norm(np.asarray(X_next) - X0))
        v_avg = dist / T
        v0 = _launch_from_avg(v_avg, T)
        dT = 2.0 / fps
        err = np.hypot(v0 * dT / T, np.sqrt(2) * POS_ERR_FT / T)
        a = arcs.get(s["frame"])
        arc_v = a["speed_mph"] / FT_S_TO_MPH if a and a.get("confident") else None
        if arc_v is not None and abs(arc_v - v0) <= AGREE * v0:
            best, conf = (arc_v + v0) / 2, "high"  # two independent measurements agree: use their mean
        else:
            best, conf = v0, ("medium" if err <= 0.25 * v0 else "low")
        out[s["frame"]] = {"mph": round(best * FT_S_TO_MPH, 1), "avg_mph": round(v_avg * FT_S_TO_MPH, 1),
                           "err_mph": round(err * FT_S_TO_MPH, 1),
                           "arc_mph": round(arc_v * FT_S_TO_MPH, 1) if arc_v else None,
                           "flight_s": round(T, 3), "dist_ft": round(dist, 1), "to": kind_next, "conf": conf}
    return out


def match_speeds(A: dict, ball: dict, P) -> dict:
    """{shot frame: speed dict} for every shot of an analysis.json dict."""
    fps = A["meta"]["fps"]
    out = {}
    for r in A["rallies"]:
        shots = [s for s in A["shots"] if s["rally"] == r["index"]]
        out.update(shot_speeds(shots, A["bounces"], ball, P, fps))
    return out
