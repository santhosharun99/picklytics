"""Player depth correction from the game rules.

A player's court position comes from the ankle keypoint mapped onto the ground. The ankle is
above the floor, so its ray meets the ground behind the player; from a low broadcast camera
(~12-14 ft up, ~70-80 ft from the far baseline) this pushes far-end players 3-5 ft too deep and
near-end players slightly too shallow (training/depth_check.py).

Fix: treat the foot point as sitting at an effective height h above the ground and move it along
the camera ray down to h (a shrink toward the camera's ground point by 1 - h / camera height).
One physical parameter, no tagging needed. Matches filmed by one fixed court + camera rig (TNPPL Court 1) share ONE
model (calibrate_rig, configs/cameras/rigs/<rig>.json), so a new match inherits it; other matches are calibrated per
match by a game rule (first that applies):
  1. server stance: servers stand just behind the baseline and the same players serve from both
     ends, so the median server stance must be equal at both ends (calibrate()).
  2. kitchen line (doubles): at the serve the receiving team is one up, one back - the returner's
     partner waits at the kitchen line at both ends, so their median distance from the net must be
     equal at both ends (kitchen_stance() / calibrate_kitchen()). Fewer stances per match, so they
     are pooled over the matches filmed with the same court and camera rig (one TNPPL tie) and the
     h found is used for every match of that rig whose server-stance rule does not hold.
  3. default h.
The model of a run is <run>/depth.json when a later step recalibrated it (training/build_tnppl.py),
else analysis.json's depth_correction (for_run). Used for every position-based metric (not for hit
detection, which was trained on uncorrected positions).
"""
import json
from pathlib import Path

import numpy as np

from .court import NET_Y, court_zone

DEFAULT_H = 0.3          # ankle joint height (ft) when there are too few serves to calibrate
H_RANGE = (0.1, 0.9)
MIN_SERVES = 8           # per end
KITCHEN_GAP_FT = 5.0     # receiving pair counts as one up / one back when this far apart in depth
MIN_KITCHEN = 10         # kitchen-line stances per end (pooled)
RIG_BOOT = 200           # bootstrap resamples for a rig's 90% range (its saved h changes only when it falls outside)


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


def from_net(xy) -> float:
    return abs(xy[1] - NET_Y)


def _solve(gap) -> float:
    """h in H_RANGE where gap(h) = 0; gap = near-end median - far-end median grows with h (the near end
    moves away from the net, the far end toward it)."""
    lo, hi = H_RANGE
    if gap(lo) >= 0:
        return lo
    if gap(hi) <= 0:
        return hi
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def kitchen_stance(receivers):
    """Raw (x, y) of the two receiving partners at the serve -> raw position of the one waiting at the kitchen line
    (the shallower), or None when they are not clearly one up and one back. Taking the shallower partner, not the
    partner of the credited returner, keeps the stance clean: a wrongly credited return would give the deep player."""
    if len(receivers) != 2:
        return None
    a, b = sorted(receivers, key=from_net)
    return a if from_net(b) - from_net(a) >= KITCHEN_GAP_FT else None


def calibrate_kitchen(stances) -> dict:
    """stances: [(raw (x, y), camera (x, y, height))] kitchen-line stances, pooled over matches with the same court and
    camera rig. -> {'h_ft': h or None, ...}: h where the median stance is equally far from the net at both ends."""
    near = [s for s in stances if s[0][1] >= NET_Y]
    far = [s for s in stances if s[0][1] < NET_Y]
    med = lambda ss, h: round(float(np.median([from_net(shrink(xy, cam, h)) for xy, cam in ss])), 2) if ss else None
    out = {"rule": "kitchen line (returner's partner at the serve)", "stances": {"near": len(near), "far": len(far)}}
    if len(near) < MIN_KITCHEN or len(far) < MIN_KITCHEN:
        return {**out, "h_ft": None, "why": f"too few stances (need {MIN_KITCHEN} per end)",
                "from_net_ft": {"raw": {"near": med(near, 0), "far": med(far, 0)}}}
    h = _solve(lambda h: med(near, h) - med(far, h))
    return {**out, "h_ft": round(float(h), 3),
            "from_net_ft": {"raw": {"near": med(near, 0), "far": med(far, 0)},
                            "corrected": {"near": med(near, h), "far": med(far, h)}}}


def calibrate(stances, cam_xyh, kitchen: dict | None = None) -> dict:
    """stances: raw (x, y) of servers at the serve hit. kitchen: calibrate_kitchen() result for this court / camera rig,
    used when the server-stance rule cannot be (too few serves, or implausible). -> depth model for correct()."""
    near = [s for s in stances if s[1] >= NET_Y]
    far = [s for s in stances if s[1] < NET_Y]
    gap = lambda h: (np.median([behind_baseline(shrink(s, cam_xyh, h)) for s in near])
                     - np.median([behind_baseline(shrink(s, cam_xyh, h)) for s in far]))
    raw = lambda ss: float(np.median([behind_baseline(x) for x in ss])) if ss else 0.0
    # the rule only holds if the first hitter really is a server standing behind the baseline; in doubles the
    # first detected hit can be another player (server measured 4 ft INSIDE the court) -> another rule / the default
    plausible = raw(near) > -1.0 and raw(far) > -1.0
    extra = {}
    if len(near) >= MIN_SERVES and len(far) >= MIN_SERVES and plausible:
        h, how = _solve(gap), "rules"
    elif kitchen and kitchen.get("h_ft"):
        h, how = kitchen["h_ft"], "kitchen line"
        extra = {"kitchen": kitchen}
    else:
        h, how = DEFAULT_H, "default" if plausible else "default (server stances implausible)"
    st = lambda ss, hh: round(float(np.median([behind_baseline(shrink(s, cam_xyh, hh)) for s in ss])), 2) if ss else None
    return {"camera": list(cam_xyh), "h_ft": round(float(h), 3), "method": how,
            "serves": {"near": len(near), "far": len(far)},
            "server_stance_ft": {"raw": {"near": st(near, 0), "far": st(far, 0)},
                                 "corrected": {"near": st(near, h), "far": st(far, h)}}, **extra}


def correct(xy, D):
    """Raw ground position -> corrected (x, y). D from calibrate(); None = no correction."""
    if xy is None or not D:
        return xy
    x, y = shrink(xy, D["camera"], D["h_ft"])
    return round(float(x), 2), round(float(y), 2)


def uncorrect(xy, D):
    """Corrected (x, y) -> raw ground position (inverse of correct)."""
    if xy is None or not D:
        return xy
    cx, cy, ch = D["camera"]
    k = 1 - D["h_ft"] / ch
    return cx + (xy[0] - cx) / k, cy + (xy[1] - cy) / k


# ------------------------------------------------------------------ the model of a run
def for_run(run_dir: Path, A: dict | None = None) -> dict | None:
    """Depth model of a run: <run>/depth.json (recalibrated after the analysis) or analysis.json's."""
    p = Path(run_dir) / "depth.json"
    if p.exists():
        return json.loads(p.read_text())
    if A is None:
        A = json.loads((Path(run_dir) / "analysis.json").read_text())
    return A.get("depth_correction")


def apply(A: dict, run_dir: Path) -> dict:
    """Bring an analysis.json dict to the run's depth model (for_run): positions the analysis stored already corrected
    (shot hitter positions and zones, the 10 Hz positions) are moved from its model to the run's. In place."""
    old, new = A.get("depth_correction"), for_run(run_dir, A)
    same = lambda a, b: (a or {}).get("h_ft") == (b or {}).get("h_ft") and (a or {}).get("camera") == (b or {}).get("camera")
    if same(new, old):
        if new:
            A["depth_correction"] = new
        return A
    move = lambda xy: correct(uncorrect(xy, old), new)
    for s in A.get("shots", []):
        if s.get("hitter_court"):
            x, y = move(s["hitter_court"])
            s["hitter_court"] = [round(x, 1), round(y, 1)]
            s["hitter_zone"] = court_zone(y)
    for p in A.get("positions", []):
        p["p"] = {k: [round(v, 1) for v in move(xy)] for k, xy in p["p"].items()}
    A["depth_correction"], A["depth_correction_analysis"] = new, old
    return A


def recalibrate(A: dict, kitchen: dict | None) -> dict | None:
    """Re-derive a run's depth model from analysis.json (server stances = first hitter of every rally, back to raw)
    with a kitchen-line calibration available as the fallback rule. None if the analysis had no model (no camera)."""
    old = A.get("depth_correction")
    if not old:
        return None
    first = {}
    for s in sorted(A["shots"], key=lambda s: s["frame"]):
        if not s.get("inferred"):
            first.setdefault(s["rally"], s)
    stances = [uncorrect(s["hitter_court"], old) for s in first.values() if s.get("hitter_court")]
    return calibrate(stances, old["camera"], kitchen)


def _servers_pooled(stances) -> dict:
    """stances: [(raw (x, y), camera)] server stances pooled over a rig -> {'h_ft', ...} from the server-stance rule"""
    near = [s for s in stances if s[0][1] >= NET_Y]
    far = [s for s in stances if s[0][1] < NET_Y]
    med = lambda ss, h: float(np.median([behind_baseline(shrink(xy, cam, h)) for xy, cam in ss]))
    out = {"rule": "server stance (pooled)", "stances": {"near": len(near), "far": len(far)}}
    if len(near) < MIN_SERVES or len(far) < MIN_SERVES or med(near, 0) <= -1.0 or med(far, 0) <= -1.0:
        return {**out, "h_ft": None, "why": "too few or implausible server stances"}
    h = _solve(lambda h: med(near, h) - med(far, h))
    return {**out, "h_ft": round(float(h), 3), "behind_baseline_ft": {"near": round(med(near, h), 2), "far": round(med(far, h), 2)}}


def calibrate_rig(run_dirs, rig_file: Path | None = None) -> dict:
    """ONE depth model per court + camera rig (e.g. TNPPL Court 1), not per match: the foot height h is physical and
    the camera is the same, so per-match fits only add noise (TNPPL matches gave 0.23-0.61 on one camera).
    Evidence pooled over EVERY run of the rig: kitchen-line stances (doubles with named players) first, else server
    stances, else the h already saved in rig_file (so a new match - even with no names - inherits the rig's value),
    else the default. Each run keeps its own camera position. Writes <run>/depth.json and rig_file.
    -> {'kitchen': ..., 'servers': ..., 'h_ft': h, 'method': ..., 'runs': {run name: model}}"""
    from .doubles_players import kitchen_stances
    from .replays import drop_replays
    runs = [Path(d) for d in run_dirs if (Path(d) / "analysis.json").exists()]
    saved = json.loads(Path(rig_file).read_text()) if rig_file and Path(rig_file).exists() else None
    kst, sst, models = [], [], {}
    for d in runs:
        A = json.loads((d / "analysis.json").read_text())
        old = A.get("depth_correction")
        if not old:
            continue
        models[d] = old
        if (d / "identity.json").exists() and (d / "tracklets.pkl").exists():
            kst += kitchen_stances(drop_replays(json.loads((d / "analysis.json").read_text()), d), d)
        first = {}
        for s in sorted(A["shots"], key=lambda s: s["frame"]):
            if not s.get("inferred"):
                first.setdefault(s["rally"], s)
        sst += [(uncorrect(s["hitter_court"], old), old["camera"]) for s in first.values() if s.get("hitter_court")]
    kitchen, servers = calibrate_kitchen(kst), _servers_pooled(sst)
    # both rules pooled, combined (mean of the rules that hold), with a bootstrap 90% range
    def est(k, sv):
        hs = [x for x in (calibrate_kitchen(k).get("h_ft") if k else None, _servers_pooled(sv).get("h_ft") if sv else None) if x]
        return float(np.mean(hs)) if hs else None
    estimate = est(kst, sst)
    rng, boot = np.random.default_rng(0), []
    if estimate is not None:
        for _ in range(RIG_BOOT):
            b = est([kst[i] for i in rng.integers(0, len(kst), len(kst))] if kst else [],
                    [sst[i] for i in rng.integers(0, len(sst), len(sst))] if sst else [])
            if b is not None:
                boot.append(b)
    rng90 = [round(float(np.percentile(boot, 5)), 3), round(float(np.percentile(boot, 95)), 3)] if boot else None
    rules = " + ".join(r for r, x in (("kitchen line", kitchen), ("server stance", servers)) if x.get("h_ft"))
    if estimate is not None:
        h, how = estimate, f"rig: {rules} (pooled)"
    elif saved and saved.get("h_ft"):
        h, how = saved["h_ft"], f"rig: saved ({saved.get('method', '')})"
    else:
        h, how = DEFAULT_H, "rig: default"
    estimate = None if estimate is None else round(estimate, 3)
    # stable rig value: keep the saved h while it lies inside the new estimate's 90% range, so adding a match does not
    # move every report of the rig for a change the evidence cannot tell apart from noise
    if saved and saved.get("h_ft") is not None and rng90 and rng90[0] <= saved["h_ft"] <= rng90[1]:
        h, how = saved["h_ft"], saved.get("method", how)
    out = {"kitchen": kitchen, "servers": servers, "h_ft": round(float(h), 3), "method": how, "latest_estimate": estimate, "range90": rng90, "runs": {}}
    for d, old in models.items():
        D = {"camera": old["camera"], "h_ft": out["h_ft"], "method": how, "rig": Path(rig_file).stem if rig_file else None}
        (d / "depth.json").write_text(json.dumps(D, indent=1))
        out["runs"][d.name] = D
    if rig_file:
        cams = np.array([m["camera"] for m in models.values()])
        Path(rig_file).parent.mkdir(parents=True, exist_ok=True)
        Path(rig_file).write_text(json.dumps({"h_ft": out["h_ft"], "method": how, "latest_estimate": estimate, "range90": rng90, "kitchen": kitchen, "servers": servers,
                                              "runs": sorted(out["runs"]), "camera_median": np.round(np.median(cams, 0), 3).tolist() if len(cams) else None,
                                              "camera_spread_ft": np.round(cams.max(0) - cams.min(0), 2).tolist() if len(cams) else None}, indent=1))
    return out
