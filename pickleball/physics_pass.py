"""Physics pass over a whole match: the rally3d reconstruction for every rally -> per-shot 3D facts.

Inputs (outputs/<run>/): analysis.json, detections.jsonl (camera), ball_v3.json (+ ball_v3_main.json) from
training/balltrack_run.py. Per rally (as eval_rally3d set-up D, docs/RESULTS.md): live window from the rally-end
model (serve .. last live touch + 2 s), rally3d.refine with the live hits as anchors, rally3d.end_of_play.
Output outputs/<run>/rally3d.json:
  {"meta": {...}, "rallies": {index: {"junctions": [[kind, frame, [x, y, h], [sd x, y, h]]], "g": [per flight],
                                      "rms_px", "inlier_share", "ending": in|out|net|close|none, ...}},
   "shots": {frame: {"contact": [x, y, h], "land": [x, y] | None, "land_sd_ft", "land_frame", "net_hit",
                     "speed_mph", "g_eff", "net_clear_ft", "ok" (physically plausible, _plausible)}}}
Shots are keyed by the detector's hit frame (anchors move at most +-2 frames). Kept apart from analysis.json until the
numbers are validated against tags (reports still use the lifter / speed modules).
"""
import json
from pathlib import Path

import numpy as np

from . import rally_end
from .camera3d import calibrate_from_run
from .court import NET_Y
from .rally3d import DRAG, MIN_GAP, _velocity, end_of_play, line_call, refine  # noqa: F401

MPH = 0.6818  # ft/s -> mph


def _net_height(x):
    """Net top: 36 in at the posts (x = -1, 21), 34 in at the centre."""
    return 3.0 - (2 / 12) * (1 - min(abs(x - 10), 11) / 11)


def _flight_facts(J, g, i, fps):
    """Launch speed and net clearance of the flight leaving junction i."""
    from .rally3d import _flight
    p0, p1 = J[i].xyh, J[i + 1].xyh
    T = (J[i + 1].frame - J[i].frame) / fps
    v = _velocity(p0, p1, T, g[i])
    out = {"speed_mph": round(float(np.linalg.norm(v)) * MPH, 1), "g_eff": round(float(g[i]), 1)}
    if (p0[1] - NET_Y) * (p1[1] - NET_Y) < 0:  # crosses the net plane in this flight
        ts = np.linspace(0, T, 200)
        P = _flight(p0, v, ts, g[i])
        k = int(np.argmin(np.abs(P[:, 1] - NET_Y)))
        out["net_clear_ft"] = round(float(P[k, 2] - _net_height(P[k, 0])), 2)
    return out


def _plausible(rec) -> bool:
    """Physically believable shot (else reports skip it): launch speed <= 75 mph (pro drives ~50-60), contact on or
    just around the court at 0.3-8 ft, landing fixed within 2 ft (1 sigma). TNPPL smoke test: an 85 mph 'serve'
    bouncing 8 frames after the hit, contacts 12 ft behind the baseline - tracking noise near the server."""
    x, y, h = rec["contact"]
    if not (rec["speed_mph"] <= 75 and -6 <= x <= 26 and -8 <= y <= 52 and 0.3 <= h <= 8):
        return False
    return rec.get("land") is None or (rec.get("land_sd_ft") is not None and rec["land_sd_ft"] <= 2.0)


def run_match(run_dir, ball_file="ball_v3.json", thr: float = 0.3, main_thr: float = 0.6, log=print, rallies=None,
              write: bool = True) -> dict:
    rd = Path(run_dir)
    A = json.loads((rd / "analysis.json").read_text())
    fps = A["meta"]["fps"]
    bf = rd / ball_file
    mf = bf.with_name(bf.stem + "_main.json")
    main = {int(k): v for k, v in json.loads(mf.read_text()).items()} if mf.exists() else {}
    ball = {int(k): v for k, v in json.loads(bf.read_text()).items() if v[2] >= thr and main.get(int(k), 1.0) >= main_thr}
    cam = calibrate_from_run(rd)
    players = {}
    for p in A.get("positions", []):
        f0 = int(round(p["t"] * fps))
        for d in range(-2, 3):
            players[f0 + d] = [tuple(v) for v in p["p"].values()]
    dead = rally_end.dead_touches(A, rd)
    out = {"meta": {"ball_file": ball_file, "thr": thr, "main_thr": main_thr, "drag": DRAG}, "rallies": {}, "shots": {}}
    for r in A["rallies"]:
        idx = r["index"]
        if rallies is not None and idx not in rallies:
            continue
        live = {s["frame"]: s for s in A["shots"] if s["rally"] == idx and not s.get("inferred") and (idx, s["index"]) not in dead}
        lv = sorted(live)
        if not lv:
            continue
        w0, w1 = lv[0], min(int((r["end_t"] + 1.0) * fps), lv[-1] + int(2.0 * fps))
        fr = [f for f in range(w0, w1 + 1) if f in ball]
        if len(fr) < 10:
            continue
        px = np.array([ball[f][:2] for f in fr], float)
        b = [x["frame"] for x in A.get("bounces", []) if fr[0] < x["frame"] < fr[-1]]
        init = sorted({(f, "contact") for f in lv if fr[0] < f < fr[-1]} | {(f, "bounce") for f in b})
        init = [x for i, x in enumerate(init) if i == 0 or x[0] != init[i - 1][0]]
        try:
            fit_ = refine(cam, fr, px, fps, [fr[0]] + [f for f, _ in init] + [fr[-1]],
                          ["start"] + [k for _, k in init] + ["end"], players, anchors=lv)
            hitter_far = live[lv[-1]].get("side") == "far" if live[lv[-1]].get("side") else None
            fit_ = end_of_play(cam, fr, px, fps, fit_, players, last_contact=lv[-1], hitter_far=hitter_far)
        except Exception as e:  # one bad rally must not stop the match
            log(f"rally {idx}: fit failed ({e})")
            continue
        J, g = fit_.junctions, fit_.g
        # contacts: the serve is the window's 'start' junction
        cidx = [i for i, j in enumerate(J) if j.kind == "contact" or (i == 0 and abs(j.frame - lv[0]) <= 3)]
        for n, i in enumerate(cidx):
            shot_f = min(lv, key=lambda f: abs(f - J[i].frame))
            if abs(shot_f - J[i].frame) > 3 or i >= len(J) - 1:
                continue
            nxt = cidx[n + 1] if n + 1 < len(cidx) else len(J)
            land = next((k for k in range(i + 1, nxt) if J[k].kind in ("bounce", "net")), None)
            rec = {"contact": [round(float(v), 2) for v in J[i].xyh], **_flight_facts(J, g, i, fps),
                   "land": None, "net_hit": False}
            if land is not None:
                rec["land_frame"] = int(J[land].frame)
                if J[land].kind == "net":
                    rec["net_hit"] = True
                else:
                    rec["land"] = [round(float(v), 2) for v in J[land].xyh[:2]]
                    rec["land_sd_ft"] = round(float(np.hypot(*J[land].sd[:2])), 2) if np.all(np.isfinite(J[land].sd[:2])) else None
            rec["ok"] = _plausible(rec)
            out["shots"][str(shot_f)] = rec
        # how the point ended: first bounce / net after the last contact
        ending = "none"
        if cidx:
            k = next((k for k in range(cidx[-1] + 1, len(J)) if J[k].kind in ("bounce", "net")), None)
            if k is not None:
                if J[k].kind == "net":
                    ending = "net"
                else:
                    hf = hitter_far if hitter_far is not None else J[cidx[-1]].xyh[1] < NET_Y
                    ending = "net" if (J[k].xyh[1] < NET_Y) == hf else line_call(J[k].xyh, J[k].sd)[0]
        out["rallies"][str(idx)] = {
            "junctions": [[j.kind, int(j.frame), [round(float(v), 2) for v in j.xyh],
                           [round(float(v), 2) if np.isfinite(v) else None for v in j.sd]] for j in J],
            "g": [round(float(x), 1) for x in g], "rms_px": round(fit_.rms_px, 2),
            "inlier_share": round(float(fit_.inliers.mean()), 2), "ending": ending}
        log(f"rally {idx}: {len(J)} junctions, rms {fit_.rms_px:.1f} px, ending {ending}")
    if write:
        (rd / "rally3d.json").write_text(json.dumps(out, separators=(",", ":")))
    return out
