"""Real-match test of the simulation-trained 2D->3D lifter vs the current event-pinned 3D arcs.

    PYTHONPATH=. python3 training/eval_lifter.py models/lifter.pt

Per shot (from one detected hit to the next): lift the measured ball pixels to 3D, then measure, on both
matches, with no tags needed except the analyst's bounce tags:
  - pixel error vs the ball seen in the video
  - serves landing in the receiver's service area (pros rarely miss: higher = better landing spots)
  - bounce timing vs analyst-tagged bounces
  - launch speed agreement with the independent stopwatch measurement (pickleball/speed.py)
  - net clearance when the rally continued (the ball must have cleared the net)
"""
import json, sys
from pathlib import Path

import cv2
import numpy as np

import pickleball.lifter as L
from pickleball.analysis import load_detections
from pickleball.ball import track_ball
from pickleball.ball3d import match_arcs, positions
from pickleball.camera3d import calibrate_from_run
from pickleball.court import NET_Y
from pickleball.speed import match_speeds
from pickleball.trajectory3d import reproject

ROOT = Path(__file__).resolve().parents[1]
MATCHES = [("US Open", ROOT / "outputs" / "full_match", ROOT / "labels" / "Mens_Singles_GOLD_2026_US_Open.json"),
           ("Las Vegas", ROOT / "outputs" / "lv_fahey_yoshitomi", ROOT / "labels" / "Fahey_Yoshitomi_LV2026.json")]


def serve_in(side, x, y):
    depth = (22 - y) if side == "near" else (y - 22)
    return -0.5 <= x <= 20.5 and 6.5 <= depth <= 22.5


def net_clear(X):
    s = np.sign(X[:, 1] - NET_Y)
    c = np.where(np.diff(s) != 0)[0]
    if not len(c):
        return None
    i = c[0]
    a = (NET_Y - X[i, 1]) / (X[i + 1, 1] - X[i, 1] + 1e-9)
    return X[i, 2] + a * (X[i + 1, 2] - X[i, 2]) - 2.9


def main(model_path):
    net = L.load(model_path)
    for mname, run, lab in MATCHES:
        meta, recs = load_detections(run)
        fps = meta["fps"]
        A = json.loads((run / "analysis.json").read_text())
        cam = calibrate_from_run(run)
        R, _ = cv2.Rodrigues(cam.rvec)
        K, t = cam.K, cam.tvec.ravel()
        ball = track_ball(recs)
        meas = {f: v[:2] for f, v in ball.items() if not v[3]}
        arcs = {a["f0"]: a for v in match_arcs(A, ball, cam.P).values() for a in v}
        speeds = match_speeds(A, ball, cam.P)
        tags = json.loads(lab.read_text())
        tag_b = sorted(e["frame"] for r in tags["rallies"] if r.get("end") and r["end"] >= r["start"]
                       and (r["end"] - r["start"]) / fps <= 60 and r.get("mode") != "hits"
                       for e in r["events"] if e["kind"] == "bounce")
        stats = {k: [] for k in ("err_lift", "err_arc", "serve_lift", "serve_arc", "speed_agree_lift", "speed_agree_arc",
                                 "clear_lift", "clear_arc", "bounce_err")}
        n_seg = 0
        for r in A["rallies"]:
            shots = sorted([s for s in A["shots"] if s["rally"] == r["index"] and not s.get("inferred")], key=lambda s: s["frame"])
            for k, (s0, s1) in enumerate(zip(shots, shots[1:])):
                f0, f1 = s0["frame"], s1["frame"]
                fr = [f for f in range(f0, f1 + 1) if f in meas]
                if len(fr) < 4 or f1 - f0 + 1 > L.MAX_T:
                    continue
                uv = np.array([meas[f] for f in fr], float)
                X = L.lift_track(K, R, t, fr, uv, f0, f1)
                if X is None:
                    continue
                n_seg += 1
                pr = reproject(cam.P, X[np.array(fr) - f0])
                stats["err_lift"].append(float(np.median(np.linalg.norm(pr - uv, axis=1))))
                a = arcs.get(f0)
                if a and a.get("confident"):
                    stats["err_arc"].append(a["err_px"])
                # bounce = lowest point of the lifted path if it reaches the ground
                ib = int(np.argmin(X[:, 2]))
                bounced = X[ib, 2] < 1.0 and 0 < ib < len(X) - 1
                if bounced:
                    fb = f0 + ib
                    near_tag = [abs(fb - b) for b in tag_b if f0 <= b <= f1]
                    if near_tag:
                        stats["bounce_err"].append(min(near_tag))
                if s0.get("shot_type") == "serve" and bounced:
                    stats["serve_lift"].append(serve_in(s0["side"], X[ib, 0], X[ib, 1]))
                if s0.get("shot_type") == "serve" and s0.get("landing_court"):
                    stats["serve_arc"].append(serve_in(s0["side"], *s0["landing_court"]))
                # launch speed from the first frames vs the stopwatch (independent)
                v_l = np.linalg.norm(X[3] - X[0]) * fps / 3 * 0.6818
                sw = speeds.get(f0)
                if sw:
                    T = sw["flight_s"]
                    stop = sw["avg_mph"] * (0.6 * T) / (1 - np.exp(-0.6 * T))
                    stats["speed_agree_lift"].append(abs(v_l - stop) <= 0.2 * stop)
                    if sw.get("arc_mph"):
                        stats["speed_agree_arc"].append(abs(sw["arc_mph"] - stop) <= 0.2 * stop)
                c = net_clear(X)
                if c is not None:
                    stats["clear_lift"].append(c >= -0.3)
                if a and a.get("confident") and a.get("net_clear_ft") is not None:
                    stats["clear_arc"].append(a["net_clear_ft"] >= -0.3)
        pct = lambda v: f"{100 * np.mean(v):.0f}% (n={len(v)})" if v else "-"
        med = lambda v: f"{np.median(v):.1f} (n={len(v)})" if v else "-"
        print(f"\n== {mname}: {n_seg} hit-to-hit shots lifted")
        print(f"  pixel error, median        lifter {med(stats['err_lift'])}   current arcs (trusted only) {med(stats['err_arc'])}")
        print(f"  serves landing in the box   lifter {pct(stats['serve_lift'])}   current {pct(stats['serve_arc'])}")
        print(f"  speed agrees w/ stopwatch   lifter {pct(stats['speed_agree_lift'])}   current arcs {pct(stats['speed_agree_arc'])}")
        print(f"  clears the net (rally on)   lifter {pct(stats['clear_lift'])}   current arcs {pct(stats['clear_arc'])}")
        be = stats["bounce_err"]
        print(f"  lifter bounce vs tagged     within 4 frames {pct([e <= 4 for e in be])}, median {med(be)} frames")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "models" / "lifter.pt"))
