"""Physics-verified training labels for the temporal ball detector (docs/BALL_RESEARCH_PLAN.md, section 2).

    PYTHONPATH=. python3 training/balltrack_pseudolabels.py [run ...]  -> data/balltrack_labels/<run>.csv
A detected ball position becomes a label only if it lies on a physically valid 3D flight: for every stretch between
two contacts, a ballistic flight (gravity + drag, seen through the calibrated camera) is fitted with RANSAC
(after_contact.fit_flight_ransac); positions within 3 px of a flight that explains >= 8 frames with < 2.5 px rms are
kept. A second flight is searched in what is left (the ball after a bounce). Labels never come from the old tracker
alone (the 2026-10-03 TrackNet learned the tracker's mistakes that way).
Excluded: held-out matches (tnppl_6_MD, lv_waters_johns, europe_feed_1) and the ground-truth clips' rallies.
CSV: frame, rally, x, y (source pixels), kind (pos), err_px
"""
import csv, json, sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from pickleball.after_contact import _contact_point, _moving, _pos, fit_flight_ransac
from pickleball.camera3d import calibrate_from_run

ROOT = Path(__file__).resolve().parents[1]
TRAIN_RUNS = ["tnppl_1_MS", "tnppl_2_WD", "tnppl_3_MD", "tnppl_4_XD", "tnppl_5_MD50", "tnppl_7_TeamDoubles",
              "tnppl_8_TeamSingles", "lv_bright_patriquin", "lv_fahey_yoshitomi", "full_match"]
MAX_ERR, MIN_INL, INL_PX = 2.5, 8, 3.0


def free_fit(cam, ts, px):
    """Flight with a free start (the ball after a bounce): RANSAC over start + velocity."""
    best = None
    rng = np.random.default_rng(1)
    for k in range(30):
        idx = np.sort(rng.choice(len(ts), 4, replace=False))

        def res(p, idx=idx):
            return (cam.project(_pos(p, ts[idx])) - px[idx]).ravel()
        p0 = [10.0, 22.0 + (-10 if k % 2 else 10), 1.0, 0.0, (-30.0, 30.0)[k % 2], 10.0]
        r = least_squares(res, p0, bounds=([-10, -20, 0.0, -50, -120, -60], [30, 64, 12, 50, 120, 60]))
        d = np.linalg.norm(cam.project(_pos(r.x, ts)) - px, axis=1)
        inl = d < INL_PX * 2
        if best is None or inl.sum() > best[0]:
            best = (inl.sum(), r.x, inl)
    if best is None or best[0] < MIN_INL:
        return None
    inl = best[2]
    r = least_squares(lambda p: (cam.project(_pos(p, ts[inl])) - px[inl]).ravel(), best[1],
                      bounds=([-10, -20, 0.0, -50, -120, -60], [30, 64, 12, 50, 120, 60]))
    return r.x, float(np.sqrt(np.mean(r.fun ** 2))), inl


def label_run(rid):
    run = ROOT / "outputs" / rid
    A = json.loads((run / "analysis.json").read_text())
    fps = A["meta"]["fps"]
    ball = {int(k): v for k, v in json.loads((run / "render_data.json").read_text())["ball"].items()}
    cam = calibrate_from_run(run)
    q = json.loads((ROOT / "labels" / "balltruth" / "queue.json").read_text())["clips"]
    test_rallies = {c["rally"] for c in q if c["run"] == rid}
    rows, segs, good = [], 0, 0
    by = {}
    for s in A["shots"]:
        if not s.get("inferred") and s["rally"] not in test_rallies:
            by.setdefault(s["rally"], []).append(s)
    for rally, sh in by.items():
        sh.sort(key=lambda s: s["frame"])
        for k, s in enumerate(sh):
            f0 = s["frame"]
            f1 = sh[k + 1]["frame"] - 1 if k + 1 < len(sh) else f0 + int(1.5 * fps)
            fr = [f for f in range(f0 + 1, f1 + 1) if f in ball and not ball[f][2]]
            if len(fr) < MIN_INL or f0 not in ball:
                continue
            xy = np.array([ball[f][:2] for f in fr], float)
            fr, xy = _moving(fr, xy)
            if len(fr) < MIN_INL:
                continue
            segs += 1
            ts = (np.array(fr, float) - f0) / fps
            start = _contact_point(cam, np.array(ball[f0][:2], float), s["side"])
            fit = fit_flight_ransac(cam, ts, xy, start, s["side"], inlier_px=INL_PX)
            used = np.zeros(len(fr), bool)
            if fit is not None and fit[1] < MAX_ERR and fit[2].sum() >= MIN_INL:
                p, err, inl = fit
                d = np.linalg.norm(cam.project(_pos(p, ts)) - xy, axis=1)
                ok = d < INL_PX
                used |= ok
                spd = float(np.hypot(*p[3:5]) * 0.6818)  # launch speed along the ground, mph
                rows += [[f, rally, round(float(x), 1), round(float(y), 1), "contact", round(float(e), 2), round(spd, 1),
                          round(float(np.hypot(*(cam.project(_pos(p, np.array([t]))) [0] - xy[0])) ), 1)]
                         for f, (x, y), e, o, t in zip(fr, xy, d, ok, ts) if o]
                good += 1
            # further flights in what is left (the ball after a bounce, flights after a missed contact): up to 3
            for _ in range(3):
                rest = np.where(~used)[0]
                if len(rest) < MIN_INL:
                    break
                t2 = ts[rest] - ts[rest][0]
                f2 = free_fit(cam, t2, xy[rest])
                if f2 is None or f2[1] >= MAX_ERR:
                    break
                d2 = np.linalg.norm(cam.project(_pos(f2[0], t2)) - xy[rest], axis=1)
                ok2 = d2 < INL_PX
                if ok2.sum() < MIN_INL:
                    break
                used[rest[ok2]] = True
                spd2 = float(np.hypot(*f2[0][3:5]) * 0.6818)
                rows += [[fr[i], rally, round(float(xy[i][0]), 1), round(float(xy[i][1]), 1), "free", round(float(e), 2), round(spd2, 1), 0.0]
                         for i, e, o in zip(rest, d2, ok2) if o]
    out = ROOT / "data" / "balltrack_labels"
    out.mkdir(parents=True, exist_ok=True)
    rows.sort()
    with (out / f"{rid}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "rally", "x", "y", "kind", "err_px", "mph", "aux"])
        w.writerows(rows)
    n_rally_frames = sum(int((r["end_t"] - r["start_t"]) * fps) for r in A["rallies"])
    print(f"{rid}: {len(rows)} labelled frames ({len(rows) / max(n_rally_frames, 1):.0%} of rally frames); "
          f"flights fitted {good}/{segs}", flush=True)


for rid in sys.argv[1:] or TRAIN_RUNS:
    label_run(rid)
