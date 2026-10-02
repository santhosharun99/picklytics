"""Check 3D flights against the labelled rallies.

    PYTHONPATH=. python3 training/test_3d.py [--beta 0.5] [--penalty 900] [run_dir ...]

For every labelled hit / bounce: is there a 3D contact within +-4 frames, and how high is it?
If hits sit in the air and bounces on the ground, contact height separates them.
"""
import argparse, json, time
from pathlib import Path

import numpy as np

from pickleball.analysis import load_detections
from pickleball.ball import track_ball
from pickleball.camera3d import calibrate_from_run
from pickleball.trajectory3d import track_flights
from training.evaluate import match

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "labels" / "Mens_Singles_GOLD_2026_US_Open.json"
RUNS = sorted((ROOT / "outputs").glob("cmp_ball_detector_v3_kaggle_*"))


def contacts_for(run: Path, beta, penalty):
    meta, recs = load_detections(run)
    cam = calibrate_from_run(run)
    main = {r["f"] for r in recs if r.get("court")}
    ball = {f: v for f, v in track_ball(recs).items() if f in main}
    flights, contacts = track_flights(cam.P, ball, meta["fps"], beta=beta, penalty=penalty)
    return meta, cam, ball, flights, contacts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--penalty", type=float, default=900.0)
    a = ap.parse_args()
    L = json.loads(LABELS.read_text())
    hits_h, bounce_h, other_h = [], [], []
    n_hit = n_bounce = found_hit = found_bounce = n_contacts = 0
    for run in [Path(r) for r in a.runs] or RUNS:
        t0 = time.time()
        meta, cam, ball, flights, contacts = contacts_for(run, a.beta, a.penalty)
        lo, hi = meta["first_frame"], meta["last_frame"]
        ral = [r for r in L["rallies"] if r["end"] is not None and lo <= r["start"] and r["end"] < hi and r.get("reviewed", True)]
        inl = lambda f: any(r["start"] <= f <= r["end"] for r in ral)
        cs = [c for c in contacts if inl(c["frame"])]
        th = [e["frame"] for r in ral for e in r["events"] if e["kind"] == "hit"]
        tb = [e["frame"] for r in ral if r.get("mode") != "hits" for e in r["events"] if e["kind"] == "bounce"]
        cf = [c["frame"] for c in cs]
        mh, mb = match(th, cf, 4), match(tb, cf, 4)
        used = {j for _, j in mh} | {j for _, j in mb}
        hits_h += [cs[j]["pos"][2] for _, j in mh]
        bounce_h += [cs[j]["pos"][2] for _, j in mb]
        other_h += [c["pos"][2] for j, c in enumerate(cs) if j not in used]
        n_hit, n_bounce, n_contacts = n_hit + len(th), n_bounce + len(tb), n_contacts + len(cs)
        found_hit, found_bounce = found_hit + len(mh), found_bounce + len(mb)
        rms = np.median([f["rms_px"] for f in flights])
        print(f"{run.name}: camera at {cam.position.round(1)} ft, f={cam.K[0, 0]:.0f}, court err {cam.err_px:.2f}px; "
              f"{len(flights)} flights (median fit error {rms:.1f}px), {time.time() - t0:.0f}s")
    q = lambda v: np.percentile(v, [10, 50, 90]).round(1).tolist() if v else []
    print(f"\nhits    found as contacts: {found_hit}/{n_hit}   height 10/50/90%: {q(hits_h)} ft")
    print(f"bounces found as contacts: {found_bounce}/{n_bounce}   height 10/50/90%: {q(bounce_h)} ft")
    print(f"other contacts (no label): {len(other_h)}   height 10/50/90%: {q(other_h)} ft   (total contacts {n_contacts})")
    for thr in (0.5, 1.0, 1.5):
        ok = sum(h > thr for h in hits_h) + sum(h <= thr for h in bounce_h)
        print(f"  split at {thr} ft: hit/bounce told apart correctly {ok}/{len(hits_h) + len(bounce_h)}")


if __name__ == "__main__":
    main()
