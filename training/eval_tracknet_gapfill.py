"""Does the TrackNet-style gap filler help? Before/after on a match it was NOT trained on.

    PYTHONPATH=. python3 training/eval_tracknet_gapfill.py outputs/<run> <pred.csv> [--names "P1=..,P2=.."]

Makes outputs/<run>_tn: the same detections, plus a TrackNet ball ONLY in frames where YOLO found no ball
(score >= MIN_SCORE). Re-runs the analysis there and prints one table, original vs gap-filled:
ball seen in play frames, trusted 3D arcs, shots with a measured speed, and (when available) hits/bounces
vs analyst tags and server/winner vs the referee log.
"""
import argparse, csv, json, shutil
from pathlib import Path

import numpy as np

from pickleball.analysis import load_detections, run_analysis
from pickleball.ball import track_ball
from pickleball.ball3d import match_arcs
from pickleball.camera3d import calibrate_from_run

ROOT = Path(__file__).resolve().parents[1]
MIN_SCORE = 0.5
BALL_PX = 14  # typical ball box size at 1920x1080


def gapfill(run: Path, pred_csv: Path) -> Path:
    out = run.parent / (run.name + "_tn")
    out.mkdir(exist_ok=True)
    meta = json.loads((run / "meta.json").read_text())
    sx, sy = meta["width"] / 512, meta["height"] / 288
    pred = {}
    for r in csv.DictReader(pred_csv.open()):
        if r["x"] and float(r["score"]) >= MIN_SCORE:
            pred[int(r["frame"])] = (float(r["x"]) * sx, float(r["y"]) * sy, float(r["score"]))
    added = 0
    with (run / "detections.jsonl").open() as src, (out / "detections.jsonl").open("w") as dst:
        for line in src:
            rec = json.loads(line)
            f = rec["f"]
            if rec.get("court") and f in pred and not rec.get("balls"):
                x, y, s = pred[f]
                rec["balls"] = [[round(x - BALL_PX / 2, 1), round(y - BALL_PX / 2, 1), round(x + BALL_PX / 2, 1),
                                 round(y + BALL_PX / 2, 1), round(min(0.9, 0.3 + 0.5 * s), 3)]]
                rec["tn"] = True
                added += 1
            dst.write(json.dumps(rec, separators=(",", ":")) + "\n")
    for name in ("meta.json", "paddles.json"):
        if (run / name).exists():
            shutil.copy(run / name, out / name)
    print(f"{out.name}: TrackNet ball added in {added} frames where YOLO had none", flush=True)
    return out


def metrics(run: Path) -> dict:
    meta, recs = load_detections(run)
    A = json.loads((run / "analysis.json").read_text())
    fps = meta["fps"]
    ball = track_ball(recs)
    in_rally = set()
    for r in A["rallies"]:
        in_rally.update(range(int(r["start_t"] * fps), int(r["end_t"] * fps) + 1))
    main = {r["f"] for r in recs if r.get("court")} & in_rally
    seen = sum(1 for f in main if f in ball and not ball[f][3]) / max(len(main), 1)
    arcs = [a for v in match_arcs(A, ball, calibrate_from_run(run).P).values() for a in v]
    shots = [s for s in A["shots"] if not s.get("inferred")]
    return {"rallies": len(A["rallies"]), "shots": len(shots), "bounces": len(A["bounces"]),
            "ball_seen_in_rallies": round(100 * seen, 1),
            "trusted_arcs_pct": round(100 * np.mean([a.get("confident", False) for a in arcs]), 1) if arcs else None,
            "shots_with_speed_pct": round(100 * np.mean([bool(s.get("ball_mph")) for s in shots]), 1) if shots else None}


def tags(run: Path) -> dict | None:
    """Hits / bounces vs analyst tags (US Open only)."""
    import contextlib, io, sys
    import training.evaluate as E
    labels = ROOT / "labels" / "Mens_Singles_GOLD_2026_US_Open.json"
    meta = json.loads((run / "meta.json").read_text())
    if "Men's Singles GOLD" not in meta["video"]:
        return None
    sys.argv = ["evaluate", str(run), str(labels)]
    with contextlib.redirect_stdout(io.StringIO()):
        E.main()
    e = json.loads((run / "eval.json").read_text())
    return {"hits_f1": e["hits"]["f1"], "hits_found": f'{e["hits"]["matched"]}/{e["hits"]["truth"]}',
            "bounces_f1": e["bounces"]["f1"], "bounces_found": f'{e["bounces"]["matched"]}/{e["bounces"]["truth"]}'}


def referee(run: Path) -> dict | None:
    """Server / winner vs the referee log (Las Vegas only), via the rally->referee map read off the scoreboard.
    Only meaningful when the rally list is unchanged; otherwise reports the referee alignment tool's view."""
    ref_p = ROOT / "data" / "reference" / "fahey_yoshitomi_lv2026.json"
    meta = json.loads((run / "meta.json").read_text())
    if "Fahey" not in meta["video"] or not ref_p.exists():
        return None
    import training.compare_referee as C
    from pickleball.outcome import match_from_analysis
    A = json.loads((run / "analysis.json").read_text())
    ref = json.loads(ref_p.read_text())["rallies"]
    ours = [o for o in match_from_analysis(A, run_dir=run) if not o.get("after_match")]
    m = C.align(ours, ref)
    return {"rallies_matched": f"{len(m)}/{len(ref)}",
            "server_ok": f'{sum(ours[i]["server"] == ref[j]["server"] for i, j in m.items())}/{len(m)}',
            "winner_ok": f'{sum(ours[i]["winner"] == ref[j]["winner"] for i, j in m.items())}/{len(m)}'}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("pred")
    ap.add_argument("--names", default="")
    a = ap.parse_args()
    run = Path(a.run)
    names = dict(kv.split("=") for kv in a.names.split(",")) if a.names else None
    tn = gapfill(run, Path(a.pred))
    run_analysis(tn, 2, names)
    rows = {"original": run, "with TrackNet gaps": tn}
    table = {}
    for label, d in rows.items():
        m = metrics(d)
        for extra in (tags(d), referee(d)):
            if extra:
                m.update(extra)
        table[label] = m
    keys = list(table["original"])
    print(f"\n{'':28s} {'original':>14s} {'with TrackNet':>14s}")
    for k in keys:
        print(f"{k:28s} {str(table['original'][k]):>14s} {str(table['with TrackNet gaps'].get(k)):>14s}")
    (tn / "tracknet_compare.json").write_text(json.dumps(table, indent=1))
