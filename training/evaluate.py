"""Score an analysis run against hand labels from the labelling tool.

    python training/evaluate.py outputs/<run> labels/<video>.json [--tol 4]

Only labelled rallies inside the run's time range are scored. Reports:
- hit & bounce detection: precision / recall / F1 (match = within --tol frames)
- bounce landing error in feet (labels with a clicked ball position)
- for matched hits: side/player, shot type, forehand/backhand, volley accuracy + confusion
- rallies: how many labelled rallies were found, shots-per-rally error
Writes <run>/eval.json.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def norm_shot(s):
    return (s or "").replace("third-shot ", "") or None


def match(truth: list[int], pred: list[int], tol: int):
    """Greedy one-to-one matching of frame numbers by closeness. Returns [(ti, pi)]."""
    pairs = sorted((abs(t - p), i, j) for i, t in enumerate(truth) for j, p in enumerate(pred) if abs(t - p) <= tol)
    used_t, used_p, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_t and j not in used_p:
            used_t.add(i)
            used_p.add(j)
            out.append((i, j))
    return out


def prf(n_match, n_truth, n_pred):
    p = n_match / n_pred if n_pred else 0.0
    r = n_match / n_truth if n_truth else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
            "truth": n_truth, "predicted": n_pred, "matched": n_match}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("labels")
    ap.add_argument("--tol", type=int, default=4, help="frame tolerance for matching events")
    args = ap.parse_args()

    run = Path(args.run)
    A = json.loads((run / "analysis.json").read_text())
    L = json.loads(Path(args.labels).read_text())
    meta = A["meta"]
    lo, hi = meta["first_frame"], meta["last_frame"]
    names = {pid: p["name"] for pid, p in A["players"].items()}
    label_names = L["players"]  # {"A": name, "B": name}

    # homographies for projecting clicked bounce positions onto the court
    H = {}
    with (run / "detections.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("court"):
                H[r["f"]] = np.asarray(r["court"]["H"])

    # auto-labelled rallies only count once a person has reviewed them
    rallies = [r for r in L["rallies"] if r["end"] is not None and lo <= r["start"] and r["end"] < hi
               and r.get("reviewed", True)]
    if not rallies:
        raise SystemExit("No completed labelled rallies inside this run's time range.")
    in_labelled = lambda f: any(r["start"] <= f <= r["end"] for r in rallies)

    t_hits = [e for r in rallies for e in r["events"] if e["kind"] == "hit"]
    t_bounces = [e for r in rallies for e in r["events"] if e["kind"] == "bounce"]
    p_hits = [s for s in A["shots"] if in_labelled(s["frame"])]
    p_bounces = [b for b in A["bounces"] if in_labelled(b["frame"])]

    hm = match([e["frame"] for e in t_hits], [s["frame"] for s in p_hits], args.tol)
    bm = match([e["frame"] for e in t_bounces], [b["frame"] for b in p_bounces], args.tol)

    # --- attributes of matched hits ---
    agree = defaultdict(lambda: [0, 0])
    confusion = defaultdict(Counter)
    offsets = []
    for i, j in hm:
        t, p = t_hits[i], p_hits[j]
        offsets.append(p["frame"] - t["frame"])
        def cmp(key, tv, pv):
            if tv is None or pv is None:
                return
            agree[key][1] += 1
            agree[key][0] += tv == pv
        cmp("side", t.get("side"), p.get("side"))
        cmp("player", label_names.get(t.get("player")), names.get(p.get("player"), p.get("player")))
        cmp("shot_type", norm_shot(t.get("shot")), norm_shot(p.get("shot_type")))
        cmp("stroke", t.get("stroke"), p.get("stroke"))
        cmp("volley", bool(t.get("volley")), bool(p.get("volley")))
        if t.get("shot"):
            confusion[norm_shot(t["shot"])][norm_shot(p["shot_type"])] += 1

    # --- bounce landing error ---
    errs = []
    for i, j in bm:
        t, p = t_bounces[i], p_bounces[j]
        f = t["frame"]
        Hf = H.get(f) if f in H else next((H[g] for g in range(f - 3, f + 4) if g in H), None)
        if t.get("xy") and Hf is not None:
            c = cv2.perspectiveTransform(np.array([[t["xy"]]], np.float32), Hf)[0, 0]
            errs.append(float(np.hypot(c[0] - p["court"][0], c[1] - p["court"][1])))

    # --- rallies ---
    found, shot_err = 0, []
    for r in rallies:
        ov = [x for x in A["rallies"] if x["start_t"] * meta["fps"] <= r["end"] and x["end_t"] * meta["fps"] >= r["start"]]
        if ov:
            found += 1
            n_true = sum(e["kind"] == "hit" for e in r["events"])
            shot_err.append(abs(sum(x["shots"] for x in ov) - n_true))

    report = {
        "labelled_rallies": len(rallies),
        "hits": prf(len(hm), len(t_hits), len(p_hits)),
        "hit_timing_offset_frames": {"mean": round(float(np.mean(offsets)), 2) if offsets else None,
                                     "abs_mean": round(float(np.mean(np.abs(offsets))), 2) if offsets else None},
        "bounces": prf(len(bm), len(t_bounces), len(p_bounces)),
        "bounce_landing_error_ft": {"mean": round(float(np.mean(errs)), 2) if errs else None,
                                    "median": round(float(np.median(errs)), 2) if errs else None, "n": len(errs)},
        "matched_hit_accuracy": {k: {"accuracy": round(a / n, 3), "n": n} for k, (a, n) in agree.items()},
        "shot_type_confusion": {k: dict(v) for k, v in confusion.items()},
        "rallies": {"found": found, "of": len(rallies),
                    "mean_abs_shot_count_error": round(float(np.mean(shot_err)), 2) if shot_err else None},
    }
    (run / "eval.json").write_text(json.dumps(report, indent=2))

    h, b = report["hits"], report["bounces"]
    print(f"Labelled rallies scored: {len(rallies)}   (tolerance ±{args.tol} frames)")
    print(f"Hits     recall {h['recall']:.0%}  precision {h['precision']:.0%}  F1 {h['f1']:.0%}  ({h['matched']}/{h['truth']} found, {h['predicted']} predicted)")
    print(f"Bounces  recall {b['recall']:.0%}  precision {b['precision']:.0%}  F1 {b['f1']:.0%}  ({b['matched']}/{b['truth']} found, {b['predicted']} predicted)")
    le = report["bounce_landing_error_ft"]
    if le["n"]:
        print(f"Landing spot error: median {le['median']} ft, mean {le['mean']} ft (n={le['n']})")
    for k, v in report["matched_hit_accuracy"].items():
        print(f"{k:10s} accuracy {v['accuracy']:.0%}  (n={v['n']})")
    print(f"Rallies found {found}/{len(rallies)}, shot-count error per rally {report['rallies']['mean_abs_shot_count_error']}")
    if confusion:
        print("\nShot type confusion (rows = truth, cols = predicted):")
        cols = sorted({c for v in confusion.values() for c in v})
        print(" " * 14 + "".join(f"{c[:10]:>11s}" for c in cols))
        for t in sorted(confusion):
            print(f"{t[:13]:14s}" + "".join(f"{confusion[t].get(c, 0):11d}" for c in cols))
    print(f"\nSaved {run / 'eval.json'}")


if __name__ == "__main__":
    main()
