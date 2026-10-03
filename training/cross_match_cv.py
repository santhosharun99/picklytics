"""Does adding a second venue's tags help? Leave-one-rally-out over BOTH matches.

    PYTHONPATH=. python3 training/cross_match_cv.py

For every tagged rally k: model A = trained on the US Open rallies only (minus k), model B = trained on the
US Open + Las Vegas rallies (minus k). Same candidates/clues/thresholds as hit_model_cv / bounce_model_cv
(hits: Random Forest +3D+paddle; bounces: Logistic Regression). Swing models are retrained per fold on the
same training rallies. Scores are reported per match.
"""
import json, time
from pathlib import Path

import numpy as np

import pickleball.hitmodel as hm
from training.bounce_model_cv import checked
from training.evaluate import match
from training.hit_model_cv import TOL, Stretch, fit, pick_threshold, train_swing

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [("US Open", ROOT / "outputs" / "full_match", ROOT / "labels" / "Mens_Singles_GOLD_2026_US_Open.json"),
           ("Las Vegas", ROOT / "outputs" / "lv_fahey_yoshitomi", ROOT / "labels" / "Fahey_Yoshitomi_LV2026.json")]
HIT_MODEL, BOUNCE_MODEL = "Random Forest", "Logistic Regression"


def main():
    t0 = time.time()
    items = []  # (match, stretch, rally)
    for name, run, lab in SOURCES:
        S = Stretch(run, json.loads(lab.read_text()))  # saved paddle detections (no detector in this process)
        items += [(name, S, r) for r in S.rallies]
        print(f"{name}: {len(S.rallies)} tagged rallies ({time.time() - t0:.0f}s)", flush=True)
    swing_rows = [s.swing_xy(r) for _, s, r in items]
    res = {}
    for k, (mname, s, r) in enumerate(items):
        for variant in ("A: US Open only", "B: both matches"):
            train_idx = [j for j, (mj, _, _) in enumerate(items) if j != k and (variant.startswith("B") or mj == "US Open")]
            sw = train_swing([swing_rows[j] for j in train_idx])

            def sp_for(j):
                sj, rj = items[j][1], items[j][2]
                d = {}
                for side in ("near", "far"):
                    fs = [f for f in sj.sf[side] if rj["start"] - 15 <= f <= rj["end"] + 15]
                    d[side] = dict(zip(fs, sw.predict_proba(np.array([sj.sf[side][f] for f in fs]))[:, 1])) if fs else {}
                return d
            # NOTE: training rows use this fold's swing model too (it never saw rally k)
            for kind in ("hits", "bounces"):
                if kind == "bounces" and not checked(r):
                    continue
                X, y, g, fr = [], [], [], []
                for j in train_idx + [k]:
                    sj, rj = items[j][1], items[j][2]
                    if kind == "bounces" and not checked(rj):
                        continue
                    spj = sp_for(j)
                    if kind == "hits":
                        cs = hm.candidates(sj.ctx, spj, rj["start"] - 5, rj["end"] + 5)
                        truth = [e["frame"] for e in rj["events"] if e["kind"] == "hit"]
                    else:
                        A = json.loads((sj.run / "analysis.json").read_text())
                        cs = hm.bounce_candidates(sj.ctx, spj, rj["start"] - 5, rj["end"] + 5, [b["frame"] for b in A["bounces"]])
                        truth = [e["frame"] for e in rj["events"] if e["kind"] == "bounce"]
                    pos = {cs[i] for _, i in match(truth, cs, TOL)}
                    for f in cs:
                        X.append(hm.row(sj.ctx, f, spj, "+3D+paddle") if kind == "hits" else hm.bounce_row(sj.ctx, f, spj))
                        y.append(int(f in pos)); g.append(j); fr.append(f)
                X, y, g, fr = np.array(X, float), np.array(y), np.array(g), np.array(fr)
                model = HIT_MODEL if kind == "hits" else BOUNCE_MODEL
                tr, te = g != k, g == k
                thr = pick_threshold(model, X[tr], y[tr], g[tr], fr[tr])
                pred = hm.nms_pick(fr[te], fit(model, X[tr], y[tr]).predict_proba(X[te])[:, 1], thr)
                truth = [e["frame"] for e in r["events"] if e["kind"] == ("hit" if kind == "hits" else "bounce")]
                m = len(match(truth, pred, TOL))
                key = (mname, variant, kind)
                tp, fp, fn = res.get(key, (0, 0, 0))
                res[key] = (tp + m, fp + len(pred) - m, fn + len(truth) - m)
        print(f"  rally {k + 1}/{len(items)} done ({time.time() - t0:.0f}s)", flush=True)
    print(f"\n{'match':10s} {'model':18s} {'what':8s} {'found':>9s} {'false':>6s} {'F1':>5s}")
    out = []
    for (mname, variant, kind), (tp, fp, fn) in sorted(res.items()):
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        print(f"{mname:10s} {variant:18s} {kind:8s} {tp:>4d}/{tp + fn:<4d} {fp:>6d} {f1:>5.0%}")
        out.append({"match": mname, "model": variant, "what": kind, "found": tp, "truth": tp + fn, "false": fp, "f1": f1})
    (ROOT / "outputs" / "cross_match_cv.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
