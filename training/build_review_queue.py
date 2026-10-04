"""Pick the shots most worth a person's time and write a review queue for dashboard/review.html.

    PYTHONPATH=. python3 training/build_review_queue.py outputs/<run> <VideoStem> [--n 50]

Mix (active learning, with a random share so a confidently wrong model still gets checked):
  40% most uncertain model predictions, 35% likely soft / rare shots (slow ball, landing in or near the kitchen,
  hitter at the net, or predicted non-drive), 25% random. Serves, returns and shots already answered are skipped.
Writes labels/<VideoStem>.queue.json; answers are saved by the review page to labels/<VideoStem>.shots.json.
"""
import argparse, json, random
from pathlib import Path

import numpy as np

from pickleball.shottype import MODEL_PATH, features

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("video")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run = Path(a.run)
    A = json.loads((run / "analysis.json").read_text())
    R = json.loads((run / "render_data.json").read_text())
    fps = A["meta"]["fps"]
    done = set()
    lab = ROOT / "labels" / f"{a.video}.json"
    if lab.exists():
        done |= {e["frame"] for r in json.loads(lab.read_text())["rallies"] for e in r["events"]
                 if e["kind"] == "hit" and (e.get("shot_tagged") or not e.get("auto")) and e.get("shot")}
    ans = ROOT / "labels" / f"{a.video}.shots.json"
    if ans.exists():
        done |= {int(f) for f in json.loads(ans.read_text()).get("answers", {})}
    import joblib
    model = joblib.load(MODEL_PATH)["model"]
    F = features(A, R["players"])
    img = {e["frame"]: e["img"] for e in R["events"] if e["kind"] == "hit"}
    cands = []
    for s in A["shots"]:
        if s.get("shot_type") in ("serve", "return") or s.get("inferred") or s["frame"] not in F:
            continue
        if any(abs(s["frame"] - d) <= 4 for d in done):
            continue
        pr = model.predict_proba(np.nan_to_num(np.array([F[s["frame"]]], float), nan=-1))[0]
        pred = str(model.classes_[int(np.argmax(pr))])
        mph = s.get("ball_mph") or s.get("speed_mph")
        land = s.get("land_3d") or s.get("landing_court")
        soft = (mph is not None and mph < 28) or (land is not None and abs(land[1] - 22) <= 9) or \
               (s.get("hitter_court") is not None and abs(s["hitter_court"][1] - 22) <= 9) or pred != "drive"
        cands.append({"frame": s["frame"], "t": round(s["frame"] / fps, 2), "rally": s["rally"], "side": s.get("side"),
                      "player": s.get("player"), "pred": pred, "prob": round(float(pr.max()), 2), "soft": bool(soft),
                      "img": img.get(s["frame"])})
    rng = random.Random(a.seed)
    picked, seen = [], set()

    def take(pool, k, reason):
        for c in pool:
            if len([p for p in picked if p["reason"] == reason]) >= k:
                break
            if c["frame"] not in seen:
                seen.add(c["frame"]); picked.append({**c, "reason": reason})

    n_unc, n_soft = int(a.n * 0.40), int(a.n * 0.35)
    take(sorted(cands, key=lambda c: c["prob"]), n_unc, "uncertain")
    soft = [c for c in cands if c["soft"]]; rng.shuffle(soft)
    take(soft, n_soft, "soft or rare")
    rest = cands[:]; rng.shuffle(rest)
    take(rest, a.n - len(picked), "random")
    rng.shuffle(picked)
    out = ROOT / "labels" / f"{a.video}.queue.json"
    out.write_text(json.dumps({"video": f"{a.video}.mp4", "fps": fps, "size": [A["meta"]["width"], A["meta"]["height"]],
                               "items": picked}, indent=1))
    from collections import Counter
    print(f"{out.name}: {len(picked)} shots ({dict(Counter(p['reason'] for p in picked))}) from {len(cands)} candidates")


if __name__ == "__main__":
    main()
