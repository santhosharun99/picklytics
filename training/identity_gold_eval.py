"""Mark-once identity vs tracklets labelled by eye (labels/<video>.identity_gold.json).

    PYTHONPATH=. python3 training/identity_gold_eval.py tnppl_2_WD [--marks 1] [--trials 20]
Simulated user: `--marks` random gold tracklets per player are 'clicked' (named); every other gold tracklet tests
pickleball/identity.assign. Repeated over random choices of the marked tracklets (a real user picks any frame).
Also reports nearest-centroid accuracy per feature, to see which look features separate partners.
"""
import argparse, json, pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from pickleball import identity as I
from pickleball.replays import drop_replays

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--marks", type=int, default=1)
ap.add_argument("--trials", type=int, default=20)
a = ap.parse_args()
run = ROOT / "outputs" / a.run
A = drop_replays(json.loads((run / "analysis.json").read_text()), run)
trk = pickle.loads((run / "tracklets.pkl").read_bytes())
I.add_kits_from_rallies(trk, run, A)
G = json.loads((ROOT / "labels" / f"{Path(A['meta']['video']).stem}.identity_gold.json").read_text())
gold = {int(k): v for k, v in G["tracklets"].items()}
C = json.loads((ROOT / "configs" / "tnppl_s2_kt_rtt.json").read_text())
team = {n: tm for tm, ns in C["rosters"].items() for n in ns}
T = {t["id"]: t for t in trk}
rng = np.random.default_rng(0)
by = defaultdict(list)
for i, n in gold.items():
    by[n].append(i)
acc = []
for _ in range(a.trials):
    marks = [{"tracklet": int(i), "name": n, "team": team[n]} for n, ids in by.items()
             for i in rng.choice(ids, a.marks, replace=False)]
    names = I.assign(trk, marks)
    marked = {m["tracklet"] for m in marks}
    test = [(i, n) for i, n in gold.items() if i not in marked]
    acc.append(np.mean([names.get(i, {}).get("name") == n for i, n in test]))
print(f"{a.run}: assign() with {a.marks} mark(s) per player: {np.mean(acc):.0%} (min {np.min(acc):.0%}, max {np.max(acc):.0%}) "
      f"over {a.trials} random choices of the marked frames; {len(gold)} gold tracklets")


# which features separate partners: leave-one-out nearest centroid within each team
def f_lower(t):
    c = t["cols"].mean(0)
    return c[72:] / (np.linalg.norm(c[72:]) + 1e-6)  # hips->knees and knees->feet colour


variants = {"embedding": lambda t: t["emb"], "colour (all)": lambda t: t["col"] / np.linalg.norm(t["col"]),
            "colour lower body": f_lower, "height": lambda t: np.array([t["height"]])}
for v, f in variants.items():
    ok = n = 0
    for tm in ("KT", "RTT"):
        cs = [(i, nm) for i, nm in gold.items() if team[nm] == tm]
        for i, nm in cs:
            cent = defaultdict(list)
            for j, m in cs:
                if j != i:
                    cent[m].append(f(T[j]))
            pred = min(cent, key=lambda m: np.linalg.norm(f(T[i]) - np.mean(cent[m], 0)))
            ok += pred == nm
            n += 1
    print(f"  nearest centroid, {v:18s} {ok}/{n}")
