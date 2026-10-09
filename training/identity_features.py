"""Which look features tell doubles partners apart? Leave-one-out nearest-centroid on overlay-labelled serves.

    PYTHONPATH=. python3 training/identity_features.py tnppl_2_WD [tnppl_3_MD ...]
Cases: rallies where the TNPPL overlay names the server and our first detected hit is the serve (on the side the
serve formation says) -> the server's tracklet carries that name. Within each team, every case is classified by
the nearest name centroid of the other cases. Needs outputs/<run>/tracklets.pkl (identity_eval.py builds it).
"""
import json, pickle, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pickleball import identity as I
from pickleball.doubles import formation_serve_side
from pickleball.overlay_truth import rally_truth_seq
from pickleball.replays import drop_replays

ROOT = Path(__file__).resolve().parents[1]
C = json.loads((ROOT / "configs" / "tnppl_s2_kt_rtt.json").read_text())
team = {n: tm for tm, ns in C["rosters"].items() for n in ns}


def serve_cases(run):
    A = drop_replays(json.loads((run / "analysis.json").read_text()), run)
    trk = pickle.loads((run / "tracklets.pkl").read_bytes())
    I.add_kits_from_rallies(trk, run, A)
    rallies = sorted(A["rallies"], key=lambda r: r["start_t"])
    truth = rally_truth_seq(run / "overlay.json", rallies, [C["rosters"]["KT"], C["rosters"]["RTT"]])
    fm = formation_serve_side(run, A)
    cases = []
    for r, t in zip(rallies, truth):
        if not t["server"]:
            continue
        real = [s for s in sorted(A["shots"], key=lambda s: s["frame"]) if s["rally"] == r["index"] and not s.get("inferred")]
        if not real or real[0]["side"] != fm.get(r["index"], {}).get("serve_side"):
            continue
        tr = I.tracklet_at(trk, real[0]["frame"], real[0]["side"], court=real[0].get("hitter_court"))
        if tr is not None and tr.get("emb") is not None:
            cases.append((t["server"], tr))
    return trk, cases


VARIANTS = ["emb", "emb_c", "col", "col_c", "h", "emb_c+col_c", "emb_c+col_c+h", "col_c+h", "partner_metric"]


def featurize(variant, t, mu):
    if variant == "partner_metric":
        return mu["pm"](t)
    parts = []
    for p in variant.split("+"):
        if p in ("emb", "emb_c"):
            v = t["emb"] - (mu["emb"] if p == "emb_c" else 0)
        elif p in ("col", "col_c"):
            v = t["col"] - (mu["col"] if p == "col_c" else 0)
        else:
            parts.append(np.array([(t["height"] - mu["h"]) / mu["h_sd"]]))
            continue
        parts.append(v / (np.linalg.norm(v) + 1e-6))
    return np.concatenate(parts)


res = defaultdict(lambda: [0, 0])
for rid in sys.argv[1:]:
    trk, cases = serve_cases(ROOT / "outputs" / rid)
    print(rid, len(cases), "labelled serves", dict(Counter(n for n, _ in cases)))
    for tm in C["rosters"]:
        cs = [(n, tr) for n, tr in cases if team[n] == tm]
        names = sorted({n for n, _ in cs})
        if len(names) < 2:
            continue
        kit = Counter(tr["kit"] for _, tr in cs).most_common(1)[0][0]
        pool = [t for t in trk if t["kit"] == kit and t.get("emb") is not None]
        hs = [t["height"] for t in pool if t.get("height")]
        mu = {"emb": np.mean([t["emb"] for t in pool], 0), "col": np.mean([t["col"] for t in pool], 0),
              "h": np.mean(hs), "h_sd": np.std(hs) + 1e-6, "pm": I.partner_metric(trk, kit)}
        for v in VARIANTS:
            if v == "partner_metric" and mu["pm"] is None:
                continue
            X = [(n, featurize(v, tr, mu)) for n, tr in cs]
            for i, (n, x) in enumerate(X):
                cent = {k: np.mean([y for j, (m, y) in enumerate(X) if m == k and j != i], 0)
                        for k in names if any(m == k and j != i for j, (m, _) in enumerate(X))}
                if len(cent) < 2:
                    continue
                res[v][0] += min(cent, key=lambda k: np.linalg.norm(x - cent[k])) == n
                res[v][1] += 1
for v in VARIANTS:
    print(f"{v:16s} {res[v][0]}/{res[v][1]}")
