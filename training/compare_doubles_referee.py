"""Doubles scoring from tracking vs a PPA referee live-scoring log (pickleball.com), rally by rally.

    PYTHONPATH=. python3 training/compare_doubles_referee.py outputs/lv_waters_johns data/reference/waters_johns_black_tardio_lv2026.json
Rallies are aligned in time with compare_referee.align (the broadcast cuts dead time). Kits are named once
(which kit is team A) by the mapping that agrees best.
"""
import argparse, json
from pathlib import Path
from pickleball.doubles import reconstruct
from training.compare_referee import align

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("ref")
ap.add_argument("--cue", default="combined")
ap.add_argument("-v", action="store_true")
a = ap.parse_args()
A = json.loads((Path(a.run) / "analysis.json").read_text())
R = json.loads(Path(a.ref).read_text())
team = {n: t for t, ns in R["teams"].items() for n in ns}
ref = []
for row in R["rallies"]:
    h, m, s = map(int, row[0].split(":"))
    h2, m2, s2 = map(int, row[4].split(":"))
    st = team[row[2]]
    ref.append({"start_s": h * 3600 + m * 60 + s, "end_s": h2 * 3600 + m2 * 60 + s2, "start_clock": row[0],
                "server_team": st, "winner_team": st if row[5] == "P" else ("B" if st == "A" else "A"),
                "score_before": row[1]})
rec = [r for r in reconstruct(a.run, A, serve_cue=a.cue) if not r.get("after_match")]
m = align(rec, ref)
best = None
for flip in ("AB", "BA"):  # kit 0 -> flip[0]
    name = lambda k: None if k is None else flip[k]
    ok = sum(name(rec[i]["serve_kit"]) == ref[j]["server_team"] for i, j in m.items())
    if best is None or ok > best[0]:
        best = (ok, flip)
name = lambda k: None if k is None else best[1][k]
srv = sum(name(rec[i]["serve_kit"]) == ref[j]["server_team"] for i, j in m.items())
win = sum(name(rec[i]["winner_kit"]) == ref[j]["winner_team"] for i, j in m.items())
sc = sum(rec[i]["score_before"] == ref[j]["score_before"] for i, j in m.items())
if a.v:
    for i, j in sorted(m.items()):
        r, f = rec[i], ref[j]
        print(f"{r['start_t'] / 60:6.2f} min | ours srv {name(r['serve_kit'])} win {name(r['winner_kit'])} {r['score_before']} {r['how']:14s}"
              f" | ref {f['start_clock']} srv {f['server_team']} win {f['winner_team']} {f['score_before']}")
print(f"{Path(a.run).name}: referee {len(ref)} rallies, ours {len(rec)}, matched {len(m)} | serving team {srv}/{len(m)}"
      f" | rally winner {win}/{len(m)} | score before rally {sc}/{len(m)} | result: {R['result']}")
