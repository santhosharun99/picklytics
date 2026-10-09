"""Doubles scoring from tracking vs the TNPPL score overlay, rally by rally (team level).

    PYTHONPATH=. python3 training/compare_doubles_overlay.py tnppl_2_WD [--scoring rally --target 25]
Kits are named once (which kit is Kodai Tigers) by the mapping that agrees best - like naming the teams when a
match is set up. Overlay row 0 = Kodai Tigers.
"""
import argparse, json
from pathlib import Path
from pickleball.doubles import reconstruct
from pickleball.overlay_truth import rally_truth, rally_truth_seq
from pickleball.replays import drop_replays

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--scoring", default="sideout")
ap.add_argument("--target", type=int, default=11)
ap.add_argument("--cue", default="combined")
ap.add_argument("--no-rally-end", action="store_true", help="keep every touch (no rally-end model)")
ap.add_argument("--truth", default="seq", help="seq (overlay transitions, robust to operator lag) or window")
ap.add_argument("-v", action="store_true")
a = ap.parse_args()
C = json.loads((ROOT / "configs" / "tnppl_s2_kt_rtt.json").read_text())
run = ROOT / "outputs" / a.run
A = drop_replays(json.loads((run / "analysis.json").read_text()), run)
if not a.no_rally_end:
    from pickleball.rally_end import apply
    A = apply(A, run)
rec = [r for r in reconstruct(run, A, a.scoring, a.target, 1, 1, serve_cue=a.cue) if not r.get("after_match")]
rost = [C["rosters"]["KT"], C["rosters"]["RTT"]]
truth = rally_truth_seq(run / "overlay.json", rec, rost) if a.truth == "seq" else \
    rally_truth(run / "overlay.json", rec, rost, scoring=a.scoring)
best = None
for flip in (0, 1):  # kit k -> overlay team (k ^ flip)
    ok = sum(1 for r, t in zip(rec, truth) if t["server_team"] is not None and r["serve_kit"] is not None
             and (r["serve_kit"] ^ flip) == t["server_team"])
    if best is None or ok > best[0]:
        best = (ok, flip)
flip = best[1]
ns = cs = nw = cw = nsc = csc = 0
for r, t in zip(rec, truth):
    st = None if r["serve_kit"] is None else r["serve_kit"] ^ flip
    wt = None if r["winner_kit"] is None else r["winner_kit"] ^ flip
    b = t["before"]
    ov_sc = None
    if b and t["server_team"] is not None:
        T = t["server_team"]
        ov_sc = f"{b['score'][T]}-{b['score'][1 - T]}" + (f"-{t['serve_no']}" if a.scoring == "sideout" and t["serve_no"] else "")
    if t["server_team"] is not None and st is not None:
        ns += 1; cs += st == t["server_team"]
    if t["winner_team"] is not None and wt is not None:
        nw += 1; cw += wt == t["winner_team"]
    if ov_sc and r["score_before"]:
        nsc += 1; csc += ov_sc == r["score_before"]
    if a.v:
        print(f"{r['index']:3d} {r['start_t']:7.1f} ours srv {st} win {wt} {str(r['score_before']):8s} {r['how']:14s} | overlay srv {t['server_team']} win {t['winner_team']} {str(ov_sc):8s} {t['server'] or ''} {t['verdict'] or ''}")
games = {}
for r in rec:
    if r.get("game") and r["winner_kit"] is not None:
        g = games.setdefault(r["game"], [0, 0])
        if a.scoring == "rally" or r["how"] in ("point", "game point won"):
            g[r["winner_kit"] ^ flip] += 1
print(f"{a.run}: {len(rec)} rallies | serving team {cs}/{ns} | rally winner {cw}/{nw} | score before rally {csc}/{nsc}"
      f" | final from tracking (KT-RTT) {games}")
