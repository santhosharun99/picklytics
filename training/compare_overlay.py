"""Rally-by-rally check of our server / winner against the TNPPL score overlay.

    PYTHONPATH=. python3 training/compare_overlay.py outputs/reports/tnppl_1_MS.json outputs/tnppl_1_MS/overlay.json \
        --teams Rahul Ramakrishnan [--scoring rally] [--singles]
--teams: report player (or team) names in overlay row order (row 0 = Kodai Tigers).
"""
import argparse, json
from pickleball.overlay_truth import rally_truth

ap = argparse.ArgumentParser()
ap.add_argument("report")
ap.add_argument("overlay")
ap.add_argument("--teams", nargs=2, required=True)
ap.add_argument("--roster0", nargs="*", default=[])
ap.add_argument("--roster1", nargs="*", default=[])
ap.add_argument("--scoring", default="sideout")
ap.add_argument("--singles", action="store_true")
a = ap.parse_args()
R = json.loads(open(a.report).read())
rallies = sorted(R["rallies"], key=lambda r: r["start_t"])
rosters = [a.roster0 or [a.teams[0]], a.roster1 or [a.teams[1]]]
truth = rally_truth(a.overlay, [{"index": r["id"], "start_t": r["start_t"], "end_t": r["end_t"]} for r in rallies],
                    rosters, scoring=a.scoring, use_server_name=not a.singles, singles=a.singles)
team = {a.teams[0]: 0, a.teams[1]: 1}
ns = nw = cs = cw = 0
for r, t in zip(rallies, truth):
    b = t["before"]["score"] if t["before"] else None
    our_s, our_w = team.get(r["server"]), team.get(r["winner"])
    ok_s = None if t["server_team"] is None or our_s is None else our_s == t["server_team"]
    ok_w = None if t["winner_team"] is None or our_w is None else our_w == t["winner_team"]
    ns += ok_s is not None; cs += bool(ok_s); nw += ok_w is not None; cw += bool(ok_w)
    mark = lambda ok: "  " if ok is None else ("ok" if ok else "XX")
    tn = lambda x: "-" if x is None else a.teams[x]
    print(f"{r['id']:3d} {r['start_t']:7.1f}  overlay {str(b):9s} srv {tn(t['server_team']):14s} win {tn(t['winner_team']):14s}"
          f" | ours srv {str(r['server']):14s}{mark(ok_s)} win {str(r['winner']):14s}{mark(ok_w)}  {t['verdict']}")
print(f"\nserver agrees {cs}/{ns}, winner agrees {cw}/{nw} (rallies with an overlay verdict; {len(rallies)} rallies)")
