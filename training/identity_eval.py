"""Mark-once identity on a TNPPL doubles run, scored against the overlay's server names.

    PYTHONPATH=. python3 training/identity_eval.py tnppl_2_WD [--marks 1]
Simulated user marks: for each player, the first rally(s) where the overlay names them as the server and our
first hit is a detected serve on the serving side -> that server's tracklet is 'clicked' with their name (what
a user would do once per player on dashboard/identify.html). Every other rally with an overlay server name tests
the identity: the name we give the serving player vs the overlay.
"""
import argparse, json, pickle
from collections import Counter
from pathlib import Path
from pickleball import identity as I
from pickleball.doubles import formation_serve_side
from pickleball.overlay_truth import rally_truth_seq
from pickleball.replays import drop_replays

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--marks", type=int, default=1, help="marks per player")
a = ap.parse_args()
C = json.loads((ROOT / "configs" / "tnppl_s2_kt_rtt.json").read_text())
run = ROOT / "outputs" / a.run
A = drop_replays(json.loads((run / "analysis.json").read_text()), run)
cache = run / "tracklets.pkl"
if cache.exists():
    trk = pickle.loads(cache.read_bytes())
else:
    trk = I.build_tracklets(run)
    I.add_looks(trk, A["meta"]["video"])
    cache.write_bytes(pickle.dumps(trk))
I.add_kits_from_rallies(trk, run, A)
print(len(trk), "tracklets")
rallies = sorted(A["rallies"], key=lambda r: r["start_t"])
truth = rally_truth_seq(run / "overlay.json", rallies, [C["rosters"]["KT"], C["rosters"]["RTT"]])
fm = formation_serve_side(run, A)
fps = A["meta"]["fps"]
cases = []  # (rally, server name, serve tracklet)
for r, t in zip(rallies, truth):
    if not t["server"]:
        continue
    sh = sorted([s for s in A["shots"] if s["rally"] == r["index"]], key=lambda s: s["frame"])
    real = [s for s in sh if not s.get("inferred")]
    side = fm.get(r["index"], {}).get("serve_side")
    if not real or real[0]["side"] != side:
        continue  # the serve itself was not seen
    tr = I.tracklet_at(trk, real[0]["frame"], real[0]["side"], court=real[0].get("hitter_court"))
    if tr is not None:
        cases.append((r, t["server"], tr))
team = {n: tm for tm, ns in C["rosters"].items() for n in ns}
kitv = Counter((team[n], tr["kit"]) for _, n, tr in cases)
team_kit = {tm: max((0, 1), key=lambda k: kitv[(tm, k)]) for tm in C["rosters"]}
used = Counter()
marks, test = [], []
for r, name, tr in cases:
    if used[name] < a.marks and tr["kit"] == team_kit[team[name]]:  # a user clicks the right player
        marks.append({"tracklet": tr["id"], "name": name, "team": team[name]})
        used[name] += 1
    else:
        test.append((r, name, tr))
names = I.assign(trk, marks)
ok = sum(names.get(tr["id"], {}).get("name") == name for _, name, tr in test)
print(f"{a.run}: marks {dict(used)} | serving player named right {ok}/{len(test)} ({ok / max(len(test), 1):.0%})")
for r, name, tr in test:
    got = names.get(tr["id"], {})
    if got.get("name") != name:
        print(f"   rally {r['index']} t {r['start_t']:.0f}: overlay {name}, ours {got.get('name')} (conf {got.get('conf')}, kit {tr['kit']}, h {tr.get('height') and round(tr['height'], 1)})")
I.save(run, trk, names)
