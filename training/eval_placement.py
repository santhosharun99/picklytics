"""Placement events vs hand tags (US Open: 16 tagged rallies, 45 tagged bounces with positions).

    PYTHONPATH=. python3 training/eval_placement.py
Builds the shot events WITHOUT the human tags (automatic sources only) and compares, on shots inside the tagged
rallies, with the human version: outgoing-flight status, observed-landing zone (same zone / boundary / different)
and distance. Matching: same shot (analysis hit frame); a tagged bounce counts for the shot whose window holds it.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from pickleball import rally_end
from pickleball.shot_events import build_run

ROOT = Path(__file__).resolve().parents[1]
run = ROOT / "outputs" / "full_match"
A = json.loads((run / "analysis.json").read_text())
dead = rally_end.dead_touches(A, run)
L = json.loads((ROOT / "labels/Mens_Singles_GOLD_2026_US_Open.json").read_text())
ranges = [(r["start"], r["end"]) for r in L["rallies"] if r.get("mode") == "hits+bounces" and r["end"] > r["start"]]
live = {}
for s in sorted(A["shots"], key=lambda s: s["frame"]):
    if (s["rally"], s["index"]) not in dead and not s.get("inferred"):
        live.setdefault(s["rally"], []).append(s)
H = build_run(run, A, live, use_human=True)
M = build_run(run, A, live, use_human=False)
tagged = [f for f in H if any(a <= f <= b for a, b in ranges)]
print(f"{len(tagged)} shots inside {len(ranges)} tagged rallies")
st = Counter((H[f]["status"], M[f]["status"]) for f in tagged)
print("status (human-tag version -> automatic):")
for (h, m), n in sorted(st.items(), key=lambda x: -x[1]):
    print(f"   {h:12s} -> {m:12s} {n}")
agree = sum(n for (h, m), n in st.items() if h == m)
print(f"status agreement {agree}/{len(tagged)} ({agree / len(tagged):.0%})")
both = [f for f in tagged if H[f]["landing"]["state"] == "observed" and M[f]["landing"]["state"] == "observed"]
hb = [f for f in tagged if H[f]["landing"]["state"] == "observed"]
d = [float(np.hypot(H[f]["landing"]["u"] - M[f]["landing"]["u"], H[f]["landing"]["d"] - M[f]["landing"]["d"])) for f in both]
zc = Counter("same zone" if H[f]["landing"]["zone"] == M[f]["landing"]["zone"] else
             ("automatic says boundary" if M[f]["landing"]["zone"] == "boundary" else "different zone") for f in both)
print(f"tagged landings {len(hb)}; automatic also found a landing for {len(both)} ({len(both) / max(len(hb), 1):.0%})")
if d:
    print(f"landing distance automatic vs tag: median {np.median(d):.2f} ft, 80% within {np.percentile(d, 80):.2f} ft; zones: {dict(zc)}")
last = {max(v["frame"] for v in shots) for shots in live.values()}
gap = [f for f in tagged if H[f]["status"] == "unresolved" and M[f]["status"] == "bounced"]
rng_end = lambda f: next(b for a, b in ranges if a <= f <= b)
after_tags = [f for f in gap if M[f]["landing"]["frame"] > max(e["frame"] for r in L["rallies"] if r.get("mode") == "hits+bounces" and r["start"] <= f <= r["end"] for e in r["events"])]
print(f"auto bounce where the tags have none: {len(gap)}; last shot of the rally: {sum(f in last for f in gap)}; "
      f"bounce after the rally's last tagged event: {len(after_tags)}")
ok = [f for f in tagged if f not in after_tags]
agree2 = sum(H[f]["status"] == M[f]["status"] for f in ok)
print(f"status agreement excluding bounces after the last tagged event: {agree2}/{len(ok)} ({agree2 / len(ok):.0%})")
