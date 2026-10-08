"""Reports for the PPA Europe matches split out of the court feed (training/europe_split.py).

    PYTHONPATH=. python3 training/build_europe.py
Kits -> team names from the overlay's serving dot (overlay_serve.json, 0 = top team) vs the serving kit found by
doubles.reconstruct: the mapping that agrees on more rallies wins. PPA doubles: side-out scoring, best of 3 to 11.
"""
import json
from pathlib import Path

from pickleball.doubles import reconstruct
from pickleball.report import _load_named, build_report_doubles
from training.europe_split import MATCHES

ROOT = Path(__file__).resolve().parents[1]
RULES = {"scoring": "sideout", "target": 11, "win_by": 2, "games_to_win": 2}
VIDEO = "data/videos/PPA_Europe_Court_feed_1.mp4"
for m in MATCHES:
    run = ROOT / "outputs" / m["id"]
    serve = {int(k): v for k, v in json.loads((run / "overlay_serve.json").read_text()).items()}
    A = _load_named(run, None, True)
    rec = reconstruct(run, A, RULES["scoring"], RULES["target"], RULES["win_by"], RULES["games_to_win"])
    pairs = [(r["serve_kit"], serve.get(r["index"])) for r in rec if r.get("serve_kit") is not None and serve.get(r["index"]) is not None]
    agree = sum(k == row for k, row in pairs)
    kit0 = 0 if agree >= len(pairs) - agree else 1
    print(f"{m['id']}: serving kit vs overlay on {len(pairs)} rallies: kit 0 = top team agrees {agree}, disagrees {len(pairs) - agree} -> kit_of_team0 = {kit0}")
    r = build_report_doubles(run, VIDEO, ROOT / "outputs" / "reports" / f"{m['id']}.json", m["teams"], kit0, RULES,
                             official=m["official"], rosters=m["rosters"], replays=False)
    rs = r["rallies"]
    print(f"  {r['title']}: {len(rs)} rallies, {sum(x['n_shots'] for x in rs)} shots; "
          f"serving team vs overlay {sum(1 for x in rs if serve.get(x['id']) is not None and x['server'] == m['teams'][serve[x['id']]])}"
          f"/{sum(1 for x in rs if serve.get(x['id']) is not None)}")
