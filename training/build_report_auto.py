"""Build a fully automatic match report from an analysis run (no labels needed).

    PYTHONPATH=. python3 training/build_report_auto.py outputs/full_match \
        --video-url data/videos/Mens_Singles_GOLD_2026_US_Open.mp4 --out outputs/reports/boyer_devilliers_full.json
Point winners come from the scoreboard layout in configs/cameras/<video>.json ("scoreboard").
Open: http://localhost:8001/dashboard/match.html?report=boyer_devilliers_full
"""
import argparse, json, re
from pathlib import Path

from pickleball.report import build_report_auto

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--video-url", required=True, help="browser-playable video path relative to the project root")
ap.add_argument("--out", required=True)
ap.add_argument("--names", help="rename tracked players, e.g. P1=Ramakrishnan,P2=Rahul")
ap.add_argument("--rules", help='scoring rules as JSON, e.g. {"target": 11, "win_by": 1, "games_to_win": 1}')
ap.add_argument("--overlay", help='score-overlay cross-check as JSON: {"path", "teams", "rosters", "scoring", "singles"}')
ap.add_argument("--official", help="official result text, e.g. 'Ramakrishnan won 11-0 (match card)'")
ap.add_argument("--replays", action="store_true", help="drop broadcast replays (TNPPL: stinger before each replay)")
a = ap.parse_args()
names = dict(kv.split("=", 1) for kv in a.names.split(",")) if a.names else None
rules = json.loads(a.rules) if a.rules else None
meta = json.loads((Path(a.run) / "meta.json").read_text())
cam = ROOT / "configs" / "cameras" / (re.sub(r"[^A-Za-z0-9]+", "_", Path(meta["video"]).stem).strip("_")[:60] + ".json")
sb = json.loads(cam.read_text()).get("scoreboard") if cam.exists() else None
print("scoreboard config:", cam.name if sb else "none (winners unknown)")
r = build_report_auto(Path(a.run), meta["video"], a.video_url, Path(a.out), sb, names=names, rules=rules,
                     overlay=json.loads(a.overlay) if a.overlay else None, official=a.official, replays=a.replays)
print(r["title"], "-", r["source"])
for name, p in r["players"].items():
    print(f"\n== {name}")
    for k, v in p["summary"].items():
        print(f"  {k}: {v['value']} (n={v['n']})")
    for i in p["insights"][:6]:
        print(f"  [{i['kind']}{' *small sample*' if i['low_sample'] else ''}] {i['text']}")
