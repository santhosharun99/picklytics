"""Build a scouting report JSON from reviewed labels + tracking runs.

    python training/build_report.py labels/<video>.json outputs/<run1> [outputs/<run2> ...] \
        --video data/videos/<video>.mp4 --out outputs/reports/<name>.json
"""
import argparse
from pathlib import Path

from pickleball.report import build_report

ap = argparse.ArgumentParser()
ap.add_argument("labels")
ap.add_argument("runs", nargs="+")
ap.add_argument("--video", required=True, help="browser-playable video path relative to the project root")
ap.add_argument("--out", required=True)
a = ap.parse_args()
r = build_report(Path(a.labels), [Path(p) for p in a.runs], a.video, Path(a.out))
print(r["title"], "-", r["source"])
for name, p in r["players"].items():
    print(f"\n== {name}")
    for k, v in p["summary"].items():
        print(f"  {k}: {v['value']} (n={v['n']})")
    print("  third shot:", {k: (v['count'], v['won_pct']) for k, v in p["third_shot"].items()})
    print("  shot mix:", p["shot_mix"], " stroke:", p["stroke"], " zones:", p["zones"])
    print("  errors:", {k: v['count'] for k, v in p["error_shots"].items()}, " finishing:", {k: v['count'] for k, v in p["finishing_shots"].items()})
    print("  movement: depth", p["movement"]["avg_depth_ft"], "kitchen%", p["movement"]["kitchen_time_pct"], "return->kitchen", p["movement"]["return_to_kitchen_s"])
    for i in p["insights"]:
        print(f"  [{i['kind']}{' *small sample*' if i['low_sample'] else ''}] {i['text']}")
