"""Analyse a pickleball match video.

    python analyze.py VIDEO [--start 590 --end 650] [--players 2] [--names P1=Boyer,P2=Devilliers]
    python analyze.py VIDEO --skip-detect      # re-run analysis on cached detections (fast)
    python analyze.py VIDEO --no-render

Steps: 1) detect (models, slow, cached)  2) analyse (tracks, shots, stats)  3) render video.
Outputs go to outputs/<video-name>[_<start>-<end>]/.
"""
import argparse
import json
import re
from pathlib import Path

from pickleball.analysis import run_analysis
from pickleball.detect import run_detection
from pickleball.render import render_video

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--end", type=float, default=None, help="seconds")
    ap.add_argument("--players", type=int, default=2, choices=[2, 4], help="2 = singles, 4 = doubles")
    ap.add_argument("--names", default="", help="e.g. P1=Boyer,P2=Devilliers")
    ap.add_argument("--out", default=None)
    ap.add_argument("--court-model", default=str(ROOT / "models" / "court_detector.pt"))
    ap.add_argument("--ball-model", default=str(ROOT / "models" / "ball_detector.pt"))
    ap.add_argument("--pose-model", default="yolo11m-pose.pt")
    ap.add_argument("--ball-imgsz", type=int, default=960)
    ap.add_argument("--skip-detect", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(args.video).stem).strip("_")[:60]
    if args.start or args.end:
        stem += f"_{int(args.start)}-{int(args.end) if args.end else 'end'}"
    out_dir = Path(args.out) if args.out else ROOT / "outputs" / stem
    names = dict(kv.split("=", 1) for kv in args.names.split(",") if "=" in kv)

    if not args.skip_detect:
        print("[1/3] detecting ...")
        run_detection(args.video, out_dir, args.court_model, args.ball_model, args.pose_model,
                      args.start, args.end, args.ball_imgsz)
    print("[2/3] analysing ...")
    res = run_analysis(out_dir, args.players, names)
    print(json.dumps(res["summary"], indent=2))
    for pid, s in res["players"].items():
        print(f"  {s['name']}: {s['shots']} shots, {s['distance_ft']} ft covered, "
              f"kitchen {s['kitchen_time_pct']}%, hand={s['dominant_hand']}, types={s['shot_types']}")
    if not args.no_render:
        print("[3/3] rendering ...")
        print("Saved", render_video(out_dir))
    print("Output:", out_dir)


if __name__ == "__main__":
    main()
