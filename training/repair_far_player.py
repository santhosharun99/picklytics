"""Add the far player to an existing detection run where the full-frame pose pass missed them.

    PYTHONPATH=. python3 training/repair_far_player.py outputs/<run> [--every 1]

For every court-view frame with nobody beyond the net, runs the pose model on an enlarged crop of the
far baseline area (detect.add_far_player, the same step new runs do inline). The original file is kept
as detections.before_far.jsonl. Re-run the analysis afterwards (analyze.py --skip-detect, or run_analysis).
"""
import argparse, json, shutil, time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from pickleball.court import NET_Y, CourtView
from pickleball.detect import add_far_player

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--every", type=int, default=1, help="repair every n-th missing frame (1 = all)")
ap.add_argument("--pose-model", default="yolo11m-pose.pt")
a = ap.parse_args()
run = Path(a.run)
meta = json.loads((run / "meta.json").read_text())
src = run / "detections.jsonl"
backup = run / "detections.before_far.jsonl"
if not backup.exists():
    shutil.copy(src, backup)
recs = [json.loads(l) for l in backup.open()]
todo = [i for i, r in enumerate(recs) if r.get("court") and not any(p["court"][1] < NET_Y for p in r.get("people", []))]
todo = set(todo[:: a.every])
print(f"{len(todo)} court frames without a far player", flush=True)
pose = YOLO(a.pose_model)
cap = cv2.VideoCapture(meta["video"])
cap.set(cv2.CAP_PROP_POS_FRAMES, recs[0]["f"])
t0, done, added, pos = time.time(), 0, 0, recs[0]["f"]
for i, r in enumerate(recs):
    f = r["f"]
    while pos < f:
        cap.grab()
        pos += 1
    if i not in todo:
        cap.grab()
        pos += 1
        continue
    ok, img = cap.read()
    pos += 1
    if not ok:
        break
    H = np.asarray(r["court"]["H"])
    view = CourtView(np.asarray(r["court"]["kp"]), np.ones(12), H, np.linalg.inv(H), r["court"].get("err", 0.0))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    people = r.get("people", [])
    new = add_far_player(pose, img, hsv, view, people)
    if len(new) > len(people):
        added += 1
    r["people"] = new
    done += 1
    if done % 500 == 0:
        rate = done / (time.time() - t0)
        print(f"  {done}/{len(todo)}  far player added in {added}  ({rate:.1f} fr/s, eta {(len(todo) - done) / rate / 60:.0f} min)", flush=True)
with src.open("w") as fh:
    for r in recs:
        fh.write(json.dumps(r, separators=(",", ":")) + "\n")
print(f"done: far player added in {added}/{done} frames", flush=True)
