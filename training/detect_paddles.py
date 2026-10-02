"""Run the paddle detector on every frame of the labelled rallies of a run (cached).

    PYTHONPATH=. python3 training/detect_paddles.py outputs/<run> [...]
Writes <run>/paddles.json: {frame: [[x1, y1, x2, y2, conf], ...]}
"""
import json, sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
L = json.loads((ROOT / "labels" / "Mens_Singles_GOLD_2026_US_Open.json").read_text())
model = YOLO(str(ROOT / "models" / "paddle_detector.pt"))
for run in map(Path, sys.argv[1:]):
    meta = json.loads((run / "meta.json").read_text())
    frames = sorted({f for r in L["rallies"] if r["end"] and meta["first_frame"] <= r["start"] and r["end"] < meta["last_frame"]
                     for f in range(r["start"] - 10, r["end"] + 11)})
    cap = cv2.VideoCapture(meta["video"])
    out, pos = {}, None
    for i, f in enumerate(frames):
        if pos != f:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        pos = f + 1
        if not ok:
            continue
        b = model(img, imgsz=960, conf=0.1, verbose=False, device="mps")[0].boxes
        out[f] = [[*map(lambda v: round(float(v), 1), xy), round(float(c), 3)] for xy, c in zip(b.xyxy.tolist(), b.conf.tolist())]
        if i % 500 == 0:
            print(run.name, i, "/", len(frames), flush=True)
    (run / "paddles.json").write_text(json.dumps(out))
    print("done", run.name, len(out), "frames", flush=True)
