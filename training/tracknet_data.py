"""Training data for a TrackNet-style gap filler (3 frames in -> ball heat-map out), labelled for free.

    PYTHONPATH=. python3 training/tracknet_data.py outputs/<run> data/tracknet/<name>

For every rally frame (rally +-0.5 s): the frame resized to 512x288 (JPEG) and a label from our own
pipeline: the YOLO ball where it was measured, else the ball position from a TRUSTED 3D arc (reprojected
into the image), which covers frames where the ball is blurred or hidden. Frames with neither are
written without a label (excluded from the loss, used only as neighbours in a 3-frame window).
labels.csv: frame,rally,x,y,src  (x, y in 512x288 pixels; src = det | arc | none)
"""
import csv, json, sys
from pathlib import Path

import cv2
import numpy as np

from pickleball.analysis import load_detections
from pickleball.ball import track_ball
from pickleball.ball3d import match_arcs, positions
from pickleball.camera3d import calibrate_from_run
from pickleball.trajectory3d import reproject

W, H = 512, 288

run, out = Path(sys.argv[1]), Path(sys.argv[2])
(out / "frames").mkdir(parents=True, exist_ok=True)
meta, recs = load_detections(run)
fps, vw, vh = meta["fps"], meta["width"], meta["height"]
A = json.loads((run / "analysis.json").read_text())
cam = calibrate_from_run(run)
ball = track_ball(recs)
arcs = match_arcs(A, ball, cam.P)

windows = []
for r in A["rallies"]:
    windows.append((r["index"], int((r["start_t"] - 0.5) * fps), int((r["end_t"] + 0.5) * fps)))
arc_px = {}
for rid, aa in arcs.items():
    for a in aa:
        if not a.get("confident"):
            continue
        fr = np.arange(a["f0"], a["f1"] + 1)
        uv = reproject(cam.P, positions(np.asarray(a["theta"]), (fr - a["f0"]) / fps, a["beta"]))
        for f, p in zip(fr, uv):
            if 0 <= p[0] < vw and 0 <= p[1] < vh:
                arc_px[int(f)] = p

sx, sy = W / vw, H / vh
cap = cv2.VideoCapture(meta["video"])
rows, n_det, n_arc = [], 0, 0
for rid, f0, f1 in windows:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    for f in range(f0, f1 + 1):
        ok, img = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out / "frames" / f"{f}.jpg"), cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        v = ball.get(f)
        if v is not None and not v[3] and v[2] >= 0.3:
            rows.append([f, rid, round(v[0] * sx, 2), round(v[1] * sy, 2), "det"]); n_det += 1
        elif f in arc_px:
            p = arc_px[f]
            rows.append([f, rid, round(p[0] * sx, 2), round(p[1] * sy, 2), "arc"]); n_arc += 1
        else:
            rows.append([f, rid, "", "", "none"])
with (out / "labels.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["frame", "rally", "x", "y", "src"])
    w.writerows(rows)
print(f"{out}: {len(rows)} frames; labels from YOLO {n_det}, from trusted 3D arcs {n_arc}, none {len(rows) - n_det - n_arc}")
