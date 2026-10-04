"""Independent check of the ball detector on PPA broadcast frames it never saw in training
(Roboflow harshita-wafmq/pickleball-ball-tracking-9s7d6 v1, CC BY 4.0: 344 frames, stretched to 640x640).

    PYTHONPATH=. python3 training/eval_ball_external.py [models/ball_detector.pt]

Frames are un-stretched back to 16:9 (1138x640) and upscaled to 1920x1080, then run like the pipeline
(conf 0.1, imgsz 960). A ball counts as found if the best detection lies within 12 px (in 1080p) of the label.
"""
import glob, sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "raw" / "harshita_ball_tracking"


def main(model_path):
    m = YOLO(model_path)
    found = labelled = false_pos = empty = 0
    errs = []
    for img_path in sorted(glob.glob(str(DS / "*" / "images" / "*.jpg"))):
        lab = Path(img_path.replace("/images/", "/labels/")).with_suffix(".txt")
        rows = [list(map(float, l.split()[1:])) for l in lab.read_text().splitlines() if l.strip()] if lab.exists() else []
        img = cv2.resize(cv2.imread(img_path), (1920, 1080), interpolation=cv2.INTER_CUBIC)
        r = m.predict(img, conf=0.1, imgsz=960, verbose=False)[0]
        det = r.boxes.xywh.cpu().numpy()[:, :2] if len(r.boxes) else np.zeros((0, 2))
        if not rows:
            empty += 1; false_pos += len(det) > 0; continue
        for cx, cy, _, _ in rows:
            labelled += 1
            gt = np.array([cx * 1920, cy * 1080])
            d = np.linalg.norm(det - gt, axis=1).min() if len(det) else np.inf
            if d <= 12:
                found += 1; errs.append(d)
    print(f"{model_path}: ball found in {found}/{labelled} labelled frames ({100 * found / max(1, labelled):.0f}%), "
          f"median position error {np.median(errs):.1f} px (1080p); detections on {false_pos}/{empty} frames without a ball")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "models" / "ball_detector.pt"))
