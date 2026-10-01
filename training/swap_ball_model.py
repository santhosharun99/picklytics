"""A/B a ball model: copy a run, re-detect only the ball on its main-view frames.

    python training/swap_ball_model.py outputs/<run> path/to/ball_model.pt outputs/<new_run> [--imgsz 640] [--class-id 0]

Court, players and everything else are kept, so differences come from the ball model only.
Then: python analyze.py x --out outputs/<new_run> --skip-detect --no-render && python training/evaluate.py ...
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

DEVICE = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--class-id", type=int, default=None, help="only keep this class (multi-class models)")
    args = ap.parse_args()

    src, dst = Path(args.run), Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "meta.json", dst / "meta.json")
    meta = json.loads((src / "meta.json").read_text())
    model = YOLO(args.model)

    cap = cv2.VideoCapture(meta["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, meta["first_frame"])
    cur = meta["first_frame"]
    with (src / "detections.jsonl").open() as fin, (dst / "detections.jsonl").open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            f = rec["f"]
            while cur < f:  # skip frames without reading the model
                cap.grab()
                cur += 1
            ok, frame = cap.read()
            cur += 1
            if ok and rec.get("court") is not None:
                r = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=DEVICE, verbose=False)[0]
                balls = []
                for xyxy, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                    if args.class_id is None or int(k) == args.class_id:
                        balls.append([round(float(v), 1) for v in xyxy] + [round(float(c), 3)])
                rec["balls"] = balls
            fout.write(json.dumps(rec, separators=(",", ":")) + "\n")
    print("wrote", dst / "detections.jsonl")


if __name__ == "__main__":
    main()
