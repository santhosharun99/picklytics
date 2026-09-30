"""Train the court keypoint detector (YOLO pose, 12 keypoints).

    python training/train_court.py
    python training/train_court.py --epochs 1 --fraction 0.1   # smoke test

Keypoint order in the Roboflow dataset (necromancer/pickleball-court), in court feet
(x across the 20 ft width, y along the 44 ft length, net at y=22):

    0 (0,0)    1 (10,0)   2 (20,0)      far baseline
    9 (0,15)  10 (10,15)  3 (20,15)     far kitchen line
    8 (0,29)  11 (10,29)  4 (20,29)     near kitchen line
    7 (0,44)   6 (10,44)  5 (20,44)     near baseline

The dataset ships an identity flip_idx, which is wrong for horizontal flips, so we
write our own data.yaml with the correct left/right swap.

The best weights are copied to models/court_detector.pt.
"""
import argparse
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "necromancer_pickleball_court"
FLIP_IDX = [2, 1, 0, 9, 8, 7, 6, 5, 4, 3, 10, 11]


def write_data_yaml() -> Path:
    cfg = {
        "path": str(RAW),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "kpt_shape": [12, 3],
        "flip_idx": FLIP_IDX,
        "names": {0: "court"},
    }
    out = ROOT / "data" / "court.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s-pose.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--name", default="court")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")
    YOLO(args.model).train(
        data=str(write_data_yaml()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        device=device,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        patience=20,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.5,  # the court should usually be seen whole
        degrees=3.0,
        perspective=0.0005,
        plots=True,
    )

    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    if best.exists():
        (ROOT / "models").mkdir(exist_ok=True)
        shutil.copy(best, ROOT / "models" / "court_detector.pt")
        print(f"Saved {ROOT / 'models' / 'court_detector.pt'}")


if __name__ == "__main__":
    main()
