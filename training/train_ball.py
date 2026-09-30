"""Train the pickleball ball detector on the merged dataset.

    python training/train_ball.py                    # full training
    python training/train_ball.py --epochs 1 --fraction 0.05   # quick smoke test

The best weights are copied to models/ball_detector.pt.
"""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt", help="pretrained start point (yolo11n/s/m.pt, yolo26s.pt, ...)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640, help="the ball is tiny; raise to 960/1280 if source images are large")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0, help="use a fraction of the train set")
    ap.add_argument("--name", default="ball")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(args.model)
    model.train(
        data=str(ROOT / "data" / "merged" / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        device=device,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        patience=15,
        # Ball-friendly augmentation: no vertical flip (gravity), mild scale, keep mosaic
        # for small-object context but turn it off for the last epochs.
        fliplr=0.5,
        flipud=0.0,
        scale=0.3,
        mosaic=1.0,
        close_mosaic=10,
        plots=True,
    )

    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    if best.exists():
        (ROOT / "models").mkdir(exist_ok=True)
        shutil.copy(best, ROOT / "models" / "ball_detector.pt")
        print(f"Saved {ROOT / 'models' / 'ball_detector.pt'}")


if __name__ == "__main__":
    main()
