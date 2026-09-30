"""Quick visual check: run the ball detector + player pose model on a video or image.

    python training/test_detect.py path/to/game.mp4
    python training/test_detect.py path/to/frame.jpg --conf 0.2

Writes an annotated copy to outputs/.
"""
import argparse
from collections import deque
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def annotate(frame, ball_model, pose_model, conf, imgsz, trail):
    players = pose_model.predict(frame, conf=0.4, classes=[0], device=DEVICE, verbose=False)[0]
    frame = players.plot(img=frame, boxes=True, labels=False, conf=False)

    balls = ball_model.predict(frame, conf=conf, imgsz=imgsz, device=DEVICE, verbose=False)[0]
    if len(balls.boxes):
        best = balls.boxes[balls.boxes.conf.argmax()]
        x1, y1, x2, y2 = best.xyxy[0].tolist()
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        trail.append((cx, cy))
        cv2.circle(frame, (cx, cy), 10, (0, 255, 255), 2)
        cv2.putText(frame, f"ball {float(best.conf):.2f}", (cx + 12, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        trail.append(None)
    pts = [p for p in trail if p]
    for a, b in zip(pts, pts[1:]):
        cv2.line(frame, a, b, (0, 200, 255), 2)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--ball-model", default=str(ROOT / "models" / "ball_detector.pt"))
    ap.add_argument("--pose-model", default="yolo11m-pose.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    args = ap.parse_args()

    ball_model, pose_model = YOLO(args.ball_model), YOLO(args.pose_model)
    src = Path(args.source)
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    trail = deque(maxlen=15)

    if src.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        out = out_dir / f"{src.stem}_detect.jpg"
        cv2.imwrite(str(out), annotate(cv2.imread(str(src)), ball_model, pose_model, args.conf, args.imgsz, trail))
        print("Saved", out)
        return

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = out_dir / f"{src.stem}_detect.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        writer.write(annotate(frame, ball_model, pose_model, args.conf, args.imgsz, trail))
        n += 1
        if n % 100 == 0:
            print(f"  {n} frames")
    cap.release()
    writer.release()
    print(f"Saved {out} ({n} frames)")


if __name__ == "__main__":
    main()
