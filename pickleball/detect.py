"""Pass 1: run the models over the video and cache raw per-frame detections.

Only frames where the court model finds the main camera view are analysed; close-ups,
crowd shots, replays and interviews are skipped (recorded as `"court": null`).
Output is one JSON object per frame in `<out>/detections.jsonl`.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .court import CourtDetector, CourtLock, in_play_area

DEVICE = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")

# COCO keypoint indices
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_ANKLE, R_ANKLE = 5, 6, 11, 12, 15, 16


def is_scene_cut(prev_small, small, threshold=0.45) -> bool:
    if prev_small is None:
        return True
    h1 = cv2.calcHist([prev_small], [0, 1], None, [16, 8], [0, 180, 0, 256])
    h2 = cv2.calcHist([small], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA) > threshold


def foot_point(box, kp) -> tuple[float, float]:
    """Ground contact point: ankle midpoint if visible, else bottom-center of the box."""
    ank = kp[[L_ANKLE, R_ANKLE]]
    vis = ank[:, 2] > 0.5
    if vis.any():
        return float(ank[vis, 0].mean()), float(ank[vis, 1].max())
    return float((box[0] + box[2]) / 2), float(box[3])


def plausible_height(view, court_xy, box, person_ft: float = 5.8) -> bool:
    """Box height must fit a standing/crouching adult at that spot on the court
    (rejects spectators' heads at the frame edge, reflections, partial bodies)."""
    x, y = float(court_xy[0]), float(court_xy[1])
    a, b = view.to_image([(x - 0.5, y), (x + 0.5, y)])
    px_per_ft = float(np.hypot(*(b - a)))
    ratio = (box[3] - box[1]) / max(person_ft * px_per_ft, 1.0)
    return 0.45 <= ratio <= 1.9


def torso_signature(hsv, box, kp) -> list[float]:
    """Colour signature of the shirt: hue histogram (16) + value histogram (4) of the torso."""
    pts = kp[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]]
    if (pts[:, 2] > 0.4).sum() >= 3:
        xs, ys = pts[pts[:, 2] > 0.4, 0], pts[pts[:, 2] > 0.4, 1]
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    else:  # fall back to the upper-middle part of the box
        bw, bh = box[2] - box[0], box[3] - box[1]
        x1, x2, y1, y2 = box[0] + 0.25 * bw, box[2] - 0.25 * bw, box[1] + 0.2 * bh, box[1] + 0.5 * bh
    h, w = hsv.shape[:2]
    x1, x2 = int(max(0, x1)), int(min(w, max(x2, x1 + 2)))
    y1, y2 = int(max(0, y1)), int(min(h, max(y2, y1 + 2)))
    patch = hsv[y1:y2, x1:x2]
    if patch.size == 0:
        return [0.0] * 20
    hue = cv2.calcHist([patch], [0], (patch[..., 1] > 50).astype(np.uint8), [16], [0, 180]).ravel()
    val = cv2.calcHist([patch], [2], None, [4], [0, 256]).ravel()
    hue = hue / (hue.sum() + 1e-6)
    val = val / (val.sum() + 1e-6)
    return [round(float(v), 3) for v in np.concatenate([hue, val])]


def run_detection(video: str, out_dir: Path, court_model: str, ball_model: str,
                  pose_model: str = "yolo11m-pose.pt", start: float = 0.0, end: float | None = None,
                  ball_imgsz: int = 960, court_every: int = 10, camera: dict | None = None,
                  pose_imgsz: int = 1280) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "detections.jsonl"

    model_court = CourtDetector(court_model, device=DEVICE) if Path(court_model).exists() else None
    if camera:  # calibrated fixed camera: line-snapping lock, model only as a fallback initialiser
        court = CourtLock({tuple(p["court"]): tuple(p["img"]) for p in camera["points"]}, fallback=model_court)
    else:
        court = model_court
    ball = YOLO(ball_model)
    pose = YOLO(pose_model)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(start * fps)
    last = min(n_total, int(end * fps)) if end else n_total
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    meta = {"video": str(video), "fps": fps, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "first_frame": first, "last_frame": last}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    prev_small, view, since_court = None, None, court_every
    t0, n_main = time.time(), 0
    with out_path.open("w") as fh:
        for f in range(first, last):
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
            cut = is_scene_cut(prev_small, small)
            prev_small = small
            if cut or since_court >= court_every:
                view = court.detect(frame)
                since_court = 0
            since_court += 1

            rec = {"f": f, "t": round(f / fps, 3), "cut": bool(cut), "court": None}
            if view is not None:
                n_main += 1
                rec["court"] = {"kp": np.round(view.keypoints, 1).tolist(),
                                "H": np.round(view.H, 8).tolist(), "err": round(view.reproj_error_ft, 3)}

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                people = []
                r = pose.predict(frame, conf=0.35, classes=[0], imgsz=pose_imgsz, device=DEVICE, verbose=False)[0]
                if r.keypoints is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    kps = r.keypoints.data.cpu().numpy()  # (n, 17, 3)
                    for box, c, kp in zip(boxes, confs, kps):
                        foot = foot_point(box, kp)
                        cxy = view.to_court([foot])[0]
                        if not in_play_area(cxy)[0] or not plausible_height(view, cxy, box):
                            continue
                        people.append({"box": np.round(box, 1).tolist(), "conf": round(float(c), 3),
                                       "kp": np.round(kp, 2).tolist(), "foot": [round(v, 1) for v in foot],
                                       "court": [round(float(v), 2) for v in cxy],
                                       "sig": torso_signature(hsv, box, kp)})
                rec["people"] = people

                b = ball.predict(frame, conf=0.1, imgsz=ball_imgsz, device=DEVICE, verbose=False)[0]
                rec["balls"] = [[round(float(v), 1) for v in xyxy] + [round(float(c), 3)]
                                for xyxy, c in zip(b.boxes.xyxy.cpu().numpy(), b.boxes.conf.cpu().numpy())]

            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            done = f - first + 1
            if done % 300 == 0:
                rate = done / (time.time() - t0)
                print(f"  {done}/{last - first} frames  {rate:.1f} fps  main-view {n_main / done:.0%}  "
                      f"eta {(last - first - done) / rate / 60:.1f} min", flush=True)
    cap.release()
    return out_path
