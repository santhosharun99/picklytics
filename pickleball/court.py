"""Court geometry, court keypoint detection and image <-> court homography.

Court coordinates are in feet: x across the 20 ft width (0 = left sideline as seen
from the near baseline), y along the 44 ft length (0 = far baseline, 44 = near
baseline), net at y = 22, kitchen (non-volley zone) lines at y = 15 and y = 29.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

COURT_W, COURT_L = 20.0, 44.0
NET_Y = 22.0
KITCHEN_FAR_Y, KITCHEN_NEAR_Y = 15.0, 29.0

# Keypoint order of the court model (see training/train_court.py).
COURT_POINTS = np.array([
    [0, 0], [10, 0], [20, 0],      # 0-2 far baseline
    [20, 15], [20, 29],            # 3-4 right sideline at kitchen lines
    [20, 44], [10, 44], [0, 44],   # 5-7 near baseline
    [0, 29], [0, 15],              # 8-9 left sideline at kitchen lines
    [10, 15], [10, 29],            # 10-11 kitchen line centers
], dtype=np.float32)

COURT_LINES = [
    (0, 2), (7, 5),                # baselines
    (0, 7), (2, 5),                # sidelines
    (9, 3), (8, 4),                # kitchen lines
    (1, 10), (11, 6),              # center lines
]


@dataclass
class CourtView:
    """Result of court detection for one frame."""
    keypoints: np.ndarray       # (12, 2) image coords
    confidence: np.ndarray      # (12,) keypoint confidences
    H: np.ndarray               # image -> court (feet)
    H_inv: np.ndarray           # court (feet) -> image
    reproj_error_ft: float      # mean error of confident keypoints after projection

    def to_court(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)


class CourtDetector:
    """Detects the 12 court keypoints and fits a homography.

    A frame counts as the main (analysable) camera view only when enough keypoints
    are confident and the homography reprojects them within `max_error_ft`.
    """

    def __init__(self, model_path: str, device: str = "cpu", min_kpt_conf: float = 0.5,
                 min_points: int = 8, max_error_ft: float = 1.5):
        self.model = YOLO(model_path)
        self.device = device
        self.min_kpt_conf = min_kpt_conf
        self.min_points = min_points
        self.max_error_ft = max_error_ft

    def detect(self, frame) -> CourtView | None:
        r = self.model.predict(frame, conf=0.25, device=self.device, verbose=False)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return None
        best = int(r.boxes.conf.argmax())
        kpts = r.keypoints.xy[best].cpu().numpy()
        conf = r.keypoints.conf[best].cpu().numpy() if r.keypoints.conf is not None else np.ones(12)
        return fit_view(kpts, conf, self.min_kpt_conf, self.min_points, self.max_error_ft)


def fit_view(kpts, conf, min_kpt_conf=0.5, min_points=8, max_error_ft=1.5) -> CourtView | None:
    ok = conf >= min_kpt_conf
    if ok.sum() < min_points:
        return None
    H, inliers = cv2.findHomography(kpts[ok], COURT_POINTS[ok], cv2.RANSAC, 1.0)
    if H is None:
        return None
    proj = cv2.perspectiveTransform(kpts[ok].reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
    err = float(np.linalg.norm(proj - COURT_POINTS[ok], axis=1).mean())
    if err > max_error_ft:
        return None
    return CourtView(kpts, conf, H, np.linalg.inv(H), err)


def in_play_area(court_xy, margin_side=8.0, margin_back=14.0) -> np.ndarray:
    """True for court positions where a player could be during a rally."""
    p = np.asarray(court_xy).reshape(-1, 2)
    return ((p[:, 0] > -margin_side) & (p[:, 0] < COURT_W + margin_side)
            & (p[:, 1] > -margin_back) & (p[:, 1] < COURT_L + margin_back))


def court_zone(y: float) -> str:
    """Depth zone for a court y coordinate (feet)."""
    d = abs(y - NET_Y)
    if d <= 7:
        return "kitchen"
    if d <= 15:
        return "transition"
    return "baseline"
