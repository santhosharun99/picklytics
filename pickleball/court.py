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
# "Up" / at the kitchen line / at the net = within UP_FT of the net (measured, depth-corrected feet). Player positions
# pile up from the kitchen line (7 ft) to ~11 ft, peaking at 8-10 ft, at every venue; 10.5 ft misclassifies the
# least rally time (docs/DOUBLES.md "Up line"). Used by kitchen time, net arrival, contact zones and doubles "up".
UP_FT = 10.5

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


# ---------------------------------------------------------------------------
# Line-based court lock: a fixed broadcast camera is calibrated once from a few
# corner points, then every frame is checked (and slightly re-fitted) against the
# painted white lines. Robust main-view detection that does not depend on a model.
# ---------------------------------------------------------------------------

def _line_samples(step_ft: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Points along every court line (court feet) and each line's direction."""
    segs = [((0, 0), (20, 0)), ((0, 44), (20, 44)), ((0, 0), (0, 44)), ((20, 0), (20, 44)),
            ((0, 15), (20, 15)), ((0, 29), (20, 29)), ((10, 0), (10, 15)), ((10, 29), (10, 44))]
    pts, dirs = [], []
    for (x1, y1), (x2, y2) in segs:
        n = int(np.hypot(x2 - x1, y2 - y1) / step_ft)
        for t in np.linspace(0.02, 0.98, n):
            pts.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
            dirs.append((x2 - x1, y2 - y1))
    return np.array(pts, np.float32), np.array(dirs, np.float32)


_SAMPLES, _DIRS = _line_samples()


def white_mask(frame) -> np.ndarray:
    """Thin bright, unsaturated structures (painted court lines).
    Two tests: plainly white pixels, and - for lines in sun shadows, where they are no longer bright - thin
    structures clearly brighter than their surroundings (top-hat on brightness). Las Vegas doubles, heavy
    shadows: line match 0.35 -> 0.64 on main-camera frames; other camera angles stay <= 0.47 (< 0.55 needed)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 70) & (hsv[..., 2] > 165)).astype(np.uint8) * 255
    # keep thin structures only: white minus its opening with a large kernel
    big = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    lit = cv2.subtract(white, big)
    v = hsv[..., 2]
    tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, np.ones((15, 15), np.uint8))
    shade = ((tophat > 15) & (hsv[..., 1] < 130) & (v > 60)).astype(np.uint8) * 255
    return cv2.max(lit, shade)


def _project(H_inv, pts):
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H_inv).reshape(-1, 2)


def snap_to_lines(frame, H_inv: np.ndarray, search_px: int = 12, mask=None):
    """Find, for each line sample, the nearest white pixel along the line normal.

    Returns (court_pts, image_pts, fraction_found, mean_offset_px)."""
    mask = white_mask(frame) if mask is None else mask
    h, w = mask.shape
    img = _project(H_inv, _SAMPLES)
    ahead = _project(H_inv, _SAMPLES + 0.01 * _DIRS)
    d = ahead - img
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    normals = np.stack([-d[:, 1], d[:, 0]], 1)
    court, found, offs = [], [], []
    visible = 0
    offsets = np.arange(-search_px, search_px + 1)
    for (cx, cy), p, nrm in zip(_SAMPLES, img, normals):
        if not (0 <= p[0] < w and 0 <= p[1] < h):
            continue
        visible += 1
        q = (p[None, :] + offsets[:, None] * nrm[None, :]).round().astype(int)
        ok = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
        if not ok.any():
            continue
        hits = np.where(mask[q[ok, 1], q[ok, 0]] > 0)[0]
        if len(hits) == 0:
            continue
        o = offsets[ok][hits]
        best = o[np.argmin(np.abs(o))]
        court.append((cx, cy))
        found.append(p + best * nrm)
        offs.append(abs(best))
    frac = len(found) / max(visible, 1)
    return (np.array(court, np.float32), np.array(found, np.float32), frac,
            float(np.mean(offs)) if offs else float(search_px))


def refine_homography(frame, H_inv, iters: int = 3, search_px: int = 12):
    """Iteratively re-fit court->image homography to the white lines."""
    mask = white_mask(frame)
    frac = 0.0
    for i in range(iters):
        c, f, frac, _ = snap_to_lines(frame, H_inv, search_px if i == 0 else 6, mask)
        if len(c) < 20:
            return None, frac
        Hn, inl = cv2.findHomography(c, f, cv2.RANSAC, 3.0)
        if Hn is None:
            return None, frac
        H_inv = Hn
    _, _, frac, off = snap_to_lines(frame, H_inv, 4, mask)
    return H_inv, frac


class CourtLock:
    """Tracks a fixed broadcast camera's court via line snapping.

    `reference_points`: {(court_x, court_y): (img_x, img_y)} from a calibration frame.
    A frame is the main view when >= `min_frac` of visible court-line samples sit on
    white pixels after refinement.
    """

    def __init__(self, reference_points: dict, min_frac: float = 0.55, fallback: "CourtDetector | None" = None):
        src = np.array(list(reference_points.keys()), np.float32)
        dst = np.array(list(reference_points.values()), np.float32)
        self.H_ref_inv, _ = cv2.findHomography(src, dst)
        self.H_cur_inv = self.H_ref_inv.copy()
        self.min_frac = min_frac
        self.fallback = fallback

    def detect(self, frame) -> CourtView | None:
        cands = [self.H_cur_inv, self.H_ref_inv]
        if self.fallback is not None:
            v = self.fallback.detect(frame)
            if v is not None:
                cands.append(v.H_inv)
        for H_inv in cands:
            Hr, frac = refine_homography(frame, H_inv)
            if Hr is not None and frac >= self.min_frac:
                self.H_cur_inv = Hr
                kp = _project(Hr, COURT_POINTS)
                return CourtView(kp, np.ones(12, np.float32), np.linalg.inv(Hr), Hr, 1.0 - frac)
        return None
