"""Full 3D camera model from the court lines.

The court homography only maps the ground. To place the ball in the air we need the real
camera: focal length + position + orientation. With a fixed broadcast camera, the 12 court
keypoints (known 20x44 ft layout) pin these down: we search the focal length that lets
solvePnP reproduce the keypoints best (principal point at the image centre, no distortion).
Check: the net top (3 ft at posts, 34 in at centre) must land on the real net in the image.

World frame: x across the court (0..20 ft), y along it (0 far baseline .. 44 near), h = height
above the ground (ft). Internally OpenCV uses z = -h.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .court import COURT_POINTS


@dataclass
class Camera:
    K: np.ndarray       # 3x3 intrinsics
    rvec: np.ndarray    # Rodrigues rotation
    tvec: np.ndarray    # translation
    err_px: float       # mean reprojection error of the court keypoints

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix acting on (x, y, z=-h, 1)."""
        R, _ = cv2.Rodrigues(self.rvec)
        return self.K @ np.c_[R, self.tvec.reshape(3, 1)]

    @property
    def position(self) -> np.ndarray:
        """Camera centre as (x, y, h) in feet."""
        R, _ = cv2.Rodrigues(self.rvec)
        c = (-R.T @ self.tvec.reshape(3, 1)).ravel()
        return np.array([c[0], c[1], -c[2]])

    def project(self, xyh) -> np.ndarray:
        """(N, 3) court points with height -> (N, 2) pixels."""
        X = np.asarray(xyh, float).reshape(-1, 3) * [1, 1, -1]
        p, _ = cv2.projectPoints(X, self.rvec, self.tvec, self.K, None)
        return p.reshape(-1, 2)

    def to_json(self) -> dict:
        return {"K": self.K.tolist(), "rvec": self.rvec.ravel().tolist(), "tvec": self.tvec.ravel().tolist(),
                "err_px": self.err_px}

    @classmethod
    def from_json(cls, d: dict) -> "Camera":
        return cls(np.array(d["K"]), np.array(d["rvec"]), np.array(d["tvec"]), d["err_px"])


def calibrate(img_pts, size=(1920, 1080), world_pts=COURT_POINTS) -> Camera:
    """Camera from image positions of the court keypoints (rows matching world_pts)."""
    img = np.asarray(img_pts, np.float64)
    W = np.c_[np.asarray(world_pts, np.float64), np.zeros(len(img))]
    cx, cy = size[0] / 2, size[1] / 2

    def solve(f):
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        _, r, t = cv2.solvePnP(W, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        p, _ = cv2.projectPoints(W, r, t, K, None)
        return float(np.linalg.norm(p.reshape(-1, 2) - img, axis=1).mean()), K, r, t

    coarse = min(np.arange(0.4, 4.0, 0.02) * size[0], key=lambda f: solve(f)[0])
    fine = min(np.arange(coarse - 40, coarse + 40, 1.0), key=lambda f: solve(f)[0])
    e, K, r, t = solve(fine)
    return Camera(K, r, t, e)


def calibrate_from_run(run_dir) -> Camera:
    """Calibrate on the median court keypoints of a detection run (fixed camera)."""
    run = Path(run_dir)
    kps = [json.loads(l)["court"]["kp"] for l in (run / "detections.jsonl").open() if '"court": {' in l or '"court":{' in l]
    meta = json.loads((run / "meta.json").read_text())
    size = (meta.get("width", 1920), meta.get("height", 1080))
    return calibrate(np.median(np.array(kps), 0), size)


def net_top(n=11) -> np.ndarray:
    """Points along the net top: 36 in at the posts (x=-1, 21), 34 in at the centre."""
    x = np.linspace(-1, 21, n)
    h = 3.0 - (2 / 12) * (1 - np.abs(x - 10) / 11)
    return np.c_[x, np.full(n, 22.0), h]
