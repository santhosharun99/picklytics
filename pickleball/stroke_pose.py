"""Stroke-phase classifier from a single body pose (COCO-17 keypoints).

Trained on Roboflow 'Tennis Pose Estimation' (degree-n9l9m, CC BY 4.0): ~4.8k player poses labelled
Backhand / Forehand / Ready_Position / Serve. Used to tell a real swing from a player just moving
their arms: P(stroke) = 1 - P(ready).
"""
import numpy as np

CLASSES = ["Backhand", "Forehand", "Ready_Position", "Serve"]
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12


def pose_features(kp) -> np.ndarray | None:
    """17x3 keypoints (x, y, conf/vis) -> scale/position-invariant feature vector."""
    kp = np.asarray(kp, float)[:17]
    vis = kp[:, 2] > 0.3
    if not (vis[[L_SH, R_SH]].all() and vis[[L_HIP, R_HIP]].any()):
        return None
    sh = kp[[L_SH, R_SH], :2].mean(0)
    hp = kp[[L_HIP, R_HIP]][vis[[L_HIP, R_HIP]], :2].mean(0)
    scale = np.linalg.norm(sh - hp)
    if scale < 1e-3:
        return None
    xy = (kp[:, :2] - hp) / scale
    xy[~vis] = 0.0
    # a few explicit swing cues: wrists relative to shoulders, arm extension, torso twist
    extra = []
    for s, e, w in ((5, 7, 9), (6, 8, 10)):
        extra += [*(kp[w, :2] - kp[s, :2]) / scale, np.linalg.norm(kp[w, :2] - kp[s, :2]) / scale]
    extra.append((kp[R_SH, 0] - kp[L_SH, 0]) / scale)       # shoulder line width (rotation proxy)
    extra.append((kp[R_HIP, 0] - kp[L_HIP, 0]) / scale)
    return np.concatenate([xy.ravel(), vis.astype(float), extra])


class StrokeClassifier:
    def __init__(self, path):
        import joblib
        self.m = joblib.load(path)

    def p_stroke(self, kp) -> float | None:
        f = pose_features(kp)
        if f is None:
            return None
        p = self.m.predict_proba(f[None])[0]
        return float(1.0 - p[list(self.m.classes_).index("Ready_Position")])
