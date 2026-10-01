"""Learned swing detector: P(this player is making contact around this frame) from the pose
sequence (body shape + wrist/elbow motion over +-3 frames). Trained on analyst-labelled hits
(training/train_swing.py). Idea from pose-sequence tennis shot models; trained on pickleball."""
import numpy as np

from .stroke_pose import pose_features


def frame_features(K, side, f):
    """K: side -> {frame: (keypoints 17x3, box height)}"""
    if f not in K[side]:
        return None
    base = pose_features(K[side][f][0])
    if base is None:
        return None
    h = max(K[side][f][1], 1)
    mot = []
    for d in (-3, -2, -1, 1, 2, 3):
        g = f + d
        if g in K[side]:
            a, b = K[side][f][0], K[side][g][0]
            mot += [np.hypot(*(b[k, :2] - a[k, :2])) / h if a[k, 2] > .3 and b[k, 2] > .3 else 0 for k in (7, 8, 9, 10)]
        else:
            mot += [0] * 4
    return np.concatenate([base, mot, [1.0 if side == "near" else 0.0]])


def pose_index(players):
    K = {"near": {}, "far": {}}
    for f, ps in players.items():
        for p in ps:
            K[p["side"]][f] = (np.asarray(p["kp"]), p["box"][3] - p["box"][1], p)
    return K


def swing_series(players, model):
    """side -> {frame: P(swing)}"""
    K = pose_index(players)
    out = {"near": {}, "far": {}}
    for side in K:
        fr = sorted(K[side])
        X, keep = [], []
        for f in fr:
            ft = frame_features(K, side, f)
            if ft is not None:
                X.append(ft)
                keep.append(f)
        if X:
            p = model.predict_proba(np.array(X))[:, 1]
            out[side] = dict(zip(keep, p))
    return out, K
