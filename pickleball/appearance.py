"""When do the players change ends? Observed from what the near player looks like, rally by rally.

Shirts can match (Las Vegas: both white tops) and TV edits can hide the break where players switch,
so neither colour-of-the-shirt nor long breaks are reliable. Upper body (shirt) or lower body (shorts /
skirt) usually differs. For every rally: brightness / saturation / colour shares of the near player's
upper and lower body over a few frames; the rallies split into two looks (2-means); an end change is
where the near look flips and stays flipped. If the two looks are not clearly apart, returns None
(the caller then falls back to the rules-only search in outcome.reconstruct).
Cached in <run>/appearance.json.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from .players import select_players


def _region_stats(img, y_a, y_b, x1, x2):
    crop = img[max(int(y_a), 0):max(int(y_b), int(y_a) + 5), max(int(x1), 0):int(x2)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(float)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    col = s > 40
    return [np.median(v) / 255, np.median(s) / 255,
            np.mean(((h < 10) | (h > 150)) & col),     # red / pink share
            np.mean((h > 85) & (h < 130) & col),       # blue share
            np.mean((h > 20) & (h < 40) & col)]        # yellow share


def _look(img, p):
    """Colour stats of the upper body (shoulders->hips) and lower body (hips->knees)."""
    kp = np.asarray(p["kp"])
    x1, y1, x2, y2 = p["box"]
    ys = lambda idx, frac: (np.mean([kp[i, 1] for i in idx if kp[i, 2] > 0.3])
                            if any(kp[i, 2] > 0.3 for i in idx) else y1 + frac * (y2 - y1))
    sh, hip, knee = ys((5, 6), 0.2), ys((11, 12), 0.5), ys((13, 14), 0.75)
    xa, xb = x1 + 0.2 * (x2 - x1), x2 - 0.2 * (x2 - x1)
    up, low = _region_stats(img, sh, hip, xa, xb), _region_stats(img, hip, knee, xa, xb)
    return np.array(up + low) if up is not None and low is not None else None


def near_looks(run_dir, rallies: list[dict], recs: list[dict] | None = None, per_rally: int = 6) -> list:
    """Lower-body look of the near player for each rally (time order); None where not seen."""
    run_dir = Path(run_dir)
    cache = run_dir / "appearance.json"
    key = [round(r["start_t"], 2) for r in rallies]
    if cache.exists():
        c = json.loads(cache.read_text())
        if c.get("key") == key and c.get("version") == 2:
            return [np.array(x) if x is not None else None for x in c["looks"]]
    meta = json.loads((run_dir / "meta.json").read_text())
    fps = meta["fps"]
    if recs is None:
        recs = [json.loads(l) for l in (run_dir / "detections.jsonl").open()]
    R = {r["f"]: r for r in recs if r.get("court")}
    cap = cv2.VideoCapture(meta["video"])
    looks = []
    for r in rallies:
        f0, f1 = int(r["start_t"] * fps), int(r["end_t"] * fps)
        fs = [f for f in range(f0, f1 + 1, max(1, (f1 - f0) // per_rally)) if f in R][:per_rally]
        vals = []
        for f in fs:
            sel = {p["side"]: p for p in select_players(R[f].get("people", []), 1)}
            if "near" not in sel:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if ok:
                x = _look(img, sel["near"])
                if x is not None:
                    vals.append(x)
        looks.append(np.median(vals, 0) if vals else None)
    cache.write_text(json.dumps({"version": 2, "key": key, "looks": [x.tolist() if x is not None else None for x in looks]}))
    return looks


def end_changes(looks: list, min_sep: float = 2.5, persist: int = 2) -> set | None:
    """Rally indices k after which the players changed ends; None if the two looks are not separable.

    Robust 1-D split: features scaled by median / MAD and clipped (one odd frame can't dominate), projected
    on their main direction of variation (skirt brightness in one match, shirt colour in another), then
    split into two groups. Separable = the group means are >= min_sep pooled standard deviations apart."""
    idx = [k for k, x in enumerate(looks) if x is not None]
    if len(idx) < 4:
        return None
    X = np.stack([looks[k] for k in idx])
    med = np.median(X, 0)
    mad = np.median(np.abs(X - med), 0) * 1.4826 + 1e-3
    Z = np.clip((X - med) / mad, -3, 3)
    u = np.linalg.svd(Z - Z.mean(0), full_matrices=False)[2][0]
    p = Z @ u
    c = np.percentile(p, [10, 90]).astype(float)
    for _ in range(50):
        lab = (np.abs(p - c[1]) < np.abs(p - c[0])).astype(int)
        if lab.min() == lab.max():
            return None
        c = np.array([p[lab == 0].mean(), p[lab == 1].mean()])
    sd = np.sqrt(((p - c[lab]) ** 2).mean()) + 1e-6
    if abs(c[1] - c[0]) / sd < min_sep:
        return None  # the two players look too alike from behind
    seq = list(zip(idx, lab))
    out, cur = set(), seq[0][1]
    for i in range(1, len(seq)):
        if seq[i][1] != cur and all(l == seq[i][1] for _, l in seq[i:i + persist]):
            out.add(seq[i - 1][0])
            cur = seq[i][1]
    return out
