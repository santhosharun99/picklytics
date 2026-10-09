"""Read the TNPPL broadcast score overlay: team scores, server name and serve number.

Layout (top-left): title bar, a row per team (team name, players, score cell, lime 'games' cell) and a bottom
line 'SERVE . <NAME> . SERVE NO <1|2>'. Two placements exist (match 1 sits ~100 px lower), so every read
anchors on the two lime games cells and takes the score cells / serve line at fixed offsets from them.
Row 0 = Kodai Tigers (the team listed first), row 1 = the opponent.

Score digits: a small classifier (models/tnppl_digits.joblib) trained on ~12k digit glyphs harvested from all
8 videos and labelled by cluster (agrees with the cluster labels on 12329/12330). The serve line is plain
text: tesseract reads it reliably.
"""
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

MODEL = Path(__file__).resolve().parents[1] / "models" / "tnppl_digits.joblib"


@lru_cache(1)
def _clf():
    import joblib
    return joblib.load(MODEL)


def anchor(img):
    """(gx, row0_y, row1_y) of the lime games cells, or None when the overlay is not on screen."""
    hsv = cv2.cvtColor(img[0:400, 0:800], cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= 25) & (hsv[..., 0] <= 40) & (hsv[..., 1] > 150) & (hsv[..., 2] > 180)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    b = sorted([tuple(st[i][:4]) for i in range(1, n)
                if 30 < st[i][2] < 70 and 25 < st[i][3] < 60 and st[i][0] > 300], key=lambda b: b[1])
    if len(b) < 2 or abs(b[0][0] - b[1][0]) > 4 or not 36 < b[1][1] - b[0][1] < 50:
        return None
    return b[0][0], b[1][1] - 42, b[1][1]


def _digit_glyphs(img, a, row):
    gx, y0, y1 = a
    y = (y0, y1)[row]
    m = (img[y - 2:y + 38, gx - 54:gx - 2].min(axis=2) > 170).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(xs) < 40:
        return None
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    w = m.shape[1]
    lo, hi = int(w * 0.35), int(w * 0.65) + 1
    s = lo + int(np.argmin(m.sum(0)[lo:hi]))  # two zero-padded digits; split at the thinnest column
    out = []
    for part in (m[:, :s], m[:, s:]):
        part = _strokes(part)
        cols = np.nonzero(part.sum(0))[0]
        if len(cols) == 0:
            return None
        out.append(cv2.resize(part[:, cols.min():cols.max() + 1] * 255, (12, 20), interpolation=cv2.INTER_AREA))
    return out


def _strokes(part):
    """Keep only the digit's tall components; stray specks (from the split or compression) shift the crop."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(part.astype(np.uint8))
    keep = [i for i in range(1, n) if st[i][3] >= 0.5 * part.shape[0]]
    return np.isin(lab, keep).astype(np.uint8) if keep else part


def _score(img, a, row):
    g = _digit_glyphs(img, a, row)
    if g is None:
        return None, 0.0
    X = np.array([x.ravel() / 255.0 for x in g])
    p = _clf().predict_proba(X)
    d = p.argmax(1)
    return int(d[0] * 10 + d[1]), float(p.max(1).min())


def _serve_line(img, a):
    gx, _, y1 = a
    g = cv2.cvtColor(img[y1 + 43:y1 + 66, gx - 313:gx - 25], cv2.COLOR_BGR2GRAY)
    big = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.copyMakeBorder(bw, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
        cv2.imwrite(tf.name, bw)
        txt = subprocess.run(["tesseract", tf.name, "-", "--psm", "7", "-c",
                              "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -."],
                             capture_output=True, text=True).stdout.strip()
    # the SSS v TEH bug is a few px wider: the crop clips the first 'S' ('-ERVE - NAME - SERVE NO 2')
    m = re.search(r"(?:S|5|-|\b)ERVE\W+([A-Z][A-Z .]*?)\W+SERVE\s*N[O0]\s*([12])?", txt)
    if not m:
        return None, None, txt
    return m.group(1).strip(), (int(m.group(2)) if m.group(2) else None), txt


def read(img, serve_line: bool = True) -> dict | None:
    """{'scores': [row0, row1], 'conf': min digit prob, 'server': NAME|None, 'serve_no': 1|2|None} or None."""
    if img.shape[1] != 1920:
        img = cv2.resize(img, (1920, 1080))
    a = anchor(img)
    if a is None:
        return None
    (s0, c0), (s1, c1) = _score(img, a, 0), _score(img, a, 1)
    if s0 is None or s1 is None:
        return None
    out = {"scores": [s0, s1], "conf": round(min(c0, c1), 3), "server": None, "serve_no": None}
    if serve_line:
        out["server"], out["serve_no"], _ = _serve_line(img, a)
    return out


def stable_reading(cap, f0: int, f1: int, n: int = 7, min_conf: float = 0.6) -> dict | None:
    """Most common confident reading among n frames spread over [f0, f1] (cap = an open cv2.VideoCapture)."""
    rs = []
    for f in np.linspace(f0, f1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if ok:
            r = read(img)
            if r and r["conf"] >= min_conf:
                rs.append(r)
    if not rs:
        return None
    keys = [tuple(r["scores"]) for r in rs]
    best = max(set(keys), key=keys.count)
    srv = [(r["server"], r["serve_no"]) for r in rs if tuple(r["scores"]) == best and r["server"]]
    server, serve_no = max(set(srv), key=srv.count) if srv else (None, None)
    return {"scores": list(best), "server": server, "serve_no": serve_no, "votes": keys.count(best), "n": len(rs)}


def monotonic_fit(values: list, weights: list | None = None, vmax: int = 40) -> list[int]:
    """Best non-decreasing sequence (TNPPL matches are one game, so each team's score only goes up) that
    agrees with the most readings; values may contain None. Fixes digit misreads such as 6 -> 8 or 0."""
    T = len(values)
    if T == 0:
        return []
    w = weights or [1.0] * T
    INF = float("inf")
    cost = [0.0] * (vmax + 1)
    back = []
    for t in range(T):
        # best previous value <= v (prefix minimum)
        pm, arg, pref = INF, 0, []
        for v in range(vmax + 1):
            if cost[v] < pm:
                pm, arg = cost[v], v
            pref.append((pm, arg))
        back.append([p[1] for p in pref])
        cost = [pref[v][0] + (0 if values[t] is None or values[t] == v else w[t]) for v in range(vmax + 1)]
    v = min(range(vmax + 1), key=lambda k: cost[k])
    out = [v]
    for t in range(T - 1, 0, -1):
        v = back[t][v]
        out.append(v)
    return out[::-1]
