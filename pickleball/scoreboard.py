"""Read the broadcast scoreboard (score per player + who serves) to know who won each rally.

Point winner = the player whose score went up between two readings; if no score changed but
the serve moved to the other player, the receiver won (side-out scoring). Works for both
side-out and rally scoring.

Layout is per broadcast; configured in configs/cameras/<video>.json under "scoreboard":
  {"rows": [[y1, y2], [y1, y2]], "score_x": [x1, x2], "serve_x": [x1, x2], "names": ["Boyer", "Devilliers"]}
(1920x1080 pixels; row order = top row first).
"""
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def _ocr_digits(gray) -> int | None:
    """Score digits (0-99) in a small crop. Tesseract is shaky on lone digits, so a few
    page-segmentation modes / scales are tried and the first clean number wins."""
    for scale in (4, 3, 6):
        big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # dark text on white
        bw = cv2.copyMakeBorder(bw, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        with tempfile.NamedTemporaryFile(suffix=".png") as tf:
            cv2.imwrite(tf.name, bw)
            for psm in ("7", "8", "10", "13"):
                out = subprocess.run(["tesseract", tf.name, "-", "--psm", psm, "-c", "tessedit_char_whitelist=0123456789"],
                                     capture_output=True, text=True).stdout.strip()
                if out.isdigit() and len(out) <= 2:
                    return int(out)
    return None


def read(img, cfg) -> dict | None:
    """{'scores': [a, b], 'server': 0|1|None} or None when the scoreboard is not on screen."""
    if img.shape[1] != 1920:
        img = cv2.resize(img, (1920, 1080))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x1, x2 = cfg["score_x"]
    scores = []
    for y1, y2 in cfg["rows"]:
        cell = gray[y1:y2, x1:x2]
        if cell.max() < 150:  # no bright digit: board hidden
            return None
        scores.append(_ocr_digits(cell))
    if None in scores:
        return None
    sx1, sx2 = cfg["serve_x"]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    marks = []
    for y1, y2 in cfg["rows"]:
        c = hsv[y1:y2, sx1:sx2]
        marks.append(int((c[..., 2] > 120).sum()))  # bright ball icon next to the server
    server = int(np.argmax(marks)) if max(marks) > 15 and min(marks) < max(marks) / 3 else None
    return {"scores": scores, "server": server}


def read_at(video: str, frames, cfg) -> dict:
    """{frame: reading} for the given frames (skips frames where the board is hidden)."""
    cap = cv2.VideoCapture(video)
    out = {}
    for f in sorted(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            r = read(img, cfg)
            if r:
                out[f] = r
    return out


def stable_reading(video, f0, f1, cfg, n=5):
    """Most common reading among n frames spread over [f0, f1] (robust to OCR slips)."""
    rs = list(read_at(video, np.linspace(f0, f1, n).astype(int), cfg).values())
    if not rs:
        return None
    keys = [(tuple(r["scores"]), r["server"]) for r in rs]
    best = max(set(keys), key=keys.count)
    return {"scores": list(best[0]), "server": best[1]}


def point_winner(before, after) -> int | None:
    """Row index (0/1) of the rally winner from readings before and after it."""
    if not before or not after:
        return None
    if after["scores"] == [0, 0] and max(before["scores"]) >= 10:  # game ended: the leader won the last point
        return int(np.argmax(before["scores"]))
    d = [a - b for a, b in zip(after["scores"], before["scores"])]
    if d == [1, 0] or d == [0, 1]:
        return d.index(1)
    if d == [0, 0] and before["server"] is not None and after["server"] is not None and before["server"] != after["server"]:
        return after["server"]  # side out: receiver won and now serves
    return None
