"""Split the PPA Europe full-day court feed (outputs/europe_feed_1) into its matches, from the broadcast overlay.

    PYTHONPATH=. python3 training/europe_split.py
The overlay (bottom left) shows the two teams; the yellow box with the current game score is there only during
match play (warm-ups show names only), and a light-blue dot marks the serving team's row. So:
  match window = stretch where the score box is visible (sampled every 5 s; gaps < 4 min = game breaks);
  per rally: serving row (0 = top team, 1 = bottom team) at the serve -> later names the kits.
Writes outputs/europe_m<k>/ (analysis.json filtered to the window's rallies, the rest symlinked to the feed run)
+ overlay_serve.json {rally index: row}. Teams / results read off the overlay by eye (2026-10-08).
"""
import json, os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEED = ROOT / "outputs" / "europe_feed_1"
VIDEO = ROOT / "data" / "videos" / "PPA_Europe_Court_feed_1.mp4"
MATCHES = [  # in feed order; read off the overlay
    {"id": "europe_m1", "teams": ["Kanichova / Biedermann", "Alvarez / Mandon"],
     "rosters": [["V. Kanichova", "M. Biedermann"], ["D. Alvarez", "G. Mandon"]],
     "official": "Kanichova / Biedermann won 11-8, 2-11, 11-9 (score overlay)"},
    {"id": "europe_m2", "teams": ["O'Donoghue / Platel", "Serrano Ruano / Pintor Rodriguez"],
     "rosters": [["M. O'Donoghue", "T. Platel"], ["A. Serrano Ruano", "M. Pintor Rodriguez"]],
     "official": "O'Donoghue / Platel won 8-11, 11-2, 11-0 (score overlay)"},
]
ROI = (slice(890, 1000), slice(0, 560))  # overlay box at 1920x1080


def overlay_state(img):
    hsv = cv2.cvtColor(img[ROI], cv2.COLOR_BGR2HSV)
    yel = (hsv[..., 0] > 20) & (hsv[..., 0] < 35) & (hsv[..., 1] > 150) & (hsv[..., 2] > 180)
    blu = (hsv[..., 0] > 90) & (hsv[..., 0] < 110) & (hsv[..., 1] > 60) & (hsv[..., 2] > 180)
    score = int(yel[:, 450:500].sum()) > 1000
    # the dot sits right after the team names: x ~260-360 for short names, up to ~430 for long ones; a wide strip
    # can catch other blue for some names -> read narrow and wide, keep a decisive answer, drop conflicts
    def read(x0, x1):
        r0, r1 = int(blu[15:55, x0:x1].sum()), int(blu[60:105, x0:x1].sum())
        return 0 if r0 > 50 and r0 > 3 * r1 else (1 if r1 > 50 and r1 > 3 * r0 else None)
    rows = {x for x in (read(250, 360), read(250, 445), read(360, 445)) if x is not None}
    return score, (rows.pop() if len(rows) == 1 else None)


def overlay_full(img):
    """(score shown, serving row 0/1/None, serve dots 1/2/None, game 1-3/None) - finished games are score columns
    left of the yellow box: x 419-435 = game 1 done, x 369-385 = game 2 done (only checked once the first exists:
    in game 1 long team names reach x ~400)."""
    score, row = overlay_state(img)
    if not score:
        return score, row, None, None
    hsv = cv2.cvtColor(img[ROI], cv2.COLOR_BGR2HSV)
    digit = ((hsv[..., 1] < 60) & (hsv[..., 2] > 200)) | ((hsv[..., 0] > 20) & (hsv[..., 0] < 35) & (hsv[..., 1] > 150) & (hsv[..., 2] > 180))
    colA, colB = int(digit[10:105, 419:436].sum()) > 20, int(digit[10:105, 369:386].sum()) > 20
    game = 1 + int(colA) + int(colA and colB)
    dots = None
    if row is not None:
        blu = (hsv[..., 0] > 90) & (hsv[..., 0] < 110) & (hsv[..., 1] > 60) & (hsv[..., 2] > 180)
        band = blu[(15, 60)[row]:(55, 105)[row], 250:445].any(0)
        dots = int(np.sum(band[1:] & ~band[:-1]) + band[0]) or None  # separate blobs along x
        dots = dots if dots in (1, 2) else None
    return score, row, dots, game


def main():
    A = json.loads((FEED / "analysis.json").read_text())
    cap = cv2.VideoCapture(str(VIDEO)); fps = cap.get(cv2.CAP_PROP_FPS)

    def at(t):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, img = cap.read()
        return overlay_state(img) if ok else (False, None)
    # 1. score-box timeline -> windows
    end = A["meta"]["last_frame"] / fps
    on = [t for t in np.arange(0, end, 5.0) if at(t)[0]]
    wins = []
    for t in on:
        if wins and t - wins[-1][1] < 240:
            wins[-1][1] = t
        else:
            wins.append([t, t])
    wins = [w for w in wins if w[1] - w[0] > 600]  # a match lasts > 10 min
    print("score windows:", [(round(a), round(b)) for a, b in wins])
    assert len(wins) == len(MATCHES), "expected one score window per match"
    # 2. per match run dir
    for m, (w0, w1) in zip(MATCHES, wins):
        rs = [r for r in A["rallies"] if w0 - 30 <= r["start_t"] <= w1 + 5]
        idx = {r["index"] for r in rs}
        t0, t1 = min(r["start_t"] for r in rs) - 1, max(r["end_t"] for r in rs) + 1
        B = {**A, "rallies": rs, "shots": [s for s in A["shots"] if s["rally"] in idx],
             "bounces": [b for b in A["bounces"] if t0 <= b["t"] <= t1],
             "positions": [p for p in A["positions"] if t0 <= p["t"] <= t1],
             "segments": [{"start_t": t0, "end_t": t1}],
             "summary": {**A["summary"], "rallies": len(rs)}, "split": {"feed": "europe_feed_1", "window_s": [w0, w1]}}
        d = ROOT / "outputs" / m["id"]; d.mkdir(exist_ok=True)
        (d / "analysis.json").write_text(json.dumps(B))
        for f in FEED.iterdir():
            if f.name != "analysis.json" and not (d / f.name).exists():
                os.symlink(f, d / f.name)
        serve = {}
        for r in rs:
            rows = [at(r["start_t"] + dt)[1] for dt in (-0.5, 0.3)]
            rows = [x for x in rows if x is not None]
            serve[r["index"]] = rows[0] if rows and all(x == rows[0] for x in rows) else None
        (d / "overlay_serve.json").write_text(json.dumps(serve))
        print(m["id"], f"{len(rs)} rallies {t0 / 60:.1f}-{t1 / 60:.1f} min, serving row read on {sum(v is not None for v in serve.values())}")


if __name__ == "__main__":
    main()
