"""Score-overlay timeline for a TNPPL video: one reading per step (default 1 s), changes only.

    PYTHONPATH=. python3 training/overlay_timeline.py data/videos/TNPPL_S2_KTvRTT_1_MS.mp4 --out outputs/tnppl_1_MS/overlay.json
"""
import argparse, json
import cv2
from pickleball import tnppl_overlay as ov

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--step", type=float, default=1.0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
cap = cv2.VideoCapture(a.video)
fps, N = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
rows = []
f = 0
while f < N:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    if not ok:
        break
    r = ov.read(img)
    if r:
        rows.append({"t": round(f / fps, 2), **r})
    f += int(a.step * fps)
# one game per match: each team's score only goes up -> best non-decreasing fit fixes digit misreads
for i in (0, 1):
    fit = ov.monotonic_fit([r["scores"][i] for r in rows], [r["conf"] for r in rows])
    for r, v in zip(rows, fit):
        r.setdefault("fit", [None, None])[i] = v
json.dump({"video": a.video, "fps": fps, "step": a.step, "readings": rows}, open(a.out, "w"), indent=0)
prev = None
for r in rows:
    k = (tuple(r["fit"]), r["server"], r["serve_no"])
    if k != prev:
        print(f"{r['t']:7.1f}  {r['fit'][0]:2d}-{r['fit'][1]:<2d} (read {r['scores'][0]}-{r['scores'][1]}) conf {r['conf']:.2f}  {r['server']} #{r['serve_no']}")
        prev = k
print(len(rows), "readings")
