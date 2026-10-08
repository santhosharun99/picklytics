"""Held-out ground-truth clips for ball tracking (docs/BALL_RESEARCH_PLAN.md, section 1).

    PYTHONPATH=. python3 training/build_balltruth.py
One random real rally (>= 6 detected touches) per run, serve - 1 s to last touch + 1.5 s (max 12 s); every 2nd frame
saved as 1280x720 JPEG in data/balltruth/<clip>/<frame>.jpg, with the current system's proposal (tracked ball) and the
detector's top candidates for the labelling page (dashboard/balltruth.html). Queue: labels/balltruth/queue.json.
Clips are drawn at random (seeded), not hand-picked. One video at a time (laptop heat).
"""
import json, random
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"tnppl_2_WD": None, "tnppl_3_MD": None, "tnppl_4_XD": None, "tnppl_5_MD50": None, "tnppl_6_MD": "held-out match",
        "tnppl_7_TeamDoubles": None, "tnppl_1_MS": None, "lv_waters_johns": "held-out match", "lv_bright_patriquin": None,
        "lv_fahey_yoshitomi": None, "full_match": None, "europe_feed_1": "held-out match"}
PROXY = {"lv_fahey_yoshitomi": "data/videos/Fahey_Yoshitomi_LV2026.mp4", "full_match": "data/videos/Mens_Singles_GOLD_2026_US_Open.mp4"}
OUT_W, OUT_H, STEP = 1280, 720, 2
rng = random.Random(2026)
queue = []
for rid, note in RUNS.items():
    run = ROOT / "outputs" / rid
    A = json.loads((run / "analysis.json").read_text())
    fps = A["meta"]["fps"]
    rep = json.loads((run / "replays.json").read_text())["rallies"] if (run / "replays.json").exists() else {}
    cands = []
    for r in A["rallies"]:
        sh = [s for s in A["shots"] if s["rally"] == r["index"] and not s.get("inferred")]
        if len(sh) >= 6 and not rep.get(str(r["index"]), {}).get("replay"):
            cands.append((r, sorted(sh, key=lambda s: s["frame"])))
    r, sh = rng.choice(cands)
    f0 = max(0, sh[0]["frame"] - int(1.0 * fps))
    f1 = min(sh[-1]["frame"] + int(1.5 * fps), f0 + int(12 * fps))
    frames = list(range(f0, f1 + 1, STEP))
    want = set(frames)
    ball = {int(k): v for k, v in json.loads((run / "render_data.json").read_text())["ball"].items()}
    cand, main = {}, set()
    with (run / "detections.jsonl").open() as fh:
        for line in fh:
            f = int(line[line.index(":") + 1:line.index(",")])
            if f in want:
                rec = json.loads(line)
                cand[f] = sorted(rec.get("balls", []), key=lambda b: -b[4])[:3]
                if rec.get("court") and not rec.get("cut"):
                    main.add(f)  # main camera only - close-ups / replays are not tracked
            elif f > f1:
                break
    clip = f"{rid}_r{r['index']}"
    d = ROOT / "data" / "balltruth" / clip
    d.mkdir(parents=True, exist_ok=True)
    video = PROXY.get(rid, A["meta"]["video"])
    cap = cv2.VideoCapture(str(ROOT / video))
    sx = OUT_W / A["meta"]["width"]
    sy = OUT_H / A["meta"]["height"]
    items = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    f = f0
    while f <= f1:
        ok, img = cap.read()
        if not ok:
            break
        if f in want and f in main:
            cv2.imwrite(str(d / f"{f}.jpg"), cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
            b = ball.get(f)
            items.append({"frame": f, "proposal": [round(b[0] * sx, 1), round(b[1] * sy, 1)] if b else None,
                          "proposal_interp": bool(b and b[2]),
                          "cands": [[round((c[0] + c[2]) / 2 * sx, 1), round((c[1] + c[3]) / 2 * sy, 1), round(c[4], 2)] for c in cand.get(f, [])]})
        f += 1
    queue.append({"clip": clip, "run": rid, "rally": r["index"], "note": note, "fps": fps, "step": STEP,
                  "size": [OUT_W, OUT_H], "frames": items})
    print(f"{clip}: {len(items)} frames ({(f1 - f0) / fps:.1f} s){' - ' + note if note else ''}", flush=True)
(ROOT / "labels" / "balltruth").mkdir(parents=True, exist_ok=True)
(ROOT / "labels" / "balltruth" / "queue.json").write_text(json.dumps({"clips": queue}))
print("total frames:", sum(len(c["frames"]) for c in queue))
