"""Pass 3: render an annotated video (players, ball trail, hit/bounce markers, shot labels,
and a top-down court minimap). Only main-view frames are written. Output is H.264 via
ffmpeg so it plays in the browser dashboard.
"""
import json
import subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .court import COURT_L, COURT_W, KITCHEN_FAR_Y, KITCHEN_NEAR_Y, NET_Y

# BGR, matching the dashboard's categorical palette (blue, orange, aqua, yellow)
PLAYER_COLORS = {"P1": (214, 120, 42), "P2": (52, 104, 235), "P3": (122, 175, 27), "P4": (0, 161, 237)}
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def _minimap(players, bounces, scale=9, pad=40):
    w, h = int(COURT_W * scale + 2 * pad), int(COURT_L * scale + 2 * pad)
    img = np.full((h, w, 3), (70, 110, 60), np.uint8)
    P = lambda x, y: (int(pad + x * scale), int(pad + y * scale))
    cv2.rectangle(img, P(0, 0), P(COURT_W, COURT_L), (160, 90, 40), -1)
    cv2.rectangle(img, P(0, KITCHEN_FAR_Y), P(COURT_W, KITCHEN_NEAR_Y), (70, 60, 170), -1)
    for a, b in [((0, 0), (COURT_W, 0)), ((0, COURT_L), (COURT_W, COURT_L)), ((0, 0), (0, COURT_L)),
                 ((COURT_W, 0), (COURT_W, COURT_L)), ((0, KITCHEN_FAR_Y), (COURT_W, KITCHEN_FAR_Y)),
                 ((0, KITCHEN_NEAR_Y), (COURT_W, KITCHEN_NEAR_Y)), ((10, 0), (10, KITCHEN_FAR_Y)),
                 ((10, KITCHEN_NEAR_Y), (10, COURT_L))]:
        cv2.line(img, P(*a), P(*b), (255, 255, 255), 2)
    cv2.line(img, P(-1, NET_Y), P(COURT_W + 1, NET_Y), (230, 230, 230), 3)
    for x, y, age in bounces:
        cv2.circle(img, P(x, y), 6, (0, 255, 255), -1 if age < 15 else 1)
    for pid, (x, y) in players.items():
        cv2.circle(img, P(x, y), 10, PLAYER_COLORS.get(pid, (255, 255, 255)), -1)
        cv2.putText(img, pid, (P(x, y)[0] + 12, P(x, y)[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return img


def render_video(out_dir: Path, width: int = 1280) -> Path:
    meta = json.loads((out_dir / "meta.json").read_text())
    analysis = json.loads((out_dir / "analysis.json").read_text())
    rd = json.loads((out_dir / "render_data.json").read_text())
    fps = meta["fps"]
    names = {pid: s["name"] for pid, s in analysis["players"].items()}

    ball = {int(k): v for k, v in rd["ball"].items()}
    players = {int(k): v for k, v in rd["players"].items()}
    events_by_frame = {e["frame"]: e for e in rd["events"]}
    shots_by_frame = {s["frame"]: s for s in analysis["shots"]}
    bounces = [(b["frame"], *b["court"]) for b in analysis["bounces"]]
    # write exactly the segments the dashboard maps its clock onto
    keep = set()
    for seg in analysis["segments"]:
        keep.update(range(round(seg["start_t"] * fps), round(seg["end_t"] * fps) + 1))

    scale = width / meta["width"]
    height = int(meta["height"] * scale)
    raw = out_dir / "annotated_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    cap = cv2.VideoCapture(meta["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, meta["first_frame"])
    trail = deque(maxlen=12)
    label, label_until = None, -1
    marks = deque(maxlen=6)
    for f in range(meta["first_frame"], meta["last_frame"]):
        ok, frame = cap.read()
        if not ok:
            break
        if f not in players or f not in keep:
            trail.clear()
            continue
        img = cv2.resize(frame, (width, height))
        S = lambda x, y: (int(x * scale), int(y * scale))

        for p in players[f]:
            col = PLAYER_COLORS.get(p["id"], (200, 200, 200))
            kp = np.asarray(p["kp"])
            for a, b in SKELETON:
                if kp[a, 2] > 0.4 and kp[b, 2] > 0.4:
                    cv2.line(img, S(*kp[a, :2]), S(*kp[b, :2]), col, 2)
            x1, y1 = S(p["box"][0], p["box"][1])
            cv2.putText(img, names.get(p["id"], p["id"] or "?"), (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        if f in ball:
            trail.append(S(ball[f][0], ball[f][1]))
        else:
            trail.append(None)
        pts = [p for p in trail if p]
        for i, (a, b) in enumerate(zip(pts, pts[1:])):
            cv2.line(img, a, b, (0, 230, 255), 1 + i // 4)
        if f in ball:
            cv2.circle(img, pts[-1], 7, (0, 255, 255), 2)

        if f in events_by_frame:
            e = events_by_frame[f]
            marks.append((S(*e["img"]), e["kind"], f))
        for pt, kind, ef in marks:
            if f - ef < fps:
                cv2.drawMarker(img, pt, (0, 0, 255) if kind == "hit" else (255, 255, 0),
                               cv2.MARKER_TILTED_CROSS if kind == "hit" else cv2.MARKER_DIAMOND, 22, 2)
        if f in shots_by_frame:
            s = shots_by_frame[f]
            parts = [names.get(s["player"], s["player"] or "?"), s["shot_type"]]
            if s["stroke"] and s["shot_type"] not in ("serve",):
                parts.append(s["stroke"])
            if s["volley"]:
                parts.append("volley")
            if s["speed_mph"]:
                parts.append(f"{s['speed_mph']:.0f} mph")
            label, label_until = "  ".join(parts), f + int(1.5 * fps)
        if label and f <= label_until:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img, (20, height - 60), (40 + tw, height - 60 + th + 20), (20, 20, 20), -1)
            cv2.putText(img, label, (30, height - 60 + th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        mm = _minimap({p["id"]: p["court"] for p in players[f] if p["id"]},
                      [(x, y, f - bf) for bf, x, y in bounces if 0 <= f - bf < 3 * fps])
        mh = int(height * 0.55)
        mm = cv2.resize(mm, (int(mm.shape[1] * mh / mm.shape[0]), mh))
        img[10:10 + mh, width - mm.shape[1] - 10:width - 10] = cv2.addWeighted(
            img[10:10 + mh, width - mm.shape[1] - 10:width - 10], 0.15, mm, 0.85, 0)
        writer.write(img)
    cap.release()
    writer.release()

    out = out_dir / "annotated.mp4"
    if raw.stat().st_size < 10_000:  # nothing written (no main-view frames)
        raw.unlink()
        return None
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
    raw.unlink()
    return out
