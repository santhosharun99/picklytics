"""Find broadcast replays among detected rallies.

Some broadcasts (TNPPL) replay points from the same fixed main camera, so a replay looks like a live rally to
the tracker. The TNPPL broadcast plays a stinger (full-screen lime / navy 'TNPPL Season 2' wipe) right before
each replay: a rally starting within STINGER_S after one is a replay.

Tried and dropped: matching a rally's foreground (players) against earlier footage. It found the men's singles
replay, but a camera with spectators' heads in the foreground (team singles) or similar serve formations
(women's doubles) gave live rallies the same overlap (checked by eye: 11 false replays over 3 videos), and
re-encoded slow-motion replays are not pixel-identical to the live frames (grey-level difference ~37).
Cached in <run>/replays.json.
"""
import json
from pathlib import Path

import cv2

STINGER_LIME = 0.08  # share of lime pixels in a full-screen TNPPL replay wipe (live frames: < 0.02)
STINGER_S = 3.0


def stingers(video: str) -> list[int]:
    """Frames (every 2nd) that show a replay stinger."""
    cap = cv2.VideoCapture(video)
    f, out = 0, []
    while cap.grab():
        if f % 2 == 0:
            ok, img = cap.retrieve()
            if ok:
                hsv = cv2.cvtColor(cv2.resize(img, (192, 108)), cv2.COLOR_BGR2HSV)
                lime = ((hsv[..., 0] >= 25) & (hsv[..., 0] <= 45) & (hsv[..., 1] > 150) & (hsv[..., 2] > 150)).mean()
                if lime > STINGER_LIME:
                    out.append(f)
        f += 1
    return out


def find_replays(run_dir, video: str | None = None) -> dict:
    """{rally index: {"replay": bool}}."""
    run_dir = Path(run_dir)
    cache = run_dir / "replays.json"
    A = json.loads((run_dir / "analysis.json").read_text())
    key = [round(r["start_t"], 2) for r in A["rallies"]]
    fps = A["meta"]["fps"]
    if cache.exists():
        c = json.loads(cache.read_text())
        if c.get("key") == key and c.get("stingers") is not None:
            st = c["stingers"]
        else:
            st = None
    else:
        st = None
    if st is None:
        video = video or A["meta"].get("video") or json.loads((run_dir / "meta.json").read_text())["video"]
        st = stingers(video)
    out = {r["index"]: {"replay": any(0 <= r["start_t"] * fps - g <= STINGER_S * fps for g in st)} for r in A["rallies"]}
    cache.write_text(json.dumps({"key": key, "stingers": st, "stingers_t": sorted({round(g / fps) for g in st}),
                                 "rallies": out}, indent=0))
    return out


def drop_replays(A: dict, run_dir) -> dict:
    """analysis dict without replayed rallies (and their shots); adds A["replays"] = dropped rally indices."""
    rep = find_replays(run_dir)
    bad = {i for i, v in rep.items() if v["replay"]}
    A = dict(A)
    A["rallies"] = [r for r in A["rallies"] if r["index"] not in bad]
    A["shots"] = [s for s in A["shots"] if s["rally"] not in bad]
    A["replays"] = sorted(bad)
    return A
