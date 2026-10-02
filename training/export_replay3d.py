"""Export 3D ball arcs of the labelled rallies for dashboard/replay3d.html.

    PYTHONPATH=. python3 training/export_replay3d.py
Writes dashboard/replay3d.json: per rally the fitted 3D arcs (auto-detected contacts, no labels
used), the detected events and the analyst-labelled hit frames for comparison.
"""
import json
from pathlib import Path

import numpy as np

from pickleball.analysis import load_detections
from pickleball.ball import track_ball
from pickleball.camera3d import calibrate_from_run
from pickleball.flights import flight_boundaries
from pickleball.players import PlayerIdentifier, select_players
from pickleball.trajectory3d import anchored_track, positions

ROOT = Path(__file__).resolve().parents[1]
L = json.loads((ROOT / "labels" / "Mens_Singles_GOLD_2026_US_Open.json").read_text())
out = []
for run in sorted((ROOT / "outputs").glob("cmp_ball_detector_v3_kaggle_*")):
    meta, recs = load_detections(run)
    fps = meta["fps"]
    cam = calibrate_from_run(run)
    ident, pl = PlayerIdentifier(2), {}
    for r in recs:
        if r.get("court"):
            pl[r["f"]] = ident.assign(select_players(r.get("people", []), 1))
    ball = {f: v for f, v in track_ball(recs).items() if f in pl}
    cands = sorted({b[0] for b in flight_boundaries(ball, penalty=400)})
    events, arcs = anchored_track(cam.P, ball, cands, pl, fps, beta=0.5, penalty=2000)
    for r in L["rallies"]:
        if r["end"] is None or not (meta["first_frame"] <= r["start"] and r["end"] < meta["last_frame"]):
            continue
        inr = lambda f: r["start"] - 15 <= f <= r["end"] + 15
        ra = []
        for a in arcs:
            if inr(a["f0"]):
                ts = np.arange(0, a["f1"] - a["f0"] + 1) / fps
                ra.append(np.round(positions(a["theta"], ts, a["beta"]), 2).tolist())
        players = {}
        for f in range(r["start"], r["end"] + 1, 3):
            for p in pl.get(f, []):
                players.setdefault(p.get("id") or "?", []).append([round(p["court"][0], 1), round(p["court"][1], 1)])
        out.append({"name": f"{r['start'] / fps // 60:.0f}:{r['start'] / fps % 60:04.1f}", "start": r["start"],
                    "arcs": ra, "events": [e for e in events if inr(e["frame"])],
                    "true_hits": [e["frame"] for e in r["events"] if e["kind"] == "hit"], "players": players,
                    "camera": cam.position.round(1).tolist()})
(ROOT / "dashboard" / "replay3d.json").write_text(json.dumps(out))
print(len(out), "rallies")
