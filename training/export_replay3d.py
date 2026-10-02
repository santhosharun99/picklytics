"""Export 3D replay data for dashboard/replay3d.html.

    PYTHONPATH=. python3 training/export_replay3d.py outputs/<run> outputs/reports/<report>.json <name>
-> dashboard/replay3d_<name>.json   (open http://localhost:8001/dashboard/replay3d.html?match=<name>)

Per rally: score/server/winner (tracking + rules), the 3D ball path per frame (event-pinned arcs,
pickleball/ball3d.py) with a confidence flag, hits (player, shot type, 3D speed), bounces (spot, in/out)
and both players' court positions at 10 Hz.
"""
import json, sys
from pathlib import Path

import numpy as np

from pickleball.analysis import load_detections
from pickleball.ball import track_ball
from pickleball.ball3d import match_arcs, positions
from pickleball.camera3d import calibrate_from_run
from pickleball.outcome import match_from_analysis
from pickleball.players import select_players

ROOT = Path(__file__).resolve().parents[1]
run, report_path, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
meta, recs = load_detections(run)
fps = meta["fps"]
A = json.loads((run / "analysis.json").read_text())
REP = json.loads(report_path.read_text())
cam = calibrate_from_run(run)
ball = track_ball(recs)
arcs_by = match_arcs(A, ball, cam.P)
outc = {o["index"]: o for o in match_from_analysis(A, run_dir=run)}
pname = {pid: p.get("name") or pid for pid, p in A["players"].items()}
R = {r["f"]: r for r in recs if r.get("court")}

rallies = []
for r in sorted(A["rallies"], key=lambda r: r["start_t"]):
    o = outc.get(r["index"], {})
    if o.get("after_match"):
        continue
    f0, f1 = int(r["start_t"] * fps) - int(0.5 * fps), int(r["end_t"] * fps) + int(0.5 * fps)
    # ball path, frame by frame
    path = []
    for a in arcs_by.get(r["index"], []):
        fr = np.arange(a["f0"], a["f1"] + 1)
        X = positions(np.asarray(a["theta"]), (fr - a["f0"]) / fps, a["beta"])
        for f, x in zip(fr, X):
            path.append([round(f / fps, 3), *[round(float(v), 2) for v in x], int(a.get("confident", True))])
    path.sort(key=lambda p: p[0])
    # hits with 3D speed (speed of the arc that starts at the hit)
    start_speed = {a["f0"]: a for a in arcs_by.get(r["index"], []) if a["kind0"] == "hit"}
    shots = sorted([s for s in A["shots"] if s["rally"] == r["index"]], key=lambda s: s["frame"])
    hits = []
    for s in shots:
        a = start_speed.get(s["frame"])
        hits.append({"t": round(s["frame"] / fps, 3), "player": pname.get(s["player"], s["player"]), "side": s["side"],
                     "shot": s.get("shot_type"), "stroke": s.get("stroke"), "inferred": bool(s.get("inferred")),
                     "speed_mph": round(a["speed_mph"]) if a and a.get("confident") else None,
                     "net_clear_ft": round(a["net_clear_ft"], 1) if a and a.get("net_clear_ft") is not None and a.get("confident") else None,
                     "pos": s.get("hitter_court")})
    bounces = [{"t": round(b["frame"] / fps, 3), "x": round(b["court"][0], 2), "y": round(b["court"][1], 2),
                "in": bool(-0.5 <= b["court"][0] <= 20.5 and -0.5 <= b["court"][1] <= 44.5)}
               for b in A["bounces"] if f0 <= b["frame"] <= f1]
    # players (court feet) at 10 Hz, named by side for this rally
    near_name = o.get("server") if o.get("server_side") == "near" else o.get("receiver")
    far_name = o.get("receiver") if o.get("server_side") == "near" else o.get("server")
    players = []
    for f in range(f0, f1 + 1, max(1, int(fps / 10))):
        rec = R.get(f)
        if not rec:
            continue
        sel = {p["side"]: p["court"] for p in select_players(rec.get("people", []), 1)}
        players.append([round(f / fps, 2), *(sel.get("near") or [None, None]), *(sel.get("far") or [None, None])])
    rallies.append({"id": r["index"], "t0": round(f0 / fps, 2), "t1": round(f1 / fps, 2), "game": o.get("game"),
                    "score": o.get("score_before"), "server": o.get("server"), "winner": o.get("winner"),
                    "near": near_name, "far": far_name, "ball": path, "hits": hits, "bounces": bounces,
                    "players": players})

out = {"title": REP["title"], "video": "../" + REP["video"], "fps": fps,
       "players": sorted({x for r in rallies for x in (r["near"], r["far"]) if x}), "rallies": rallies}
dst = ROOT / "dashboard" / f"replay3d_{name}.json"
dst.write_text(json.dumps(out, separators=(",", ":")))
print(f"{dst.name}: {len(rallies)} rallies, {sum(len(r['ball']) for r in rallies)} ball samples, {dst.stat().st_size / 1e6:.1f} MB")
