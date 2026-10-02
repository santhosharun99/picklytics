"""Pass 2: turn cached detections into tracks, events, rallies, shots and stats.

Fast (no models), so thresholds can be tuned and re-run without re-processing video.
Writes `<out>/analysis.json`.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .ball import track_ball
from .court import KITCHEN_FAR_Y, KITCHEN_NEAR_Y, NET_Y
from .events import detect_events, segment_rallies
from .players import PlayerIdentifier, select_players, smooth_positions
from .shots import classify_rally, infer_handedness, shots_to_dicts

HEAT_CELL_FT = 2.0


def load_detections(out_dir: Path):
    meta = json.loads((out_dir / "meta.json").read_text())
    with (out_dir / "detections.jsonl").open() as fh:
        recs = [json.loads(line) for line in fh]
    return meta, recs


def _segments(main_frames: list[int]) -> list[tuple[int, int]]:
    segs = []
    for f in main_frames:
        if segs and f - segs[-1][1] <= 1:
            segs[-1][1] = f
        else:
            segs.append([f, f])
    return [tuple(s) for s in segs]


def _other_camera_frames(recs, max_dev_px: float = 40.0, min_frames: int = 300) -> set:
    """Frames whose court sits far from where the (fixed) main camera puts it: replays or
    cut-aways from other cameras that the court check let through. Dropped so a replayed
    point is not counted twice and the single 3D camera stays valid."""
    kp = {r["f"]: np.asarray(r["court"]["kp"]) for r in recs if r.get("court")}
    if len(kp) < min_frames:
        return set()
    med = np.median(np.stack(list(kp.values())), 0)
    return {f for f, k in kp.items() if np.linalg.norm(k - med, axis=1).max() > max_dev_px}


def _hit_context(out_dir: Path, meta: dict, ball: dict, players_by_frame: dict):
    """Inputs for the learned hit decider: 3D camera, main-view ball track, cached paddles."""
    from .events import HIT_MODEL
    from .hitmodel import MODEL_PATH, Context
    if not HIT_MODEL["enabled"] or not MODEL_PATH.exists() or not players_by_frame:
        return None
    from .camera3d import calibrate_from_run
    from .paddles import PaddleCache
    try:
        cam = calibrate_from_run(out_dir)
    except Exception as e:  # no usable court view
        print("hit model skipped (camera):", e)
        return None
    main_ball = {f: v for f, v in ball.items() if f in players_by_frame}
    return Context.build(main_ball, players_by_frame, cam.P, meta["fps"], PaddleCache(out_dir, meta.get("video")))


def run_analysis(out_dir: Path, n_players: int = 2, names: dict[str, str] | None = None) -> dict:
    meta, recs = load_detections(out_dir)
    fps = meta["fps"]
    per_side = 1 if n_players == 2 else 2

    homs, players_by_frame, main = {}, {}, set()
    ident = PlayerIdentifier(n_players)
    off_camera = _other_camera_frames(recs)
    for r in recs:
        if r.get("court") is None or r["f"] in off_camera:
            continue
        f = r["f"]
        main.add(f)
        homs[f] = np.asarray(r["court"]["H"])
        players_by_frame[f] = ident.assign(select_players(r.get("people", []), per_side))

    ball = track_ball(recs)
    events = detect_events(ball, players_by_frame, homs, fps, hit_ctx=_hit_context(out_dir, meta, ball, players_by_frame))
    rallies = segment_rallies(events, main, fps)
    hand = infer_handedness(rallies)
    shots = [s for r in rallies for s in classify_rally(r, hand)]

    # --- player movement during rallies ---
    in_rally = np.zeros(max(main) + 2 if main else 1, bool)
    for r in rallies:
        in_rally[r.start_frame:min(r.end_frame, len(in_rally) - 1) + 1] = True
    tracks = defaultdict(list)
    for f in sorted(players_by_frame):
        for p in players_by_frame[f]:
            if p.get("id"):
                tracks[p["id"]].append((f, *p["court"]))

    player_stats = {}
    for pid, tr in tracks.items():
        tr = smooth_positions(tr)
        dist = top_speed = 0.0
        rally_pts, kitchen = [], 0
        for (f0, x0, y0), (f1, x1, y1) in zip(tr, tr[1:]):
            if f1 - f0 > 2 or not in_rally[f1]:
                continue
            d = float(np.hypot(x1 - x0, y1 - y0))
            v = d / ((f1 - f0) / fps)
            if v > 30:  # > 20 mph on foot: identity/detection glitch
                continue
            dist += d
        for f, x, y in tr:
            if in_rally[f]:
                rally_pts.append((x, y))
                kitchen += KITCHEN_FAR_Y <= y <= KITCHEN_NEAR_Y
        # top speed over 0.5 s windows
        step = max(1, int(fps / 2))
        for (f0, x0, y0), (f1, x1, y1) in zip(tr, tr[step:]):
            if f1 - f0 == step and in_rally[f1]:
                top_speed = max(top_speed, np.hypot(x1 - x0, y1 - y0) / (step / fps))
        # heatmap on the player's own half, normalised as if always on the near side
        heat = np.zeros((int(34 / HEAT_CELL_FT), int(32 / HEAT_CELL_FT)))  # y 22..56, x -6..26
        for x, y in rally_pts:
            if y < NET_Y:
                x, y = 20 - x, 44 - y
            gi, gj = int((y - NET_Y) / HEAT_CELL_FT), int((x + 6) / HEAT_CELL_FT)
            if 0 <= gi < heat.shape[0] and 0 <= gj < heat.shape[1]:
                heat[gi, gj] += 1
        my = [s for s in shots if s.player == pid]
        speeds = [s.speed_mph for s in my if s.speed_mph]
        player_stats[pid] = {
            "name": (names or {}).get(pid, pid),
            "dominant_hand": hand.get(pid),
            "distance_ft": round(dist, 1),
            "top_speed_mph": round(top_speed * 0.6818, 1),
            "kitchen_time_pct": round(100 * kitchen / max(1, len(rally_pts)), 1),
            "avg_depth_from_net_ft": round(float(np.mean([abs(y - NET_Y) for _, y in rally_pts])), 1) if rally_pts else None,
            "shots": len(my),
            "shot_types": dict(Counter(s.shot_type for s in my).most_common()),
            "strokes": dict(Counter(s.stroke for s in my if s.stroke)),
            "volleys": sum(s.volley for s in my),
            "avg_shot_mph": round(float(np.mean(speeds)), 1) if speeds else None,
            "max_shot_mph": round(float(max(speeds)), 1) if speeds else None,
            "heatmap": {"cell_ft": HEAT_CELL_FT, "origin": [-6, NET_Y], "grid": heat.astype(int).tolist()},
        }

    rally_list = []
    for r in rallies:
        rs = [s for s in shots if s.rally == r.index]
        rally_list.append({
            "index": r.index, "start_t": round(r.start_frame / fps, 2), "end_t": round(r.end_frame / fps, 2),
            "duration_s": round((r.end_frame - r.start_frame) / fps, 2), "shots": len(rs),
            "server": rs[0].player if rs else None,
            "last_hitter": rs[-1].player if rs else None,
        })

    lengths = [r["shots"] for r in rally_list]
    segs = _segments(sorted(main))
    result = {
        "meta": {**meta, "n_players": n_players},
        "summary": {
            "video_s": round((meta["last_frame"] - meta["first_frame"]) / fps, 1),
            "main_view_s": round(len(main) / fps, 1),
            "rallies": len(rallies),
            "shots": len(shots),
            "avg_rally_shots": round(float(np.mean(lengths)), 1) if lengths else 0,
            "longest_rally_shots": max(lengths) if lengths else 0,
            "shot_types": dict(Counter(s.shot_type for s in shots).most_common()),
            "ball_frames_pct": round(100 * sum(1 for f in ball if f in main) / max(1, len(main)), 1),
        },
        "players": player_stats,
        "rallies": rally_list,
        "shots": shots_to_dicts(shots),
        "bounces": [{"t": round(e.t, 2), "frame": e.frame, "court": [round(v, 2) for v in e.court], "side": e.side}
                    for e in events if e.kind == "bounce"],
        "segments": [{"start_t": round(a / fps, 2), "end_t": round(b / fps, 2)} for a, b in segs if b - a > fps],
        # positions at ~10 Hz for the dashboard's live court view
        "positions": [
            {"t": round(f / fps, 2),
             "p": {p["id"]: [round(p["court"][0], 1), round(p["court"][1], 1)] for p in players_by_frame[f] if p.get("id")}}
            for f in sorted(players_by_frame)[:: max(1, int(fps / 10))]
        ],
    }
    (out_dir / "analysis.json").write_text(json.dumps(result, separators=(",", ":")))
    _save_intermediate(out_dir, ball, events, players_by_frame)
    return result


def _save_intermediate(out_dir, ball, events, players_by_frame):
    """Per-frame data the renderer needs (kept out of analysis.json to keep it small)."""
    data = {
        "ball": {str(f): [round(v[0], 1), round(v[1], 1), bool(v[3])] for f, v in ball.items()},
        "events": [{"kind": e.kind, "frame": e.frame, "img": [round(v, 1) for v in e.img], "player": e.player}
                   for e in events],
        "players": {str(f): [{"id": p.get("id"), "box": p["box"], "court": p["court"], "kp": p["kp"]} for p in ps]
                    for f, ps in players_by_frame.items()},
    }
    (out_dir / "render_data.json").write_text(json.dumps(data, separators=(",", ":")))
