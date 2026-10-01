"""Scouting report: per-player performance profile from reviewed rallies.

Event truth (who hit, shot type, stroke, rally winner) comes from analyst labels; where
players stood and how they moved comes from the tracking cache (detections.jsonl) of
analysis runs covering the same frames. Every stat keeps the rallies behind it so the
report can play the clips.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .court import KITCHEN_FAR_Y, KITCHEN_NEAR_Y, NET_Y, court_zone
from .players import select_players

HEAT_FT = 2.0
SHORT, LONG = 3, 7  # rally length buckets (shots): <=3 short, >=7 long


def _own_half(x, y):
    """Normalise a position to the player's own half, net at y=0, baseline at y=22, x from their right."""
    return (20 - x, y - NET_Y) if y >= NET_Y else (x, NET_Y - y)


def load_tracks(run_dirs: list[Path]):
    """frame -> {'near': (x, y), 'far': (x, y)} court positions of the two players."""
    pos = {}
    for d in run_dirs:
        with (Path(d) / "detections.jsonl").open() as fh:
            for line in fh:
                r = json.loads(line)
                if not r.get("court"):
                    continue
                sel = select_players(r.get("people", []), 1)
                if sel:
                    pos[r["f"]] = {p["side"]: tuple(p["court"]) for p in sel}
    return pos


def _pos_at(tracks, f, side, win=4):
    for d in range(win + 1):
        for g in (f - d, f + d):
            if g in tracks and side in tracks[g]:
                return tracks[g][side]
    return None


def build_rallies(labels: dict, tracks: dict, fps: float):
    names = labels["players"]
    out = []
    for r in sorted(labels["rallies"], key=lambda r: r["start"]):
        if r.get("reviewed") is False or r.get("end") is None:
            continue
        hits = [e for e in r["events"] if e["kind"] == "hit"]
        if not hits:
            continue
        near = r.get("near", "A")
        side_of = {near: "near", ("B" if near == "A" else "A"): "far"}
        shots = []
        for i, h in enumerate(hits):
            pid = h.get("player") or ("A" if (h.get("side") == "near") == (near == "A") else "B")
            side = side_of[pid]
            p = _pos_at(tracks, h["frame"], side)
            shots.append({
                "i": i, "t": round(h["frame"] / fps, 2), "player": names[pid], "side": side,
                "shot": h.get("shot"), "stroke": h.get("stroke"), "volley": bool(h.get("volley")),
                "pos": [round(v, 1) for v in p] if p else None,
                "zone": court_zone(p[1]) if p else None,
            })
        for s, nxt in zip(shots, shots[1:]):
            s["next_pos"] = nxt["pos"]
        # player movement during the rally (sampled at 10 Hz)
        move = defaultdict(list)
        step = max(1, int(fps / 10))
        for f in range(r["start"], r["end"] + 1, step):
            for side, xy in tracks.get(f, {}).items():
                move[side].append([round(v, 1) for v in xy])
        winner = names[r["winner"]] if r.get("winner") else None
        out.append({
            "id": r["id"], "start_t": round(r["start"] / fps, 2), "end_t": round(r["end"] / fps, 2),
            "server": shots[0]["player"], "winner": winner, "ending": r.get("ending"),
            "shots": shots, "n_shots": len(shots),
            "movement": {names[pid]: move.get(side, []) for pid, side in side_of.items()},
            "detail": any(s["shot"] not in (None, "serve", "return") for s in shots[2:]) or len(shots) <= 2,
        })
    return out


def _pct(a, b):
    return round(100 * a / b) if b else None


def _stat(value, n, rallies, note=None):
    return {"value": value, "n": n, "rallies": sorted(set(rallies)), **({"note": note} if note else {})}


def player_profile(name: str, opponent: str, rallies: list[dict]) -> dict:
    mine = [s for r in rallies for s in r["shots"] if s["player"] == name]
    decided = [r for r in rallies if r["winner"]]
    won = [r for r in decided if r["winner"] == name]

    serve_r = [r for r in decided if r["server"] == name]
    ret_r = [r for r in decided if r["server"] == opponent]

    # third shot (when serving)
    third = Counter()
    third_won = Counter()
    third_rallies = defaultdict(list)
    for r in rallies:
        if r["server"] == name and len(r["shots"]) >= 3 and r["shots"][2]["shot"]:
            k = r["shots"][2]["shot"]
            third[k] += 1
            third_rallies[k].append(r["id"])
            if r["winner"] == name:
                third_won[k] += 1

    typed = [s for s in mine if s["shot"]]
    by_type = Counter(s["shot"] for s in typed)
    stroke = Counter(s["stroke"] for s in mine if s["stroke"])
    zones = Counter(s["zone"] for s in mine if s["zone"])

    # how rallies end: whoever hit the last shot either won it (winner / forced error) or lost it (error)
    finishing, errors = Counter(), Counter()
    fin_r, err_r = defaultdict(list), defaultdict(list)
    for r in decided:
        last = r["shots"][-1]
        if last["player"] != name:
            continue
        key = f"{last['shot'] or 'shot'} ({last['stroke'] or '?'})"
        if r["winner"] == name:
            finishing[key] += 1
            fin_r[key].append(r["id"])
        else:
            errors[key] += 1
            err_r[key].append(r["id"])

    # rally length
    length = {}
    for label, cond in (("short (<=3 shots)", lambda n: n <= SHORT), ("medium (4-6)", lambda n: SHORT < n < LONG),
                        ("long (7+)", lambda n: n >= LONG)):
        rs = [r for r in decided if cond(r["n_shots"])]
        length[label] = _stat(_pct(sum(r["winner"] == name for r in rs), len(rs)), len(rs), [r["id"] for r in rs])

    # movement: coverage heatmap on own half + depth + kitchen time
    pts = [_own_half(*xy) for r in rallies for xy in r["movement"].get(name, [])]
    heat = np.zeros((int(34 / HEAT_FT), int(32 / HEAT_FT)))  # depth 0..34 ft from net, x -6..26
    for x, d in pts:
        i, j = int(d / HEAT_FT), int((x + 6) / HEAT_FT)
        if 0 <= i < heat.shape[0] and 0 <= j < heat.shape[1]:
            heat[i, j] += 1
    depths = [d for _, d in pts]
    contact = [_own_half(*s["pos"]) for s in mine if s["pos"]]

    # time to reach the kitchen line (within 9 ft of net) after hitting the return
    arrive = []
    for r in rallies:
        if r["server"] == opponent and len(r["shots"]) >= 2:
            t0 = r["shots"][1]["t"]
            track = r["movement"].get(name, [])
            dt = 0.1
            for k, xy in enumerate(track):
                t = r["start_t"] + k * dt
                if t >= t0 and _own_half(*xy)[1] <= 9:
                    arrive.append(round(t - t0, 2))
                    break

    profile = {
        "name": name,
        "summary": {
            "rallies": _stat(len(rallies), len(rallies), [r["id"] for r in rallies]),
            "rallies_won_pct": _stat(_pct(len(won), len(decided)), len(decided), [r["id"] for r in won]),
            "won_on_serve_pct": _stat(_pct(sum(r["winner"] == name for r in serve_r), len(serve_r)), len(serve_r),
                                      [r["id"] for r in serve_r]),
            "won_on_return_pct": _stat(_pct(sum(r["winner"] == name for r in ret_r), len(ret_r)), len(ret_r),
                                       [r["id"] for r in ret_r]),
            "shots": _stat(len(mine), len(mine), [r["id"] for r in rallies]),
        },
        "third_shot": {k: {"count": v, "won_pct": _pct(third_won[k], v), "rallies": third_rallies[k]}
                       for k, v in third.most_common()},
        "shot_mix": dict(by_type.most_common()),
        "stroke": dict(stroke),
        "stroke_by_shot": {k: dict(Counter(s["stroke"] for s in typed if s["shot"] == k and s["stroke"]))
                           for k in by_type},
        "zones": dict(zones),
        "finishing_shots": {k: {"count": v, "rallies": fin_r[k]} for k, v in finishing.most_common()},
        "error_shots": {k: {"count": v, "rallies": err_r[k]} for k, v in errors.most_common()},
        "rally_length": length,
        "movement": {
            "heatmap": {"cell_ft": HEAT_FT, "grid": heat.astype(int).tolist()},
            "avg_depth_ft": round(float(np.mean(depths)), 1) if depths else None,
            "kitchen_time_pct": _pct(sum(d <= 9 for d in depths), len(depths)),
            "contact_points": [[round(x, 1), round(d, 1)] for x, d in contact],
            "return_to_kitchen_s": {"median": float(np.median(arrive)) if arrive else None, "n": len(arrive)},
        },
    }
    profile["insights"] = insights(profile, opponent)
    return profile


def insights(p: dict, opponent: str) -> list[dict]:
    """Plain-language strengths / weaknesses with their sample size; small samples are flagged."""
    out = []

    def add(kind, text, n, rallies):
        out.append({"kind": kind, "text": text, "n": n, "rallies": rallies, "low_sample": n < 5})

    s = p["summary"]
    if s["won_on_serve_pct"]["value"] is not None and s["won_on_return_pct"]["value"] is not None:
        a, b = s["won_on_serve_pct"], s["won_on_return_pct"]
        if a["value"] - b["value"] >= 20:
            add("strength", f"Much stronger when serving ({a['value']}% rallies won) than returning ({b['value']}%).", a["n"] + b["n"], a["rallies"])
        elif b["value"] - a["value"] >= 20:
            add("strength", f"Dangerous on return: wins {b['value']}% of return rallies vs {a['value']}% on serve.", a["n"] + b["n"], b["rallies"])
    st = p["stroke"]
    tot = sum(st.values())
    if tot:
        bh = st.get("backhand", 0)
        add("pattern", f"Backhand on {round(100 * bh / tot)}% of shots ({bh}/{tot}).", tot, [])
    errs = p["error_shots"]
    if errs:
        k, v = next(iter(errs.items()))
        add("weakness", f"Most rallies lost on their own {k} ({v['count']}x).", v["count"], v["rallies"])
    fins = p["finishing_shots"]
    if fins:
        k, v = next(iter(fins.items()))
        add("strength", f"Most rallies won with a {k} ({v['count']}x).", v["count"], v["rallies"])
    for k, v in p["third_shot"].items():
        if v["count"] >= 2:
            add("pattern", f"Third shot: {k} {v['count']}x, rallies won {v['won_pct']}%.", v["count"], v["rallies"])
    rl = p["rally_length"]
    sh, lg = rl["short (<=3 shots)"], rl["long (7+)"]
    if sh["value"] is not None and lg["value"] is not None and abs(sh["value"] - lg["value"]) >= 25:
        better = "short" if sh["value"] > lg["value"] else "long"
        add("strength" if better == "long" else "weakness",
            f"Wins {sh['value']}% of short rallies vs {lg['value']}% of long ones: better in {better} rallies.",
            sh["n"] + lg["n"], lg["rallies"] if better == "long" else sh["rallies"])
    mv = p["movement"]
    if mv["return_to_kitchen_s"]["median"] is not None:
        add("pattern", f"Reaches the kitchen line about {mv['return_to_kitchen_s']['median']} s after returning.",
            mv["return_to_kitchen_s"]["n"], [])
    if mv["avg_depth_ft"] is not None:
        add("pattern", f"Plays on average {mv['avg_depth_ft']} ft from the net; {mv['kitchen_time_pct']}% of rally time at the kitchen.",
            len(mv["contact_points"]), [])
    return out


def build_report(labels_path: Path, run_dirs: list[Path], video: str, out_path: Path) -> dict:
    labels = json.loads(Path(labels_path).read_text())
    fps = labels.get("fps", 29.97)
    tracks = load_tracks(run_dirs)
    rallies = build_rallies(labels, tracks, fps)
    a, b = labels["players"]["A"], labels["players"]["B"]
    report = {
        "title": f"{a} vs {b}",
        "source": f"{len(rallies)} analyst-reviewed rallies ({sum(r['detail'] for r in rallies)} with full shot detail)",
        "video": video,
        "players": {a: player_profile(a, b, rallies), b: player_profile(b, a, rallies)},
        "rallies": [{k: v for k, v in r.items() if k != "movement"} for r in rallies],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1))
    return report
