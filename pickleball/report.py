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

from . import winprob
from .depth import correct as correct_depth
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
        a = Path(d) / "analysis.json"
        D = json.loads(a.read_text()).get("depth_correction") if a.exists() else None  # pickleball/depth.py
        with (Path(d) / "detections.jsonl").open() as fh:
            for line in fh:
                r = json.loads(line)
                if not r.get("court"):
                    continue
                sel = select_players(r.get("people", []), 1)
                if sel:
                    pos[r["f"]] = {p["side"]: tuple(correct_depth(p["court"], D)) for p in sel}
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


def _third_quality(shots: list[dict], nxt: list[dict]) -> dict:
    """Quality of third shots: pace, low over the net (< 1.5 ft clearance), deep (>= 15 ft past the net), and how
    often the opponent's 4th shot was rushed. Note: an earlier finding (third-shot drive wins 61% vs 19%) came from
    the rule-based shot types and did not survive the learned shot types (87 of 89 third shots are drives)."""
    mph = [s["speed_mph"] for s in shots if s.get("speed_mph") and s.get("speed_conf") in ("high", "medium")]
    clr = [s["net_clear_ft"] for s in shots if s.get("net_clear_ft") is not None]
    dep = [(22 - s["land"][1]) if s["side"] == "near" else (s["land"][1] - 22) for s in shots if s.get("land")]
    rush = [s["spare_s"] < RUSHED_S for s in nxt if s.get("spare_s") is not None]
    return {"mph": round(float(np.median(mph))) if mph else None, "n_mph": len(mph),
            "low_pct": _pct(sum(c < 1.5 for c in clr), len(clr)), "n_clear": len(clr),
            "deep_pct": _pct(sum(d >= 15 for d in dep), len(dep)), "n_land": len(dep),
            "rushed_next_pct": _pct(sum(rush), len(rush)), "n_next": len(rush)}


def player_profile(name: str, opponent: str, rallies: list[dict], hands: dict | None = None) -> dict:
    add_time_pressure(rallies)
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
            for t, x, y in r.get("movement_t", {}).get(name, []):
                if t >= t0 and _own_half(x, y)[1] <= 9:
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
        "third_shot": {k: {"count": v, "won_pct": _pct(third_won[k], v), "rallies": third_rallies[k],
                           **_third_quality([r["shots"][2] for r in rallies if r["id"] in third_rallies[k]],
                                            [r["shots"][3] for r in rallies if r["id"] in third_rallies[k] and len(r["shots"]) > 3])}
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
    profile["speed"] = speed_profile(name, rallies)
    profile["running"] = running_profile(name, rallies)
    profile["placement"] = placement_profile(name, rallies)
    profile["pressure"] = pressure_profile(name, rallies)
    profile["big_points"] = winprob.big_points(name, rallies)
    profile["net"] = net_profile(name, rallies)
    profile["tactics"] = tactics_profile(name, rallies, (hands or {}).get(name, "right"))
    profile["insights"] = insights(profile, opponent)
    return profile


def _rally_running(track, win=5, max_gap=0.35, glitch_mph=20.0, sprint_mph=6.0):
    """Distance / speeds for one player in one rally from [t, x, y] samples (~10 Hz). Feet positions are
    noisy, so positions are averaged over `win` samples (0.5 s) inside continuous stretches; steps faster
    than a human can run (identity / detection glitches) are dropped."""
    if len(track) < win + 2:
        return None
    T = np.array(track, float)
    runs, cur = [], [T[0]]
    for a, b in zip(T, T[1:]):
        if b[0] - a[0] > max_gap:
            runs.append(np.array(cur)); cur = []
        cur.append(b)
    runs.append(np.array(cur))
    dist, speeds, dur, fast = 0.0, [], 0.0, 0.0
    k = np.ones(win) / win
    for run in runs:
        if len(run) < win + 2:
            continue
        xs = np.convolve(run[:, 1], k, "valid")
        ys = np.convolve(run[:, 2], k, "valid")
        ts = run[win // 2: win // 2 + len(xs), 0]
        step = np.hypot(np.diff(xs), np.diff(ys))
        dt = np.diff(ts)
        v = step / np.maximum(dt, 1e-3) * 0.6818
        ok = v <= glitch_mph
        dist += float(step[ok].sum())
        speeds += v[ok].tolist()
        dur += float(dt[ok].sum())
        fast += float(dt[ok & (v >= sprint_mph)].sum())
    if not speeds:
        return None
    sp = np.array(speeds)
    # top speed must be held: highest speed sustained for 0.5 s (5 consecutive steps)
    held = max((sp[i:i + 5].min() for i in range(len(sp) - 4)), default=float(np.median(sp)))
    return {"dist_ft": dist, "peak_mph": float(held), "avg_mph": float(dist / max(dur, 1e-3) * 0.6818),
            "fast_s": fast, "dur_s": dur}


def running_profile(name: str, rallies: list[dict]) -> dict:
    """How much and how fast a player moves during rallies (from the timed court tracks)."""
    per = [(r["id"], x) for r in rallies
           if (x := _rally_running(r.get("movement_t", {}).get(name, []))) is not None]
    if not per:
        return {}
    d = [x["dist_ft"] for _, x in per]
    pk = [x["peak_mph"] for _, x in per]
    top = sorted(per, key=lambda p: -p[1]["peak_mph"])[:3]
    # movement efficiency (cf. distance per minute in elite padel, Miralles et al. 2026 Biol Sport)
    ids = {rid for rid, _ in per}
    n_shots = sum(1 for r in rallies if r["id"] in ids for s in r["shots"] if s["player"] == name)
    minutes = sum(r["end_t"] - r["start_t"] for r in rallies if r["id"] in ids) / 60
    return {"rallies": len(per),
            "ft_per_shot": round(float(np.sum(d)) / n_shots, 1) if n_shots else None,
            "ft_per_min": round(float(np.sum(d)) / minutes) if minutes > 0 else None,
            "dist_per_rally_ft": round(float(np.median(d))),
            "total_dist_ft": round(float(np.sum(d))),
            "avg_speed_mph": round(float(np.median([x["avg_mph"] for _, x in per])), 1),
            "typical_peak_mph": round(float(np.median(pk)), 1),
            "top_sprint_mph": round(float(max(pk)), 1),
            "sprint_time_pct": round(100 * sum(x["fast_s"] for _, x in per) / max(sum(x["dur_s"] for _, x in per), 1e-3)),
            "top_sprints": [{"rally": rid, "mph": round(x["peak_mph"], 1)} for rid, x in top]}


REACT_S, RUN_FT_S, RUSHED_S = 0.25, 12.0, 0.3  # reaction time, movement speed, "rushed" below this spare time


def _pos_near(track, t, w=0.2):
    v = [(x, y) for tt, x, y in track if abs(tt - t) <= w]
    return np.median(np.array(v), 0) if v else None


def add_time_pressure(rallies: list[dict]):
    """Time pressure per shot (after Mlakar & Kovalchik 2020, J Sports Analytics, tennis): the time the hitter had
    since the opponent's shot, minus a reaction time and the time needed to cover the distance they actually moved
    (feet positions at the two hits, from the court tracks). spare_s < RUSHED_S = rushed. Validation on our two
    matches: rushed shots end the rally as the hitter's error 22% of the time vs 15% otherwise."""
    for r in rallies:
        sh = r["shots"]
        for a, b in zip(sh, sh[1:]):
            if a["player"] == b["player"] or "spare_s" in b:
                continue
            tr = r.get("movement_t", {}).get(b["player"], [])
            p0, p1 = _pos_near(tr, a["t"]), _pos_near(tr, b["t"])
            if p0 is None or p1 is None:
                continue
            dt, dist = b["t"] - a["t"], float(np.hypot(*(p1 - p0)))
            b["time_s"], b["moved_ft"] = round(dt, 2), round(dist, 1)
            b["spare_s"] = round(dt - REACT_S - dist / RUN_FT_S, 2)


def pressure_profile(name: str, rallies: list[dict]) -> dict:
    """Pressure the player faced (their shots) and applied (the opponent's next shot), and how rushed shots end."""
    add_time_pressure(rallies)
    faced, applied = [], []   # (spare_s, rally won by the hitter, shot ended the rally as the hitter's error, rally, shot)
    for r in rallies:
        if not r["winner"]:
            continue
        sh = r["shots"]
        for k, s in enumerate(sh):
            if "spare_s" not in s:
                continue
            row = (s["spare_s"], r["winner"] == s["player"], k == len(sh) - 1 and r["winner"] != s["player"], r["id"],
                   sh[k - 1].get("shot"))
            if s["player"] == name:
                faced.append(row)
            elif sh[k - 1]["player"] == name:
                applied.append(row)
    if not faced and not applied:
        return {}
    rate = lambda rows, i: _pct(sum(x[i] for x in rows), len(rows))
    rushed = [x for x in faced if x[0] < RUSHED_S]
    calm = [x for x in faced if x[0] >= RUSHED_S]
    opp_rushed = [x for x in applied if x[0] < RUSHED_S]
    by_shot = defaultdict(list)
    for x in applied:
        if x[4]:
            by_shot[x[4]].append(x[0] < RUSHED_S)
    return {
        "model": {"reaction_s": REACT_S, "run_ft_s": RUN_FT_S, "rushed_below_s": RUSHED_S},
        "faced": {"n": len(faced), "median_spare_s": round(float(np.median([x[0] for x in faced])), 2) if faced else None,
                  "rushed_pct": _pct(len(rushed), len(faced)), "n_rushed": len(rushed),
                  "error_when_rushed_pct": rate(rushed, 2), "error_otherwise_pct": rate(calm, 2),
                  "won_when_rushed_pct": rate(rushed, 1), "won_otherwise_pct": rate(calm, 1),
                  "rushed_rallies": sorted({x[3] for x in rushed})},
        "applied": {"n": len(applied), "rushed_opponent_pct": _pct(len(opp_rushed), len(applied)), "n_rushed": len(opp_rushed),
                    "opponent_error_when_rushed_pct": rate(opp_rushed, 2),
                    "by_shot": {k: {"n": len(v), "rushed_pct": _pct(sum(v), len(v))}
                                for k, v in sorted(by_shot.items(), key=lambda kv: -len(kv[1])) if len(v) >= 5},
                    "rushed_rallies": sorted({x[3] for x in opp_rushed})},
    }


NET_FT, NET_HOLD_S = 9.0, 0.5  # "at the net" = within 9 ft of it (kitchen line + a step), held for 0.5 s


def _net_arrival(track, t0):
    """Seconds after t0 until the player first stands at the net for NET_HOLD_S, or None."""
    pts = [(t, abs(y - NET_Y) <= NET_FT) for t, x, y in track if t >= t0]
    for i, (t, ok) in enumerate(pts):
        if ok and all(o for tt, o in pts[i:] if tt <= t + NET_HOLD_S):
            return t - t0
    return None


def net_profile(name: str, rallies: list[dict]) -> dict:
    """Kitchen-line control: how often and how fast the player gets to the net, and who gets there first.
    Pooled over our two matches the first player to the net won 59% of those rallies (n=46, not significant yet;
    net position is the key tactical factor in padel and pickleball doubles studies)."""
    reached, times, first, first_won, opp_first, opp_first_won, n = 0, [], 0, 0, 0, 0, 0
    first_r = []
    for r in rallies:
        mt = r.get("movement_t", {})
        if not r["winner"] or len(r["shots"]) < 2 or name not in mt or len(mt) != 2:
            continue
        n += 1
        t0 = r["shots"][0]["t"]
        arr = {p: _net_arrival(tr, t0) for p, tr in mt.items()}
        if arr[name] is not None:
            reached += 1
            times.append(arr[name])
        got = {p: t for p, t in arr.items() if t is not None}
        if got:
            f = min(got, key=got.get)
            if f == name:
                first += 1; first_won += r["winner"] == name; first_r.append(r["id"])
            else:
                opp_first += 1; opp_first_won += r["winner"] == name
    if not n:
        return {}
    return {"rallies": n, "reached_pct": _pct(reached, n), "median_arrival_s": round(float(np.median(times)), 1) if times else None,
            "first_pct": _pct(first, first + opp_first), "n_contested": first + opp_first,
            "won_when_first_pct": _pct(first_won, first), "n_first": first,
            "won_when_opp_first_pct": _pct(opp_first_won, opp_first), "n_opp_first": opp_first, "first_rallies": first_r}


MOVE_FT = 3.0  # lateral move that counts as being pulled wide / running around
SHOT_GROUP = {"serve": "Serve", "return": "Return", "drive": "Drive", "third-shot drive": "Drive",
              "groundstroke": "Groundstroke", "third-shot groundstroke": "Groundstroke", "volley": "Volley",
              "drop": "Drop / dink / reset", "third-shot drop": "Drop / dink / reset", "dink": "Drop / dink / reset",
              "reset": "Drop / dink / reset", "lob": "Lob", "overhead": "Overhead"}


def _zone(y):
    d = abs(y - NET_Y)
    return "net" if d <= NET_FT else ("mid" if d <= 18 else "back")


def tactics_profile(name: str, rallies: list[dict], hand: str = "right") -> dict:
    """How the player moves and chooses shots, from the court tracks (approximate, rule-based shot types):
    - strokes: each forehand / backhand split by the move before it (from where the player stood at the opponent's
      hit to the contact): pulled wide to that side (>= MOVE_FT), ball near the body (choice / positioning), or
      ran around (moved toward the other side, e.g. run-around forehand).
    - situations: shot choice by where the player and the opponent stand at contact (net / mid / back), with the
      direction (cross-court vs down the line), depth of the landing and the rally win rate.
    - response: for each kind of opponent shot, the average move from start to contact (own-half coordinates)."""
    strokes = {"forehand": Counter(), "backhand": Counter()}
    stroke_r = defaultdict(list)
    sit = defaultdict(list)
    resp = defaultdict(list)
    for r in rallies:
        sh, mt = r["shots"], r.get("movement_t", {})
        if name not in mt or len(mt) != 2:
            continue
        opp = next(n for n in mt if n != name)
        for k in range(1, len(sh)):
            a, b = sh[k - 1], sh[k]
            if b["player"] != name or a["player"] == name:
                continue
            p0, p1, q = _pos_near(mt[name], a["t"]), _pos_near(mt[name], b["t"]), _pos_near(mt[opp], b["t"])
            if p0 is None or p1 is None:
                continue
            right = (p1[0] - p0[0]) * (1 if p1[1] >= NET_Y else -1)       # lateral move to the player's right
            fh = right if hand != "left" else -right                       # ... toward the forehand side
            if b.get("stroke") in strokes:
                to_side = fh if b["stroke"] == "forehand" else -fh
                kind = "pulled_wide" if to_side >= MOVE_FT else ("ran_around" if to_side <= -MOVE_FT else "near_body")
                strokes[b["stroke"]][kind] += 1
                stroke_r[(b["stroke"], kind)].append(r["id"])
            s0, s1 = _own_half(*p0), _own_half(*p1)
            resp[SHOT_GROUP.get(a.get("shot"), "Other")].append((s0, s1, b.get("spare_s"), r["id"]))
            if q is None or not r["winner"]:
                continue
            cross = deep = short = None
            if b.get("land") and abs(p1[0] - 10) >= 2:
                lx, ly = b["land"]
                cross = (p1[0] - 10) * (lx - 10) < 0
                dep = abs(ly - NET_Y)
                deep, short = dep >= 15, dep <= 7
            sit[(_zone(p1[1]), _zone(q[1]))].append((b.get("shot"), cross, deep, short, r["winner"] == name, r["id"], k))
    pct = lambda xs: _pct(sum(1 for x in xs if x), sum(1 for x in xs if x is not None))
    situations = []
    for (me, op), v in sorted(sit.items(), key=lambda kv: -len(kv[1])):
        c = Counter(x[0] for x in v if x[0])
        situations.append({"me": me, "opp": op, "n": len(v),
                           "shots": [[k, n] for k, n in c.most_common(3)],
                           "cross_pct": pct([x[1] for x in v]), "deep_pct": pct([x[2] for x in v]),
                           "short_pct": pct([x[3] for x in v]), "won_pct": _pct(sum(x[4] for x in v), len(v)),
                           "rallies": sorted({x[5] for x in v})})
    response = {}
    for g, v in sorted(resp.items(), key=lambda kv: -len(kv[1])):
        if len(v) < 3:
            continue
        a0 = np.mean([x[0] for x in v], 0); a1 = np.mean([x[1] for x in v], 0)
        sp = [x[2] for x in v if x[2] is not None]
        response[g] = {"n": len(v), "from": [round(float(a0[0]), 1), round(float(a0[1]), 1)],
                       "to": [round(float(a1[0]), 1), round(float(a1[1]), 1)],
                       "moved_ft": round(float(np.median([np.hypot(x[1][0] - x[0][0], x[1][1] - x[0][1]) for x in v])), 1),
                       "spare_s": round(float(np.median(sp)), 2) if sp else None,
                       "rushed_pct": _pct(sum(x < RUSHED_S for x in sp), len(sp)), "rallies": sorted({x[3] for x in v})}
    out = {"strokes": {k: dict(v) for k, v in strokes.items()},
           "stroke_rallies": {f"{a}:{b}": sorted(set(v)) for (a, b), v in stroke_r.items()},
           "situations": situations, "response": response, "move_ft": MOVE_FT}
    out["takeaways"] = _tactics_takeaways(out)
    return out


def _tactics_takeaways(t: dict) -> dict:
    """One plain sentence per tactics card (numbers only; no claim the data cannot carry)."""
    tk = {}
    bh, fh = t["strokes"].get("backhand", {}), t["strokes"].get("forehand", {})
    nb, nf = sum(bh.values()), sum(fh.values())
    if nb >= 8:
        wide, body = _pct(bh.get("pulled_wide", 0), nb), _pct(bh.get("near_body", 0), nb)
        tk["strokes"] = (f"{wide}% of backhands came after being pulled wide to that side (forced); {body}% with the ball "
                         f"near the body (positioning or choice)." +
                         (f" Ran around the backhand to hit a forehand {fh.get('ran_around')}x." if fh.get("ran_around", 0) >= 3 else ""))
    main = [x for x in t["situations"] if not any(k in ("serve", "return") for k, _ in x["shots"][:1]) and x["n"] >= 5]
    if main:
        x = main[0]
        top, n = x["shots"][0]
        where = {"net": "at the net", "mid": "in mid-court", "back": "at the back"}
        dirn = (f", {x['cross_pct']}% cross-court" if x["cross_pct"] is not None else "")
        tk["situations"] = (f"Most common spot: {where[x['me']]} with the opponent {where[x['opp']]} ({x['n']} shots) - "
                            f"plays {top} {_pct(n, x['n'])}% of the time{dirn}; wins {x['won_pct']}% of those rallies.")
    if t["response"]:
        g, v = max(t["response"].items(), key=lambda kv: kv[1]["moved_ft"] if kv[1]["n"] >= 5 else -1)
        if v["n"] >= 5:
            tk["response"] = (f"Moves most against the opponent's {g.lower()}: {v['moved_ft']} ft on average "
                              f"(rushed on {v['rushed_pct']}% of them, {v['n']} shots).")
    return tk


def placement_profile(name: str, rallies: list[dict]) -> dict:
    """Where this player's shots land on the opponent's half, by shot group. Positions are normalised to
    the receiving half as seen from the hitter: x from the hitter's left (0..20), depth from the net (0..22)."""
    groups = {"Serve": ("serve",), "Return": ("return",), "Drive": ("drive", "third-shot drive"),
              "Drop / reset": ("drop", "third-shot drop", "reset", "dink")}
    out = {}
    for label, types in groups.items():
        pts = []
        for r in rallies:
            for s in r["shots"]:
                if s["player"] != name or s.get("shot") not in types or not s.get("land"):
                    continue
                x, y = s["land"]
                if s["side"] == "near":      # hitting from the near side: lands on the far half (y < 22)
                    pts.append([round(20 - x, 1), round(22 - y, 1), r["id"]])
                else:                        # hitting from the far side: lands on the near half
                    pts.append([round(x, 1), round(y - 22, 1), r["id"]])
        if pts:
            deep = sum(p[1] >= 15 for p in pts)
            out[label] = {"points": pts, "n": len(pts), "deep_pct": round(100 * deep / len(pts)),
                          "out_pct": round(100 * sum(not (0 <= p[0] <= 20 and 0 <= p[1] <= 22) for p in pts) / len(pts))}
    return out


def speed_profile(name: str, rallies: list[dict]) -> dict:
    """Ball speed off the paddle by shot group, from shots with a trustworthy speed (high/medium confidence)."""
    groups = {"Serve": ("serve",), "Return": ("return",), "Drive": ("drive", "third-shot drive"),
              "Third-shot drop": ("third-shot drop",), "Reset / drop": ("reset", "drop", "dink"), "Volley": ("volley",)}
    mine = [(r["id"], s) for r in rallies for s in r["shots"]
            if s["player"] == name and s.get("speed_mph") and s.get("speed_conf") in ("high", "medium")]
    out = {"groups": {}, "fastest": []}
    for label, types in groups.items():
        # volleys: the ball was hit before bouncing (a flag; learned shot types do not include "volley")
        v = [(rid, s["speed_mph"], s.get("speed_conf")) for rid, s in mine
             if ((s.get("volley") and s.get("shot") not in ("serve", "return")) if label == "Volley" else s.get("shot") in types)]
        if v:
            mph = [x for _, x, _ in v]
            sure = [x for _, x, c in v if c == "high"]  # top speeds only where both measurements agree
            out["groups"][label] = {"avg": round(float(np.median(mph))), "max": round(max(sure)) if sure else None,
                                    "n": len(v), "rallies": sorted({rid for rid, _, _ in v})}
    sure_shots = [(rid, s) for rid, s in mine if s.get("speed_conf") == "high"]
    for rid, s in sorted(sure_shots, key=lambda x: -x[1]["speed_mph"])[:5]:
        out["fastest"].append({"rally": rid, "t": s["t"], "shot": s.get("shot"), "mph": s["speed_mph"],
                               "err": s.get("speed_err")})
    return out


def fisher_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact test: are k1/n1 and k2/n2 different rates?"""
    from math import comb
    K, N = k1 + k2, n1 + n2
    pr = lambda k: comb(n1, k) * comb(n2, K - k) / comb(N, K)
    p0 = pr(k1)
    return min(1.0, sum(pr(k) for k in range(max(0, K - n2), min(n1, K) + 1) if pr(k) <= p0 * (1 + 1e-9)))


def insights(p: dict, opponent: str) -> list[dict]:
    """Plain-language strengths / weaknesses with their sample size; small samples are flagged."""
    out = []

    def add(kind, text, n, rallies):
        out.append({"kind": kind, "text": text, "n": n, "rallies": rallies, "low_sample": n < 5})

    s = p["summary"]
    if s["won_on_serve_pct"]["value"] is not None and s["won_on_return_pct"]["value"] is not None:
        a, b = s["won_on_serve_pct"], s["won_on_return_pct"]
        k1, k2 = round(a["value"] * a["n"] / 100), round(b["value"] * b["n"] / 100)
        sig = fisher_p(k1, a["n"], k2, b["n"]) < 0.1
        if a["value"] - b["value"] >= 20:
            add("strength" if sig else "pattern", f"Much stronger when serving ({a['value']}% rallies won) than returning ({b['value']}%).", a["n"] + b["n"], a["rallies"])
        elif b["value"] - a["value"] >= 20:
            add("strength" if sig else "pattern", f"Dangerous on return: wins {b['value']}% of return rallies vs {a['value']}% on serve.", a["n"] + b["n"], b["rallies"])
    st = p["stroke"]
    tot = sum(st.values())
    if tot:
        bh = st.get("backhand", 0)
        add("pattern", f"Backhand on {round(100 * bh / tot)}% of shots ({bh}/{tot}).", tot, [])
    errs = p["error_shots"]
    if errs:
        k, v = next(iter(errs.items()))
        add("pattern", f"Most rallies lost on their own {k} ({v['count']}x).", v["count"], v["rallies"])  # descriptive, not tested
    fins = p["finishing_shots"]
    if fins:
        k, v = next(iter(fins.items()))
        add("pattern", f"Most rallies won with a {k} ({v['count']}x).", v["count"], v["rallies"])  # descriptive, not tested
    for k, v in p["third_shot"].items():
        if v["count"] >= 2:
            add("pattern", f"Third shot: {k} {v['count']}x, rallies won {v['won_pct']}%.", v["count"], v["rallies"])
    rl = p["rally_length"]
    sh, lg = rl["short (<=3 shots)"], rl["long (7+)"]
    if sh["value"] is not None and lg["value"] is not None and abs(sh["value"] - lg["value"]) >= 25:
        better = "short" if sh["value"] > lg["value"] else "long"
        k1, k2 = round(sh["value"] * sh["n"] / 100), round(lg["value"] * lg["n"] / 100)
        sig = fisher_p(k1, sh["n"], k2, lg["n"]) < 0.1
        add(("strength" if better == "long" else "weakness") if sig else "pattern",
            f"Wins {sh['value']}% of short rallies vs {lg['value']}% of long ones: better in {better} rallies.",
            sh["n"] + lg["n"], lg["rallies"] if better == "long" else sh["rallies"])
    bp = p.get("big_points") or {}
    if bp.get("n_big", 0) >= 10 and bp["big_won_pct"] is not None and bp["rest_won_pct"] is not None:
        d = bp["big_won_pct"] - bp["rest_won_pct"]
        k1, k2 = round(bp["big_won_pct"] * bp["n_big"] / 100), round(bp["rest_won_pct"] * bp["n_rest"] / 100)
        if abs(d) >= 15 and fisher_p(k1, bp["n_big"], k2, bp["n_rest"]) < 0.1:  # only claim it when unlikely to be chance
            add("strength" if d > 0 else "weakness",
                f"{'Raises' if d > 0 else 'Drops'} their level on big points: wins {bp['big_won_pct']}% of the most important rallies "
                f"vs {bp['rest_won_pct']}% of the rest ({bp['n_big']} big rallies).", bp["n_big"], bp["big_rallies"])
    pr = p.get("pressure") or {}
    fc, ap = pr.get("faced", {}), pr.get("applied", {})
    if fc.get("n_rushed", 0) >= 5 and fc.get("error_when_rushed_pct") is not None and fc.get("error_otherwise_pct") is not None:
        a, b = fc["error_when_rushed_pct"], fc["error_otherwise_pct"]
        n_calm = fc["n"] - fc["n_rushed"]
        sig = fisher_p(round(a * fc["n_rushed"] / 100), fc["n_rushed"], round(b * n_calm / 100), n_calm) < 0.1
        if a - b >= 10 and sig:
            add("weakness", f"Breaks down under time pressure: {a}% of rushed shots end the rally as an error vs {b}% otherwise "
                            f"({fc['n_rushed']} rushed shots).", fc["n_rushed"], fc["rushed_rallies"])
        elif a < b and sig:
            add("strength", f"Holds up under time pressure: errors on {a}% of rushed shots vs {b}% otherwise "
                            f"({fc['n_rushed']} rushed shots).", fc["n_rushed"], fc["rushed_rallies"])
    if ap.get("by_shot"):
        k, v = max(ap["by_shot"].items(), key=lambda kv: kv[1]["rushed_pct"] or 0)
        add("pattern", f"Rushes the opponent on {ap['rushed_opponent_pct']}% of shots; most with the {k} ({v['rushed_pct']}% of {v['n']}).",
            ap["n"], ap["rushed_rallies"])
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
        "rallies": [{k: v for k, v in r.items() if k not in ("movement", "movement_t")} for r in rallies],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1))
    return report


# ---------------------------------------------------------------------------------------------
# Fully automatic report: rallies, shots and positions from an analysis run (analysis.json),
# point winners from the broadcast scoreboard (scoreboard.py). No analyst labels needed.
# ---------------------------------------------------------------------------------------------

def build_rallies_auto(run_dir: Path, video_file: str, sb_cfg: dict | None, max_next_s: float = 40.0):
    from . import scoreboard as sb
    run_dir = Path(run_dir)
    A = json.loads((run_dir / "analysis.json").read_text())
    fps = A["meta"]["fps"]
    name_of = {pid: p.get("name") or pid for pid, p in A["players"].items()}
    tracks = load_tracks([run_dir])
    shots_by = defaultdict(list)
    for s in A["shots"]:
        shots_by[s["rally"]].append(s)
    rallies = sorted(A["rallies"], key=lambda r: r["start_t"])
    readings = {}
    # tracking + game rules decide server, winner and games (end changes observed from the players' looks)
    from .outcome import match_from_analysis
    track_w = {x["index"]: x for x in match_from_analysis(A, run_dir=run_dir)}

    def reading(t0, t1):
        key = (round(t0, 1), round(t1, 1))
        if key not in readings:
            readings[key] = sb.stable_reading(video_file, int(t0 * fps), int(t1 * fps), sb_cfg) if sb_cfg else None
        return readings[key]

    out = []
    for k, r in enumerate(rallies):
        shots = []
        for i, s in enumerate(sorted(shots_by[r["index"]], key=lambda s: s["frame"])):
            p = s.get("hitter_court")
            shots.append({"i": i, "t": round(s["t"], 2), "player": name_of.get(s["player"], s["player"]), "side": s["side"],
                          "shot": s.get("shot_type"), "stroke": s.get("stroke"), "volley": bool(s.get("volley")),
                          "pos": [round(v, 1) for v in p] if p else None, "zone": s.get("hitter_zone"),
                          "land": s.get("land_3d") or s.get("landing_court"),  # 3D lifter landing spot first
                          "speed_mph": s.get("ball_mph"), "speed_err": s.get("ball_mph_err"),
                          "speed_conf": s.get("speed_conf"), "net_clear_ft": s.get("net_clear_ft")})
        if not shots:
            continue
        for s, nxt in zip(shots, shots[1:]):
            s["next_pos"] = nxt["pos"]
        # tracking decides: server = first hitter, winner from the next rally's server (side-out logic)
        tw = track_w.get(r["index"], {})
        if tw.get("after_match"):
            continue  # play after the match was decided (warm-down, celebrations) is not part of the analysis
        server, winner = tw.get("server") or shots[0]["player"], tw.get("winner")
        sb_server = sb_winner = None
        if sb_cfg and sb_cfg.get("mode", "ocr") == "ocr":  # scoreboard only as an independent cross-check
            before = reading(r["start_t"] - 4, r["start_t"] - 0.2)
            if before and before["server"] is not None:
                sb_server = sb_cfg["names"][before["server"]]
            nxt = rallies[k + 1] if k + 1 < len(rallies) else None
            if nxt and nxt["start_t"] - r["end_t"] <= max_next_s:
                after = reading(nxt["start_t"] - 4, nxt["start_t"] - 0.2)
            else:
                after = reading(r["end_t"] + 5, r["end_t"] + 20)
            w = sb.point_winner(before, after)
            sb_winner = sb_cfg["names"][w] if w is not None else None
        side_of = {}
        for s in shots:
            side_of.setdefault(s["player"], s["side"])
        move, move_t = defaultdict(list), defaultdict(list)
        step = max(1, int(fps / 10))
        for f in range(int(r["start_t"] * fps), int(r["end_t"] * fps) + 1, step):
            for side, xy in tracks.get(f, {}).items():
                move[side].append([round(v, 1) for v in xy])
                move_t[side].append([round(f / fps, 2), round(xy[0], 2), round(xy[1], 2)])
        out.append({
            "id": r["index"], "start_t": round(r["start_t"], 2), "end_t": round(r["end_t"], 2),
            "server": server, "winner": winner, "ending": None, "game": tw.get("game"), "score_before": tw.get("score_before"),
            "scoreboard": {"server": sb_server, "winner": sb_winner,
                           "agrees": None if sb_winner is None else (sb_winner == winner and (sb_server in (None, server)))},
            "shots": shots, "n_shots": len(shots),
            "movement": {n: move.get(sd, []) for n, sd in side_of.items()},
            "movement_t": {n: move_t.get(sd, []) for n, sd in side_of.items()},
            "detail": True,
        })
    return out, name_of


def build_report_auto(run_dir: Path, video_file: str, video_url: str, out_path: Path, sb_cfg: dict | None) -> dict:
    rallies, name_of = build_rallies_auto(run_dir, video_file, sb_cfg)
    a, b = list(name_of.values())[:2]
    A = json.loads((Path(run_dir) / "analysis.json").read_text())
    hands = {p.get("name") or pid: p.get("dominant_hand") or "right" for pid, p in A["players"].items()}
    checked = [r for r in rallies if r["scoreboard"]["agrees"] is not None]
    agree = sum(r["scoreboard"]["agrees"] for r in checked)
    games = max((r["game"] or 1 for r in rallies), default=1)
    report = {
        "title": f"{a} vs {b}",
        "source": f"{len(rallies)} automatically detected rallies in {games} games (full match; server and point winner from tracking"
                  + (f"; scoreboard agrees on {agree}/{len(checked)}" if checked else "") + ")",
        "video": video_url,
        "win_probability": winprob.annotate(rallies, a, b),  # adds wp_before / wp_after / importance per rally
        "players": {a: player_profile(a, b, rallies, hands), b: player_profile(b, a, rallies, hands)},
        "rallies": [{k: v for k, v in r.items() if k not in ("movement", "movement_t")} for r in rallies],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1))
    return report
