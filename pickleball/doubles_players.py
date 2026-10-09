"""Doubles: individual and partnership analytics from NAMED player tracks (docs/DOUBLES.md).

Inputs per match (all cached, nothing re-detected): the match report (outputs/reports/<report>.json: rallies, shots
credited to a named hitter), the run's player tracklets (outputs/<run>/tracklets.pkl: court position per frame) and
their names (outputs/<run>/identity.json, each player marked once, the rest named by look - pickleball/identity.py).

Coordinates follow the placement convention (docs/PLACEMENT.md): every position is seen from the PLAYER, looking over
the net - u = 0 at their left sideline, d = feet from the net on their own half. "Up" = within UP_FT (10.5 ft) of the
net (the kitchen-line zone used for singles kitchen time, pickleball/court.py). Positions carry the same depth
correction as all other movement numbers (pickleball/depth.py, the run's model from depth.for_run), about +-1 ft.
Frames whose player is not named are left out (coverage is reported).

Output (one dict, written to outputs/reports/<report>.doubles.json by training/build_doubles_players.py):
  players[name]: movement (same method as singles: report._rally_running), depth / up time, left-right side,
                 net arrival when serving / receiving, shot roles and share of the team's shots, middle balls taken,
                 cross-overs (took the ball on the partner's side), heatmap
  teams[team]:   formation over rally time and at the opponents' contacts (both up / one up / both back), partner
                 spacing, side switches, both-up arrival after the serve, middle-ball split, who plays left
Every count keeps its denominator; clip lists are [rally id, feed time].
"""
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import depth
from .court import UP_FT
from .report import _rally_running

NET_Y = 22.0
BACK_FT = 15.0         # "back" = deeper than 15 ft (only for the depth split; formation uses up / not up)
BIN_S = 0.1            # pair metrics are computed on 0.1 s bins where both partners are seen
SWITCH_GAP_FT = 2.0    # a side switch needs the partners at least this far apart sideways ...
SWITCH_HOLD_BINS = 5   # ... for 0.5 s on the new sides
MIN_SAMPLES = 10       # a player counts as present in a rally with at least ~1 s of named positions
HEAT_FT = 2.0
CENTRE = (20 / 3, 40 / 3)


def own(x, y):
    """court (x, y) -> (u from the player's left, d from the net, near end?)"""
    near = y >= NET_Y
    return (x if near else 20.0 - x), abs(y - NET_Y), near


def named_tracks(run_dir: Path, fps: float, raw: bool = False):
    """name -> sorted array [t, x, y] (depth-corrected court positions of every frame where the player is named;
    raw=True: as tracked, for calibrating the correction)."""
    run_dir = Path(run_dir)
    names = {int(k): v.get("name") for k, v in json.loads((run_dir / "identity.json").read_text())["tracklets"].items()}
    D = None if raw else depth.for_run(run_dir)
    T = pickle.loads((run_dir / "tracklets.pkl").read_bytes())
    per, n_all, n_named = defaultdict(list), 0, 0
    for t in T:
        n_all += len(t["frames"])
        nm = names.get(t["id"])
        if not nm:
            continue
        n_named += len(t["frames"])
        for f, (x, y) in zip(t["frames"], t["court"]):
            cx, cy = depth.correct((float(x), float(y)), D)
            per[nm].append((f / fps, cx, cy))
    out = {n: np.array(sorted(v)) for n, v in per.items()}
    return out, (n_named / n_all if n_all else 0.0)


def _window(track, t0, t1):
    if track is None or not len(track):
        return np.zeros((0, 3))
    i, j = np.searchsorted(track[:, 0], [t0, t1])
    return track[i:j]


def _at(track, t, w=0.15):
    """median position of a track within +-w s of t, or None"""
    s = _window(track, t - w, t + w)
    return None if not len(s) else (float(np.median(s[:, 1])), float(np.median(s[:, 2])))


def _bins(s, t0):
    """samples -> {bin index: (u, d)} (median per 0.1 s bin)"""
    by = defaultdict(list)
    for t, x, y in s:
        u, d, _ = own(x, y)
        by[int((t - t0) / BIN_S)].append((u, d))
    return {k: tuple(np.median(np.array(v), 0)) for k, v in by.items()}


def _resample(s):
    """~10 Hz [t, x, y] list (median per 0.1 s bin) - the sampling the movement method was tuned on"""
    by = defaultdict(list)
    for t, x, y in s:
        by[int(t / BIN_S)].append((x, y))
    return [[(k + 0.5) * BIN_S, *np.median(np.array(v), 0)] for k, v in sorted(by.items())]


def _switch(seq):
    """first bin where the left-side player changes and stays changed for SWITCH_HOLD_BINS, else None"""
    w = [who for _, who in seq]
    for i in range(SWITCH_HOLD_BINS, len(w) - SWITCH_HOLD_BINS + 1):
        before, after = set(w[i - SWITCH_HOLD_BINS:i]), set(w[i:i + SWITCH_HOLD_BINS])
        if len(before) == 1 and len(after) == 1 and before != after:
            return seq[i][0]
    return None


def _pct(a, b):
    return round(100 * a / b) if b else None


def _med(v, nd=1):
    return round(float(np.median(v)), nd) if len(v) else None


# ------------------------------------------------------------------ who hit it (doubles)
# The report's hitter (training/identify.py: the named tracklet nearest the shot's 'hitter_court') picks the partner
# at the kitchen line far too often: right on 21 of 39 overlay-named serves and 7 of 45 returns where the depth rule
# decides (docs/DOUBLES.md). attribute_hitters() replaces it, shot by shot, with the method recorded:
#   'overlay'   serve: the server named by the broadcast score overlay (TNPPL)
#   'depth'     return: the deeper receiving player when the partners are >= 5 ft apart in depth (the returner lets
#               the serve bounce; the partner waits at the kitchen line)
#   'ball'      otherwise: the partner whose image box (widened for paddle reach) is nearest the detected ball around
#               the contact frame - only with a clear margin; else None (the shot stays team-level)
REACH_W, REACH_TOP, MARGIN_PX = 0.5, 0.3, 25.0


def _ball_dist(T, names, ball, frame, cands, W=4):
    best = {}
    fr = sorted((f for f in range(frame - W, frame + W + 1) if str(f) in ball), key=lambda f: abs(f - frame))[:3]
    for f in fr:
        bx, by = ball[str(f)][:2]
        for t in T:
            nm = names.get(t["id"])
            if nm not in cands or not (t["f0"] <= f <= t["f1"]):
                continue
            i = int(np.argmin(np.abs(np.asarray(t["frames"]) - f)))
            if abs(t["frames"][i] - f) > 3:
                continue
            x0, y0, x1, y1 = t["box"][i]
            h = y1 - y0
            dx = max(x0 - h * REACH_W - bx, 0, bx - x1 - h * REACH_W)
            dy = max(y0 - h * REACH_TOP - by, 0, by - y1)
            d = float(np.hypot(dx, dy) + 0.15 * abs(bx - (x0 + x1) / 2))
            best[nm] = min(best.get(nm, 1e9), d)
    return best


def attribute_hitters(report, run_dir: Path, tracks) -> dict:
    """'rally_id:index' -> {'name': str | None, 'method': str, 'old': report hitter}"""
    run_dir = Path(run_dir)
    names = {int(k): v.get("name") for k, v in json.loads((run_dir / "identity.json").read_text())["tracklets"].items()}
    T = pickle.loads((run_dir / "tracklets.pkl").read_bytes())
    bp = run_dir / "ball_v3.json"
    ball = json.loads(bp.read_text()) if bp.exists() else {}
    teams = report.get("teams") or {}
    out = {}
    for r in report["rallies"]:
        shots = [s for s in r["shots"] if not s.get("inferred")]
        sb = r.get("scoreboard") or {}
        for i, s in enumerate(shots):
            cands = teams.get(s["player"], [])
            role = (s.get("pl") or {}).get("role")
            on = [n for n in cands if _at(tracks.get(n), s["t"], 0.3)]
            key, name, how = f"{r['id']}:{s['i']}", None, "unattributed"
            if len(on) == 1:
                name, how = on[0], "only player seen"
            elif role == "serve" and sb.get("server_name") in cands and sb.get("server") == s["player"]:
                name, how = sb["server_name"], "overlay"
            else:
                if role == "return" and len(on) == 2:
                    pa, pb = _at(tracks.get(on[0]), s["t"], 0.2), _at(tracks.get(on[1]), s["t"], 0.2)
                    if pa and pb and abs(own(*pa)[1] - own(*pb)[1]) >= 5:
                        name, how = (on[0] if own(*pa)[1] > own(*pb)[1] else on[1]), "depth"
                if name is None and len(on) == 2:
                    d = _ball_dist(T, names, ball, s["f"], on)
                    if len(d) == 2:
                        (n1, d1), (n2, d2) = sorted(d.items(), key=lambda kv: kv[1])
                        if d2 - d1 >= MARGIN_PX:
                            name, how = n1, "ball"
            out[key] = {"name": name, "method": how, "old": s.get("hitter")}
    return out


def kitchen_stances(A: dict, run_dir: Path) -> list:
    """Kitchen-line stances of a doubles run for the depth calibration (pickleball/depth.py, rule 2): at every serve
    (first detected hit of a rally) the two named players at the receiving end, raw positions; the shallower one when
    they are clearly one up, one back. -> [(raw (x, y), camera (x, y, height))]. A: analysis.json dict (replays dropped)."""
    cam = (A.get("depth_correction") or {}).get("camera")
    if not cam:
        return []
    tracks, _ = named_tracks(run_dir, A["meta"]["fps"], raw=True)
    first = {}
    for s in sorted(A["shots"], key=lambda s: s["frame"]):
        if not s.get("inferred"):
            first.setdefault(s["rally"], s)
    out = []
    for s in first.values():
        recv_near = s["side"] == "far"
        ps = [p for p in (_at(tr, s["t"], 0.3) for tr in tracks.values()) if p and (p[1] >= NET_Y) == recv_near]
        k = depth.kitchen_stance(ps)
        if k:
            out.append((k, cam))
    return out


def build(report: dict, run_dir: Path, fps: float) -> dict:
    tracks, coverage = named_tracks(run_dir, fps)
    hit = attribute_hitters(report, run_dir, tracks)
    report = json.loads(json.dumps(report))     # local copy: shots get the new hitter, the report file is untouched
    for r in report["rallies"]:
        for s in r["shots"]:
            if not s.get("inferred"):
                s["hitter"] = hit.get(f"{r['id']}:{s['i']}", {}).get("name")
    teams = report.get("teams") or {}
    team_of = {n: t for t, ns in teams.items() for n in ns}
    rallies = report["rallies"]
    P = {n: {"team": team_of.get(n), "samples": 0, "up_s": 0, "back_s": 0, "left_s": 0, "depth": [], "runs": [],
             "arr": {"server": [], "server_partner": [], "returner": [], "returner_partner": []}, "heat": np.zeros((int(34 / HEAT_FT), int(32 / HEAT_FT))),
             "shots": defaultdict(int), "middle": 0, "middle_clips": [], "cross": 0, "cross_volley": 0, "cross_clips": [],
             "rallies": set()} for n in team_of}
    Tm = {t: {"bins": 0, "both_up": 0, "one_up": 0, "both_back": 0, "opp_contacts": defaultdict(int), "dist": [], "lat": [],
              "dgap": [], "switch_rallies": 0, "pair_rallies": 0, "switch_clips": [], "left": defaultdict(int),
              "both_up_after": {"serving": [], "receiving": []}, "middle": defaultdict(int), "middle_n": 0}
          for t in teams}
    for r in rallies:
        t0, t1 = r["start_t"], r["end_t"]
        shots = [s for s in r["shots"] if not s.get("inferred")]
        serve_t = shots[0]["t"] if shots else t0
        win = {n: _window(tracks.get(n), t0, t1) for n in team_of}
        present = {n for n, s in win.items() if len(s) >= MIN_SAMPLES}
        # the pair on court for each team in this rally (team events rotate players)
        onc = {t: [n for n in ns if n in present] for t, ns in teams.items()}
        pair = {t: v for t, v in onc.items() if len(v) == 2}
        # rally roles: server / server's partner / returner / returner's partner (as in padel role analyses)
        roles = {}
        sv = shots[0].get("hitter") if shots and (shots[0].get("pl") or {}).get("role") == "serve" else None
        rt = shots[1].get("hitter") if len(shots) > 1 and (shots[1].get("pl") or {}).get("role") == "return" else None
        for who, mate in ((sv, "server_partner"), (rt, "returner_partner")):
            if who in team_of:
                roles[who] = "server" if mate == "server_partner" else "returner"
                for n in pair.get(team_of[who], []):
                    if n != who:
                        roles[n] = mate
        # ---- per player: time-based
        for n in present:
            s, p = win[n], P[n]
            p["rallies"].add(r["id"])
            ud = np.array([own(x, y)[:2] for _, x, y in s])
            p["samples"] += len(s)
            p["up_s"] += int((ud[:, 1] <= UP_FT).sum())
            p["back_s"] += int((ud[:, 1] > BACK_FT).sum())
            p["left_s"] += int((ud[:, 0] < 10).sum())
            p["depth"] += ud[:, 1].tolist()
            for u, d in ud:
                i, j = int(d / HEAT_FT), int((u + 6) / HEAT_FT)
                if 0 <= i < p["heat"].shape[0] and 0 <= j < p["heat"].shape[1]:
                    p["heat"][i, j] += 1
            run = _rally_running(_resample(s))
            if run:
                p["runs"].append((r["id"], run))
            role = roles.get(n)
            if role:
                after = s[s[:, 0] >= serve_t]
                up = [t for t, x, y in after if own(x, y)[1] <= UP_FT]
                p["arr"][role].append((r["id"], round(up[0] - serve_t, 2) if up else None, run["dist_ft"] if run else None))
        # ---- per team: both partners seen
        for team, (a, b) in pair.items():
            ba, bb = _bins(win[a], t0), _bins(win[b], t0)
            common = sorted(set(ba) & set(bb))
            if len(common) < MIN_SAMPLES:
                continue
            m = Tm[team]
            m["pair_rallies"] += 1
            seq = []
            for k in common:
                (ua, da), (ub, db) = ba[k], bb[k]
                ups = int(da <= UP_FT) + int(db <= UP_FT)   # int(): numpy bools add as logical OR
                m["bins"] += 1
                m["both_up" if ups == 2 else "one_up" if ups == 1 else "both_back"] += 1
                m["lat"].append(abs(ua - ub)); m["dgap"].append(abs(da - db)); m["dist"].append(float(np.hypot(ua - ub, da - db)))
                if abs(ua - ub) >= SWITCH_GAP_FT:
                    who = a if ua < ub else b
                    seq.append((k, who))
                    m["left"][who] += 1
            at = _switch(seq)
            if at is not None:
                m["switch_rallies"] += 1
                m["switch_clips"].append([r["id"], round(t0 + at * BIN_S, 1)])
            role = "serving" if r.get("server") == team else "receiving"
            kk = [k for k in common if t0 + k * BIN_S >= serve_t and ba[k][1] <= UP_FT and bb[k][1] <= UP_FT]
            m["both_up_after"][role].append((r["id"], round(t0 + kk[0] * BIN_S - serve_t, 2) if kk else None))
        # ---- shots: roles, formation at the opponents' contacts, middle balls, cross-overs
        for i, s in enumerate(shots):
            team, hitter = s["player"], s.get("hitter")
            if hitter in P:
                P[hitter]["shots"][(s.get("pl") or {}).get("role", "later")] += 1
            for tm, (a, b) in pair.items():   # formation of the team that must answer this shot
                if tm == team:
                    continue
                pa, pb = _at(tracks.get(a), s["t"]), _at(tracks.get(b), s["t"])
                if pa and pb:
                    ups = int(own(*pa)[1] <= UP_FT) + int(own(*pb)[1] <= UP_FT)
                    Tm[tm]["opp_contacts"]["both_up" if ups == 2 else "one_up" if ups == 1 else "both_back"] += 1
            if hitter not in P or team not in pair or hitter not in pair[team]:
                continue
            partner = [n for n in pair[team] if n != hitter][0]
            ph, pp = _at(tracks.get(hitter), s["t"]), _at(tracks.get(partner), s["t"])
            if not (ph and pp):
                continue
            uh, up_ = own(*ph)[0], own(*pp)[0]
            role = (s.get("pl") or {}).get("role")
            if role not in ("third", "later"):
                continue
            if CENTRE[0] <= uh < CENTRE[1]:
                Tm[team]["middle_n"] += 1
                Tm[team]["middle"][hitter] += 1
                P[hitter]["middle"] += 1
                P[hitter]["middle_clips"].append([r["id"], s["t"]])
            # cross-over: at the opponents' previous contact the hitter was on one side of the partner, at this
            # contact on the other side (poach / switch to take the ball)
            if i > 0:
                q = shots[i - 1]
                ph0, pp0 = _at(tracks.get(hitter), q["t"]), _at(tracks.get(partner), q["t"])
                if ph0 and pp0:
                    before, now = own(*ph0)[0] - own(*pp0)[0], uh - up_
                    if abs(before) >= SWITCH_GAP_FT and abs(now) >= SWITCH_GAP_FT and np.sign(before) != np.sign(now):
                        P[hitter]["cross"] += 1
                        P[hitter]["cross_volley"] += bool(s.get("volley"))
                        P[hitter]["cross_clips"].append([r["id"], s["t"]])
    # ---- summarise
    team_shots = {t: sum(sum(P[n]["shots"].values()) for n in ns if n in P) for t, ns in teams.items()}
    team_rally_shots = {t: sum(P[n]["shots"]["third"] + P[n]["shots"]["later"] for n in ns if n in P) for t, ns in teams.items()}
    out_p = {}
    for n, p in P.items():
        if not p["samples"]:
            continue
        runs = p["runs"]
        arr = {k: {"n": len(v), "reached": sum(1 for _, x, _d in v if x is not None), "reached_pct": _pct(sum(1 for _, x, _d in v if x is not None), len(v)),
                   "median_s": _med([x for _, x, _d in v if x is not None]), "dist_per_rally_ft": _med([d for *_, d in v if d is not None], 0)}
               for k, v in p["arr"].items()}
        sh = dict(p["shots"]); tot = sum(sh.values()); rally = sh.get("third", 0) + sh.get("later", 0)
        out_p[n] = {
            "team": p["team"], "rallies": len(p["rallies"]),
            "movement": {"rallies": len(runs), "dist_per_rally_ft": round(float(np.median([x["dist_ft"] for _, x in runs]))) if runs else None,
                         "typical_peak_mph": _med([x["peak_mph"] for _, x in runs]), "top_sprint_mph": round(max(x["peak_mph"] for _, x in runs), 1) if runs else None,
                         "top_sprints": [{"rally": rid, "mph": round(x["peak_mph"], 1), "t": round(x["peak_t"], 1)} for rid, x in sorted(runs, key=lambda q: -q[1]["peak_mph"])[:3]]},
            "position": {"avg_depth_ft": _med(p["depth"]), "up_pct": _pct(p["up_s"], p["samples"]), "back_pct": _pct(p["back_s"], p["samples"]),
                         "left_pct": _pct(p["left_s"], p["samples"])},
            "by_rally_role": arr,
            "shots": {"total": tot, "by_role": sh, "team_share_pct": _pct(tot, team_shots.get(p["team"], 0)),
                      "rally_shots": rally, "team_rally_share_pct": _pct(rally, team_rally_shots.get(p["team"], 0))},
            "middle": {"taken": p["middle"], "clips": p["middle_clips"][:12]},
            "cross": {"n": p["cross"], "volleys": p["cross_volley"], "of_rally_shots": rally, "clips": p["cross_clips"][:12]},
            "heatmap": {"cell_ft": HEAT_FT, "u0": -6, "grid": p["heat"].astype(int).tolist()},
        }
    out_t = {}
    for t, m in Tm.items():
        b = m["bins"]
        oc = dict(m["opp_contacts"]); noc = sum(oc.values())
        both = {k: {"n": len(v), "reached": sum(1 for _, x in v if x is not None), "reached_pct": _pct(sum(1 for _, x in v if x is not None), len(v)),
                    "median_s": _med([x for _, x in v if x is not None])} for k, v in m["both_up_after"].items()}
        lt = dict(m["left"]); nl = sum(lt.values())
        out_t[t] = {"players": teams[t], "pair_rallies": m["pair_rallies"], "pair_seconds": round(b * BIN_S),
                    "formation_time_pct": {k: _pct(m[k], b) for k in ("both_up", "one_up", "both_back")},
                    "formation_at_opp_contact": {"n": noc, **{k: _pct(oc.get(k, 0), noc) for k in ("both_up", "one_up", "both_back")}},
                    "spacing": {"median_ft": _med(m["dist"]), "lateral_ft": _med(m["lat"]), "depth_gap_ft": _med(m["dgap"])},
                    "switches": {"rallies": m["switch_rallies"], "of": m["pair_rallies"], "clips": m["switch_clips"][:12]},
                    "left_side_pct": {n: _pct(v, nl) for n, v in lt.items()},
                    "both_up_after_serve": both,
                    "middle": {"n": m["middle_n"], "by_player": dict(m["middle"])},
                    "shots": {"total": team_shots.get(t, 0), "rally": team_rally_shots.get(t, 0)}}
    methods = defaultdict(int)
    for v in hit.values():
        methods[v["method"]] += 1
    changed = sum(1 for v in hit.values() if v["name"] and v["old"] and v["name"] != v["old"])
    return {"named_coverage_pct": round(100 * coverage), "up_ft": UP_FT, "back_ft": BACK_FT, "depth": depth.for_run(run_dir), "players": out_p, "teams": out_t,
            "hitters": hit, "hitter_methods": dict(methods), "hitters_changed": changed}
