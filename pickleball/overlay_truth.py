"""Per-rally ground truth from the TNPPL score overlay timeline (training/overlay_timeline.py -> overlay.json).

For each detected rally: the overlay state just before it (score, server name, serve number) and the state
just before the next serve. From the two:
  - a team's score went up by one       -> that team won the rally (and, in side-out scoring, was serving);
  - nothing changed (side-out scoring)  -> the serving team lost (2nd server or side out);
  - the score jumped by more than one   -> rallies were missed in between: no verdict.
Team = row on the overlay (0 = Kodai Tigers / first team, 1 = second team). The serve line names the server,
mapped to a team with the match-card roster (fuzzy match: OCR slips like 'VIRINA REODY').
Note: in the men's singles video the serve line never changed from RAMAKRISHNAN, so for singles the server
name is ignored and only score changes count.
"""
import difflib
import json
from pathlib import Path


def roster_lookup(rosters: list[list[str]]):
    names = {n.upper(): t for t, team in enumerate(rosters) for n in team}

    def team_of(ocr: str | None):
        if not ocr:
            return None, None
        m = difflib.get_close_matches(ocr.upper(), list(names), n=1, cutoff=0.6)
        return (names[m[0]], m[0].title()) if m else (None, None)
    return team_of


def _state(readings, t0, t1):
    """Most common (fit score, server, serve_no) among readings in [t0, t1]."""
    rs = [r for r in readings if t0 <= r["t"] <= t1][-5:]  # the latest readings: the score may change in the window
    if not rs:
        return None
    keys = [tuple(r["fit"]) for r in rs]
    sc = max(set(keys), key=keys.count)
    srv = [(r["server"], r["serve_no"]) for r in rs if tuple(r["fit"]) == sc and r["server"]]
    srv_last = [x for x in srv if x[1]]
    server, no = (max(set(srv_last), key=srv_last.count) if srv_last else (srv[-1] if srv else (None, None)))
    return {"score": list(sc), "server": server, "serve_no": no, "t": rs[-1]["t"]}


def rally_truth(overlay_path, rallies: list[dict], rosters: list[list[str]], scoring: str = "sideout",
                use_server_name: bool = True, singles: bool = False, window_s: float = 25.0) -> list[dict]:
    """rallies: time-ordered {"index", "start_t", "end_t"}. Returns one dict per rally with
    before/after overlay states, "winner_team", "server_team", "server" (name) where the overlay tells.
    singles: without server names, the server is carried forward (the rally winner serves next) as long as
    the score before a rally equals the score after the previous one (no rally missed in between)."""
    ov = json.loads(Path(overlay_path).read_text())
    R = ov["readings"]
    team_of = roster_lookup(rosters)
    out = []
    for k, r in enumerate(rallies):
        prev_end = rallies[k - 1]["end_t"] if k else r["start_t"] - window_s
        nxt_start = rallies[k + 1]["start_t"] if k + 1 < len(rallies) else r["end_t"] + window_s
        before = _state(R, max(prev_end + 1.0, r["start_t"] - window_s), r["start_t"] - 0.3)
        after = _state(R, max(r["end_t"] + 1.0, nxt_start - window_s), nxt_start - 0.3)
        res = {"index": r["index"], "before": before, "after": after, "winner_team": None, "server_team": None,
               "server": None, "serve_no": None, "verdict": None}
        if before and use_server_name:
            res["server_team"], res["server"] = team_of(before["server"])
            res["serve_no"] = before["serve_no"]
        if res["server_team"] is None and singles and out:
            p = out[-1]
            if p["winner_team"] is not None and p["after"] and before and p["after"]["score"] == before["score"]:
                res["server_team"] = p["winner_team"]
        if before and after:
            d = [a - b for a, b in zip(after["score"], before["score"])]
            if sorted(d) == [0, 1]:
                res["winner_team"] = d.index(1)
                res["verdict"] = "scored"
                if scoring == "sideout":
                    res["server_team"] = res["winner_team"]  # only the serving team scores
            elif d == [0, 0] and scoring == "sideout" and res["server_team"] is not None:
                res["winner_team"] = 1 - res["server_team"]
                res["verdict"] = "no score"
            elif d == [0, 0]:
                res["verdict"] = "no score (server unknown)"
            else:
                res["verdict"] = f"score jump {d}"
        out.append(res)
    return out


def points_from_overlay(overlay_path) -> list[dict]:
    """Every point the overlay shows (also points whose rally tracking missed): one entry per +1 step of the
    fitted score, {"t", "before": [a, b], "winner_team", "server" (name on the serve line before it)}.
    A jump of more than one point between readings yields one entry per point (time = the jump)."""
    R = json.loads(Path(overlay_path).read_text())["readings"]
    out, prev, srv = [], None, None
    for r in R:
        sc = list(r["fit"])
        if prev is not None and sc != prev and all(s >= p for s, p in zip(sc, prev)):
            cur = list(prev)
            for team in (0, 1):
                for _ in range(sc[team] - prev[team]):
                    out.append({"t": r["t"], "before": list(cur), "winner_team": team, "server": srv,
                                "exact": sum(sc) - sum(prev) == 1})
                    cur[team] += 1
        if r["server"]:
            srv = r["server"]
        prev = sc
    return out


def rotation_results(overlay_path, rosters: list[list[str]], every: int = 5) -> list[dict]:
    """Rotating team singles (TNPPL Super Singles): who was on court for each point follows from the combined
    score (slot = combined // every, cycling through the roster order). Per matchup: points won by each side and
    how often the serve-line name agrees with the slot's players (a check on the rotation rule)."""
    team_of = roster_lookup(rosters)
    n = len(rosters[0])
    res = {}
    for p in points_from_overlay(overlay_path):
        slot = (sum(p["before"]) // every) % n
        e = res.setdefault(slot, {"slot": slot + 1, "players": [rosters[0][slot], rosters[1][slot]], "points": [0, 0],
                                  "serve_line_checked": 0, "serve_line_agrees": 0})
        e["points"][p["winner_team"]] += 1
        _, name = team_of(p["server"])
        if name:
            e["serve_line_checked"] += 1
            e["serve_line_agrees"] += name.lower() in (x.lower() for x in e["players"])
    return [res[k] for k in sorted(res)]


def transitions(overlay_path, rosters: list[list[str]], min_stable: int = 2) -> list[dict]:
    """The overlay's state changes in time order; each change = one finished rally. State = (fitted score,
    server name matched to the roster, serve number). A state must hold for min_stable readings (OCR slips).
    Each entry: {"t": time the new state first shows, "before", "after" (states), "winner_team",
    "server_team", "server", "serve_no", "verdict"}; jumps of more than one point -> winner None."""
    R = json.loads(Path(overlay_path).read_text())["readings"]
    team_of = roster_lookup(rosters)
    states = []  # [state, first t, count]
    for r in R:
        t_, name = team_of(r["server"]) if r["server"] else (None, None)
        st = (tuple(r["fit"]), name, r["serve_no"])
        if states and states[-1][0][0] == st[0] and (st[1] is None or st[1] == states[-1][0][1]) \
                and (st[2] is None or st[2] == states[-1][0][2]):
            states[-1][2] += 1
            continue
        if states and states[-1][0][0] == st[0] and states[-1][0][1] is None:  # fill a missing name / number
            states[-1][0] = (st[0], st[1] or states[-1][0][1], st[2] or states[-1][0][2])
            states[-1][2] += 1
            continue
        states.append([st, r["t"], 1])
    states = [s for s in states if s[2] >= min_stable]
    out = []
    for (a, _, _), (b, t, _) in zip(states, states[1:]):
        if a == b:
            continue
        srv_team = team_of(a[1])[0] if a[1] else None
        d = [x - y for x, y in zip(b[0], a[0])]
        if sorted(d) == [0, 1]:
            w, v = d.index(1), "scored"
        elif d == [0, 0]:
            w, v = (1 - srv_team if srv_team is not None else None), "no score"
        else:
            w, v = None, f"score jump {d}"
        out.append({"t": t, "before": {"score": list(a[0]), "server": a[1], "serve_no": a[2]},
                    "after": {"score": list(b[0]), "server": b[1], "serve_no": b[2]},
                    "winner_team": w, "server_team": srv_team, "server": a[1], "serve_no": a[2], "verdict": v})
    return out


def rally_truth_seq(overlay_path, rallies: list[dict], rosters: list[list[str]], max_lag_s: float = 30.0) -> list[dict]:
    """Truth per rally by matching rallies to overlay transitions in order: rally k gets the first unused
    transition that shows after it ends and no later than max_lag_s after it (the operator may update late,
    even during the next rally). Unmatched rallies -> all None (a rally the overlay never registered: let,
    split rally, or warm-up)."""
    tr = transitions(overlay_path, rosters)
    out, j = [], 0
    for r in rallies:
        while j < len(tr) and tr[j]["t"] < r["end_t"] - 1.0:  # transitions before this rally ended: rallies we missed
            j += 1
        if j < len(tr) and tr[j]["t"] <= r["end_t"] + max_lag_s:
            out.append({"index": r["index"], **tr[j]})
            j += 1
        else:
            out.append({"index": r["index"], "before": None, "after": None, "winner_team": None, "server_team": None,
                        "server": None, "serve_no": None, "verdict": None})
    return out
