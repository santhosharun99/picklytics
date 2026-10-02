"""Who served and who won each rally, from our own tracking (no scoreboard needed).

Server = hitter of the rally's first shot.
Winner: look at what happened after the LAST hit of the rally
  - next bounce lands on the hitter's own side   -> ball never crossed the net: hitter LOSES ("net")
  - next bounce lands outside the court lines    -> hitter LOSES ("out")
  - next bounce lands inside the opponent's court and the opponent never hits it -> hitter WINS ("winner")
  - no bounce seen before the rally ends         -> opponent failed to return it in the air or it flew
                                                    out unseen; default: hitter wins ("unreturned", low confidence)
The scoreboard (scoreboard.py) is only used as an independent cross-check.
"""
from .court import COURT_L, COURT_W, NET_Y

IN_MARGIN_FT = 0.5  # line calls within this of a line count as in (ball touching the line is in)


def _inside(xy, margin=IN_MARGIN_FT):
    return -margin <= xy[0] <= COURT_W + margin and -margin <= xy[1] <= COURT_L + margin


def rally_outcome(shots: list[dict], bounces: list[dict], end_frame: int, fps: float, players: tuple[str, str],
                  next_start_frame: int | None = None) -> dict:
    """shots: this rally's shots (with frame, player, side); bounces: all bounces (frame, court, side)."""
    if not shots:
        return {"server": None, "winner": None, "reason": None, "confidence": 0.0}
    shots = sorted(shots, key=lambda s: s["frame"])
    server, last = shots[0]["player"], shots[-1]
    hitter = last["player"]
    opponent = players[1] if hitter == players[0] else players[0]
    horizon = min(end_frame + int(2.5 * fps), next_start_frame or 10 ** 9)
    after = [b for b in sorted(bounces, key=lambda b: b["frame"]) if last["frame"] < b["frame"] <= horizon]
    if after:
        b = after[0]
        own_side = (b["court"][1] < NET_Y) == (last["side"] == "far")
        if own_side:
            return {"server": server, "winner": opponent, "reason": "net", "confidence": 0.7, "last_hitter": hitter}
        if not _inside(b["court"]):
            return {"server": server, "winner": opponent, "reason": "out", "confidence": 0.6, "last_hitter": hitter}
        return {"server": server, "winner": hitter, "reason": "winner", "confidence": 0.8, "last_hitter": hitter}
    return {"server": server, "winner": hitter, "reason": "unreturned", "confidence": 0.4, "last_hitter": hitter}


def outcomes_for_analysis(A: dict) -> dict:
    """{rally index: outcome} for an analysis.json dict (names resolved)."""
    fps = A["meta"]["fps"]
    name = {pid: p.get("name") or pid for pid, p in A["players"].items()}
    players = tuple(name.values())[:2]
    rallies = sorted(A["rallies"], key=lambda r: r["start_t"])
    out = {}
    for k, r in enumerate(rallies):
        shots = [dict(s, player=name.get(s["player"], s["player"])) for s in A["shots"] if s["rally"] == r["index"]]
        nxt = int(rallies[k + 1]["start_t"] * fps) if k + 1 < len(rallies) else None
        out[r["index"]] = rally_outcome(shots, A["bounces"], int(r["end_t"] * fps), fps, players, nxt)
    return out


def winners_from_serves(rallies: list[dict], game_gap_s: float = 100.0) -> list[dict]:
    """Side-out scoring logic: the server of the NEXT rally tells who won this one.

    rallies: in time order, each {"start_t", "end_t", "server", "receiver"} (from tracking: server =
    first hitter). Same server next rally -> server won; serve changed -> receiver won (side out).
    The last rally before a long break (end of a game) is credited to the server: in side-out
    scoring only the serving player can win a game. Returns a copy with "winner" and "game".
    """
    out, game = [], 1
    for k, r in enumerate(rallies):
        nxt = rallies[k + 1] if k + 1 < len(rallies) else None
        gap = (nxt["start_t"] - r["end_t"]) if nxt else None
        if nxt is None or gap > game_gap_s:
            w, how = r["server"], "game end"
        elif nxt["server"] == r["server"]:
            w, how = r["server"], "kept serve"
        else:
            w, how = r["receiver"], "side out"
        out.append({**r, "winner": w, "how": how, "game": game})
        if nxt is not None and gap > game_gap_s:
            game += 1
    return out
