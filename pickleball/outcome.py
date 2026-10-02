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


# ---------------------------------------------------------------------------------------------
# Rules-based match reconstruction (singles, side-out scoring). Works on court SIDES, not player
# names, because which side served is what tracking sees reliably:
#   - rally winner = the side that serves next (the rally winner always serves next in singles);
#   - only the server scores; a game ends at 11, win by 2; players change ends after each game;
#   - serve position check: a player serves from the RIGHT half when their own score is even,
#     from the LEFT when odd. A mismatch means a point was missed -> credit it to the server.
#   - at game point a long break before the next rally (> game_break_s) = the game ended (server won).
# Names are then attached per game (one side->player vote per game; players swap ends between games).
# ---------------------------------------------------------------------------------------------

def _right_half(side: str, x: float) -> bool:
    """Is court x on the right half from that player's point of view? (camera behind the near baseline)"""
    return x > 10.0 if side == "near" else x < 10.0


def reconstruct(rallies: list[dict], target: int = 11, game_break_s: float = 40.0, use_parity: bool = True) -> list[dict]:
    """rallies (time order): {"start_t","end_t","server_side","serve_x"}. Returns copies with game, score_before
    (server-receiver), winner_side, parity_ok, missed_points_before."""
    out, game = [], 1
    score = {"near": 0, "far": 0}
    for k, r in enumerate(rallies):
        s = r["server_side"]
        o = "far" if s == "near" else "near"
        missed = 0
        parity_ok = None
        if use_parity and r.get("serve_x") is not None:
            parity_ok = _right_half(s, r["serve_x"]) == (score[s] % 2 == 0)
            if not parity_ok:  # a point the server won is missing (the server kept the serve)
                score[s] += 1
                missed = 1
        before = f"{score[s]}-{score[o]}"
        nxt = rallies[k + 1] if k + 1 < len(rallies) else None
        gap = (nxt["start_t"] - r["end_t"]) if nxt else None
        game_point = score[s] + 1 >= target and score[s] + 1 - score[o] >= 2
        if nxt is None:
            w = s
        elif game_point and gap > game_break_s:
            w = s
        else:
            w = s if nxt["server_side"] == s else o
        if w == s:
            score[s] += 1
        out.append({**r, "game": game, "score_before": before, "winner_side": w, "parity_ok": parity_ok,
                    "missed_points_before": missed})
        if score[s] >= target and score[s] - score[o] >= 2:  # game over: new game, players change ends
            game += 1
            score = {"near": 0, "far": 0}
    return out


def name_sides(recon: list[dict], side_votes: list[dict]) -> list[dict]:
    """Attach player names: side_votes[k] = {"near": Counter(names), "far": Counter(names)} for rally k.
    One mapping per game by majority; adds "server", "receiver", "winner"."""
    from collections import Counter
    out = []
    games = sorted({r["game"] for r in recon})
    mapping = {}
    for g in games:
        tally = Counter()
        for r, v in zip(recon, side_votes):
            if r["game"] == g:
                for name, n in v.get("near", {}).items():
                    tally[name] += n
                for name, n in v.get("far", {}).items():
                    tally[name] -= n
        names = sorted({n for v in side_votes for side in v.values() for n in side})
        if len(names) == 2:
            a, b = names
            near = a if tally[a] - tally[b] >= 0 else b
            mapping[g] = {"near": near, "far": b if near == a else a}
    for r in recon:
        m = mapping.get(r["game"], {})
        srv, rcv = m.get(r["server_side"]), m.get("far" if r["server_side"] == "near" else "near")
        out.append({**r, "server": srv, "receiver": rcv, "winner": m.get(r["winner_side"])})
    return out


def match_from_analysis(A: dict, **kw) -> list[dict]:
    """Rules-based reconstruction straight from an analysis.json dict (one entry per rally, time order)."""
    from collections import Counter
    name = {pid: p.get("name") or pid for pid, p in A["players"].items()}
    seq, votes = [], []
    for r in sorted(A["rallies"], key=lambda r: r["start_t"]):
        sh = sorted([s for s in A["shots"] if s["rally"] == r["index"]], key=lambda s: s["frame"])
        if not sh:
            continue
        seq.append({"index": r["index"], "start_t": r["start_t"], "end_t": r["end_t"], "server_side": sh[0]["side"],
                    "serve_x": sh[0]["hitter_court"][0] if sh[0].get("hitter_court") else None})
        v = {"near": Counter(), "far": Counter()}
        for s in sh:
            if not s.get("inferred"):
                v[s["side"]][name.get(s["player"], s["player"])] += 1
        votes.append(v)
    return name_sides(reconstruct(seq, **kw), votes)
