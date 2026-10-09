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


def _reconstruct_once(rallies, switch_after: set, target: int, game_break_s: float, add_missed: bool = False,
                      win_by: int = 2, games_to_win: int = 2, scoring: str = "sideout"):
    """One pass of the rules for a given set of end changes (players change ends after rally k for k in
    switch_after). Scores are per player SLOT ("A" = near player at the start).
    scoring "sideout": only the server scores; "rally": the rally winner scores (and serves next)."""
    out, game, swapped = [], 1, False
    score = {"A": 0, "B": 0}
    parity_hits = 0
    first_server, violations = {}, 0  # rule: the first server alternates from game to game
    games_won = {"A": 0, "B": 0}      # best of 3: a player's 2nd game ends the match (games_to_win)
    match_over = False
    for k, r in enumerate(rallies):
        slot = lambda side, sw: ("A" if side == "near" else "B") if not sw else ("B" if side == "near" else "A")
        s = slot(r["server_side"], swapped)
        o = "B" if s == "A" else "A"
        if game not in first_server:
            first_server[game] = s
            violations += game > 1 and first_server.get(game - 1) == s
        missed, parity_ok = 0, None
        x = r.get("serve_x")
        if x is not None and abs(x - 10.0) >= 2.5 and not r.get("serve_inferred"):  # clearly on one half
            parity_ok = _right_half(r["server_side"], x) == (score[s] % 2 == 0)
            parity_hits += parity_ok
            if not parity_ok and add_missed:  # a point the server won may be missing
                score[s] += 1
                missed = 1
        before = f"{score[s]}-{score[o]}"
        nxt = rallies[k + 1] if k + 1 < len(rallies) else None
        gap = (nxt["start_t"] - r["end_t"]) if nxt else None
        game_point = score[s] + 1 >= target and score[s] + 1 - score[o] >= win_by
        sw_next = swapped != (k in switch_after)
        if nxt is None or (game_point and gap > game_break_s):
            w = s
        else:
            w = slot(nxt["server_side"], sw_next)  # the rally winner serves next
        if w == s or scoring == "rally":
            score[w] += 1
        if match_over:  # after someone won 2 games: warm-down / post-match play, not part of the match
            out.append({**r, "game": None, "after_match": True, "server_slot": s, "winner_slot": w,
                        "score_before": None, "parity_ok": None, "missed_points_before": 0, "swapped": swapped})
            swapped = sw_next
            continue
        out.append({**r, "game": game, "server_slot": s, "winner_slot": w, "score_before": before,
                    "parity_ok": parity_ok, "missed_points_before": missed, "swapped": swapped})
        long_break = gap is not None and gap > game_break_s
        # a long break near 11 ends the game only if the NEXT game's first server is the other player
        # (the first server alternates by game); a timeout keeps the same server
        next_first_ok = nxt is not None and slot(nxt["server_side"], sw_next) != first_server.get(game)
        rallies_left = 0  # rallies still to come before the next long break (post-match hitting comes after one)
        for j in range(k + 1, len(rallies)):
            if rallies[j]["start_t"] - rallies[j - 1]["end_t"] > game_break_s and j > k + 1:
                break
            rallies_left += 1
        match_would_end = games_won[w] + 1 >= games_to_win
        reached = score[w] >= target and score[w] - score[o if w == s else s] >= win_by
        assumed = long_break and next_first_ok and w == s and score[s] >= target - 2 and score[s] - score[o] >= win_by
        fake_match_end = match_would_end and rallies_left >= 3  # play continues: that was not match point
        if (reached or assumed) and not fake_match_end:
            games_won[w] += 1
            match_over = games_won[w] >= games_to_win
            game += 1  # game over (or its last points were not seen and a long break followed)
            score = {"A": 0, "B": 0}
        swapped = sw_next
    return out, (parity_hits, violations)


def reconstruct(rallies: list[dict], target: int = 11, game_break_s: float = 40.0, switch_break_s: float = 40.0,
                switch_after: set | None = None, max_breaks: int = 10, **rules) -> list[dict]:
    """Rules-based reconstruction (singles, side-out scoring). rallies (time order):
    {"start_t","end_t","server_side","serve_x"}.

    Players only change ends during a long break, but tournaments differ (between games, at 6, both).
    So the end changes are NOT assumed: every long break (> switch_break_s) is a candidate, and the
    combination that best fits the rules is chosen: the first server alternates from game to game, and
    the serve-position rule (right half on an even own score, left on odd) over the whole match. Returns copies with game, server_slot, winner_slot, score_before
    (server-receiver), parity_ok, missed_points_before, swapped (True = slot A is on the far side)."""
    if switch_after is not None:
        return _reconstruct_once(rallies, set(switch_after), target, game_break_s, **rules)[0]
    cands = [k for k in range(len(rallies) - 1)
             if rallies[k + 1]["start_t"] - rallies[k]["end_t"] > switch_break_s]
    cands = sorted(cands, key=lambda k: rallies[k + 1]["start_t"] - rallies[k]["end_t"], reverse=True)[:max_breaks]
    best = None
    for mask in range(1 << len(cands)):
        sw = {cands[i] for i in range(len(cands)) if mask >> i & 1}
        out, (hits, viol) = _reconstruct_once(rallies, sw, target, game_break_s, **rules)
        key = (-viol, hits, -len(sw))  # first server alternates by game; best serve-position fit; fewest end changes
        if best is None or key > best[0]:
            best = (key, out)
    return best[1]


def name_sides(recon: list[dict], side_votes: list[dict]) -> list[dict]:
    """Attach player names to slots A/B with ONE decision for the whole match: every rally's shirt-colour
    votes (by side) are converted to slot votes using that rally's side->slot link, then pooled.
    side_votes[k] = {"near": Counter(names), "far": Counter(names)}. Adds "server", "receiver", "winner"."""
    from collections import Counter
    names = sorted({n for v in side_votes for side in v.values() for n in side})
    tally = Counter()
    for r, v in zip(recon, side_votes):
        near_slot = "B" if r["swapped"] else "A"
        for side, cnt in v.items():
            sl = near_slot if side == "near" else ("A" if near_slot == "B" else "B")
            for name, n in cnt.items():
                tally[(sl, name)] += n
    if len(names) == 2:
        a, b = names
        keep = tally[("A", a)] + tally[("B", b)] >= tally[("A", b)] + tally[("B", a)]
        slot_name = {"A": a, "B": b} if keep else {"A": b, "B": a}
    else:
        slot_name = {"A": names[0] if names else "A", "B": names[1] if len(names) > 1 else "B"}
    out = []
    for r in recon:
        other = "B" if r["server_slot"] == "A" else "A"
        out.append({**r, "server": slot_name[r["server_slot"]], "receiver": slot_name[other],
                    "winner": slot_name[r["winner_slot"]]})
    return out


def _kit_end_changes(votes: list[dict], persist: int = 2) -> set:
    """End changes from the kit-colour identity of the near-side hitters (teams in clearly different kits):
    an end change after rally k when the near player's identity flips and stays flipped for `persist` rallies.
    Needed for rotating team formats, where the near player's LOOK changes at every rotation."""
    near = [v["near"].most_common(1)[0][0] if v["near"] else
            (next((n for n in {x for vv in votes for x in vv["far"]} if n not in v["far"]), None) if v["far"] else None)
            for v in votes]
    out, cur = set(), next((n for n in near if n), None)
    for k in range(len(near)):
        nxt = [n for n in near[k + 1:k + 1 + persist] if n]
        if cur and len(nxt) == persist and all(n != cur for n in nxt):
            out.add(k)
            cur = nxt[0]
    return out


def match_from_analysis(A: dict, run_dir=None, **kw) -> list[dict]:
    """Rules-based reconstruction straight from an analysis.json dict (one entry per rally, time order).
    With run_dir, end changes are OBSERVED from the near player's look (appearance.end_changes);
    otherwise (or if the players look alike) they are searched with the rules."""
    from collections import Counter
    name = {pid: p.get("name") or pid for pid, p in A["players"].items()}
    seq, votes = [], []
    for r in sorted(A["rallies"], key=lambda r: r["start_t"]):
        sh = sorted([s for s in A["shots"] if s["rally"] == r["index"]], key=lambda s: s["frame"])
        if not sh:
            continue
        seq.append({"index": r["index"], "start_t": r["start_t"], "end_t": r["end_t"], "server_side": sh[0]["side"],
                    "serve_x": sh[0]["hitter_court"][0] if sh[0].get("hitter_court") else None,
                    "serve_inferred": bool(sh[0].get("inferred"))})
        v = {"near": Counter(), "far": Counter()}
        for s in sh:
            if not s.get("inferred"):
                v[s["side"]][name.get(s["player"], s["player"])] += 1
        votes.append(v)
    if kw.pop("ends_from_kit", False) and "switch_after" not in kw:
        kw["switch_after"] = _kit_end_changes(votes)
    if run_dir is not None and "switch_after" not in kw:
        from .appearance import end_changes, near_looks
        sw = end_changes(near_looks(run_dir, seq))
        if sw is not None:
            kw["switch_after"] = sw
    return name_sides(reconstruct(seq, **kw), votes)
