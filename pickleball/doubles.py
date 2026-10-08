"""Doubles scoring from tracking, at TEAM level.

Which team is on which side: teams wear different kits, partners the same, so the kit separates teams (not
partners). Per rally the mean shirt-colour signature of the two near-side and the two far-side players; 2-means
over all (rally, side) signatures -> kit A / kit B; each rally gets the kit on the near side (end changes come
out of this for free).

Serving team = team on the side of the rally's first hit (the serve, or the inferred serve).

Side-out scoring (standard doubles, 0-0-2 start): a team keeps serving through server 1 and server 2. In the
sequence of serving teams, a SERVICE TURN is a run of rallies served by one team. A turn of n rallies that ends
in a side-out holds exactly one rally lost by server 1 and the last rally lost by server 2, so the team scored
n - 2 points (n - 1 for the game's first turn, which has only server 2, and +1 if the game ends inside the turn,
since the last point is won by the server). So the score after every turn follows from turn lengths alone.
Inside a turn, which rally server 1 lost is not visible without telling partners apart (the server's side flips
every rally either way). Default: the first rally (the likeliest: P(first rally lost) ~ 0.5-0.55); a per-rally
'partner changed' cue (looks of consecutive servers) picks a better one when it is clear. Player identity
(mark-once) settles it.

Rally scoring (TNPPL super doubles, to 25): the rally winner serves next; always server 1.
"""
import json
from pathlib import Path

import numpy as np

from .court import NET_Y
from .players import select_players


def _rally_frames(A, step=3):
    fps = A["meta"]["fps"]
    for r in sorted(A["rallies"], key=lambda r: r["start_t"]):
        yield r, range(int(r["start_t"] * fps), int(r["end_t"] * fps) + 1, step)


def team_sides(run_dir, A: dict | None = None) -> dict:
    """{rally index: {"near_kit": 0|1|None, "margin": float}}; kit 0 = the far-side kit of the first rally.
    Cached in <run>/team_sides.json."""
    run_dir = Path(run_dir)
    A = A or json.loads((run_dir / "analysis.json").read_text())
    ov = run_dir / "sides_override.json"  # sides known from elsewhere (PPA Europe: overlay serve dot), kit = team
    if ov.exists():
        return {int(k): {"near_kit": v, "margin": 1.0} for k, v in json.loads(ov.read_text()).items()}
    cache = run_dir / "team_sides.json"
    key = [round(r["start_t"], 2) for r in A["rallies"]]
    if cache.exists():
        c = json.loads(cache.read_text())
        if c.get("key") == key:
            return {int(k): v for k, v in c["rallies"].items()}
    recs = {}
    with (run_dir / "detections.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("court") and r.get("people"):
                recs[r["f"]] = r["people"]
    rows = []  # (rally index, side, mean sqrt-signature)
    for r, frames in _rally_frames(A):
        acc = {"near": [], "far": []}
        for f in frames:
            for p in select_players(recs.get(f, []), 2):
                if p.get("sig") is not None:
                    acc[p["side"]].append(np.sqrt(np.asarray(p["sig"], float)))
        for side, v in acc.items():
            if len(v) >= 5:
                rows.append((r["index"], side, np.mean(v, 0)))
    out = {}
    if len(rows) >= 4:
        X = np.stack([x[2] for x in rows])
        first = rows[0][0]
        init = [next((x[2] for x in rows if x[0] == first and x[1] == s), X[i]) for i, s in enumerate(("far", "near"))]
        C = np.stack(init)
        for _ in range(50):
            d = ((X[:, None] - C[None]) ** 2).sum(2)
            lab = d.argmin(1)
            newC = np.stack([X[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(2)])
            if np.allclose(newC, C):
                break
            C = newC
        d = ((X[:, None] - C[None]) ** 2).sum(2)
        by = {}
        for (idx, side, _), dd in zip(rows, d):
            # evidence that the NEAR kit is kit 1: near side close to kit 1 / far side close to kit 0
            e = (dd[0] - dd[1]) if side == "near" else (dd[1] - dd[0])
            by.setdefault(idx, []).append(float(e))
        for idx, es in by.items():
            e = float(np.sum(es))
            out[idx] = {"near_kit": int(e > 0), "margin": round(abs(e), 4)}
    cache.write_text(json.dumps({"key": key, "rallies": out}, indent=0))
    return out


def _smooth_sides(seq: list[dict], min_run: int = 2) -> None:
    """Teams only change ends between games / at the TNPPL switch point: a near-kit flip that does not last
    min_run rallies is a colour slip -> take the neighbours' kit (in place)."""
    k = [s["near_kit"] for s in seq]
    for i in range(len(k)):
        if k[i] is None:
            continue
        run = 1
        j = i + 1
        while j < len(k) and k[j] == k[i]:
            run += 1
            j += 1
        prev = next((k[x] for x in range(i - 1, -1, -1) if k[x] is not None), None)
        nxt = k[j] if j < len(k) else None
        if run < min_run and prev is not None and prev == nxt and prev != k[i]:
            for x in range(i, j):
                seq[x]["near_kit"] = prev


def serving_sequence(run_dir, A: dict | None = None) -> list[dict]:
    """Time-ordered rallies with the serving KIT: {"index","start_t","end_t","near_kit","serve_side","serve_kit"}."""
    A = A or json.loads((Path(run_dir) / "analysis.json").read_text())
    ts = team_sides(run_dir, A)
    seq = []
    for r in sorted(A["rallies"], key=lambda r: r["start_t"]):
        sh = sorted([s for s in A["shots"] if s["rally"] == r["index"]], key=lambda s: s["frame"])
        if not sh:
            continue
        t = ts.get(r["index"], {})
        seq.append({"index": r["index"], "start_t": r["start_t"], "end_t": r["end_t"], "near_kit": t.get("near_kit"),
                    "serve_side": sh[0]["side"], "serve_inferred": bool(sh[0].get("inferred")), "n_shots": len(sh)})
    _smooth_sides(seq)
    for s in seq:
        nk = s["near_kit"]
        s["serve_kit"] = None if nk is None else (nk if s["serve_side"] == "near" else 1 - nk)
    return seq


def _serve_override(run_dir, seq: list[dict]) -> None:
    """Serving team known from elsewhere (PPA Europe: overlay serve dot; kit = team) wins over tracking (in place)."""
    so = Path(run_dir) / "serve_override.json"
    if so.exists():
        ov = {int(k): v for k, v in json.loads(so.read_text()).items()}
        for s in seq:
            if ov.get(s["index"]) is not None:
                s["serve_kit"] = ov[s["index"]]


def _end_changes(seq: list[dict]) -> set:
    """Rally positions where the near kit changes (teams changed ends just before that rally)."""
    out, cur = set(), next((s["near_kit"] for s in seq if s["near_kit"] is not None), None)
    for i, s in enumerate(seq):
        if s["near_kit"] is not None and s["near_kit"] != cur:
            out.add(i)
            cur = s["near_kit"]
    return out


def _resync(score: list, out: list[dict], target: int) -> None:
    """Teams change ends when one team reaches half the target (6 in games to 11, 13 to 25 - TNPPL and PPA
    Las Vegas, checked on every end change: overlay / referee score). At an end change the team that won the
    last rally has exactly that score: resets the drift from missed rallies (in place)."""
    w = next((r["winner_kit"] for r in reversed(out) if r.get("winner_kit") is not None), None)
    if w is None:
        return
    mid = (target + 1) // 2
    score[w] = mid
    score[1 - w] = min(score[1 - w], mid)


def score_sideout(seq: list[dict], target: int = 11, win_by: int = 2, games_to_win: int = 2,
                  game_break_s: float = 60.0, l1_pick=None) -> list[dict]:
    """Side-out doubles reconstruction from the serving-kit sequence. Adds per rally: game, score_before
    ('server-receiver-server#', as referees write it), server_no, winner_kit, how.
    l1_pick(turn: list[dict]) -> position in the turn of the rally server 1 lost (default 0 = the first).
    At an end change inside a game the score is resynced (_resync)."""
    out, game, score, server_no, games_won = [], 1, [0, 0], 2, [0, 0]
    ends = _end_changes(seq)
    i = 0
    while i < len(seq):
        if i in ends and any(score):
            _resync(score, out, target)
        if max(games_won) >= games_to_win:  # match over: warm-down / celebrations
            out += [{**r, "game": None, "after_match": True, "score_before": None, "server_no": None,
                     "winner_kit": None, "how": "after match"} for r in seq[i:]]
            break
        T = seq[i]["serve_kit"]
        if T is None:
            out.append({**seq[i], "game": game, "score_before": None, "server_no": None, "winner_kit": None, "how": "unknown"})
            i += 1
            continue
        j = i  # service turn: the run of rallies served by kit T (a long break or an end change ends it)
        while j + 1 < len(seq) and seq[j + 1]["serve_kit"] == T and seq[j + 1]["start_t"] - seq[j]["end_t"] < game_break_s \
                and j + 1 not in ends:
            j += 1
        turn = seq[i:j + 1]
        l1 = None
        if server_no == 1 and len(turn) >= 2:
            l1 = i + (l1_pick(turn[:-1]) if l1_pick else 0)
        o = 1 - T
        nxt_i = j + 1
        for x in range(i, j + 1):
            r = seq[x]
            before = f"{score[T]}-{score[o]}-{server_no}"
            if x == l1:
                win, how = o, "server 1 lost"
            elif x == j:
                nxt = seq[x + 1] if x + 1 < len(seq) else None
                brk = nxt is None or nxt["start_t"] - r["end_t"] >= game_break_s
                at_gp = score[T] + 1 >= target and score[T] + 1 - score[o] >= win_by
                win, how = (T, "game point won") if (brk and at_gp) else (o, "side out")
            else:
                win, how = T, "point"
            out.append({**r, "game": game, "score_before": before, "server_no": server_no, "winner_kit": win, "how": how})
            if x == l1:
                server_no = 2
            if win == T:
                score[T] += 1
                if score[T] >= target and score[T] - score[o] >= win_by:  # game over
                    games_won[T] += 1
                    game, score, server_no = game + 1, [0, 0], 2
                    nxt_i = x + 1
                    break
        else:
            server_no = 1  # side out: the other team starts with its first server
        i = nxt_i
    return out


def score_rally(seq: list[dict], target: int = 25, win_by: int = 1) -> list[dict]:
    """Rally scoring (TNPPL super doubles): the rally winner serves next. score_before = 'server-receiver'."""
    out, score = [], [0, 0]
    ends = _end_changes(seq)
    for x, r in enumerate(seq):
        if x in ends and any(score):
            _resync(score, out, target)
        T = r["serve_kit"]
        nxt = seq[x + 1] if x + 1 < len(seq) else None
        if T is None:
            out.append({**r, "game": 1, "score_before": None, "server_no": 1, "winner_kit": None, "how": "unknown"})
            continue
        o = 1 - T
        before = f"{score[T]}-{score[o]}"
        if nxt is None or nxt["serve_kit"] is None:
            win, how = T, "last rally" if nxt is None else "unknown next server"
        else:
            win = nxt["serve_kit"]
            how = "kept serve" if win == T else "receiver won"
        score[win] += 1
        out.append({**r, "game": 1, "score_before": before, "server_no": 1, "winner_kit": win, "how": how})
        if score[win] >= target and score[win] - score[1 - win] >= win_by:
            for rr in seq[x + 1:]:
                out.append({**rr, "game": None, "after_match": True, "score_before": None, "server_no": None,
                            "winner_kit": None, "how": "after match"})
            break
    return out


def formation_serve_side(run_dir, A: dict, window_s: float = 1.2) -> dict:
    """Serving side from the doubles serve formation: at the serve both servers stand back at the baseline,
    while the receiving team has the receiver back and the partner up near the kitchen line. Over the first
    main-view frames of each rally (before the 3rd shot): per side, the distance from the net of the side's
    player closest to the net; the side whose closest player is nearer the net is RECEIVING.
    Returns {rally index: {"serve_side": "near"|"far"|None, "gap_ft": far-min minus near-min distance}}."""
    run_dir = Path(run_dir)
    fps = A["meta"]["fps"]
    want = {}
    for r in A["rallies"]:
        sh = sorted([s for s in A["shots"] if s["rally"] == r["index"]], key=lambda s: s["frame"])
        real = [s for s in sh if not s.get("inferred")]
        if not real:
            continue
        # frames from the rally start up to the 2nd real hit (the receiving formation holds until the return)
        f_end = real[1]["frame"] if len(real) > 1 else real[0]["frame"] + int(0.5 * fps)
        f0 = max(int(r["start_t"] * fps) - int(window_s * fps), f_end - int(3 * fps))
        want[r["index"]] = (f0, f_end)
    near_d = {k: [] for k in want}
    far_d = {k: [] for k in want}
    frames = {}
    for k, (a, b) in want.items():
        for f in range(a, b + 1):
            frames.setdefault(f, []).append(k)
    with (run_dir / "detections.jsonl").open() as fh:
        for line in fh:
            rec = json.loads(line)
            ks = frames.get(rec["f"])
            if not ks or not rec.get("court"):
                continue
            ps = select_players(rec.get("people", []), 2)
            nd = [abs(p["court"][1] - NET_Y) for p in ps if p["side"] == "near"]
            fd = [abs(p["court"][1] - NET_Y) for p in ps if p["side"] == "far"]
            if len(nd) == 2 and len(fd) == 2:
                for k in ks:
                    near_d[k].append(min(nd))
                    far_d[k].append(min(fd))
    out = {}
    for k in want:
        if len(near_d[k]) < 3:
            out[k] = {"serve_side": None, "gap_ft": None}
            continue
        gap = float(np.median(far_d[k]) - np.median(near_d[k]))  # > 0: near side has a player closer to the net
        out[k] = {"serve_side": "far" if gap > 0 else "near", "gap_ft": round(gap, 1)}
    return out


FORMATION_MIN_GAP_FT = 5.0  # formation decides the serving side when the gap is at least this (else first hit)


def reconstruct(run_dir, A: dict, scoring: str = "sideout", target: int = 11, win_by: int = 2,
                games_to_win: int = 2, game_break_s: float = 60.0, serve_cue: str = "combined") -> list[dict]:
    """Team-level doubles match reconstruction. serve_cue: 'firsthit' (side of the first hit), 'formation'
    (formation_serve_side) or 'combined' (formation when its gap >= FORMATION_MIN_GAP_FT, else first hit)."""
    seq = serving_sequence(run_dir, A)
    if serve_cue != "firsthit":
        fm = formation_serve_side(run_dir, A)
        for s in seq:
            f = fm.get(s["index"], {})
            if f.get("serve_side") and (serve_cue == "formation" or abs(f["gap_ft"]) >= FORMATION_MIN_GAP_FT):
                s["serve_side"] = f["serve_side"]
                s["serve_cue"] = "formation"
            else:
                s["serve_cue"] = "first hit"
            nk = s["near_kit"]
            s["serve_kit"] = None if nk is None else (nk if s["serve_side"] == "near" else 1 - nk)
    _serve_override(run_dir, seq)
    if scoring == "rally":
        return score_rally(seq, target, win_by)
    return score_sideout(seq, target, win_by, games_to_win, game_break_s)
