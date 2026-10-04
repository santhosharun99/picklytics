"""Win probability and point importance for side-out scoring (singles, games to 11 win by 2, best of 3).

Markov model of the rally sequence: only the server scores; a lost serve passes the serve.
- Live win probability: before each rally, each player's chance of winning a rally on their own serve is
  estimated from the rallies played SO FAR, shrunk toward P0 (empirical Bayes, PRIOR_N pseudo-rallies), so the
  curve starts near 50% and learns as the match goes (no hindsight).
- Importance of a rally = P(match | A wins it) - P(match | A loses it) with both players at P0: it depends only
  on the situation (score, games, server), the classic definition (Morris 1977; tennis: Klaassen & Magnus 2001),
  here for side-out scoring. Big-point performance = rally win rate on the top quarter by importance.
"""
from functools import lru_cache

PRIOR_N = 20
P0 = 0.48   # server wins the rally: pooled over our two pro singles matches (51/107)
TARGET, GAMES = 11, 2


def game_table(p_a: float, p_b: float, target: int = TARGET):
    """W[(a, b, srv)] = P(A wins the game) at score a (A) - b (B), srv 0 = A serving, 1 = B serving."""
    top = target + 1
    W = {}

    def norm(a, b):
        m = min(a, b) - (target - 1)
        return (a - m, b - m) if m > 0 else (a, b)

    def val(a, b, s):
        if a >= target and a - b >= 2:
            return 1.0
        if b >= target and b - a >= 2:
            return 0.0
        return W.get((*norm(a, b), s), 0.5)

    states = [(a, b, s) for a in range(top + 1) for b in range(top + 1) for s in (0, 1)
              if not (a >= target and a - b >= 2) and not (b >= target and b - a >= 2)]
    for _ in range(400):  # value iteration (deuce states loop)
        delta = 0.0
        for a, b, s in states:
            v = (p_a * val(a + 1, b, 0) + (1 - p_a) * val(a, b, 1)) if s == 0 else \
                (p_b * val(a, b + 1, 1) + (1 - p_b) * val(a, b, 0))
            delta = max(delta, abs(v - W.get((a, b, s), 0.5)))
            W[(a, b, s)] = v
        if delta < 1e-9:
            break
    return lambda a, b, s: val(a, b, s)


def match_prob(game_w, ga: int, gb: int, a: int, b: int, srv: int, first: int) -> float:
    """P(A wins the match): games ga-gb, in-game score a-b, srv serving, first = first server of this game."""
    @lru_cache(None)
    def fresh(ga, gb, first):
        if ga >= GAMES:
            return 1.0
        if gb >= GAMES:
            return 0.0
        w = game_w(0, 0, first)
        return w * fresh(ga + 1, gb, 1 - first) + (1 - w) * fresh(ga, gb + 1, 1 - first)
    if ga >= GAMES:
        return 1.0
    if gb >= GAMES:
        return 0.0
    w = game_w(a, b, srv)
    return w * fresh(ga + 1, gb, 1 - first) + (1 - w) * fresh(ga, gb + 1, 1 - first)


def annotate(rallies: list[dict], a: str, b: str) -> dict:
    """Adds wp_before / wp_after (live P(a wins the match)) and importance to each rally with a known score."""
    tables = {}

    def table(pa, pb):
        k = (round(pa, 2), round(pb, 2))
        if k not in tables:
            tables[k] = game_table(*k)
        return tables[k]

    neutral = table(P0, P0)
    won, n = {a: 0, b: 0}, {a: 0, b: 0}
    games, first, cur = {a: 0, b: 0}, {}, None
    seq = [r for r in rallies if r.get("game") and r.get("score_before") and r.get("server") in (a, b)]
    for k, r in enumerate(seq):
        g = r["game"]
        if cur is not None and g != cur:  # previous game finished: its last rally's winner took it
            last = seq[k - 1]
            if last.get("winner"):
                games[last["winner"]] += 1
        cur = g
        first.setdefault(g, r["server"])
        s_pts, o_pts = map(int, r["score_before"].split("-"))
        srv = 0 if r["server"] == a else 1
        pa, pb = (s_pts, o_pts) if srv == 0 else (o_pts, s_pts)
        f = 0 if first[g] == a else 1
        rate = lambda x: (won[x] + PRIOR_N * P0) / (n[x] + PRIOR_N)
        live = table(rate(a), rate(b))
        mp = lambda W, x, y, s: match_prob(W, games[a], games[b], x, y, s, f)
        outcomes = lambda W: (mp(W, pa + 1, pb, 0), mp(W, pa, pb, 1)) if srv == 0 else (mp(W, pa, pb, 0), mp(W, pa, pb + 1, 1))
        if_a, if_b = outcomes(live)
        r["wp_before"] = round(mp(live, pa, pb, srv), 3)
        if r.get("winner"):
            r["wp_after"] = round(if_a if r["winner"] == a else if_b, 3)
        na, nb = outcomes(neutral)
        r["importance"] = round(na - nb, 3)
        if r.get("winner"):  # learn from this rally for the next ones
            won[r["server"]] += r["winner"] == r["server"]
            n[r["server"]] += 1
    return {"player": a, "opponent": b, "p0": P0, "prior_n": PRIOR_N,
            "serve_rally_win": {x: {"won": won[x], "n": n[x]} for x in (a, b)}}


def big_points(name: str, rallies: list[dict], top_frac: float = 0.25) -> dict:
    """Rally win rate on the most important rallies (top quarter by importance) vs the rest."""
    rs = [r for r in rallies if r.get("winner") and r.get("importance") is not None]
    if len(rs) < 8:
        return {}
    cut = sorted(r["importance"] for r in rs)[int(len(rs) * (1 - top_frac))]
    big = [r for r in rs if r["importance"] >= cut]
    rest = [r for r in rs if r["importance"] < cut]
    pct = lambda x: round(100 * sum(r["winner"] == name for r in x) / len(x)) if x else None
    return {"big_won_pct": pct(big), "n_big": len(big), "rest_won_pct": pct(rest), "n_rest": len(rest),
            "big_rallies": [r["id"] for r in big]}
