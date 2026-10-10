"""Evidence-fusion scoring: the single most likely sequence of points of a match (TNPPL: one game per match).

No source is trusted blindly. The decoder walks the detected rallies in time order over hidden score states
    (score A, score B, serving team, server number 1/2, which partner serves)      side-out doubles
    (score A, score B, serving team)                                                side-out singles, rally scoring
and finds the cheapest path (Viterbi, costs in nats = -log probability) that fits:
  1. tracking cues, each with a measured reliability (SCORING_CUES below; training/validate_scoring.py --calibrate):
     serving team (doubles serve formation / first hit; singles first hitter), which team is on the near side
     (teams change ends when one reaches half the target: 6 of 11, 13 of 25), singles serve position (right half
     on an even own score), doubles server's name from the mark-once identity (server 1 -> partner = server 2);
  2. the score overlay, only where it is visible: every stable overlay state is a soft anchor "the score was X
     before the next rally", with a lag allowance (operator one point late), and serve-line slips cost less than
     score disagreements;
  3. game rules: only the server scores in side-out, server 1 -> server 2 -> side out (0-0-2 start), rally
     scoring = winner serves next, scores never go down, the game ends at the target;
  4. the official final score (match card) as a hard end constraint.
Points the cameras never showed (cut-aways, replays, close-ups) are explicit MISSED points between detected
rallies: their number is a Poisson guess from the time gap, and the overlay (when visible) or the rules / final
score say what they were. A detected rally can also be "not a point" (let, warm-up, split rally) or after the
match. Teams are in OVERLAY ROW / config 'teams' order (team 0, team 1); tracking kits map onto them by a flip
that is decided by identity marks (doubles names), config names (singles) or, failing both, by the decode itself.
Confidence per rally from max-marginals (cost of the best path with a different winner / server):
    provenance 'overlay'  - the overlay showed the state before and after the point and the decode agrees;
               'rules'    - no overlay, but every other answer costs >= CERTAIN_NATS more (forced);
               'tracking' - decided by the tracking cues within the rules;
    confidence 'certain' for overlay / rules, 'inferred' for tracking.
"""
import difflib
import json
import math
from pathlib import Path

P = dict(
    spurious=3.0,     # a detected rally that is not a point (let, warm-up, split rally, replay not caught)
    after=1.0,        # a detected rally after the game was decided
    ovl_bad=10.0,     # overlay score contradicts the decoded state
    ovl_lag=4.0,      # overlay still shows the state before the previous point (operator late)
    ovl_team=2.5,     # score agrees, serving team differs (serve-line OCR / operator slip)
    ovl_no=1.5,       # serve number differs
    ovl_player=0.7,   # serve-line name is the partner
    ovl_skip=5.0,     # an overlay state between two rallies that the decoded points never pass through
    point_s=25.0,     # one missed point per this many seconds of gap beyond gap0_s (Poisson mean)
    gap0_s=12.0,
    lam_min=0.03,
    miss_every_s=9.0, # at most one missed point per this many seconds of gap
    max_miss=14,
    unfinished=8.0,   # no official final: path that does not finish the game
    beam=30.0,
)
CERTAIN_NATS = 6.0
# tracking-cue reliabilities (probability the cue is right), measured against the full-overlay decode on the
# 23 TNPPL matches (training/validate_scoring.py --calibrate, 2026-10-10: formation 355/397, first hit 18/23,
# inferred first hit 40/56, near side 475/476 (capped), server name 68/91, singles first hit 54/69 (inferred 49/77),
# serve position 21/24). The rally-end model (net / out / in after the last hit) agreed on 51% -> not used.
SCORING_CUES = {
    "formation": 0.89, "first hit": 0.78, "first hit inferred": 0.71, "singles first hit": 0.78,
    "singles first hit inferred": 0.64, "near": 0.97, "player": 0.75, "parity": 0.87,
}


def _nll(rel, ok):
    rel = min(max(rel, 0.5), 0.99)
    return -math.log(rel if ok else 1.0 - rel)


class Game:
    """Rules of one game. State = (a, b, T, n, p): scores of team 0 / 1, serving team, server number (side-out
    doubles; else 0), partner index of the server within the serving team (side-out doubles; else 0)."""

    def __init__(self, scoring="sideout", target=11, win_by=1, doubles=True):
        self.sideout = scoring == "sideout"
        self.doubles = doubles
        self.players = doubles and self.sideout
        self.target, self.win_by = target, win_by
        self.mid = (target + 1) // 2  # TNPPL / PPA: teams change ends when one team reaches half the target

    def starts(self):
        if self.players:  # 0-0-2: the first serving team has only its second server
            return [(0, 0, T, 2, p) for T in (0, 1) for p in (0, 1)]
        return [(0, 0, T, 0, 0) for T in (0, 1)]

    def over(self, s):
        a, b = s[0], s[1]
        return max(a, b) >= self.target and abs(a - b) >= self.win_by

    def succ(self, s):
        """[(next state, winning team, how)] for one point played from state s."""
        a, b, T, n, p = s
        sc = [a, b]
        sc[T] += 1
        won = (sc[0], sc[1], T, n, p)
        if self.sideout:
            if self.players:
                if n == 1:
                    return [(won, T, "point"), ((a, b, T, 2, 1 - p), 1 - T, "server 1 lost")]
                return [(won, T, "point")] + [((a, b, 1 - T, 1, q), 1 - T, "side out") for q in (0, 1)]
            return [(won, T, "point"), ((a, b, 1 - T, 0, 0), 1 - T, "side out")]
        lost = [a, b]
        lost[1 - T] += 1
        return [(won, T, "point"), ((lost[0], lost[1], 1 - T, 0, 0), 1 - T, "receiver won")]

    def switched(self, s):
        return max(s[0], s[1]) >= self.mid

    def score_text(self, s):
        """'server-receiver[-server#]' as referees call it."""
        a, b, T, n, _ = s
        sc = (a, b)
        return f"{sc[T]}-{sc[1 - T]}" + (f"-{n}" if self.players else "")


# --------------------------------------------------------------------------------------------- overlay

def _roster_match(name, names, cutoff=0.6):
    if not name:
        return None
    m = difflib.get_close_matches(name.upper(), [n.upper() for n in names], n=1, cutoff=cutoff)
    return [n.upper() for n in names].index(m[0]) if m else None


def overlay_segments(readings, squads, team_rosters=None, use_server=True, use_serve_no=True, hide=None,
                     min_stable=2, burst_s=9.0):
    """Stable overlay states in time order: [{"score": (a, b), "team", "no", "p", "name", "t0", "t1", "n"}].
    squads: [[team 0 names], [team 1 names]] (whole squads: serve-line OCR slips); team_rosters: the match's own
    players per team (partner index p). hide: [(t0, t1), ...] time spans treated as invisible (validation); the
    one-game monotonic score fit is then redone on what stays visible. team / no / p None = not read."""
    rs = [r for r in readings if not hide or not any(a <= r["t"] < b for a, b in hide)]
    if not rs:
        return []
    if hide:
        from .tnppl_overlay import monotonic_fit
        fits = [monotonic_fit([r["scores"][i] for r in rs], [r["conf"] for r in rs]) for i in (0, 1)]
        rs = [dict(r, fit=[fits[0][k], fits[1][k]]) for k, r in enumerate(rs)]
    allnames = [n for sq in squads for n in sq]
    segs = []
    for r in rs:
        team = p = None
        name = r.get("server") if use_server else None
        if name:
            k = _roster_match(name, allnames)
            if k is not None:
                team = 0 if k < len(squads[0]) else 1
                if team_rosters:
                    p = _roster_match(name, team_rosters[team], 0.5)
        no = r.get("serve_no") if (use_server and use_serve_no) else None
        st = {"score": tuple(r["fit"]), "team": team, "no": no, "p": p, "name": name.title() if team is not None and name else None}
        last = segs[-1] if segs else None
        if last and last["score"] == st["score"] and all(st[k] is None or last[k] is None or st[k] == last[k]
                                                         for k in ("team", "no", "p")):
            for k in ("team", "no", "p", "name"):
                if last[k] is None:
                    last[k] = st[k]
            last["t1"], last["n"] = r["t"], last["n"] + 1
            continue
        segs.append({**st, "t0": r["t"], "t1": r["t"], "n": 1})
    segs = [s for s in segs if s["n"] >= min_stable]
    out = []  # merge neighbours that became identical after dropping flickers
    for s in segs:
        if out and all(out[-1][k] == s[k] for k in ("score", "team", "no", "p")):
            out[-1]["t1"], out[-1]["n"] = s["t1"], out[-1]["n"] + s["n"]
        else:
            out.append(dict(s))
    # catch-up bursts: states shown for less than burst_s before the next change (no point is that short) are the
    # operator typing in points late - their times say nothing; keep only the state the burst ends on
    return [s for k, s in enumerate(out) if k + 1 == len(out) or out[k + 1]["t0"] - s["t0"] >= burst_s]


def assign_anchors(events, segs):
    """Each overlay state anchors the state BEFORE the first rally that ends after the state first shows (an
    update shown during a rally = the operator was late with the previous point). Several states before one
    rally: the last one is that rally's 'pre' anchor, the earlier ones are 'mid' anchors = points we missed in
    the gap before it (in order). Returns (pre: {event: seg}, mid: {event: [seg, ...]}); event len(events) =
    after the last rally."""
    pre, mid = {}, {}
    for s in segs:
        e = next((k for k, ev in enumerate(events) if ev["end_t"] > s["t0"] + 0.5), len(events))
        mid.setdefault(e, []).append(s)
    for e, lst in list(mid.items()):
        if e < len(events):
            pre[e] = lst[-1]
            mid[e] = lst[:-1]
    return pre, mid


# --------------------------------------------------------------------------------------------- decoder

class _Trellis:
    def __init__(self):
        self.F, self.key, self.par, self.lab, self.edges = [], [], [], [], []

    def node(self, layer, key, u, w, lab):
        c = (self.F[u] if u is not None else 0.0) + w
        v = layer.get(key)
        if v is None:
            v = layer[key] = len(self.F)
            self.F.append(c), self.key.append(key), self.par.append(u), self.lab.append(lab)
        elif c < self.F[v]:
            self.F[v], self.par[v], self.lab[v] = c, u, lab
        if u is not None:
            self.edges.append((u, v, w, lab))
        return v

    def prune(self, layer, beam):
        if not layer:
            return layer
        m = min(self.F[v] for v in layer.values())
        return {k: v for k, v in layer.items() if self.F[v] <= m + beam}


def _poisson_nll(j, lam):
    return lam - j * math.log(lam) + math.lgamma(j + 1)


class Decoder:
    def __init__(self, game: Game, events, segs, final=None, flip=0, n0=0, video_end=None, params=None):
        """events: time-ordered detected rallies with cues in KIT space (see cues_doubles / cues_singles);
        team = kit ^ flip; n0 = team on the near side at the start. final: (score team 0, score team 1)."""
        self.g, self.ev, self.final, self.flip, self.n0 = game, events, final, flip, n0
        self.p = {**P, **(params or {})}
        self.pre_a, self.mid_a = assign_anchors(events, segs or [])
        self.video_end = video_end if video_end is not None else (events[-1]["end_t"] + 60 if events else 60)
        self._ac = {}

    # ---- evidence
    def anchor_cost(self, X, s):
        k = (X["score"], X["team"], X["no"], X["p"], s)
        if k in self._ac:
            return self._ac[k]
        g, p = self.g, self.p
        a, b, T, n, pp = s
        if X["score"] == (a, b):
            c = 0.0
            if X["team"] is not None and X["team"] != T:
                c = p["ovl_team"]
            else:
                if g.players and X["no"] is not None and X["no"] != n:
                    c += p["ovl_no"]
                if g.players and X["p"] is not None and X["team"] == T and X["p"] != pp:
                    c += p["ovl_player"]
        else:
            c = p["ovl_bad"]
            for T0 in ((X["team"],) if X["team"] is not None else (0, 1)):  # lag: X is one point behind s
                for n0 in ((X["no"],) if (g.players and X["no"] is not None) else ((1, 2) if g.players else (0,))):
                    for p0 in ((0, 1) if g.players else (0,)):
                        q = (X["score"][0], X["score"][1], T0, n0, p0)
                        if not g.over(q) and any(s2[:2] == (a, b) and s2[2] == T for s2, _, _ in g.succ(q)):
                            c = p["ovl_lag"]
        self._ac[k] = c
        return c

    def track_cost(self, ev, s):
        g = self.g
        a, b, T, n, pp = s
        c = 0.0
        if ev.get("serve_kit") is not None:
            c += _nll(ev["serve_rel"], (ev["serve_kit"] ^ self.flip) == T)
        if ev.get("near_kit") is not None:
            exp = self.n0 if not g.switched(s) else 1 - self.n0
            c += _nll(ev["near_rel"], (ev["near_kit"] ^ self.flip) == exp)
        if g.players and ev.get("server_player") is not None:
            team_p, idx = ev["server_player"]
            if team_p == T:
                c += _nll(ev["player_rel"], idx == pp)
        if not g.doubles and g.sideout and ev.get("serve_right") is not None:
            own = (a, b)[T]
            c += _nll(ev["parity_rel"], ev["serve_right"] == (own % 2 == 0))
        return c

    def _advance(self, anchors, i, s):
        while i < len(anchors) and self.anchor_cost(anchors[i], s) < self.p["ovl_team"]:
            i += 1
        return i

    def _gap(self, tr, post, anchors, gap_s, gid):
        """Missed points in a gap: post {(state, i): node} -> {state: node} (after j >= 0 missed points)."""
        p, g = self.p, self.g
        lam = max(p["lam_min"], (gap_s - p["gap0_s"]) / p["point_s"])
        M = min(p["max_miss"], int(max(gap_s, 0) // p["miss_every_s"]))
        out = {}
        layer = post
        for j in range(M + 1):
            for (s, i), u in layer.items():
                w = _poisson_nll(j, lam) + p["ovl_skip"] * (len(anchors) - i)
                tr.node(out, s, u, w, ("close", gid, j))
            if j == M:
                break
            nxt = {}
            for (s, i), u in tr.prune(layer, p["beam"]).items():
                if g.over(s):
                    continue
                for s2, w_, how in g.succ(s):
                    tr.node(nxt, (s2, self._advance(anchors, i, s2)), u, 0.0, ("miss", gid, s, w_, how))
            layer = nxt
            if not layer:
                break
        return tr.prune(out, p["beam"])

    def run(self):
        g, p, E = self.g, self.p, self.ev
        tr = _Trellis()
        self.tr = tr
        post = {}
        a0 = self.mid_a.get(0, []) if E else self.mid_a.get(0, [])
        for s in g.starts():
            tr.node(post, (s, self._advance(a0, 0, s)), None, 0.0, ("start",))
        t_prev = 0.0
        self.pre_layers, self.out_edges = [], []
        for e, ev in enumerate(E):
            pre = self._gap(tr, post, self.mid_a.get(e, []), ev["start_t"] - t_prev, e)
            X = self.pre_a.get(e)
            emit = {}
            for s, u in pre.items():
                w = self.track_cost(ev, s) if not g.over(s) else 0.0
                if X is not None:
                    w += self.anchor_cost(X, s) * (1.0 if X["n"] >= 3 else 0.6)
                tr.node(emit, s, u, w, ("emit", e))
            emit = tr.prune(emit, p["beam"])
            self.pre_layers.append(emit)
            nxt_a = self.mid_a.get(e + 1, [])
            post = {}
            e0 = len(tr.edges)
            for s, u in emit.items():
                i0 = self._advance(nxt_a, 0, s)
                if g.over(s):
                    tr.node(post, (s, i0), u, p["after"], ("after", e))
                    continue
                tr.node(post, (s, i0), u, p["spurious"], ("spurious", e))
                for s2, w_, how in g.succ(s):
                    tr.node(post, (s2, self._advance(nxt_a, i0, s2)), u, 0.0, ("rally", e, s, w_, how))
            self.out_edges.append((e0, len(tr.edges)))
            post = tr.prune(post, p["beam"])
            t_prev = ev["end_t"]
        end = self._gap(tr, post, self.mid_a.get(len(E), []), self.video_end - t_prev, len(E))
        fin = {}
        for s, u in end.items():
            if self.final is not None:
                w = 0.0 if (s[0], s[1]) == tuple(self.final) else math.inf
            else:
                w = 0.0 if g.over(s) else p["unfinished"]
            if w < math.inf:
                tr.node(fin, "end", u, w, ("final",))
        self.fin = fin.get("end")
        self.cost = tr.F[self.fin] if self.fin is not None else math.inf
        return self.cost

    def backward(self):
        tr = self.tr
        B = [math.inf] * len(tr.F)
        if self.fin is None:
            return B
        B[self.fin] = 0.0
        for u, v, w, _ in reversed(tr.edges):
            if B[v] + w < B[u]:
                B[u] = B[v] + w
        self.B = B
        return B

    def path(self):
        """Labels of the best path in time order."""
        tr, v, out = self.tr, self.fin, []
        while v is not None:
            out.append((tr.lab[v], tr.key[v]))
            v = tr.par[v]
        return out[::-1]


# --------------------------------------------------------------------------------------------- top level

def parse_official(text, teams):
    """'Kodai Tigers won 11-5 (match card)' -> (11, 5) in team order (team 0, team 1); None if unreadable."""
    import re
    if not text:
        return None
    m = re.search(r"^(.*?)\s+won\s+(\d+)-(\d+)", text)
    if not m:
        return None
    who, w, l = m.group(1).strip().lower(), int(m.group(2)), int(m.group(3))
    k = next((i for i, t in enumerate(teams) if t.lower() in who or who.split(" (")[0] in t.lower()), None)
    if k is None:
        return None
    return (w, l) if k == 0 else (l, w)


def decode(game: Game, events, segs, final=None, flips=(0, 1), video_end=None, params=None):
    """Best decode over the kit -> team flips allowed and the starting near team. Returns the Decoder (with
    .cost, .flip, .n0, .alt_cost = best cost with the other flip, if tried)."""
    best, by_flip = None, {}
    has_near = any(ev.get("near_kit") is not None for ev in events)
    for flip in flips:
        for n0 in ((0, 1) if has_near else (0,)):
            d = Decoder(game, events, segs, final, flip, n0, video_end, params)
            c = d.run()
            by_flip[flip] = min(by_flip.get(flip, math.inf), c)
            if best is None or c < best.cost:
                best = d
    best.alt_cost = min((c for f, c in by_flip.items() if f != best.flip), default=None)
    return best


def summarize(d: Decoder):
    """Per detected rally + missed points from a decoded path.
    rallies: [{"index", "role": point|spurious|after, "state": (a,b,T,n,p) before, "server_team", "server_no",
               "server_p", "winner_team", "how", "score_before", "score": [a, b], "provenance", "confidence",
               "margin_winner", "margin_server"}]
    missed:  [{"after_index" (rally index before the gap or None), "state", "winner_team", "how", "score_before",
               "source": overlay|rules}]"""
    g, tr = d.g, d.tr
    B = d.backward()
    path = d.path()
    E = d.ev
    rallies, missed = [], []
    for k, (lab, key) in enumerate(path):
        kind = lab[0]
        if kind == "miss":
            _, gid, s, w, how = lab
            s2 = key[0]
            anchors = d.mid_a.get(gid, []) + ([d.pre_a[gid]] if gid in d.pre_a else [])
            seen = any(X["score"] == (s2[0], s2[1]) for X in anchors)
            missed.append({"gap": gid, "after_index": E[gid - 1]["index"] if gid > 0 else None, "state": s,
                           "winner_team": w, "how": how, "score_before": g.score_text(s), "score": [s[0], s[1]],
                           "source": "overlay" if seen else "rules"})
        elif kind in ("rally", "spurious", "after"):
            e = lab[1]
            ev = E[e]
            prev_key = path[k - 1][1]
            s = prev_key
            if kind == "rally":
                _, _, s, w, how = lab
            else:
                w, how = None, ("not a point" if kind == "spurious" else "after match")
            rallies.append({"index": ev["index"], "event": e, "role": "point" if kind == "rally" else kind, "state": s,
                            "server_team": s[2] if kind != "after" else None, "server_no": s[3] if g.players else None,
                            "server_p": s[4] if g.players else None, "winner_team": w, "how": how,
                            "score_before": g.score_text(s) if kind != "after" else None, "score": [s[0], s[1]]})
    # max-marginals per rally: winner (None = not a point) and serving team
    for r in rallies:
        e = r["event"]
        a, b = d.out_edges[e]
        best_w, best_t = {}, {}
        for u, v, w, lab in tr.edges[a:b]:
            tot = tr.F[u] + w + B[v]
            if tot == math.inf:
                continue
            win = lab[3] if lab[0] == "rally" else None
            T = tr.key[u][2]
            best_w[win] = min(best_w.get(win, math.inf), tot)
            best_t[T] = min(best_t.get(T, math.inf), tot)
        bw = best_w.get(r["winner_team"], math.inf)
        r["margin_winner"] = round(min((c for k, c in best_w.items() if k != r["winner_team"]), default=math.inf) - bw, 2)
        bt = best_t.get(r["state"][2], math.inf)
        r["margin_server"] = round(min((c for k, c in best_t.items() if k != r["state"][2]), default=math.inf) - bt, 2)
        X = d.pre_a.get(e)
        before_ok = X is not None and d.anchor_cost(X, r["state"]) < P["ovl_team"]
        after_ok = False
        if r["role"] == "point":
            nxt = [r2 for r2 in rallies if r2["event"] > e][:1]
            s_after = None
            for lab, key in path:
                if lab[0] == "rally" and lab[1] == e:
                    s_after = key[0]
            cands = d.mid_a.get(e + 1, []) + ([d.pre_a[e + 1]] if e + 1 in d.pre_a else [])
            after_ok = s_after is not None and bool(cands) and d.anchor_cost(cands[0], s_after) < P["ovl_team"]
        if r["role"] == "point" and before_ok and after_ok:
            r["provenance"] = "overlay"
        elif min(r["margin_winner"], r["margin_server"]) >= CERTAIN_NATS:
            r["provenance"] = "rules"
        else:
            r["provenance"] = "tracking"
        r["confidence"] = "certain" if r["provenance"] in ("overlay", "rules") else "inferred"
        for k in ("margin_winner", "margin_server"):
            if r[k] == math.inf:
                r[k] = 99.0
    return {"rallies": rallies, "missed": missed, "cost": round(d.cost, 2), "flip": d.flip, "n0": d.n0,
            "flip_margin": None if d.alt_cost is None else round(d.alt_cost - d.cost, 2)}


# --------------------------------------------------------------------------------------------- cues

def cues_doubles(run_dir, A: dict, rosters=None, cues=None):
    """Detected rallies (time order) with tracking cues in KIT space, from doubles.serve_cues (team sides from
    kits, serving side from the serve formation / first hit) and the mark-once identity (<run>/hitters.json):
    the serve's hitter name -> (team, partner index). Returns (events, identity flip or None)."""
    from .doubles import serve_cues
    cues = {**SCORING_CUES, **(cues or {})}
    run_dir = Path(run_dir)
    seq = serve_cues(run_dir, A)
    hp = run_dir / "hitters.json"
    hitters = json.loads(hp.read_text()) if hp.exists() else {}
    first = {}
    for s in sorted(A["shots"], key=lambda s: s["frame"]):
        first.setdefault(s["rally"], s)
    votes = [0, 0]  # identity: kit ^ team == 0 / 1
    events = []
    for r in seq:
        cue = r.get("serve_cue") or "first hit"
        if cue == "first hit" and r.get("serve_inferred"):
            cue = "first hit inferred"
        ev = {"index": r["index"], "start_t": r["start_t"], "end_t": r["end_t"], "near_kit": r["near_kit"],
              "near_rel": cues["near"], "serve_kit": r["serve_kit"], "serve_rel": cues[cue], "cue": cue,
              "server_player": None, "player_rel": cues["player"]}
        sh = first.get(r["index"])
        h = hitters.get(f"{sh['rally']}:{sh['index']}") if sh else None
        if rosters and h and h.get("name") and not sh.get("inferred") and sh["side"] == r["serve_side"]:
            for t, team in enumerate(rosters):
                if h["name"] in team:
                    ev["server_player"] = (t, team.index(h["name"]))
        events.append(ev)
    if rosters and hitters:
        side_kit = {r["index"]: r["near_kit"] for r in seq}
        for s in A["shots"]:
            h = hitters.get(f"{s['rally']}:{s['index']}")
            nk = side_kit.get(s["rally"])
            if not h or not h.get("name") or nk is None:
                continue
            kit = nk if s["side"] == "near" else 1 - nk
            for t, team in enumerate(rosters):
                if h["name"] in team:
                    votes[kit ^ t] += 1
    flip = None
    if sum(votes) >= 10 and max(votes) >= 0.8 * sum(votes):
        flip = 0 if votes[0] > votes[1] else 1
    return events, flip


def cues_singles(A: dict, cues=None):
    """Singles: kit = analysis player (P1 -> 0, P2 -> 1, clustered by kit colour). Server = first hitter, near
    kit = the near side's hitter, serve half (right / left) for the serve-position rule."""
    from .outcome import _right_half
    cues = {**SCORING_CUES, **(cues or {})}
    kit = {pid: k for k, pid in enumerate(sorted(A["players"]))}
    events = []
    for r in sorted(A["rallies"], key=lambda r: r["start_t"]):
        sh = sorted([s for s in A["shots"] if s["rally"] == r["index"]], key=lambda s: s["frame"])
        if not sh:
            continue
        near = [kit[s["player"]] for s in sh if not s.get("inferred") and s["side"] == "near"]
        far = [kit[s["player"]] for s in sh if not s.get("inferred") and s["side"] == "far"]
        nk = max(set(near), key=near.count) if near else (1 - max(set(far), key=far.count) if far else None)
        s0 = sh[0]
        cue = "singles first hit inferred" if s0.get("inferred") else "singles first hit"
        x = s0["hitter_court"][0] if s0.get("hitter_court") else None
        right = _right_half(s0["side"], x) if (x is not None and abs(x - 10.0) >= 2.5 and not s0.get("inferred")) else None
        events.append({"index": r["index"], "start_t": r["start_t"], "end_t": r["end_t"], "near_kit": nk,
                       "near_rel": cues["near"], "serve_kit": kit.get(s0["player"]), "serve_rel": cues[cue], "cue": cue,
                       "serve_right": right, "parity_rel": cues["parity"]})
    return events, kit


def decode_match(run_dir, A: dict, m: dict, squads, hide=None, params=None, cues=None, use_overlay=True, prepared=None):
    """Decode one TNPPL match. m: the config entry (type, teams, rosters, rules, official, names, singles_overlay);
    squads: [[team 0 squad], [team 1 squad]] for serve-line names; A: analysis (replays already dropped).
    hide: overlay time spans to treat as invisible (validation). Returns summarize() + {"events", "game",
    "final", "segments", "teams"}; teams are in config order (= overlay rows). prepared: match_cues() result
    (reuse across calls: reading the tracking cues is the slow part)."""
    run_dir = Path(run_dir)
    game, events, flips = prepared or match_cues(run_dir, A, m, cues)
    segs = []
    ovp = run_dir / "overlay.json"
    if use_overlay and ovp.exists():
        R = json.loads(ovp.read_text())["readings"]
        segs = overlay_segments(R, squads, m.get("rosters") if game.players else None,
                                use_server=not m.get("singles_overlay"), use_serve_no=game.players, hide=hide)
    final = parse_official(m.get("official"), m["teams"])
    fps = A["meta"]["fps"]
    vend = A["meta"].get("last_frame", 0) / fps if A["meta"].get("last_frame") else None
    d = decode(game, events, segs, final, flips, vend, params)
    out = summarize(d)
    out.update({"events": events, "game": game, "final": final, "segments": segs, "teams": m["teams"]})
    return out


def match_cues(run_dir, A: dict, m: dict, cues=None):
    """(Game, events, kit flips to try) for a config match entry (see decode_match)."""
    rules = m["rules"]
    doubles = m["type"] == "doubles"
    game = Game(rules.get("scoring", "sideout"), rules.get("target", 11), rules.get("win_by", 1), doubles)
    if doubles:
        events, flip = cues_doubles(run_dir, A, m.get("rosters") if game.players else None, cues)
        flips = (flip,) if flip is not None else (0, 1)
    else:
        events, kit = cues_singles(A, cues)
        names = m.get("names") or {}
        tk = [next((t for t, nm in enumerate(m["teams"]) if nm == names.get(pid)), None) for pid in sorted(kit)]
        flips = (0 if tk[0] == 0 or tk[1] == 1 else 1,) if any(t is not None for t in tk) else (0, 1)
    return game, events, flips


def kit_signatures(run_dir, events, per_side=2, step=6):
    """Mean shirt-colour signature (sqrt histogram, as doubles.team_sides) of each tracking kit over the detected
    rallies: {kit: vector}. Used to name kits by the team colours learned on other matches of the same tie."""
    import numpy as np
    from .players import select_players
    want = {}
    fps = 30.0
    for ev in events:
        if ev.get("near_kit") is None:
            continue
        for f in range(int(ev["start_t"] * fps), int(ev["end_t"] * fps) + 1, step):
            want[f] = ev["near_kit"]
    acc = {0: [], 1: []}
    with (Path(run_dir) / "detections.jsonl").open() as fh:
        for line in fh:
            if not want:
                break
            r = json.loads(line)
            nk = want.get(r["f"])
            if nk is None or not r.get("court") or not r.get("people"):
                continue
            for p in select_players(r["people"], per_side):
                if p.get("sig") is not None:
                    acc[nk if p["side"] == "near" else 1 - nk].append(np.sqrt(np.asarray(p["sig"], float)))
    return {k: np.mean(v, 0) for k, v in acc.items() if len(v) >= 20}


def flip_from_colours(sig: dict, refs: dict):
    """refs: {team: vector} (team colours from matches of the same tie whose kit naming is known). The kit ->
    team flip whose colours are closer, with the margin (relative distance gain); None if undecidable."""
    import numpy as np
    if len(sig) < 2 or len(refs) < 2:
        return None, 0.0
    d = [sum(float(np.sum((sig[k] - refs[k ^ f]) ** 2)) for k in (0, 1)) for f in (0, 1)]
    f = int(d[1] < d[0])
    return f, (max(d) - min(d)) / max(max(d), 1e-9)


def name_kits_by_colour(preps: dict, min_margin=0.3):
    """preps: {match id: (run_dir, m, (game, events, flips))} for ONE tie (teams keep their kits for the whole
    tie). Matches whose kit naming is unknown (flips == (0, 1): no identity marks, no names) get the flip whose
    colours match the team colours of the matches with a known naming. Returns {match id: (flip, margin)} for
    the matches it named (in place: preps entries get flips = (flip,))."""
    import numpy as np
    sig = {}
    for k, (run, m, (g, ev, fl)) in preps.items():
        sig[k] = kit_signatures(run, ev, 2 if m["type"] == "doubles" else 1)
    out = {}
    for k, (run, m, (g, ev, fl)) in preps.items():
        if len(fl) == 1:
            continue
        refs = {}
        for k2, (_, _, (_, _, fl2)) in preps.items():
            if k2 != k and len(fl2) == 1:
                for kit, v in sig[k2].items():
                    refs.setdefault(kit ^ fl2[0], []).append(v)
        f, margin = flip_from_colours(sig[k], {t: np.mean(v, 0) for t, v in refs.items()})
        if f is not None and margin >= min_margin:
            preps[k] = (run, m, (g, ev, (f,)))
            out[k] = (f, round(margin, 3))
    return out
