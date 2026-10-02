"""Hit and bounce detection from the ball trajectory, and rally segmentation.

A candidate event is a sharp change in the ball's image-space velocity (direction or
speed). It is a HIT if a player's wrist is close to the ball at that moment, a BOUNCE
if the ball is over the court surface with no player near, otherwise it is ignored
(net cord, occlusion noise).

Rules applied afterwards:
- consecutive hits must alternate court sides (you can't return your own shot);
- at most one bounce between two hits (the strongest candidate is kept).
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .court import COURT_L, COURT_W, NET_Y

NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 0, 5, 6, 7, 8, 9, 10


@dataclass
class Event:
    kind: str                 # "hit" | "bounce"
    frame: int
    t: float
    img: tuple[float, float]  # ball position in the image
    score: float
    player: str | None = None
    side: str | None = None   # side of the court the event happened on ("near" / "far")
    court: tuple[float, float] | None = None  # bounce location, or hitter's feet for hits
    wrist: str | None = None  # "left" / "right" wrist closest to the ball (hits)
    body_x: float | None = None     # hitter's shoulder-midpoint x in the image (hits)
    above_head: bool = False        # hitting wrist above the head (overheads)
    ground: tuple[float, float] | None = None  # ball image point projected onto the court plane
    inferred: bool = False          # not seen: deduced from the rules (e.g. a missed serve)


@dataclass
class Rally:
    index: int
    start_frame: int
    end_frame: int
    events: list[Event] = field(default_factory=list)

    @property
    def hits(self):
        return [e for e in self.events if e.kind == "hit"]


def _chunks(frames: list[int], max_gap: int):
    chunk = [frames[0]]
    for a, b in zip(frames, frames[1:]):
        if b - a > max_gap:
            yield chunk
            chunk = []
        chunk.append(b)
    yield chunk


def _change_scores(pts: np.ndarray, k: int = 3) -> np.ndarray:
    """Per-point score of how sharply the trajectory changes there."""
    n = len(pts)
    s = np.zeros(n)
    for i in range(k, n - k):
        v_in = (pts[i] - pts[i - k]) / k
        v_out = (pts[i + k] - pts[i]) / k
        a, b = np.linalg.norm(v_in), np.linalg.norm(v_out)
        if a < 1.0 and b < 1.0:
            continue
        cos = float(np.dot(v_in, v_out) / (a * b + 1e-6))
        angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        ratio = max(a, b) / (min(a, b) + 1.0)
        s[i] = angle / 45.0 + max(0.0, ratio - 1.5)
    return s


def _court_polygon_img(H_inv: np.ndarray, margin: float = 1.0) -> np.ndarray:
    corners = np.array([[-margin, -margin], [COURT_W + margin, -margin],
                        [COURT_W + margin, COURT_L + margin], [-margin, COURT_L + margin]], np.float32)
    return cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H_inv).reshape(-1, 2)


FLIGHT_SETTINGS = {"enabled": True, "penalty": 900.0, "reach": 0.45, "only": True}  # tuned on 15 labelled rallies (70 hits)


def _flight_events(ball, players, homographies, fps, penalty, reach) -> list[Event]:
    """Contacts as boundaries between smooth ball flights (see flights.py)."""
    from .flights import flight_boundaries
    out = []
    for f, pos, v_in, v_out in flight_boundaries(ball, penalty=penalty):
        best = None
        for ff in range(f - 4, f + 5):
            for p in players.get(ff, []):
                pts, h = _reach_points(p)
                for xy, name in pts:
                    d = np.hypot(xy[0] - pos[0], xy[1] - pos[1]) / h
                    if best is None or d < best[0]:
                        best = (d, p, name)
        if best and best[0] <= reach:
            d, p, name = best
            kp = np.asarray(p["kp"])
            wy = kp[L_WRIST if name == "left" else R_WRIST, 1]
            out.append(Event("hit", f, f / fps, pos, 2.5 + (reach - d), player=p.get("id"), side=p["side"],
                             court=tuple(p["court"]), wrist=name,
                             body_x=float(kp[[L_SHOULDER, R_SHOULDER], 0].mean()),
                             above_head=bool(kp[NOSE, 2] > 0.3 and wy < kp[NOSE, 1])))
            continue
        H = next((homographies[g] for g in (f, *range(f - 3, f + 4)) if g in homographies), None)
        if H is None:
            continue
        poly = _court_polygon_img(np.linalg.inv(H))
        if cv2.pointPolygonTest(poly.astype(np.float32), (float(pos[0]), float(pos[1])), False) < 0:
            continue
        cxy = cv2.perspectiveTransform(np.array([[pos]], np.float32), H)[0, 0]
        out.append(Event("bounce", f, f / fps, pos, 2.0, side="far" if cxy[1] < COURT_L / 2 else "near",
                         court=(float(cxy[0]), float(cxy[1]))))
    return out


HIT_MODEL = {"enabled": True, "bundle": None, "bounce_bundle": None}  # learned hit/bounce deciders (hitmodel.py), used when a Context is given


def model_bounces(ctx, players, homographies, fps, rule_bounces) -> list[Event]:
    """Bounces from the learned bounce detector; the rule-based bounces are extra candidates."""
    from . import hitmodel as hm
    from .swing import swing_series
    if HIT_MODEL["bounce_bundle"] is None:
        HIT_MODEL["bounce_bundle"] = hm.load_bounce_bundle()
    if HIT_MODEL["bounce_bundle"] is None or SWING_MODEL["model"] is None or not ctx.ball:
        return []
    sp, _ = swing_series(players, SWING_MODEL["model"])
    out = []
    for f, prob, G in hm.predict_bounces(ctx, sp, HIT_MODEL["bounce_bundle"], min(ctx.ball), max(ctx.ball),
                                         [e.frame for e in rule_bounces]):
        xy = hm.ball_at(ctx, f)
        H = next((homographies[g] for g in (f, *range(f - 3, f + 4)) if g in homographies), None)
        if xy is None or H is None:
            continue
        cxy = cv2.perspectiveTransform(np.array([[xy]], np.float32), H)[0, 0]
        out.append(Event("bounce", f, f / fps, tuple(xy), 2.0 + prob, side="far" if cxy[1] < COURT_L / 2 else "near",
                         court=(float(cxy[0]), float(cxy[1]))))
    return out


def model_hits(ctx, players, fps) -> list[Event]:
    """Hits from the learned hit decider (Random Forest on swing/ball/3D/paddle clues)."""
    from . import hitmodel as hm
    from .swing import swing_series
    _load_default_swing_model()
    if HIT_MODEL["bundle"] is None:
        HIT_MODEL["bundle"] = hm.load_bundle()
    if HIT_MODEL["bundle"] is None or SWING_MODEL["model"] is None or not ctx.ball:
        return []
    sp, _ = swing_series(players, SWING_MODEL["model"])
    out = []
    for f, prob, side in hm.predict_hits(ctx, sp, HIT_MODEL["bundle"], min(ctx.ball), max(ctx.ball)):
        p = hm.player_at(ctx, f, side)
        if p is None:
            continue
        kp = np.asarray(p["kp"])
        xy = hm.ball_at(ctx, f)
        name = "right" if kp[R_WRIST, 2] >= kp[L_WRIST, 2] else "left"
        if xy is not None and kp[L_WRIST, 2] > 0.3 and kp[R_WRIST, 2] > 0.3:
            name = "left" if np.hypot(*(kp[L_WRIST, :2] - xy)) < np.hypot(*(kp[R_WRIST, :2] - xy)) else "right"
        wy = kp[L_WRIST if name == "left" else R_WRIST, 1]
        out.append(Event("hit", f, f / fps, tuple(xy) if xy is not None else tuple(p.get("foot", p["court"])),
                         3.0 + prob, player=p.get("id"), side=side, court=tuple(p["court"]), wrist=name,
                         body_x=float(kp[[L_SHOULDER, R_SHOULDER], 0].mean()),
                         above_head=bool(kp[NOSE, 2] > 0.3 and wy < kp[NOSE, 1])))
    return out


def detect_events(ball: dict, players: dict[int, list[dict]], homographies: dict[int, np.ndarray],
                  fps: float, max_gap: int = 6, min_score: float = 1.2,
                  hit_radius: float = 0.45, hit_ctx=None) -> list[Event]:
    frames = sorted(ball)
    if not frames:
        return []
    events: list[Event] = []
    fs = FLIGHT_SETTINGS
    flight = _flight_events(ball, players, homographies, fps, fs["penalty"], fs["reach"]) if fs["enabled"] else []
    chunks = list(_chunks(frames, max_gap))
    events += _gap_reversals(chunks, ball, players, homographies, fps)
    events += _proximity_hits(ball, players, fps)
    for chunk in chunks:
        if len(chunk) < 9:
            continue
        pts = np.array([ball[f][:2] for f in chunk], dtype=float)
        scores = _change_scores(pts)
        # every local maximum above threshold is classified; duplicates are merged after
        taken = [i for i in range(len(scores)) if scores[i] >= min_score
                 and scores[i] == scores[max(0, i - 3):i + 4].max()]
        # the first ball positions of a track can be a contact (e.g. the serve)
        if not any(i < 6 for i in taken):
            taken.append(1)
        for i in sorted(taken):
            f = chunk[i]
            bx, by = pts[i]
            a, b = max(0, i - 3), min(len(pts) - 1, i + 3)
            v_in = (pts[i] - pts[a]) / max(i - a, 1)
            v_out = (pts[b] - pts[i]) / max(b - i, 1)
            ev = _classify(f, (bx, by), float(scores[i]), players, homographies, fps, hit_radius, v_in, v_out)
            if ev:
                events.append(ev)
    mh = model_hits(hit_ctx, players, fps) if hit_ctx is not None and HIT_MODEL["enabled"] else []
    if mh:
        # learned hit decider; learned bounce detector (falls back to trajectory-rule bounces)
        rule_b = [e for e in flight + events if e.kind == "bounce"]
        mb = model_bounces(hit_ctx, players, homographies, fps, rule_b)
        events = mh + (mb if mb else rule_b)
    elif SWING_SETTINGS["enabled"]:
        # hits from swing + ball fusion; bounces as before
        events = swing_hits(ball, players, flight, fps) + [e for e in flight + events if e.kind == "bounce"]
    elif fs["enabled"] and fs["only"]:
        # hits from flight boundaries; bounces from both (trajectory kinks find bounces that
        # a single smooth fit can absorb)
        fhits = [e for e in flight if e.kind == "hit"]
        if PERSIST["mode"] == "only":
            fhits = persistent_reversal_hits(ball, players, fps, PERSIST["m"])
        elif PERSIST["mode"] == "add":
            ph = persistent_reversal_hits(ball, players, fps, PERSIST["m"])
            fhits += [e for e in ph if not any(abs(e.frame - h.frame) < 10 for h in fhits)]
        _load_default_swing_model()
        if SWING_MODEL["mode"] != "off" and SWING_MODEL["model"] is not None:
            fhits = learned_swing_hits(ball, players, fhits, fps, SWING_MODEL)
        elif GAPFILL["enabled"]:
            fhits = fill_gaps_with_swings(fhits, ball, players, fps, GAPFILL["min_score"])
        events = fhits + [e for e in flight + events if e.kind == "bounce"]
    else:
        events += flight
    return _clean(events)


def _reach_points(p):
    """Wrists plus estimated paddle centres (wrist extended along the forearm)."""
    kp = np.asarray(p["kp"])
    h = max(p["box"][3] - p["box"][1], 1.0)
    pts = []
    for e, w, name in ((L_ELBOW, L_WRIST, "left"), (R_ELBOW, R_WRIST, "right")):
        if kp[w, 2] < 0.3:
            continue
        pts.append((kp[w, :2], name))
        if kp[e, 2] >= 0.3:
            d = kp[w, :2] - kp[e, :2]
            n = np.linalg.norm(d)
            if n > 1:
                pts.append((kp[w, :2] + d / n * 0.22 * h, name))
    return pts, h


def _proximity_hits(ball, players, fps, reach: float = 0.55, win: int = 6, rise: float = 0.2) -> list[Event]:
    """Contacts seen as the ball reaching a player's paddle and then leaving it, whatever
    the ball does on screen afterwards (lobs keep rising after a far-side contact)."""
    out = []
    for side in ("near", "far"):
        series = {}
        for f, (bx, by, _, _) in ball.items():
            best = None
            for p in players.get(f, []):
                if p["side"] != side:
                    continue
                pts, h = _reach_points(p)
                for xy, name in pts:
                    d = np.hypot(xy[0] - bx, xy[1] - by) / h
                    if best is None or d < best[0]:
                        best = (d, p, name)
            if best:
                series[f] = best
        for f, (d, p, name) in series.items():
            if d > reach:
                continue
            around = [series[g][0] for g in range(f - win, f + win + 1) if g in series and g != f]
            if not around or d > min(around):
                continue  # not the closest approach
            before = [series[g][0] for g in range(f - win, f) if g in series]
            after = [series[g][0] for g in range(f + 1, f + win + 1) if g in series]
            if not after or max(after) - d < rise or (before and max(before) - d < rise * 0.5):
                continue  # the ball must come in and then leave
            kp = np.asarray(p["kp"])
            wy = kp[L_WRIST if name == "left" else R_WRIST, 1]
            out.append(Event("hit", f, f / fps, tuple(ball[f][:2]), 2.0 + (reach - d), player=p.get("id"),
                             side=side, court=tuple(p["court"]), wrist=name,
                             body_x=float(kp[[L_SHOULDER, R_SHOULDER], 0].mean()),
                             above_head=bool(kp[NOSE, 2] > 0.3 and wy < kp[NOSE, 1])))
    return out


def _gap_reversals(chunks, ball, players, homographies, fps, max_gap_s: float = 0.5) -> list[Event]:
    """Contacts hidden by the hitter's body: the ball vanishes for a few frames and comes
    back travelling the other way along the court (up-screen <-> down-screen)."""
    out = []
    for c1, c2 in zip(chunks, chunks[1:]):
        gap = c2[0] - c1[-1]
        if gap > max_gap_s * fps or len(c1) < 3 or len(c2) < 3:
            continue
        # direction a few frames before the ball vanished (it may already be turning
        # right before the gap) and right after it reappears
        k0, k1 = max(0, len(c1) - 9), max(1, len(c1) - 4)
        p1 = np.array([ball[c1[k0]][:2], ball[c1[k1]][:2], ball[c1[-1]][:2]], float)
        p2 = np.array([ball[f][:2] for f in c2[:3]], float)
        vy_in = (p1[1, 1] - p1[0, 1]) / max(c1[k1] - c1[k0], 1)
        vy_out = (p2[-1, 1] - p2[0, 1]) / (c2[2] - c2[0])
        if vy_in < -1.0 and vy_out > 1.0:
            side = "far"
        elif vy_in > 1.0 and vy_out < -1.0:
            side = "near"
        else:
            continue
        f = (c1[-1] + c2[0]) // 2
        xy = tuple(((p1[-1] + p2[0]) / 2).tolist())
        cand = [p for ff in range(f - 3, f + 4) for p in players.get(ff, []) if p["side"] == side]
        p = cand[0] if cand else None
        out.append(Event("hit", f, f / fps, xy, 3.0, player=p.get("id") if p else None, side=side,
                         court=tuple(p["court"]) if p else None))
    return out


def _classify(f, ball_xy, score, players, homographies, fps, hit_radius, v_in=None, v_out=None) -> Event | None:
    t = f / fps
    # nearest wrist over a small window (hit moment and detection moment differ by a frame or two)
    best = None
    for ff in range(f - 2, f + 3):
        for p in players.get(ff, []):
            kp = np.asarray(p["kp"])
            h = max(p["box"][3] - p["box"][1], 1.0)
            for idx, name in ((L_WRIST, "left"), (R_WRIST, "right")):
                if kp[idx, 2] < 0.3:
                    continue
                d = np.hypot(kp[idx, 0] - ball_xy[0], kp[idx, 1] - ball_xy[1]) / h
                if best is None or d < best[0]:
                    best = (d, p, name)
    at_ground = False
    if best:
        kp = np.asarray(best[1]["kp"])
        h = max(best[1]["box"][3] - best[1]["box"][1], 1.0)
        feet_y = best[1]["foot"][1]
        # a ball at the player's feet is a bounce, not a contact
        at_ground = ball_xy[1] > feet_y - 0.12 * h and best[0] > 0.15
    if best and best[0] < hit_radius and not at_ground:
        _, p, wrist = best
        kp = np.asarray(p["kp"])
        wy = kp[L_WRIST if wrist == "left" else R_WRIST, 1]
        above = bool(kp[NOSE, 2] > 0.3 and wy < kp[NOSE, 1])
        body_x = float(kp[[L_SHOULDER, R_SHOULDER], 0].mean())
        ground = None
        if homographies.get(f) is not None:
            g = cv2.perspectiveTransform(np.array([[ball_xy]], np.float32), homographies[f])[0, 0]
            ground = (float(g[0]), float(g[1]))
        return Event("hit", f, t, ball_xy, score, player=p.get("id"), side=p["side"],
                     court=tuple(p["court"]), wrist=wrist, body_x=body_x, above_head=above, ground=ground)

    H = homographies.get(f)
    if H is None:
        return None
    H_inv = np.linalg.inv(H)
    # Far-side contact without a matched wrist (far player small/occluded): the ball was
    # travelling away from the camera (up the screen) and now comes back, above the net line.
    if v_in is not None:
        net_y = cv2.perspectiveTransform(np.array([[[10.0, 22.0]]], np.float32), H_inv)[0, 0, 1]
        if ball_xy[1] < net_y and v_in[1] < -1.0 and v_out[1] > 1.0:
            far = [p for p in players.get(f, []) if p["side"] == "far"]
            p = far[0] if far else None
            return Event("hit", f, t, ball_xy, score, player=p.get("id") if p else None, side="far",
                         court=tuple(p["court"]) if p else None)
    poly = _court_polygon_img(H_inv)
    if cv2.pointPolygonTest(poly.astype(np.float32), (float(ball_xy[0]), float(ball_xy[1])), False) < 0:
        return None
    cxy = cv2.perspectiveTransform(np.array([[ball_xy]], np.float32), H)[0, 0]
    side = "far" if cxy[1] < COURT_L / 2 else "near"
    return Event("bounce", f, t, ball_xy, score, side=side, court=(float(cxy[0]), float(cxy[1])))


def _dedupe(events: list[Event], window: int = 6) -> list[Event]:
    """Merge same-kind events closer than `window` frames, keeping the strongest."""
    out: list[Event] = []
    for e in sorted(events, key=lambda e: e.frame):
        twin = next((o for o in out if o.kind == e.kind and abs(o.frame - e.frame) < window), None)
        if twin is None:
            out.append(e)
        elif e.score > twin.score:
            out[out.index(twin)] = e
    return out


def _clean(events: list[Event]) -> list[Event]:
    events = _dedupe(events)
    # 1. consecutive hits must alternate sides. Two close "hits" on the same side are
    #    really bounce-then-hit: the earlier one becomes a bounce at its ground point.
    hits: list[Event] = []
    bounces = [e for e in events if e.kind == "bounce"]
    for e in (e for e in events if e.kind == "hit"):
        if hits and hits[-1].side == e.side and e.t - hits[-1].t < 1.5:
            prev = hits.pop()
            if prev.ground and -1 <= prev.ground[0] <= COURT_W + 1 and -1 <= prev.ground[1] <= COURT_L + 1:
                bounces.append(Event("bounce", prev.frame, prev.t, prev.img, prev.score,
                                     side=prev.side, court=prev.ground))
        hits.append(e)
    # 2. at most one bounce between consecutive hits
    kept: list[Event] = []
    bounds = [-np.inf] + [h.frame for h in hits] + [np.inf]
    for a, b in zip(bounds, bounds[1:]):
        inside = [x for x in bounces if a < x.frame < b]
        if inside:
            kept.append(max(inside, key=lambda x: x.score))
    return sorted(hits + kept, key=lambda e: e.frame)


def segment_rallies(events: list[Event], main_view: set[int], fps: float,
                    max_pause: float = 2.5, min_hits: int = 2) -> list[Rally]:
    """Group events into rallies: split on long pauses or camera cuts between events."""
    rallies: list[Rally] = []
    cur: list[Event] = []

    def continuous(a: Event, b: Event) -> bool:
        if b.t - a.t > max_pause:
            return False
        return all(f in main_view for f in range(a.frame, b.frame + 1, 3))

    def serve_only(evs) -> bool:
        """A 1-hit rally is still a point (ace, unreturned serve) when that hit looks like a serve:
        hitter behind the baseline and the ball then bounces on the other side of the net."""
        hits = [e for e in evs if e.kind == "hit"]
        if len(hits) != 1 or not hits[0].court:
            return False
        h = hits[0]
        behind = h.court[1] <= 1.0 if h.side == "far" else h.court[1] >= COURT_L - 1.0
        lands = any(e.kind == "bounce" and e.side != h.side and 0 < e.t - h.t <= 2.5 for e in evs)
        return behind and lands

    def flush():
        if sum(e.kind == "hit" for e in cur) >= min_hits or serve_only(cur):
            # a rally starts with a hit (the serve); drop leading bounces
            while cur and cur[0].kind != "hit":
                cur.pop(0)
            rallies.append(Rally(len(rallies) + 1, cur[0].frame, cur[-1].frame + int(fps), list(cur)))

    for e in events:
        if cur and not continuous(cur[-1], e):
            flush()
            cur = []
        cur.append(e)
    if cur:
        flush()
    return rallies


# ---------------------------------------------------------------------------
# Swing + ball fusion (idea from pose-sequence shot models, e.g. Tennis_Shot_Prediction):
# a contact shows up as a wrist-speed peak of the hitter. Arms move constantly, so a
# swing only counts with ball evidence, and the rally's hits must alternate sides; the
# best alternating sequence is chosen jointly (dynamic programming).
# ---------------------------------------------------------------------------
SWING_SETTINGS = {"enabled": False, "thr": 0.06, "w_swing": 6.0, "w_ball": 2.0, "w_flight": 1.5,
                  "bias": 1.2, "min_gap": 12, "reach": 1.0, "win": 6}


def _wrist_speeds(players):
    """side -> {frame: normalised max wrist speed (body heights per frame)}"""
    W = {"near": {}, "far": {}}
    for f, ps in players.items():
        for p in ps:
            kp = np.asarray(p["kp"])
            W[p["side"]][f] = (kp[[L_WRIST, R_WRIST], :2], kp[[L_WRIST, R_WRIST], 2], max(p["box"][3] - p["box"][1], 1.0), p)
    S = {"near": {}, "far": {}}
    for side, d in W.items():
        for f in d:
            if f - 1 not in d:
                continue
            (wa, ca, _, _), (wb, cb, hb, _) = d[f - 1], d[f]
            v = [np.hypot(*(wb[k] - wa[k])) / hb for k in range(2) if ca[k] > 0.3 and cb[k] > 0.3]
            if v:
                S[side][f] = max(v)
    return S, W


def swing_hits(ball, players, flight_events, fps, cfg=None) -> list[Event]:
    cfg = {**SWING_SETTINGS, **(cfg or {})}
    S, W = _wrist_speeds(players)
    fb = [e for e in flight_events if e.kind == "hit"]
    cands = []
    for side in ("near", "far"):
        sp = S[side]
        for f, v in sp.items():
            if v < cfg["thr"] or v < max((sp.get(g, 0) for g in range(f - 4, f + 5) if g != f), default=0):
                continue
            # ball evidence: closest approach of the ball to this player's reach points around the swing
            d_ball = np.inf
            for g in range(f - cfg["win"], f + cfg["win"] + 1):
                if g in ball and g in W[side]:
                    pts, h = _reach_points(W[side][g][3])
                    for xy, _ in pts:
                        d_ball = min(d_ball, np.hypot(xy[0] - ball[g][0], xy[1] - ball[g][1]) / h)
            near_flight = any(e.side == side and abs(e.frame - f) <= cfg["win"] for e in fb)
            score = (cfg["w_swing"] * v + cfg["w_ball"] * max(0.0, 1 - d_ball / cfg["reach"])
                     + cfg["w_flight"] * near_flight - cfg["bias"])
            cands.append((f, side, score, d_ball))
    cands.sort()
    # best alternating subsequence (each pick must switch side and be >= min_gap frames later)
    n = len(cands)
    best = [c[2] for c in cands]
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if cands[j][1] != cands[i][1] and cands[i][0] - cands[j][0] >= cfg["min_gap"] \
                    and cands[i][0] - cands[j][0] <= 3 * fps and best[j] + cands[i][2] > best[i]:
                best[i], prev[i] = best[j] + cands[i][2], j
    # collect chains greedily from the highest-scoring ends (several rallies per run)
    used, out = set(), []
    for i in sorted(range(n), key=lambda k: -best[k]):
        chain, k = [], i
        while k != -1 and k not in used:
            chain.append(k)
            k = prev[k]
        if not chain or best[i] <= 0:
            continue
        for k in chain:
            used.add(k)
            if cands[k][2] > -1.0:  # keep weak links only inside a strong chain
                f, side, sc, _ = cands[k]
                p = W[side][f][3]
                kp = np.asarray(p["kp"])
                out.append(Event("hit", f, f / fps, tuple(ball[f][:2]) if f in ball else tuple(p["foot"]), 3.0 + sc,
                                 player=p.get("id"), side=side, court=tuple(p["court"]),
                                 wrist="right" if kp[R_WRIST, 2] >= kp[L_WRIST, 2] else "left",
                                 body_x=float(kp[[L_SHOULDER, R_SHOULDER], 0].mean())))
    return out


def fill_gaps_with_swings(hits: list[Event], ball, players, fps, min_score: float = 0.0,
                          long_gap_s: float = 1.6) -> list[Event]:
    """Insert a missed contact between two detected hits when the sequence says one is missing
    (same side twice in a row, or an unusually long gap) and the opponent shows a swing with ball support."""
    S, W = _wrist_speeds(players)
    hits = sorted(hits, key=lambda e: e.frame)
    added = []
    for a, b in zip(hits, hits[1:]):
        gap = (b.frame - a.frame) / fps
        if gap > 4.0:
            continue  # different rallies
        need = None
        if a.side == b.side:
            need = "far" if a.side == "near" else "near"
        elif gap > long_gap_s:
            continue  # alternating already; a long gap alone isn't proof of a missing pair
        if need is None:
            continue
        best = None
        sp = S[need]
        for f in range(a.frame + 8, b.frame - 7):
            v = sp.get(f)
            if v is None or v < max((sp.get(g, 0) for g in range(f - 4, f + 5) if g != f), default=0):
                continue
            d_ball = np.inf
            for g in range(f - 6, f + 7):
                if g in ball and g in W[need]:
                    pts, h = _reach_points(W[need][g][3])
                    for xy, _ in pts:
                        d_ball = min(d_ball, np.hypot(xy[0] - ball[g][0], xy[1] - ball[g][1]) / h)
            sc = 6.0 * v + 2.0 * max(0.0, 1 - d_ball / 1.0)
            if best is None or sc > best[0]:
                best = (sc, f)
        if best and best[0] >= min_score:
            f = best[1]
            p = W[need][f][3]
            kp = np.asarray(p["kp"])
            added.append(Event("hit", f, f / fps, tuple(ball[f][:2]) if f in ball else tuple(p["foot"]), 2.0,
                               player=p.get("id"), side=need, court=tuple(p["court"]),
                               wrist="right" if kp[R_WRIST, 2] >= kp[L_WRIST, 2] else "left",
                               body_x=float(kp[[L_SHOULDER, R_SHOULDER], 0].mean())))
    return hits + added


GAPFILL = {"enabled": True, "min_score": 1.0}  # +5 pts recall at same precision on 15 labelled rallies


def persistent_reversal_hits(ball, players, fps, m: int = 25, smooth: int = 5) -> list[Event]:
    """Hit rule from ameynarwadkar/Tennis-Analysis-System (Apache-2.0), adapted: a hit is where the
    ball's (smoothed) up/down screen direction flips and the new direction holds for >= m of the
    next 1.2*m frames. Side from the new direction: moving up-screen = hit by the near player."""
    fr = sorted(ball)
    if len(fr) < 3:
        return []
    f0, f1 = fr[0], fr[-1]
    ys = np.full(f1 - f0 + 1, np.nan)
    for f in fr:
        ys[f - f0] = ball[f][1]
    idx = np.arange(len(ys))
    ok = ~np.isnan(ys)
    ys = np.interp(idx, idx[ok], ys[ok])  # their pipeline interpolates every gap
    ym = np.convolve(ys, np.ones(smooth) / smooth, mode="full")[:len(ys)]  # trailing rolling mean
    dy = np.diff(ym, prepend=ym[0])
    out = []
    span = int(m * 1.2)
    for i in range(1, len(dy) - span):
        if not ((dy[i] > 0 and dy[i + 1] < 0) or (dy[i] < 0 and dy[i + 1] > 0)):
            continue
        cnt = int(np.sum(np.sign(dy[i + 1:i + span + 1]) == np.sign(dy[i + 1])))
        if cnt >= m:
            f = f0 + i
            if f not in ball:  # only where we actually track the ball nearby
                if not any(g in ball for g in range(f - 3, f + 4)):
                    continue
            side = "near" if dy[i + 1] < 0 else "far"
            cand = [p for g in range(f - 3, f + 4) for p in players.get(g, []) if p["side"] == side]
            p = cand[0] if cand else None
            out.append(Event("hit", f, f / fps, (0.0, float(ys[i])), 2.0, player=p.get("id") if p else None,
                             side=side, court=tuple(p["court"]) if p else None))
    return out


PERSIST = {"mode": "off", "m": 25}  # off | only | add


# ---------------------------------------------------------------------------
# Learned swing detector (pickleball/swing.py) as a hit signal.
# ---------------------------------------------------------------------------
SWING_MODEL = {"model": None, "mode": "confirm", "thr": 0.95, "keep": 0.2, "min_gap": 12}  # off | primary | confirm
# cross-set validated on 15 labelled rallies: flights confirmed by learned swing -> found 79%, real 81%, F1 80%


def _load_default_swing_model():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "models" / "swing_clf.joblib"
    if SWING_MODEL["model"] is None and path.exists():
        import joblib
        SWING_MODEL["model"] = joblib.load(path)


def learned_swing_hits(ball, players, flight_hits, fps, cfg) -> list[Event]:
    from .swing import swing_series
    P, K = swing_series(players, cfg["model"])

    def peaks(side, thr):
        sp = P[side]
        return [f for f, v in sp.items() if v >= thr and v >= max((sp.get(g, 0) for g in range(f - 5, f + 6) if g != f), default=0)]

    def mk(f, side, sc):
        p = K[side][f][2]
        kp = np.asarray(p["kp"])
        return Event("hit", f, f / fps, tuple(ball[f][:2]) if f in ball else tuple(p["foot"]), sc,
                     player=p.get("id"), side=side, court=tuple(p["court"]),
                     wrist="right" if kp[R_WRIST, 2] >= kp[L_WRIST, 2] else "left",
                     body_x=float(kp[[L_SHOULDER, R_SHOULDER], 0].mean()))

    if cfg["mode"] == "primary":
        cands = sorted((f, s, P[s][f]) for s in ("near", "far") for f in peaks(s, cfg["thr"]))
        # best alternating chain(s)
        n = len(cands)
        best, prev = [c[2] for c in cands], [-1] * n
        for i in range(n):
            for j in range(i):
                gap = cands[i][0] - cands[j][0]
                if cands[j][1] != cands[i][1] and cfg["min_gap"] <= gap <= 3 * fps and best[j] + cands[i][2] > best[i]:
                    best[i], prev[i] = best[j] + cands[i][2], j
        used, out = set(), []
        for i in sorted(range(n), key=lambda k: -best[k]):
            k = i
            while k != -1 and k not in used:
                used.add(k)
                out.append(mk(cands[k][0], cands[k][1], 3.0 + cands[k][2]))
                k = prev[k]
        return out
    # confirm: keep flight hits whose hitter swings around them, add confident swings that flights missed
    out = []
    for h in flight_hits:
        ps = max((P[h.side].get(g, 0) for g in range(h.frame - 6, h.frame + 7)), default=0)
        if ps >= cfg["keep"]:
            out.append(h)
    for s in ("near", "far"):
        for f in peaks(s, cfg["thr"]):
            if not any(abs(f - h.frame) <= 8 for h in out):
                out.append(mk(f, s, 2.5 + P[s][f]))
    return out


# ---------------------------------------------------------------------------
# Missed serves. A rally's first detected hit cannot be the serve if the ball reached the
# hitter from across the net (or bounced on the hitter's side) just before it: by the
# two-bounce rule that hit is the RETURN, so the opponent served and the detector missed it
# (typically the far player's serve). Such rallies get an inferred serve by the opponent,
# so server, point winner (serve sequence) and serve/return/third-shot labels stay right.
# ---------------------------------------------------------------------------

def _net_image_y(H):
    """Image y of the net's ground line at the court centre, from an image->court homography."""
    p = cv2.perspectiveTransform(np.array([[[COURT_W / 2, NET_Y]]], np.float32), np.linalg.inv(H))[0, 0]
    return float(p[1])


def crossings_before(h1: Event, ball: dict, all_events: list[Event], homographies: dict, fps: float,
                     max_s: float = 3.5, max_gap: int = 8, margin_px: float = 20.0, min_run: int = 3):
    """Ball sides before the first detected hit, oldest first, e.g. ["far", "near"] = it came from the
    far side, crossed the net once, then h1 (near) hit it. Each crossing is one shot we missed; the
    oldest side is where the rally began (the server's side). Follows the ball back only while it is
    tracked continuously, so a ball fed back between points is not mistaken for a shot."""
    H = next((homographies[g] for g in (h1.frame, *range(h1.frame - 10, h1.frame + 10)) if g in homographies), None)
    runs = []
    if H is not None:
        ny = _net_image_y(H)
        seq, last = [], h1.frame
        for f in range(h1.frame - 1, h1.frame - int(max_s * fps), -1):
            if f in ball and not ball[f][3]:
                if last - f > max_gap:
                    break
                last = f
                y = ball[f][1]
                side = "far" if y < ny - margin_px else ("near" if y > ny + margin_px else None)
                if side:
                    seq.append(side)
        for side in reversed(seq):  # oldest first, runs of the same side
            if runs and runs[-1][0] == side:
                runs[-1][1] += 1
            else:
                runs.append([side, 1])
        runs = [r for r in runs if r[1] >= min_run]
        merged = []
        for side, n in runs:
            if merged and merged[-1][0] == side:
                merged[-1][1] += n
            else:
                merged.append([side, n])
        runs = merged
    sides = [r[0] for r in runs]
    if sides and sides[-1] != h1.side:  # ball not yet seen arriving on the hitter's side
        sides.append(h1.side)
    if len(sides) < 2:
        lo = h1.frame - int(1.6 * fps)
        if any(e.kind == "bounce" and lo <= e.frame < h1.frame - 1 and e.side == h1.side for e in all_events):
            return ["far" if h1.side == "near" else "near", h1.side]  # incoming bounce: at least the serve
        return [h1.side]
    return sides


def add_missed_serves(rallies: list[Rally], ball: dict, players: dict, all_events: list[Event],
                      homographies: dict, fps: float) -> int:
    """Insert inferred shots for every net crossing seen before the first detected hit (missed serve,
    or serve + return, ...), so the server and the serve/return/third-shot order stay right."""
    n = 0
    for r in rallies:
        hits = [e for e in r.events if e.kind == "hit"]
        if not hits:
            continue
        h1 = hits[0]
        sides = crossings_before(h1, ball, all_events, homographies, fps)
        missed = sides[:-1]  # every side before the hitter's own run had a shot that sent the ball over
        if len(missed) > 2:  # more than serve + return missed: the older crossings are balls fed back between points
            missed = missed[-2:]
        if not missed:
            continue
        new = []
        for k, side in enumerate(missed):
            f = max(h1.frame - int((len(missed) - k) * 1.1 * fps), 0)  # ~1.1 s per missed shot
            p = None
            for g in [f + d for d in range(0, 30)] + [f - d for d in range(1, 30)]:
                p = next((q for q in players.get(g, []) if q["side"] == side), None)
                if p:
                    break
            if p is None:
                new = []
                break
            new.append(Event("hit", f, f / fps, tuple(p.get("foot", p["court"])), 0.5, player=p.get("id"),
                             side=side, court=tuple(p["court"]), inferred=True))
        if new:
            r.events[:0] = new
            r.start_frame = new[0].frame
            n += len(new)
    return n
