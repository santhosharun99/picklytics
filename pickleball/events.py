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

from .court import COURT_L, COURT_W

NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 0, 5, 6, 9, 10


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


def detect_events(ball: dict, players: dict[int, list[dict]], homographies: dict[int, np.ndarray],
                  fps: float, max_gap: int = 6, min_score: float = 1.2,
                  hit_radius: float = 0.45) -> list[Event]:
    frames = sorted(ball)
    if not frames:
        return []
    events: list[Event] = []
    chunks = list(_chunks(frames, max_gap))
    events += _gap_reversals(chunks, ball, players, homographies, fps)
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
    return _clean(events)


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

    def flush():
        if sum(e.kind == "hit" for e in cur) >= min_hits:
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
