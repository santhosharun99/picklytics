"""Per-shot observed event sequence: hitter contact -> outgoing flight -> first bounce | opponent contact | net.

The outgoing flight of shot k is the window from its contact to the next contact (shot k+1, normally the opponent);
nothing observed after that next contact belongs to shot k. Status of the outgoing flight:
  bounced       an observed bounce on the opponent's half comes first (it is the observed landing; a later
                opponent contact is kept as the next shot);
  intercepted   the opponent hit it before any bounce - ONLY when evidence supports that no bounce was missed:
                the next contact is flagged a volley AND the ball was tracked densely enough through the window to
                see a bounce, AND the rules allow it (serve and return must bounce: two-bounce rule);
  net           a net event (physics 'net' junction) comes first - no extrapolation through the net;
  unresolved    anything else (missed bounce, sparse tracking, rally ended with nothing observed, a bounce on the
                hitter's own half without a confirmed net event).
Locations are kept apart (docs/PLACEMENT.md): origin (ball at contact from the 3D fit, else the hitter's feet as a
proxy), observed first bounce (human tag > physics fit > bounce detector), interception (ball at the opponent's
contact, else the opponent's feet as a proxy) and projected landing (counterfactual; 'unknown' here - no validated
projection model yet). Landing state 'not-applicable' (intercepted / net) is never mixed with 'unknown'.
Speed: ONE definition - estimated launch speed off the paddle from the physics fit of the outgoing flight
(pickleball/rally3d.py; pre-interception observations only), used when the flight is physically plausible and at
least MIN_OBS ball positions were seen in the window; else unknown with a reason.
"""
import json
from pathlib import Path

import numpy as np

from .zones import NET_Y, locate

MIN_OBS = 6                   # ball positions in the outgoing window for a speed estimate
COVER_FRAC, COVER_GAP_S = 0.6, 0.2   # tracking dense enough to rule out a missed bounce
MERGE_FRAMES = 4              # physics and detector bounce within this many frames = the same bounce
LABELS = {"full_match": "labels/Mens_Singles_GOLD_2026_US_Open.json",   # hand tags with bounce positions
          "tnppl_1_MS": "labels/TNPPL_S2_KTvRTT_1_MS.json"}


def _opp_half(y: float, hitter_near: bool) -> bool:
    return (y < NET_Y) == hitter_near


def resolve(role: str, hitter_near: bool, events: list[dict], has_next: bool, next_volley: bool | None,
            coverage_ok: bool) -> tuple[str, dict | None, list[str]]:
    """Outgoing-flight status from the events inside the shot's window (pure; tested in tests/test_shot_events.py).
    events: [{"kind": "bounce"|"net", "frame", "x", "y", ...}] between this contact and the next one.
    Returns (status, first_bounce or None, reasons)."""
    evs = sorted(events, key=lambda e: e["frame"])
    first = evs[0] if evs else None
    if first is not None:
        if first["kind"] == "net":
            return "net", None, ["net event before any bounce or contact"]
        if _opp_half(first["y"], hitter_near):
            return "bounced", first, [f"first bounce observed ({first['src']})"]
        return "unresolved", None, ["first bounce on the hitter's own half, no confirmed net event"]
    if not has_next:
        return "unresolved", None, ["no bounce or opponent contact observed after the shot"]
    if role in ("serve", "return"):
        return "unresolved", None, ["the rules require a bounce here; none was observed (missed bounce)"]
    if next_volley and coverage_ok:
        return "intercepted", None, ["opponent contact flagged as a volley and ball tracked through the flight"]
    why = []
    if not next_volley:
        why.append("no bounce observed, but the opponent's contact is not flagged as a volley")
    if not coverage_ok:
        why.append("ball tracking too sparse in the flight to rule out a missed bounce")
    return "unresolved", None, why


def _coverage(ball: dict, f0: int, f1: int, fps: float) -> tuple[bool, int]:
    """(dense enough to rule out a missed bounce, ball positions seen) between two contacts."""
    fr = range(f0 + 2, f1 - 1)
    if len(fr) < 3:
        return False, 0
    seen = [f in ball for f in fr]
    gap = run = 0
    for s in seen:
        run = 0 if s else run + 1
        gap = max(gap, run)
    return (np.mean(seen) >= COVER_FRAC and gap <= COVER_GAP_S * fps), int(sum(seen))


def _human_bounces(run_dir: Path, A: dict):
    """Human bounce events (court position via the ground homography), the frame ranges the tags cover (inside
    them the human tags are the complete bounce record) and the flights the tagger could not judge (bounce not
    visible: 'next_bounce_unknown' on the hit) as (from, to) frames. Positions filled in automatically
    ('xy_src' auto ...) are NOT human and are skipped."""
    lf = LABELS.get(Path(run_dir).name)
    root = Path(run_dir).resolve().parents[1]
    if not lf or not (root / lf).exists():
        return [], [], []
    from .camera3d import calibrate_from_run
    L = json.loads((root / lf).read_text())
    cam = calibrate_from_run(run_dir)
    Hinv = np.linalg.inv(cam.P[:, [0, 1, 3]])
    sw, sh = L.get("source_size", [A["meta"]["width"], A["meta"]["height"]])
    sx, sy = A["meta"]["width"] / sw, A["meta"]["height"] / sh
    out, ranges, unknown = [], [], []
    for r in L["rallies"]:
        if r.get("mode") != "hits+bounces" or r["end"] <= r["start"]:
            continue
        ranges.append((int(r["start"]), int(r["end"])))
        ev = sorted(r["events"], key=lambda e: e["frame"])
        for i, e in enumerate(ev):
            if e["kind"] == "hit" and e.get("next_bounce_unknown"):
                nxt = next((x for x in ev[i + 1:] if x["kind"] == "hit"), None)
                unknown.append((int(e["frame"]), int(nxt["frame"]) if nxt else int(e["frame"]) + 90))
        for e in r["events"]:
            if e["kind"] == "bounce" and "xy" in e and not str(e.get("xy_src", "")).startswith("auto"):
                v = Hinv @ np.array([e["xy"][0] * sx, e["xy"][1] * sy, 1.0])
                out.append({"kind": "bounce", "frame": int(e["frame"]), "x": float(v[0] / v[2]), "y": float(v[1] / v[2]),
                            "src": "human tag", "sd": None, "human": True})
    return out, ranges, unknown


def build_run(run_dir, A: dict, live_by_rally: dict, use_human: bool = True) -> dict:
    """{shot frame: placement record} for the live shots of a run. live_by_rally: {rally index: [analysis shots,
    time-ordered, dead ball removed]}. Uses <run>/rally3d.json (physics), ball_v3.json (+_main) and A['bounces']."""
    rd = Path(run_dir)
    fps = A["meta"]["fps"]
    P = json.loads((rd / "rally3d.json").read_text()) if (rd / "rally3d.json").exists() else {"rallies": {}, "shots": {}}
    bf = rd / "ball_v3.json"
    main = {int(k): v for k, v in json.loads((rd / "ball_v3_main.json").read_text()).items()} if (rd / "ball_v3_main.json").exists() else {}
    ball = {int(k): v for k, v in json.loads(bf.read_text()).items() if v[2] >= 0.3 and main.get(int(k), 1.0) >= 0.6} if bf.exists() else {}
    human, h_ranges, h_unknown = _human_bounces(rd, A) if use_human else ([], [], [])
    in_tags = lambda f: any(a <= f <= b for a, b in h_ranges) and not any(a - 3 <= f <= a + 3 for a, b in h_unknown)
    det = [{"kind": "bounce", "frame": int(b["frame"]), "x": b["court"][0], "y": b["court"][1], "src": "bounce detector", "sd": None}
           for b in A.get("bounces", []) if b.get("court")]
    out = {}
    for idx, shots in live_by_rally.items():
        junc = (P["rallies"].get(str(idx)) or {}).get("junctions", [])
        phys_b = [{"kind": "bounce", "frame": j[1], "x": j[2][0], "y": j[2][1], "src": "physics fit",
                   "sd": float(np.hypot(*(j[3][:2]))) if j[3] and None not in j[3][:2] else None} for j in junc if j[0] == "bounce"]
        phys_n = [{"kind": "net", "frame": j[1], "x": j[2][0], "y": NET_Y, "h": j[2][2], "src": "physics fit"} for j in junc if j[0] == "net"]
        contacts = {j[1]: j for j in junc if j[0] in ("contact", "start")}
        for k, s in enumerate(shots):
            f = s["frame"]
            nxt = shots[k + 1] if k + 1 < len(shots) else None
            f_end = nxt["frame"] if nxt else f + int(3 * fps)
            near = s["side"] == "near"
            role = ("serve", "return", "third")[k] if k < 3 else "later"
            win = lambda e: f < e["frame"] < f_end
            if in_tags(f):  # inside hand-tagged rallies the human bounces are the full record
                evs = [e for e in human if win(e)] + [e for e in phys_n if win(e)]
            else:
                pb = [e for e in phys_b if win(e)]
                db = [e for e in det if win(e) and not any(abs(e["frame"] - p["frame"]) <= MERGE_FRAMES for p in pb)]
                for p in pb:
                    p["also"] = any(abs(e["frame"] - p["frame"]) <= MERGE_FRAMES for e in det if win(e))
                evs = pb + db + [e for e in phys_n if win(e)]
            cov_ok, n_obs = _coverage(ball, f, f_end, fps)
            status, bounce, why = resolve(role, near, evs, nxt is not None, bool(nxt and nxt.get("volley")), cov_ok)
            ph = P["shots"].get(str(f)) or {}
            # origin: ball at contact (3D fit) or the hitter's feet
            cj = contacts.get(min(contacts, key=lambda g: abs(g - f))) if contacts else None
            if ph.get("ok") and ph.get("contact"):
                ox, oy, oev, osd = ph["contact"][0], ph["contact"][1], "ball position at contact (3D fit)", None
            elif s.get("hitter_court"):
                ox, oy, oev, osd = s["hitter_court"][0], s["hitter_court"][1], "hitter's feet (player-position proxy)", None
            else:
                ox = oy = None
            rec = {"role": role, "incoming": "serve" if k == 0 else ("volley" if s.get("volley") else "after bounce"),
                   "incoming_src": "automatic volley flag", "status": status, "status_why": why,
                   "end_frame": int(bounce["frame"]) if bounce else (int(nxt["frame"]) if nxt else None),
                   "origin": None, "landing": None, "interception": None, "projected": None, "speed": None}
            if ox is not None:
                rec["origin"] = {**locate(ox, oy, near, osd).to_json(), "evidence": oev}
            # observed first bounce: observed / not applicable / unknown
            if status == "bounced":
                q = "human-reviewed" if bounce.get("human") else ("two automatic sources agree" if bounce.get("also") else "one automatic source")
                rec["landing"] = {"state": "observed", **locate(bounce["x"], bounce["y"], near, bounce.get("sd")).to_json(),
                                  "source": bounce["src"], "quality": q, "frame": int(bounce["frame"])}
            elif status in ("intercepted", "net"):
                rec["landing"] = {"state": "not-applicable", "why": "the ball did not reach the ground before " + ("the opponent's contact" if status == "intercepted" else "the net")}
            else:
                rec["landing"] = {"state": "unknown", "why": "; ".join(why)}
            # interception: only for intercepted shots
            if status == "intercepted":
                nj = contacts.get(min(contacts, key=lambda g: abs(g - nxt["frame"]))) if contacts else None
                if nj is not None and abs(nj[1] - nxt["frame"]) <= 3 and nj[0] == "contact":
                    ix, iy, iev, isd = nj[2][0], nj[2][1], "ball position at the opponent's contact (3D fit)", (float(np.hypot(*nj[3][:2])) if nj[3] and None not in nj[3][:2] else None)
                else:
                    ix, iy, iev, isd = (nxt.get("hitter_court") or [None, None]) + ["opponent's feet (player-position proxy)", None]
                if ix is not None:
                    rec["interception"] = {"state": "observed", **locate(ix, iy, near, isd).to_json(), "evidence": iev, "frame": int(nxt["frame"])}
                else:
                    rec["interception"] = {"state": "unknown", "why": "no position for the opponent's contact"}
                rec["projected"] = {"state": "unknown", "why": "no validated projection model yet (docs/PLACEMENT.md)"}
            else:
                rec["interception"] = {"state": "not-applicable"}
                rec["projected"] = {"state": "not-applicable" if status == "bounced" else "unknown",
                                    "why": "an observed bounce exists" if status == "bounced" else "outgoing flight not resolved"}
            # speed: one definition (physics launch speed), pre-next-contact observations only
            if ph.get("ok") and ph.get("speed_mph") is not None and n_obs >= MIN_OBS:
                rec["speed"] = {"mph": int(round(ph["speed_mph"])), "definition": "estimated launch speed off the paddle",
                                "source": "physics fit of the outgoing flight", "obs": n_obs}
            else:
                rec["speed"] = {"mph": None, "why": "no plausible 3D flight" if not ph.get("ok") else f"only {n_obs} ball positions in the flight"}
            out[f] = rec
    return out
