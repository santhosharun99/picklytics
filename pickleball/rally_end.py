"""Rally-end model: which detected touches are live play, and how each point ended.

Trained on labels/endings.json (dashboard/endings.html: per rally the moment of the serve, the moment the point is
over and how it ended - net / out / in-not-returned - or 'only ball handling'). Two learned models:
  1. LIVE TOUCH: every detected paddle-ball touch -> live play or dead ball (picking the ball up, passing it to the
     server, warm-up knocks). Live = between the serve and the end of the point.
  2. ENDING: for the last live touch of a rally -> net / out / in, from what the ball did after it.
Features come from tracking only (ball flight, bounces after the touch, sides, timing, how fast the players move),
never from a scoreboard. training/train_rally_end.py trains and cross-validates (leave one match out).
"""
import json
from pathlib import Path

import numpy as np

from .court import COURT_L, COURT_W, NET_Y

MODEL = Path(__file__).resolve().parents[1] / "models" / "rally_end.joblib"
BALL_FEATS = ["lift_has", "lift_reach", "lift_z_net", "lift_land", "lift_dur",
              "trk_vis", "trk_reach", "trk_end_line", "trk_end_side"]
LIVE_FEATS = ["mph", "mph_known", "flight", "prev_same", "next_same", "gap_prev", "gap_next", "k_from_start",
              "k_from_end", "n_touches", "net_dist", "behind_base", "wide", "b_own", "b_opp_in", "b_opp_out",
              "net_clear", "move_before", "move_after", "move_after2", "t_from_start", "t_to_end"] + BALL_FEATS
FOLLOW_FEATS = ["f_ok", "f_h_net", "f_crossed", "f_land_line", "f_inl_share", "f_inliers", "f_err", "f_piece_s"]
END_FEATS = ["mph", "mph_known", "flight", "net_clear", "b_own", "b_opp_in", "b_opp_out", "b_dist_line", "b_dt",
             "land_out", "land_known", "move_after", "move_after2", "next_touch_gap", "next_same"] + BALL_FEATS + FOLLOW_FEATS


def follow_feats(run_dir, A: dict, shot: dict, cache: dict = {}) -> dict:
    """The after-contact 3D follower (after_contact.follow) as model features for one touch (no next touch)."""
    from .after_contact import follow
    from .camera3d import calibrate_from_run
    run_dir = Path(run_dir)
    if run_dir not in cache:
        cache[run_dir] = (calibrate_from_run(run_dir), _ball_context(run_dir)[1])
    cam, ball = cache[run_dir]
    o = follow(cam, ball, shot["frame"], None, shot.get("hitter_court") or [10.0, 22.0], shot["side"], A["meta"]["fps"])
    ok = o["fit_err_px"] is not None
    return {"f_ok": float(ok), "f_h_net": o["h_at_net"] if o["h_at_net"] is not None else -1.0,
            "f_crossed": -1.0 if o["crossed"] is None else float(o["crossed"]),
            "f_land_line": o["land_line_ft"] if o["land_line_ft"] is not None else 0.0,
            "f_inl_share": o["inlier_share"] or 0.0, "f_inliers": float(o["inliers"] or 0),
            "f_err": o["fit_err_px"] if ok else -1.0, "f_piece_s": o["piece_s"] or 0.0}


def _line_dist(xy) -> float:
    """Signed distance (ft) from the court boundary: + inside, - outside."""
    x, y = xy
    return float(min(x, COURT_W - x, y, COURT_L - y))


def _motion(A: dict):
    """t -> mean player speed (ft/s) from the 10 Hz positions (nearest-neighbour matching between samples,
    robust to identity swaps)."""
    P = [(x["t"], np.array(list(x["p"].values()), float)) for x in A.get("positions", []) if x["p"]]
    ts, sp = [], []
    for (t0, a), (t1, b) in zip(P, P[1:]):
        if 0 < t1 - t0 <= 0.25 and len(a) and len(b):
            d = np.sqrt(((a[:, None] - b[None]) ** 2).sum(2)).min(1)
            ts.append(t1)
            sp.append(float(np.median(d)) / (t1 - t0))
    ts, sp = np.array(ts), np.clip(np.array(sp), 0, 30)

    def mean_speed(a, b):
        m = (ts >= a) & (ts <= b)
        return float(np.mean(sp[m])) if m.any() else -1.0
    return mean_speed


def _ball_context(run_dir):
    """3D paths per hit frame (lift.json), image ball track (render_data.json) and per-frame image->court
    homographies (detections.jsonl) for a run; cached in <run>/ball_context.pkl."""
    import pickle
    run_dir = Path(run_dir)
    cache = run_dir / "ball_context.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    lift = {int(k): v for k, v in json.loads((run_dir / "lift.json").read_text()).items()} if (run_dir / "lift.json").exists() else {}
    rd = json.loads((run_dir / "render_data.json").read_text()) if (run_dir / "render_data.json").exists() else {}
    ball = {int(k): v for k, v in rd.get("ball", {}).items()}
    H = {}
    with (run_dir / "detections.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("court"):
                H[r["f"]] = np.asarray(r["court"]["H"], float)
    ctx = (lift, ball, H)
    cache.write_bytes(pickle.dumps(ctx))
    return ctx


def _toward(side, y) -> float:
    """How far past the net (ft) a court y is, seen from a hitter on `side` (+ = on the opponent's half)."""
    return (NET_Y - y) if side == "near" else (y - NET_Y)


def _ball_feats(s, fps, ctx) -> dict:
    lift, ball, H = ctx
    f0 = s["frame"]
    out = {k: 0.0 for k in BALL_FEATS}
    out.update({"lift_z_net": -1.0, "lift_reach": -30.0, "trk_reach": -30.0})
    L = lift.get(f0)
    if L and L.get("path"):
        P = np.asarray(L["path"], float)
        tw = np.array([_toward(s["side"], y) for y in P[:, 1]])
        out["lift_has"] = 1.0
        out["lift_reach"] = float(tw.max())
        cross = np.where((tw[:-1] < 0) & (tw[1:] >= 0))[0]
        if len(cross):
            i = cross[0]
            a = -tw[i] / max(tw[i + 1] - tw[i], 1e-6)
            out["lift_z_net"] = float(P[i, 2] + a * (P[i + 1, 2] - P[i, 2]))
        land = P[int(np.argmin(P[:, 2]))] if tw.max() > 0 else P[-1]
        out["lift_land"] = _line_dist(land[:2])
        out["lift_dur"] = (L["f1"] - L["f0"]) / fps
    pts = []
    for f in range(f0 + 2, f0 + int(2.5 * fps)):
        b = ball.get(f)
        Hf = H.get(f)
        if b and Hf is not None and not b[2]:
            c = Hf @ np.array([b[0], b[1], 1.0])
            if abs(c[2]) > 1e-9:
                pts.append(c[:2] / c[2])
    out["trk_vis"] = len(pts) / (2.5 * fps)
    if pts:
        pts = np.array(pts)
        tw = np.array([_toward(s["side"], y) for y in pts[:, 1]])
        out["trk_reach"] = float(tw.max())
        out["trk_end_line"] = _line_dist(pts[-1])
        out["trk_end_side"] = float(tw[-1] > 0)
    return out


def touch_features(A: dict, run_dir=None) -> list[dict]:
    """One row per detected (not inferred) touch: {"rally", "index", "t", "side", **features}.
    run_dir adds the ball-flight features (3D path, image track); without it they are zeros."""
    move = _motion(A)
    ctx = _ball_context(run_dir) if run_dir else ({}, {}, {})
    fps = A["meta"]["fps"]
    bounces = sorted(A.get("bounces", []), key=lambda b: b["t"])
    by = {}
    for s in A["shots"]:
        if not s.get("inferred"):
            by.setdefault(s["rally"], []).append(s)
    rows = []
    for r in A["rallies"]:
        sh = sorted(by.get(r["index"], []), key=lambda s: s["t"])
        n = len(sh)
        for k, s in enumerate(sh):
            prv, nxt = (sh[k - 1] if k else None), (sh[k + 1] if k + 1 < n else None)
            horizon = nxt["t"] if nxt else s["t"] + 2.5
            after = [b for b in bounces if s["t"] + 0.05 < b["t"] <= horizon]
            own = [b for b in after if b["side"] == s["side"]]
            opp = [b for b in after if b["side"] != s["side"]]
            hc = s.get("hitter_court") or [COURT_W / 2, NET_Y]
            mph = s.get("speed_mph") or s.get("ball_mph")
            fb = (opp or own or [None])[0]
            land = s.get("land_3d") or s.get("landing_court")
            rows.append({
                "rally": r["index"], "index": s["index"], "t": s["t"], "side": s["side"],
                "mph": mph or 0.0, "mph_known": float(mph is not None), "flight": s.get("flight_s") or 0.0,
                "prev_same": -1.0 if prv is None else float(prv["side"] == s["side"]),
                "next_same": -1.0 if nxt is None else float(nxt["side"] == s["side"]),
                "gap_prev": min(s["t"] - prv["t"], 10.0) if prv else 10.0,
                "gap_next": min(nxt["t"] - s["t"], 10.0) if nxt else 10.0,
                "next_touch_gap": min(nxt["t"] - s["t"], 10.0) if nxt else 10.0,
                "k_from_start": float(k), "k_from_end": float(n - 1 - k), "n_touches": float(n),
                "net_dist": abs(hc[1] - NET_Y), "behind_base": max(0.0, abs(hc[1] - NET_Y) - NET_Y),
                "wide": max(0.0, -hc[0], hc[0] - COURT_W),
                "b_own": float(len(own)), "b_opp_in": float(sum(_line_dist(b["court"]) >= 0 for b in opp)),
                "b_opp_out": float(sum(_line_dist(b["court"]) < 0 for b in opp)),
                "b_dist_line": _line_dist(fb["court"]) if fb else 0.0, "b_dt": (fb["t"] - s["t"]) if fb else -1.0,
                "land_out": float(_line_dist(land) < 0) if land else 0.0, "land_known": float(land is not None),
                "net_clear": s.get("net_clear_ft") if s.get("net_clear_ft") is not None else -9.0,
                "move_before": move(s["t"] - 1.0, s["t"]), "move_after": move(s["t"], s["t"] + 1.5),
                "move_after2": move(s["t"] + 1.5, s["t"] + 4.0),
                "t_from_start": s["t"] - sh[0]["t"], "t_to_end": sh[-1]["t"] - s["t"],
                **_ball_feats(s, fps, ctx),
            })
    return rows


def labels_for(rows: list[dict], ans: dict, slack: float = 0.6) -> list[int | None]:
    """Live (1) / dead (0) per touch row from one rally's answer; None = cannot tell (point unclear)."""
    if ans.get("no_rally"):
        return [0] * len(rows)
    lo = ans["serve_t"] - slack if ans.get("serve_t") is not None else -1e9
    hi = ans["end_t"] + 0.2
    return [int(lo <= r["t"] <= hi) for r in rows]


def apply(A: dict, run_dir, threshold: float = 0.5) -> dict:
    """analysis dict with dead-ball touches removed (and rallies left with no live touch), using the trained
    models; each kept rally gets "ending" {"type", "p"} from the ending model. Adds A["rally_end"] stats."""
    import joblib
    M = joblib.load(MODEL)
    rows = touch_features(A, run_dir)
    if not rows:
        return A
    X = np.array([[r[f] for f in LIVE_FEATS] for r in rows])
    p = M["live"].predict_proba(X)[:, 1]
    live = {(r["rally"], r["index"]): float(q) for r, q in zip(rows, p)}
    keep_shot, keep_r, endings = [], [], {}
    by = {}
    for s in A["shots"]:
        by.setdefault(s["rally"], []).append(s)
    row_of = {(r["rally"], r["index"]): r for r in rows}
    for r in A["rallies"]:
        sh = sorted(by.get(r["index"], []), key=lambda s: s["t"])
        real = [s for s in sh if not s.get("inferred")]
        ok = [s for s in real if live[(s["rally"], s["index"])] >= threshold]
        if not ok:
            continue
        first, last = ok[0], ok[-1]
        # live play is one stretch: from the first to the last live touch; inferred serves before it stay
        kept = [s for s in sh if (s.get("inferred") and s["t"] <= first["t"]) or (first["t"] <= s["t"] <= last["t"] and not s.get("inferred"))]
        keep_shot += kept
        lr = row_of[(last["rally"], last["index"])]
        q = M["ending"].predict_proba(np.array([[lr[f] for f in END_FEATS]]))[0]
        endings[r["index"]] = {"type": M["ending"].classes_[int(q.argmax())], "p": round(float(q.max()), 2),
                               "last_touch": last["index"]}
        keep_r.append({**r, "start_t": kept[0]["t"], "end_t": max(last["t"] + 1.0, kept[-1]["t"]), "ending": endings[r["index"]]})
    out = dict(A)
    out["rallies"], out["shots"] = keep_r, keep_shot
    out["rally_end"] = {"touches_dropped": sum(1 for s in A["shots"] if not s.get("inferred")) - sum(1 for s in keep_shot if not s.get("inferred")),
                        "rallies_dropped": len(A["rallies"]) - len(keep_r)}
    return out


def dead_touches(A: dict, run_dir, threshold: float = 0.5) -> dict:
    """{(rally, shot index): p_live} for the touches the model calls dead ball. Per rally, live play is ONE
    stretch (serve .. last shot): the most likely contiguous block of live touches given the per-touch
    probabilities (no block = the whole 'rally' was ball handling). Inferred serves before a dead first
    touch are dead too. Report builders leave these out of every shot statistic; scoring is unchanged."""
    import joblib
    M = joblib.load(MODEL)
    rows = touch_features(A, run_dir)
    if not rows:
        return {}
    p = M["live"].predict_proba(np.array([[r[f] for f in LIVE_FEATS] for r in rows]))[:, 1]
    by = {}
    for r, q in zip(rows, p):
        by.setdefault(r["rally"], []).append((r["t"], r["index"], float(q)))
    dead = {}
    for rally, xs in by.items():
        xs.sort()
        lp = np.log(np.clip([q for _, _, q in xs], 1e-4, 1))
        ln = np.log(np.clip([1 - q for _, _, q in xs], 1e-4, 1))
        best, ab = ln.sum(), None
        for a in range(len(xs)):
            for b in range(a, len(xs)):
                sc = ln[:a].sum() + lp[a:b + 1].sum() + ln[b + 1:].sum()
                if sc > best:
                    best, ab = sc, (a, b)
        for k, (t, idx, q) in enumerate(xs):
            if ab is None or not (ab[0] <= k <= ab[1]):
                dead[(rally, idx)] = q
        first_live_t = xs[ab[0]][0] if ab else 1e9
        for s in A["shots"]:
            if s["rally"] == rally and s.get("inferred") and (ab is None or s["t"] > first_live_t or
                                                                 (rally, xs[0][1]) in dead and s["t"] < xs[0][0]):
                dead[(rally, s["index"])] = 0.0
    return dead


def endings(A: dict, run_dir, dead: dict) -> dict:
    """{rally: {"type": net|out|in, "p": confidence}} from the ending model, for each rally's last live touch."""
    import joblib
    M = joblib.load(MODEL)
    last = {}
    for r in touch_features(A, run_dir):
        if (r["rally"], r["index"]) not in dead and (r["rally"] not in last or r["t"] > last[r["rally"]]["t"]):
            last[r["rally"]] = r
    out = {}
    shot_of = {(s["rally"], s["index"]): s for s in A["shots"]}
    for rally, r in last.items():
        r = {**r, **follow_feats(run_dir, A, shot_of[(r["rally"], r["index"])])}
        q = M["ending"].predict_proba(np.array([[r[f] for f in END_FEATS]]))[0]
        out[rally] = {"type": str(M["ending"].classes_[int(q.argmax())]), "p": round(float(q.max()), 2)}
    return out
