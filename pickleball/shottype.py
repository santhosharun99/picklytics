"""Learned shot types (replaces the rule-based type for rally shots; serves and returns keep the rules).

Features per shot: hitter / opponent depth, ball speed, net clearance, landing depth, flight time, volley,
shot number, previous shot, and body pose around contact. Trained on analyst-tagged shots
(training/shot_type_cv.py --save -> models/shot_clf.joblib). Taxonomy follows the tags: drive, drop, dink,
reset, groundstroke, lob, overhead (volley is a separate flag). Leave-one-rally-out CV on 100 tagged rally shots:
83% vs 57% for the rules - BUT leave-one-MATCH-out (the honest test) gives 74% = always "drive", macro-F1 0.12
(rules 0.16): soft shots are not learned yet. Needs varied tags from several matches (doubles).
"""
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "shot_clf.joblib"
KP_CONF = 0.5     # pose keypoints below this confidence are ignored
UNSURE = 0.6     # below this top-class probability a shot is flagged shot_unsure (review queue)
_model = None


def _pose(players: dict, pid, f, fps: float = 30.0):
    """Body cues around contact (COCO keypoints, image y grows downward), scaled by torso length:
    highest wrist above the shoulders (overheads / high volleys), fastest wrist speed in the 0.2 s before contact
    (big swing = drive, compact push = dink / reset), and crouch (shoulder-hip height vs box height).
    Keypoints below KP_CONF confidence are ignored; wrist speed is per second (torso lengths / s)."""
    hi, sp, cr, prev = [], [], [], None
    for g in range(f - 6, f + 3):
        p = next((q for q in players.get(str(g), []) if q.get("id") == pid), None)
        if not p:
            prev = None; continue
        k = np.array(p["kp"], float)
        ok = k[:, 2] >= KP_CONF if k.shape[1] > 2 else np.ones(len(k), bool)
        if not (ok[[5, 6, 11, 12]].all()):           # no reliable torso: skip this frame
            prev = None; continue
        torso = max(1.0, (k[11, 1] + k[12, 1]) / 2 - (k[5, 1] + k[6, 1]) / 2)
        sh = (k[5, 1] + k[6, 1]) / 2
        wr = [i for i in (9, 10) if ok[i]]
        if wr:
            hi.append((sh - min(k[i, 1] for i in wr)) / torso)
        cr.append(torso / max(1.0, p["box"][3] - p["box"][1]))
        w = np.where(ok[[9, 10], None], k[[9, 10], :2], np.nan)
        if prev is not None and g <= f:
            d = np.linalg.norm(w - prev, axis=1)
            if np.isfinite(d).any():
                sp.append(float(np.nanmax(d)) / torso * fps)
        prev = w
    return [max(hi) if hi else np.nan, max(sp) if sp else np.nan, float(np.median(cr)) if cr else np.nan]


def features(A: dict, players: dict | None = None) -> dict:
    """frame -> feature vector for every analysed shot (also used at prediction time)."""
    fps = A["meta"]["fps"]
    pos = sorted(((p["t"], p["p"]) for p in A["positions"]), key=lambda x: x[0])
    ts = np.array([t for t, _ in pos])
    by_rally = {}
    for s in sorted(A["shots"], key=lambda s: s["frame"]):
        by_rally.setdefault(s["rally"], []).append(s)
    out = {}
    for shots in by_rally.values():
        for k, s in enumerate(shots):
            hx, hy = s.get("hitter_court") or (np.nan, np.nan)
            depth = abs(hy - 22) if hy == hy else np.nan
            i = int(np.clip(np.searchsorted(ts, s["frame"] / fps), 0, len(ts) - 1)) if len(ts) else None
            # opponents = players on the other side of the net (in doubles the other non-hitter is a teammate);
            # the one nearest the net matters most (is someone up at the kitchen?)
            mine_far = hy < 22 if hy == hy else s.get("side") == "far"
            opp = [v for pid, v in (pos[i][1].items() if i is not None else []) if pid != s["player"] and (v[1] < 22) != mine_far]
            od = min(abs(v[1] - 22) for v in opp) if opp else np.nan
            land = s.get("land_3d") or s.get("landing_court")
            ld = abs(land[1] - 22) if land else np.nan
            prev = shots[k - 1] if k else {}
            mph = s.get("ball_mph") or s.get("speed_mph") or np.nan
            out[s["frame"]] = [depth, abs(hx - 10) if hx == hx else np.nan, od, mph,
                               s.get("net_clear_ft") if s.get("net_clear_ft") is not None else np.nan,
                               ld, s.get("flight_s") or np.nan, float(bool(s.get("volley"))), min(k, 6),
                               (prev.get("ball_mph") or prev.get("speed_mph") or np.nan) if prev else np.nan,
                               abs(prev["hitter_court"][1] - 22) if prev.get("hitter_court") else np.nan,
                               *(_pose(players, s["player"], s["frame"], fps) if players else [np.nan] * 3)]
    return out


FEATURES = ["hitter_depth", "hitter_off_centre", "opp_depth", "mph", "net_clear", "land_depth", "flight_s",
            "volley", "shot_no", "prev_mph", "prev_depth", "wrist_above_shoulder", "swing_speed", "crouch"]



def apply(result: dict, players_by_frame: dict) -> int:
    """Overwrite shot_type of rally shots in an analysis result (dict) with the learned model. -> shots changed."""
    global _model
    if not MODEL_PATH.exists():
        return 0
    if _model is None:
        import joblib
        _model = joblib.load(MODEL_PATH)
    players = {str(f): ps for f, ps in players_by_frame.items()}
    F = features(result, players)
    by_rally = {}
    for s in sorted(result["shots"], key=lambda s: s["frame"]):
        by_rally.setdefault(s["rally"], []).append(s)
    changed = 0
    for shots in by_rally.values():
        for k, s in enumerate(shots):
            if s.get("shot_type") in ("serve", "return") or s["frame"] not in F:
                continue
            x = np.nan_to_num(np.array([F[s["frame"]]], float), nan=-1)
            pr = _model["model"].predict_proba(x)[0]
            t = str(_model["model"].classes_[int(np.argmax(pr))])
            s["shot_prob"] = round(float(pr.max()), 2)
            s["shot_unsure"] = bool(pr.max() < UNSURE)
            server_third = k == 2 and shots[0]["player"] == s["player"]
            t = f"third-shot {t}" if server_third and t in ("drive", "drop", "groundstroke") else t
            s["shot_type_rules"] = s.get("shot_type")
            changed += t != s.get("shot_type")
            s["shot_type"] = t
    return changed
