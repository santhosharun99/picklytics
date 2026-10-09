"""Mark-once player identity for doubles: the user names each player once, the system follows them.

Partners wear the same kit, so shirt colour separates TEAMS only. Here:
1. Tracklets: the 2+2 on-court players (players.select_players) linked frame to frame by court position
   inside one camera shot (a cut ends every tracklet).
2. Look of a tracklet: ImageNet MobileNetV2 embedding of a few person crops (whole body: shorts, shoes, hair,
   build - not just the shirt) + upper / lower body colour + height (box height over the court scale at the feet).
3. Team of a tracklet: its kit (2-means on shirt signatures, as doubles.team_sides).
4. Marks (labels/<video>.identity.json, made on dashboard/identify.html): {"marks": [{"t", "box", "name",
   "team"}], "answers": [{"tracklet", "name", "yes"}]}. A mark names the tracklet under the clicked box.
5. Per team: names from the marks; each tracklet gets the name whose look is closest, with the rule that two
   tracklets on court at the same time on the same team are different players (pairs solved jointly).
   Prototypes are refined with the confident assignments (self-training, a few rounds). Answers from the
   confirmation queue are hard constraints.
6. Low-margin tracklets -> confirmation queue ("is this Joshika?").
Output: <run>/identity.json  {"tracklets": {id: {...}}, "queue": [...]} and helpers to name hitters.
"""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .players import select_players

LINK_FT = 3.0        # max court distance (ft) between consecutive positions of one tracklet
MAX_GAP = 6          # frames a tracklet may go unseen
MIN_LEN = 12         # frames; shorter tracklets are dropped
CROPS = 6            # crops per tracklet for the look


# ------------------------------------------------------------------ tracklets
def build_tracklets(run_dir) -> list[dict]:
    run_dir = Path(run_dir)
    active, done, nid = {}, [], 0
    prev_f = None
    with (run_dir / "detections.jsonl").open() as fh:
        for line in fh:
            rec = json.loads(line)
            f = rec["f"]
            if not rec.get("court") or rec.get("cut"):
                done += active.values()
                active = {}
                prev_f = None
                continue
            if prev_f is not None and f - prev_f > MAX_GAP:
                done += active.values()
                active = {}
            prev_f = f
            ps = select_players(rec.get("people", []), 2)
            for side in ("near", "far"):
                cur = [p for p in ps if p["side"] == side]
                tr = [t for t in active.values() if t["side"] == side and f - t["frames"][-1] <= MAX_GAP]
                used = set()
                pairs = sorted(((np.hypot(*(np.array(p["court"]) - t["court"][-1])), i, t["id"])
                                for i, p in enumerate(cur) for t in tr), key=lambda x: x[0])
                taken = set()
                for d, i, tid in pairs:
                    if d > LINK_FT or i in used or tid in taken:
                        continue
                    t = active[tid]
                    p = cur[i]
                    t["frames"].append(f)
                    t["court"].append(np.array(p["court"], float))
                    t["box"].append(p["box"])
                    if p.get("sig") is not None:
                        t["sig"].append(np.sqrt(np.asarray(p["sig"], float)))
                    used.add(i)
                    taken.add(tid)
                for i, p in enumerate(cur):
                    if i in used:
                        continue
                    active[nid] = {"id": nid, "side": side, "frames": [f], "court": [np.array(p["court"], float)],
                                   "box": [p["box"]], "sig": [np.sqrt(np.asarray(p["sig"], float))] if p.get("sig") is not None else [],
                                   "H": rec["court"]["H"]}
                    nid += 1
                # close tracklets of this side that were not continued for too long
                for t in list(active.values()):
                    if t["side"] == side and f - t["frames"][-1] > MAX_GAP:
                        done.append(active.pop(t["id"]))
    done += active.values()
    out = []
    for t in done:
        if len(t["frames"]) < MIN_LEN:
            continue
        out.append({"id": len(out), "side": t["side"], "f0": t["frames"][0], "f1": t["frames"][-1],
                    "frames": t["frames"], "court": np.array(t["court"]), "box": np.array(t["box"]),
                    "sig": np.mean(t["sig"], 0) if t["sig"] else None, "H": t["H"]})
    return out


# ------------------------------------------------------------------ who is a player
IMG_H = 1080
SIDELINE_FT = 2.5   # players stay within this of the sidelines; referees (net post) and line judges do not


def box_ok(box, court) -> bool:
    """A player box worth showing / learning from: inside the sidelines and not cut off by the bottom of the
    picture (spectators' heads in front of a low camera are cut off there, and cut-off feet give wrong positions)."""
    return box[3] < IMG_H - 15 and -SIDELINE_FT <= court[0] <= 20 + SIDELINE_FT


def is_player(t) -> bool:
    """Most of the tracklet's boxes are player-like (see box_ok)."""
    ok = [box_ok(b, c) for b, c in zip(t["box"], t["court"])]
    return sum(ok) >= 0.6 * len(ok)


# ------------------------------------------------------------------ looks
_NET = None


def _net():
    global _NET
    if _NET is None:
        import torch
        import torchvision
        m = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier = torch.nn.Identity()
        m.eval()
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _NET = (m.to(dev), dev)
    return _NET


def _embed(crops: list[np.ndarray]) -> np.ndarray:
    import torch
    m, dev = _net()
    x = np.stack([cv2.resize(c, (128, 256))[:, :, ::-1] for c in crops]).astype(np.float32) / 255.0
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    with torch.no_grad():
        e = m(torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2).to(dev)).cpu().numpy()
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def _colour(crop) -> np.ndarray:
    h = crop.shape[0]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    out = []
    for part in (hsv[int(0.15 * h):int(0.5 * h)], hsv[int(0.5 * h):int(0.8 * h)], hsv[int(0.8 * h):]):
        hist = cv2.calcHist([part], [0, 1, 2], None, [8, 3, 3], [0, 180, 0, 256, 0, 256]).ravel()
        out.append(np.sqrt(hist / max(hist.sum(), 1)))
    return np.concatenate(out)


def _height_ft(box, H) -> float:
    """Box height in feet at the player's depth: pixel height over the image length of 1 ft of court width there."""
    x1, y1, x2, y2 = box
    H = np.asarray(H)
    Hi = np.linalg.inv(H)
    c = H @ np.array([(x1 + x2) / 2, y2, 1.0])
    c = c[:2] / c[2]  # image -> court (ft) at the feet
    p = [Hi @ np.array([c[0] + d, c[1], 1.0]) for d in (-1.0, 1.0)]
    px_per_ft = np.hypot(*(p[1][:2] / p[1][2] - p[0][:2] / p[0][2])) / 2.0
    return (y2 - y1) / max(px_per_ft, 1e-6)


def add_looks(tracklets: list[dict], video: str) -> None:
    """Fills t["emb"], t["col"], t["height"] (in place). One sequential pass over the video."""
    want = defaultdict(list)
    for t in tracklets:
        idx = np.linspace(0, len(t["frames"]) - 1, min(CROPS, len(t["frames"]))).astype(int)
        for i in idx:
            want[t["frames"][i]].append((t["id"], i))
    crops = defaultdict(list)
    cap = cv2.VideoCapture(video)
    f, last = 0, max(want) if want else -1
    while f <= last and cap.grab():
        if f in want:
            ok, img = cap.retrieve()
            if ok:
                for tid, i in want[f]:
                    x1, y1, x2, y2 = tracklets[tid]["box"][i].astype(int)
                    x1, y1 = max(x1, 0), max(y1, 0)
                    c = img[y1:y2, x1:x2]
                    if c.size and c.shape[0] > 20 and c.shape[1] > 8:
                        crops[tid].append((c, i))
        f += 1
    for t in tracklets:
        cs = crops.get(t["id"], [])
        if not cs:
            t["emb"] = t["col"] = None
            t["height"] = None
            continue
        t["embs"] = _embed([c for c, _ in cs])                  # per crop (for the per-match partner metric)
        t["cols"] = np.stack([_colour(c) for c, _ in cs])
        t["heights"] = np.array([_height_ft(t["box"][i], t["H"]) for _, i in cs])
        t["emb"] = t["embs"].mean(0)
        t["emb"] /= np.linalg.norm(t["emb"])
        t["col"] = t["cols"].mean(0)
        t["height"] = float(np.median(t["heights"]))
        t["thumb"] = max(cs, key=lambda x: x[0].shape[0])  # biggest crop, for the confirmation queue


# ------------------------------------------------------------------ teams
def add_kits_from_rallies(tracklets: list[dict], run_dir, A: dict) -> None:
    """t["kit"] from the rally-level kit sides (doubles.team_sides: clean, one end change per match): a
    tracklet on the near side during a rally has that rally's near kit. Tracklets between rallies take the
    nearest rally's sides. More reliable than clustering single tracklets, which also picks up officials."""
    from .doubles import team_sides
    ts = team_sides(run_dir, A)
    fps = A["meta"]["fps"]
    rs = [(r["start_t"] * fps, r["end_t"] * fps, ts.get(r["index"], {}).get("near_kit")) for r in A["rallies"]]
    rs = [x for x in rs if x[2] is not None]
    for t in tracklets:
        mid = (t["f0"] + t["f1"]) / 2
        a, b, nk = min(rs, key=lambda x: 0 if x[0] <= mid <= x[1] else min(abs(mid - x[0]), abs(mid - x[1])))
        t["kit"] = nk if t["side"] == "near" else 1 - nk
        t["in_rally"] = any(x[0] <= mid <= x[1] for x in rs)


def add_kits(tracklets: list[dict]) -> None:
    """t["kit"] = 0/1 by 2-means on shirt signatures, weighted by tracklet length (in place)."""
    X = [t for t in tracklets if t["sig"] is not None]
    S = np.stack([t["sig"] for t in X])
    w = np.array([len(t["frames"]) for t in X], float)
    far = np.array([t["side"] == "far" for t in X])
    C = np.stack([np.average(S[far], 0, w[far]), np.average(S[~far], 0, w[~far])])
    for _ in range(50):
        lab = ((S[:, None] - C[None]) ** 2).sum(2).argmin(1)
        newC = np.stack([np.average(S[lab == k], 0, w[lab == k]) if (lab == k).any() else C[k] for k in range(2)])
        if np.allclose(newC, C):
            break
        C = newC
    d = ((S[:, None] - C[None]) ** 2).sum(2)
    for t, dd in zip(X, d):
        t["kit"] = int(dd.argmin())
        t["kit_margin"] = float(abs(dd[0] - dd[1]))
    for t in tracklets:
        t.setdefault("kit", None)


# ------------------------------------------------------------------ assignment
def _feat(t) -> np.ndarray | None:
    if t.get("emb") is None:
        return None
    h = np.array([(t["height"] or 5.5) / 6.0])
    return np.concatenate([t["emb"], 0.6 * t["col"] / max(np.linalg.norm(t["col"]), 1e-6), 0.8 * h])


def _overlaps(a, b) -> bool:
    return a["f0"] <= b["f1"] and b["f0"] <= a["f1"]


def assign(tracklets: list[dict], marks: list[dict], answers: list[dict] | None = None, rounds: int = 4) -> dict:
    """marks: [{"tracklet": id, "name": str, "team": str}] (already resolved to tracklets); answers:
    [{"tracklet", "name", "yes": bool}]. Returns {tracklet id: {"name", "conf", "fixed"}}."""
    T = {t["id"]: t for t in tracklets}
    feats = {i: (_feat(t) if is_player(t) else None) for i, t in T.items()}  # referees / spectators get no name
    fixed, banned = {}, defaultdict(set)
    for m in marks:
        fixed[m["tracklet"]] = m["name"]
    for a in answers or []:
        if a["yes"]:
            fixed[a["tracklet"]] = a["name"]
        else:
            banned[a["tracklet"]].add(a["name"])
    # names of each kit: a name's team comes from the marks' "team" (match-card roster); the team's kit is the
    # majority kit of its marked tracklets (one slip in the tracklet's kit cannot move a player to the other team)
    team_of = {m["name"]: m.get("team") for m in marks}
    votes = defaultdict(lambda: [0, 0])
    for m in marks:
        if T[m["tracklet"]]["kit"] is not None:
            votes[m.get("team") or m["name"]][T[m["tracklet"]]["kit"]] += 1
    kit_of = {tm: int(v[1] > v[0]) for tm, v in votes.items()}
    kit_names = defaultdict(list)
    for n in dict.fromkeys(m["name"] for m in marks):
        k = kit_of[team_of[n] or n]
        kit_names[k].append(n)
    for tid in list(fixed):  # a marked tracklet belongs to its player's team kit
        T[tid] = {**T[tid], "kit": kit_of[team_of[fixed[tid]] or fixed[tid]]}
    protos = {}

    def update_protos(lab):
        for n in {n for ns in kit_names.values() for n in ns}:
            vs = [feats[i] * (3.0 if i in fixed else c) for i, (nm, c) in lab.items() if nm == n and feats[i] is not None]
            if vs:
                p = np.sum(vs, 0)
                protos[n] = p / np.linalg.norm(p)

    lab = {i: (n, 1.0) for i, n in fixed.items()}
    update_protos(lab)
    for _ in range(rounds):
        new = dict((i, v) for i, v in lab.items() if i in fixed)
        for k, names in kit_names.items():
            members = [t for t in tracklets if t["kit"] == k and t["id"] not in fixed and feats[t["id"]] is not None]
            sims = {t["id"]: {n: float(feats[t["id"]] @ protos[n]) if n in protos else -1.0 for n in names} for t in members}
            for t in members:
                for n in banned[t["id"]]:
                    if n in sims[t["id"]]:
                        sims[t["id"]][n] = -9.0
            # partners on court at the same time must differ: solve overlapping same-kit, same-side pairs jointly
            done = set()
            for t in sorted(members, key=lambda t: -len(t["frames"])):
                if t["id"] in done:
                    continue
                mates = [u for u in members if u["id"] != t["id"] and u["id"] not in done and u["side"] == t["side"] and _overlaps(t, u)]
                mates += [T[i] for i in fixed if T[i]["kit"] == k and T[i]["side"] == t["side"] and _overlaps(t, T[i])]
                s = sims[t["id"]]
                ranked = sorted(s.items(), key=lambda x: -x[1])
                if mates and len(names) >= 2:
                    u = max(mates, key=lambda u: min(u["f1"], t["f1"]) - max(u["f0"], t["f0"]))
                    if u["id"] in fixed:
                        opts = [(n, v) for n, v in ranked if n != fixed[u["id"]]]
                        best = opts[0]
                        margin = best[1] - (opts[1][1] if len(opts) > 1 else -1.0)
                        new[t["id"]] = (best[0], max(margin, 0.0))
                        done.add(t["id"])
                        continue
                    su = sims[u["id"]]
                    combos = sorted(((s[a] + su[b], a, b) for a in names for b in names if a != b), reverse=True)
                    (v1, a, b), v2 = combos[0], (combos[1][0] if len(combos) > 1 else v1 - 1)
                    margin = max(v1 - v2, 0.0)
                    new[t["id"]], new[u["id"]] = (a, margin), (b, margin)
                    done |= {t["id"], u["id"]}
                else:
                    margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else -1.0)
                    new[t["id"]] = (ranked[0][0], max(margin, 0.0))
                    done.add(t["id"])
        lab = new
        update_protos(lab)
    return {i: {"name": n, "conf": round(c, 4) if i not in fixed else 1.0, "fixed": i in fixed} for i, (n, c) in lab.items()}


# ------------------------------------------------------------------ helpers
def tracklet_at(tracklets: list[dict], frame: int, side: str | None = None, court=None, box=None):
    """The tracklet seen at `frame` (on `side`), nearest to a court position or best overlapping a box."""
    best, bd = None, 1e9
    for t in tracklets:
        if not (t["f0"] <= frame <= t["f1"]) or (side and t["side"] != side):
            continue
        i = int(np.argmin(np.abs(np.array(t["frames"]) - frame)))
        if abs(t["frames"][i] - frame) > MAX_GAP:
            continue
        if box is not None:
            a, b = np.asarray(box, float), t["box"][i]
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            d = -ix / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix)
        elif court is not None:
            d = float(np.hypot(*(t["court"][i] - np.asarray(court, float))))
        else:
            d = 0.0
        if d < bd:
            best, bd = t, d
    return best


def save(run_dir, tracklets: list[dict], names: dict, queue_n: int = 30) -> dict:
    """<run>/identity.json + thumbnails for the confirmation queue in <run>/identity/."""
    run_dir = Path(run_dir)
    (run_dir / "identity").mkdir(exist_ok=True)
    out = {"tracklets": {}, "queue": []}
    for t in tracklets:
        n = names.get(t["id"])
        out["tracklets"][str(t["id"])] = {"side": t["side"], "f0": int(t["f0"]), "f1": int(t["f1"]), "kit": t["kit"],
                                          "name": n["name"] if n else None, "conf": n["conf"] if n else None,
                                          "height_ft": round(t["height"], 2) if t.get("height") else None}
    cands = sorted([t for t in tracklets if t["id"] in names and not names[t["id"]]["fixed"] and t.get("thumb")],
                   key=lambda t: names[t["id"]]["conf"] / np.sqrt(len(t["frames"])))[:queue_n]
    for t in cands:
        p = run_dir / "identity" / f"t{t['id']}.jpg"
        cv2.imwrite(str(p), cv2.resize(t["thumb"][0], (96, int(96 * t["thumb"][0].shape[0] / max(t["thumb"][0].shape[1], 1)))))
        out["queue"].append({"tracklet": t["id"], "name": names[t["id"]]["name"], "conf": names[t["id"]]["conf"],
                             "t": round(t["f0"] / 30.0, 1), "thumb": f"identity/t{t['id']}.jpg"})
    (run_dir / "identity.json").write_text(json.dumps(out, indent=0))
    return out


# ------------------------------------------------------------------ per-match partner metric
def _crop_feats(t) -> np.ndarray:
    """Per-crop feature rows: CNN embedding + body colour (both unit length) + height."""
    c = t["cols"] / np.maximum(np.linalg.norm(t["cols"], axis=1, keepdims=True), 1e-6)
    return np.hstack([t["embs"], c, (t["heights"][:, None] / 6.0)])


def partner_metric(tracklets: list[dict], kit: int, dims: int = 4, pca: int = 48, min_overlap: int = 15):
    """Feature directions that separate the two partners of one team, learned per match WITHOUT labels:
    crops of one tracklet = same person (within scatter), two same-kit tracklets on court at the same time on the
    same side = the two partners (between scatter, their mean difference). Fisher discriminant (generalised
    eigenvectors) in a PCA space. Returns project(t) -> vector, or None if too few partner pairs."""
    pool = [t for t in tracklets if t.get("kit") == kit and t.get("embs") is not None and len(t["embs"]) >= 2]
    if len(pool) < 6:
        return None
    X = np.vstack([_crop_feats(t) for t in pool])
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = Vt[:pca].T

    def red(t):
        return (_crop_feats(t) - mu) @ P

    R = {t["id"]: red(t) for t in pool}
    k = P.shape[1]
    Sw = np.zeros((k, k))
    for z in R.values():
        d = z - z.mean(0)
        Sw += d.T @ d
    Sb, npair = np.zeros((k, k)), 0
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            if a["side"] == b["side"] and min(a["f1"], b["f1"]) - max(a["f0"], b["f0"]) >= min_overlap:
                d = R[a["id"]].mean(0) - R[b["id"]].mean(0)
                Sb += np.outer(d, d)
                npair += 1
    if npair < 3:
        return None
    Sw /= max(len(X) - len(pool), 1)
    Sb /= npair
    w, V = np.linalg.eig(np.linalg.solve(Sw + 1e-3 * np.trace(Sw) / k * np.eye(k), Sb))
    W = np.real(V[:, np.argsort(-np.real(w))[:dims]])

    def project(t):
        return (red(t) @ W).mean(0)
    return project
