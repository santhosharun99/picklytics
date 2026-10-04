"""2D -> 3D ball "lifter": a small network that turns what one camera saw into the ball's 3D path.

Idea from recent research (e.g. "Where Is The Ball", CVPR workshops 2025): train ONLY on simulated
flights, filmed by simulated cameras, so no tagging is needed and the network can't memorise a match.

Simulator (one shot = from one hit to the next hit):
  - hitter on one side, ball aimed at the other side, flight time and drag drawn at random
  - gravity + linear air drag; optional bounce (restitution, friction) -> then flies on to the next hit;
    some shots are volleys (no bounce)
  - a random broadcast camera behind the baseline (position, height, zoom) or one of our real cameras
  - realistic tracking: pixel noise, missing frames, blur gaps near contacts, occasional wrong detections
Network input per frame: the camera ray through the ball pixel (unit direction, world coordinates), the
camera centre, a visible flag and time -> it is camera-independent, so it works for any calibrated venue.
Output per frame: the ball's (x, y, h) in court feet (also for frames where the ball was not seen).

Court frame: x across (0..20), y along (0 far baseline .. 44 near), h = height (ft). Train with
`python -m pickleball.lifter train` (GPU recommended), apply with `lift_track`.
"""
import math, os, time

import numpy as np

G = 32.17
FPS = 29.97
MAX_T = 96          # frames per sample (3.2 s)
COURT_C = np.array([10.0, 22.0, 0.0])
SCALE = 25.0


# ------------------------------------------------------------------ cameras
def look_at_camera(C, target, f, size=(1920, 1080), roll=0.0):
    """K, R, t (OpenCV, world = (x, y, z=-h)) for a camera at C=(x, y, h) looking at target=(x, y, h)."""
    Cw, Tw = np.array([C[0], C[1], -C[2]]), np.array([target[0], target[1], -target[2]])
    z = Tw - Cw; z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, -1.0])               # world "up" is -z in OpenCV coordinates
    x = np.cross(z, up); x /= np.linalg.norm(x)  # image right
    y = np.cross(z, x)                           # image down
    R = np.stack([x, y, z])
    if roll:
        c, s = math.cos(roll), math.sin(roll)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
    return K, R, -R @ Cw


def random_camera(rng):
    """Broadcast-like camera: behind the near baseline, raised, looking at the court."""
    C = (10 + rng.uniform(-4, 4), 44 + rng.uniform(18, 55), rng.uniform(9, 30))
    target = (10 + rng.uniform(-1.5, 1.5), rng.uniform(14, 26), 0.0)
    f = rng.uniform(1800, 3600)
    return look_at_camera(C, target, f, roll=rng.uniform(-0.02, 0.02))


def project(K, R, t, X):
    Xw = X * np.array([1.0, 1.0, -1.0])
    c = Xw @ R.T + t
    p = c @ K.T
    return p[:, :2] / p[:, 2:3], c[:, 2]


def rays(K, R, t, uv):
    """Unit ray directions in (x, y, h) world coordinates and the camera centre (x, y, h)."""
    Cw = -R.T @ t
    d = np.linalg.solve(K, np.c_[uv, np.ones(len(uv))].T).T @ R  # R^T applied to each row
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d * np.array([1.0, 1.0, -1.0]), Cw * np.array([1.0, 1.0, -1.0])


# ------------------------------------------------------------------ physics
def _fly(p0, v0, beta, ts):
    """Positions with gravity + linear drag (beta 1/s) from p0, v0 at times ts (s)."""
    ts = np.asarray(ts, float)[:, None]
    if beta < 1e-6:
        P = p0 + v0 * ts
        P[:, 2] -= 0.5 * G * ts[:, 0] ** 2
        return P
    e = np.exp(-beta * ts)
    P = p0 + v0 * (1 - e) / beta
    P[:, 2] -= G / beta * ts[:, 0] - G / beta ** 2 * (1 - e[:, 0])
    return P


def _vel(v0, beta, t):
    if beta < 1e-6:
        return v0 - np.array([0, 0, G * t])
    e = math.exp(-beta * t)
    return v0 * e - np.array([0, 0, G / beta * (1 - e)])


def _aim(p0, p1, T, beta):
    """Initial velocity to go from p0 to p1 in T seconds."""
    if beta < 1e-6:
        v = (p1 - p0) / T
        v[2] += 0.5 * G * T
        return v
    e = math.exp(-beta * T)
    k = (1 - e) / beta
    v = (p1 - p0) / k
    v[2] += (G / beta * T - G / beta ** 2 * (1 - e)) / k
    return v


def simulate_shot(rng):
    """One shot from a hit to the next hit. Returns positions (n, 3) at FPS, bounce index or -1."""
    near = rng.random() < 0.5
    hy = rng.uniform(30, 50) if near else rng.uniform(-6, 14)
    p0 = np.array([rng.uniform(-2, 22), hy, rng.uniform(0.6, 7.5)])
    kind = rng.choice(["drive", "soft", "lob", "serve"], p=[0.45, 0.3, 0.1, 0.15])
    other_lo, other_hi = (0.0, 21.5) if near else (22.5, 44.0)
    if kind == "soft":   # dinks / drops land near the net
        ly = rng.uniform(15, 22) if near else rng.uniform(22, 29)
        T1 = rng.uniform(0.7, 1.6)
    elif kind == "lob":
        ly = rng.uniform(other_lo + 1, other_lo + 8) if near else rng.uniform(other_hi - 8, other_hi - 1)
        T1 = rng.uniform(1.4, 2.4)
    else:
        ly = rng.uniform(other_lo, other_hi)
        T1 = rng.uniform(0.35, 1.1)
    p1 = np.array([rng.uniform(-1, 21), ly, 0.0])
    beta = rng.uniform(0.2, 1.3)
    v0 = _aim(p0, p1, T1, beta)
    volley = kind == "drive" and rng.random() < 0.25
    if volley:  # intercepted in the air before the bounce
        Tv = T1 * rng.uniform(0.55, 0.95)
        ts = np.arange(0, Tv, 1 / FPS)
        return _fly(p0, v0, beta, ts), -1
    ts1 = np.arange(0, T1, 1 / FPS)
    A = _fly(p0, v0, beta, ts1)
    vb = _vel(v0, beta, T1)
    e, mu = rng.uniform(0.55, 0.8), rng.uniform(0.65, 0.9)
    v2 = np.array([vb[0] * mu, vb[1] * mu, -vb[2] * e])
    T2 = rng.uniform(0.15, 1.0)
    ts2 = np.arange(ts1[-1] + 1 / FPS - T1, T2, 1 / FPS)
    Bp = _fly(np.array([p1[0], p1[1], 0.0]), v2, beta, ts2)
    return np.vstack([A, Bp]), len(A)


def make_sample(rng, camera=None):
    """Synthetic (inputs (MAX_T, 9), target (MAX_T, 3), mask (MAX_T,)) like real tracking of one shot."""
    while True:
        X, bi = simulate_shot(rng)
        if 6 <= len(X) <= MAX_T and X[:, 2].min() > -0.2:
            break
    K, R, t = camera if camera is not None else random_camera(rng)
    uv, depth = project(K, R, t, X)
    n = len(X)
    vis = (depth > 1) & (uv[:, 0] > 0) & (uv[:, 0] < 1920) & (uv[:, 1] > 0) & (uv[:, 1] < 1080)
    vis &= rng.random(n) > rng.uniform(0.0, 0.25)                 # random misses
    for c in [0, n - 1] + ([bi] if bi > 0 else []):               # blur / occlusion near contacts
        if rng.random() < 0.6:
            g = rng.integers(1, 6)
            vis[max(0, c - g // 2): c + g] = False
    uvn = uv + rng.normal(0, rng.uniform(0.8, 3.5), uv.shape)     # pixel noise
    out = rng.random(n) < 0.02                                    # wrong detections
    uvn[out] += rng.normal(0, 80, (out.sum(), 2))
    return encode(K, R, t, uvn, vis, n), _pad((X - COURT_C) / SCALE, MAX_T), _pad(np.ones(n), MAX_T)


def encode(K, R, t, uv, vis, n):
    d, C = rays(K, R, t, np.nan_to_num(uv))
    feats = np.zeros((MAX_T, 9), np.float32)
    feats[:n, 0:3] = d * vis[:, None]
    feats[:n, 3:6] = (C - COURT_C) / SCALE
    feats[:n, 6] = vis
    feats[:n, 7] = np.arange(n) / FPS
    feats[:n, 8] = 1.0                                            # inside the shot
    return feats


def _pad(a, L):
    a = np.asarray(a, np.float32)
    out = np.zeros((L,) + a.shape[1:], np.float32)
    out[:len(a)] = a
    return out


# ------------------------------------------------------------------ network
def build_net():
    import torch.nn as nn

    class Lifter(nn.Module):
        def __init__(self, h=192):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(9, h), nn.GELU())
            self.rnn = nn.GRU(h, h, num_layers=3, batch_first=True, bidirectional=True, dropout=0.1)
            self.out = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 3))

        def forward(self, x):
            y, _ = self.rnn(self.inp(x))
            return self.out(y)
    return Lifter()


def train(out_path, minutes=60.0, batch=256, seed=0, real_cameras=()):
    import torch
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    net = build_net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    cams = [tuple(np.asarray(a) for a in c) for c in real_cameras]

    def batch_data(n):
        xs, ys, ms = [], [], []
        for _ in range(n):
            cam = cams[rng.integers(len(cams))] if cams and rng.random() < 0.3 else None
            x, y, m = make_sample(rng, cam)
            xs.append(x); ys.append(y); ms.append(m)
        return (torch.tensor(np.stack(xs)).to(dev), torch.tensor(np.stack(ys)).to(dev), torch.tensor(np.stack(ms)).to(dev))

    val = batch_data(1024)
    t0, step, best = time.time(), 0, 1e9
    budget = minutes * 60
    while time.time() - t0 < budget:
        x, y, m = batch_data(batch)
        pred = net(x)
        err = ((pred - y) ** 2).sum(-1)
        vel = ((pred[:, 1:] - pred[:, :-1]) - (y[:, 1:] - y[:, :-1])) ** 2
        loss = (err * m).sum() / m.sum() + 20 * (vel.sum(-1) * m[:, 1:]).sum() / m[:, 1:].sum()
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
        for g in opt.param_groups:
            g["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * min(1.0, (time.time() - t0) / budget)))
        if step % 200 == 0:
            with torch.no_grad():
                p = net(val[0])
                e = (((p - val[1]) ** 2).sum(-1).sqrt() * SCALE * val[2]).sum() / val[2].sum()
            print(f"[{(time.time() - t0) / 60:5.1f} min] step {step} loss {loss.item():.4f}  val mean 3D error {e.item():.2f} ft", flush=True)
            if e.item() < best:
                best = e.item()
                torch.save(net.state_dict(), out_path)
    return best


# ------------------------------------------------------------------ use on real tracks
_NET = None


def load(path):
    global _NET
    import torch
    net = build_net()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    _NET = net
    return net


def lift_track(K, R, t, frames, uv, f0, f1, net=None):
    """3D positions (f1-f0+1, 3) in court feet for the ball between hits at f0 and f1, from the measured
    pixels uv at `frames` (missing frames are fine)."""
    import torch
    n = f1 - f0 + 1
    if n > MAX_T or n < 3:
        return None
    full = np.zeros((n, 2))
    vis = np.zeros(n, bool)
    for f, p in zip(frames, uv):
        if f0 <= f <= f1:
            full[f - f0] = p
            vis[f - f0] = True
    x = encode(K, R, t, full, vis, n)
    with torch.no_grad():
        y = (net or _NET)(torch.tensor(x[None]))[0, :n].numpy()
    return y * SCALE + COURT_C


def _plausible_tail(X, K, R, t, fr, uv, f0, side, max_px=12.0):
    """A rally's last flight has no next hit, which the lifter never saw in training; after the ball goes out
    the tracker may also follow something else. Keep it only up to the first ground contact, and only if it fits
    the video (median <= max_px) and moves like a ball: steadily away from the hitter along the court, without
    turning back sideways. Returns the cut path or None."""
    import cv2
    ground = np.where(X[:, 2] < 0.4)[0]
    if len(ground):
        X = X[:max(int(ground[0]) + 1, 4)]
    keep = np.array(fr) - f0 < len(X)
    if keep.sum() < 4:
        return None
    rv, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints((X[np.array(fr)[keep] - f0] * [1, 1, -1]).astype(np.float64), rv, t.astype(np.float64), K, None)
    if np.median(np.linalg.norm(proj.reshape(-1, 2) - uv[keep], axis=1)) > max_px:
        return None
    dy = np.diff(X[:, 1])
    away = -1 if side == "near" else 1                         # near player hits toward the far end (y down)
    if np.mean(np.sign(dy) == away) < 0.9:
        return None
    dx = np.diff(X[:, 0])
    if dx.max() > 0.15 and dx.min() < -0.15 and min(dx.max(), -dx.min()) > 0.3 * max(dx.max(), -dx.min()):
        return None                                            # sideways U-turn: not a ball flight
    return X


def lift_match(A: dict, ball: dict, cam, model_path=None) -> dict:
    """3D path for every shot (from a detected hit to the next detected hit) of an analysis dict.
    Returns {shot frame: {"f0", "f1", "path" [[x, y, h], ...], "land" (x, y) | None, "bounce_frame", "net_clear_ft"}}."""
    import cv2
    from pathlib import Path
    if _NET is None:
        load(model_path or str(Path(__file__).resolve().parents[1] / "models" / "lifter.pt"))
    R, _ = cv2.Rodrigues(cam.rvec)
    K, t = cam.K, cam.tvec.ravel()
    meas = {f: v[:2] for f, v in ball.items() if not v[3]}
    out = {}
    fps = A["meta"]["fps"]
    for r in A["rallies"]:
        # inferred hits (missed by the detector, deduced from the rules) count as flight boundaries too: the ball
        # itself is usually well seen, and skipping them left whole serves / returns without a 3D path
        shots = sorted([s for s in A["shots"] if s["rally"] == r["index"]], key=lambda s: s["frame"])
        # the rally's last shot: from the last hit to the last time the ball is seen (into the net, out, a winner)
        end_seen = [f for f in range(shots[-1]["frame"] + 1, int(r["end_t"] * fps) + int(fps)) if f in meas] if shots else []
        tail = [{"frame": min(end_seen[-1], shots[-1]["frame"] + MAX_T - 1)}] if len(end_seen) >= 4 else []
        for s0, s1 in zip(shots, shots[1:] + tail):
            f0, f1 = s0["frame"], s1["frame"]
            fr = [f for f in range(f0, f1 + 1) if f in meas]
            if len(fr) < 4 or f1 - f0 + 1 > MAX_T:
                continue
            X = lift_track(K, R, t, fr, np.array([meas[f] for f in fr], float), f0, f1)
            if X is None:
                continue
            if "rally" not in s1:   # rally-ending shot: keep only a trustworthy, physically plausible flight
                X = _plausible_tail(X, K, R, t, fr, np.array([meas[f] for f in fr], float), f0, s0.get("side"))
                if X is None:
                    continue
                f1 = f0 + len(X) - 1
            ib = int(np.argmin(X[:, 2]))
            bounced = X[ib, 2] < 1.0 and 0 < ib < len(X) - 1
            side = np.sign(X[:, 1] - 22.0)
            c = np.where(np.diff(side) != 0)[0]
            clear = None
            if len(c):
                i = c[0]
                a = (22.0 - X[i, 1]) / (X[i + 1, 1] - X[i, 1] + 1e-9)
                clear = float(X[i, 2] + a * (X[i + 1, 2] - X[i, 2]) - 2.9)
            rough = s0.get("inferred") or "rally" not in s1        # flight after a deduced hit, or the rally's last shot
            out[f0] = {"f0": f0, "f1": f1, "path": np.round(X, 2).tolist(), "rough": bool(rough),
                       "land": [round(float(X[ib, 0]), 2), round(float(X[ib, 1]), 2)] if bounced else None,
                       "bounce_frame": f0 + ib if bounced else None, "net_clear_ft": round(clear, 2) if clear is not None else None}
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        out = os.environ.get("LIFTER_OUT", "lifter.pt")
        minutes = float(os.environ.get("LIFTER_MINUTES", "60"))
        cams = json_cams = []
        cp = os.environ.get("LIFTER_CAMERAS")
        if cp and os.path.exists(cp):
            import json
            json_cams = [(np.array(c["K"]), np.array(c["R"]), np.array(c["t"])) for c in json.load(open(cp))]
        print("best val 3D error (ft):", train(out, minutes, real_cameras=json_cams))
