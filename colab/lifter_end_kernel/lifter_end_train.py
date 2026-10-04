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


def _bounce_on(p, v, beta, t_left, rng, max_bounces=3):
    """Continue a ball from p (on the ground) with velocity v, bouncing up to max_bounces times, for t_left s."""
    out, t_used = [], 0.0
    for _ in range(max_bounces):
        e, mu = rng.uniform(0.5, 0.8), rng.uniform(0.6, 0.9)
        v = np.array([v[0] * mu, v[1] * mu, abs(v[2]) * e])
        # time to come down again (no drag on the vertical for this estimate)
        tf = max(2 * v[2] / G, 1 / FPS)
        T = min(tf, t_left - t_used)
        if T <= 0:
            break
        ts = np.arange(1 / FPS, T + 1e-9, 1 / FPS)
        if not len(ts):
            break
        seg = _fly(p, v, beta, ts)
        seg[:, 2] = np.maximum(seg[:, 2], 0.0)
        out.append(seg)
        t_used += T
        p = np.array([seg[-1, 0], seg[-1, 1], 0.0])
        v = _vel(v, beta, T)
        if t_used >= t_left or v[2] > -1.0:   # rolling: stop simulating
            break
    return np.vstack(out) if out else np.zeros((0, 3))


def simulate_end_shot(rng):
    """A rally's LAST shot: no next hit. The ball goes into the net, long, wide, or is a winner that bounces on and
    away; the track ends 0.2-1.5 s after the decisive moment (or when it leaves the picture). Returns (positions, -1)."""
    near = rng.random() < 0.5
    hy = rng.uniform(30, 50) if near else rng.uniform(-6, 14)
    p0 = np.array([rng.uniform(-2, 22), hy, rng.uniform(0.6, 7.5)])
    kind = rng.choice(["net", "long", "wide", "winner"], p=[0.3, 0.25, 0.15, 0.3])
    beta = rng.uniform(0.2, 1.3)
    sgn = -1 if near else 1                                # direction of travel along y
    tail = rng.uniform(0.2, 1.5)
    if kind == "net":
        T1 = rng.uniform(0.25, 0.9)
        pn = np.array([rng.uniform(0, 20), 22.0, rng.uniform(0.3, 2.7)])     # hits the net below the tape
        v0 = _aim(p0, pn, T1, beta)
        A = _fly(p0, v0, beta, np.arange(0, T1, 1 / FPS))
        vb = _vel(v0, beta, T1)
        v = np.array([vb[0] * 0.3, -vb[1] * rng.uniform(0.02, 0.15), min(vb[2], 0.0) * 0.2])  # dies in the net
        ts = np.arange(1 / FPS, 3.0, 1 / FPS)
        B = _fly(pn + np.array([0, -sgn * 0.3, 0]), v, beta, ts)
        hit = np.where(B[:, 2] <= 0)[0]
        B = B[: hit[0] + 1] if len(hit) else B
        B[:, 2] = np.maximum(B[:, 2], 0.0)
        C = _bounce_on(np.array([B[-1, 0], B[-1, 1], 0.0]), _vel(v, beta, len(B) / FPS) * [1, 1, -1], beta, tail, rng, 2)
        return np.vstack([A, B, C]) if len(C) else np.vstack([A, B]), -1
    if kind == "long":
        ly = (rng.uniform(-6, -0.3) if near else rng.uniform(44.3, 50))
        lx = rng.uniform(-1, 21)
    elif kind == "wide":
        ly = rng.uniform(0, 21) if near else rng.uniform(23, 44)
        lx = rng.uniform(-6, -0.3) if rng.random() < 0.5 else rng.uniform(20.3, 26)
    else:   # winner: lands in, nobody gets it
        ly = rng.uniform(0, 21.5) if near else rng.uniform(22.5, 44)
        lx = rng.uniform(-0.5, 20.5)
    T1 = rng.uniform(1.3, 2.3) if rng.random() < 0.15 else rng.uniform(0.35, 1.1)   # some lobs
    p1 = np.array([lx, ly, 0.0])
    v0 = _aim(p0, p1, T1, beta)
    A = _fly(p0, v0, beta, np.arange(0, T1, 1 / FPS))
    C = _bounce_on(p1, _vel(v0, beta, T1), beta, tail, rng, 3)
    return (np.vstack([A, C]) if len(C) else A), -1


def make_sample(rng, camera=None):
    """Synthetic (inputs (MAX_T, 9), target (MAX_T, 3), mask (MAX_T,)) like real tracking of one shot."""
    while True:
        X, bi = simulate_end_shot(rng)   # END-OF-RALLY model: only last shots
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



# ---------------- Kaggle training run ----------------
REAL_CAMS = [{"K": [[2660.6000000000013, 0.0, 960.0], [0.0, 2660.6000000000013, 540.0], [0.0, 0.0, 1.0]], "R": [[0.9995125147485258, 0.030189780336356716, -0.00795675966210267], [-0.0004584625672833624, 0.2690204306663479, 0.9631343611854823], [0.031217345709594446, -0.9626611995127545, 0.26890312805825156]], "t": [-10.920202738013465, -7.273018299215378, 79.4386078054547]}, {"K": [[1861.200000000001, 0.0, 960.0], [0.0, 1861.200000000001, 540.0], [0.0, 0.0, 1.0]], "R": [[0.9999791376078349, 0.005970319033143306, 0.0024656925464746193], [-0.004058057125909332, 0.2836674241705262, 0.9589141383027062], [0.005025586677603939, -0.9589041389811086, 0.28368573408517506]], "t": [-10.274179702088675, -8.396251990144243, 69.30718848409425]}]
N_SAMPLES = int(os.environ.get("LIFTER_N", "300000"))
TRAIN_MIN = float(os.environ.get("LIFTER_MINUTES", "60"))
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."


def _chunk(args):
    seed, n = args
    rng = np.random.default_rng(seed)
    cams = [(np.array(c["K"]), np.array(c["R"]), np.array(c["t"])) for c in REAL_CAMS]
    xs, ys, ms = [], [], []
    for _ in range(n):
        cam = cams[rng.integers(len(cams))] if rng.random() < 0.3 else None
        x, y, m = make_sample(rng, cam)
        xs.append(x); ys.append(y); ms.append(m)
    return np.stack(xs), np.stack(ys), np.stack(ms)


if __name__ == "__main__":
    import multiprocessing as mp, torch, json as _json
    t0 = time.time()
    workers = max(1, mp.cpu_count())
    per = 5000
    with mp.Pool(workers) as pool:
        parts = pool.map(_chunk, [(1000 + i, per) for i in range(N_SAMPLES // per)])
    X = np.concatenate([p[0] for p in parts]); Y = np.concatenate([p[1] for p in parts]); M = np.concatenate([p[2] for p in parts])
    print(f"generated {len(X)} simulated shots with {workers} workers in {(time.time() - t0) / 60:.1f} min", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nval = 5000
    Xv, Yv, Mv = (torch.tensor(a[:nval]).to(dev) for a in (X, Y, M))
    Xt, Yt, Mt = (torch.tensor(a[nval:]) for a in (X, Y, M))
    net = build_net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    t1, step, best = time.time(), 0, 1e9
    budget = TRAIN_MIN * 60
    while time.time() - t1 < budget:
        idx = torch.randint(0, len(Xt), (512,))
        x, y, m = Xt[idx].to(dev), Yt[idx].to(dev), Mt[idx].to(dev)
        p = net(x)
        err = ((p - y) ** 2).sum(-1)
        vel = ((p[:, 1:] - p[:, :-1]) - (y[:, 1:] - y[:, :-1])) ** 2
        loss = (err * m).sum() / m.sum() + 20 * (vel.sum(-1) * m[:, 1:]).sum() / m[:, 1:].sum()
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
        for g in opt.param_groups:
            g["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * min(1.0, (time.time() - t1) / budget)))
        if step % 500 == 0:
            with torch.no_grad():
                pv = torch.cat([net(Xv[i:i + 1000]) for i in range(0, nval, 1000)])
                d = ((pv - Yv) ** 2).sum(-1).sqrt() * SCALE
                e = (d * Mv).sum() / Mv.sum()
                # launch speed error from the first 3 frames
                v_p = (pv[:, 2] - pv[:, 0]) * SCALE * FPS / 2 * 0.6818
                v_y = (Yv[:, 2] - Yv[:, 0]) * SCALE * FPS / 2 * 0.6818
                se = (v_p.norm(dim=-1) - v_y.norm(dim=-1)).abs().median()
            print(f"[{(time.time() - t1) / 60:5.1f} min] step {step} loss {loss.item():.4f}  val 3D error {e.item():.2f} ft, launch speed error {se.item():.1f} mph", flush=True)
            if e.item() < best:
                best = e.item(); torch.save(net.state_dict(), f"{OUT}/lifter_end.pt")
    _json.dump({"best_val_3d_err_ft": best, "steps": step, "samples": len(X)}, open(f"{OUT}/lifter_end_summary.json", "w"))
    print("DONE best val 3D error (ft):", best, flush=True)
