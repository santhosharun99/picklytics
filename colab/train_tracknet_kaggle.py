# Kaggle (GPU): TrackNet-style ball gap filler, our own compact implementation of the published idea
# (3 consecutive frames in -> ball heat-map out). Labels come from our own pipeline (YOLO ball + trusted
# 3D arcs), see training/tracknet_data.py. Two folds: train on one match, predict the other.
#
# Data: a Kaggle dataset with <match>/frames/<frame>.jpg and <match>/labels.csv (frame,rally,x,y,src).
# Env: TN_DATA (dataset root), TN_OUT (outputs), TN_HOURS (training time per fold), TN_TEST=1 (tiny smoke test).
# Outputs in TN_OUT: tracknet_<train>.pt, pred_<test>.csv (frame, x, y, score in 512x288 px), tracknet_summary.json
import csv, json, math, os, random, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path(os.environ.get("TN_DATA", "/kaggle/input/pickleball-tracknet"))
OUT = Path(os.environ.get("TN_OUT", "/kaggle/working")); OUT.mkdir(parents=True, exist_ok=True)
HOURS = float(os.environ.get("TN_HOURS", "1.5"))
TEST = os.environ.get("TN_TEST") == "1"
W, H, SIGMA = 512, 288, 2.5
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def find_match(name):
    for p in [DATA / name, *DATA.glob(f"*/{name}")]:
        if (p / "labels.csv").exists():
            return p
    raise FileNotFoundError(name)


def load_labels(root):
    rows = list(csv.DictReader((root / "labels.csv").open()))
    lab = {}
    for r in rows:
        f = int(r["frame"])
        lab[f] = (float(r["x"]), float(r["y"]), r["src"]) if r["src"] != "none" else None
    return lab


class Frames(torch.utils.data.Dataset):
    """Samples = frames whose middle frame has a label and whose neighbours exist."""

    def __init__(self, root, frames, lab, train):
        self.root, self.lab, self.train = root, lab, train
        have = set(lab)
        self.items = [f for f in frames if lab.get(f) and f - 1 in have and f + 1 in have]

    def __len__(self):
        return len(self.items)

    def img(self, f):
        im = cv2.imread(str(self.root / "frames" / f"{f}.jpg"))
        return im if im is not None else np.zeros((H, W, 3), np.uint8)

    def __getitem__(self, i):
        f = self.items[i]
        x = np.concatenate([self.img(g) for g in (f - 1, f, f + 1)], 2).astype(np.float32) / 255.0
        bx, by, _ = self.lab[f]
        if self.train:
            if random.random() < 0.5:  # mirror
                x = x[:, ::-1].copy(); bx = W - 1 - bx
            x = np.clip(x * random.uniform(0.8, 1.2) + random.uniform(-0.08, 0.08), 0, 1)
        yy, xx = np.mgrid[0:H, 0:W]
        hm = np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2 * SIGMA ** 2)).astype(np.float32)
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(hm[None])


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class BallNet(nn.Module):
    """Small U-Net: 9 channels (3 RGB frames) -> 1 heat-map channel (logits)."""

    def __init__(self, c=32):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4 = block(9, c), block(c, 2 * c), block(2 * c, 4 * c), block(4 * c, 8 * c)
        self.d3, self.d2, self.d1 = block(12 * c, 4 * c), block(6 * c, 2 * c), block(3 * c, c)
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(F.max_pool2d(e1, 2)); e3 = self.e3(F.max_pool2d(e2, 2)); e4 = self.e4(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([F.interpolate(e4, scale_factor=2), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))
        return self.head(d1)


def loss_fn(logits, hm):
    # heat-map focal-style weighting: the few ball pixels matter far more than the background
    p = torch.sigmoid(logits)
    w = 1 + 20 * hm
    return F.binary_cross_entropy_with_logits(logits, hm, weight=w) + 0.5 * ((p - hm) ** 2 * w).mean()


def peak(hm, thr=0.5):
    m = float(hm.max())
    if m < thr:
        return None, m
    yy, xx = np.where(hm >= m * 0.6)
    w = hm[yy, xx]
    return (float((xx * w).sum() / w.sum()), float((yy * w).sum() / w.sum())), m


def run_fold(train_name, test_name):
    tr_root, te_root = find_match(train_name), find_match(test_name)
    lab = load_labels(tr_root)
    rallies = {}
    for r in csv.DictReader((tr_root / "labels.csv").open()):
        rallies.setdefault(r["rally"], []).append(int(r["frame"]))
    keys = sorted(rallies, key=int)
    random.Random(0).shuffle(keys)
    val_keys = set(keys[: max(1, len(keys) // 8)])  # held-out rallies of the training match
    tr_frames = [f for k in keys if k not in val_keys for f in rallies[k]]
    va_frames = [f for k in val_keys for f in rallies[k]]
    if TEST:
        tr_frames, va_frames = tr_frames[:200], va_frames[:60]
    dtr, dva = Frames(tr_root, tr_frames, lab, True), Frames(tr_root, va_frames, lab, False)
    log(f"fold train={train_name}: {len(dtr)} train / {len(dva)} val samples")
    ltr = torch.utils.data.DataLoader(dtr, batch_size=8 if dev == "cuda" else 2, shuffle=True, num_workers=4 if dev == "cuda" else 0, drop_last=True)
    net = BallNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    budget = (0.02 if TEST else HOURS) * 3600
    t_start, step, best = time.time(), 0, (1e9, None)
    while time.time() - t_start < budget:
        net.train()
        for x, hm in ltr:
            x, hm = x.to(dev), hm.to(dev)
            loss = loss_fn(net(x), hm)
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            for g in opt.param_groups:  # cosine decay over the time budget
                g["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * min(1.0, (time.time() - t_start) / budget)))
            if step % 200 == 0:
                log(f"  step {step} loss {loss.item():.4f}")
            if time.time() - t_start >= budget:
                break
        # validation: detection rate and pixel error on held-out rallies
        net.eval(); errs, found = [], 0
        with torch.no_grad():
            for i in range(0, len(dva), 16):
                xs = torch.stack([dva[j][0] for j in range(i, min(i + 16, len(dva)))]).to(dev)
                hms = torch.sigmoid(net(xs)).cpu().numpy()[:, 0]
                for k, h in enumerate(hms):
                    bx, by, _ = dva.lab[dva.items[i + k]]
                    p, _ = peak(h)
                    if p is not None:
                        found += 1; errs.append(math.hypot(p[0] - bx, p[1] - by))
        score = (np.median(errs) if errs else 99) - 10 * found / max(len(dva), 1)
        log(f"  val: found {found}/{len(dva)}, median error {np.median(errs) if errs else float('nan'):.1f}px")
        if score < best[0]:
            best = (score, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    torch.save(net.state_dict(), OUT / f"tracknet_{train_name}.pt")
    # predict EVERY frame of the other match
    te_lab = load_labels(te_root)
    frames = sorted(te_lab)
    if TEST:
        frames = frames[:120]
    have = set(frames)
    rows, net = [], net.eval()
    ds = Frames(te_root, [], te_lab, False)
    with torch.no_grad():
        batch = []
        for f in frames:
            if f - 1 in have and f + 1 in have:
                batch.append(f)
            if len(batch) == 16 or (f == frames[-1] and batch):
                xs = torch.stack([torch.from_numpy((np.concatenate([ds.img(g) for g in (b - 1, b, b + 1)], 2)
                                                   .astype(np.float32) / 255.0).transpose(2, 0, 1).copy()) for b in batch]).to(dev)
                hms = torch.sigmoid(net(xs)).cpu().numpy()[:, 0]
                for b, h in zip(batch, hms):
                    p, m = peak(h, thr=0.3)
                    rows.append([b, round(p[0], 2) if p else "", round(p[1], 2) if p else "", round(m, 3)])
                batch = []
    with (OUT / f"pred_{test_name}.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame", "x", "y", "score"]); w.writerows(rows)
    # quick check on the test match: agreement with YOLO where YOLO saw the ball, and gap filling vs trusted arcs
    res = {}
    for src in ("det", "arc"):
        e, n, hit = [], 0, 0
        pred = {r[0]: r for r in rows}
        for f, l in te_lab.items():
            if l and l[2] == src and f in pred:
                n += 1
                if pred[f][1] != "" and pred[f][3] >= 0.5:
                    hit += 1; e.append(math.hypot(pred[f][1] - l[0], pred[f][2] - l[1]))
        res[src] = {"frames": n, "found": hit, "median_err_px": float(np.median(e)) if e else None}
    gaps = sum(1 for f, l in te_lab.items() if l is None)
    filled = sum(1 for r in rows if te_lab.get(r[0]) is None and r[1] != "" and r[3] >= 0.5)
    res["no_label_frames"] = gaps
    res["filled_no_label_frames"] = filled
    log(f"fold {train_name}->{test_name}: {json.dumps(res)}")
    return res


summary = {}
for tr, te in (("lasvegas", "usopen"), ("usopen", "lasvegas")):
    try:
        summary[f"{tr}->{te}"] = run_fold(tr, te)
    except Exception as ex:
        import traceback
        summary[f"{tr}->{te}"] = {"error": traceback.format_exc()}
        log("FAILED", tr, te, ex)
    (OUT / "tracknet_summary.json").write_text(json.dumps(summary, indent=1))
log("ALL DONE", json.dumps(summary))
