# Kaggle GPU: train the temporal heat-map ball detector (pickleball/balltrack_net.py, inlined below by build.py).
# Inputs: santhosharun/pb-balltrack-labels (runs.json + <run>.csv physics-verified labels) and the match videos
# (santhosharun/pbvid-*). Frames are decoded here (nothing big is uploaded from the laptop).
# Env: BT_HOURS (training time, default 6), BT_TEST=1 (smoke test).
# Outputs: balltrack.pt (best val F1), balltrack_metrics.json, train_log.txt
import csv, glob, json, math, os, random, time, zlib
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# @@NET@@

W = int(os.environ.get("BT_W", W)); H = int(os.environ.get("BT_H", H))  # input resolution (default 768x432)

def _find_labels():
    if os.environ.get("BT_LABELS"):
        return Path(os.environ["BT_LABELS"])
    hits = glob.glob("/kaggle/input/**/runs.json", recursive=True)  # Kaggle's mount layout differs between images
    return Path(hits[0]).parent if hits else Path("/kaggle/input/pb-balltrack-labels")


LAB = _find_labels()
OUT = Path(os.environ.get("BT_OUT", "/kaggle/working")); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("BT_CACHE", "/tmp/bt_cache"))  # not /kaggle/working: 20 GB output cap, huge output listing
HOURS = float(os.environ.get("BT_HOURS", "6"))
TEST = os.environ.get("BT_TEST") == "1"
WORKERS = int(os.environ.get("BT_WORKERS", "4"))
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
T0 = time.time()
LOG = (OUT / "train_log.txt").open("w")


def log(*a):
    s = f"[{(time.time() - T0) / 60:6.1f} min] " + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def find_video(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) + glob.glob(os.environ.get("BT_VIDEOS", "/nonexistent") + f"/{name}")
    return hits[0] if hits else None


def prepare():
    runs = json.loads((LAB / "runs.json").read_text())
    samples = []
    for rid, info in runs.items():
        vid = find_video(info["video"])
        if vid is None:
            log("no video for", rid, "- skipped"); continue
        sx, sy = W / info["size"][0], H / info["size"][1]
        lab = {}
        for r in csv.DictReader((LAB / f"{rid}.csv").open()):
            lab[int(r["frame"])] = (float(r["x"]) * sx, float(r["y"]) * sy, int(r["rally"]))
        frames = sorted(lab)
        if TEST:
            frames = frames[:300]
        need = set()
        for f in frames:
            need.update(range(f - 1, f + 2))
        rally_of = {}
        bg_need = {}
        for idx, f0, f1 in info["rallies"]:
            if any(lab.get(f, (0, 0, -1))[2] == idx for f in range(f0, f1 + 1)):
                pts = np.linspace(f0, f1, 15).astype(int).tolist()
                bg_need[idx] = pts
                need.update(pts)
                for f in range(f0, f1 + 1):
                    rally_of[f] = idx
        d = CACHE / rid; d.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(vid)
        f, last = 0, max(need)
        bg_imgs = {k: [] for k in bg_need}
        want_bg = {}
        for k, pts in bg_need.items():
            for p in pts:
                want_bg.setdefault(p, []).append(k)
        while f <= last and cap.grab():
            if f in need:
                ok, img = cap.retrieve()
                if ok:
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(d / f"{f}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    for k in want_bg.get(f, []):
                        bg_imgs[k].append(img)
            f += 1
        for k, ims in bg_imgs.items():
            if ims:
                cv2.imwrite(str(d / f"bg_{k}.jpg"), np.median(np.stack(ims), 0).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
        n = 0
        for f in frames:
            k = rally_of.get(f)
            if k is None or not (d / f"bg_{k}.jpg").exists() or not all((d / f"{g}.jpg").exists() for g in (f - 1, f, f + 1)):
                continue
            tg = [lab.get(g) if lab.get(g, (0, 0, -1))[2] == k else None for g in (f - 1, f, f + 1)]
            samples.append({"run": rid, "f": f, "rally": k, "t": [None if t is None else t[:2] for t in tg]})
            n += 1
        log(f"{rid}: {n} samples from {len(frames)} labelled frames")
    return samples


class DS(torch.utils.data.Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s = self.items[i]
        d = CACHE / s["run"]
        ims = [cv2.imread(str(d / f"{g}.jpg")) for g in (s["f"] - 1, s["f"], s["f"] + 1)] + [cv2.imread(str(d / f"bg_{s['rally']}.jpg"))]
        x = np.concatenate(ims, 2).astype(np.float32) / 255.0
        pts = [None if t is None else list(t) for t in s["t"]]
        if self.train:
            if random.random() < 0.5:
                x = x[:, ::-1].copy()
                pts = [None if p is None else [W - 1 - p[0], p[1]] for p in pts]
            sc = random.uniform(0.9, 1.1); tx, ty = random.uniform(-24, 24), random.uniform(-16, 16)
            M = np.float32([[sc, 0, (1 - sc) * W / 2 + tx], [0, sc, (1 - sc) * H / 2 + ty]])
            x = cv2.warpAffine(x, M, (W, H), borderMode=cv2.BORDER_REFLECT)
            pts = [None if p is None else [sc * p[0] + (1 - sc) * W / 2 + tx, sc * p[1] + (1 - sc) * H / 2 + ty] for p in pts]
            x = np.clip(x * random.uniform(0.8, 1.2) + random.uniform(-0.06, 0.06), 0, 1)
        tgt = torch.zeros(K, H, W)
        mask = torch.zeros(K)
        for j, p in enumerate(pts):
            if p is not None and 0 <= p[0] < W and 0 <= p[1] < H:
                tgt[j] = gaussian(p[0], p[1], W, H); mask[j] = 1
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), tgt, mask


def evaluate(net, items, px=None):
    px = px or 4.0 * W / 1280  # 4 px at 1280x720
    net.eval(); tp = fp = fn = 0; errs = []
    dl = torch.utils.data.DataLoader(DS(items, False), batch_size=8, num_workers=WORKERS)
    with torch.no_grad():
        for x, tgt, mask in dl:
            with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
                p = torch.sigmoid(net(x.to(dev)).float()).cpu()
            for b in range(len(x)):
                pr, _ = peaks(p[b, 1])
                has = mask[b, 1] > 0
                if has:
                    gy, gx = divmod(int(tgt[b, 1].argmax()), W)
                    if pr is not None and math.hypot(pr[0] - gx, pr[1] - gy) <= px:
                        tp += 1; errs.append(math.hypot(pr[0] - gx, pr[1] - gy))
                    else:
                        fn += 1; fp += pr is not None
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9), "median_err_px": float(np.median(errs)) if errs else None}


def main():
    samples = prepare()
    rng = random.Random(0)
    val = [s for s in samples if (zlib.crc32(f"{s['run']}:{s['rally']}".encode()) % 10) == 0]
    tr = [s for s in samples if (zlib.crc32(f"{s['run']}:{s['rally']}".encode()) % 10) != 0]
    log(f"train {len(tr)} / val {len(val)} samples")
    net = BallTrackNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    dl = torch.utils.data.DataLoader(DS(tr, True), batch_size=int(os.environ.get("BT_BS", "6")), shuffle=True, num_workers=WORKERS, drop_last=True, persistent_workers=WORKERS > 0)
    budget = (0.05 if TEST else HOURS) * 3600
    t0, step, best = time.time(), 0, -1.0
    hist = []
    while time.time() - t0 < budget:
        net.train()
        for x, tgt, mask in dl:
            frac = min(1.0, (time.time() - t0) / budget)
            lr = 5e-4 * min(1.0, step / 500) * 0.5 * (1 + math.cos(math.pi * frac))
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
                out = net(x.to(dev))
            loss = focal_loss(out, tgt.to(dev), mask.to(dev))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); step += 1
            if step % 200 == 0:
                log(f"step {step} loss {loss.item():.4f} lr {lr:.2e}")
            if time.time() - t0 >= budget:
                break
        m = evaluate(net, val)
        hist.append({"step": step, "min": round((time.time() - t0) / 60, 1), **m})
        log("val", m)
        if m["f1"] > best:
            best = m["f1"]; torch.save(net.state_dict(), OUT / "balltrack.pt")
    (OUT / "balltrack_metrics.json").write_text(json.dumps({"best_val_f1": best, "history": hist, "n_train": len(tr), "n_val": len(val),
                                                            "input_size": [W, H]}, indent=1))
    log("done; best val F1", best)


main()
