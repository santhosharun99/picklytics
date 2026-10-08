# Kaggle GPU: run the temporal ball detector (pickleball/balltrack_net.py, inlined by build_infer.py) over every
# rally of the matches in runs_infer.json - same output as training/balltrack_run.py (laptop): per run
# <run>_ball_v3.json {frame: [x, y, conf]} in source pixels, <run>_ball_v3_cands.json (top-3 peaks),
# <run>_ball_v3_main.json (main-camera score: fixed-camera court-line samples on white pixels).
# Inputs: santhosharun/pb-balltrack-infer (balltrack.pt, balltrack_metrics.json, runs_infer.json) + pbvid-* videos.
import glob, json, os, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

W, H, K, SIGMA = 768, 432, 3, 2.0


W, H, K, SIGMA = 768, 432, 3, 2.0


def _cbr(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Up(nn.Module):
    def __init__(self, i, skip, o):
        super().__init__()
        self.c = nn.Sequential(_cbr(i + skip, o), _cbr(o, o))

    def forward(self, x, s):
        x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
        return self.c(torch.cat([x, s], 1))


class BallTrackNet(nn.Module):
    def __init__(self, k: int = K, pretrained: bool = True):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        cin = 3 * (k + 1)
        conv1 = nn.Conv2d(cin, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            conv1.weight.copy_(r.conv1.weight.repeat(1, k + 1, 1, 1) / (k + 1))
        self.stem0 = nn.Sequential(_cbr(cin, 32), _cbr(32, 32))            # full resolution
        self.enc1 = nn.Sequential(conv1, r.bn1, r.relu)                     # 1/2, 64
        self.enc2 = nn.Sequential(r.maxpool, r.layer1)                      # 1/4, 64
        self.enc3, self.enc4, self.enc5 = r.layer2, r.layer3, r.layer4      # 1/8 128, 1/16 256, 1/32 512
        self.up4, self.up3 = Up(512, 256, 256), Up(256, 128, 128)
        self.up2, self.up1 = Up(128, 64, 64), Up(64, 64, 48)
        self.up0 = Up(48, 32, 32)
        self.head = nn.Conv2d(32, k, 1)
        nn.init.constant_(self.head.bias, -4.6)  # start at p ~ 0.01 (focal-loss practice)

    def forward(self, x):
        s0 = self.stem0(x)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        d = self.up4(e5, e4)
        d = self.up3(d, e3)
        d = self.up2(d, e2)
        d = self.up1(d, e1)
        d = self.up0(d, s0)
        return self.head(d)


def focal_loss(logits, target, mask, alpha: float = 2.0, beta: float = 4.0):
    """CenterNet penalty-reduced focal loss. target: Gaussian heat-maps (peak 1 at the ball), mask: (B, K) 1 where
    the frame is labelled (a 'no ball' frame has an all-zero target and still counts)."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = (target >= 0.999).float()
    neg = 1 - pos
    pos_l = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_l = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * neg
    m = mask[:, :, None, None]
    n_pos = (pos * m).sum().clamp(min=1.0)
    return ((pos_l + neg_l) * m).sum() / n_pos


def gaussian(x, y, w: int = W, h: int = H, sigma: float = SIGMA):
    yy = torch.arange(h, dtype=torch.float32)[:, None]
    xx = torch.arange(w, dtype=torch.float32)[None, :]
    g = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    g[int(round(min(max(y, 0), h - 1))), int(round(min(max(x, 0), w - 1)))] = 1.0
    return g


def peaks(hm: torch.Tensor, thr: float = 0.5):
    """Sub-pixel ball position from one heat-map (probabilities) or None: weighted centre of the blob around the max."""
    m = float(hm.max())
    if m < thr:
        return None, m
    i = int(hm.argmax())
    y, x = divmod(i, hm.shape[-1])
    y0, y1, x0, x1 = max(0, y - 3), y + 4, max(0, x - 3), x + 4
    patch = hm[y0:y1, x0:x1]
    yy, xx = torch.meshgrid(torch.arange(y0, y0 + patch.shape[0]), torch.arange(x0, x0 + patch.shape[1]), indexing="ij")
    w = patch.clamp(min=0)
    return (float((xx * w).sum() / w.sum()), float((yy * w).sum() / w.sum())), m

COURT_POINTS = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [20.0, 15.0], [20.0, 29.0], [20.0, 44.0], [10.0, 44.0], [0.0, 44.0], [0.0, 29.0], [0.0, 15.0], [10.0, 15.0], [10.0, 29.0]], dtype=np.float32)
def _line_samples(step_ft: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Points along every court line (court feet) and each line's direction."""
    segs = [((0, 0), (20, 0)), ((0, 44), (20, 44)), ((0, 0), (0, 44)), ((20, 0), (20, 44)),
            ((0, 15), (20, 15)), ((0, 29), (20, 29)), ((10, 0), (10, 15)), ((10, 29), (10, 44))]
    pts, dirs = [], []
    for (x1, y1), (x2, y2) in segs:
        n = int(np.hypot(x2 - x1, y2 - y1) / step_ft)
        for t in np.linspace(0.02, 0.98, n):
            pts.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
            dirs.append((x2 - x1, y2 - y1))
    return np.array(pts, np.float32), np.array(dirs, np.float32)

_SAMPLES, _DIRS = _line_samples()
def white_mask(frame) -> np.ndarray:
    """Thin bright, unsaturated structures (painted court lines).
    Two tests: plainly white pixels, and - for lines in sun shadows, where they are no longer bright - thin
    structures clearly brighter than their surroundings (top-hat on brightness). Las Vegas doubles, heavy
    shadows: line match 0.35 -> 0.64 on main-camera frames; other camera angles stay <= 0.47 (< 0.55 needed)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 70) & (hsv[..., 2] > 165)).astype(np.uint8) * 255
    # keep thin structures only: white minus its opening with a large kernel
    big = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    lit = cv2.subtract(white, big)
    v = hsv[..., 2]
    tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, np.ones((15, 15), np.uint8))
    shade = ((tophat > 15) & (hsv[..., 1] < 130) & (v > 60)).astype(np.uint8) * 255
    return cv2.max(lit, shade)

def _project(H_inv, pts):
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H_inv).reshape(-1, 2)

def snap_to_lines(frame, H_inv: np.ndarray, search_px: int = 12, mask=None):
    """Find, for each line sample, the nearest white pixel along the line normal.

    Returns (court_pts, image_pts, fraction_found, mean_offset_px)."""
    mask = white_mask(frame) if mask is None else mask
    h, w = mask.shape
    img = _project(H_inv, _SAMPLES)
    ahead = _project(H_inv, _SAMPLES + 0.01 * _DIRS)
    d = ahead - img
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    normals = np.stack([-d[:, 1], d[:, 0]], 1)
    court, found, offs = [], [], []
    visible = 0
    offsets = np.arange(-search_px, search_px + 1)
    for (cx, cy), p, nrm in zip(_SAMPLES, img, normals):
        if not (0 <= p[0] < w and 0 <= p[1] < h):
            continue
        visible += 1
        q = (p[None, :] + offsets[:, None] * nrm[None, :]).round().astype(int)
        ok = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
        if not ok.any():
            continue
        hits = np.where(mask[q[ok, 1], q[ok, 0]] > 0)[0]
        if len(hits) == 0:
            continue
        o = offsets[ok][hits]
        best = o[np.argmin(np.abs(o))]
        court.append((cx, cy))
        found.append(p + best * nrm)
        offs.append(abs(best))
    frac = len(found) / max(visible, 1)
    return (np.array(court, np.float32), np.array(found, np.float32), frac,
            float(np.mean(offs)) if offs else float(search_px))



def _find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return Path(hits[0]) if hits else None


INP = _find("runs_infer.json").parent
OUT = Path("/kaggle/working")
dev = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()
m = json.loads((INP / "balltrack_metrics.json").read_text())
if "input_size" in m:
    W, H = m["input_size"]
net = BallTrackNet(pretrained=False)
net.load_state_dict(torch.load(INP / "balltrack.pt", map_location="cpu"))
net.eval().to(dev)
THR, BS = 0.2, 16
runs = json.loads((INP / "runs_infer.json").read_text())
only = os.environ.get("BT_RUNS")
for rid, info in runs.items():
    if only and rid not in only.split(","):
        continue
    vid = _find(info["video"])
    if vid is None:
        print("no video for", rid, flush=True); continue
    sx, sy = info["size"][0] / W, info["size"][1] / H
    H_small = np.diag([1 / sx, 1 / sy, 1.0]) @ np.array(info["H_inv"]) if info.get("H_inv") else None
    cap = cv2.VideoCapture(str(vid))
    out, cands, main = {}, {}, {}
    for k, (f0, f1) in enumerate(info["ranges"]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0 - 1)
        imgs = {}
        for f in range(f0 - 1, f1 + 2):
            ok, img = cap.read()
            if not ok:
                break
            imgs[f] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        if len(imgs) < 3:
            continue
        bg = np.median(np.stack([imgs[f] for f in np.linspace(min(imgs), max(imgs), 15).astype(int) if f in imgs]), 0).astype(np.uint8)
        todo = [f for f in range(f0, f1 + 1) if all(g in imgs for g in (f - 1, f, f + 1))]
        for i in range(0, len(todo), BS):
            fs = todo[i:i + BS]
            x = np.stack([np.concatenate([imgs[f - 1], imgs[f], imgs[f + 1], bg], 2) for f in fs]).astype(np.float32) / 255.0
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=dev == "cuda"):
                hm = torch.sigmoid(net(torch.from_numpy(x.transpose(0, 3, 1, 2)).to(dev)).float()).cpu()[:, 1]
            for f, h in zip(fs, hm):
                p, conf = peaks(h, THR)
                if p:
                    out[f] = [round(p[0] * sx, 1), round(p[1] * sy, 1), round(conf, 3)]
                mx = F.max_pool2d(h[None, None], 7, 1, 3)[0, 0]
                ys, xs = torch.where((h == mx) & (h >= 0.05))
                vals = h[ys, xs]
                order = torch.argsort(vals, descending=True)[:3]
                cands[f] = [[round(float(xs[j]) * sx, 1), round(float(ys[j]) * sy, 1), round(float(vals[j]), 3)] for j in order]
                if H_small is not None:
                    main[f] = round(float(snap_to_lines(imgs[f], H_small, 3)[2]), 2)
        if k % 10 == 0:
            print(f"[{(time.time() - T0) / 60:5.1f} min] {rid} range {k + 1}/{len(info['ranges'])}", flush=True)
    for name, d in (("ball_v3", out), ("ball_v3_cands", cands), ("ball_v3_main", main)):
        (OUT / f"{rid}_{name}.json").write_text(json.dumps(d))
    print(f"[{(time.time() - T0) / 60:5.1f} min] {rid} done: {len(out)} ball frames", flush=True)
