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

# @@NET@@

# @@COURT@@


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
