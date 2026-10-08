"""Run the temporal ball detector on the ground-truth clips -> predictions for training/eval_balltruth.py.

    PYTHONPATH=. python3 training/balltrack_infer.py models/balltrack.pt outputs/balltrack/pred_v1.json [--thr 0.5]
For each clip frame f: frames f-1, f, f+1 + the median background of the clip (15 frames), 768x432 -> heat-map of f
-> ball position (1280x720 pixels) or none. One clip at a time (laptop heat).
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np
import torch

from pickleball.balltrack_net import BallTrackNet, H, W, peaks

ROOT = Path(__file__).resolve().parents[1]
PROXY = {"lv_fahey_yoshitomi": "data/videos/Fahey_Yoshitomi_LV2026.mp4", "full_match": "data/videos/Mens_Singles_GOLD_2026_US_Open.mp4"}
ap = argparse.ArgumentParser()
ap.add_argument("weights")
ap.add_argument("out")
ap.add_argument("--all-frames", action="store_true", help="every frame of the clip (for tracking), not only the labelled ones")
ap.add_argument("--thr", type=float, default=0.05, help="keep peaks above this; eval_balltruth --thr picks the operating point")
a = ap.parse_args()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
mfile = Path(a.weights).with_name("balltrack_metrics.json")
if mfile.exists() and "input_size" in json.loads(mfile.read_text()):
    W, H = json.loads(mfile.read_text())["input_size"]  # the resolution the model was trained at
net = BallTrackNet(pretrained=False)
net.load_state_dict(torch.load(a.weights, map_location="cpu"))
net.eval().to(dev)
Q = json.loads((ROOT / "labels" / "balltruth" / "queue.json").read_text())["clips"]
out, cands = {}, {}
for c in Q:
    A = json.loads((ROOT / "outputs" / c["run"] / "analysis.json").read_text())
    video = ROOT / PROXY.get(c["run"], A["meta"]["video"])
    fs = [f["frame"] for f in c["frames"]]
    f0, f1 = min(fs) - 1, max(fs) + 1
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    imgs = {}
    for f in range(f0, f1 + 1):
        ok, img = cap.read()
        if not ok:
            break
        imgs[f] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    bg = np.median(np.stack([imgs[f] for f in np.linspace(f0, f1, 15).astype(int) if f in imgs]), 0).astype(np.uint8)
    sx, sy = c["size"][0] / W, c["size"][1] / H
    pred = {}
    run_fs = list(range(min(fs), max(fs) + 1)) if a.all_frames else fs
    with torch.no_grad():
        for f in run_fs:
            if not all(g in imgs for g in (f - 1, f, f + 1)):
                pred[f] = None
                continue
            x = np.concatenate([imgs[f - 1], imgs[f], imgs[f + 1], bg], 2).astype(np.float32) / 255.0
            hm = torch.sigmoid(net(torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(dev))).cpu()[0, 1]
            p, conf = peaks(hm, a.thr)
            pred[f] = [round(p[0] * sx, 1), round(p[1] * sy, 1), round(conf, 3)] if p else None  # x, y, confidence
            # top-3 local maxima (for tracking over candidates): non-maximum suppression with a 7x7 window
            mx = torch.nn.functional.max_pool2d(hm[None, None], 7, 1, 3)[0, 0]
            ys, xs = torch.where((hm == mx) & (hm >= a.thr))
            vals = hm[ys, xs]
            order = torch.argsort(vals, descending=True)[:3]
            cands.setdefault(c["clip"], {})[f] = [[round(float(xs[i]) * sx, 1), round(float(ys[i]) * sy, 1), round(float(vals[i]), 3)] for i in order]
    out[c["clip"]] = pred
    print(c["clip"], sum(v is not None for v in pred.values()), "/", len(pred), "frames with a ball", flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(out))
Path(a.out).with_name(Path(a.out).stem + "_cands.json").write_text(json.dumps(cands))
