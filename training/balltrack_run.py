"""Run the temporal ball detector over frame ranges of a match -> ball track for the physics layer.

    PYTHONPATH=. python3 training/balltrack_run.py outputs/full_match models/balltrack_v3.pt outputs/full_match/ball_v3.json \
        [--ranges labels/Mens_Singles_GOLD_2026_US_Open.json] [--thr 0.2]
Ranges: the tagged rallies of a labels file, or (default) every rally of the run +-0.5 s. Per range: median background
of 15 frames, then every frame (needs f-1, f, f+1). Output {frame: [x, y, conf]} in SOURCE pixels (analysis
resolution) + top-3 candidates in <out>_cands.json + main-camera score per frame in <out>_main.json (share of the
fixed camera's court-line samples on white pixels: main view ~0.7-0.8, cut-aways / replays from other cameras mostly
< 0.5; use >= 0.6). One match at a time (laptop heat).
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np
import torch

from pickleball import balltrack_net as BN
from pickleball.court import COURT_POINTS, snap_to_lines

ROOT = Path(__file__).resolve().parents[1]
PROXY = {"lv_fahey_yoshitomi": "data/videos/Fahey_Yoshitomi_LV2026.mp4", "full_match": "data/videos/Mens_Singles_GOLD_2026_US_Open.mp4"}
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("weights")
ap.add_argument("out")
ap.add_argument("--ranges")
ap.add_argument("--thr", type=float, default=0.2)
ap.add_argument("--rallies", type=int, nargs="*", help="only these rally indices of the run (+-0.5 s); merged into an existing --out")
a = ap.parse_args()
run = Path(a.run)
A = json.loads((run / "analysis.json").read_text())
fps = A["meta"]["fps"]
W, H = BN.W, BN.H
mfile = Path(a.weights).with_name("balltrack_metrics.json")
if mfile.exists() and "input_size" in json.loads(mfile.read_text()):
    W, H = json.loads(mfile.read_text())["input_size"]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
net = BN.BallTrackNet(pretrained=False)
net.load_state_dict(torch.load(a.weights, map_location="cpu"))
net.eval().to(dev)
if a.ranges:
    L = json.loads(Path(a.ranges).read_text())
    ranges = [(int(r["start"]), int(r["end"])) for r in L["rallies"] if r.get("mode") == "hits+bounces" and r["end"] > r["start"]]
else:
    ranges = [(int((r["start_t"] - 0.5) * fps), int((r["end_t"] + 0.5) * fps)) for r in A["rallies"]
              if a.rallies is None or r["index"] in a.rallies]
video = ROOT / PROXY.get(run.name, A["meta"]["video"])
cap = cv2.VideoCapture(str(video))
sx, sy = A["meta"]["width"] / W, A["meta"]["height"] / H
kps = [json.loads(l)["court"]["kp"] for l in (run / "detections.jsonl").open() if '"court":{' in l or '"court": {' in l] \
    if (run / "detections.jsonl").exists() else []
H_small = None
if len(kps) > 100:  # fixed camera: median court -> image homography, scaled to the network input size
    H_inv, _ = cv2.findHomography(np.asarray(COURT_POINTS, np.float32), np.median(np.array(kps), 0).astype(np.float32))
    H_small = np.diag([1 / sx, 1 / sy, 1.0]) @ H_inv
out, cands, main = {}, {}, {}
if a.rallies is not None and Path(a.out).exists():  # partial run: add to what is there
    out = {int(k): v for k, v in json.loads(Path(a.out).read_text()).items()}
    for d_, suf in ((cands, "_cands.json"), (main, "_main.json")):
        f_ = Path(a.out).with_name(Path(a.out).stem + suf)
        if f_.exists():
            d_.update({int(k): v for k, v in json.loads(f_.read_text()).items()})
for k, (f0, f1) in enumerate(sorted(ranges)):
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
    with torch.no_grad():
        for f in range(f0, f1 + 1):
            if not all(g in imgs for g in (f - 1, f, f + 1)):
                continue
            if H_small is not None:
                main[f] = round(float(snap_to_lines(imgs[f], H_small, 3)[2]), 2)
            x = np.concatenate([imgs[f - 1], imgs[f], imgs[f + 1], bg], 2).astype(np.float32) / 255.0
            hm = torch.sigmoid(net(torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(dev))).cpu()[0, 1]
            p, conf = BN.peaks(hm, a.thr)
            if p:
                out[f] = [round(p[0] * sx, 1), round(p[1] * sy, 1), round(conf, 3)]
            mx = torch.nn.functional.max_pool2d(hm[None, None], 7, 1, 3)[0, 0]
            ys, xs = torch.where((hm == mx) & (hm >= 0.05))
            vals = hm[ys, xs]
            order = torch.argsort(vals, descending=True)[:3]
            cands[f] = [[round(float(xs[i]) * sx, 1), round(float(ys[i]) * sy, 1), round(float(vals[i]), 3)] for i in order]
    print(f"range {k + 1}/{len(ranges)} frames {f0}-{f1}: ball in {sum(1 for f in range(f0, f1 + 1) if f in out)}", flush=True)
Path(a.out).write_text(json.dumps(out))
Path(a.out).with_name(Path(a.out).stem + "_cands.json").write_text(json.dumps(cands))
if main:
    Path(a.out).with_name(Path(a.out).stem + "_main.json").write_text(json.dumps(main))
