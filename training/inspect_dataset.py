"""Quick quality check of a downloaded YOLO dataset: classes, counts, box sizes, sample grid."""
import glob, sys, collections, random
import cv2, numpy as np, yaml
from pathlib import Path
root = Path(sys.argv[1]); out = sys.argv[2]
names = yaml.safe_load(next(root.rglob("data.yaml")).read_text())["names"]
names = list(names.values()) if isinstance(names, dict) else names
cnt, sizes, poly = collections.Counter(), collections.defaultdict(list), 0
files = glob.glob(str(root / "*/labels/*.txt"))
for f in files:
    for r in open(f).read().split("\n"):
        p = r.split()
        if len(p) < 5: continue
        c = int(p[0]); v = list(map(float, p[1:]))
        if len(v) == 4: w, h = v[2], v[3]
        else: poly += 1; xs, ys = v[0::2], v[1::2]; w, h = max(xs)-min(xs), max(ys)-min(ys)
        cnt[names[c]] += 1; sizes[names[c]].append(max(w, h))
imgs = glob.glob(str(root / "*/images/*"))
print(f"{root.name}: images {len(imgs)}, label files {len(files)}, polygon rows {poly}")
for k, v in cnt.most_common():
    s = np.array(sizes[k]); print(f"  {k!r:40s} n={v:6d}  box max-side median {np.median(s):.3f}  p95 {np.percentile(s,95):.3f}  >0.06: {np.mean(s>0.06):.0%}")
random.seed(0); tiles = []
for f in random.sample(files, min(8, len(files))):
    im = cv2.imread(f.replace("/labels/", "/images/").rsplit(".", 1)[0] + ".jpg")
    if im is None: continue
    H, W = im.shape[:2]
    for r in open(f).read().split("\n"):
        p = r.split()
        if len(p) < 5: continue
        v = list(map(float, p[1:]))
        if len(v) == 4: x, y, w, h = v
        else: xs, ys = v[0::2], v[1::2]; x, y, w, h = (max(xs)+min(xs))/2, (max(ys)+min(ys))/2, max(xs)-min(xs), max(ys)-min(ys)
        x1, y1, x2, y2 = int((x-w/2)*W)-8, int((y-h/2)*H)-8, int((x+w/2)*W)+8, int((y+h/2)*H)+8
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 3); cv2.putText(im, names[int(p[0])][:10], (x1, y1-4), 0, 0.9, (0, 255, 255), 2)
    tiles.append(cv2.resize(im, (480, 270)))
while len(tiles) < 8: tiles.append(np.zeros((270, 480, 3), np.uint8))
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i+4]) for i in (0, 4)]))
