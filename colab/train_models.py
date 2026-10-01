import subprocess
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True, capture_output=True, text=True).stdout)
subprocess.run("pip -q install ultralytics", shell=True, check=True)

from google.colab import drive, userdata
drive.mount('/content/drive')
import os, pathlib
OUT = pathlib.Path('/content/drive/MyDrive/pickleball-analyzer')
(OUT / 'models').mkdir(parents=True, exist_ok=True)
KEY = userdata.get('ROBOFLOW_API_KEY')
print('key loaded:', bool(KEY))

# Download the Roboflow datasets (YOLO format) to local Colab disk
import io, zipfile, requests
DATA = pathlib.Path('/content/data/raw'); DATA.mkdir(parents=True, exist_ok=True)
API = 'https://api.roboflow.com'
DATASETS = [
    ('liberin_pickleball_vision', 'liberin-technologies', 'pickleball-vision'),
    ('uninu_pickleball', 'pickleball-1uztf', 'pickleball-uninu'),
    ('necromancer_pickleball_court', 'necromancer', 'pickleball-court-vbmkq'),
]
for name, ws, proj in DATASETS:
    out = DATA / name
    if out.exists(): print('have', name); continue
    info = requests.get(f'{API}/{ws}/{proj}', params={'api_key': KEY}, timeout=60).json()
    ver = max(int(v['id'].rsplit('/', 1)[-1]) for v in info['versions'])
    link = requests.get(f'{API}/{ws}/{proj}/{ver}/yolov8', params={'api_key': KEY}, timeout=300).json()['export']['link']
    z = requests.get(link, timeout=1800); z.raise_for_status()
    zipfile.ZipFile(io.BytesIO(z.content)).extractall(out)
    print(f'{name} v{ver}: {len(z.content)/1e6:.0f} MB')

# Merge the two ball datasets into one class 'ball' (Court/Paddle/Player labels dropped)
import shutil, yaml
from collections import Counter
MERGED = pathlib.Path('/content/data/merged')
shutil.rmtree(MERGED, ignore_errors=True)
CLASS_MAP = {'liberin_pickleball_vision': {'ball'}, 'uninu_pickleball': {'pickleball'}}
SPLITS = {'train': 'train', 'valid': 'val', 'test': 'test'}
stats = Counter()
for ds, keep in CLASS_MAP.items():
    root = DATA / ds
    names = yaml.safe_load(next(root.rglob('data.yaml')).read_text())['names']
    names = list(names.values()) if isinstance(names, dict) else names
    keep_ids = {i for i, n in enumerate(names) if n.lower() in keep}
    for src, dst in SPLITS.items():
        img_dir = root / src / 'images'
        if not img_dir.exists(): continue
        (MERGED / dst / 'images').mkdir(parents=True, exist_ok=True); (MERGED / dst / 'labels').mkdir(parents=True, exist_ok=True)
        for img in img_dir.iterdir():
            stem = f'{ds}__{img.stem}'
            lbl = root / src / 'labels' / f'{img.stem}.txt'
            rows = []
            if lbl.exists():
                for r in lbl.read_text().splitlines():
                    p = r.split()
                    if len(p) == 5 and int(p[0]) in keep_ids:
                        rows.append(' '.join(['0', *p[1:]]))
            os.symlink(img, MERGED / dst / 'images' / f'{stem}{img.suffix}')
            (MERGED / dst / 'labels' / f'{stem}.txt').write_text('\n'.join(rows) + ('\n' if rows else ''))
            stats[f'{dst} images'] += 1; stats[f'{dst} balls'] += len(rows)
(MERGED / 'data.yaml').write_text(yaml.safe_dump({'path': str(MERGED), 'train': 'train/images', 'val': 'val/images',
                                                  'test': 'test/images', 'names': {0: 'ball'}}))
print(dict(stats))

# Court keypoints: our own data.yaml with the correct left/right flip mapping
COURT = DATA / 'necromancer_pickleball_court'
(COURT / 'court.yaml').write_text(yaml.safe_dump({'path': str(COURT), 'train': 'train/images', 'val': 'valid/images',
    'test': 'test/images', 'kpt_shape': [12, 3], 'flip_idx': [2, 1, 0, 9, 8, 7, 6, 5, 4, 3, 10, 11], 'names': {0: 'court'}}))

# 1) Court detector: YOLO11s-pose. Resumes automatically from Drive if interrupted.
from ultralytics import YOLO
last = OUT / 'runs/court/weights/last.pt'
if (OUT / 'models/court_detector.pt').exists():
    print('court model already trained')
elif last.exists():
    YOLO(str(last)).train(resume=True)
else:
    YOLO('yolo11s-pose.pt').train(data=str(COURT / 'court.yaml'), epochs=80, imgsz=640, batch=16, patience=20,
        fliplr=0.5, flipud=0.0, mosaic=0.5, degrees=3.0, perspective=0.0005,
        project=str(OUT / 'runs'), name='court', exist_ok=True, plots=True)
shutil.copy(OUT / 'runs/court/weights/best.pt', OUT / 'models/court_detector.pt')
print('saved', OUT / 'models/court_detector.pt')

# 2) Ball detector: YOLO11s at 960 px (small objects). Resumes automatically from Drive if interrupted.
last = OUT / 'runs/ball/weights/last.pt'
if (OUT / 'models/ball_detector.pt').exists():
    print('ball model already trained')
elif last.exists():
    YOLO(str(last)).train(resume=True)
else:
    YOLO('yolo11s.pt').train(data=str(MERGED / 'data.yaml'), epochs=80, imgsz=960, batch=16, patience=20,
        fliplr=0.5, flipud=0.0, scale=0.3, mosaic=1.0, close_mosaic=10,
        project=str(OUT / 'runs'), name='ball', exist_ok=True, plots=True)
shutil.copy(OUT / 'runs/ball/weights/best.pt', OUT / 'models/ball_detector.pt')
print('saved', OUT / 'models/ball_detector.pt')

# Final metrics
import pandas as pd
for run, cols in [('court', ['metrics/mAP50(P)', 'metrics/mAP50-95(P)']), ('ball', ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)'])]:
    df = pd.read_csv(OUT / f'runs/{run}/results.csv'); df.columns = df.columns.str.strip()
    print(run, 'best epoch:'); print(df.loc[df[cols[0]].idxmax(), ['epoch', *cols]].to_string(), '\n')