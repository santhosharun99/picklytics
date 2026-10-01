# Colab (T4): ball detector v2 = fine-tune the v1 ball model on 3 merged Roboflow datasets
# (~18k train images, adds the 16.5k-image pickleball-bal-detection set incl. PPA broadcast frames).
# Resumable: checkpoints in MyDrive/pickleball-analyzer/runs/ball_v2; re-run to continue.
import io, os, pathlib, shutil, subprocess, zipfile
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True, capture_output=True, text=True).stdout)
subprocess.run("pip -q install ultralytics", shell=True, check=True)
import requests, yaml
from google.colab import drive, userdata
drive.mount('/content/drive')
OUT = pathlib.Path('/content/drive/MyDrive/pickleball-analyzer')
KEY = userdata.get('ROBOFLOW_API_KEY')
DATA = pathlib.Path('/content/data/raw'); DATA.mkdir(parents=True, exist_ok=True)
API = 'https://api.roboflow.com'
# name, workspace, project, kept classes (lower-case), max box size (fraction of image)
DATASETS = [
    ('liberin_pickleball_vision', 'liberin-technologies', 'pickleball-vision', {'ball'}, 1.0),
    ('uninu_pickleball', 'pickleball-1uztf', 'pickleball-uninu', {'pickleball'}, 1.0),
    ('pbd_pickleball_bal_detection', 'pickleball-ball-detector', 'pickleball-bal-detection',
     {'ball', 'pickleball - v5 2024-09-06 7-41am'}, 0.06),
]
for name, ws, proj, _, _ in DATASETS:
    out = DATA / name
    if out.exists(): print('have', name); continue
    info = requests.get(f'{API}/{ws}/{proj}', params={'api_key': KEY}, timeout=60).json()
    ver = max(int(v['id'].rsplit('/', 1)[-1]) for v in info['versions'])
    link = requests.get(f'{API}/{ws}/{proj}/{ver}/yolov8', params={'api_key': KEY}, timeout=600).json()['export']['link']
    z = requests.get(link, timeout=3600); z.raise_for_status()
    zipfile.ZipFile(io.BytesIO(z.content)).extractall(out)
    print(f'{name} v{ver}: {len(z.content)/1e6:.0f} MB')

MERGED = pathlib.Path('/content/data/merged_v2'); shutil.rmtree(MERGED, ignore_errors=True)
n_img = n_ball = 0
for name, _, _, keep, max_box in DATASETS:
    root = DATA / name
    names = yaml.safe_load(next(root.rglob('data.yaml')).read_text())['names']
    names = list(names.values()) if isinstance(names, dict) else names
    keep_ids = {i for i, n in enumerate(names) if n.lower() in keep}
    for src, dst in {'train': 'train', 'valid': 'val', 'test': 'test'}.items():
        img_dir = root / src / 'images'
        if not img_dir.exists(): continue
        (MERGED / dst / 'images').mkdir(parents=True, exist_ok=True); (MERGED / dst / 'labels').mkdir(parents=True, exist_ok=True)
        for img in img_dir.iterdir():
            rows = []
            lbl = root / src / 'labels' / f'{img.stem}.txt'
            if lbl.exists():
                for r in lbl.read_text().splitlines():
                    p = r.split()
                    if len(p) == 5 and int(p[0]) in keep_ids and max(float(p[3]), float(p[4])) <= max_box:
                        rows.append(' '.join(['0', *p[1:]]))
            stem = f'{name}__{img.stem}'
            os.symlink(img, MERGED / dst / 'images' / f'{stem}{img.suffix}')
            (MERGED / dst / 'labels' / f'{stem}.txt').write_text('\n'.join(rows) + ('\n' if rows else ''))
            n_img += 1; n_ball += len(rows)
(MERGED / 'data.yaml').write_text(yaml.safe_dump({'path': str(MERGED), 'train': 'train/images', 'val': 'val/images',
                                                  'test': 'test/images', 'names': {0: 'ball'}}))
print('merged images', n_img, 'balls', n_ball)

from ultralytics import YOLO
last = OUT / 'runs/ball_v2/weights/last.pt'
if (OUT / 'models/ball_detector_v2.pt').exists():
    print('ball v2 already trained')
elif last.exists():
    YOLO(str(last)).train(resume=True)
else:
    start = OUT / 'runs/ball/weights/best.pt'   # v1 ball model as the starting point
    YOLO(str(start) if start.exists() else 'yolo11s.pt').train(
        data=str(MERGED / 'data.yaml'), epochs=60, imgsz=960, batch=16, patience=15,
        fliplr=0.5, flipud=0.0, scale=0.3, mosaic=1.0, close_mosaic=10,
        project=str(OUT / 'runs'), name='ball_v2', exist_ok=True, plots=True)
shutil.copy(OUT / 'runs/ball_v2/weights/best.pt', OUT / 'models/ball_detector_v2.pt')
print('saved', OUT / 'models/ball_detector_v2.pt')
import pandas as pd
df = pd.read_csv(OUT / 'runs/ball_v2/results.csv'); df.columns = df.columns.str.strip()
print(df.loc[df['metrics/mAP50(B)'].idxmax(), ['epoch', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)']].to_string())
