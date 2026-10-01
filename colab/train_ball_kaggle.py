# Kaggle (2x T4): ball detector v2 on 3 merged Roboflow datasets (~18k train images).
# Run from the notebook with ROBOFLOW_API_KEY in the environment (from Kaggle Secrets).
# Output: /kaggle/working/ball_detector_v2.pt (+ runs/ball_v2 with curves).
import io, os, pathlib, shutil, zipfile
import requests, yaml

KEY = os.environ["ROBOFLOW_API_KEY"]
DATA = pathlib.Path("/kaggle/temp/raw"); DATA.mkdir(parents=True, exist_ok=True)
API = "https://api.roboflow.com"
DATASETS = [  # name, workspace, project, kept classes (lower-case), max box size (fraction of image)
    ("liberin_pickleball_vision", "liberin-technologies", "pickleball-vision", {"ball"}, 1.0),
    ("uninu_pickleball", "pickleball-1uztf", "pickleball-uninu", {"pickleball"}, 1.0),
    ("pbd_pickleball_bal_detection", "pickleball-ball-detector", "pickleball-bal-detection",
     {"ball", "pickleball - v5 2024-09-06 7-41am"}, 0.06),
]
for name, ws, proj, _, _ in DATASETS:
    out = DATA / name
    if out.exists():
        continue
    info = requests.get(f"{API}/{ws}/{proj}", params={"api_key": KEY}, timeout=60).json()
    ver = max(int(v["id"].rsplit("/", 1)[-1]) for v in info["versions"])
    link = requests.get(f"{API}/{ws}/{proj}/{ver}/yolov8", params={"api_key": KEY}, timeout=900).json()["export"]["link"]
    z = requests.get(link, timeout=3600); z.raise_for_status()
    zipfile.ZipFile(io.BytesIO(z.content)).extractall(out)
    print(f"{name} v{ver}: {len(z.content) / 1e6:.0f} MB", flush=True)

MERGED = pathlib.Path("/kaggle/temp/merged"); shutil.rmtree(MERGED, ignore_errors=True)
n_img = n_ball = 0
for name, _, _, keep, max_box in DATASETS:
    root = DATA / name
    names = yaml.safe_load(next(root.rglob("data.yaml")).read_text())["names"]
    names = list(names.values()) if isinstance(names, dict) else names
    keep_ids = {i for i, n in enumerate(names) if n.lower() in keep}
    for src, dst in {"train": "train", "valid": "val", "test": "test"}.items():
        img_dir = root / src / "images"
        if not img_dir.exists():
            continue
        (MERGED / dst / "images").mkdir(parents=True, exist_ok=True)
        (MERGED / dst / "labels").mkdir(parents=True, exist_ok=True)
        for img in img_dir.iterdir():
            rows = []
            lbl = root / src / "labels" / f"{img.stem}.txt"
            if lbl.exists():
                for r in lbl.read_text().splitlines():
                    p = r.split()
                    if len(p) == 5 and int(p[0]) in keep_ids and max(float(p[3]), float(p[4])) <= max_box:
                        rows.append(" ".join(["0", *p[1:]]))
            stem = f"{name}__{img.stem}"
            os.symlink(img, MERGED / dst / "images" / f"{stem}{img.suffix}")
            (MERGED / dst / "labels" / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            n_img += 1
            n_ball += len(rows)
(MERGED / "data.yaml").write_text(yaml.safe_dump({"path": str(MERGED), "train": "train/images", "val": "val/images",
                                                  "test": "test/images", "names": {0: "ball"}}))
print("merged images", n_img, "balls", n_ball, flush=True)

from ultralytics import YOLO
import torch
devices = list(range(torch.cuda.device_count())) or "cpu"
YOLO("yolo11s.pt").train(
    data=str(MERGED / "data.yaml"), epochs=60, imgsz=960, batch=16 * max(1, len(devices) if isinstance(devices, list) else 1),
    device=devices, workers=4, patience=12,  # NOTE: ultralytics `time=` overrides `epochs` (trains for the full time) - keep epochs as the limit
    fliplr=0.5, flipud=0.0, scale=0.3, mosaic=1.0, close_mosaic=10,
    project="/kaggle/working/runs", name="ball_v2", exist_ok=True, plots=True)
shutil.copy("/kaggle/working/runs/ball_v2/weights/best.pt", "/kaggle/working/ball_detector_v2.pt")
import pandas as pd
df = pd.read_csv("/kaggle/working/runs/ball_v2/results.csv"); df.columns = df.columns.str.strip()
print(df.loc[df["metrics/mAP50(B)"].idxmax(), ["epoch", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)"]].to_string())
