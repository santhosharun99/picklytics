# Kaggle (2x T4): v3 specialist models, trained one after the other in a single run.
#   1) paddle detector  (new)      - ~1.25 h
#   2) ball detector v3 (more data) - ~6.5 h
# ROBOFLOW_API_KEY must be in the environment (Kaggle Secrets). Outputs in /kaggle/working:
#   paddle_detector.pt, ball_detector_v3.pt, v3_summary.json, runs/<model>/...
# NOTE: ultralytics `time=` (hours) OVERRIDES `epochs`: it trains until the time is used up and
# shapes the learning-rate schedule to that budget. Used here on purpose as a hard end time.
import io, json, os, pathlib, shutil, time, traceback, zipfile
import requests, yaml

KEY = os.environ.get("ROBOFLOW_API_KEY", "")
API = "https://api.roboflow.com"
TEST = os.environ.get("V3_TEST") == "1"  # local dry run: tiny time budget, local data, same code path
RAW = pathlib.Path(os.environ.get("V3_RAW", "/kaggle/temp/raw")); RAW.mkdir(parents=True, exist_ok=True)
OUT = pathlib.Path(os.environ.get("V3_OUT", "/kaggle/working")); OUT.mkdir(parents=True, exist_ok=True)
TMP = pathlib.Path(os.environ.get("V3_TMP", "/kaggle/temp"))
T0 = time.time()

# (local name, workspace, project, classes to keep (lower-case), max box size as fraction of image)
BALL = [
    ("liberin_pickleball_vision", "liberin-technologies", "pickleball-vision", {"ball"}, 0.06),
    ("uninu_pickleball", "pickleball-1uztf", "pickleball-uninu", {"pickleball"}, 0.06),
    ("pbd_pickleball_bal_detection", "pickleball-ball-detector", "pickleball-bal-detection",
     {"ball", "pickleball - v5 2024-09-06 7-41am"}, 0.06),
# new for v3 (checked on 2026-10-01: PPA/MLP/other broadcast frames, labels on the ball)
    ("pbb_annotations_pickleball", "pickleballbadminton", "annotations_pickleball", {"ball"}, 0.12),
    ("duong_pb_ball_core", "duong-0nbek", "pb_ball_core", {"ball"}, 0.06),
    ("pbds_bigannotation", "pickleball-dataset", "bigannotationds-rxifp", {"pickleball"}, 0.08),
    ("pbbd_detector", "pickleball-ball-detection", "pickleball-detector", {"pickleball"}, 0.06),
    ("pbtraining_ball_paddle", "pbtrainingset", "pickleball-object-detection", {"ball"}, 0.08),
    # still downloading / retrying at the time of writing - included if the export works on Kaggle
    ("trns_06_09_lan1", "trns-workspace-pwwqf", "06-09-lan1", {"ball"}, 0.06),
    ("pkball", "pk-ball", "pk_ball", {"ball"}, 0.06),
    ("swz_pickleball", "s-workspace-zbpen", "pickleball-7kgpr", {"ball", "pickleball"}, 0.06),
    ("od_pickleball_pbcbd", "object-detection-sb2zh", "pickleball-pbcbd", {"ball"}, 0.06),
]
PADDLE = [
    ("od8al_version2_seg", "pickleball-od8al", "pickleball-version2", {"t1_paddle", "t2_paddle"}, 0.15),
    ("pbtraining_ball_paddle", "pbtrainingset", "pickleball-object-detection", {"paddle"}, 0.25),
]
MODELS = [  # name, datasets, hours
    ("paddle", PADDLE, 1.25),
    ("ball_v3", BALL, 6.5),
]


def log(*a):
    print(f"[{(time.time() - T0) / 3600:5.2f} h]", *a, flush=True)


def download(name, ws, proj):
    out = RAW / name
    if out.exists():
        return out
    if TEST:
        raise RuntimeError("not downloaded locally (test mode)")
    info = requests.get(f"{API}/{ws}/{proj}", params={"api_key": KEY}, timeout=60).json()
    ver = max(int(v["id"].rsplit("/", 1)[-1]) for v in info["versions"])
    for attempt in range(3):
        try:
            link = None
            for _ in range(80):  # export may still be being prepared: poll up to ~20 min
                j = requests.get(f"{API}/{ws}/{proj}/{ver}/yolov8", params={"api_key": KEY}, timeout=1800).json()
                if "export" in j and j["export"].get("link"):
                    link = j["export"]["link"]
                    break
                time.sleep(15)
            if link is None:
                raise RuntimeError("export not ready after 20 min")
            z = requests.get(link, timeout=3600); z.raise_for_status()
            zipfile.ZipFile(io.BytesIO(z.content)).extractall(out)
            log(f"downloaded {name} v{ver}: {len(z.content) / 1e6:.0f} MB")
            return out
        except Exception as e:
            log(f"download {name} attempt {attempt + 1} failed: {e}")
            time.sleep(30)
    raise RuntimeError(f"could not download {name}")


def row_to_box(parts):
    """YOLO row -> (cls, xc, yc, w, h). Polygon rows (segmentation exports) become their bounding box."""
    c = int(parts[0]); v = list(map(float, parts[1:]))
    if len(v) == 4:
        return c, *v
    xs, ys = v[0::2], v[1::2]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def merge(model_name, datasets):
    merged = TMP / f"merged_{model_name}"; shutil.rmtree(merged, ignore_errors=True)
    stats = {}
    for name, ws, proj, keep, max_box in datasets:
        try:
            root = download(name, ws, proj)
        except Exception as e:
            log(f"SKIP {name}: {e}")
            stats[name] = "download failed"
            continue
        names = yaml.safe_load(next(root.rglob("data.yaml")).read_text())["names"]
        names = list(names.values()) if isinstance(names, dict) else names
        keep_ids = {i for i, n in enumerate(names) if str(n).lower() in keep}
        n_img = n_obj = n_drop = 0
        for src, dst in {"train": "train", "valid": "val", "test": "test"}.items():
            img_dir = root / src / "images"
            if not img_dir.exists():
                continue
            (merged / dst / "images").mkdir(parents=True, exist_ok=True)
            (merged / dst / "labels").mkdir(parents=True, exist_ok=True)
            for img in img_dir.iterdir():
                rows = []
                lbl = root / src / "labels" / f"{img.stem}.txt"
                if lbl.exists():
                    for r in lbl.read_text().splitlines():
                        p = r.split()
                        if len(p) < 5 or int(p[0]) not in keep_ids:
                            continue
                        c, xc, yc, w, h = row_to_box(p)
                        if max(w, h) > max_box or w <= 0 or h <= 0:
                            n_drop += 1
                            continue
                        rows.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                stem = f"{name}__{img.stem}"
                os.symlink(img, merged / dst / "images" / f"{stem}{img.suffix}")
                (merged / dst / "labels" / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
                n_img += 1
                n_obj += len(rows)
        stats[name] = {"images": n_img, "objects": n_obj, "dropped_boxes": n_drop}
        log(f"{model_name}: {name} images={n_img} objects={n_obj} dropped={n_drop}")
    cls = "paddle" if model_name == "paddle" else "ball"
    (merged / "data.yaml").write_text(yaml.safe_dump({"path": str(merged), "train": "train/images", "val": "val/images",
                                                      "names": {0: cls}}))
    return merged, stats


summary = {}
from ultralytics import YOLO
import torch
devices = list(range(torch.cuda.device_count())) or ("mps" if torch.backends.mps.is_available() else "cpu")
ngpu = len(devices) if isinstance(devices, list) else 1
for model_name, datasets, hours in MODELS:
    try:
        merged, stats = merge(model_name, datasets)
        log(f"training {model_name} for {hours} h")
        YOLO("yolo11s.pt").train(
            data=str(merged / "data.yaml"), epochs=300, time=(0.01 if TEST else hours), imgsz=(320 if TEST else 960),
            batch=(8 if TEST else 16 * ngpu), fraction=(0.01 if TEST else 1.0),
            device=devices, workers=(0 if TEST else 4), patience=15, fliplr=0.5, flipud=0.0, scale=0.3, mosaic=1.0, close_mosaic=5,
            project=str(OUT / "runs"), name=model_name, exist_ok=True, plots=True)
        best = OUT / "runs" / model_name / "weights" / "best.pt"
        dst = OUT / ("paddle_detector.pt" if model_name == "paddle" else "ball_detector_v3.pt")
        shutil.copy(best, dst)
        import pandas as pd
        df = pd.read_csv(OUT / "runs" / model_name / "results.csv"); df.columns = df.columns.str.strip()
        row = df.loc[df["metrics/mAP50(B)"].idxmax()]
        summary[model_name] = {"data": stats, "epochs_run": int(df["epoch"].max()), "best_epoch": int(row["epoch"]),
                               "precision": float(row["metrics/precision(B)"]), "recall": float(row["metrics/recall(B)"]),
                               "mAP50": float(row["metrics/mAP50(B)"]), "weights": dst.name}
        log(f"DONE {model_name}: {summary[model_name]}")
    except Exception:
        summary[model_name] = {"error": traceback.format_exc()}
        log(f"FAILED {model_name}:\n{summary[model_name]['error']}")
    (OUT / "v3_summary.json").write_text(json.dumps(summary, indent=2))  # written after every model
log("ALL DONE", json.dumps({k: {kk: v.get(kk) for kk in ("mAP50", "precision", "recall", "error")} for k, v in summary.items()}))
