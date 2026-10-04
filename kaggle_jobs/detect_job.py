"""Kaggle job: run the heavy per-frame detection (court lock, players + pose, far-player zoom, ball) on a GPU.

Pushed by training/kaggle_detect.py, which fills in STEM and attaches two private datasets:
  pickleball-code  (pickleball/ package, configs/cameras/, models: ball, court, yolo11m-pose)
  pbvid-<slug>     (the match video, <STEM>.mp4)
The video is split in two halves, one per T4 GPU, and the results are merged into
/kaggle/working/out/{detections.jsonl, meta.json}. Analysis (fast) runs locally after `fetch`.
"""
import glob, json, os, subprocess, sys, time
from pathlib import Path

STEM = "__STEM__"
T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)

try:
    import ultralytics  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3"], check=True)

code = Path(glob.glob("/kaggle/input/**/pickleball/detect.py", recursive=True)[0]).parents[1]
video = glob.glob(f"/kaggle/input/**/{STEM}.mp4", recursive=True) or glob.glob("/kaggle/input/**/*.mp4", recursive=True)
video = video[0]
log("code", code, "video", video)

import cv2  # noqa: E402
cap = cv2.VideoCapture(video)
fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
mid = n / 2 / fps
log(f"{n} frames at {fps:.2f} fps ({n / fps / 60:.1f} min); halves split at {mid:.0f} s")

work = Path("/kaggle/working")
cam = code / "configs" / "cameras" / f"{STEM}.json"
job = """
import json, sys
from pathlib import Path
sys.path.insert(0, {code!r})
from pickleball.detect import run_detection
cam = Path({cam!r})
run_detection({video!r}, Path({out!r}), {court!r}, {ball!r}, {pose!r}, {start}, {end},
              960, camera=json.loads(cam.read_text()) if cam.exists() else None, pose_imgsz=1280)
"""
procs = []
for gpu, (start, end) in enumerate([(0.0, mid), (mid, None)]):
    src = job.format(code=str(code), cam=str(cam), video=video, out=str(work / f"part{gpu}"),
                     court=str(code / "models" / "court_detector.pt"), ball=str(code / "models" / "ball_detector.pt"),
                     pose=str(code / "models" / "yolo11m-pose.pt"), start=start, end=end)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    procs.append(subprocess.Popen([sys.executable, "-c", src], env=env,
                                  stdout=open(work / f"part{gpu}.log", "w"), stderr=subprocess.STDOUT))
    log(f"started half {gpu} on GPU {gpu}")
while any(p.poll() is None for p in procs):
    time.sleep(120)
    for g in range(2):
        tail = (work / f"part{g}.log").read_text().splitlines()[-1:] if (work / f"part{g}.log").exists() else []
        log(f"half {g}:", tail[0][-90:] if tail else "")
codes = [p.returncode for p in procs]
log("exit codes", codes)
if any(codes):
    for g in range(2):
        print((work / f"part{g}.log").read_text()[-3000:])
    raise SystemExit("detection failed")

out = work / "out"
out.mkdir(exist_ok=True)
metas = [json.loads((work / f"part{g}" / "meta.json").read_text()) for g in range(2)]
with open(out / "detections.jsonl", "w") as fo:
    for g in range(2):
        with open(work / f"part{g}" / "detections.jsonl") as fi:
            for line in fi:
                fo.write(line)
meta = {**metas[0], "first_frame": metas[0]["first_frame"], "last_frame": metas[1]["last_frame"], "video": f"{STEM}.mp4",
        "kaggle_minutes": round((time.time() - T0) / 60, 1)}
(out / "meta.json").write_text(json.dumps(meta, indent=2))
for g in range(2):
    for f in (work / f"part{g}").iterdir():
        f.unlink()
    (work / f"part{g}").rmdir()
log("done:", meta)
