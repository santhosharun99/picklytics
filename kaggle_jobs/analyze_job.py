"""Kaggle job: run only the ANALYSIS (hit / bounce models, paddle checks on the GPU, 3D paths, shot types) for a match
whose detections already exist. Pushed by `training/kaggle_detect.py analyze <video>`, which attaches:
  pickleball-code  (code + models), pbvid-<slug> (the video), pbrun-<slug> (detections.jsonl.gz, meta.json,
  paddles.json so far). Outputs analysis.json, render_data.json, paddles.json, lift.json in /kaggle/working/out.
"""
import glob, gzip, json, shutil, subprocess, sys, time
from pathlib import Path

STEM = "__STEM__"
PLAYERS = int("__PLAYERS__")
T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
try:
    import ultralytics  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3"], check=True)

code = Path(glob.glob("/kaggle/input/**/pickleball/detect.py", recursive=True)[0]).parents[1]
video = (glob.glob(f"/kaggle/input/**/{STEM}.mp4", recursive=True) or glob.glob("/kaggle/input/**/*.mp4", recursive=True))[0]
# Kaggle unpacks .gz files in datasets, so accept the detections either compressed or already unpacked
det = (glob.glob("/kaggle/input/**/detections.jsonl.gz", recursive=True) + glob.glob("/kaggle/input/**/detections.jsonl", recursive=True))[0]
run_in = Path(det).parent
out = Path("/kaggle/working/out")
out.mkdir(parents=True, exist_ok=True)
with (gzip.open(det, "rb") if det.endswith(".gz") else open(det, "rb")) as fi, open(out / "detections.jsonl", "wb") as fo:
    shutil.copyfileobj(fi, fo)
meta = json.loads((run_in / "meta.json").read_text())
meta["video"] = video
(out / "meta.json").write_text(json.dumps(meta, indent=2))
if (run_in / "paddles.json").exists():
    shutil.copy(run_in / "paddles.json", out / "paddles.json")    # paddle checks already done locally are reused
log("inputs ready:", video)

sys.path.insert(0, str(code))
from pickleball.analysis import run_analysis  # noqa: E402
res = run_analysis(out, PLAYERS)
log("analysis done:", res["summary"]["rallies"], "rallies,", res["summary"]["shots"], "shots")
(out / "detections.jsonl").unlink()   # already on the Mac; keep the output small
