"""Run the heavy detection for a match on Kaggle's GPUs instead of the Mac.

    PYTHONPATH=. python3 training/kaggle_detect.py push  data/videos/<Video>.mp4      # upload + start (returns at once)
    PYTHONPATH=. python3 training/kaggle_detect.py status data/videos/<Video>.mp4     # queued / running / complete
    PYTHONPATH=. python3 training/kaggle_detect.py fetch  data/videos/<Video>.mp4 --players 4 [--out outputs/<run>]
                                                                                       # download + local analysis

Needs ~/.kaggle/access_token (Kaggle CLI). Uploads two PRIVATE datasets: santhosharun/pickleball-code (code,
camera configs, ball/court/pose models; re-uploaded only when they change) and santhosharun/pbvid-<slug> (the
video). A camera config (configs/cameras/<stem>.json) must exist before `push` for courts the court model does
not detect. Kaggle job: kaggle_jobs/detect_job.py (two halves in parallel on 2x T4).
"""
import argparse, hashlib, json, re, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "santhosharun"
STAGE = ROOT / ".kaggle_stage"          # gitignored staging folders
CODE_DS = f"{USER}/pickleball-code"
MODELS = ["ball_detector.pt", "court_detector.pt"]


def kaggle(*args, check=True) -> str:
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def stem_of(video: Path) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", video.stem).strip("_")[:60]


def slug_of(stem: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")[:40]


def wait_ready(ds: str, minutes: int = 60):
    for _ in range(minutes * 4):
        s = kaggle("datasets", "status", ds, check=False).strip().lower()
        if "ready" in s:
            return
        time.sleep(15)
    raise SystemExit(f"dataset {ds} not ready after {minutes} min")


def push_code():
    d = STAGE / "code"
    shutil.rmtree(d, ignore_errors=True)
    (d / "pickleball").mkdir(parents=True)
    (d / "models").mkdir()
    (d / "configs" / "cameras").mkdir(parents=True)
    for f in (ROOT / "pickleball").glob("*.py"):
        shutil.copy2(f, d / "pickleball" / f.name)
    for f in (ROOT / "configs" / "cameras").glob("*.json"):
        shutil.copy2(f, d / "configs" / "cameras" / f.name)
    for m in MODELS:
        shutil.copy2(ROOT / "models" / m, d / "models" / m)
    shutil.copy2(ROOT / "yolo11m-pose.pt", d / "models" / "yolo11m-pose.pt")
    h = hashlib.sha256()
    for f in sorted(p for p in d.rglob("*") if p.is_file()):
        h.update(f.relative_to(d).as_posix().encode()); h.update(f.read_bytes())
    digest = h.hexdigest()
    mark = STAGE / "code.sha256"
    if mark.exists() and mark.read_text() == digest:
        print("code dataset unchanged")
        return
    (d / "dataset-metadata.json").write_text(json.dumps({"title": "pickleball-code", "id": CODE_DS,
                                                           "licenses": [{"name": "other"}]}))
    exists = "ready" in kaggle("datasets", "status", CODE_DS, check=False).lower()
    print(("updating" if exists else "creating"), CODE_DS)
    if exists:
        kaggle("datasets", "version", "-p", str(d), "-m", "code/models update", "-r", "zip")
    else:
        kaggle("datasets", "create", "-p", str(d), "-r", "zip")
    wait_ready(CODE_DS)
    mark.write_text(digest)


def push_video(video: Path, stem: str) -> str:
    ds = f"{USER}/pbvid-{slug_of(stem)}"
    if "ready" in kaggle("datasets", "status", ds, check=False).lower():
        print("video dataset already on Kaggle:", ds)
        return ds
    d = STAGE / f"video-{slug_of(stem)}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / f"{stem}.mp4").symlink_to(video.resolve())
    (d / "dataset-metadata.json").write_text(json.dumps({"title": f"pbvid-{slug_of(stem)}"[:50], "id": ds,
                                                           "licenses": [{"name": "other"}]}))
    size = video.resolve().stat().st_size / 1e9
    print(f"uploading video ({size:.1f} GB) to private dataset {ds} - this can take a while ...")
    kaggle("datasets", "create", "-p", str(d))
    wait_ready(ds, minutes=120)
    return ds


def kernel_id(stem: str) -> str:
    return f"{USER}/pb-detect-{slug_of(stem)}"


def push(video: Path):
    stem = stem_of(video)
    if not (ROOT / "configs" / "cameras" / f"{stem}.json").exists():
        print(f"note: no configs/cameras/{stem}.json - relying on the court model (fine for courts it detects)")
    push_code()
    vds = push_video(video, stem)
    k = STAGE / f"kernel-{slug_of(stem)}"
    shutil.rmtree(k, ignore_errors=True)
    k.mkdir(parents=True)
    (k / "detect_job.py").write_text((ROOT / "kaggle_jobs" / "detect_job.py").read_text().replace("__STEM__", stem))
    kid = kernel_id(stem)
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": kid.split("/")[1], "code_file": "detect_job.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [CODE_DS, vds], "machine_shape": "NvidiaTeslaT4"}))
    print(kaggle("kernels", "push", "-p", str(k)))
    print(f"started. check with: PYTHONPATH=. python3 training/kaggle_detect.py status {video}")


def status(video: Path):
    print(kaggle("kernels", "status", kernel_id(stem_of(video)), check=False))


def fetch(video: Path, players: int, out: Path | None, names: str):
    stem = stem_of(video)
    out = out or ROOT / "outputs" / stem
    tmp = STAGE / f"fetch-{slug_of(stem)}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    print(kaggle("kernels", "output", kernel_id(stem), "-p", str(tmp)))
    src = next(tmp.rglob("detections.jsonl"), None)
    if src is None:
        raise SystemExit("no detections in the job output yet - is the job complete? (status)")
    out.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), out / "detections.jsonl")
    meta = json.loads(next(tmp.rglob("meta.json")).read_text())
    meta["video"] = str(video)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"detections -> {out} (Kaggle took {meta.get('kaggle_minutes')} min). Running the analysis ...")
    from pickleball.analysis import run_analysis
    res = run_analysis(out, players, dict(kv.split("=", 1) for kv in names.split(",") if "=" in kv))
    print(json.dumps(res["summary"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "status", "fetch"])
    ap.add_argument("video")
    ap.add_argument("--players", type=int, default=2, choices=[2, 4])
    ap.add_argument("--names", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    video = Path(a.video)
    if a.cmd == "push":
        push(video)
    elif a.cmd == "status":
        status(video)
    else:
        fetch(video, a.players, Path(a.out) if a.out else None, a.names)


if __name__ == "__main__":
    sys.exit(main())
