"""Build the Kaggle ball-detector INFERENCE kernel + its input dataset (laptop: training/balltrack_run.py, ~8 fps).

    PYTHONPATH=. python3 kaggle_jobs/balltrack/build_infer.py tnppl_8_TeamSingles full_match ...
-> data/balltrack_infer/{balltrack.pt, balltrack_metrics.json, runs_infer.json, dataset-metadata.json}
   (upload: kaggle datasets create -p data/balltrack_infer --dir-mode zip, or `version` after the first time)
-> kaggle_jobs/balltrack/kernel_infer/{balltrack_infer_kernel.py, kernel-metadata.json}  (kaggle kernels push -p ...)
Then: kaggle kernels output santhosharun/pb-balltrack-infer -p outputs/balltrack/infer --file-pattern '_ball_v3'
and move <run>_ball_v3*.json to outputs/<run>/ball_v3*.json.
"""
import inspect, json, re, shutil, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pickleball import court  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
_b = (Path(__file__).resolve().parent / "build.py").read_text()  # same video datasets as training (build.py runs on import)
exec(_b[_b.index("VIDEO_DS = "):_b.index("\nnet = ")])
HERE = Path(__file__).resolve().parent
DS = ROOT / "data" / "balltrack_infer"
DS.mkdir(parents=True, exist_ok=True)
runs = {}
for rid in sys.argv[1:]:
    if rid not in VIDEO_DS:
        print("no Kaggle video dataset for", rid, "- skipped"); continue
    run = ROOT / "outputs" / rid
    A = json.loads((run / "analysis.json").read_text())
    fps = A["meta"]["fps"]
    kps = [json.loads(l)["court"]["kp"] for l in (run / "detections.jsonl").open() if '"court":{' in l or '"court": {' in l]
    H_inv = None
    if len(kps) > 100:
        H_inv, _ = cv2.findHomography(court.COURT_POINTS, np.median(np.array(kps), 0).astype(np.float32))
    runs[rid] = {"video": PROXY.get(rid, Path(A["meta"]["video"]).name), "size": [A["meta"]["width"], A["meta"]["height"]],
                 "ranges": [[int((r["start_t"] - 0.5) * fps), int((r["end_t"] + 0.5) * fps)] for r in A["rallies"]],
                 "H_inv": H_inv.tolist() if H_inv is not None else None}
(DS / "runs_infer.json").write_text(json.dumps(runs))
shutil.copy(ROOT / "models" / "balltrack_v3.pt", DS / "balltrack.pt")
shutil.copy(ROOT / "models" / "balltrack_metrics.json", DS / "balltrack_metrics.json")
(DS / "dataset-metadata.json").write_text(json.dumps({"title": "pb-balltrack-infer", "id": "santhosharun/pb-balltrack-infer",
                                                      "licenses": [{"name": "CC0-1.0"}]}))
net = (ROOT / "pickleball" / "balltrack_net.py").read_text()
net = re.sub(r'^""".*?"""\n', "", net, flags=re.S)
net = "\n".join(l for l in net.splitlines() if not l.startswith(("import torch", "import torch.nn")))
court_src = "\n".join([
    "COURT_POINTS = np.array(" + json.dumps(court.COURT_POINTS.tolist()) + ", dtype=np.float32)",
    inspect.getsource(court._line_samples), "_SAMPLES, _DIRS = _line_samples()",
    inspect.getsource(court.white_mask), inspect.getsource(court._project), inspect.getsource(court.snap_to_lines)])
src = (HERE / "kernel_infer_src.py").read_text().replace("# @@NET@@", net).replace("# @@COURT@@", court_src)
kd = HERE / "kernel_infer"
kd.mkdir(exist_ok=True)
(kd / "balltrack_infer_kernel.py").write_text(src)
(kd / "kernel-metadata.json").write_text(json.dumps({
    "id": "santhosharun/pb-balltrack-infer-run", "title": "pb-balltrack-infer-run", "code_file": "balltrack_infer_kernel.py",
    "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": ["santhosharun/pb-balltrack-infer"] + [f"santhosharun/{VIDEO_DS[r]}" for r in runs],
    "machine_shape": "NvidiaTeslaT4"}, indent=1))
print("runs:", list(runs), "| frames:", sum(b - a for r in runs.values() for a, b in r["ranges"]))
