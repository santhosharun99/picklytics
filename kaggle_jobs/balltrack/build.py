"""Build the Kaggle kernel + labels dataset for the ball detector.

    PYTHONPATH=. python3 kaggle_jobs/balltrack/build.py
-> kaggle_jobs/balltrack/kernel/{balltrack_kernel.py, kernel-metadata.json}  (net inlined from pickleball/balltrack_net.py)
-> data/balltrack_labels/{runs.json, dataset-metadata.json}  (upload: kaggle datasets create/version -p data/balltrack_labels)
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LAB = ROOT / "data" / "balltrack_labels"
VIDEO_DS = {"tnppl_1_MS": "pbvid-tnppl-s2-ktvrtt-1-ms", "tnppl_2_WD": "pbvid-tnppl-s2-ktvrtt-2-wd",
            "tnppl_3_MD": "pbvid-tnppl-s2-ktvrtt-3-md", "tnppl_4_XD": "pbvid-tnppl-s2-ktvrtt-4-xd",
            "tnppl_5_MD50": "pbvid-tnppl-s2-ktvrtt-5-md50", "tnppl_7_TeamDoubles": "pbvid-tnppl-s2-ktvrtt-7-teamdoubles",
            "tnppl_8_TeamSingles": "pbvid-tnppl-s2-ktvrtt-8-teamsingles", "lv_bright_patriquin": "pbvid-bright-patriquin-fahey-staksrud-lv2026",
            "lv_fahey_yoshitomi": "pbvid-fahey-yoshitomi-lv2026", "full_match": "pbvid-mens-singles-gold-2026-us-open"}
PROXY = {"lv_fahey_yoshitomi": "Fahey_Yoshitomi_LV2026.mp4", "full_match": "Mens_Singles_GOLD_2026_US_Open.mp4"}  # same timing

net = (ROOT / "pickleball" / "balltrack_net.py").read_text()
net = re.sub(r'^""".*?"""\n', "", net, flags=re.S)
net = "\n".join(l for l in net.splitlines() if not l.startswith(("import torch", "import torch.nn")))
src = (HERE / "kernel_src.py").read_text().replace("# @@NET@@", net)
kd = HERE / "kernel"; kd.mkdir(exist_ok=True)
(kd / "balltrack_kernel.py").write_text(src)
runs = {}
import sys
ONLY = set(sys.argv[1:])  # optional: restrict to runs whose video dataset is ready
for rid in VIDEO_DS:
    if ONLY and rid not in ONLY:
        continue
    run = ROOT / "outputs" / rid
    if not (LAB / f"{rid}.csv").exists():
        continue
    A = json.loads((run / "analysis.json").read_text())
    fps = A["meta"]["fps"]
    runs[rid] = {"video": PROXY.get(rid, Path(A["meta"]["video"]).name), "size": [A["meta"]["width"], A["meta"]["height"]], "fps": fps,
                 "rallies": [[r["index"], int((r["start_t"] - 0.5) * fps), int((r["end_t"] + 0.5) * fps)] for r in A["rallies"]]}
(LAB / "runs.json").write_text(json.dumps(runs))
(LAB / "dataset-metadata.json").write_text(json.dumps({"title": "pb-balltrack-labels", "id": "santhosharun/pb-balltrack-labels",
                                                       "licenses": [{"name": "CC0-1.0"}]}))
(kd / "kernel-metadata.json").write_text(json.dumps({
    "id": "santhosharun/pb-balltrack-train", "title": "pb-balltrack-train", "code_file": "balltrack_kernel.py",
    "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": ["santhosharun/pb-balltrack-labels"] + [f"santhosharun/{VIDEO_DS[r]}" for r in runs],
    "machine_shape": "NvidiaTeslaT4"}, indent=1))
# high-resolution variant (round 3): same code, env defaults set at the top
hd = HERE / "kernel_hr"; hd.mkdir(exist_ok=True)
(hd / "balltrack_kernel.py").write_text('import os\nos.environ.setdefault("BT_W", "1024"); os.environ.setdefault("BT_H", "576"); os.environ.setdefault("BT_BS", "3")\n' + src)
meta = json.loads((kd / "kernel-metadata.json").read_text())
(hd / "kernel-metadata.json").write_text(json.dumps({**meta, "id": "santhosharun/pb-balltrack-train-hr", "title": "pb-balltrack-train-hr"}, indent=1))
print("runs:", list(runs), "| kernel:", kd / "balltrack_kernel.py")
