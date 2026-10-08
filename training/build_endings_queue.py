"""Queue of rallies for labelling how each rally starts and ends (dashboard/endings.html -> labels/endings.json).

    PYTHONPATH=. python3 training/build_endings_queue.py [--per-video 8]
Per rally the labeller marks the serve, the last live shot and how the point ended (net / out / in-not-returned),
so every detected touch gets a live / dead label and every rally an ending - training and test data for the
rally-end model (pickleball/rally_end.py). Rallies are drawn at random from every processed match (no hand-picking).
"""
import argparse, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["tnppl_1_MS", "tnppl_2_WD", "tnppl_3_MD", "tnppl_4_XD", "tnppl_5_MD50", "tnppl_6_MD", "tnppl_7_TeamDoubles",
        "tnppl_8_TeamSingles", "lv_waters_johns", "lv_bright_patriquin", "lv_fahey_yoshitomi", "full_match"]
PROXY = {"lv_fahey_yoshitomi": "data/videos/Fahey_Yoshitomi_LV2026.mp4",  # same timing as the originals
         "full_match": "data/videos/Mens_Singles_GOLD_2026_US_Open.mp4"}
ap = argparse.ArgumentParser()
ap.add_argument("--per-video", type=int, default=8)
a = ap.parse_args()
rng = random.Random(7)
items = []
for rid in RUNS:
    run = ROOT / "outputs" / rid
    if not (run / "analysis.json").exists():
        continue
    A = json.loads((run / "analysis.json").read_text())
    rep = json.loads((run / "replays.json").read_text())["rallies"] if (run / "replays.json").exists() else {}
    rallies = [r for r in A["rallies"] if not rep.get(str(r["index"]), {}).get("replay")]
    for r in rng.sample(rallies, min(a.per_video, len(rallies))):
        sh = sorted([s for s in A["shots"] if s["rally"] == r["index"]], key=lambda s: s["frame"])
        if not sh:
            continue
        items.append({"id": f"{rid}:{r['index']}", "run": rid, "rally": r["index"], "video": PROXY.get(rid, A["meta"]["video"]),
                      "fps": A["meta"]["fps"], "t0": round(max(0.0, sh[0]["t"] - 4.0), 2), "t1": round(sh[-1]["t"] + 4.0, 2),
                      "touches": [{"i": s["index"], "t": round(s["t"], 2), "side": s["side"], "shot": s.get("shot_type"),
                                   "inferred": bool(s.get("inferred"))} for s in sh]})
rng.shuffle(items)
(ROOT / "labels" / "endings.queue.json").write_text(json.dumps({"items": items}, indent=0))
print(len(items), "rallies queued -> dashboard/endings.html")
