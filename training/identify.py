"""Mark-once player identity for a doubles run (pickleball/identity.py).

    PYTHONPATH=. python3 training/identify.py tnppl_2_WD            # prepare frames / apply marks + answers
Step 1 (no marks yet): writes outputs/<run>/identity_frames.json - frames with every on-court player boxed - for
dashboard/identify.html?run=<run>, where the user clicks each player once and names them (labels/<video>.identity.json).
Step 2 (marks saved): names every tracklet, writes outputs/<run>/identity.json (+ the 'is this X?' confirmation queue
with thumbnails) and outputs/<run>/hitters.json (name of the hitter of every shot). Answers from the queue are
applied on the next run. Run heavy steps one at a time (laptop overheats with parallel video jobs).
"""
import json, pickle, sys
from pathlib import Path

import numpy as np

from pickleball import identity as I
from pickleball.replays import drop_replays

ROOT = Path(__file__).resolve().parents[1]
rid = sys.argv[1]
run = ROOT / "outputs" / rid
A = json.loads((run / "analysis.json").read_text())
if (run / "replays.json").exists():
    A = drop_replays(A, run)
fps = A["meta"]["fps"]
video_stem = Path(A["meta"]["video"]).stem
cache = run / "tracklets.pkl"
if cache.exists():
    trk = pickle.loads(cache.read_bytes())
else:
    trk = I.build_tracklets(run)
    I.add_looks(trk, A["meta"]["video"])
    cache.write_bytes(pickle.dumps(trk))
I.add_kits_from_rallies(trk, run, A)

# frames for marking: shortly before rallies' first real hit, all four players tracked; spread over the match
frames = []
for r in sorted(A["rallies"], key=lambda r: r["start_t"]):
    real = [s for s in A["shots"] if s["rally"] == r["index"] and not s.get("inferred")]
    if not real:
        continue
    f = min(s["frame"] for s in real) - int(0.5 * fps)
    here = []
    for t in trk:
        if t["f0"] <= f <= t["f1"] and I.is_player(t):
            i = int(np.argmin(np.abs(np.array(t["frames"]) - f)))
            if abs(t["frames"][i] - f) <= 3 and I.box_ok(t["box"][i], t["court"][i]):
                here.append((t, i))
    # exactly the four players, all fully in the picture and inside the court (no referee / spectator boxes)
    if len([t for t, _ in here if t["side"] == "near"]) == 2 and len([t for t, _ in here if t["side"] == "far"]) == 2:
        boxes = [{"tracklet": t["id"], "box": [round(float(v), 1) for v in t["box"][i]], "side": t["side"], "kit": t["kit"]}
                 for t, i in here]
        frames.append({"t": round(f / fps, 3), "frame": int(f), "rally": r["index"], "boxes": boxes})
pick = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, min(16, len(frames)))] if frames else []
(run / "identity_frames.json").write_text(json.dumps({"run": rid, "video": A["meta"]["video"], "label": video_stem,
                                                       "fps": fps, "frames": pick}, indent=0))
print(f"{len(trk)} tracklets; {len(pick)} frames for marking -> dashboard/identify.html?run={rid}")

lab_path = ROOT / "labels" / f"{video_stem}.identity.json"
if not lab_path.exists():
    sys.exit(0)
L = json.loads(lab_path.read_text())
marks, dropped = [], []
for m in L.get("marks", []):
    t = I.tracklet_at(trk, m["frame"], box=m["box"])
    if t is not None and I.is_player(t):
        marks.append({"tracklet": t["id"], "name": m["name"], "team": m.get("team")})
    else:
        dropped.append(m)
if dropped:  # marks on a referee / spectator box (old marking frames): removed from the labels file
    L["marks"] = [m for m in L["marks"] if m not in dropped]
    lab_path.write_text(json.dumps(L, indent=1))
    print("dropped marks on non-player boxes:", [(m["name"], m["t"]) for m in dropped])
names = I.assign(trk, marks, L.get("answers", []))
out = I.save(run, trk, names)
hitters = {}
for s in A["shots"]:
    if s.get("inferred"):
        continue
    t = I.tracklet_at(trk, s["frame"], s["side"], court=s.get("hitter_court"))
    n = names.get(t["id"]) if t is not None else None
    hitters[f"{s['rally']}:{s['index']}"] = {"name": n["name"], "conf": n["conf"]} if n else None
(run / "hitters.json").write_text(json.dumps(hitters, indent=0))
named = sum(1 for v in hitters.values() if v)
print(f"marks {len(marks)}, answers {len(L.get('answers', []))}; shots named {named}/{len(hitters)}; "
      f"queue {len(out['queue'])} -> confirm on dashboard/identify.html?run={rid}")
