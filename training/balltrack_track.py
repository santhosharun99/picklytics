"""Tracking on top of the heat-map detector: global best path over its top-3 candidates per frame.

    PYTHONPATH=. python3 training/balltrack_track.py outputs/balltrack/pred_v2_all_cands.json outputs/balltrack/pred_v2_tracked.json
Same segmental Viterbi as the old tracker (pickleball/ball.track_ball_global: speed-limited links, restarts, short
gap filling) but over the new detector's candidates (1280x720 pixels; speeds scaled from 1920 px).
"""
import json, sys
from pathlib import Path

from pickleball.ball import track_ball_global

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
min_conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
C = json.loads(src.read_text())
out = {}
for clip, frames in C.items():
    recs = [{"f": int(f), "court": True, "balls": [[x - 4, y - 4, x + 4, y + 4, c] for x, y, c in cs]}
            for f, cs in sorted(frames.items(), key=lambda kv: int(kv[0]))]
    tr = track_ball_global(recs, min_conf=min_conf, free_speed=45.0 * 1280 / 1920, cell=8)
    out[clip] = {str(f): [round(v[0], 1), round(v[1], 1), round(float(v[2]), 3)] for f, v in tr.items()}
    print(clip, len(out[clip]), "frames with a ball", flush=True)
dst.write_text(json.dumps(out))
