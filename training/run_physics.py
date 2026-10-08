"""Physics pass (pickleball/physics_pass.py) over matches -> outputs/<run>/rally3d.json.

    PYTHONPATH=. python3 training/run_physics.py tnppl_1_MS [tnppl_6_MD ...]
Needs outputs/<run>/ball_v3.json (training/balltrack_run.py, all rallies). CPU heavy: one match at a time.
"""
import sys
from pathlib import Path

from pickleball.physics_pass import run_match

ROOT = Path(__file__).resolve().parents[1]
for run in sys.argv[1:]:
    print(f"== {run}", flush=True)
    out = run_match(ROOT / "outputs" / run, log=lambda m: print(" ", m, flush=True))
    print(f"   {len(out['rallies'])} rallies, {len(out['shots'])} shots -> outputs/{run}/rally3d.json", flush=True)
