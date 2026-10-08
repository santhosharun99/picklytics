"""Physics pass for one big match split over N processes (rallies round-robin), then merged into rally3d.json.

    PYTHONPATH=. python3 training/run_physics_split.py lv_bright_patriquin 3
Each part runs single-threaded (OMP/BLAS 1 thread); one part per spare core.
"""
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if len(sys.argv) == 4:  # worker: run, n, k
    from pickleball.physics_pass import run_match
    run, n, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    A = json.loads((ROOT / "outputs" / run / "analysis.json").read_text())
    mine = {r["index"] for i, r in enumerate(A["rallies"]) if i % n == k}
    out = run_match(ROOT / "outputs" / run, rallies=mine, write=False, log=lambda m: print(" ", m, flush=True))
    (ROOT / "outputs" / run / f"rally3d.part{k}.json").write_text(json.dumps(out))
    sys.exit(0)
run, n = sys.argv[1], int(sys.argv[2])
env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "VECLIB_MAXIMUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
procs = [subprocess.Popen([sys.executable, __file__, run, str(n), str(k)], cwd=ROOT, env=env,
                          stdout=open(ROOT / "outputs/balltrack" / f"physics_{run}.part{k}.log", "w"), stderr=subprocess.STDOUT)
         for k in range(n)]
codes = [p.wait() for p in procs]
merged = None
for k in range(n):
    part = json.loads((ROOT / "outputs" / run / f"rally3d.part{k}.json").read_text())
    if merged is None:
        merged = part
    else:
        merged["rallies"].update(part["rallies"]); merged["shots"].update(part["shots"])
(ROOT / "outputs" / run / "rally3d.json").write_text(json.dumps(merged, separators=(",", ":")))
for k in range(n):
    (ROOT / "outputs" / run / f"rally3d.part{k}.json").unlink()
print(f"{run}: exit codes {codes}, {len(merged['rallies'])} rallies, {len(merged['shots'])} shots")
