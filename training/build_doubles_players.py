"""Individual + partnership analytics for every doubles match with named players (pickleball/doubles_players.py).

    PYTHONPATH=. python3 training/build_doubles_players.py            # all doubles reports in dashboard/matches.json
    -> outputs/reports/<report>.doubles.json  (read by the dashboard; the report itself is not changed)
"""
import json
from pathlib import Path

from pickleball.doubles_players import build

ROOT = Path(__file__).resolve().parents[1]


def main():
    J = json.loads((ROOT / "dashboard/matches.json").read_text())
    todo = []
    for m in J["matches"]:
        if m.get("type") != "doubles" or not m.get("report") or not m.get("run"):
            continue
        run = ROOT / "outputs" / m["run"]
        if not (run / "identity.json").exists() or not (run / "tracklets.pkl").exists():
            print(f"{m['id']}: no named players (mark them in dashboard/identify.html) - skipped")
            continue
        R = json.loads((ROOT / "outputs/reports" / f"{m['report']}.json").read_text())
        A = json.loads((run / "analysis.json").read_text())
        todo.append((m, run, R, float((A.get("meta") or {}).get("fps") or 30.0)))
    for m, run, R, fps in todo:
        out = build(R, run, fps)   # positions on the run's depth model (pickleball/depth.py, training/build_tnppl.py)
        out.update({"match": m["id"], "report": m["report"], "run": m["run"]})
        (ROOT / "outputs/reports" / f"{m['report']}.doubles.json").write_text(json.dumps(out, indent=1))
        tm = {t: (v["pair_rallies"], v["formation_time_pct"], v["spacing"]["median_ft"]) for t, v in out["teams"].items()}
        print(f"{m['id']}: coverage {out['named_coverage_pct']}%, players {len(out['players'])}, teams {tm}")


if __name__ == "__main__":
    main()
