"""Score event-detection settings against hand labels (analysis only, uses cached detections).

    python training/sweep_events.py outputs/<run> labels/<video>.json
"""
import contextlib, io, json, sys
from pathlib import Path

import pickleball.events as ev
from pickleball.analysis import run_analysis

run, labels = Path(sys.argv[1]), sys.argv[2]


def score():
    with contextlib.redirect_stdout(io.StringIO()):
        run_analysis(run, 2, {"P1": "Devilliers", "P2": "Boyer"})
        import training.evaluate as E
        sys.argv = ["evaluate", str(run), labels]
        E.main()
    r = json.loads((run / "eval.json").read_text())
    h = r["hits"]
    return h["recall"], h["precision"], h["f1"], r["matched_hit_accuracy"].get("side", {}).get("accuracy")


configs = [("old detectors only", dict(enabled=False))]
for pen in (1200, 2500, 5000, 10000):
    for reach in (0.45, 0.6, 0.75):
        configs.append((f"flights only  pen={pen} reach={reach}", dict(enabled=True, penalty=pen, reach=reach, only=True)))
for name, cfg in configs:
    ev.FLIGHT_SETTINGS.update({"only": False, **cfg})
    rec, prec, f1, side = score()
    print(f"{name:38s} recall {rec:.0%}  precision {prec:.0%}  F1 {f1:.0%}  side {side}", flush=True)
