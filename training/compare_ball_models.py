"""One-command comparison of ball models on the analyst-labelled rallies.

    python training/compare_ball_models.py models/ball_detector.pt ~/Downloads/ball_detector_v2.pt ...

For each model: re-detect the ball on the labelled stretches (court/players kept), run the full
analysis (flight hits + learned swing detector), and score hits against labels/<video>.json.
Prints one table: hits found (recall), reported hits that are real (precision), F1, ball coverage.
"""
import contextlib, io, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "labels" / "Mens_Singles_GOLD_2026_US_Open.json"
BASE_RUNS = [ROOT / "outputs" / "Men_s_Singles_GOLD_2026_Franklin_US_Open_Pickleball_Champion_170-340",
             next((ROOT / "outputs").glob("*530-710*"))]
NAMES = {"P1": "Devilliers", "P2": "Boyer"}


_CROSS = {}


def cross_swing_models():
    """Swing detector for each labelled stretch trained on the OTHER stretch (honest scores)."""
    if not _CROSS:
        from training.train_swing import train
        labels = json.loads(LABELS.read_text())
        a, b = BASE_RUNS
        _CROSS[a.name[-7:]] = train(labels, [b])
        _CROSS[b.name[-7:]] = train(labels, [a])
    return _CROSS


def score_model(model: Path, imgsz: int) -> dict:
    sys.path.insert(0, str(ROOT))
    from pickleball.analysis import run_analysis
    import pickleball.events as ev
    import training.evaluate as E
    cross = cross_swing_models()
    m = t = p = 0
    cover = []
    for base in BASE_RUNS:
        out = ROOT / "outputs" / f"cmp_{model.stem}_{base.name[-7:]}"
        if not (out / "detections.jsonl").exists():
            subprocess.run([sys.executable, str(ROOT / "training" / "swap_ball_model.py"), str(base), str(model), str(out),
                            "--imgsz", str(imgsz)], check=True, capture_output=True)
        ev.SWING_MODEL["model"] = cross[base.name[-7:]]
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_analysis(out, 2, NAMES)
            sys.argv = ["evaluate", str(out), str(LABELS)]
            E.main()
        h = json.loads((out / "eval.json").read_text())["hits"]
        m, t, p = m + h["matched"], t + h["truth"], p + h["predicted"]
        cover.append(res["summary"]["ball_frames_pct"])
    rec, prec = m / t, m / max(p, 1)
    return {"recall": rec, "precision": prec, "f1": 2 * rec * prec / max(rec + prec, 1e-9),
            "ball_cover": sum(cover) / len(cover), "hits": f"{m}/{t}"}


if __name__ == "__main__":
    rows = []
    for arg in sys.argv[1:]:
        path, _, size = arg.partition("@")
        r = score_model(Path(path).expanduser(), int(size or 960))
        rows.append((Path(path).name + (f"@{size}" if size else ""), r))
        print(f"done {rows[-1][0]}", flush=True)
    print(f"\n{'ball model':40s} {'hits found':>11s} {'real':>6s} {'F1':>5s} {'ball seen':>10s}")
    for name, r in rows:
        print(f"{name:40s} {r['recall']:>6.0%} ({r['hits']:>5s}) {r['precision']:>5.0%} {r['f1']:>5.0%} {r['ball_cover']:>9.0f}%")
