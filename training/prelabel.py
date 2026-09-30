"""Turn an analysis run into draft labels for the labelling tool ("pre-labels").

    python training/prelabel.py outputs/<run> --video Mens_Singles_GOLD_2026_US_Open [--a Boyer --b Devilliers]

Writes labels/<video>.prelabel.json. In the labelling page, "Import auto-labels" merges these
rallies (marked as not reviewed); the person fixes them and presses Enter to mark each rally
reviewed. Only reviewed rallies count in training/evaluate.py, so the answer key stays human.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--video", required=True, help="label file stem, e.g. the name of the video in data/videos")
    ap.add_argument("--a", default=None, help="name of player A in the labeller (default: first analysed player)")
    ap.add_argument("--b", default=None, help="name of player B")
    args = ap.parse_args()

    run = Path(args.run)
    A = json.loads((run / "analysis.json").read_text())
    R = json.loads((run / "render_data.json").read_text())
    fps = A["meta"]["fps"]
    names = {pid: p["name"] for pid, p in sorted(A["players"].items())}
    order = list(names.values())
    name_a = args.a or order[0]
    name_b = args.b or (order[1] if len(order) > 1 else "B")
    to_ab = {pid: ("A" if n == name_a else "B" if n == name_b else None) for pid, n in names.items()}
    img_by_frame = {(e["kind"], e["frame"]): e["img"] for e in R["events"]}

    rallies = []
    for r in A["rallies"]:
        start, end = round(r["start_t"] * fps), round(r["end_t"] * fps)
        events, near = [], None
        for s in A["shots"]:
            if s["rally"] != r["index"]:
                continue
            who = to_ab.get(s["player"])
            if s["side"] == "near" and who:
                near = near or who
            ev = {"kind": "hit", "frame": s["frame"], "side": s["side"], "player": who,
                  "shot": (s["shot_type"] or "").replace("third-shot ", "") or None,
                  "stroke": s["stroke"], "volley": bool(s["volley"]), "auto": True}
            if ("hit", s["frame"]) in img_by_frame:
                ev["xy"] = [round(v) for v in img_by_frame[("hit", s["frame"])]]
            events.append(ev)
        for b in A["bounces"]:
            if start <= b["frame"] <= end:
                ev = {"kind": "bounce", "frame": b["frame"], "auto": True}
                if ("bounce", b["frame"]) in img_by_frame:
                    ev["xy"] = [round(v) for v in img_by_frame[("bounce", b["frame"])]]
                events.append(ev)
        events.sort(key=lambda e: e["frame"])
        if near is None:  # no near-side hit attributed: infer from a far-side one
            far = next((e["player"] for e in events if e["kind"] == "hit" and e["player"]), None)
            near = {"A": "B", "B": "A"}.get(far, "A")
        rallies.append({"id": r["index"], "start": start, "end": end, "near": near, "winner": None,
                        "ending": None, "auto": True, "reviewed": False, "events": events})

    out = ROOT / "labels" / f"{args.video}.prelabel.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"video": f"{args.video}.mp4", "fps": fps, "source_size": [A["meta"]["width"], A["meta"]["height"]],
                               "players": {"A": name_a, "B": name_b}, "from_run": run.name, "rallies": rallies}, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}: {len(rallies)} rallies, {sum(len(r['events']) for r in rallies)} events")


if __name__ == "__main__":
    main()
