"""Build the TNPPL reports (configs/tnppl_s2_kt_rtt.json): tracking report + score-overlay cross-check.

    PYTHONPATH=. python3 training/build_tnppl.py [tnppl_1_MS ...] [--config configs/tnppl_s2_sss_teh.json]
Needs outputs/<id>/analysis.json (Kaggle run) and outputs/<id>/overlay.json (training/overlay_timeline.py).
First recalibrates the player depth correction of every match of the tie (one court and camera rig; pickleball/depth.py:
kitchen-line rule pooled over the doubles matches with named players) -> outputs/<id>/depth.json, used by every report.
"""
import json, sys
from pathlib import Path
from pickleball import depth
from pickleball.report import build_report_auto, build_report_doubles
from pickleball.overlay_truth import rotation_results

ROOT = Path(__file__).resolve().parents[1]
args = sys.argv[1:]
cfg = "configs/tnppl_s2_kt_rtt.json"
if "--config" in args:
    i = args.index("--config"); cfg = args[i + 1]; args = args[:i] + args[i + 2:]
C = json.loads((ROOT / cfg).read_text())
ids = args or [k for k, m in C["matches"].items() if m["type"] == "singles"]
# one depth model per court + camera rig, pooled over EVERY tie filmed with it (all configs naming the same "rig")
rig_id = C.get("rig")
rig_runs = [ROOT / "outputs" / k for c in sorted((ROOT / "configs").glob("tnppl_s2_*.json"))
            for k in json.loads(c.read_text())["matches"] if json.loads(c.read_text()).get("rig") == rig_id] if rig_id else \
           [ROOT / "outputs" / k for k in C["matches"]]
rig = depth.calibrate_rig(rig_runs, ROOT / "configs" / "cameras" / "rigs" / f"{rig_id}.json" if rig_id else None)
print(f"depth ({rig_id or 'this tie'}): h={rig['h_ft']} ft, {rig['method']}; kitchen {rig['kitchen'].get('h_ft')} "
      f"({rig['kitchen']['stances']}), servers {rig['servers'].get('h_ft')} ({rig['servers']['stances']}); {len(rig['runs'])} runs")
for mid in ids:
    m = C["matches"][mid]
    run = ROOT / "outputs" / mid
    meta = json.loads((run / "meta.json").read_text())
    if m["type"] == "doubles":
        allr = list(C["rosters"].values())  # overlay names: match the whole squads (OCR slips); order = overlay rows
        r = build_report_doubles(run, meta["video"], ROOT / "outputs" / "reports" / f"{mid}.json", m["teams"], None,
                                 m["rules"], overlay={"path": str(run / "overlay.json"), "rosters": allr},
                                 official=m["official"], rosters=m["rosters"], replays=True)
        rs = r["rallies"]
        sv = [x["server"] == x["scoreboard"]["server"] for x in rs if x["scoreboard"].get("server")]
        wn = [x["winner"] == x["scoreboard"]["winner"] for x in rs if x["scoreboard"].get("winner")]
        print(f"{mid}: {r['source']}\n  vs overlay: serving team {sum(sv)}/{len(sv)}, rally winner {sum(wn)}/{len(wn)};"
              f" players named: {list(r['individuals'])}; replays dropped {r['replays_dropped']}")
        continue
    overlay = {"path": str(run / "overlay.json"), "teams": m["teams"], "rosters": m["rosters"],
               "scoring": m["rules"].get("scoring", "sideout"), "singles": bool(m.get("singles_overlay"))}
    r = build_report_auto(run, meta["video"], meta["video"], ROOT / "outputs" / "reports" / f"{mid}.json", None,
                          names=m["names"], rules=m["rules"], overlay=overlay, official=m["official"], replays=True)
    rs = r["rallies"]
    if m.get("rotation"):  # rotating team singles: who was on court, from the overlay score
        every = m["rotation"]["every"]
        r["rotations"] = rotation_results(overlay["path"], m["rosters"], every)
        r["overlay_teams"] = m["teams"]
        for x in rs:
            sc = x["scoreboard"].get("score")
            if sc is not None:
                slot = (sum(sc) // every) % len(m["rosters"][0])
                x["on_court"] = {m["teams"][0]: m["rosters"][0][slot], m["teams"][1]: m["rosters"][1][slot]}
        (ROOT / "outputs" / "reports" / f"{mid}.json").write_text(json.dumps(r, indent=1))
    chk = [x["scoreboard"] for x in rs]
    sv = [(x["server"] == c["server"]) for x, c in zip(rs, chk) if c.get("server")]
    wn = [(x["winner"] == c["winner"]) for x, c in zip(rs, chk) if c.get("winner")]
    print(f"{mid}: {r['source']}\n  vs overlay: server {sum(sv)}/{len(sv)}, winner {sum(wn)}/{len(wn)}; replays dropped {r['replays_dropped']}")
