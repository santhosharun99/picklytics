"""Adds the TNPPL Salem Super Smashers v Twin Eagles Hosur matches that have a report to dashboard/matches.json (re-runnable)."""
import json
from pathlib import Path
ROOT = Path("/Users/santhosharun/Desktop/pickleball-analyzer")
C = json.loads((ROOT / "configs/tnppl_s2_sss_teh.json").read_text())
J = json.loads((ROOT / "dashboard/matches.json").read_text())
ROUND = {"1_MS": "Men's singles (open) · match 1 of the tie", "2_WD": "Women's doubles", "3_MD": "Men's doubles (open)",
         "4_XD": "Mixed doubles", "5_MD50": "Men's 50+ doubles", "6_MD": "Men's doubles (open)",
         "7_TeamDoubles": "Team event - super doubles · rotating doubles", "8_TeamSingles": "Team event - super singles · rotating singles, rally scoring to 25"}
GROUP = "Salem Super Smashers vs Twin Eagles Hosur"
have = {m["id"] for m in J["matches"]}
added = []
for mid, m in C["matches"].items():
    rep = ROOT / "outputs/reports" / f"{mid}.json"
    if not rep.exists():
        continue
    R = json.loads(rep.read_text())
    key = mid.replace("tnppl_st_", "")
    rs = R["rallies"]
    if R.get("type") == "doubles":
        sv = [x["server"] == x["scoreboard"]["server"] for x in rs if (x.get("scoreboard") or {}).get("server")]
        wn = [x["winner"] == x["scoreboard"]["winner"] for x in rs if (x.get("scoreboard") or {}).get("winner")]
        check = f"Score overlay: serving team right on {sum(sv)} of {len(sv)}, rally winner on {sum(wn)} of {len(wn)}"
        title = " vs ".join(" / ".join(r) for r in m["rosters"]) if len(m["rosters"][0]) == 2 else "Salem Super Smashers vs Twin Eagles Hosur (team)"
        players = [f"{t}: {' / '.join(r)}" for t, r in zip(m["teams"], m["rosters"])]
    else:
        sv = [x["server"] == x["scoreboard"]["server"] for x in rs if (x.get("scoreboard") or {}).get("server")]
        wn = [x["winner"] == x["scoreboard"]["winner"] for x in rs if (x.get("scoreboard") or {}).get("winner")]
        check = f"Score overlay: winner right on {sum(wn)} of {len(wn)}, server on {sum(sv)} of {len(sv)} checked rallies"
        title = " vs ".join(m["teams"]) + (" (super singles)" if "Team" in key else "")
        players = m["teams"]
    e = {"id": mid, "type": m["type"], "title": title, "players": players,
         "event": f"TNPPL Season 2 · {GROUP}", "round": ROUND[key], "venue": "Court 1 · indoor", "result": m["official"],
         "report": mid, "run": mid, "video": f"data/videos/TNPPL_S2_SSSvTEH_{key}.mp4", "thumb": f"thumbs/{mid}.jpg",
         "check": check, "tournament": "tnppl_s2", "group": GROUP}
    if (ROOT / f"dashboard/replay3d_{mid}.json").exists():
        e["replay"] = mid
    J["matches"] = [x for x in J["matches"] if x["id"] != mid] + [e]
    added.append((mid, check))
(ROOT / "dashboard/matches.json").write_text(json.dumps(J, indent=1, ensure_ascii=False))
for a in added: print(a)
