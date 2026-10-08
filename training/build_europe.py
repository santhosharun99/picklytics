"""Reports for the PPA Europe matches split out of the court feed (training/europe_split.py).

    PYTHONPATH=. python3 training/build_europe.py
Team sides come from the overlay, not the kits (outdoors the kit colours did not separate the teams: serving team
vs overlay dot only 10/17 and 28/50): serving team (dot, overlay_serve.json) + the serve's end (tracking) ->
sides_override.json, which doubles.team_sides uses. PPA doubles: side-out scoring, best of 3 to 11.
"""
import json
from pathlib import Path

from pickleball.doubles import reconstruct
from pickleball.report import _load_named, build_report_doubles
from training.europe_split import MATCHES

ROOT = Path(__file__).resolve().parents[1]
RULES = {"scoring": "sideout", "target": 11, "win_by": 2, "games_to_win": 2}
def sides_from_overlay(A, ov) -> dict:
    """Near team per rally (0 = overlay's top team). Per rally a raw vote: serving team (overlay dot) + the serve's
    end from tracking (often wrong on this low camera: the serve is missed and the first hit is the return). Teams
    keep their ends for a whole game (overlay game number) except the switch at 6 in game 3 -> majority per game,
    game 3 as two parts at its best single change point."""
    first = {}
    for s in sorted(A["shots"], key=lambda s: s["frame"]):
        first.setdefault(s["rally"], s)
    rs = sorted(A["rallies"], key=lambda r: r["start_t"])
    games, cur = [], None
    for r in rs:  # game per rally, holes filled from the previous rally
        g = (ov.get(r["index"]) or {}).get("game") or cur
        games.append(g); cur = g
    raw = []
    for r in rs:
        o, sh = ov.get(r["index"]) or {}, first.get(r["index"])
        raw.append(None if o.get("row") is None or sh is None or sh.get("inferred") else (o["row"] if sh["side"] == "near" else 1 - o["row"]))
    out = {}
    for g in sorted({x for x in games if x}):
        ix = [i for i, x in enumerate(games) if x == g]
        maj = lambda part: (1 if sum(raw[i] for i in part if raw[i] is not None) * 2 > sum(raw[i] is not None for i in part) else 0)
        parts = [ix]
        if g == 3 and len(ix) >= 6:  # ends change at 6: best split, if it explains the votes clearly better
            def fit(k):
                a, b = ix[:k], ix[k:]
                ma, mb = maj(a), maj(b)
                return sum(raw[i] == ma for i in a if raw[i] is not None) + sum(raw[i] == mb for i in b if raw[i] is not None), k, ma, mb
            best = max(fit(k) for k in range(3, len(ix) - 2))
            one = sum(raw[i] == maj(ix) for i in ix if raw[i] is not None)
            if best[2] != best[3] and best[0] >= one + 3:
                parts = [ix[:best[1]], ix[best[1]:]]
        for part in parts:
            m_ = maj(part)
            for i in part:
                out[rs[i]["index"]] = m_
    return out


for m in MATCHES:
    run = ROOT / "outputs" / m["id"]
    ov = {int(k): v for k, v in json.loads((run / "overlay_rally.json").read_text()).items()}
    serve = {k: v["row"] for k, v in ov.items()}
    A = _load_named(run, None, True)
    (run / "sides_override.json").write_text(json.dumps(sides_from_overlay(A, ov)))
    (run / "serve_override.json").write_text(json.dumps(serve))
    rec = reconstruct(run, A, RULES["scoring"], RULES["target"], RULES["win_by"], RULES["games_to_win"])
    pairs = [(r["serve_kit"], serve.get(r["index"])) for r in rec if r.get("serve_kit") is not None and serve.get(r["index"]) is not None]
    agree = sum(k == row for k, row in pairs)
    kit0 = 0  # sides_override.json: kit = overlay team index
    assert agree >= len(pairs) - agree, "overlay sides should agree with themselves"
    print(f"{m['id']}: serving kit vs overlay on {len(pairs)} rallies: kit 0 = top team agrees {agree}, disagrees {len(pairs) - agree} -> kit_of_team0 = {kit0}")
    cut = json.loads((run / "video_cut.json").read_text())
    r = build_report_doubles(run, f"data/videos/PPA_Europe_{m['id']}.mp4", ROOT / "outputs" / "reports" / f"{m['id']}.json",
                             m["teams"], kit0, RULES, official=m["official"], rosters=m["rosters"], replays=False)
    r["video_offset"] = cut["offset_s"]  # the match's own video file starts here in the feed
    (ROOT / "outputs" / "reports" / f"{m['id']}.json").write_text(json.dumps(r, indent=1))
    rs = r["rallies"]
    print(f"  {r['title']}: {len(rs)} rallies, {sum(x['n_shots'] for x in rs)} shots; "
          f"serving team vs overlay {sum(1 for x in rs if serve.get(x['id']) is not None and x['server'] == m['teams'][serve[x['id']]])}"
          f"/{sum(1 for x in rs if serve.get(x['id']) is not None)}")
    g = {}
    for x in rs:
        if x.get("game") and x.get("winner") and x["server"] == x["winner"]:
            g.setdefault(x["game"], {}).setdefault(x["winner"], 0)
            g[x["game"]][x["winner"]] += 1
    print("  reconstructed games:", {k: f"{v.get(m['teams'][0], 0)}-{v.get(m['teams'][1], 0)}" for k, v in sorted(g.items())},
          "| official:", m["official"])
