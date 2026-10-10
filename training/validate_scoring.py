"""Validate evidence-fusion scoring (pickleball/score_decode.py) by HIDING the score overlay.

    PYTHONPATH=. python3 training/validate_scoring.py [--matches tnppl_2_WD ...] [--calibrate] [--out docs/scoring_validation.json]
Reference = decode with every visible overlay reading. Scenarios hide the overlay: all of it, the first 5 minutes
(the overlay is often missing at the start), random 60 s blocks covering ~30% / ~50% of the match (3 seeds each).
Per scenario: serving team, rally winner and score-before agreement with the reference on the rallies that are
points in the reference, and missed-point counts. --calibrate prints how often each tracking cue agrees with the
reference (the reliabilities used in score_decode.SCORING_CUES).
Also: the reference vs the raw overlay transitions (overlay_truth.rally_truth_seq) as a sanity check.
"""
import argparse, json, random, sys, time
from collections import defaultdict
from pathlib import Path
from pickleball.score_decode import decode_match, match_cues, name_kits_by_colour
from pickleball.replays import drop_replays
from pickleball.overlay_truth import rally_truth_seq

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["tnppl_s2_kt_rtt", "tnppl_s2_sss_teh", "tnppl_s2_rr_cs"]
TIE = {"tnppl_s2_kt_rtt": "KT v RTT", "tnppl_s2_sss_teh": "SSS v TEH", "tnppl_s2_rr_cs": "RR v CS"}


def matches(only=None):
    for c in CONFIGS:
        C = json.loads((ROOT / "configs" / f"{c}.json").read_text())
        for k, m in C["matches"].items():
            if m.get("skip") or (only and k not in only):
                continue
            yield c, k, m, list(C["rosters"].values())


def scenarios(t_end, seeds=3):
    yield "all hidden", None
    yield "first 5 min hidden", [(0, 300)]
    for frac in (0.3, 0.5):
        for sd in range(seeds):
            rnd = random.Random(sd)
            blocks = [(t, t + 60) for t in range(0, int(t_end) + 60, 60) if rnd.random() < frac]
            yield f"random {int(frac * 100)}% hidden", blocks


def compare(ref, got):
    """Agreement on the reference's points (by rally index)."""
    g = {r["index"]: r for r in got["rallies"]}
    n = sv = wn = sc = 0
    by_conf = defaultdict(lambda: [0, 0])
    for r in ref["rallies"]:
        if r["role"] != "point":
            continue
        x = g.get(r["index"])
        n += 1
        ok_s = x is not None and x["server_team"] == r["server_team"]
        ok_w = x is not None and x["winner_team"] == r["winner_team"]
        sv += ok_s
        wn += ok_w
        sc += x is not None and x["score_before"] == r["score_before"] and x["server_team"] == r["server_team"]
        if x is not None:
            by_conf[x["confidence"]][0] += ok_s and ok_w
            by_conf[x["confidence"]][1] += 1
    return {"n": n, "server": sv, "winner": wn, "both": None, "score": sc, "by_conf": dict(by_conf),
            "missed_ref": len(ref["missed"]), "missed": len(got["missed"])}


def evaluate(job):
    """One match: reference decode, sanity check vs the raw overlay, hidden-overlay scenarios, cue calibration."""
    cfg, k, m, squads, prep, meta, seeds = job
    t0 = time.time()
    run = ROOT / "outputs" / k
    A = {"meta": meta}
    cal = defaultdict(lambda: [0, 0])
    ref = decode_match(run, A, m, squads, prepared=prep)
    pts = [r for r in ref["rallies"] if r["role"] == "point"]
    truth = {t["index"]: t for t in rally_truth_seq(run / "overlay.json", [e for e in ref["events"]], squads)}
    tw = [(r["winner_team"] == truth[r["index"]]["winner_team"]) for r in pts if truth[r["index"]]["winner_team"] is not None]
    ts = [(r["server_team"] == truth[r["index"]]["server_team"]) for r in pts if truth[r["index"]]["server_team"] is not None]
    covered = sum(1 for r in ref["rallies"] if r["provenance"] == "overlay")
    row = {"tie": TIE[cfg], "match": k, "type": m["type"], "scoring": m["rules"].get("scoring", "sideout"),
           "rallies": len(ref["rallies"]), "points": len(pts), "missed": len(ref["missed"]),
           "final": ref["final"], "overlay_confirmed": covered, "flip": ref["flip"], "flip_margin": ref["flip_margin"],
           "vs_overlay": {"winner": [sum(tw), len(tw)], "server": [sum(ts), len(ts)]}, "scenarios": {}}
    st = {r["index"]: r["state"] for r in pts}
    fl, n0, game = ref["flip"], ref["n0"], ref["game"]
    for ev in ref["events"]:
        s = st.get(ev["index"])
        if s is None:
            continue
        if ev.get("serve_kit") is not None:
            c = cal[(m["type"], ev["cue"])]
            c[0] += (ev["serve_kit"] ^ fl) == s[2]; c[1] += 1
        if ev.get("near_kit") is not None:
            c = cal[(m["type"], "near")]
            c[0] += (ev["near_kit"] ^ fl) == (n0 if not game.switched(s) else 1 - n0); c[1] += 1
        if ev.get("server_player") and ev["server_player"][0] == s[2]:
            c = cal[(m["type"], "player")]
            c[0] += ev["server_player"][1] == s[4]; c[1] += 1
        if ev.get("serve_right") is not None and game.sideout:
            c = cal[(m["type"], "parity")]
            c[0] += ev["serve_right"] == ((s[0], s[1])[s[2]] % 2 == 0); c[1] += 1
    t_end = meta.get("last_frame", 0) / meta["fps"]
    agg = defaultdict(list)
    for name, hide in scenarios(t_end, seeds):
        got = decode_match(run, A, m, squads, hide=hide, use_overlay=hide is not None, prepared=prep)
        agg[name].append(compare(ref, got))
    for name, lst in agg.items():
        tot = {key: sum(x[key] for x in lst) for key in ("n", "server", "winner", "score", "missed_ref", "missed")}
        bc = defaultdict(lambda: [0, 0])
        for x in lst:
            for c, (ok, nn) in x["by_conf"].items():
                bc[c][0] += ok; bc[c][1] += nn
        tot["by_conf"], tot["runs"] = dict(bc), len(lst)
        row["scenarios"][name] = tot
    s_all = row["scenarios"]["all hidden"]
    msg = (f"{k:24s} {len(pts):3d} pts, {row['missed']:2d} missed, overlay-confirmed {covered:3d} | vs overlay winner "
           f"{sum(tw)}/{len(tw)} server {sum(ts)}/{len(ts)} | all hidden: server {s_all['server']}/{s_all['n']} winner "
           f"{s_all['winner']}/{s_all['n']} ({time.time() - t0:.0f}s)")
    return row, dict(cal), msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="*")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "docs" / "scoring_validation.json"))
    a = ap.parse_args()
    preps, jobs = defaultdict(dict), []
    want_cfg = {c for c, _, _, _ in matches(a.matches)}
    for cfg, k, m, squads in matches():  # cues for the whole tie: kits are named by colour across its matches
        if cfg not in want_cfg:
            continue
        run = ROOT / "outputs" / k
        A = drop_replays(json.loads((run / "analysis.json").read_text()), run)
        preps[cfg][k] = (run, m, match_cues(run, A, m), A["meta"])
    for cfg in preps:
        named = name_kits_by_colour({k: v[:3] for k, v in (x for x in preps[cfg].items())}, )
        for k, (f, mg) in named.items():
            run, m, (g, ev, _), meta = preps[cfg][k]
            preps[cfg][k] = (run, m, (g, ev, (f,)), meta)
            print(f"{TIE[cfg]} {k}: kits named by colour (flip {f}, margin {mg})")
    for cfg, k, m, squads in matches(a.matches):
        _, _, prep, meta = preps[cfg][k]
        jobs.append((cfg, k, m, squads, prep, meta, a.seeds))
    from multiprocessing import Pool
    res, cal = [], defaultdict(lambda: [0, 0])
    with Pool(a.jobs) as pool:
        for row, c, msg in pool.imap(evaluate, jobs):
            print(msg, flush=True)
            res.append(row)
            for key, (ok, n) in c.items():
                cal[key][0] += ok; cal[key][1] += n
    if a.calibrate:
        print("\ncue reliability vs the reference decode:")
        for (typ, cue), (ok, n) in sorted(cal.items()):
            print(f"  {typ:8s} {cue:28s} {ok:4d}/{n:<4d} {ok / max(n, 1):.3f}")
    print()
    names = list(res[0]["scenarios"]) if res else []
    for name in names:
        print(name)
        for tie in dict.fromkeys(r["tie"] for r in res):
            rows = [r["scenarios"][name] for r in res if r["tie"] == tie]
            n = sum(x["n"] for x in rows)
            s = sum(x["server"] for x in rows); w = sum(x["winner"] for x in rows); sc = sum(x["score"] for x in rows)
            mr = sum(x["missed_ref"] for x in rows); mg = sum(x["missed"] for x in rows)
            print(f"  {tie:10s} server {s}/{n} {s / max(n, 1):.1%}  winner {w}/{n} {w / max(n, 1):.1%}  score {sc / max(n, 1):.1%}"
                  f"  missed points {mg} (reference {mr})")
    if a.out and not a.matches:
        Path(a.out).write_text(json.dumps({"date": time.strftime("%Y-%m-%d"), "matches": res,
                                           "calibration": {f"{t}|{c}": v for (t, c), v in cal.items()}}, indent=1))


if __name__ == "__main__":
    main()
