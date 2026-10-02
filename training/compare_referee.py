"""Compare an automatic match analysis with a referee live-scoring log (pickleball.com).

    PYTHONPATH=. python3 training/compare_referee.py outputs/<run> data/reference/<match>.json

The broadcast cuts dead time, so video time and the referee's wall clock differ by an offset that
changes at breaks. Alignment: for every window of our rallies, the offset (referee start - our start)
that lines up the most rallies within +-TOL s; each of our rallies is then matched to the nearest
referee rally under its local offset. Player names are mapped once (shirt-colour ids P1/P2 -> names:
the mapping that agrees best), like naming the players when setting up a match.
"""
import json, sys
from collections import Counter
from pathlib import Path

import numpy as np

from pickleball.outcome import winners_from_serves

TOL = 4.0  # seconds


def our_rallies(A, rules=True):
    if rules:
        from pickleball.outcome import match_from_analysis
        return match_from_analysis(A)
    name = {pid: p.get("name") or pid for pid, p in A["players"].items()}
    seq = []
    for r in sorted(A["rallies"], key=lambda r: r["start_t"]):
        sh = sorted([s for s in A["shots"] if s["rally"] == r["index"]], key=lambda s: s["frame"])
        if not sh:
            continue
        srv = name.get(sh[0]["player"], sh[0]["player"])
        seq.append({"index": r["index"], "start_t": r["start_t"], "end_t": r["end_t"], "server": srv,
                    "receiver": next((n for n in name.values() if n != srv), None), "n_shots": len(sh),
                    "inferred": sum(1 for s in sh if s.get("inferred"))})
    return winners_from_serves(seq)


def align(ours, ref, jump_penalty=0.15, max_back=TOL):
    """Order-preserving matching (dynamic programming). offset = referee time - video time can only
    grow (the broadcast removes dead time, never adds it) and should stay steady between cuts.
    Score = matched pairs - jump_penalty * (offset change in seconds) / 10; offset may shrink by at
    most max_back s (timing noise)."""
    t = [r["start_t"] for r in ours]
    s = [r["start_s"] for r in ref]
    pairs = [(i, j) for i in range(len(t)) for j in range(len(s))]
    best, prev = {}, {}
    for i, j in pairs:  # pairs in (i, j) order
        off = s[j] - t[i]
        b, p = 1.0, None
        for (i2, j2), v in best.items():
            if i2 < i and j2 < j:
                d = off - (s[j2] - t[i2])
                if d < -max_back:
                    continue
                cand = v + 1.0 - jump_penalty * abs(d) / 10.0
                if cand > b:
                    b, p = cand, (i2, j2)
        best[(i, j)], prev[(i, j)] = b, p
    if not best:
        return {}
    cur = max(best, key=best.get)
    out = {}
    while cur is not None:
        out[cur[0]] = cur[1]
        cur = prev[cur]
    return out


def main(run, ref_path):
    A = json.loads((Path(run) / "analysis.json").read_text())
    R = json.loads(Path(ref_path).read_text())
    ref = R["rallies"]
    ours = our_rallies(A)
    m = align(ours, ref)
    names = R["players"]
    # player naming: identity or swapped, whichever agrees more on servers
    def swap(n):
        return {names[0]: names[1], names[1]: names[0]}.get(n, n)
    agree = sum(ours[i]["server"] == ref[j]["server"] for i, j in m.items())
    flip = sum(swap(ours[i]["server"]) == ref[j]["server"] for i, j in m.items())
    fix = (lambda n: swap(n)) if flip > agree else (lambda n: n)
    srv_ok = sum(fix(ours[i]["server"]) == ref[j]["server"] for i, j in m.items())
    win_ok = sum(fix(ours[i]["winner"]) == ref[j]["winner"] for i, j in m.items())
    extra = [ours[i] for i in range(len(ours)) if i not in m]
    missed = [ref[j] for j in range(len(ref)) if j not in m.values()]
    print(f"referee rallies: {len(ref)} | ours: {len(ours)} | matched: {len(m)}"
          f" | referee rallies we missed: {len(missed)} | our rallies with no referee rally: {len(extra)}")
    print(f"player names {'swapped once' if flip > agree else 'as given'}")
    print(f"server right: {srv_ok}/{len(m)} ({srv_ok / max(len(m), 1):.0%})   point winner right: {win_ok}/{len(m)} ({win_ok / max(len(m), 1):.0%})")
    dur = [abs((ours[i]["end_t"] - ours[i]["start_t"]) - (ref[j]["end_s"] - ref[j]["start_s"])) for i, j in m.items()]
    print(f"rally length difference: median {np.median(dur):.1f} s")
    games = Counter(r["game"] for r in ours)
    print(f"games found: {len(games)} (referee: {max(r['game'] for r in ref)})")
    # score from the serve sequence: points = rallies won by the server, per game
    for g in sorted(games):
        pts = Counter(fix(r["winner"]) for r in ours if r["game"] == g and r["winner"] == r["server"])
        print(f"  game {g}: points from tracking {dict(pts)}")
    print(f"  referee result: {R['result']}")
    if missed:
        print("missed referee rallies (wall clock):", [r["start_clock"] for r in missed][:20])
    if extra:
        print("our unmatched rallies (video min):", [round(r["start_t"] / 60, 2) for r in extra][:20])
    bad = [(round(ours[i]["start_t"] / 60, 2), ref[j]["start_clock"], fix(ours[i]["server"]), ref[j]["server"])
           for i, j in m.items() if fix(ours[i]["server"]) != ref[j]["server"]]
    if bad:
        print("server misses (video min, referee clock, ours, referee):", bad)
    out = {"matched": len(m), "ref": len(ref), "ours": len(ours), "server_ok": srv_ok, "winner_ok": win_ok,
           "missed": [r["start_clock"] for r in missed], "extra_min": [round(r["start_t"] / 60, 2) for r in extra],
           "pairs": {str(ours[i]["index"]): ref[j]["start_clock"] for i, j in m.items()}}
    (Path(run) / "referee_compare.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
