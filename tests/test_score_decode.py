"""Evidence-fusion scoring decoder (pickleball/score_decode.py): game rules, end constraint, missed points,
overlay anchors (visible / hidden / late / catch-up bursts)."""
import random

from pickleball.score_decode import Game, decode, summarize, overlay_segments, assign_anchors, parse_official


def simulate(game, seed=0, p_server=0.55):
    """A random game played to the end: [(state before, winner, next state)]."""
    rnd = random.Random(seed)
    s = rnd.choice(game.starts())
    pts = []
    while not game.over(s):
        opts = game.succ(s)
        won = [o for o in opts if o[1] == s[2]]
        lost = [o for o in opts if o[1] != s[2]]
        nxt = rnd.choice(won) if rnd.random() < p_server else rnd.choice(lost)
        pts.append((s, nxt[1], nxt[0]))
        s = nxt[0]
    return pts


def events_from(game, pts, n0=0, keep=None, serve_rel=0.9, cue_ok=None):
    """Detected rallies with clean tracking cues (kit = team): one every 25 s; keep = point positions detected."""
    evs = []
    for k, (s, w, _) in enumerate(pts):
        if keep is not None and k not in keep:
            continue
        T = s[2]
        ok = cue_ok(k) if cue_ok else True
        evs.append({"index": k, "start_t": 30 + 25 * k, "end_t": 38 + 25 * k,
                    "serve_kit": T if ok else 1 - T, "serve_rel": serve_rel, "cue": "formation",
                    "near_kit": n0 if not game.switched(s) else 1 - n0, "near_rel": 0.95,
                    "server_player": (T, s[4]) if game.players else None, "player_rel": 0.8})
    return evs


def segs_from(pts, times=None, lag_s=4.0, team=True):
    """Overlay states as overlay_segments would give them: the state after each point shows lag_s after the
    point's rally ends (times: point positions where the overlay is visible)."""
    out = [{"score": (0, 0), "team": pts[0][0][2] if team else None, "no": None, "p": None, "name": None,
            "t0": 5.0, "t1": 25.0, "n": 5}]
    for k, (_, _, s2) in enumerate(pts):
        if times is not None and k not in times:
            continue
        t0 = 38 + 25 * k + lag_s
        out.append({"score": (s2[0], s2[1]), "team": s2[2] if team else None, "no": None, "p": None, "name": None,
                    "t0": t0, "t1": t0 + 10, "n": 5})
    return out


def run(game, evs, segs, final, flips=(0,)):
    return summarize(decode(game, evs, segs, final, flips=flips, video_end=evs[-1]["end_t"] + 30 if evs else 60))


def test_rules_sideout_doubles():
    g = Game("sideout", 11, 1, doubles=True)
    assert set(g.starts()) == {(0, 0, T, 2, p) for T in (0, 1) for p in (0, 1)}  # 0-0-2
    s = (3, 2, 0, 1, 0)
    nxt = {(o[2], o[0]) for o in g.succ(s)}
    assert ("point", (4, 2, 0, 1, 0)) in nxt            # only the server scores
    assert ("server 1 lost", (3, 2, 0, 2, 1)) in nxt    # partner serves as server 2, no score change
    side = {o[0] for o in g.succ((3, 2, 0, 2, 1))}
    assert (3, 2, 1, 1, 0) in side and (3, 2, 1, 1, 1) in side  # side out: other team's first server
    assert all(o[0][0] >= 3 and o[0][1] >= 2 for o in g.succ(s))  # scores never go down
    assert g.over((11, 9, 0, 1, 0)) and not g.over((10, 10, 0, 1, 0))
    assert g.score_text((4, 7, 1, 2, 0)) == "7-4-2"
    assert g.switched((6, 2, 0, 1, 0)) and not g.switched((5, 5, 0, 1, 0))


def test_rules_rally_scoring():
    g = Game("rally", 25, 1, doubles=True)
    nxt = {o[0]: o[1] for o in g.succ((10, 12, 1, 0, 0))}
    assert nxt == {(10, 13, 1, 0, 0): 1, (11, 12, 0, 0, 0): 0}  # winner scores and serves next
    assert g.mid == 13


def test_parse_official():
    assert parse_official("Kodai Tigers won 11-5 (match card)", ["Kodai Tigers", "Rockfort Terminatrz"]) == (11, 5)
    assert parse_official("Rockfort Terminatrz won 25-10 (match card)", ["Kodai Tigers", "Rockfort Terminatrz"]) == (10, 25)
    assert parse_official("Kannan (Twin Eagles Hosur) won 11-7 (match card)", ["Rishi Raghavendra", "Kannan"]) == (7, 11)
    assert parse_official(None, ["a", "b"]) is None


def _check(out, pts, keep=None):
    by = {r["index"]: r for r in out["rallies"]}
    for k, (s, w, _) in enumerate(pts):
        if keep is not None and k not in keep:
            continue
        assert by[k]["role"] == "point"
        assert by[k]["winner_team"] == w, (k, by[k], s, w)
        assert by[k]["server_team"] == s[2]


def test_clean_tracking_no_overlay():
    for scoring, target, doubles in (("sideout", 11, True), ("sideout", 11, False), ("rally", 25, True)):
        g = Game(scoring, target, 1, doubles)
        pts = simulate(g, seed=3)
        out = run(g, events_from(g, pts), [], pts[-1][2][:2])
        _check(out, pts)
        assert out["missed"] == []


def test_final_score_is_a_hard_constraint():
    g = Game("sideout", 11, 1, True)
    pts = simulate(g, seed=5)
    out = run(g, events_from(g, pts), [], pts[-1][2][:2])
    last = [r for r in out["rallies"] if r["role"] == "point"][-1]
    sc = list(last["score"])
    sc[last["winner_team"]] += last["how"] == "point"
    assert tuple(sc) == pts[-1][2][:2]


def test_missed_points_are_explicit():
    g = Game("sideout", 11, 1, True)
    pts = simulate(g, seed=7)
    drop = {4, 5, 12}  # cut-aways: these points were never seen by the cameras
    keep = set(range(len(pts))) - drop
    out = run(g, events_from(g, pts, keep=keep), segs_from(pts), pts[-1][2][:2])
    _check(out, pts, keep)
    assert len(out["missed"]) == len(drop)
    assert all(m["source"] == "overlay" for m in out["missed"])


def test_missed_points_without_overlay_reach_the_final_score():
    g = Game("rally", 25, 1, True)
    pts = simulate(g, seed=11)
    keep = set(range(len(pts) - 3))  # the last three points not filmed, overlay gone
    out = run(g, events_from(g, pts, keep=keep), [], pts[-1][2][:2])
    assert len(out["missed"]) == 3
    assert all(m["source"] == "rules" for m in out["missed"])


def test_overlay_beats_noisy_tracking():
    g = Game("sideout", 11, 1, True)
    pts = simulate(g, seed=2)
    rnd = random.Random(1)
    flips = {k for k in range(len(pts)) if rnd.random() < 0.3}  # 30% of serve cues wrong
    evs = events_from(g, pts, serve_rel=0.7, cue_ok=lambda k: k not in flips)
    out = run(g, evs, segs_from(pts), pts[-1][2][:2])
    _check(out, pts)
    pts_r = [r for r in out["rallies"] if r["role"] == "point"]
    assert sum(r["provenance"] == "overlay" for r in pts_r) >= 0.8 * len(pts_r)
    assert all(r["confidence"] == "certain" for r in pts_r if r["provenance"] == "overlay")


def test_overlay_hidden_in_parts_and_late():
    g = Game("sideout", 11, 1, True)
    pts = simulate(g, seed=4)
    vis = set(range(len(pts) // 3, len(pts)))  # overlay missing for the first third of the match
    out = run(g, events_from(g, pts), segs_from(pts, times=vis, lag_s=12.0), pts[-1][2][:2])  # and updated late
    _check(out, pts)
    early = [r for r in out["rallies"] if r["index"] < len(pts) // 3 - 1]
    assert all(r["provenance"] != "overlay" for r in early)


def test_overlay_disagreeing_with_final_is_not_trusted_blindly():
    g = Game("rally", 25, 1, True)
    pts = simulate(g, seed=9)
    segs = segs_from(pts)
    bad = segs[len(segs) // 2]
    bad["score"] = (bad["score"][0] + 3, bad["score"][1])  # one misread overlay state
    out = run(g, events_from(g, pts), segs, pts[-1][2][:2])
    _check(out, pts)
    assert out["missed"] == []


def test_kit_flip_decided_without_overlay():
    g = Game("sideout", 11, 1, True)
    pts = simulate(g, seed=12)
    evs = events_from(g, pts)
    for e in evs:  # tracking calls team 0 'kit 1'
        e["serve_kit"] = 1 - e["serve_kit"]
        e["near_kit"] = 1 - e["near_kit"]
        e["server_player"] = None
    out = run(g, evs, [], pts[-1][2][:2], flips=(0, 1))
    assert out["flip"] == 1
    # without names the server-1-lost placement is ambiguous; the serving teams must be right
    by = {r["index"]: r for r in out["rallies"]}
    assert all(by[k]["server_team"] == s[2] for k, (s, _, _) in enumerate(pts))


def test_overlay_segments_hide_and_bursts():
    R = []
    def add(t0, t1, sc, name="RAHUL", no=1):
        for t in range(t0, t1):
            R.append({"t": float(t), "scores": list(sc), "conf": 0.99, "server": name, "serve_no": no, "fit": list(sc)})
    add(0, 30, (0, 0)); add(30, 60, (1, 0)); add(60, 63, (2, 0)); add(63, 66, (3, 0)); add(66, 120, (3, 1), "RAM")
    squads = [["Rahul"], ["Ram"]]
    segs = overlay_segments(R, squads)
    assert [s["score"] for s in segs] == [(0, 0), (1, 0), (3, 1)]  # 2-0 / 3-0 typed in late: burst
    assert segs[0]["team"] == 0 and segs[-1]["team"] == 1
    hidden = overlay_segments(R, squads, hide=[(20, 70)])
    assert [s["score"] for s in hidden] == [(0, 0), (3, 1)]
    pre, mid = assign_anchors([{"start_t": 40, "end_t": 50}, {"start_t": 100, "end_t": 110}], segs)
    assert pre[0]["score"] == (1, 0) and pre[1]["score"] == (3, 1)
