"""Player depth correction (pickleball/depth.py): one model, calibrated by the server-stance or kitchen-line rule."""
from pickleball import depth

CAM = (10.0, 63.0, 9.0)   # TNPPL-like: 19 ft behind the near baseline, 9 ft up


def _raw(xy, h):
    """the raw (tracked) position of a player truly at xy, for a foot point at effective height h"""
    return depth.uncorrect(xy, {"camera": CAM, "h_ft": h})


def test_correct_and_uncorrect_are_inverse():
    D = {"camera": CAM, "h_ft": 0.5}
    for xy in [(3.0, 2.0), (10.0, 15.0), (17.0, 30.0), (8.0, 45.0)]:
        x, y = depth.correct(depth.uncorrect(xy, D), D)
        assert abs(x - xy[0]) < 0.01 and abs(y - xy[1]) < 0.01


def test_kitchen_stance_is_the_shallower_partner_when_one_up_one_back():
    up, back = (5.0, 29.5), (14.0, 45.0)              # near end: 7.5 ft and 23 ft from the net
    assert depth.kitchen_stance([back, up]) == up
    assert depth.kitchen_stance([(5.0, 29.5), (14.0, 31.0)]) is None   # both up: no stance
    assert depth.kitchen_stance([up]) is None


def test_kitchen_rule_recovers_the_height():
    # kitchen-line waiters truly 8 ft from the net at both ends, tracked with a foot point 0.55 ft up
    st = [(_raw((x, 30.0), 0.55), CAM) for x in (4.0, 6.0, 8.0) * 4] + [(_raw((x, 14.0), 0.55), CAM) for x in (12.0, 14.0, 16.0) * 4]
    K = depth.calibrate_kitchen(st)
    assert abs(K["h_ft"] - 0.55) < 0.01
    assert K["from_net_ft"]["raw"]["far"] > K["from_net_ft"]["raw"]["near"] + 3      # far end reads too deep
    assert abs(K["from_net_ft"]["corrected"]["near"] - K["from_net_ft"]["corrected"]["far"]) < 0.05
    assert depth.calibrate_kitchen(st[:5])["h_ft"] is None                          # too few to calibrate


def test_kitchen_rule_is_the_fallback_for_implausible_server_stances():
    servers = [(10.0, 30.0)] * 10 + [(10.0, -1.0)] * 10   # "server" 8 ft inside the near court: not a server
    K = {"h_ft": 0.55}
    D = depth.calibrate(servers, CAM, K)
    assert D["method"] == "kitchen line" and D["h_ft"] == 0.55
    assert depth.calibrate(servers, CAM)["method"] == "default (server stances implausible)"
    good = [_raw((10.0, 45.5), 0.5)] * 10 + [_raw((10.0, -1.5), 0.5)] * 10
    D = depth.calibrate(good, CAM, K)                     # the server rule holds: it wins
    assert D["method"] == "rules" and abs(D["h_ft"] - 0.5) < 0.01


def test_apply_moves_stored_positions_to_the_run_model(tmp_path):
    old = {"camera": list(CAM), "h_ft": 0.3}
    new = {"camera": list(CAM), "h_ft": 0.6}
    raw = (10.0, 10.0)
    A = {"depth_correction": old, "shots": [{"hitter_court": list(depth.correct(raw, old)), "hitter_zone": "x"}],
         "positions": [{"t": 0, "p": {"P1": list(depth.correct(raw, old))}}]}
    assert depth.apply(A, tmp_path)["shots"][0]["hitter_court"] == A["shots"][0]["hitter_court"]  # no depth.json: unchanged
    (tmp_path / "depth.json").write_text(__import__("json").dumps(new))
    depth.apply(A, tmp_path)
    want = depth.correct(raw, new)
    assert abs(A["shots"][0]["hitter_court"][1] - want[1]) < 0.06
    assert A["depth_correction"] == new and A["depth_correction_analysis"] == old
    assert A["shots"][0]["hitter_zone"] != "x"


def test_pooled_server_rule_finds_one_height_for_a_rig():
    # servers stand 2 ft behind the baseline at both ends; raw positions made with h = 0.5 from one camera
    from pickleball.depth import _servers_pooled, uncorrect
    cam = [10.0, 63.0, 9.0]
    D = {"camera": cam, "h_ft": 0.5}
    near = [(uncorrect((x, 46.0), D), cam) for x in (3, 5, 8, 12, 15, 17, 6, 14)]
    far = [(uncorrect((x, -2.0), D), cam) for x in (3, 5, 8, 12, 15, 17, 6, 14)]
    r = _servers_pooled(near + far)
    assert abs(r["h_ft"] - 0.5) < 0.02
    assert _servers_pooled(near[:3] + far[:3])["h_ft"] is None      # too few serves per end
