"""Outgoing-flight status of a shot (pickleball/shot_events.resolve): bounce / interception / net / unresolved."""
from pickleball.shot_events import resolve

NEAR = True  # hitter at the near end -> the opponent's half is y < 22


def b(frame, y, x=10.0, src="physics fit"):
    return {"kind": "bounce", "frame": frame, "x": x, "y": y, "src": src}


def test_regression_fast_shot_volleyed_at_kitchen_is_interception_not_kitchen_landing():
    # 42 mph drive, the opponent volleys it at their kitchen line: no bounce in the window, dense tracking
    status, bounce, _ = resolve("later", NEAR, [], has_next=True, next_volley=True, coverage_ok=True)
    assert status == "intercepted" and bounce is None   # interception stats, never an observed landing


def test_bounce_then_opponent_contact_is_bounced_with_first_bounce():
    status, bounce, _ = resolve("later", NEAR, [b(30, 4.0), b(40, 2.0)], has_next=True, next_volley=False, coverage_ok=True)
    assert status == "bounced" and bounce["frame"] == 30


def test_missed_bounce_does_not_become_a_volley():
    # no bounce seen but tracking sparse -> unresolved, even if the next hit is flagged a volley
    assert resolve("later", NEAR, [], True, True, coverage_ok=False)[0] == "unresolved"
    # no bounce seen, dense tracking, but the next contact is not a volley -> unresolved
    assert resolve("later", NEAR, [], True, False, coverage_ok=True)[0] == "unresolved"


def test_serve_and_return_cannot_be_intercepted():
    for role in ("serve", "return"):
        assert resolve(role, NEAR, [], True, True, True)[0] == "unresolved"


def test_net_first_no_extrapolation():
    evs = [{"kind": "net", "frame": 12, "x": 10.0, "y": 22.0, "src": "physics fit"}, b(20, 26.0)]
    status, bounce, _ = resolve("later", NEAR, evs, True, False, True)
    assert status == "net" and bounce is None


def test_own_half_bounce_without_net_is_unresolved():
    assert resolve("later", NEAR, [b(15, 30.0)], True, False, True)[0] == "unresolved"


def test_last_shot_nothing_observed_is_unresolved():
    assert resolve("later", NEAR, [], has_next=False, next_volley=None, coverage_ok=True)[0] == "unresolved"


def test_far_side_hitter_opponent_half_is_near():
    assert resolve("later", False, [b(30, 35.0)], True, False, True)[0] == "bounced"


def test_net_cord_then_opponent_bounce_is_bounced_with_flag():
    evs = [{"kind": "net", "frame": 12, "x": 10.0, "y": 22.0, "src": "physics fit"}, b(20, 18.0)]
    status, bounce, _ = resolve("later", NEAR, evs, True, False, True)
    assert status == "bounced" and bounce["net_cord"] and bounce["frame"] == 20


def test_net_then_own_half_bounce_stays_net():
    evs = [{"kind": "net", "frame": 12, "x": 10.0, "y": 22.0, "src": "physics fit"}, b(20, 25.0)]
    assert resolve("later", NEAR, evs, True, False, True)[0] == "net"
