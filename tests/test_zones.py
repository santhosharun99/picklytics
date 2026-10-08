"""Court zones: orientation at both ends, boundaries, outside positions, uncertainty (pickleball/zones.py)."""
from pickleball.zones import locate, zone_name


def test_orientation_near_and_far_ends_mirror():
    # near-side hitter (faces the far end): x small = their left
    assert locate(2, 40, hitter_near=True).name == "deep left"
    # far-side hitter (faces the near end): the same court x is on their RIGHT
    assert locate(2, 4, hitter_near=False).name == "deep right"
    # destination on the far half for a near hitter: x small = hitter's left
    assert locate(2, 4, hitter_near=True).name == "deep left"
    assert locate(2, 4, hitter_near=True).half == "opponent"


def test_end_change_uses_the_hitters_own_end():
    # same player, same court point after changing ends: zone follows the hitter's end, not the match
    before = locate(15, 30, hitter_near=True)
    after = locate(5, 14, hitter_near=False)   # the mirrored point after the end change
    assert before.name == after.name == "midcourt right"


def test_boundaries_half_open_and_lines_in():
    assert zone_name(5, 7.0) == "midcourt left"          # 7 ft belongs to midcourt
    assert zone_name(5, 6.999) == "kitchen left"
    assert zone_name(20 / 3, 3) == "kitchen centre"       # 20/3 belongs to centre
    assert zone_name(40 / 3, 3) == "kitchen right"
    assert zone_name(20.0, 22.0) == "deep right"          # sideline and baseline are in
    assert zone_name(0.0, 14.5) == "deep left"


def test_outside_positions_not_clamped():
    z = locate(10, 47, hitter_near=True)                  # hitter 3 ft behind the near baseline
    assert z.name == "behind centre" and not z.inside and z.half == "own"
    z = locate(21.5, 5, hitter_near=True)                 # estimated landing wide of the sideline
    assert z.name == "deep wide-right" and not z.inside


def test_uncertainty_spanning_zones_is_boundary():
    z = locate(6.5, 15.0, hitter_near=True, sd=0.8)      # near the left/centre line and the kitchen line
    assert z.name == "boundary" and len(z.candidates) > 1
    z = locate(3, 3, hitter_near=True, sd=0.5)           # well inside one zone
    assert z.name == "deep left" and z.candidates == ["deep left"]
