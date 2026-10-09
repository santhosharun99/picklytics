"""Doubles individual / partnership helpers (pickleball/doubles_players.py)."""
import numpy as np

from pickleball.doubles_players import SWITCH_HOLD_BINS, _resample, _switch, own


def test_own_half_is_seen_from_the_player():
    # near-end player (y > 22) faces the far end: x is their left-right as is
    assert own(2, 40) == (2, 18, True)
    # far-end player faces the near end: the same x is on their right
    u, d, near = own(2, 4)
    assert (u, d, near) == (18, 18, False)


def test_side_switch_needs_a_held_change():
    a, b = "A", "B"
    steady = [(k, a) for k in range(20)]
    assert _switch(steady) is None
    blip = [(k, a) for k in range(10)] + [(10, b)] + [(k, a) for k in range(11, 20)]
    assert _switch(blip) is None                       # one bin on the other side is noise
    real = [(k, a) for k in range(10)] + [(k, b) for k in range(10, 10 + SWITCH_HOLD_BINS + 2)]
    assert _switch(real) == 10


def test_resample_to_ten_hertz():
    s = np.array([[t / 30, 1.0, 2.0] for t in range(30)])     # 1 s at 30 fps
    r = _resample(s)
    assert len(r) == 10 and all(abs(x[1] - 1.0) < 1e-9 for x in r)
