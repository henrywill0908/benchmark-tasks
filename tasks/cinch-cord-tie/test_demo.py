"""Checks of demo.py's draft pieces with the official yam_arena IK (no GPU): the hold the jig stands in for is
reachable and leaves the slot's faces open; the phases borrowed from the probe stay reachable.

    python -m pytest -q test_demo.py
"""

import numpy as np

import demo as D
import tie_logic as L
from yam_arena import ik
from yam_arena.constants import READY
from yam_arena.station import BASES


def _reach(arm, pos, rot, q=None):
    q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot),
                         np.array(READY, float) if q is None else q, base_pos=BASES[arm])
    return q, pe, re


def test_the_hold_from_behind_is_reachable_at_the_working_pose():
    tcp, rot = D.hold_pose()
    _, pe, re = _reach("right", tcp, rot)
    assert pe < 1e-3 and re < 1e-2


def test_the_hold_from_behind_leaves_both_slot_faces_open():
    gaps = D.hold_clears_the_slot()
    assert gaps["entry"] > 1.0 * L.MM and gaps["exit"] > 1.0 * L.MM


def test_the_hold_from_behind_is_not_reachable_where_the_tie_is_picked():
    """Why demo.py regrips near the cords: from behind, at the pick spot, yam_arena's IK misses."""
    b = L.BUILDS["free"]
    lifted = L.Build("lifted", False, tuple(np.asarray(b.head_pos) + [0, 0, L.LIFT]), b.head_rot, b.points)
    tcp, rot = D.hold_pose(lifted)
    _, pe, _ = _reach("right", tcp, rot)
    assert pe > 2e-3


def test_the_borrowed_phases_are_reachable():
    for arm, steps in (("right", D.pick_steps()), ("left", D.insert_steps())):
        q = np.array(READY, float)
        for s in steps:
            if isinstance(s[3], str) or s[1] is None:
                q = np.array(READY, float)
                continue
            q, pe, re = _reach(arm, s[1], s[3], q)
            assert pe < 1e-3 and re < 1e-2, (arm, s[4], pe)
