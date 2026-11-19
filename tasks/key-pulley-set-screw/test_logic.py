"""Offline checks of pulley_logic (no simulator): the key's fit, the hub's gate, the screw and clutch, the driver's bit,
the score's levels and every demo variant played through the tracker in the kinematic stand-in (dryrun.py).

    python -m pytest -q test_logic.py test_task.py
"""

import math

import numpy as np
import pytest

import dryrun
import pulley_logic as L

ID = np.array([0.0, 0.0, 0.0, 1.0])
REST_POS = np.array([L.SEAT_CX, 0.0, L.AX_Z + L.SEAT_FLOOR_R + L.PAD_RISE + L.KEY[2] / 2])


def test_the_parts_have_the_standard_sizes():
    assert np.allclose(L.KEY, [0.025, 0.006, 0.006])  # ISO/R 773 6 x 6 for a 20 mm shaft
    assert math.isclose(L.SHAFT_R - L.SEAT_FLOOR_R, 3.5e-3) and math.isclose(L.KEYWAY_ROOF_R - L.BORE_R, 2.8e-3)
    assert math.isclose(L.KEYWAY_ROOF_R - L.KEY_HOME_TOP_R, 0.3e-3, abs_tol=1e-9)  # the key's top clears the hub
    assert L.SEAT_W_SIM - L.KEY_SIM[1] <= 0.5e-3  # the simulated slot: under 0.25 mm a side
    assert L.SEAT_X[1] - L.SEAT_X[0] > L.KEY_SIM[0] and L.SEAT_X[1] < L.SHOULDER_X
    assert math.isclose(L.TOUCH_KEYED, 270.0, abs_tol=1e-6) and L.TOUCH_BARE > 3 * 360  # bare shaft: almost 4 turns


def test_the_key_fits_only_home_and_square():
    assert not L.key_fit(L.KEY_START, ID).in_slot
    resting = L.key_fit(REST_POS, ID)
    assert resting.in_slot and not resting.home
    home = L.key_fit(*L.key_home_pose())
    assert home.in_slot and home.home and math.isclose(home.top_r, L.KEY_HOME_TOP_R)
    skew = L.matrix_to_quat_xyzw(L.rot_z(math.radians(5)))
    assert not L.key_fit(L.key_home_pose()[0], skew).home
    off = L.key_home_pose()[0] + (0.0, 0.8e-3, 0.0)
    assert not L.key_fit(off, ID).home


def _stop(fit):
    return fit.front_x - 0.5e-3 - L.REAR0


def test_the_hub_stops_at_the_key_unless_lined_up_and_the_key_is_home():
    home = L.key_fit(*L.key_home_pose())
    stop = _stop(home)
    assert L.gate(stop - 0.01, 0.0, home, 0.0, None) == L.Gate()  # short of the key: free
    g = L.gate(stop + 0.001, math.radians(-10), home, 0.0, None)
    assert g.block_at == pytest.approx(stop) and not g.jam  # keyway off the key: it butts on the key's end
    g = L.gate(stop + 0.001, math.radians(3.5), home, 0.0, None)
    assert g.snap and g.block_at is None  # within the 4 deg capture: it goes on
    g = L.gate(stop + 0.001, 0.0, home, 300.0, None)
    assert g.block_at is not None  # a screw run down into the keyway stops it too
    proud = L.key_fit(REST_POS, ID)
    g = L.gate(_stop(proud) + 0.0005, 0.0, proud, 0.0, None)
    assert g.block_at is not None and not g.jam
    assert L.gate(_stop(proud) + 0.0012, 0.0, proud, 0.0, g.block_at).jam  # forced on: the key cocks
    assert not L.gate(_stop(proud) + 0.0012, math.radians(20), proud, 0.0, g.block_at).jam  # off line: it only butts


def test_the_screw_touches_the_key_then_the_clutch_slips_at_snug():
    s = L.Screw()
    s.drive(269.0, True, L.SLIDE_LIMITS[1])
    assert not s.tight and L.screw_tip_r(s.psi) > L.KEY_HOME_TOP_R
    s.drive(30.0, True, L.SLIDE_LIMITS[1])
    assert not s.tight  # at the clutch's setting, not past it
    s.drive(20.0, True, L.SLIDE_LIMITS[1])
    assert s.tight and s.psi == pytest.approx(L.TOUCH_KEYED + L.SNUG_DEG)
    assert s.tight_where == (L.SLIDE_LIMITS[1], True)
    s.drive(-30.0, True, L.SLIDE_LIMITS[1])
    assert not s.tight  # backed off
    bare = L.Screw()
    bare.drive(360.0, False, 0.0)
    assert not bare.tight  # over bare shaft it has 3.1 mm to go before it touches


def test_turning_the_handle_clockwise_from_above_tightens():
    mouth, up = L.screw_mouth(L.SLIDE_LIMITS[1], 0.0), L.up_dir(0.0)
    q0 = L.matrix_to_quat_xyzw(L.rot_z(0.0))
    pos0, quat0, tw0 = L.seated_driver_pose(q0, mouth, up)
    tip, shank = L.driver_tip(pos0, quat0)
    assert np.allclose(tip, mouth - L.SOCKET_DEPTH * up) and L.engages(tip + 1.5e-3 * up, shank, mouth, up)
    _, _, tw1 = L.seated_driver_pose(L.matrix_to_quat_xyzw(L.rot_z(math.radians(-30))), mouth, up)
    assert -math.degrees(L.wrap(tw1 - tw0)) == pytest.approx(30.0)  # clockwise seen from above: +30 (in)
    tilted = L.matrix_to_quat_xyzw(L.rot_x(math.radians(12)))
    t2, s2 = L.driver_tip(pos0, tilted)
    assert not L.engages(t2, s2, mouth, up)


def test_the_score_levels():
    f = L.Facts()
    assert L.task_score(f) == 0.0
    f.key_home = True
    assert L.task_score(f) == 0.25
    f.keyed = f.seated = True
    assert L.task_score(f) == 0.5
    f.tight = True
    assert L.task_score(f) == 0.75
    f.driver_out, f.hand_clear, f.still = True, 0.05, True
    assert L.task_score(f) == 1.0
    f.hand_clear = 0.015
    assert L.task_score(f) == 0.75
    f.failed = "jam"
    assert L.task_score(f) == 0.0


def _inputs(**kw):
    base = dict(slide=0.0, turn=math.radians(L.PHI0), key_pos=np.array(L.KEY_START), key_quat=ID,
                driver_pos=np.array(L.BAR_START), driver_quat=ID, driver_ang_vel=np.zeros(3), pad=0.0,
                grippers=[np.array([0.0, 0.3, 1.2])], tcps=[(np.array([0.0, 0.3, 1.2]), 0.094)])
    base.update(kw)
    return L.Inputs(**base)


def test_the_tracker_needs_a_press_to_seat_the_key_and_then_welds_it():
    t = L.Tracker()
    assert t.step(_inputs()).hold_pad is True
    a = t.step(_inputs(key_pos=REST_POS, pad=-0.2e-3))
    assert a.hold_pad is None and not t.key_welded  # resting on the held pad: not pressed
    a = t.step(_inputs(key_pos=REST_POS - (0, 0, 0.6e-3), pad=-0.6e-3))
    assert a.hold_pad is False  # the press broke it free
    a = t.step(_inputs(key_pos=L.key_home_pose()[0], pad=-L.PAD_RISE))
    assert t.key_welded and a.key_pose is not None and t.facts.key_home


def test_a_key_on_the_hub_or_a_screw_without_the_key_scores_nothing():
    t = L.Tracker()
    on_hub = np.array([L.REAR0 - 0.008, 0.0, L.AX_Z + L.BOSS_R + 0.003])
    for _ in range(30):
        t.step(_inputs(key_pos=on_hub, slide=L.SLIDE_LIMITS[1], turn=0.0))
    assert L.task_score(t.facts) == 0.0 and not t.facts.key_home


def test_a_bit_resting_on_the_screw_drops_in_once_it_lines_up():
    up = L.up_dir(0.0)
    mouth = L.screw_mouth(L.SLIDE_LIMITS[1], 0.0)
    down = -up
    assert L.engages(mouth + 1.5e-3 * up, down, mouth, up)  # just above the socket, on its axis
    assert L.engages(mouth - 5.5e-3 * up, down, mouth, up)  # pushed onto the screw's face (round 2's no_key)
    assert not L.engages(mouth - 5.5e-3 * up + np.array([0.0, 2.5e-3, 0.0]), down, mouth, up)  # not lined up
    assert not L.engages(mouth + 4e-3 * up, down, mouth, up)  # still in the air
    tilted = L.rot_x(math.radians(12.0)) @ down
    assert not L.engages(mouth, tilted, mouth, up)


def test_the_tracker_drops_the_bit_when_the_handle_is_pulled_up():
    t = L.Tracker()
    mouth, up = L.screw_mouth(L.SLIDE_LIMITS[1], 0.0), L.up_dir(0.0)
    pos, quat, _ = L.seated_driver_pose(ID, mouth, up)
    grip = pos + (0, 0, L.BAR_GRIP_ABOVE)
    t.step(_inputs(slide=L.SLIDE_LIMITS[1], turn=0.0, driver_pos=pos, driver_quat=quat, tcps=[(grip, 0.016)]))
    assert t.engaged
    t.step(_inputs(slide=L.SLIDE_LIMITS[1], turn=0.0, driver_pos=pos, driver_quat=quat,
                   tcps=[(grip + (0, 0, 0.02), 0.030)]))
    assert t.engaged  # an open hand above it doesn't pull it
    t.step(_inputs(slide=L.SLIDE_LIMITS[1], turn=0.0, driver_pos=pos, driver_quat=quat,
                   tcps=[(grip + (0, 0, 0.008), 0.016)]))
    assert not t.engaged


@pytest.mark.parametrize("variant", sorted(L.VARIANTS))
def test_each_demo_variant_scores_as_designed_in_the_kinematic_stand_in(variant):
    assert dryrun.run(variant)["score"] == dryrun.EXPECT[variant]


def test_the_full_demo_fits_the_run_limit():
    assert L.plan_seconds(L.demo_plan("full")) + 2.0 < L.RUN_LIMIT_STEPS / L.CONTROL_HZ
