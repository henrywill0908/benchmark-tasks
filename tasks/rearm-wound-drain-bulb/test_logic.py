"""Scoring and demo-plan checks that need no simulator and no yam_arena (numpy and pytest only):

    python -m pytest -q test_logic.py
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import drain_logic as D

HERE = Path(__file__).parent
OPEN, SHUT = D.PLUG_OPEN, 0.0
FAR = [np.array([0.05, 0.30, D.TOP + 0.20]), np.array([0.05, 0.30, D.TOP + 0.187]),
       np.array([0.05, -0.30, D.TOP + 0.20]), np.array([0.05, -0.30, D.TOP + 0.187])]
HOLDING = [D.CENTRE, D.CENTRE + 0.013 * D.U, FAR[2], FAR[3]]  # the left gripper still at the bulb
FLAT = (-D.TRAVEL, D.TRAVEL)  # both faces pushed all the way in
ROUND = (0.0, 0.0)


def play(*steps):
    """Each step: (plug angle, latched, (left, right) face positions, gripper points, repeat). Returns the trial."""
    t = D.DrainTrial()
    for angle, latched, faces, points, n in steps:
        for _ in range(n):
            t.step(angle, latched, *faces, points)
    return t


def test_doing_nothing_scores_zero():
    assert play((OPEN, False, ROUND, FAR, 40)).score() == 0.0


def test_squeezing_without_capping_scores_zero():
    assert play((OPEN, False, FLAT, HOLDING, 40), (OPEN, False, ROUND, FAR, 20)).score() == 0.0


def test_capping_the_round_bulb_scores_a_quarter():
    assert play((OPEN, False, ROUND, FAR, 10), (SHUT, True, ROUND, FAR, 20)).score() == 0.25


def test_squeeze_then_cap_then_let_go_scores_full():
    t = play((OPEN, False, FLAT, HOLDING, 20), (SHUT, True, FLAT, HOLDING, 20), (SHUT, True, FLAT, FAR, 20))
    assert t.score() == 1.0 and t.done()


def test_still_holding_the_bulb_scores_half():
    t = play((OPEN, False, FLAT, HOLDING, 20), (SHUT, True, FLAT, HOLDING, 20))
    assert t.score() == 0.5 and not t.done()


def test_capping_first_then_squeezing_is_not_suction():
    # In the simulator the sealed bulb is held round; even if it read flat, it was round when the cap went on.
    t = play((SHUT, True, ROUND, FAR, 10), (SHUT, True, FLAT, HOLDING, 10), (SHUT, True, FLAT, FAR, 10))
    assert t.score() == 0.25


def test_half_squeezed_when_capped_is_not_enough():
    part = (-0.6 * D.TRAVEL, 0.6 * D.TRAVEL)
    assert play((OPEN, False, part, HOLDING, 10), (SHUT, True, part, FAR, 20)).score() == 0.25


def test_uncapping_undoes_the_seal_and_a_second_try_counts():
    t = play((OPEN, False, FLAT, HOLDING, 10), (SHUT, True, FLAT, HOLDING, 10), (OPEN, False, ROUND, FAR, 10))
    assert t.score() == 0.0 and not t.armed
    t2 = play((OPEN, False, ROUND, FAR, 5), (SHUT, True, ROUND, FAR, 5), (OPEN, False, ROUND, FAR, 5),
              (OPEN, False, FLAT, HOLDING, 10), (SHUT, True, FLAT, HOLDING, 10), (SHUT, True, FLAT, FAR, 10))
    assert t2.score() == 1.0


def test_the_seal_actions_fire_once_each_way():
    t = D.DrainTrial()
    assert t.step(OPEN, False, *FLAT, HOLDING) is None
    assert t.step(SHUT, True, *FLAT, HOLDING) == "seal"
    assert t.step(SHUT, True, *FLAT, HOLDING) is None
    assert t.step(OPEN, False, *FLAT, HOLDING) == "unseal"
    assert t.step(OPEN, False, *FLAT, HOLDING) is None


def test_a_pad_in_the_open_mouth_is_a_fault_that_sticks():
    tip_in_mouth = [FAR[0], FAR[1], D.MOUTH + np.array([0.0, 0.0, 0.016]), D.MOUTH + np.array([0.0, 0.0, 0.002])]
    t = play((OPEN, False, FLAT, tip_in_mouth, 1), (OPEN, False, FLAT, HOLDING, 10), (SHUT, True, FLAT, HOLDING, 10),
             (SHUT, True, FLAT, FAR, 20))
    assert t.contaminated and t.score() == 0.0


def test_pressing_the_seated_cap_is_not_touching_the_mouth():
    on_cap = [D.CENTRE, D.CENTRE, D.MOUTH + np.array([0, 0, D.PLUG_T + 0.013]), D.MOUTH + np.array([0, 0, 0.002])]
    t = play((OPEN, False, FLAT, HOLDING, 10), (SHUT, True, FLAT, on_cap, 10), (SHUT, True, FLAT, FAR, 20))
    assert not t.contaminated and t.score() == 1.0


def test_a_slipped_hold_scores_half():
    slipped = (-0.5 * D.TRAVEL, 0.5 * D.TRAVEL)
    t = play((OPEN, False, FLAT, HOLDING, 10), (SHUT, True, FLAT, HOLDING, 10), (SHUT, True, slipped, FAR, 20))
    assert t.score() == 0.5


def _swung(dx):
    """The bulb's centre and mouth moved dx metres sideways (along V)."""
    off = dx * D.V
    return D.CENTRE + off, D.MOUTH + off


def test_tugging_the_drain_is_a_fault_that_sticks():
    t = D.DrainTrial()
    centre, mouth = _swung(0.045)
    t.step(OPEN, False, *FLAT, HOLDING, centre, mouth)  # yanked 4.5 cm aside once
    for _ in range(10):
        t.step(OPEN, False, *FLAT, HOLDING)
    for _ in range(10):
        t.step(SHUT, True, *FLAT, HOLDING)
    for _ in range(20):
        t.step(SHUT, True, *FLAT, FAR)
    assert t.tugged and t.score() == 0.0 and not t.done()


def test_a_small_sway_is_allowed():
    centre, mouth = _swung(0.02)
    t = D.DrainTrial()
    for args in ((OPEN, False, FLAT, HOLDING, 10), (SHUT, True, FLAT, HOLDING, 10), (SHUT, True, FLAT, FAR, 20)):
        angle, latched, faces, points, n = args
        for _ in range(n):
            t.step(angle, latched, *faces, points, centre, mouth)
    assert not t.tugged and t.score() == 1.0


def test_the_mouth_and_hands_are_judged_where_the_bulb_is_now():
    centre, mouth = _swung(0.02)
    tip_in_mouth = [FAR[0], FAR[1], mouth + np.array([0.0, 0.0, 0.016]), mouth + np.array([0.0, 0.0, 0.002])]
    t = D.DrainTrial()
    t.step(OPEN, False, *FLAT, tip_in_mouth, centre, mouth)
    assert t.contaminated  # judged at the swung mouth, not where the mouth was at rest
    holding_swung = [centre, centre + 0.013 * D.U, FAR[2], FAR[3]]
    t2 = D.DrainTrial()
    t2.step(SHUT, True, *FLAT, holding_swung, centre, mouth)
    assert not t2.let_go()


def test_squeeze_and_thickness():
    assert D.squeeze(*ROUND) == 0.0 and D.squeeze(*FLAT) == 1.0
    assert D.squeeze(0.004, -0.004) == 0.0  # pulled outward counts as round, not negative
    assert D.thickness(*ROUND) == pytest.approx(0.048) and D.thickness(*FLAT) == pytest.approx(0.016)


def test_the_bulb_frame_is_turned_as_stated():
    assert np.allclose(D.U, [math.cos(D.YAW), math.sin(D.YAW), 0.0])
    assert abs(D.U @ D.V) < 1e-12


def test_the_cap_cannot_seat_under_its_own_weight():
    weight = D.PLUG_MASS * 9.81 * math.hypot(D.PLUG_AT[1], D.PLUG_AT[2])  # its largest closing torque
    spring_at_flat = D.PLUG_SPRING * D.PLUG_OPEN  # the tether pushing it open when flat
    assert spring_at_flat > 10 * weight
    push = spring_at_flat / (2 * D.PLUG_R)  # N, pressing at the cap's far edge
    assert push < 1.0  # a light press seats it


def test_the_layout_holds_together():
    assert D.PORT_Z - D.PORT[2] / 2 < D.BULB_Z + D.HALF_H < D.PORT_TOP  # port sits on the bulb's top
    assert D.BULB_Z - D.HALF_H > D.TOP  # bulb clear of the table
    assert D.CENTRE[0] + D.HALF_W < D.RAIL_X - D.RAIL["size"][0] / 2  # bulb in front of the rail
    assert D.PADDLE[2] / 2 < D.HALF_H and D.PADDLE[0] / 2 < D.HALF_W  # faces inside the visible body
    assert D.HALF_T - D.TRAVEL > D.PADDLE[1]  # flat faces don't meet
    assert 2 * D.HALF_T < 0.094 - 0.02  # the open gripper (94 mm) clears the round bulb with room to spare


# -- the demo plan --------------------------------------------------------------------------------------------------------
PLAN = D.demo_plan()


def _times(steps):
    out, t = [], 0.0
    for s in steps:
        t += s[0]
        out.append((t, s))
    return out


def test_the_demo_fits_the_run_limit_with_room():
    steps = json.loads((HERE / "task.json").read_text(encoding="utf-8"))["run_limit"]
    limit = int(steps.split()[0].replace(",", ""))
    assert (D.demo_seconds() + 1.0 + 1.0) * D.CONTROL_HZ < limit / 2


def test_the_right_arm_reaches_the_cap_only_after_the_bulb_is_flat():
    left_flat = _times(PLAN["left"])[2][0]
    assert left_flat <= D.SQUEEZED_AT + 1e-9
    descend_start = _times(PLAN["right"])[1][0]  # end of the wait = start of the descent
    assert descend_start >= left_flat - 1e-9


def test_the_left_arm_holds_until_the_right_arm_has_lifted_away():
    right_clear = _times(PLAN["right"])[-2][0]
    left_release_start = _times(PLAN["left"])[3][0]
    assert left_release_start >= right_clear - 1e-9


def test_the_sweep_clears_the_mouth_and_ends_past_the_port():
    _, sweep_start = _times(PLAN["right"])[2]
    _, sweep_end = _times(PLAN["right"])[3]
    for s in (sweep_start, sweep_end):
        tip = np.array(s[1]) - np.array([0.0, 0.0, D.PAD_TIP])
        assert not D.touches_mouth(tip)
        assert tip[2] - D.PORT_TOP > D.PLUG_T  # passes over the seated cap
    v0 = (np.array(sweep_start[1][:2]) - D.MOUTH[:2]) @ D.V[:2]
    v1 = (np.array(sweep_end[1][:2]) - D.MOUTH[:2]) @ D.V[:2]
    assert v0 < -0.025 and v1 > 0.0  # starts beside the open cap, ends past the hinge line


def test_both_arms_end_away_from_the_bulb():
    left_out = np.array(PLAN["left"][-2][1])
    right_up = np.array(PLAN["right"][-2][1])
    assert not D.at_bulb(left_out) and not D.at_bulb(left_out + D.PAD_TIP * D.U)
    assert not D.at_bulb(right_up) and not D.at_bulb(right_up - np.array([0, 0, D.PAD_TIP]))


def test_the_left_grip_is_square_to_the_bulb():
    for s in PLAN["left"]:
        if s[3] == "forward":
            off = np.array(s[1]) - D.CENTRE
            assert abs(off @ D.V) < 1e-9 and abs(off[2]) < 1e-9  # only ever along the approach axis
