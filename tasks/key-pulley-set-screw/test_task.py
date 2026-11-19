"""Checks of task.py with the official yam_arena package (no GPU, no Isaac Sim): the task file loads, its parts are
where pulley_logic says, the shaft's slot is the keyseat, every demo waypoint (and points between) is reachable with
yam_arena's own IK, and the YAM finger envelope clears the shaft, the pulley and the stand where the demo needs it.

    python -m pytest -q test_logic.py test_task.py
"""

import math
from pathlib import Path

import numpy as np
import pytest
from yam_arena.task_file import load

import armcheck

HERE = Path(__file__).parent
TASK = load(HERE / "task.py")
M = TASK.module
L = M.L
MM = 1e-3


def test_the_task_file_loads_with_unique_names():
    assert TASK.success is not None and TASK.score is not None and TASK.demo is not None
    assert len(set(TASK.names)) == len(TASK.names)
    assert set(TASK.free_objects) == {"key", "driver"}
    for o in TASK.objects:
        o._check()


def test_the_instruction_states_the_goal_only():
    text = M.INSTRUCTION.lower()
    assert "pulley" in text and "set screw" in text
    for technique in ("key", "turn", "line", "press", "push", "torque", "clutch", "driver", "shoulder"):
        assert technique not in text


def test_the_docstring_names_every_object_the_policy_sees():
    doc = TASK.docs.lower()
    for word in ("shaft", "keyseat", "pulley", "set screw", "key", "driver", "cup", "mat"):
        assert word in doc


def test_nothing_scores_at_the_start():
    assert M.score({}) == 0.0 and not M.success({})


def test_the_shaft_pieces_leave_the_keyseat_slot_and_nothing_else():
    parts = {p.name: np.array(p.mesh[0]) + np.asarray(M.PULLEY.pos) for p in M.PULLEY.base_parts}
    left, right, low = parts["seat_left"], parts["seat_right"], parts["seat_low"]
    assert left[:, 1].min() == pytest.approx(L.SEAT_W_SIM / 2) and right[:, 1].max() == pytest.approx(-L.SEAT_W_SIM / 2)
    assert low[:, 2].max() == pytest.approx(L.AX_Z + L.SEAT_FLOOR_R - 1 * MM)
    for p in (left, right, low, parts["shaft_front"], parts["shaft_back"]):
        assert np.hypot(p[:, 1], p[:, 2] - L.AX_Z).max() <= L.SHAFT_R + 1e-6  # mesh points are rounded to 0.1 um
    assert parts["shaft_front"][:, 0].min() == pytest.approx(L.SHAFT_END)
    assert low[:, 0].min() == pytest.approx(L.SEAT_X_SIM[0]) and low[:, 0].max() == pytest.approx(L.SEAT_X_SIM[1])
    # the slot's top edges are broken: its wall stops LEAD_IN under where a square edge would meet the round
    wall = left[np.isclose(left[:, 1], L.SEAT_W_SIM / 2)]
    square_top = L.AX_Z + np.sqrt(L.SHAFT_R ** 2 - (L.SEAT_W_SIM / 2) ** 2)
    assert wall[:, 2].max() == pytest.approx(square_top - L.LEAD_IN, abs=1e-6)
    assert L.SEAT_X_SIM[1] - L.SEAT_X_SIM[0] - L.KEY_SIM[0] >= 2.0 * MM  # room for a key placed a millimetre off


def test_the_pad_holds_a_key_proud_and_lets_it_reach_the_floor():
    pad = M.PAD
    top0 = pad.pos[2] + pad.panel_offset[2] + pad.panel[2] / 2
    assert top0 == pytest.approx(L.AX_Z + L.SEAT_FLOOR_R + L.PAD_RISE)
    assert top0 + pad.limits[0] == pytest.approx(L.AX_Z + L.SEAT_FLOOR_R)
    bottom_low = top0 + pad.limits[0] - pad.panel[2]
    assert bottom_low >= L.AX_Z + L.SEAT_FLOOR_R - 1 * MM - 1e-9  # it never sinks into the slot's floor piece


def test_the_driver_hangs_in_its_stand():
    tip, _ = L.driver_tip(L.BAR_START, (0, 0, 0, 1))
    assert tip[2] > L.TOP + L.CUP_WALL + 1 * MM  # its bit clear of the cup's floor
    assert L.BAR_START[2] - L.BAR[2] / 2 == pytest.approx(L.TOP + L.CUP_H)  # the handle on the rim
    assert L.SHANK[0] < L.CUP_IN - 4 * MM


def test_the_pulley_starts_clear_of_the_keyseat_and_seats_on_the_shoulder():
    assert L.REAR0 < L.SEAT_X[0] - 0.02
    assert L.rear_x(L.SLIDE_LIMITS[1]) == pytest.approx(L.SHOULDER_X)
    assert L.rear_x(L.SLIDE_LIMITS[0]) - L.HUB_L == pytest.approx(L.SHAFT_END)
    assert L.SEAT_X[0] > L.SHOULDER_X - L.HUB_L  # the key ends under the seated hub
    mouth = L.screw_mouth(L.SLIDE_LIMITS[1], 0.0)
    assert L.SEAT_X[0] < mouth[0] < L.SEAT_X[1]  # the screw lands on the key


@pytest.mark.parametrize("variant", sorted(L.VARIANTS))
def test_every_demo_waypoint_is_reachable(variant):
    for arm, w in armcheck.worst(variant).items():
        assert w["pos_mm"] < 1.0 and w["rot_deg"] < 1.0, (arm, w)


# -- the YAM finger envelope (the same boxes as the other tasks' checks: pads, then the finger's wider body) ----------
def finger_boxes(tcp, rot, gap):
    out = []
    for side in (1.0, -1.0):
        for (x0, x1), half_y, (z0, z1) in (((0.0, 6.4), 13.7, (-5.0, 12.8)), ((1.0, 10.0), 20.1, (-20.0, -5.0)),
                                           ((2.2, 18.0), 27.6, (-40.0, -20.0))):
            lo = np.array([gap / 2 + x0 * MM, -half_y * MM, z0 * MM])
            hi = np.array([gap / 2 + x1 * MM, half_y * MM, z1 * MM])
            c = (lo + hi) / 2
            c[0] *= side
            out.append((np.asarray(tcp) + np.asarray(rot) @ c, np.asarray(rot), (hi - lo) / 2))
    return out


def box_samples(box, n=7):
    c, R, h = box
    g = np.linspace(-1, 1, n)
    return np.array([c + R @ (h * np.array([a, b, d])) for a in g for b in g for d in g])


def shaft_gap(points):
    """Clearance from points to the shaft (a round 20 mm bar from its end to the collar), ignoring the slot."""
    p = np.asarray(points)
    inside_x = (p[:, 0] > L.SHAFT_END) & (p[:, 0] < L.SHOULDER_X)
    return float(np.min(np.where(inside_x, np.hypot(p[:, 1], p[:, 2] - L.AX_Z) - L.SHAFT_R, 1.0)))


def _step(plan, arm, label):
    return next(s for s in plan[arm] if s[4] == label)


def test_the_key_hand_s_fingertips_stay_off_the_shaft():
    plan = L.demo_plan("full")
    for label, gap in (("set it in the seat", 0.0), ("let go", L.FINGER_GAP["wide"] * MM)):
        s = _step(plan, "right", label)
        pts = np.vstack([box_samples(b) for b in finger_boxes(s[1], s[3], max(gap, L.KEY[1]))])
        assert shaft_gap(pts) > 1.0 * MM, label
    s = _step(plan, "right", "press it home")
    pts = np.vstack([box_samples(b) for b in finger_boxes(s[1], s[3], 0.0)])
    tips_low = pts[:, 2].min()
    assert tips_low < L.AX_Z + L.KEY_HOME_TOP_R  # the press aims below home (the arm gives)


def test_the_rim_grip_stays_off_the_boss_and_the_shaft():
    for th in np.linspace(*L.TURN_ARC, 6):
        p, R = L.rim_grip(th)
        pts = np.vstack([box_samples(b) for b in finger_boxes(p, R, L.RIM_W + 2 * MM)])
        u = L.up_dir(math.radians(th))
        rel = pts - np.array([0.0, L.AX_Y, L.AX_Z])
        r = np.abs(rel[:, 1:] @ np.array([u[1], u[2]]))  # distance along the grip direction from the axis
        boss_side = pts[:, 0] > L.REAR0 - L.HUB_L + L.RIM_W  # behind the rim, where the boss is
        assert np.all(np.hypot(rel[boss_side, 1], rel[boss_side, 2]) > L.BOSS_R + 1 * MM) or not boss_side.any()
        assert shaft_gap(pts) > 2 * MM
        assert r.min() > L.SHAFT_R


def test_the_push_block_rides_over_the_shaft():
    plan = L.demo_plan("full")
    s = _step(plan, "left", "push it on to the shoulder")
    pts = np.vstack([box_samples(b) for b in finger_boxes(s[1], s[3], 0.0)])
    assert shaft_gap(pts) > 1 * MM
    face = s[1][0] + L.PAD_HALF
    assert face > L.SHOULDER_X - L.HUB_L  # it reaches past the rim's seated front face


def test_the_driver_hand_clears_the_stand_and_the_rim():
    plan = L.demo_plan("full")
    s = _step(plan, "left", "grip the handle")
    pts = np.vstack([box_samples(b) for b in finger_boxes(s[1], s[3], L.BAR[1])])
    cup_top = L.TOP + L.CUP_H
    half = L.CUP_IN / 2 + L.CUP_WALL
    near_cup = (np.abs(pts[:, 0] - L.HOLDER[0]) < half) & (np.abs(pts[:, 1] - L.HOLDER[1]) < half)
    assert np.all(pts[near_cup, 2] > cup_top + 0.5 * MM)
    s = _step(plan, "left", "bit in")
    for yaw in np.linspace(L.STROKE[1], L.STROKE[0], 7):
        pts = np.vstack([box_samples(b) for b in finger_boxes(s[1], L.rot_down(yaw), L.BAR[1])])
        assert pts[:, 2].min() > L.AX_Z + L.RIM_R + 5 * MM  # the turning hand passes over the rim
