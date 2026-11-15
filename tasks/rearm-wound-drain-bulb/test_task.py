"""Checks of task.py with the official yam_arena package (no GPU, no Isaac Sim): the task file loads, the documents
agree, nothing leaks the technique to the policy, and every demo waypoint is reachable with yam_arena's own IK.

    python -m pytest -q test_task.py test_logic.py
"""

import json
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik
from yam_arena.constants import READY
from yam_arena.demo import grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import load

HERE = Path(__file__).parent
TASK = load(HERE / "task.py")
M = TASK.module


def test_the_task_file_loads_with_unique_names_and_nothing_loose():
    assert TASK.success is not None and TASK.score is not None and TASK.demo is not None
    assert len(set(TASK.names)) == len(TASK.names)
    assert TASK.free_objects == ()  # everything is fixed or jointed, so object_jitter changes nothing


def test_doing_nothing_scores_zero():
    M._new_trial()
    assert M.score({}) == 0.0 and not M.success({})


def test_the_instruction_is_the_same_everywhere():
    meta = json.loads((HERE / "task.json").read_text(encoding="utf-8"))
    assert meta["instruction"] == M.INSTRUCTION
    assert M.INSTRUCTION in (HERE / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("word", ["squeeze", "flat", "press", "compress", "vacuum", "before", "while"])
def test_neither_instruction_nor_policy_docs_give_the_technique_away(word):
    assert word not in M.INSTRUCTION.lower()
    assert word not in (M.__doc__ or "").lower()


def test_the_text_fields_fit_the_upload_limits():
    fields = json.loads((HERE / "task.json").read_text(encoding="utf-8"))
    for key, limit in (("title", 120), ("instruction", 1000), ("context", 1000), ("setup", 4000),
                       ("run_limit", 200), ("fidelity", 500), ("scores", 300)):
        assert len(fields[key]) <= limit, key


B = M.BULB


def _turn_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _core_to_world(p):
    """A point in the bulb's core frame at rest, in the world."""
    c, s = np.cos(B.yaw), np.sin(B.yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.asarray(B.pos, float) + R @ np.asarray(p, float)


def test_the_cap_starts_open_on_its_tether_and_the_bulb_round():
    assert B.cap_start == M.D.PLUG_OPEN > np.pi / 2 and B.cap_rest == B.cap_start  # held open by its tether at rest
    closing_weight = B.cap_mass * 9.81 * np.hypot(B.cap_at[1], B.cap_at[2])  # the cap's weight when flat (N m)
    opening_spring = B.cap_spring * (B.cap_rest - 0.0)  # the tether pushing it open when flat (N m)
    assert opening_spring > 10 * closing_weight  # it can't seat under its own weight


def test_the_bulb_hangs_where_the_scorer_expects():
    from labkit import start_state

    s = start_state(B)
    assert np.allclose(s.centre, M.D.CENTRE, atol=1e-9) and np.allclose(s.mouth, M.D.MOUTH, atol=1e-9)
    assert s.squeeze_left == 0.0 and s.squeeze_right == 0.0 and s.cap_angle == B.cap_start


def test_the_seated_cap_covers_the_port():
    centre = _core_to_world(B.cap_hinge_point + np.asarray(B.cap_at))
    assert np.linalg.norm(centre[:2] - M.D.MOUTH[:2]) < 0.004
    assert centre[2] == pytest.approx(M.D.PORT_TOP + M.D.PLUG_T / 2, abs=1e-6)


def test_the_open_cap_leans_toward_the_right_arms_side():
    centre = _core_to_world(B.cap_hinge_point + _turn_x(B.cap_start) @ np.asarray(B.cap_at))
    assert (centre[:2] - M.D.MOUTH[:2]) @ M.D.V[:2] < -0.005  # toward -V, where the sweep starts


def test_a_one_newton_push_sways_the_bulb_less_than_the_tug_limit():
    sway = M.D.SWING_DROP * (1.0 * M.D.SWING_DROP) / M.D.SWING_SPRING  # small-angle: lever x torque / stiffness
    assert sway < M.D.TUG_LIMIT


def _rotation(w):
    rot = getattr(w, "rot", None)
    return rot if rot is not None else grip_rotation(w.yaw)


@pytest.mark.parametrize("arm", ["left", "right"])
def test_every_demo_waypoint_is_reachable(arm):
    q = np.array(READY, float)
    for w in TASK.demo()[arm]:
        q, pos_err, rot_err = ik.solve(np.asarray(w.pos, float), ik.matrix_to_quat_xyzw(_rotation(w)), q,
                                       base_pos=BASES[arm])
        assert pos_err < 1e-3 and rot_err < 1e-2, (arm, w.label, pos_err, rot_err)


def test_the_demo_fits_the_run_limit():
    total = max(sum(w.duration for w in ws) for ws in TASK.demo().values())
    assert (total + M.SUCCESS_HOLD_S + 1.0) * M.CONTROL_HZ < 1200
