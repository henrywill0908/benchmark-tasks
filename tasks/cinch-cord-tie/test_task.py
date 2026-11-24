"""Checks of task.py and labkit's CableTie with the official yam_arena package (no GPU, no Isaac Sim): the task file
loads in both builds, the tie's links chain from the head to the tip, its state reads back, the ratchet hook brakes and
lets go as specified, and every probe waypoint (and the steering envelope round the insert) is reachable with
yam_arena's own IK.

    python -m pytest -q test_logic.py test_task.py
"""

import dataclasses
import importlib
import itertools
import json
import math
import os
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
L = M.L


def test_the_task_file_loads_with_unique_names_and_nothing_loose():
    assert TASK.success is not None and TASK.score is not None and TASK.demo is not None
    assert len(set(TASK.names)) == len(TASK.names) and "tie" in TASK.names
    assert TASK.free_objects == ()  # room and cords fixed; the tie is an articulation labkit resets


def test_the_instruction_states_the_goal_only():
    text = M.INSTRUCTION.lower()
    assert "snug" in text and "let go" in text
    for technique in ("thread", "slot", "head", "tail", "hold", "pawl", "pull the"):
        assert technique not in text


@pytest.mark.parametrize("build", ["free", "jig", "tight", "lift"])
def test_both_builds_make_a_valid_tie_that_starts_where_tie_logic_says(build):
    from labkit import start_state

    tie = M.build_tie(build)
    tie._check()
    b = L.BUILDS[build]
    s = start_state(tie)
    assert np.allclose(s.polyline, L.start_polyline(b), atol=1e-12)
    assert np.allclose(s.pos, b.head_pos) and math.isclose(s.pawl_angle, L.PAWL_REST)
    assert M.score({"tie": s}) == 0.0 and not M.success({"tie": s})
    assert 1 + 1 + tie.n_links <= 64


def test_each_link_chains_from_the_head_to_the_next():
    tie = M.build_tie("jig")
    frames, c, half = tie.link_frames(), tie.link_centres(), tie.link_length() / 2
    x = np.array([f[:, 0] for f in frames])
    assert np.allclose(c[0] - x[0] * half, tie.points[0])
    assert np.allclose(c[:-1] + x[:-1] * half, c[1:] - x[1:] * half)
    for a, b in zip(frames[:-1], frames[1:]):
        rel = a.T @ b  # a turn about y only: the strap bends like a ribbon
        assert np.allclose(rel[:, 1], [0.0, 1.0, 0.0], atol=1e-12) and np.isclose(np.linalg.det(rel), 1.0)
        assert math.degrees(math.acos(np.clip(rel[0, 0], -1, 1))) < math.degrees(L.MAX_BEND)


def test_the_state_reads_links_back_into_the_same_polyline():
    from labkit import start_state

    tie = M.build_tie("jig")
    r = tie.head_rotation()
    q = np.array([ik.matrix_to_quat_xyzw(r @ f) for f in tie.link_frames()])
    c = tie.link_centres() @ r.T + np.asarray(tie.pos)
    s = tie.state_at(tie.pos, tie.quat, (c, q), pawl=0.3, link_speed=0.02)
    assert np.allclose(s.polyline, start_state(tie).polyline, atol=1e-12)
    assert s.pawl_angle == 0.3 and s.link_speed_max == 0.02


def test_a_tie_with_uneven_or_twisted_links_is_rejected():
    tie = M.build_tie("jig")
    pts = np.array(tie.points)
    bent = pts.copy()
    bent[5, 1] += 0.001
    with pytest.raises(ValueError):
        type(tie)(**{**tie.__dict__, "points": tuple(map(tuple, bent))})._check()
    stretched = pts.copy()
    stretched[-1] += [0.0, 0.0, 0.001]
    with pytest.raises(ValueError):
        type(tie)(**{**tie.__dict__, "points": tuple(map(tuple, stretched))})._check()


def _lying_tie(ring_mm: float, tail_mm: float = 150.0, cord_out: bool = False):
    """A hand-built tie closed round the lying cords (world): the head beside the bundle, its root down and its exit
    toward -y, the strap under the cords, up their far side and back over them into the entry on a circle ``ring_mm``
    in radius, ``tail_mm`` of tail out. Returns (polyline, head centre, head quaternion, the cords' polylines)."""
    hr = L.head_rot_pitched(90.0)
    centre = np.array([0.33, 0.100, L.TOP + ring_mm * L.MM + 0.002])
    hc = centre + np.array([0.0, -ring_mm * L.MM - 0.004, 0.004])
    root = hc + hr @ L.ROOT
    entry = hc + hr @ np.array([L.SLOT_C, 0.0, L.ENTRY_Z + 0.001])
    exit_ = hc + hr @ np.array([L.SLOT_C, 0.0, L.EXIT_Z])
    tip = exit_ + hr @ np.array([0.0, 0.0, -tail_mm * L.MM])
    a = np.linspace(math.radians(215.0), math.radians(215.0 + 300.0), 40)  # from below the root, under, round, over
    ring = centre + ring_mm * L.MM * np.column_stack([np.zeros_like(a), np.cos(a), np.sin(a)])
    poly = np.vstack([root, ring, entry, exit_, tip])
    cords = {}
    for k, (name, *_rest) in enumerate(L.LIFT_CORDS):
        y = centre[1] + 0.003 * (k - 1) + (0.060 if cord_out and k == 0 else 0.0)
        cords[name] = np.array([[0.20, y, centre[2]], [0.45, y, centre[2]]])
    return poly, hc, L.matrix_to_quat_xyzw(hr), cords


class _Cords:
    def __init__(self, polylines):
        self.polylines = polylines


def test_the_lying_cords_score_reads_the_marks_on_the_strap():
    """Round 26's scorer: closed round all three cords, hands clear and still; then 1.0 with the full mark out of the
    head, 0.5 with the half mark out, 0.25 otherwise; 0 round two cords only, with a gripper near or while moving."""
    def scored(ring_mm, hand=0.05, speed=0.0, **kw):
        poly, hc, q, cords = _lying_tie(ring_mm, **kw)
        return L.lift_score(poly, hc, q, cords, hand, speed), poly, hc, q

    tight, poly, hc, q = scored(7.0)
    th = L.threading(poly, hc, q)
    full, half = L.lift_marks(L.arc_length(poly))
    assert tight == 1.0 and th.through == pytest.approx(0.150, abs=1e-6) and th.through >= full
    loose, poly, hc, q = scored(10.0)
    full, half = L.lift_marks(L.arc_length(poly))
    assert loose == 0.5 and half <= L.threading(poly, hc, q).through < full
    assert scored(18.0)[0] == 0.25
    assert scored(7.0, cord_out=True)[0] == 0.0  # one cord beside the loop
    assert scored(7.0, hand=0.02)[0] == 0.0 and scored(7.0, speed=0.02)[0] == 0.0
    assert scored(7.0, tail_mm=1.0)[0] == 0.0  # the tip barely through: not closed
    # the marks sit halfway between two of the 6.25 mm steps the tail rests at, 3.1 mm from either
    for mark in L.lift_marks(L.BUILDS["lift25"].tie_l):
        k = mark / L.LINK
        assert abs(k - round(k)) == pytest.approx(0.5, abs=1e-6)
    assert L.lift_marks(0.250) == pytest.approx((0.184375, 0.165625), abs=1e-9)
    if M.BUILD == "lift25":
        assert [round(1000 * (0.250 - s), 1) for s, _ in M.TIE.marks] == [184.4, 165.6]


def _hanging_tie(ring_mm: float, cord_out: bool = False, z: float = 0.140):
    """A hand-built 370 mm tie closed round the hanging cords (world): a level ring ``ring_mm`` in radius about the
    bundle, the head in front of it, whatever the ring leaves of the strap out of the head as the tail. Returns
    (polyline, head centre, head quaternion)."""
    turn = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # _lying_tie's frame stood on end
    hr = L.head_rot_pitched(90.0)
    r = ring_mm * L.MM
    hc = np.array([0.0, -r - 0.004, 0.004])
    root = hc + hr @ L.ROOT
    entry = hc + hr @ np.array([L.SLOT_C, 0.0, L.ENTRY_Z + 0.001])
    exit_ = hc + hr @ np.array([L.SLOT_C, 0.0, L.EXIT_Z])
    a = np.linspace(math.radians(215.0), math.radians(515.0), 40)
    ring = r * np.column_stack([np.zeros_like(a), np.cos(a), np.sin(a)])
    loop = np.vstack([root, ring, entry, exit_])
    tail = L.BUILDS["hang37"].tie_l - L.arc_length(loop)
    poly = np.vstack([loop, exit_ + hr @ np.array([0.0, 0.0, -tail])])
    centre = np.array([np.mean([c.xy[0] for c in L.CORDS]) + (0.060 if cord_out else 0.0),
                       np.mean([c.xy[1] for c in L.CORDS]), L.TOP + z])
    return centre + poly @ turn.T, centre + turn @ hc, L.matrix_to_quat_xyzw(turn @ hr)


def test_the_task_scores_the_hanging_cords_by_the_marks_through_score():
    """Round 38, the task: closed round all three hanging cords, hands clear and still; then 1.0 with the full mark
    out of the head, 0.5 with the half mark out, 0.25 otherwise."""
    from labkit import CableTieState

    full, half = L.lift_marks(L.BUILDS["hang37"].tie_l)
    assert (full, half) == pytest.approx((0.290, 0.270), abs=1e-9)  # whole millimetres from the tip (round 48)
    got = {}
    for ring in (9.0, 12.5, 20.0):
        poly, hc, q = _hanging_tie(ring)
        got[ring] = (L.hang_score(poly, hc, q, 0.05, 0.0), L.threading(poly, hc, q).through)
        assert L.arc_length(poly) == pytest.approx(L.BUILDS["hang37"].tie_l, abs=1e-9)
    assert got[9.0][0] == 1.0 and got[9.0][1] >= full
    assert got[12.5][0] == 0.5 and half <= got[12.5][1] < full
    assert got[20.0][0] == 0.25 and got[20.0][1] < half
    poly, hc, q = _hanging_tie(9.0)
    assert L.hang_score(poly, hc, q, 0.02, 0.0) == 0.0 and L.hang_score(poly, hc, q, 0.05, 0.02) == 0.0
    out = _hanging_tie(9.0, cord_out=True)
    assert L.hang_score(*out, 0.05, 0.0) == 0.0  # the loop closed beside the cords
    if M.BUILD != "hang37":
        return
    assert [round(1000 * (L.BUILDS["hang37"].tie_l - s), 1) for s, _ in M.TIE.marks] == [290.0, 270.0]
    s = CableTieState(pos=hc, quat_xyzw=q, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=poly.min(0), high=poly.max(0),
                      polyline=poly, pawl_angle=1.0, link_speed_max=0.0)
    M.SEEN["hand_clear"] = 0.05
    assert M.score({"tie": s}) == 1.0 and M.success({"tie": s})
    s2 = CableTieState(**{**s.__dict__, "link_speed_max": 0.02})
    assert M.score({"tie": s2}) == 0.0
    M.SEEN["hand_clear"] = 0.01
    assert M.score({"tie": s}) == 0.0
    M._new_trial()


@pytest.mark.skipif(M.BUILD not in L.LIFT_BUILDS, reason="the lying cords of rounds 19-30 (TIE_BUILD=lift25)")
def test_the_task_scores_the_lying_cords_through_score():
    from labkit import CableTieState

    poly, hc, q, cords = _lying_tie(7.0)
    s = CableTieState(pos=hc, quat_xyzw=q, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=poly.min(0), high=poly.max(0),
                      polyline=poly, pawl_angle=1.0, link_speed_max=0.0)
    M.SEEN["hand_clear"] = 0.05
    assert M.score({"tie": s, "cords": _Cords(cords)}) == 1.0 and M.success({"tie": s, "cords": _Cords(cords)})
    assert M.score({"tie": s}) == 0.0  # no cords to check: no credit
    s2 = CableTieState(**{**s.__dict__, "link_speed_max": 0.02})
    assert M.score({"tie": s2, "cords": _Cords(cords)}) == 0.0
    M.SEEN["hand_clear"] = 0.01
    assert M.score({"tie": s, "cords": _Cords(cords)}) == 0.0
    M._new_trial()


@pytest.mark.skipif(M.BUILD not in L.LIFT_BUILDS, reason="the lying cords of rounds 19-30 (TIE_BUILD=lift25)")
def test_the_keys_start_ends_with_the_head_held_by_the_bundle_and_the_left_open():
    """On the lying cords task.demo() plays tie_logic.KEYS_START: round 24's plan up to the head held by the bundle,
    then nothing: no hand lets go or goes back to ready, so a person can carry on from there on the keyboard."""
    tr = L.trial(M.TRIAL)
    assert M.TRIAL == L.KEYS_START and tr.b_keys and tr.build == M.BUILD == "lift25" and tr in L.KEPT
    plan = L.trial_plan(tr)
    old = L.trial_plan(L.trial("l25-thread"))

    def same(a, b):
        return len(a) == len(b) and all(x[0] == y[0] and x[2] == y[2] and x[4] == y[4] and np.allclose(x[1], y[1])
                                        and np.allclose(x[3], y[3]) for x, y in zip(a, b))

    n = [s[4] for s in old["right"]].index("level head") + 1
    assert same(plan["right"][:n], old["right"][:n])  # the right's steps as jobs 98 and 101 played them
    m = [s[4] for s in old["left"]].index("clear the head's way") + 1
    assert same(plan["left"][:m], old["left"][:m])
    for arm in plan:
        assert all(s[4] != "ready" and not isinstance(s[3], str) for s in plan[arm]), arm
    assert plan["right"][-1][2] == "head" and plan["left"][-1][2] == "closed"
    # the left takes the tail where it lies (a down grip, as rounds 23-25) and lifts it with strap to spare
    ls = {s[4]: s for s in plan["left"]}
    assert [s[4] for s in plan["left"]][-5:] == ["to the tail", "down to the tail", "take tail", "lift tail", "watch"]
    assert np.allclose(ls["take tail"][3], L.thread_left_rot(90.0)) and ls["take tail"][2] == "closed"
    held = np.asarray(ls["lift tail"][1], float) - L.B_TAIL_DEPTH * np.array([0.0, 0.0, 1.0])
    assert L.strap_slack(tr, held) >= L.B_SLACK_MIN
    assert np.allclose(plan["right"][-1][1], plan["right"][n - 1][1])  # still holding the head where it levelled it
    assert L.plan_seconds(plan) < 50.0
    try:
        ws = M.demo()
        assert isinstance(M.DEMO["steering"], M.ThreadSteering)
        assert [w.label for w in ws["right"]][-1] == "watch"
    finally:
        M.stop_demo()


def test_the_brake_engages_when_the_strap_backs_out_and_lets_go_when_pulled():
    import labkit

    labkit._SPECS["tie"] = M.TIE  # what the scene hook does when the sim spawns it
    try:
        M._new_trial()
        M.set_ratchet("brake")
        s = labkit.start_state(M.TIE)
        M._ratchet(None, s, 0.010)
        assert not M.SEEN["brake_on"] and M.SEEN["s_max"] == 0.010
        M._ratchet(None, s, 0.010 - 0.1 * L.MM)  # within the slip allowance (0.2 mm)
        assert not M.SEEN["brake_on"]
        M._ratchet(None, s, 0.010 - 0.6 * L.MM)  # backing out
        assert M.SEEN["brake_on"] and labkit.held("tie", "pawl") == pytest.approx(
            min(L.PAWL_REST, s.pawl_angle + math.radians(L.BRAKE_DEG)))
        M._ratchet(None, s, 0.010 - 0.1 * L.MM)  # pulled again past where it braked + 0.4 mm
        assert not M.SEEN["brake_on"] and labkit.held("tie", "pawl") is None and M.SEEN["brakes"] == 1
        M._new_trial()
        M.set_ratchet("open")
        M._ratchet(None, s, 0.0)
        assert labkit.held("tie", "pawl") == 0.0
    finally:
        labkit._HELD.clear()
        labkit._SPECS.pop("tie", None)
        M.set_ratchet(os.environ.get("TIE_RATCHET", "latch"))
        M._new_trial()
    with pytest.raises(ValueError):
        M.set_ratchet("glue")


def _rotation(w):
    rot = getattr(w, "rot", None)
    return rot if rot is not None else grip_rotation(w.yaw)


def _reach(arm, waypoints, shift=None):
    q = np.array(READY, float)
    for w in waypoints:
        pos = np.asarray(w.pos, float) + (shift(w) if shift else 0.0)
        q, pos_err, rot_err = ik.solve(pos, ik.matrix_to_quat_xyzw(_rotation(w)), q, base_pos=BASES[arm])
        assert pos_err < 1e-3 and rot_err < 1e-2, (arm, w.label, pos_err, rot_err)


@pytest.mark.parametrize("name", L.TRIAL_NAMES)
def test_every_probe_waypoint_is_reachable(name):
    for arm, ws in M.plan_waypoints(L.trial_plan(L.trial(name))).items():
        _reach(arm, ws)


@pytest.mark.parametrize("dx, dz", [(-1, 0), (1, 0), (0, -1), (0, 1)])
def test_the_steered_insert_stays_reachable_at_the_steering_caps(dx, dz):
    """The probe shifts the inserting hand by up to STEER_CAP across the slot's line (world x and z) while it lines
    up and pushes: those shifted waypoints must still be reachable."""
    ws = M.plan_waypoints(L.trial_plan(L.trial("c-heavier")))["left"]
    cap = L.STEER_CAP
    _reach("left", ws, lambda w: np.array([dx * cap, 0.0, dz * cap]) if w.label in L.STEER_LABELS else 0.0)


@pytest.mark.parametrize("name", [t.name for t in L.CHECKED if t.steer_yaw])
@pytest.mark.parametrize("sign", [-1, 1])
def test_the_yaw_steered_insert_stays_reachable(name, sign):
    """Round 8: the probe turns the inserting hand by up to YAW_CAP about the strap in its fingers and shifts it by up to
    the trial's cap while it lines up and pushes: those poses must still be reachable."""
    tr = L.trial(name)
    ws = M.plan_waypoints(L.trial_plan(tr))["left"]
    for dx in (-1, 1):
        q = np.array(READY, float)
        for w in ws:
            pos, rot = np.asarray(w.pos, float), _rotation(w)
            if w.label in L.STEER_LABELS:
                pos, rot = L.steer_pose(pos, rot, [dx * tr.steer_cap, 0.0, 0.0], sign * L.YAW_CAP, tr.grip_depth)
            q, pos_err, rot_err = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), q, base_pos=BASES["left"])
            assert pos_err < 1e-3 and rot_err < 1e-2, (w.label, dx, sign, pos_err, rot_err)


def test_rot_down_is_yam_arenas_down_grip():
    for yaw in (-90.0, 0.0, 30.0):
        assert np.allclose(L.rot_down(yaw), grip_rotation(math.radians(yaw)), atol=1e-12)


def test_finger_names_map_to_finger_joint_values():
    assert M.FINGERS["closed"] == 0.0 and M.FINGERS["open"] < M.FINGERS["wide"] < M.FINGERS["loose"] < 0.0
    assert M.FINGERS["head"] == pytest.approx(-0.5e-3)


def test_the_default_build_is_the_tasks_start():
    if "TIE_BUILD" not in os.environ:
        assert M.BUILD == "hang37" and not M.TIE.fixed  # round 38: nothing holds the head
        if "TIE_PHYS" not in os.environ:
            assert M.PHYS == L.HANG_PHYS == "nylonlead" and M.TIE.lead_in > 0 and M.TIE.head_friction == 0.25
            assert any(type(o).__name__ == "Prop" and o.friction == 0.35 for o in M.OBJECTS)  # the cords, their own friction
        assert M.CORDS_HARNESS is None and M.RATCHET["latch_at"] == L.LIFT_LATCH_AT
        names = [o.name for o in M.OBJECTS]
        assert not any("stand" in n or "clip" in n or "post" in n for n in names), names  # the review: no stand
        tr = L.trial(M.TRIAL)
        assert M.TRIAL == "h37-hold-l" and tr.build == "hang37" and tr.demo and tr.h_thread and tr.h_held
        assert tr.phys == L.HANG_PHYS and not tr.h_pull_to  # the draw stops at the full mark
    importlib.reload  # noqa: B018 - keep the import used


def test_task_json_matches_the_code():
    import json

    card = json.loads((HERE / "task.json").read_text(encoding="utf-8"))
    for key in ("title", "category", "instruction", "setup", "scoring", "run_limit", "fidelity", "scores"):
        assert card.get(key), key
    assert card["instruction"] == M.INSTRUCTION
    assert card["category"] == "Societal relatability"

    full, half = (f"{1000 * v:.1f}".removesuffix(".0") for v in L.lift_marks(L.BUILDS[M.BUILD].tie_l))
    for text in (card["scoring"], card["setup"], card["reset_notes"]):
        assert f"{full} mm" in text and f"{half} mm" in text
    assert f"{round(1000 * L.CLEAR / 10)} cm" in card["scoring"] and "1 cm" in card["scoring"]
    assert f"{round(1000 * L.BUILDS[M.BUILD].tie_l)} x 4.8" in card["setup"]
    steps = int(card["run_limit"].split(" steps")[0].replace(",", ""))
    assert L.plan_seconds(L.trial_plan(L.trial(M.TRIAL))) * L.CONTROL_HZ < steps - 400  # the demo fits, with room
    assert len(card["fidelity"]) <= 500 and len(card["context"]) <= 500
    texts = [json.dumps(card)] + [(HERE / f).read_text(encoding="utf-8") for f in ("README.md", "SPEC.md")]
    for name in ("henry", "khoi", "lamp-tasks", "claudeimplementation", "lenovo", "/home/"):  # blind review
        for text in texts:
            assert name not in text.lower()


def test_no_arm_path_touches_the_tv_the_wall_or_the_cords():
    """Round 1's lesson: replay every trial's path as the probe plays it and keep the arms' collision meshes 2 cm off
    the TV and wall and the cords (fingertips excepted near the cords: test_logic checks those with the finger
    envelope). The round-1 tilted grip from above must fail this (its wrist went 20 mm into the TV)."""
    import armcheck

    for t in L.CHECKED:
        for arm, (d, when, label, link, what) in armcheck.plan_clearance(M, L.trial_plan(t), build=t.build).items():
            assert d > 0.02, (t.name, arm, round(d * 1000, 1), when, label, link, what)
    old = L.Trial("round1-flat", "jig", "", insert="flat", pull=None, watch_s=2.0)
    left = armcheck.plan_clearance(M, L.trial_plan(old))["left"]
    assert left[0] < 0 and left[4] == "tv"


# -- round 6: the latch and the plastic strap -------------------------------------------------------------------------
def test_the_latch_holds_the_pawl_clear_until_the_tail_is_out_then_brakes():
    """Round 5: 3 of 5 pushes blew up where the tip met the sprung pawl. The latch holds the pawl flat against the far
    wall while the tail goes in, lets it onto the strap once the task's latch length is out (16 mm on the first start, 8 mm on the lying cords),
    and brakes from then on."""
    import labkit

    labkit._SPECS["tie"] = M.TIE
    try:
        M._new_trial()
        M.set_ratchet("latch")
        at = M.RATCHET["latch_at"]
        assert at == (L.LIFT_LATCH_AT if M.BUILD in L.LIFT_BUILDS + L.HANG_BUILDS else L.LATCH_AT)
        s = labkit.start_state(M.TIE)
        M._ratchet(None, s, 0.0)
        assert labkit.held("tie", "pawl") == 0.0 and M.SEEN["opened"]
        M._ratchet(None, s, at - 1 * L.MM)
        M._ratchet(None, s, at - 3 * L.MM)  # backing out before the latch: no brake, the pawl stays clear
        assert labkit.held("tie", "pawl") == 0.0 and not M.SEEN["brake_on"]
        M._ratchet(None, s, at + 1 * L.MM)
        assert labkit.held("tie", "pawl") is None and M.SEEN["latched"] and not M.SEEN["opened"]
        M._ratchet(None, s, at + 0.4 * L.MM)  # now it backs out: the brake
        assert M.SEEN["brake_on"] and labkit.held("tie", "pawl") is not None
    finally:
        labkit._HELD.clear()
        labkit._SPECS.pop("tie", None)
        M.set_ratchet(os.environ.get("TIE_RATCHET", "latch"))
        M._new_trial()


class _T:
    def __init__(self, a):
        self.a = a

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _FakeTie:
    def __init__(self, n):
        import types

        self.joint_names = ["pawl"] + [f"bend_{i:02d}" for i in range(n)]
        self.device = "cpu"
        self.q = np.zeros((1, n + 1))
        self.data = types.SimpleNamespace()

        self.targets = []
        self.data.joint_pos = _T(self.q)

    def set_joint_position_target_index(self, target):
        self.targets.append(np.asarray(target)[0].tolist())

    def write_joint_stiffness_to_sim_index(self, stiffness):
        self.stiffness = np.asarray(stiffness)[0].tolist()


def test_the_strap_takes_a_set_and_a_reset_clears_it():
    """Round 5's pull jammed when the D-loop's kink (55 degrees, permanent in an elastic strap) reached the slot.
    With a yield angle, a joint bent past it keeps all but the yield angle of the bend, and its drive aims there."""
    import sys
    import types

    import labkit

    if "torch" not in sys.modules:  # offline: a stand-in that keeps the written tensors as arrays
        try:
            import torch  # noqa: F401
        except ImportError:
            sys.modules["torch"] = types.SimpleNamespace(float32="float32",
                                                         tensor=lambda v, dtype=None, device=None: np.asarray(v, float))

    tie = M.TIE
    assert tie.yield_angle == pytest.approx(math.radians(L.YIELD_DEG))
    fake = _FakeTie(tie.n_links)
    env = types.SimpleNamespace(scene={"tie": fake})
    env.unwrapped = env
    labkit._SPECS["tie"] = tie
    try:
        labkit._PLASTIC.pop("tie", None)
        y = math.radians(L.YIELD_DEG)
        fake.q[0, 21] = 1.0  # bend_20 straightened 1 rad from its start shape
        fake.q[0, 5] = 0.5 * y  # bend_04 within the yield: elastic
        labkit._tie_plastic(env)
        rest = labkit.tie_set("tie")
        assert rest[21] == pytest.approx(1.0 - y) and rest[5] == 0.0 and rest[0] == 0.0
        assert fake.targets[-1][21] == pytest.approx(1.0 - y) and fake.targets[-1][0] == 0.0
        labkit.hold_joint(env, "tie", 0.0, "pawl")  # a pawl hold rewrites every target: the set must survive it
        assert fake.targets[-1][21] == pytest.approx(1.0 - y)
        assert fake.targets[-1][0] == pytest.approx(-L.PAWL_REST)
        labkit.release_joint(env, "tie", "pawl")
        labkit.set_tie_yield("tie", None)  # purely elastic: no further set
        fake.q[0, 21] = 1.5
        labkit._tie_plastic(env)
        assert labkit.tie_set("tie")[21] == pytest.approx(1.0 - y)
        labkit._YIELD.clear()
        fake.q[0, 21] = 0.0
        fake.write_joint_position_to_sim_index = lambda **k: None
        fake.write_joint_velocity_to_sim_index = lambda **k: None
        fake.data.default_joint_pos = types.SimpleNamespace(torch=types.SimpleNamespace(clone=lambda: None))
        fake.data.default_joint_vel = fake.data.default_joint_pos
        labkit._set_joint_starts(env)
        assert labkit.tie_set("tie") is None and fake.targets[-1][21] == 0.0
    finally:
        labkit._HELD.clear()
        labkit._YIELD.clear()
        labkit._PLASTIC.pop("tie", None)
        labkit._SPECS.pop("tie", None)


def test_the_bend_limits_leave_room_to_pull_every_start_bend_straight():
    """Round 5: 60 degree limits left the D-loop's 55 degree turn 5 degrees from its stop when pulled straight."""
    p = np.asarray(M.TIE.points)
    u = np.diff(p, axis=0)
    bends = [abs(math.atan2(np.cross(u[i], u[i + 1])[1], u[i] @ u[i + 1])) for i in range(len(u) - 1)]
    assert max(bends) + math.radians(30.0) < M.TIE.max_bend


def test_the_clip_stand_is_drawn_on_the_welded_head_where_tie_logic_puts_it_and_collides_with_nothing():
    names = {o.name for o in M.OBJECTS}
    assert not names & {"clip_bottom", "clip_top", "clip_back", "stand_rod", "stand_base"}  # no colliders (round 13)
    for build in ("jig", "wrap", "tight"):
        b = L.BUILDS[build]
        tie = M.build_tie(build)
        tie._check()
        decor = {d[0]: d for d in tie.decor}
        hp, rot = np.asarray(b.head_pos), np.asarray(b.head_rot)
        for name, kind, geo in L.stand_parts(b):
            _, k, at, dims, _ = decor[name]
            assert k == kind
            centre = hp + rot @ np.asarray(at)
            if kind == "box":
                lo, hi = geo
                assert np.allclose(centre, (lo + hi) / 2, atol=1e-12)
                assert np.allclose(np.abs(rot) @ np.asarray(dims), hi - lo, atol=1e-12)
            else:
                xy, r, (z0, z1) = geo
                assert np.allclose(centre, [xy[0], xy[1], (z0 + z1) / 2], atol=1e-12)
                assert dims[0] == pytest.approx(r) and dims[1] == pytest.approx(z1 - z0)
                assert np.allclose(np.abs(rot @ np.eye(3)["XYZ".index(dims[2])]), [0, 0, 1])  # upright in the world
    assert M.build_tie("free").decor == ()


def test_the_demo_steering_shifts_every_left_waypoint_like_the_probe_and_follows_its_labels():
    from yam_arena.demo import ArmPlan
    from yam_arena.station import BASES

    tr = L.trial("c-heavier")
    plan = M.demo_plan(tr)
    try:
        st = M.DEMO["steering"]
        assert st is not None and st.poses is plan["left"]
        ref = ArmPlan(M.plan_waypoints(L.trial_plan(tr))["left"], BASES["left"])  # the probe's own, unshifted
        arm = ArmPlan(plan["left"], BASES["left"])
        for k in range(1, int(L.plan_seconds(L.trial_plan(tr)) * M.CONTROL_HZ)):
            assert st.label(k / M.CONTROL_HZ) == ref.target(k / M.CONTROL_HZ)[3]
        # the probe's rule, written out: corr every STEER_EVERY steps in STEER_LABELS before capture; depth in the push
        corr, depth, rng = np.zeros(3), 0.0, np.random.default_rng(3)
        for k in range(1, 400):
            through = 0.0 if k < 200 else 0.004 * (k - 200) / 20
            err = rng.normal(0.0, 0.002, 3)
            label = ref.target(k / M.CONTROL_HZ)[3]
            if k % L.STEER_EVERY == 0:
                if label in L.STEER_LABELS and through <= 0.0:
                    corr = np.clip(corr - L.STEER_GAIN * err, -tr.steer_cap, tr.steer_cap)
                if label in L.HANDOVER_DEPTH_LABELS:
                    depth = float(np.clip(depth + 1000 * L.DEPTH_GAIN * max(L.HANDOVER_OUT - through, 0.0), 0.0,
                                          1000 * tr.depth_cap))
            st.update(k, through, err)
            shift = corr + st.d_dir * depth / 1000.0
            assert np.allclose(st.corr, corr, atol=1e-15) and st.depth_mm == pytest.approx(depth, abs=1e-12)
            t = k / M.CONTROL_HZ  # the step update(k) prepares
            assert np.allclose(arm.target(t)[0], ref.target(t)[0] + shift, atol=1e-12)
        assert np.any(corr != 0.0) and depth > 0.0  # the synthetic run exercised both
    finally:
        M.stop_demo()
    assert M.DEMO["steering"] is None


def test_the_demo_plays_the_candidate_and_the_probe_can_check_it_fresh():
    cand = L.trial("m-demo")  # the first submission's demo, on the pre-looped start (TIE_BUILD=jig plays it)
    assert cand is L.CANDIDATE and cand in L.KEPT
    assert cand.handover and cand.build == "jig" and cand.demo and cand.ratchet == "latch" and cand.tighten > 0.0
    assert not cand.unstick  # as job 73 ran it
    demo, old = L.trial("c-demo"), L.trial("c-heavier")  # the earlier candidate and its demo-played twin
    assert demo.demo and not old.demo
    assert {k: v for k, v in vars(demo).items() if k not in ("name", "why", "demo")} == \
        {k: v for k, v in vars(old).items() if k not in ("name", "why", "demo")}
    launch = [t for t in L.TRIALS if (t.build, t.phys) == (cand.build, cand.phys)]
    assert cand not in L.TRIALS or launch[0] is cand  # first in its launch: fresh, as a reviewer's run is
    assert all(t.demo for t in L.CHECKED if t.tighten > 0.0)  # the tighten lives only in task.Steering


def test_the_tighten_aims_past_the_joint_the_jig_loop_rests_at():
    """Round 17: a let-go at 137.9 mm out or more ended at JIG_REST (137.5, 1.0); at 137.3 or less, a link back (0.5).
    So the tighten's target is past that, and the rest itself is inside the 1.0 line."""
    snug = L.snug_through(L.BUILDS["jig"])
    for tr in L.CHECKED:
        if tr.tighten > 0.0:
            assert snug - tr.tighten_slack >= L.JIG_REST + 0.4 * L.MM, tr.name
    assert snug - L.JIG_REST <= L.SNUG_SLACK

def test_the_demo_task_ratchet_is_the_one_its_trial_was_probed_with():
    """yam_scripted plays demo() under the task's default ratchet; the probe set the trial's. They must agree."""
    assert L.trial(M.TRIAL).ratchet == os.environ.get("TIE_RATCHET", "latch") == M.RATCHET["mode"]


def test_the_demo_tighten_carries_the_pulling_hand_on_along_its_last_stroke_until_snug():
    from yam_arena.demo import ArmPlan
    from yam_arena.station import BASES

    tr = L.trial("m-demo")
    plan = M.demo_plan(tr)
    try:
        st = M.DEMO["steering"]
        assert st is not None and st.right is plan["right"]
        ref = ArmPlan(M.plan_waypoints(L.trial_plan(tr))["right"], BASES["right"])
        arm = ArmPlan(plan["right"], BASES["right"])
        at = {w.label: np.asarray(w.pos, float) for w in plan["right"]}
        d = (at["pull 2"] - at["pull 1"]) / np.linalg.norm(at["pull 2"] - at["pull 1"])
        assert np.allclose(st.pull_dir, d)
        goal = L.snug_through(L.BUILDS["jig"]) - tr.tighten_slack
        steps = int(L.plan_seconds(L.trial_plan(tr)) * M.CONTROL_HZ)
        tight = [k for k in range(1, steps) if ref.target(k / M.CONTROL_HZ)[3] == "tighten"]
        assert len(tight) == round(L.TIGHTEN_S * M.CONTROL_HZ)
        through, k0 = goal - 3 * L.MM, tight[0]
        for k in range(1, steps):
            if k < k0:
                through = 0.1  # the pulls: far short, but not in "tighten", so the hand isn't carried on yet
            elif k < k0 + 6:
                through = goal - 3 * L.MM
            else:
                through = goal + 0.5 * L.MM  # snug: it stops going on
            st.update(k, through, np.zeros(3))
            if k == k0 - 1:
                assert st.tighten_mm == 0.0
            t = k / M.CONTROL_HZ
            moved = []  # the plan as it would be written with this shift: the tighten's waypoints along the pull
            for w in M.plan_waypoints(L.trial_plan(tr))["right"]:
                if w.label in L.TIGHTEN_SHIFT_LABELS:
                    w.pos = tuple(float(v) for v in np.asarray(w.pos, float) + d * st.tighten_mm / 1000.0)
                moved.append(w)
            want = ArmPlan(moved, BASES["right"]).target(t)[0]
            assert np.allclose(arm.target(t)[0], want, atol=1e-12), (k, ref.target(t)[3])
            if ref.target(t)[3] == "tighten":
                assert np.allclose(arm.target(t)[0], ref.target(t)[0] + d * st.tighten_mm / 1000.0, atol=1e-12)
        rate = 1000 * L.TIGHTEN_RATE * L.STEER_EVERY / M.CONTROL_HZ  # mm a short update carries it on
        assert st.tighten_mm == pytest.approx(rate * len([k for k in range(k0, k0 + 6) if k % L.STEER_EVERY == 0]))
        held = st.tighten_mm
        k1 = (k0 // L.STEER_EVERY + 2) * L.STEER_EVERY  # a step in "tighten" that updates
        assert k1 in tight
        for _ in range(200):  # far short for long (the same step, again and again): capped
            st.update(k1, goal - 50 * L.MM, np.zeros(3))
        assert st.tighten_mm == pytest.approx(1000 * tr.tighten) and st.tighten_mm > held
        others = [w for w in plan["right"] if w.label not in L.TIGHTEN_SHIFT_LABELS]
        assert all(np.allclose(w.pos, b) for w, b in zip(plan["right"], st.right_base) if w in others)
    finally:
        M.stop_demo()


@pytest.mark.parametrize("name", [t.name for t in L.CHECKED if t.tighten > 0.0])
def test_the_tightened_pull_stays_reachable_at_its_cap(name):
    tr = L.trial(name)
    ws = M.plan_waypoints(L.trial_plan(tr))["right"]
    at = {w.label: np.asarray(w.pos, float) for w in ws}
    d = (at["pull 2"] - at["pull 1"]) / np.linalg.norm(at["pull 2"] - at["pull 1"])
    _reach("right", ws, lambda w: d * tr.tighten if w.label in L.TIGHTEN_SHIFT_LABELS[1:] else 0.0)


def test_the_tighten_is_off_unless_asked_and_the_old_candidate_plan_is_unchanged():
    old = L.trial_plan(L.trial("c-heavier"))["right"]
    assert not any(s[4] == "tighten" for s in old)
    assert [s[0] for s in old if s[4] == "hold snug"] == [1.0]
    new = L.trial_plan(L.trial("m-demo"))["right"]
    assert [s[0] for s in new if s[4] == "tighten"] == [L.TIGHTEN_S]
    assert [s[0] for s in new if s[4] == "hold snug"] == [1.5]
    assert L.plan_seconds(L.trial_plan(L.trial("m-demo"))) * M.CONTROL_HZ < 2400


def _pull_labels(tr):
    from yam_arena.demo import ArmPlan
    from yam_arena.station import BASES

    steps = int(L.plan_seconds(L.trial_plan(tr)) * M.CONTROL_HZ)
    ref = ArmPlan(M.plan_waypoints(L.trial_plan(tr))["right"], BASES["right"])
    return steps, ref, [ref.target(k / M.CONTROL_HZ)[3] for k in range(steps)]


def test_the_unstick_never_acts_while_the_strap_keeps_up_so_unstalled_runs_repeat():
    """Round 18 replays round 17's order with the unstick on: a pull whose strap stays within UNSTICK_LEAD of the plan
    must leave every waypoint exactly as it was, so the runs that didn't stall repeat to the 0.1 mm."""
    tr = L.trial("u-deep")
    steps, ref, labels = _pull_labels(tr)
    plan = M.demo_plan(tr)
    try:
        st = M.DEMO["steering"]
        base = [w.pos for w in plan["right"]]
        for k in range(1, steps):
            t = k / M.CONTROL_HZ
            if labels[k] in L.UNSTICK_LABELS:
                assert np.allclose(st.planned(t), ref.target(t)[0], atol=1e-12)  # ArmPlan's own walk
                through = 0.0227 + max(float(np.linalg.norm(st.planned(t) - st.grip_at)) - 0.010, 0.0)  # 10 mm behind
            else:
                through = 0.0227 if k < labels.index("pull 1") else 0.139  # past the tighten's target: it stays put
            st.update(k, through, np.zeros(3))
        assert st.back_mm == 0.0 and st.side_mm == 0.0 and [w.pos for w in plan["right"]] == base
    finally:
        M.stop_demo()


def test_the_unstick_holds_the_hand_near_a_stalled_strap_moves_it_aside_and_lets_go_past_the_zone():
    from yam_arena.demo import ArmPlan
    from yam_arena.station import BASES

    tr = L.trial("u-deep")
    steps, ref, labels = _pull_labels(tr)
    plan = M.demo_plan(tr)
    try:
        st = M.DEMO["steering"]
        arm = ArmPlan(plan["right"], BASES["right"])
        k0 = next(k for k in range(steps) if labels[k] == "pull 1" and k % L.STEER_EVERY == 0)
        free_at = k0 + 40  # stalled 2 s, 25.6 mm out (round 17's stall), then it runs out past the zone
        held = 0
        for k in range(1, steps):
            through = 0.0227 if k < k0 else (0.0256 if k < free_at else L.UNSTICK_ZONE + 0.01)
            st.update(k, through, np.zeros(3))
            t = k / M.CONTROL_HZ
            if k % L.STEER_EVERY or labels[k] not in L.UNSTICK_LABELS or k >= free_at:
                continue
            side = np.array([st.side_mm / 1000.0, 0.0, 0.0])
            lead = float((arm.target(t)[0] - side - st.grip_at) @ st.back_dir)  # how far along the plan's way it is
            assert lead <= (through - st.through0) + L.UNSTICK_LEAD + 1e-9, k
            if st.back_mm > 0.0:
                held += 1
                assert lead == pytest.approx((through - st.through0) + L.UNSTICK_LEAD, abs=1e-9)
        assert held > 10 and st.through0 == pytest.approx(0.0256)  # where the strap was as the pull began
        assert st.back_mm == 0.0 and st.side_mm == 0.0  # let go of, once the strap was past the zone
    finally:
        M.stop_demo()
    side_seen = []
    plan = M.demo_plan(tr)
    try:
        st = M.DEMO["steering"]
        for k in range(1, k0 + 40):
            st.update(k, 0.0227 if k < k0 else 0.0256, np.zeros(3))
            side_seen.append(st.side_mm)
        assert min(side_seen) == pytest.approx(1000 * tr.unstick_side)  # held long enough: all the way aside
        assert all(b <= a for a, b in zip(side_seen, side_seen[1:]))  # toward the robot only, while held
    finally:
        M.stop_demo()


@pytest.mark.parametrize("name", [t.name for t in L.CHECKED if t.unstick])
def test_the_unstick_side_shift_stays_reachable(name):
    tr = L.trial(name)
    ws = M.plan_waypoints(L.trial_plan(tr))["right"]
    _reach("right", ws, lambda w: np.array([tr.unstick_side, 0.0, 0.0]) if w.label in L.UNSTICK_SHIFT_LABELS else 0.0)


# -- round 19: the lift start -------------------------------------------------------------------------------------------


def test_the_lift_cords_lie_on_the_console_in_equal_links_and_plug_into_the_strip():
    h = M.build_cords()
    h._check()
    assert 2 + sum(w.segments for w in h.wires) <= 64
    strip = L.lift_strip_centre()
    for (name, r, off, kg_m, _), w in zip(L.LIFT_CORDS, h.wires):
        p = np.asarray(w.points) + np.asarray(h.pos)
        chords = np.linalg.norm(np.diff(p, axis=0), axis=1)
        assert np.ptp(chords) < 1e-9 and w.radius == r and w.link_mass == pytest.approx(kg_m * chords[0])
        assert p[:, 2].min() == pytest.approx(L.TOP + r + 0.2 * L.MM)  # lying on the console, not in it
        assert np.allclose(p[-1], strip + np.asarray(w.terminal))  # on the strip's back face
        assert p[-1][0] == pytest.approx(L.LIFT_STRIP["x"][1])
        assert np.allclose(p[0], h.pos + np.asarray(w.root)) and p[0][2] == pytest.approx(L.LIFT_ANCHOR_Z)
    assert h.device_mass == L.LIFT_STRIP_MASS and h.parts[0].size == pytest.approx((0.055, 0.28, 0.04))


def test_the_lift_start_tie_lies_clear_of_the_cords_and_points_its_strap_the_slides_way():
    from labkit import start_state

    s = start_state(M.build_tie("lift"))
    h = M.build_cords()
    for w in h.wires:
        p = np.asarray(w.points) + np.asarray(h.pos)
        d = np.min(np.linalg.norm(s.polyline[:, None, :2] - p[None, :, :2], axis=2))
        assert d > 0.10
    strap = s.polyline[-1] - s.polyline[0]
    assert np.allclose(strap[:2] / np.linalg.norm(strap[:2]), L.SLIDE_DIR, atol=1e-9)
    assert s.polyline[-1][1] < L.SLIDE_TIP[0]  # it starts short of where the slide begins


def test_equal_chords_end_on_the_path_ends_with_equal_steps():
    path = np.column_stack([np.linspace(0, 1, 500), np.sin(np.linspace(0, 3, 500)), np.zeros(500)])
    p = L.equal_chords(path, 13)
    chords = np.linalg.norm(np.diff(p, axis=0), axis=1)
    assert len(p) == 14 and np.ptp(chords) < 1e-9
    assert np.allclose(p[0], path[0]) and np.allclose(p[-1], path[-1])


@pytest.mark.parametrize("name", [t.name for t in L.CHECKED if t.stage == "slide"])
def test_the_slide_waits_for_the_lift_and_the_bundle_comes_down_only_after_it(name):
    tr = L.trial(name)
    plan = L.trial_plan(tr)

    def ends(arm, label):
        t, out = 0.0, None
        for s in plan[arm]:
            t += s[0]
            if s[4] == label:
                out = t
        return out

    def starts(arm, label):
        t = 0.0
        for s in plan[arm]:
            if s[4] == label:
                return t
            t += s[0]

    assert starts("right", "slide under") >= ends("left", "lift cords") - 1e-9
    assert starts("left", "lower cords") >= ends("right", "slide under") - 1e-9
    assert starts("right", "let go tie") >= ends("left", "let go cords") - 1e-9
    slide = [s for s in plan["right"] if s[4] == "slide under"][0]
    hold = [s for s in plan["right"] if s[4] == "carry tie"][0]
    assert np.allclose(np.subtract(slide[1], hold[1])[[0, 2]], 0.0)  # straight along y at one height
    assert slide[1][1] - hold[1][1] == pytest.approx(L.SLIDE_TIP[1] - L.SLIDE_TIP[0])


def test_the_stiff_presets_scale_the_straps_stiffness_and_nothing_else_of_the_tie():
    base, k5, k15 = (M.build_tie("lift", p) for p in ("heavier", "stiff5", "stiff15"))
    assert k5.stiffness == pytest.approx(5 * base.stiffness) and k15.stiffness == pytest.approx(15 * base.stiffness)
    assert k15.link_mass == base.link_mass and k15.head_mass == base.head_mass and k15.points == base.points


# -- round 20: the lift start, stage B --------------------------------------------------------------------------------
WEDGE = [t.name for t in L.CHECKED if t.b_wedge]  # round 29: the tail locked tip-down, then threaded
THREADS = [t.name for t in L.CHECKED if t.stage == "thread" and not t.b_keys and not t.b_cinch and not t.b_wedge]  # rounds 20-25's
CINCH = [t.name for t in L.CHECKED if t.b_cinch]  # round 26's scripted finish


def _times(plan, arm):
    """{label: (start, end)} of an arm's steps (a label's last occurrence)."""
    t, out = 0.0, {}
    for s in plan[arm]:
        out[s[4]] = (t, t + s[0])
        t += s[0]
    return out


def _box_points(box, n: int = 5) -> np.ndarray:
    c, r, h = box
    g = np.linspace(-1.0, 1.0, n)
    pts = []
    for ax in range(3):
        o = [i for i in range(3) if i != ax]
        for sgn in (-1.0, 1.0):
            for a in g:
                for b in g:
                    v = np.zeros(3)
                    v[ax], v[o[0]], v[o[1]] = sgn, a, b
                    pts.append(c + r @ (v * h))
    return np.array(pts)


def _gap_between(boxes_a, boxes_b) -> float:
    return min(min(L.box_distance(_box_points(a), b), L.box_distance(_box_points(b), a)) for a in boxes_a for b in boxes_b)


@pytest.mark.parametrize("name", THREADS)
def test_stage_b_hands_take_turns_so_the_tie_is_always_held_and_nothing_moves_under_a_hand(name):
    plan = L.trial_plan(L.trial(name))
    lt, rt = _times(plan, "left"), _times(plan, "right")
    assert lt["hold strap"][1] <= rt["release pick"][0] + 1e-9  # the strap held before the right lets go of the head
    assert rt["grip head"][1] <= lt["let go strap"][0] + 1e-9  # the head held again before the left lets go
    assert lt["hold cords"][1] <= rt["slide under"][0] + 1e-9  # the bundle up before the strap slides
    assert rt["hold tie"][1] <= lt["lower cords"][0] + 1e-9  # ... and down only after
    assert lt["clear cords"][1] <= rt["level head"][0] + 1e-9  # the head raised once the left is off the bundle
    assert rt["level head"][1] <= lt["to the tail"][0] + 1e-9  # the tail taken where it then lies
    assert lt["back off tail"][1] <= rt["let go head"][0] + 1e-9  # the head held until the left has let go and left
    for arm, labels in (("left", ("hold strap", "take tail", "lift tail", "over", "above head", "align", "push",
                                  "push hold")), ("right", ("grip", "raise tie", "to regrasp", "grip head", "carry tie",
                                                            "slide under", "hold tie", "level head"))):
        steps = {s[4]: s for s in plan[arm]}
        for lab in labels:
            assert steps[lab][2] in ("closed", "head"), (arm, lab)


@pytest.mark.parametrize("name", THREADS)
def test_stage_b_fingers_keep_clear_of_each_other_and_of_the_tails_way_through_the_slot(name):
    """The fat finger envelopes: the left holding the strap vs every pose of the right while it changes grips (the
    hands' fingers overlap in x, so they must be apart along y); the left at the end of its push (with the extra depth
    and either steering cap along y) vs the right holding the head; the right's fingers 3 mm or more off the slot's
    centre line where the tail goes through (the strap's half width is 2.4 mm)."""
    tr = L.trial(name)
    plan = L.trial_plan(tr)
    ls, rs = {s[4]: s for s in plan["left"]}, {s[4]: s for s in plan["right"]}
    hold = L.finger_boxes(ls["hold strap"][1], ls["hold strap"][3], L.W)
    for lab in ("to regrasp", "release pick", "up from head", "around", "behind head", "onto head", "grip head"):
        s = rs[lab]
        gap = L.HEAD[1] if s[2] == "head" else L.FINGER_GAP[s[2]] * L.MM
        assert _gap_between(hold, L.finger_boxes(s[1], s[3], gap)) > 0.010, lab
    tg = L.thread_targets(tr)
    right = L.finger_boxes(rs["level head"][1], rs["level head"][3], L.HEAD[1])
    for dy in (L.B_STEER_LO[1], 0.0, L.B_STEER_CAP[1]):
        p = np.asarray(ls["push"][1]) + tg["through"] * L.B_DEPTH_CAP + np.array([0.0, dy, 0.0])
        assert _gap_between(L.finger_boxes(p, ls["push"][3], L.W), right) > 0.002, dy
    line = tg["entry"] - np.outer(np.linspace(-0.06, 0.03, 91), tg["through"])  # 60 mm above the entry, 30 below
    assert min(L.box_distance(line, b) for b in right) > 0.003


@pytest.mark.parametrize("name", THREADS)
def test_stage_b_lines_the_tip_up_over_the_entry_and_pushes_it_out_of_the_exit(name):
    tr = L.trial(name)
    ls = {s[4]: s for s in L.trial_plan(tr)["left"]}
    tg = L.thread_targets(tr)

    def tip(lab):
        r = ls[lab][3]
        return np.asarray(ls[lab][1]) + L.B_TAIL_DEPTH * r[:, 2] + tr.b_gft * r[:, 1]

    assert np.allclose(tip("align"), tg["entry"] - tg["through"] * L.B_TIP_GAP, atol=1e-9)
    assert np.allclose(tip("push"), tg["entry"] + tg["through"] * (L.HEAD[2] + tr.b_out), atol=1e-9)
    assert np.allclose(ls["align"][3][:, 1], tg["through"], atol=1e-9)  # the tail held along the slot's axis
    assert np.allclose(L.thread_left_rot(90.0), L.rot_axes((0, 0, -1), (-1, 0, 0)))  # the pick: a down grip


@pytest.mark.parametrize("name", THREADS)
def test_stage_b_closed_loop_shifts_stay_reachable(name):
    """ThreadSteering moves each group onto where things really are, at most L.B_SNAP (round 21: round 20's snaps were
    open): every corner of the strap hold's, the head's and the tail's boxes, and the line-up's B_STEER_LO..B_STEER_CAP in x and y
    plus B_DEPTH_CAP down."""
    ws = M.plan_waypoints(L.trial_plan(L.trial(name)))
    lo_, c, k = L.B_STEER_LO, L.B_STEER_CAP, L.B_DEPTH_CAP
    cases = []
    for key, arm, labels in (("strap", "left", L.THREAD_STRAP_LABELS), ("head", "right", L.THREAD_HEAD_LABELS),
                             ("tail", "left", L.THREAD_TAIL_LABELS)):
        lo, hi = L.B_SNAP[key]
        cases += [(arm, labels, (x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in {lo[2], hi[2]}]
    cases += [("left", L.THREAD_STEER_LABELS, (dx, dy, -k)) for dx in (lo_[0], c[0]) for dy in (lo_[1], c[1])]
    for arm, labels, v in cases:
        _reach(arm, ws[arm], lambda w, v=v, labels=labels: np.asarray(v, float) if w.label in labels else 0.0)


class _TieAt:
    def __init__(self, pos, polyline):
        self.pos, self.polyline = np.asarray(pos, float), np.asarray(polyline, float)


def _step_in(when, label, after):
    k = int(round((when[label] + after) * M.CONTROL_HZ))
    return k + k % L.STEER_EVERY


def test_thread_steering_moves_each_group_onto_the_real_strap_head_and_tail_and_freezes_it_once_closed():
    tr = L.trial(THREADS[0])
    plan = M.demo_plan(tr)
    try:
        st = M.DEMO["steering"]
        assert isinstance(st, M.ThreadSteering)
        tg = L.thread_targets(tr)
        base = {w.label: np.asarray(w.pos) for w in plan["left"]}
        when, t = {}, 0.0
        for w in plan["left"]:
            when.setdefault(w.label, t)
            t += w.duration
        w = {x.label: x for x in plan["left"]}
        off = np.array([0.0, 0.004, -0.006])
        straight = tg["regrasp"] + L.R_FREE @ L.ROOT + np.outer(np.linspace(0, L.TIE_L, 33), L.R_FREE[:, 0])
        st.update(_step_in(when, "down to the strap", 0.5), 0.0, np.zeros(3), state=_TieAt(tg["regrasp"], straight + off))
        assert np.allclose(np.asarray(w["hold strap"].pos) - base["hold strap"], off, atol=1e-9)
        assert np.allclose(np.asarray(w["up from strap"].pos) - base["up from strap"], off, atol=1e-9)
        st.update(_step_in(when, "hold strap", 0.3), 0.0, np.zeros(3), state=_TieAt(tg["regrasp"], straight + 2 * off))
        assert np.allclose(np.asarray(w["hold strap"].pos) - base["hold strap"], off, atol=1e-9)  # frozen once closed
        # round 21: a snap moves a hand at most B_SNAP, however far off the strap is (job 95: 57 mm, then 203 mm)
        st2 = M.ThreadSteering(tr, M.plan_waypoints(L.trial_plan(tr))["left"], [])
        st2.update(_step_in(when, "down to the strap", 0.5), 0.0, np.zeros(3),
                   state=_TieAt(tg["regrasp"], straight + np.array([0.013, -0.057, -0.031])))
        assert np.allclose(st2.shift["strap"], [0.006, -0.012, -0.020]) and st2.want["strap"][1] == pytest.approx(-0.057)
        # the tail: the strap's point b_gft from its tip, sideways only
        lying = tg["tail"] + np.outer(np.linspace(-0.15, tr.b_gft, 33), [0.0, 1.0, 0.0]) + np.array([0.003, 0.010, 0.0])
        st.update(_step_in(when, "down to the tail", 0.4), 0.0, np.zeros(3), state=_TieAt(tg["head"], lying))
        assert np.allclose(np.asarray(w["take tail"].pos) - base["take tail"], [0.003, 0.010, 0.0], atol=1e-6)
        # the line-up: against the tip's offset, capped
        st.update(_step_in(when, "align", 0.2), 0.0, np.array([0.05, 0.0, 0.0]), state=_TieAt(tg["head"], lying))
        assert np.asarray(w["align"].pos)[0] - base["align"][0] == pytest.approx(-L.B_STEER_CAP[0])
    finally:
        M.stop_demo()


def test_the_head_snap_reads_the_head_before_the_fingers_reach_it_and_then_holds():
    """Round 22 (job 96): on the heavier strap the head rose 7.7 mm once the left alone held it, and the 6 mm cap left the
    right's fingers under it; on stiff15 a fingertip pushed the head 10 mm during "onto head" and the snap, still live,
    followed it. The snap now reads the head only while the right comes round behind it, up to 12 mm high."""
    tr = L.trial(THREADS[0])
    plan = M.demo_plan(tr)
    try:
        st = M.DEMO["steering"]
        tg = L.thread_targets(tr)
        when, t = {}, 0.0
        for w in plan["right"]:
            when.setdefault(w.label, t)
            t += w.duration
        base = {w.label: np.asarray(w.pos) for w in plan["right"] if w.pos is not None}
        w = {x.label: x for x in plan["right"]}
        poly = tg["regrasp"] + L.R_FREE @ L.ROOT + np.outer(np.linspace(0, L.TIE_L, 33), L.R_FREE[:, 0])
        risen = np.array([0.0, 0.0, 0.0077])
        st.update(_step_in(when, "behind head", 0.5), 0.0, np.zeros(3), state=_TieAt(tg["regrasp"] + risen, poly + risen))
        assert np.allclose(np.asarray(w["grip head"].pos) - base["grip head"], risen, atol=1e-9)
        pushed = risen + np.array([0.003, 0.010, -0.005])
        st.update(_step_in(when, "onto head", 0.8), 0.0, np.zeros(3), state=_TieAt(tg["regrasp"] + pushed, poly + pushed))
        assert np.allclose(np.asarray(w["grip head"].pos) - base["grip head"], risen, atol=1e-9)  # frozen
        st.update(_step_in(when, "around", 0.5), 0.0, np.zeros(3),
                  state=_TieAt(tg["regrasp"] + np.array([0.0, 0.0, 0.030]), poly))
        assert st.shift["head"][2] == pytest.approx(0.012)  # capped
    finally:
        M.stop_demo()


def test_the_right_holds_still_until_the_left_is_off_the_strap():
    """Round 22 (job 96): the right began its carry as the left let go of the strap and the head squeezed out."""
    for name in THREADS:
        tr = L.trial(name)
        plan = L.trial_plan(tr)
        lt, rt = _times(plan, "left"), _times(plan, "right")
        carry = "carry head" if tr.b_slide_first else "carry tie"  # round 24: the slide came before the change of grips
        assert lt["up from strap"][1] <= rt[carry][0] + 1e-9, name
        assert all(s[2] == "head_open" for s in plan["right"] if s[4] in ("around", "behind head", "onto head")), name


@pytest.mark.parametrize("name", THREADS)
def test_stage_b_fingers_stay_over_the_console(name):
    """Round 23 holds the head 33 mm up, the right's fingers tilted 5 degrees: their base (the fat envelope's third box)
    hangs lowest. No finger envelope of either hand goes into the console at any stage B waypoint (the tail's pick and
    stage A's grips come within 0.3-1.3 mm by design)."""
    plan = L.trial_plan(L.trial(name))
    for arm, steps in plan.items():
        for s in steps:
            if isinstance(s[3], str):
                continue
            gap = (L.HEAD[1] if s[2] == "head" else L.W if (arm, s[2]) == ("left", "closed")
                   else L.FINGER_GAP[s[2]] * L.MM)
            for c, r, h in L.finger_boxes(s[1], s[3], gap):
                low = min((c + r @ (np.array(sg) * h))[2] for sg in itertools.product((-1, 1), repeat=3))
                assert low >= L.TOP, (arm, s[4], round((low - L.TOP) * 1000, 1))


def test_the_strap_need_is_the_taut_path_under_and_round_the_bundle():
    """strap_need: from a root level with the bundle's top, 50 mm off, to a point straight over the root, the strap
    goes down under the cords and back over them: longer than the way out and back; to a point on the console past the
    bundle, it runs flat under it."""
    ring = L.lift_bundle_ring()
    y0, y1, top = ring[:, 0].min(), ring[:, 0].max(), ring[:, 1].max()
    root = np.array([y0 - 0.050, top])
    flat = L.strap_need(root, np.array([y1 + 0.050, L.TOP + L.T / 2]))
    assert flat == pytest.approx(np.linalg.norm(root - [y0, L.TOP + L.T / 2]) + (y1 - y0) + 0.050, abs=1e-6)
    over = L.strap_need(root, root + np.array([0.0, 0.040]))
    assert over > 2 * (y1 - y0) + 2 * 0.050


@pytest.mark.parametrize("name", THREADS)
def test_stage_b_leaves_strap_to_spare_at_every_step_the_left_holds_the_tail(name):
    """Round 23 (job 97): round 22's plan needed 29-88 mm more strap than the tie has from the head round the bundle to
    the left's fingers; the left dragged the cords up and pulled the head out of the right's fingers. Every point the
    left holds the tail at leaves B_SLACK_MIN to spare, the cords lying low or propped up; on the way up from the pick
    (where the strap lies flat, nothing to spare) the lift only ever adds slack."""
    tr = L.trial(name)
    pts = L.tail_held_points(L.trial_plan(tr)["left"])
    assert {lab for lab, _ in pts} == set(L.B_HOLD_LABELS)
    lift = [L.strap_slack(tr, p) for lab, p in pts if lab == "lift tail"]
    assert lift[0] >= 0.0 and all(b >= a for a, b in zip(lift, lift[1:])) and lift[-1] >= L.B_SLACK_MIN, lift
    for lab, p in pts:
        if lab != "lift tail":
            s = L.strap_slack(tr, p)
            assert s >= L.B_SLACK_MIN, (lab, np.round(p * 1000, 1), round(s * 1000, 1))


def test_round_22s_threading_runs_out_of_strap():
    """The check fails round 22's layout: the head 55 mm off the bundle and 32 mm over it, the tail held 50 mm from its
    tip, lifted straight up, over the bundle 120 mm up."""
    old = dataclasses.replace(L.trial("b-thread"), b_head=(0.330, 0.045, L.TOP + 0.060), b_gft=0.050, b_lift=(0.0, 0.060),
                              b_over=(0.330, 0.125, L.TOP + 0.120), b_out=0.022)
    worst = min(L.strap_slack(old, p) for _, p in L.tail_held_points(L.trial_plan(old)["left"]))
    assert worst < -0.020


def test_the_longer_ties_are_the_lift_start_with_more_of_the_same_strap():
    """Round 24: lift25 and lift30 are the lift start with a 250 and a 300 mm tie, 40 and 48 links of the same 6.25 mm
    (so the same stiffness per joint and mass per link), the head where lift's is; the jig's tie is still 200 mm. With
    the head and the pawl the tie stays inside PhysX's 64 links per articulation."""
    lift = L.BUILDS["lift"]
    for name, n in (("lift25", 40), ("lift30", 48)):
        b = L.BUILDS[name]
        assert b.n_links == n and b.tie_l == pytest.approx(n * L.LINK) and n + 2 <= 64
        assert b.head_pos == lift.head_pos and np.allclose(b.head_rot, lift.head_rot)
        assert np.allclose(b.points[: lift.n_links + 1], lift.points)
        tie = M.build_tie(name)
        assert len(tie.points) == n + 1
    assert L.BUILDS["jig"].tie_l == pytest.approx(L.TIE_L) and L.BUILDS["lift"].tie_l == pytest.approx(L.TIE_L)


SLIDE_FIRST = [t.name for t in L.CHECKED if t.stage == "thread" and t.b_slide_first and not t.b_keys and not t.b_wedge
               and not t.b_cinch]


@pytest.mark.parametrize("name", SLIDE_FIRST)
def test_the_tail_comes_down_onto_the_slot_with_strap_for_a_smooth_turn(name):
    """Round 24 (job 98): with 14 mm to spare the strap turned from heading -y to heading down at the one joint where it
    left the left's pads (to its 100-degree stop) and the rest slipped in them. From above the head to the end of the
    push the plan leaves B_TURN_SLACK, room for the turn to happen above the fingers; round 23's plan doesn't."""
    def worst(tr):
        pts = L.tail_held_points(L.trial_plan(tr)["left"])
        return min(L.strap_slack(tr, p) for lab, p in pts if lab in ("align", "push", "push hold"))
    assert worst(L.trial(name)) >= L.B_TURN_SLACK
    assert worst(L.trial("b-thread")) < L.B_TURN_SLACK / 2


@pytest.mark.parametrize("name", SLIDE_FIRST)
def test_slide_first_changes_grips_clear_of_the_cords_after_the_slide(name):
    """Round 24's order: lift and slide (the pick's down grip) before the change of grips, which happens with the head
    80 mm or more from the bundle and the left's fingers on the strap 30 mm or more from it; the left has let go of
    the cords before the right raises the head; the strap is under the cords before the right carries the head on."""
    tr = L.trial(name)
    plan = L.trial_plan(tr)
    lt, rt = _times(plan, "left"), _times(plan, "right")
    near = min(L.LIFT_HANG_Y - off - r for _, r, off, _, _ in L.LIFT_CORDS)
    tg = L.thread_targets(tr)
    assert near - tg["regrasp"][1] >= 0.080 and near - tg["strap"][1] >= 0.030
    assert rt["hold tie"][1] <= lt["lower cords"][0] + 1e-9
    assert lt["clear cords"][1] <= rt["to regrasp"][0] + 1e-9
    assert rt["slide under"][1] <= rt["release pick"][0] and rt["grip head"][1] <= rt["carry head"][0]
    tip_y = tg["regrasp"][1] + L.tip_from_head(L.BUILDS[tr.build])  # the tip, the strap straight, at the slide's end
    assert tip_y - max(L.LIFT_HANG_Y - off + r for _, r, off, _, _ in L.LIFT_CORDS) >= 0.080


def test_enclosed_lift_finds_a_cord_through_a_loop_and_not_one_beside_it():
    hc = np.array([0.33, 0.05, L.TOP + 0.06])
    q = L.matrix_to_quat_xyzw(L.R_FREE)
    root = hc + L.R_FREE @ L.ROOT
    entry = hc + L.R_FREE @ np.array([L.SLOT_C, 0.0, L.ENTRY_Z + 0.002])
    exit_ = hc + L.R_FREE @ np.array([L.SLOT_C, 0.0, L.EXIT_Z - 0.010])
    centre = np.array([0.33, 0.075, L.TOP + 0.06])
    a = np.linspace(-math.pi / 2, math.pi, 30)  # from below the root, round the far side and over, to above the entry
    ring = centre + 0.02 * np.column_stack([np.zeros_like(a), np.cos(a), np.sin(a)])
    poly = np.vstack([root, ring, entry, exit_])
    cord_in = {"in": np.array([[0.20, 0.075, L.TOP + 0.06], [0.45, 0.075, L.TOP + 0.06]])}
    cord_out = {"out": np.array([[0.20, 0.140, L.TOP + 0.06], [0.45, 0.140, L.TOP + 0.06]])}
    assert L.enclosed_lift(poly, hc, q, {**cord_in, **cord_out}) == ("in",)


# -- round 21: the arms against each other ----------------------------------------------------------------------------
def _plan_key(tr):
    return json.dumps({arm: [[round(float(x), 5) for x in (st[1] or ())] + [st[2], st[4]] for st in steps]
                       for arm, steps in L.trial_plan(tr).items()})


_ARMS_SEEN: dict = {}


@pytest.mark.parametrize("name", THREADS)
def test_stage_b_arms_keep_clear_of_each_other(name):
    """Job 95: the left's down grip on the strap 82 mm from the right's on the head put the two forearms (link4) 3 mm
    apart; in the simulator the left pushed the right 57 mm and the head fell. Both arms' meshes, replayed together,
    must stay 10 mm apart over the whole plan (subsampled meshes: the real gap is a little smaller)."""
    import armcheck

    tr = L.trial(name)
    key = _plan_key(tr)
    if key not in _ARMS_SEEN:  # the thread trials differ only in physics: one replay per distinct plan
        _ARMS_SEEN[key] = armcheck.arms_clearance(M, L.trial_plan(tr))
    d, t, ll, lr, la, lb = _ARMS_SEEN[key]
    assert d > 0.010, (name, round(d * 1000, 1), t, ll, lr, la, lb)


@pytest.mark.parametrize("name", ["b-thread", *{L.trial(n).build: n for n in SLIDE_FIRST[::-1]}.values()])  # one a build
def test_round_20s_grip_change_fails_the_arm_check(name):
    """The check catches job 95's layout (the left's down grip 76 mm out) and passes round 21's (45 degrees, 50 mm),
    the right's closed-loop head and the left's strap snap at the corners of B_SNAP included; round 24's grip change
    (after the slide, b_regrasp's y) passes it too."""
    import armcheck

    tr = L.trial(name)
    plan = L.trial_plan(tr)
    t = _times(plan, "left")
    t0, t1 = t["to the strap"][0], t["up from strap"][1]
    if name == "b-thread":
        old = dataclasses.replace(tr, b_hold_s=0.076, b_hold_phi=90.0)
        assert armcheck.arms_clearance(M, L.trial_plan(old), t0=t0, t1=t1)[0] < 0.005
    for k in range(2):
        snap = {key: L.B_SNAP[key][k] for key in ("strap", "head")}

        def shift(arm, w, snap=snap):
            if arm == "left" and w.label in L.THREAD_STRAP_LABELS:
                return snap["strap"]
            return snap["head"] if arm == "right" and w.label in L.THREAD_HEAD_LABELS else 0.0

        assert armcheck.arms_clearance(M, plan, t0=t0, t1=t1, shift=shift)[0] > 0.010, k


@pytest.mark.parametrize("name", THREADS)
def test_the_strap_past_the_lefts_hold_clears_its_palm(name):
    """Round 21's hold points the left at the head, tilted: the strap runs on past its fingertips toward the tip and
    must pass under the palm and wrist (flat, or drooping up to 20 degrees), 20 mm clear of them."""
    import armcheck
    from yam_arena.urdf import forward_kinematics

    tr = L.trial(name)
    ls = {s[4]: s for s in L.trial_plan(tr)["left"]}
    tcp, rot = np.asarray(ls["hold strap"][1], float), ls["hold strap"][3]
    q, pos_err, _ = ik.solve(tcp, ik.matrix_to_quat_xyzw(rot), np.array(READY, float), base_pos=BASES["left"])
    assert pos_err < 1e-3
    poses = forward_kinematics(Path(ik.__file__).parent / "assets" / "yam" / "yam.urdf",
                               {f"joint{i + 1}": float(q[i]) for i in range(6)} | {"joint7": 0.0, "joint8": 0.0})
    strap = tcp + L.GRIP_DEPTH * rot[:, 2]
    for slope in (0.0, 10.0, 20.0):
        s_ = math.radians(slope)
        line = strap + np.outer(np.linspace(0.0, 0.15, 151), [0.0, math.cos(s_), -math.sin(s_)])
        for link in ("gripper", "link5", "link4"):
            pts = armcheck._meshes()[link] @ poses[link][:3, :3].T + poses[link][:3, 3] + np.asarray(BASES["left"])
            d = float(np.sqrt(((line[:, None] - pts[None]) ** 2).sum(-1)).min()) - L.W / 2
            assert d > 0.020, (slope, link, round(d * 1000, 1))


# -- round 26: the scripted finish (the cinch) --------------------------------------------------------------------------
class _Tie:
    def __init__(self, pos, quat, polyline):
        self.pos, self.quat_xyzw, self.polyline = np.asarray(pos, float), np.asarray(quat, float), np.asarray(polyline, float)


def _play_cinch(name, aim0, err, head_off=(0.0, 0.0, 0.0)):
    """Play a cinch trial's demo plan with CinchSteering fed a made-up tie: the head where the right holds it (off
    by ``head_off``), the tail a straight stick from the left's fingers, aimed ``aim0`` below level when the pinch
    closes and then turning with the wrist less ``err``. Returns (the steering, each arm's worst IK error in mm)."""
    from yam_arena.demo import ArmPlan

    tr = L.trial(name)
    plan = M.demo_plan(tr)
    try:
        st = M.DEMO["steering"]
        arms = {arm: ArmPlan(ws, BASES[arm]) for arm, ws in plan.items()}
        hc = np.asarray(tr.b_head, float) + head_off
        through = 0.0
        for k in range(1, int(L.plan_seconds(L.trial_plan(tr)) * M.CONTROL_HZ) + 1):
            t = k / M.CONTROL_HZ
            lab = {arm: st.label(t, ws) for arm, ws in st.arms.items()}
            ltcp, lrot, _, _ = arms["left"].target(t)
            held = ltcp + L.B_TAIL_DEPTH * lrot[:, 2]
            aim = aim0
            if t >= st.when["left"]["turn tail"]:
                aim = aim0 + (math.degrees(math.atan2(-lrot[:, 2][2], -lrot[:, 2][1])) - tr.b_feed_phi) - err
            u = np.array([0.0, -math.cos(math.radians(aim)), -math.sin(math.radians(aim))])
            tip = held + u * (tr.b_gft - 0.004)
            poly = np.vstack([hc + [0.0, 0.3, -0.02], held - 0.05 * u, held, tip - 0.03 * u, tip - 0.02 * u, tip - 0.01 * u, tip])
            if lab["right"] in ("pull", "pull hold"):
                s = min(max((t - st.when["right"]["pull"]) / tr.b_pull_s, 0.0), 1.0)
                through = min(0.034 + tr.b_pull * s * s * (3 - 2 * s) * 0.97, 0.1905)
            elif t >= st.when["left"]["push hold"]:
                through = 0.030
            f = L.cinch_frame(hc, st.pitch)
            off = tip - f["entry"]
            tip_err = off - (off @ f["d"]) * f["d"] if lab["left"] == "align" else np.zeros(3)
            st.update(k + 1, through, tip_err, state=_Tie(hc, L.matrix_to_quat_xyzw(L.head_rot_pitched(st.pitch)), poly))
            for p in arms.values():
                p.action7(t)
        return st, {arm: p.worst_ik_mm for arm, p in arms.items()}
    finally:
        M.stop_demo()


@pytest.mark.parametrize("name", CINCH[:1])
@pytest.mark.parametrize("aim0, err, head_off", [(15.0, 0.0, (0.0, 0.0, 0.0)), (-15.0, 5.0, (0.0, 0.004, -0.003)),
                                                 (-50.0, 0.0, (0.0, 0.0, 0.006)), (35.0, -5.0, (0.0, -0.004, 0.0))])
def test_the_cinch_closed_loop_turns_wrist_and_head_onto_the_tail_and_stays_in_reach(name, aim0, err, head_off):
    """CinchSteering: the wrist turns the tail on toward B_AIM (0 to 45 degrees, no further: its hand would stand
    beside the right's), the head's pitch faces the tail as measured after the turn (30 to 75 degrees), the pull stops
    with the full mark 4 mm out, and every target stays within the arms' reach."""
    tr = L.trial(name)
    st, ik_mm = _play_cinch(name, aim0, err, head_off)
    assert max(ik_mm.values()) < 1.0, ik_mm
    assert st.turn_deg == pytest.approx(float(np.clip(L.B_AIM - aim0, *L.CINCH_TURN)))
    after = aim0 + st.turn_deg - err
    assert st.aims == [round(aim0, 1), pytest.approx(after, abs=0.6)]
    assert st.pitch == pytest.approx(float(np.clip(90.0 - after, L.CINCH_PITCH_MIN, L.CINCH_PITCH_MAX)), abs=0.7)
    assert L.CINCH_PITCH_MAX + L.B_GRIP_TILT <= 90.0  # the right's hand never tips past straight down
    assert tr.b_feed_phi + L.CINCH_TURN[1] <= 55.0 and tr.b_feed_phi + L.CINCH_TURN[0] >= 10.0
    assert st.freeze is not None and st.back > 0.0  # the pull stopped short of its planned length
    full = L.lift_marks(L.BUILDS[tr.build].tie_l)[0]
    assert 0.1905 >= full + st.PAST and full + st.PAST > 0.1875 + 0.0005  # past the joint it rests at


@pytest.mark.parametrize("name", CINCH)
def test_the_cinch_hands_take_turns_so_the_tie_is_always_held(name):
    tr = L.trial(name)
    plan = L.trial_plan(tr)
    lt, rt = _times(plan, "left"), _times(plan, "right")
    ls, rs = {s[4]: s for s in plan["left"]}, {s[4]: s for s in plan["right"]}
    assert rt["level head"][1] <= lt["to the tail"][0] + 1e-9  # the head in place before the tail is taken
    assert ls["over"][2] == ls["top"][2] == ls["feed"][2] == tr.b_soft and ls["settle"][2] == "closed"  # eased only
    #                                                                                          on the way over
    # the tail goes up over the bundle before it comes down past its far side (taken straight there, the strap arches
    # and the tail points away from the head); the head waits low and level under it, then turns to face it
    held = {k: np.asarray(ls[k][1]) + L.B_TAIL_DEPTH * ls[k][3][:, 2] for k in ("lift tail", "over", "top", "feed")}
    far = max(L.LIFT_HANG_Y - off + r for _, r, off, _, _ in L.LIFT_CORDS)
    assert held["top"][2] - L.TOP >= 0.110 and abs(held["top"][1] - far) <= 0.010 and held["feed"][1] - far >= 0.025
    assert held["top"][2] > held["over"][2] > held["lift tail"][2] and held["feed"][2] < held["over"][2]
    assert rs["level head"][3][:, 2][2] > -0.2 and tr.b_wait_pitch == -10.0  # the right's hand near level, behind the head
    assert np.allclose(rs["face tail"][3], L.back_rot(L.B_GRIP_TILT + tr.b_insert_pitch))
    assert lt["settle 2"][1] <= rt["face tail"][0] + 1e-9 and rt["face tail"][1] <= lt["push"][0] + 1e-9
    assert lt["push hold"][1] <= rt["release head"][0] + 1e-9  # the tail in and held before the right lets go
    assert rt["take tail 2"][1] <= lt["let go strap 2"][0] + 1e-9  # the right has the tail before the left lets go
    assert lt["grip head 2"][1] <= rt["raise tail"][0] + 1e-9  # the left has the head before the pull
    assert rt["raise tail"] == pytest.approx(lt["raise head"])  # both raise the tie together
    assert rt["let go tail 2"][1] <= lt["lower head"][0] + 1e-9
    for arm, labels in (("left", ("take tail", "lift tail", "settle", "turn tail", "align", "push", "push hold",
                                  "grip head 2", "raise head", "hold head", "lower head")),
                        ("right", ("level head", "face tail", "take tail 2", "raise tail", "pull", "pull hold"))):
        steps = {s[4]: s for s in plan[arm]}
        for lab in labels:
            assert steps[lab][2] in ("closed", "head"), (arm, lab)
    # one push leaves the tail far enough out for the right's pads, past the latch
    assert L.CINCH_OUT >= tr.b_tail_at + 1.8 * L.CINCH_PAD and L.CINCH_OUT > tr.latch_mm == L.LIFT_LATCH_AT
    f = L.cinch_frame(tr.b_head, tr.b_insert_pitch)
    assert tr.b_insert_pitch == pytest.approx(90.0 - L.B_AIM) and f["d"][1] < 0 and f["d"][2] < 0  # in toward -y, down
    tip = np.asarray(ls["push"][1]) + L.B_TAIL_DEPTH * ls["push"][3][:, 2] + tr.b_gft * f["d"]
    assert np.allclose(tip, f["exit"] + f["d"] * L.CINCH_OUT, atol=1e-9)
    assert L.plan_seconds(plan) * L.CONTROL_HZ < 2400  # within the first submission's limit, well inside 3,600


@pytest.mark.parametrize("name", CINCH[:1])
def test_the_cinch_fingers_keep_clear_of_each_other_and_the_console(name):
    """The fat finger envelopes: the left at the end of its push (the closed loop's extra depth included) against the
    right holding the head; the right on the tail against the left on the head; nothing in the console."""
    tr = L.trial(name)
    plan = L.trial_plan(tr)
    ls, rs = {s[4]: s for s in plan["left"]}, {s[4]: s for s in plan["right"]}
    f = L.cinch_frame(tr.b_head, tr.b_insert_pitch)
    right = L.finger_boxes(rs["face tail"][1], rs["face tail"][3], L.HEAD[1])
    pushed = np.asarray(ls["push"][1]) + f["d"] * L.CINCH_DEPTH_CAP
    assert _gap_between(L.finger_boxes(pushed, ls["push"][3], L.W), right) > 0.004
    on_tail = L.finger_boxes(rs["take tail 2"][1], rs["take tail 2"][3], L.W)
    on_head = L.finger_boxes(ls["grip head 2"][1], ls["grip head 2"][3], L.HEAD[1])
    assert _gap_between(on_tail, on_head) > 0.003
    for arm, steps in plan.items():
        for s in steps:
            if isinstance(s[3], str):
                continue
            gap = (L.HEAD[1] if s[2] == "head" else L.W if s[2] in ("closed", "soft", "soft3", "soft5")
                   else L.FINGER_GAP[s[2]] * L.MM)
            for c, r, h in L.finger_boxes(s[1], s[3], gap):
                low = min((c + r @ (np.array(sg) * h))[2] for sg in itertools.product((-1, 1), repeat=3))
                assert low >= L.TOP, (arm, s[4], round((low - L.TOP) * 1000, 1))


@pytest.mark.parametrize("name", CINCH[:1])
def test_the_cinch_arms_keep_clear_of_each_other(name):
    import armcheck

    d, t, ll, lr, la, lb = armcheck.arms_clearance(M, L.trial_plan(L.trial(name)))
    assert d > 0.010, (name, round(d * 1000, 1), t, ll, lr, la, lb)


def test_the_cinch_trials_differ_only_in_the_pinch_on_the_way_over():
    base = L.trial(CINCH[0])
    for name in CINCH[1:]:
        tr = L.trial(name)
        assert {k: v for k, v in vars(tr).items() if k not in ("name", "why", "b_soft")} == \
            {k: v for k, v in vars(base).items() if k not in ("name", "why", "b_soft")}
    assert [L.trial(n).b_soft for n in CINCH] == ["soft", "soft3", "soft5"]
    assert [M.FINGERS[k] for k in ("soft", "soft3", "soft5")] == pytest.approx([-0.002, -0.0016, -0.0022])


# -- round 29: the wedge (the tail locked tip-down in the pinch, then threaded) -------------------------------------------
def _play_wedge(name, offsets, head_off=(0.0, 0.0, 0.0), pitch_off=0.0):
    """Play a wedge trial's demo plan with WedgeSteering fed a made-up tie: the head where the right holds it (off by
    ``head_off``, pitched ``pitch_off`` more), the tail a straight stick from the left's fingers that sits
    ``offsets[0]`` degrees off the wrist's line until the left is over the bundle and ``offsets[1]`` from then on.
    Returns (the steering, each arm's worst IK error in mm, the tip at the end of the line-up)."""
    from yam_arena.demo import ArmPlan

    tr = L.trial(name)
    plan = M.demo_plan(tr)
    try:
        st = M.DEMO["steering"]
        arms = {arm: ArmPlan(ws, BASES[arm]) for arm, ws in plan.items()}
        hc = np.asarray(tr.b_head, float) + head_off
        hr = L.head_rot_pitched(tr.b_insert_pitch + pitch_off)
        tip_at_align = None
        for k in range(1, int(L.plan_seconds(L.trial_plan(tr)) * M.CONTROL_HZ) + 1):
            t = k / M.CONTROL_HZ
            lab = st.label(t, st.arms["left"])
            ltcp, lrot, _, _ = arms["left"].target(t)
            held = ltcp + L.B_TAIL_DEPTH * lrot[:, 2]
            phi = math.degrees(math.atan2(-lrot[:, 2][2], -lrot[:, 2][1]))
            a = math.radians(phi - 90.0 + offsets[0 if t < st.when["left"]["settle 2"] else 1])
            u = np.array([0.0, math.cos(a), math.sin(a)])
            tip = held + u * tr.b_gft
            poly = np.vstack([hc + [0.0, 0.3, -0.02], held - 0.05 * u, held, tip - 0.02 * u, tip - 0.01 * u, tip])
            if lab == "align":
                tip_at_align = tip
            st.update(k + 1, 0.0, np.zeros(3), state=_Tie(hc, L.matrix_to_quat_xyzw(hr), poly))
            for p in arms.values():
                p.action7(t)
        return st, {arm: p.worst_ik_mm for arm, p in arms.items()}, tip_at_align, (hc, hr)
    finally:
        M.stop_demo()


@pytest.mark.parametrize("name", WEDGE[:2])
@pytest.mark.parametrize("offsets, head_off, pitch_off", [
    ((-48.0, -48.0), (0.0, 0.0, 0.0), 0.0), ((-30.0, -42.0), (0.002, 0.004, -0.003), 6.0),
    ((-55.0, -50.0), (0.0, -0.004, 0.004), -5.0), ((10.0, 48.0), (0.0, 0.0, 0.0), 0.0)])
def test_the_wedge_closed_loop_points_the_tail_down_the_slot_and_stays_in_reach(name, offsets, head_off, pitch_off):
    """WedgeSteering: whatever angle the strap locks at in the pinch, the wrist's tilt is set so the tail points down
    the slot of the head as it is really held (within WEDGE_PHI), the tip lines up B_TIP_GAP before the entry, and
    every target stays within the arms' reach."""
    tr = L.trial(name)
    st, ik_mm, tip, (hc, hr) = _play_wedge(name, offsets, head_off, pitch_off)
    assert max(ik_mm.values()) < 1.0, ik_mm
    d = hr @ np.array([0.0, 0.0, -1.0])
    entry = hc + hr @ np.array([L.SLOT_C, 0.0, L.ENTRY_Z])
    assert [r["offset_deg"] for r in st.reads] == pytest.approx(list(offsets), abs=0.3)
    want = 90.0 + L.heading_deg(d) - offsets[1]
    assert st.reads[1]["phi_deg"] == pytest.approx(float(np.clip(want, *L.WEDGE_PHI)), abs=0.3)
    assert np.allclose(tip, entry - d * L.B_TIP_GAP, atol=3e-4)  # the tip where the line-up wants it
    if L.WEDGE_PHI[0] < want < L.WEDGE_PHI[1]:  # ... and the tail along the slot
        lrot = next(w for w in st.arms["left"] if w.label == "align").rot
        phi = math.degrees(math.atan2(-lrot[:, 2][2], -lrot[:, 2][1]))
        assert phi - 90.0 + offsets[1] == pytest.approx(L.heading_deg(d), abs=0.5)


@pytest.mark.parametrize("name", WEDGE)
def test_the_wedge_plan_arches_the_tail_before_it_lifts_and_threads_from_over_the_bundle(name):
    tr = L.trial(name)
    plan = L.trial_plan(tr)
    lt, rt = _times(plan, "left"), _times(plan, "right")
    ls, rs = {s[4]: s for s in plan["left"]}, {s[4]: s for s in plan["right"]}
    tg = L.thread_targets(tr)
    held = {k: np.asarray(ls[k][1]) + L.B_TAIL_DEPTH * ls[k][3][:, 2] for k in
            ("take tail", "arch", "lift tail", "settle", "over", "settle 2", "above head", "align", "push")}
    assert rt["level head"][1] <= lt["to the tail"][0] + 1e-9  # the head in place before the tail is taken
    assert np.allclose(held["take tail"], tg["tail"]) and ls["take tail"][2] == "closed"
    # pushed toward the bundle along the console (the strap behind arches up), then lifted
    assert held["arch"][1] == pytest.approx(held["take tail"][1] - tr.b_arch) and tr.b_arch >= 0.030
    assert 0.001 <= held["arch"][2] - held["take tail"][2] <= 0.003
    assert held["lift tail"][2] - held["arch"][2] >= 0.045 and np.allclose(held["settle"], held["lift tail"])
    # the strap behind stays an arch while the tail is lifted (pulled straight, it locks tip-up again) ...
    assert min(L.strap_slack(tr, held[k]) for k in ("arch", "lift tail", "over", "above head", "align", "push")) >= 0.040
    # ... and the lifted tail, turned WEDGE_DEG down in the pinch, keeps its tip off the console
    assert held["lift tail"][2] - L.TOP - tr.b_gft * math.sin(math.radians(L.WEDGE_DEG)) >= 0.010
    far = max(L.LIFT_HANG_Y - off + r for _, r, off, _, _ in L.LIFT_CORDS)
    assert held["lift tail"][1] - far >= 0.060  # the arch has room between the fingers and the bundle
    # the tail of the plan: locked WEDGE_DEG below the wrist's line, pointing down the slot from "over" on
    d, e = tg["through"], tg["entry"]
    phi = L.wedge_phi(d, -L.WEDGE_DEG)
    assert L.WEDGE_PHI[0] < phi < L.WEDGE_PHI[1]
    for k in ("over", "settle 2", "above head", "align", "push", "push hold", "let go tail"):
        assert np.allclose(ls[k][3], L.thread_left_rot(phi))
    assert np.allclose(held["over"], tr.b_over) and np.allclose(held["settle 2"], tr.b_over)
    assert np.allclose(held["align"] + tr.b_gft * d, e - d * L.B_TIP_GAP, atol=1e-9)
    assert np.allclose(held["push"] + tr.b_gft * d, e + d * (L.HEAD[2] + tr.b_out), atol=1e-9)
    assert tr.b_out + L.B_DEPTH_CAP <= tr.b_gft - 0.0262 - 0.002  # the left's pads stay over the right's plates
    assert tr.b_out >= tr.latch_mm + 0.004 and tr.latch_mm == L.LIFT_LATCH_AT  # one push passes the latch
    for lab in ("take tail", "arch", "lift tail", "settle", "over", "settle 2", "above head", "align", "push", "push hold"):
        assert ls[lab][2] == "closed", lab
    if tr.b_draw:  # round 30: the right has the tail below the head before the left lets go, and draws after
        assert lt["push hold"][1] <= rt["release head"][0] + 1e-9
        assert rt["take tail 2"][1] <= lt["let go tail"][0] + 1e-9 and lt["back off tail"][1] <= rt["pull"][0] + 1e-9
        gp = np.asarray(rs["take tail 2"][1]) + L.GRIP_DEPTH * rs["take tail 2"][3][:, 2]
        below = float((gp - (e + d * L.HEAD[2])) @ d)  # the strap's point in the right's pads, below the exit
        assert below == pytest.approx(tr.b_tail_at + L.CINCH_PAD) and below <= tr.b_out - 0.010  # pads 8 mm below
        #                                                    the exit, the strap's last 12 mm and more in them
        assert rs["take tail 2"][2] == rs["pull"][2] == rs["pull hold"][2] == "closed"
        drawn = np.asarray(rs["pull"][1]) - np.asarray(rs["take tail 2"][1])
        assert np.linalg.norm(drawn) == pytest.approx(tr.b_pull) and drawn[1] < 0 < drawn[2]
        full = L.lift_marks(L.BUILDS[tr.build].tie_l)[0]
        assert tr.b_pull >= full + 0.004 - tr.b_out  # far enough to bring the full mark out
        if tr.b_hold_cords:
            assert lt["pin cords"][1] <= rt["pull"][0] + 1e-9 and rt["let go tail 2"][1] <= lt["let go cords 2"][0] + 1e-9
    else:
        assert lt["push hold"][1] <= rt["let go head"][0] + 1e-9 and lt["let go tail"][1] <= rt["let go head"][0] + 1e-9
    assert L.plan_seconds(plan) * L.CONTROL_HZ < 2400


@pytest.mark.parametrize("name", WEDGE)
def test_the_wedge_fingers_and_arms_keep_clear(name):
    import armcheck

    tr = L.trial(name)
    plan = L.trial_plan(tr)
    ls, rs = {s[4]: s for s in plan["left"]}, {s[4]: s for s in plan["right"]}
    d = L.thread_targets(tr)["through"]
    right = L.finger_boxes(rs["level head"][1], rs["level head"][3], L.HEAD[1])
    pushed = np.asarray(ls["push"][1]) + d * L.B_DEPTH_CAP
    assert _gap_between(L.finger_boxes(pushed, ls["push"][3], L.W), right) > 0.001
    if tr.b_draw:  # the right on the tail below the head against the left still holding the strap above it
        on_tail = L.finger_boxes(rs["take tail 2"][1], rs["take tail 2"][3], L.W)
        assert _gap_between(on_tail, L.finger_boxes(ls["push"][1], ls["push"][3], L.W)) > 0.010
    for arm, steps in plan.items():
        for s in steps:
            if isinstance(s[3], str):
                continue
            gap = L.HEAD[1] if s[2] == "head" else L.W if s[2] == "closed" else L.FINGER_GAP[s[2]] * L.MM
            for c, r, h in L.finger_boxes(s[1], s[3], gap):
                low = min((c + r @ (np.array(sg) * h))[2] for sg in itertools.product((-1, 1), repeat=3))
                assert low >= L.TOP, (arm, s[4], round((low - L.TOP) * 1000, 1))
    gap, t, ll, lr, la, lb = armcheck.arms_clearance(M, plan)
    assert gap > 0.010, (name, round(gap * 1000, 1), t, ll, lr, la, lb)


@pytest.mark.parametrize("name, tight_at", [("w30-draw", None), ("w30-draw-pin", 0.150)])
def test_the_draw_stops_at_the_full_mark_or_when_the_tie_is_tight(name, tight_at):
    """Round 30's closed loop: the right's grip goes onto the strap's real point below the real exit; the draw stops
    with the full mark 4 mm out, or (a tie that tightens sooner) once the hand is 12 mm on with no more tail coming."""
    from yam_arena.demo import ArmPlan

    tr = L.trial(name)
    plan = M.demo_plan(tr)
    try:
        st = M.DEMO["steering"]
        arms = {arm: ArmPlan(ws, BASES[arm]) for arm, ws in plan.items()}
        hc = np.asarray(tr.b_head, float) + [0.002, -0.003, 0.004]
        hr = L.head_rot_pitched(tr.b_insert_pitch + 4.0)
        d = hr @ np.array([0.0, 0.0, -1.0])
        ex = hc + hr @ np.array([L.SLOT_C, 0.0, L.EXIT_Z])
        full = L.lift_marks(L.BUILDS[tr.build].tie_l)[0]
        t_pull, at_freeze, through = st.when["right"]["pull"], None, 0.0
        for k in range(1, int(L.plan_seconds(L.trial_plan(tr)) * M.CONTROL_HZ) + 1):
            t = k / M.CONTROL_HZ
            rtcp = arms["right"].target(t)[0]
            if t >= st.when["left"]["push hold"]:
                through = tr.b_out
            if t >= t_pull:  # the first 60 mm of the draw take up the loop, then the tail follows the hand
                hand = float(np.linalg.norm(rtcp - start))
                through = min(tr.b_out + max(hand - 0.060, 0.0), tight_at or 1.0)
            else:
                start = rtcp
            tip = ex + d * through
            poly = np.vstack([hc + [0.0, 0.3, -0.02], hc + [0.0, 0.2, 0.05], ex - d * 0.01, ex, tip])
            st.update(k + 1, through, np.zeros(3), state=_Tie(hc, L.matrix_to_quat_xyzw(hr), poly))
            if st.freeze is not None and at_freeze is None:
                at_freeze = (t, rtcp.copy(), through)
            for p in arms.values():
                p.action7(t)
        assert max(p.worst_ik_mm for p in arms.values()) < 1.0
        grip = next(w for w in st.arms["right"] if w.label == "take tail 2")
        want = ex + d * (tr.b_tail_at + L.CINCH_PAD)
        assert np.allclose(np.asarray(grip.pos) + L.GRIP_DEPTH * np.asarray(grip.rot)[:, 2], want, atol=1e-6)
        assert at_freeze is not None and st.why == ("tight" if tight_at else "mark")
        end = arms["right"].target(st.when["right"]["pull hold"] + 0.5)[0]
        assert np.linalg.norm(end - at_freeze[1]) < 0.004  # the hand stays where it stopped
        if tight_at:
            assert at_freeze[2] == pytest.approx(tight_at) and np.linalg.norm(at_freeze[1] - start) < 0.060 + tight_at
        else:
            assert full + st.PAST <= at_freeze[2] < full + st.PAST + 0.006
    finally:
        M.stop_demo()


# -- round 31: the hanging cords without the stand, the strap held by its faces -----------------------------------------
HANG = [t.name for t in L.CHECKED if t.stage == "hang"]


@pytest.mark.parametrize("name", HANG[:1])
def test_the_hang_start_lifts_the_tie_straight_turns_it_on_edge_and_brings_its_tail_to_the_front(name):
    from yam_arena.demo import ArmPlan

    tr = L.trial(name)
    b = L.BUILDS[tr.build]
    plan = L.trial_plan(tr)
    rs, ls = {s[4]: s for s in plan["right"]}, {s[4]: s for s in plan["left"]}
    rt, lt = _times(plan, "right"), _times(plan, "left")
    assert not b.fixed and tr.build in L.HANG_BUILDS and b.tie_l == pytest.approx(0.300)
    start = L.start_polyline(b)
    assert start[:, 0].max() < L.STRIP["x"][0] - 0.05 and np.allclose(start[:, 2], start[0, 2])  # flat, clear of the strip
    # both hands take the tie where it lies, lift it and turn it together: the strap between them keeps its length
    at, r0, o = L.hang_pick(b)
    for a, c in (("grip", "take tail"), ("lift", "lift tail"), ("turn on edge", "turn tail")):
        assert rt[a] == pytest.approx(lt[c])
        head = np.asarray(rs[a][1]) + rs[a][3] @ o
        pinch = np.asarray(ls[c][1]) + L.B_TAIL_DEPTH * ls[c][3][:, 2]
        assert np.linalg.norm(pinch - head) == pytest.approx(np.linalg.norm(
            L.ROOT + np.array([b.tie_l - tr.h_first, 0.0, 0.0])), abs=1e-6), a
    assert np.allclose(rs["turn on edge"][3][:, 2], (1.0, 0.0, 0.0)) and abs(rs["turn on edge"][3][:, 0][2]) == pytest.approx(1.0)
    assert np.allclose(ls["turn tail"][3], L.RY_ROLL @ ls["take tail"][3])
    assert lt["let go tail"][1] <= rt["out"][0] + 1e-9  # the left has let go before the right carries the ruler off
    # behind the cords: the ruler's tip starts short of them, its line passes 10 mm and more behind their backs
    back = max(c.xy[0] + c.diameter / 2 for c in L.CORDS)
    head = np.asarray(rs["behind"][1]) + rs["behind"][3] @ o
    root, u = head + (L.RY_ROLL @ b.head_rot) @ L.ROOT, (L.RY_ROLL @ b.head_rot)[:, 0]
    assert np.allclose(u, (0.0, 1.0, 0.0)) and (root + u * b.tie_l)[1] < min(c.xy[1] - c.diameter / 2 for c in L.CORDS) - 0.015
    assert root[0] - L.T / 2 - back >= 0.010 and L.WALL["x"][0] - root[0] >= 0.050
    # swung, the tail is in front of the cords' plane where the left reaches down to it, by its faces
    f = L.hang_ruler(b, tr.h_swing, L.TOP + tr.h_low, tr.h_root_back)
    tail = f["root"] + f["u"] * (b.tie_l - tr.h_gft)
    grip = np.asarray(ls["take tail 2"][1]) - np.array([0.0, 0.0, L.GRIP_DEPTH])
    assert np.allclose(grip, tail) and tail[0] < min(c.xy[0] for c in L.CORDS) - 0.10
    lr = ls["take tail 2"][3]
    assert np.allclose(lr[:, 2], (0.0, 0.0, -1.0)) and abs(float(lr[:, 0] @ f["u"])) < 1e-9  # closing across the strap
    assert ls["take tail 2"][2] == "closed" and rt["hold tie"][1] <= lt["take tail 2"][0] + 1e-9
    assert lt["take tail 2"][1] <= rt["release head"][0] + 1e-9  # the left has it before the right lets go
    # within reach all the way, and off the TV, the wall and the cords
    arms = {arm: ArmPlan(ws, BASES[arm]) for arm, ws in M.plan_waypoints(plan).items()}
    for k in range(1, int(L.plan_seconds(plan) * M.CONTROL_HZ) + 1):
        for p in arms.values():
            p.action7(k / M.CONTROL_HZ)
    assert max(p.worst_ik_mm for p in arms.values()) < 1.0 and max(p.worst_ik_deg for p in arms.values()) < 1.0


def test_the_hang_closed_loop_takes_the_tail_where_it_really_is():
    tr = L.trial(HANG[0])
    b = L.BUILDS[tr.build]
    plan = M.demo_plan(tr)
    try:
        st = M.DEMO["steering"]
        f = L.hang_ruler(b, tr.h_swing + 6.0, L.TOP + tr.h_low - 0.004, tr.h_root_back)  # swung 6 degrees more, 4 mm low
        poly = np.array([f["root"] + f["u"] * k * L.LINK for k in range(b.n_links + 1)])
        t = next(t0 for t0, w in zip(np.cumsum([0.0] + [w.duration for w in plan["left"]]), plan["left"])
                 if w.label == "to the tail 2") + 0.5
        st.update(int(t * M.CONTROL_HZ) // L.STEER_EVERY * L.STEER_EVERY, 0.0, np.zeros(3), state=_Tie(f["head"], (0, 0, 0, 1), poly))
        w = next(w for w in plan["left"] if w.label == "take tail 2")
        want = L.face_tcp(poly[0] + f["u"] * (b.tie_l - tr.h_gft))
        assert np.allclose(w.pos, want, atol=1e-6) and abs(float(np.asarray(w.rot)[:, 0] @ f["u"])) < 1e-6
    finally:
        M.stop_demo()


def test_the_hanging_cords_are_scored_by_the_marks_too():
    b = L.BUILDS["hang30"]
    full, half = L.lift_marks(b.tie_l)
    assert (full, half) == pytest.approx((0.300 - 10.5 * L.LINK, 0.300 - 13.5 * L.LINK))
    start = L.start_polyline(b)
    q = L.matrix_to_quat_xyzw(b.head_rot)
    assert L.hang_score(start, b.head_pos, q, 1.0, 0.0) == 0.0  # lying open on the console


# -- rounds 32-33: the hanging cords, the whole tie -----------------------------------------------------------------------
def _play_hang(name="h37-tie", tail_off=-3.0, pivot_to=8.0, taut_at=0.034, slot_off=0.0, tight_at=None):
    """Play a whole-tie trial's demo plan with HangSteering fed a made-up tie. Until the left has brought the tail
    in: a straight ruler from the head the right holds. Then: the head's centre where the right's fingers hold it; its
    strap leaving it 70 degrees left of the hand's approach (job 118, after the hand squared up) until the draw at
    the front has gone ``taut_at``, where it turns to ``pivot_to`` degrees off the strap's line from the cords and
    from then on turns with the hand, ``slot_off`` degrees out; the tail a straight stick from the left's fingers,
    ``tail_off`` degrees off the hand's strap axis; after the push, through the slot, following the right's draw
    less 50 mm (to ``tight_at`` at most). Returns (the steering, {(arm, label): worst IK mm, deg} over 1, the tip's
    offset from the slot's line at the line-up, the full mark)."""
    from yam_arena.demo import ArmPlan

    tr = L.trial(name)
    b = L.BUILDS[tr.build]
    plan = M.demo_plan(tr)
    try:
        st = M.DEMO["steering"]
        arms = {arm: ArmPlan(ws, BASES[arm]) for arm, ws in plan.items()}
        o = L.hang_pick(b)[2]
        full = L.lift_marks(b.tie_l)[0]
        f = L.hang_ruler(b, tr.h_swing + 4.0, L.TOP + tr.h_low - 0.003, tr.h_root_back)
        ruler = np.array([f["root"] + f["u"] * k * L.LINK for k in range(b.n_links + 1)])
        through, start, tip_off, per, rel, held = 0.0, None, None, {}, None, None
        regrip_at, let_go, shift, root_now = None, False, np.zeros(3), 0.0
        when = st.when
        t_front = when["right"]["hold head"]
        T = L.plan_seconds(L.trial_plan(tr))
        for k in range(1, int(T * M.CONTROL_HZ) + 1):
            t = k / M.CONTROL_HZ
            lab = {arm: st.label(t, ws) for arm, ws in st.arms.items()}
            ltcp, lrot, _, _ = arms["left"].target(t)
            rtcp, rrot, _, _ = arms["right"].target(t)
            if t < when["left"]["feed"]:
                state = _Tie(f["head"], L.matrix_to_quat_xyzw(f["rot"]), ruler)
            else:
                opened = "open head" in when["right"] and when["right"]["open head"] <= t < when["right"]["close head"] + 0.8
                if t < when["right"]["release head"] and not opened:
                    hc = rtcp + rrot @ o
                    hand = L.heading_xy(rrot[:, 2])
                    root = hand + 70.0
                    if "tauten" in when["right"] and t >= when["right"]["tauten"]:
                        drawn = float(np.linalg.norm(hc - (np.asarray(st.base["right"]["hold head"][0]) + st.base["right"]["hold head"][1] @ o)))
                        if rel is None and drawn >= taut_at:
                            rel = L.hang_leg(hc)[0] + pivot_to - hand  # from here the head turns with the hand
                        if rel is not None:
                            if regrip_at is not None:  # taken again: it turns with the hand from where it was let go
                                rel = regrip_at - hand
                                regrip_at = None
                            root = hand + rel
                    if held is not None and "close head" in when["right"] and t >= when["right"]["close head"] + 0.8:
                        hc = hc + shift  # the head stayed where it was let go; the hand came onto it there
                    hr = L.rot_z(root - 90.0 + slot_off) @ L.RY_ROLL @ np.asarray(b.head_rot, float)
                    held = (hc, hr)
                    root_now = root
                elif opened and not let_go:
                    let_go = True
                    held = (held[0] + np.array([0.003, -0.002, 0.0]), held[1])  # let go, the head shifts a little
                    regrip_at = root_now
                    shift = np.array([0.003, -0.002, 0.0])
                hc, hr = held
                d = hr @ np.array([0.0, 0.0, -1.0])
                ex = hc + hr @ np.array([L.SLOT_C, 0.0, L.EXIT_Z])
                en = hc + hr @ np.array([L.SLOT_C, 0.0, L.ENTRY_Z])
                grip = ltcp + L.GRIP_DEPTH * lrot[:, 2]
                y = lrot[:, 1]
                a = math.atan2(y[1], y[0]) + math.radians(tail_off)
                u = np.array([math.cos(a), math.sin(a), 0.0])
                tip = grip + u * tr.h_gft
                if t >= when["left"]["push hold"]:
                    through = max(through, tr.h_out)
                if lab["right"] in ("pull out", "pull", "pull hold"):
                    if start is None:
                        start = rtcp.copy()
                    through = min(tr.h_out + max(float(np.linalg.norm(rtcp - start)) - 0.05, 0.0), tight_at or 1.0)
                if through > 0:
                    poly = np.vstack([hc + [0.1, 0.0, 0.0], en - d * 0.05, en, ex, ex + d * through])
                else:
                    poly = np.vstack([hc + [0.1, 0.0, 0.0], grip - u * 0.05, grip, tip - u * 0.02, tip])
                if lab["left"] == "align":
                    off = tip - en
                    tip_off = off - (off @ u) * u  # off the tail's own line through the entry
                state = _Tie(hc, L.matrix_to_quat_xyzw(hr), poly)
            st.update(k + 1, through, np.zeros(3), state=state)
            for arm, pl in arms.items():
                before = (pl.worst_ik_mm, pl.worst_ik_deg)
                pl.worst_ik_mm = pl.worst_ik_deg = 0.0
                pl.action7(t)
                w = per.setdefault((arm, lab[arm]), [0.0, 0.0])
                w[0], w[1] = max(w[0], pl.worst_ik_mm), max(w[1], pl.worst_ik_deg)
                pl.worst_ik_mm, pl.worst_ik_deg = max(before[0], pl.worst_ik_mm), max(before[1], pl.worst_ik_deg)
        bad = {k: (round(v[0], 1), round(v[1], 1)) for k, v in per.items() if v[0] > 1.0 or v[1] > 1.0}
        return st, bad, tip_off, full
    finally:
        M.stop_demo()


@pytest.mark.parametrize("kw, why", [
    (dict(), "mark"), (dict(tail_off=3.0, pivot_to=18.0, taut_at=0.020, slot_off=-6.0), "mark"),
    (dict(pivot_to=30.0, taut_at=0.050, tight_at=0.26), "tight")])
@pytest.mark.parametrize("name", ["h37-tie", "h37-regrip", "h37-firm"])
def test_the_hang_closed_loop_draws_taut_turns_the_head_aims_the_tail_and_stops_the_draw(name, kw, why):
    """HangSteering on the 370 mm tie, fed a made-up tie: the draw at the front stops where the head has turned to
    within PIVOT_OK of the strap's line (or at its end), the right's hand turns back by what the slot is round of the
    tail's line (at most what it kept), the left's wrist goes no further round than tr.h_tail_max, the tip lines up on
    the entry, the right's grip goes onto the real tail below the real exit, the draw stops at the full mark or when
    the tie is tight, and every target is within reach."""
    tr = L.trial(name)
    st, bad, tip, full = _play_hang(name, **kw)
    assert not bad, bad
    taut, turn, aim = st.reads["taut"], st.reads["turn"], st.reads["aim"]
    if tr.h_regrip_yaw:
        assert np.linalg.norm(st.moved["head 2"]) <= 1000 * st.SNAP2 and "regrip" in st.reads
    assert taut["why"] in ("turned", "end") and 0.0 < taut["drawn_mm"] <= 1000 * tr.h_tauten + 0.1
    if taut["why"] == "turned":
        assert abs(taut["gap_deg"]) <= st.PIVOT_OK
    most = tr.h_carry_yaw + tr.h_regrip_yaw
    assert 0.0 <= turn["turn_deg"] <= most and turn["turn_deg"] == pytest.approx(
        float(np.clip(turn["wanted_deg"], 0.0, most)))
    assert aim["hand_deg"] <= tr.h_tail_max + 0.2 and abs(aim["turn_deg"]) <= st.TURN
    assert abs((aim["slot_deg"] - aim["tail_deg"] + 180.0) % 360.0 - 180.0) <= 26.0  # within the slot's play
    assert np.linalg.norm(tip) < 0.002  # the tip on the entry at the line-up
    assert st.why == why and st.freeze is not None
    assert np.linalg.norm(st.moved["stub"]) <= 1000 * st.SNAP2


def test_the_hang_tie_plan_wraps_the_head_round_threads_and_draws():
    tr = L.trial("h37-tie")
    b = L.BUILDS[tr.build]
    plan = L.trial_plan(tr)
    rs, ls = {s[4]: s for s in plan["right"]}, {s[4]: s for s in plan["left"]}
    rt, lt = _times(plan, "right"), _times(plan, "left")
    o = L.hang_pick(b)[2]
    assert b.tie_l == pytest.approx(59 * L.LINK) and len(M.build_tie(tr.build).points) == 60
    # the left has the tail, by its faces, before the right eases its squeeze and carries the head round the cords,
    # its hand still turned as it swung the ruler
    assert lt["feed"][1] <= rt["ease"][0] + 1e-9 and rs["ease"][2] == rs["to front"][2] == "head_light"
    assert np.allclose(rs["round 2"][3], L.rot_z(tr.h_carry_yaw) @ rs["pull"][3]) and tr.h_carry_yaw == tr.h_swing
    heads = [np.asarray(rs[k][1]) + rs[k][3] @ o for k in ("round 1", "round 2", "round 3", "to front")]
    cords = np.mean([c.xy for c in L.CORDS], axis=0)
    assert all(p[1] < cords[1] for p in heads[:3]) and [p[0] for p in heads] == sorted((p[0] for p in heads), reverse=True)
    assert heads[0][0] > cords[0] > heads[2][0] and min(np.linalg.norm(p[:2] - cords) for p in heads) >= 0.045
    # at the front: slack first (the strap's budget, with job 118's 45 mm for the loop's bends), then out along the
    # strap's line from the cords, then the hand turned back
    leg_deg, leg = L.hang_leg(heads[-1])
    feed = np.asarray(ls["feed"][1]) + L.GRIP_DEPTH * ls["feed"][3][:, 2]
    spare = b.tie_l - tr.h_gft - 0.045 - leg - np.linalg.norm(feed[:2] - (cords + [0.0, 0.008]))
    assert 0.015 <= spare <= tr.h_tauten - 0.010  # slack at the front, taut before the draw's end
    far = np.asarray(rs["tauten"][1]) + rs["tauten"][3] @ o
    away = (far - heads[-1])[:2] / np.linalg.norm((far - heads[-1])[:2])
    assert np.allclose(away, -np.array([math.cos(math.radians(leg_deg)), math.sin(math.radians(leg_deg))]), atol=1e-9)
    assert np.linalg.norm(far - heads[-1]) == pytest.approx(tr.h_tauten)
    assert np.allclose(rs["turn head"][3], L.rot_z(tr.h_carry_yaw - tr.h_turn_head) @ rs["pull"][3])
    assert rt["hold head 2"][1] <= lt["settle 3"][0] + 1e-9  # the head is where it will be before the left reads it
    # the tail's line as the left threads it: tr.h_tail_deg, the tip on the entry, pushed along that line
    lr = ls["push"][3]
    a = math.radians(tr.h_tail_deg)
    t_hat = np.array([math.cos(a), math.sin(a), 0.0])
    head = np.asarray(rs["hold head 2"][1]) + rs["hold head 2"][3] @ o
    hr = L.rot_z(tr.h_tail_deg) @ L.RY_ROLL @ np.asarray(b.head_rot, float)
    entry = head + hr @ np.array([L.SLOT_C, 0.0, L.ENTRY_Z])
    grip = np.asarray(ls["align"][1]) + L.GRIP_DEPTH * lr[:, 2]
    assert np.allclose(grip + t_hat * tr.h_gft, entry - t_hat * L.B_TIP_GAP, atol=1e-9)
    assert tr.h_tail_max - tr.h_tail_deg >= 2.0
    assert math.degrees(math.atan((L.PASSAGE_X[1] - L.PASSAGE_X[0] - L.T) / L.HEAD[2])) >= 25.0  # the slot's play
    pushed = np.asarray(ls["push"][1]) + L.GRIP_DEPTH * lr[:, 2] + t_hat * tr.h_gft
    assert np.allclose(pushed, entry + t_hat * (L.HEAD[2] + tr.h_out), atol=1e-9) and tr.h_out > tr.latch_mm + 0.01
    # the right takes the tail below the exit before the left lets go, then draws it far enough for the full mark
    assert lt["push hold"][1] <= rt["release head"][0] + 1e-9 and rt["take stub"][1] <= lt["let go tail 3"][0] + 1e-9
    assert rs["take stub"][2] == rs["pull"][2] == "closed" and lt["clear 2"][0] <= rt["pull out"][0] + 2.0
    drawn = np.asarray(rs["pull"][1]) - np.asarray(rs["take stub"][1])
    assert np.linalg.norm(drawn) >= L.lift_marks(b.tie_l)[0] + 0.004 - tr.h_out + 0.03 and drawn[1] < -0.2
    assert L.plan_seconds(plan) * L.CONTROL_HZ < 2400


def test_the_300_mm_whole_tie_trials_differ_only_in_the_squeeze():
    a, b = L.trial_plan(L.trial("h30-tie"))["right"], L.trial_plan(L.trial("h30-tie-firm"))["right"]
    assert [s[4] for s in a] == [s[4] for s in b]
    ra, rb = {s[4]: s for s in a}, {s[4]: s for s in b}
    assert ra["to front"][2] == "head_light" and rb["to front"][2] == "head"
    assert all(np.allclose(ra[k][1], rb[k][1]) for k in ra if ra[k][1] is not None)


def test_the_firm_trial_differs_only_in_the_squeeze_from_the_turn_back_on():
    a, b = L.trial_plan(L.trial("h37-tie"))["right"], L.trial_plan(L.trial("h37-firm"))["right"]
    assert [s[4] for s in a] == [s[4] for s in b]
    for x, y in zip(a, b):
        assert x[1] is None or np.allclose(x[1], y[1])
        if x[4] in ("settle 2b", "turn head", "hold head 2") or (x[4] == "wait" and x[2] == "head_light" and y[2] == "head"):
            assert (x[2], y[2]) == ("head_light", "head"), x[4]
        else:
            assert x[2] == y[2], x[4]


def test_a_strap_dragged_back_through_the_brake_can_come_forward_again():
    """Round 37 (job 121): the brake's mark follows a strap that is dragged back through it, and letting go starts the
    best pull afresh, so the next 0.4 mm forward frees the pawl and the brake does not come straight back on."""
    import labkit

    labkit._SPECS["tie"] = M.TIE
    try:
        M._new_trial()
        M.set_ratchet("latch")
        at = M.RATCHET["latch_at"]
        s = dataclasses.replace(labkit.start_state(M.TIE), pawl_angle=math.radians(56.0))  # the pawl on the strap
        M._ratchet(None, s, 0.0)
        M._ratchet(None, s, at + 0.016)  # 24 mm out: latched
        M._ratchet(None, s, at + 0.0157)  # backs out 0.3 mm: the brake
        assert M.SEEN["latched"] and M.SEEN["brake_on"] and M.SEEN["brakes"] == 1
        first = M.SEEN["brake_at"]
        assert first == pytest.approx(math.radians(56.0 + L.BRAKE_DEG)) and L.BRAKE_SEAT <= 56.0
        for through in (at + 0.010, at + 0.006):  # dragged 10 mm back through it
            M._ratchet(None, s, through)
        assert M.SEEN["brake_on"] and M.SEEN["brake_s"] == pytest.approx(at + 0.006)
        # still backing out, the brake has closed further each time (round 45), never past the pawl's own stop
        assert M.SEEN["brake_at"] == pytest.approx(first + 2 * math.radians(L.BRAKE_MORE)) and M.SEEN["brake_at"] <= L.PAWL_REST
        M._ratchet(None, s, at + 0.0062)  # 0.2 mm forward: still held
        assert M.SEEN["brake_on"]
        M._ratchet(None, s, at + 0.0066)  # 0.6 mm forward: let go
        assert not M.SEEN["brake_on"] and M.SEEN["releases"] == 1
        M._ratchet(None, s, at + 0.0080)  # on forward: free, no new brake
        assert not M.SEEN["brake_on"] and M.SEEN["brakes"] == 1
        M._ratchet(None, s, at + 0.0077)  # backs out 0.3 mm from there: the brake again
        assert M.SEEN["brake_on"] and M.SEEN["brakes"] == 2 and M.SEEN["latched"]
    finally:
        labkit._HELD.clear()
        labkit._SPECS.pop("tie", None)
        M.set_ratchet(os.environ.get("TIE_RATCHET", "latch"))
        M._new_trial()


# -- round 40: the left holds the head for the draw ----------------------------------------------------------------------
def _spans(steps):
    out, t = {}, 0.0
    for sec, *_rest, label in steps:
        if label != "wait":
            out.setdefault(label, (t, t + sec))
        t += sec
    return out


def test_the_held_draw_changes_hands_carries_the_head_to_the_cords_and_draws_along_the_slot():
    from yam_arena.demo import ArmPlan

    tr = L.trial("h37-hold")
    b = L.BUILDS[tr.build]
    plan = L.trial_plan(tr)
    ls, rs = ({s[4]: s for s in plan[arm]} for arm in ("left", "right"))
    lt, rt = _spans(plan["left"]), _spans(plan["right"])
    r1 = L.RY_ROLL @ L.hang_pick(b)[1]
    o = L.hang_pick(b)[2]
    # the hands change over one at a time: the tie is never let go of
    assert rt["take stub"][1] <= lt["let go tail 3"][0] + 1e-9  # the right has the stub before the left lets go
    assert lt["take head 2"][1] <= rt["let go stub 1"][0] + 1e-9  # the left has the head before the right lets go
    assert rt["back off stub 1"][1] <= lt["carry head"][0] + 1e-9  # the right is off before the head is carried
    assert lt["carry head"][1] <= rt["take stub 2"][0] + 1e-9  # the head is at the cords before the right takes the tail
    assert rt["let go stub"][0] <= lt["let go head 2"][0] + 1e-9 <= rt["back off stub"][1] + 1e-9
    # the left's hold: level, pointing the strap's way out of the head, its pads flush with the exit face
    rot = ls["take head 2"][3]
    assert abs(rot[:, 2][2]) < 1e-9 and abs(abs(rot[:, 0][2]) - 1.0) < 1e-9  # level, its fingers above and below
    head = np.asarray(ls["take head 2"][1]) + rot @ L.HOLD_O
    pad_mid = np.asarray(ls["take head 2"][1]) + rot[:, 2] * 0.011
    across = float((head - pad_mid) @ rot[:, 1])
    assert across == pytest.approx(-(0.0137 - L.HEAD[2] / 2 - L.HOLD_PAST), abs=1e-9) and 0.004 <= L.HOLD_PAST <= 0.008
    # (the head 6 mm in from the pads' edge on the exit side: on their flat, short of the right's pads on the tail)
    assert ls["take head 2"][2] == tr.head_fingers and ls["onto the head"][2] == "head_open"
    # by the cords: the strap's way out of the head at the wall, the way through toward the right (-y), the head as
    # near the cords as a tie hugging them has it, and nothing of the left's pads in the cords
    tight = L.hang_tight(tr, b, L.TOP + tr.h_low)
    deg = tr.h_tight[2]
    assert L.heading_xy(tight["rot"][:, 0]) == pytest.approx(deg) and L.heading_xy(tight["d"]) == pytest.approx(deg - 90.0)
    assert 5.0 <= deg <= 15.0  # turned toward the cords: the loose loop comes into the mouth on the slot's near side
    lc = ls["at the cords"][3]
    assert np.allclose(np.asarray(ls["at the cords"][1]) + lc @ L.HOLD_O, tight["head"], atol=1e-9)
    assert np.allclose(lc, L.rot_z(deg) @ r1) and deg + 11.0 <= tr.h_square_max  # room to square a head 11 degrees off
    cords = np.array([[c.xy[0], c.xy[1], tight["head"][2] + dz] for c in L.CORDS for dz in (-0.01, 0.0, 0.01)])
    radius = max(c.diameter for c in L.CORDS) / 2
    for hand in (deg, tr.h_square_max):  # nothing of the left's fingers in the cords, however far its wrist squares
        rot = L.rot_z(hand) @ r1
        assert min(L.box_distance(cords, bx) for bx in L.finger_boxes(tight["head"] - rot @ L.HOLD_O, rot, L.HEAD[1])) - radius >= 0.003
    root = tight["head"] + tight["rot"] @ L.ROOT
    for c in L.CORDS:  # the strap's way out of the head passes the cords (3 mm and more off them), the head clear too
        v = np.array([c.xy[0] - root[0], c.xy[1] - root[1]])
        off = abs(float(v[0] * tight["rot"][1, 0] - v[1] * tight["rot"][0, 0])) - c.diameter / 2 - L.T / 2
        assert off >= 0.003 and np.linalg.norm(np.array(c.xy) - tight["head"][:2]) - c.diameter / 2 >= L.HEAD[0] / 2 + 0.001
    assert [lab for lab in ("carry head", "set head", "square head", "at the cords") if lab in ls] == [
        "carry head", "set head", "square head", "at the cords"] and lt["square head"][1] <= rt["take stub 2"][0] + 1e-9
    # the right's second take: the tail below the exit, then the draw along the way through
    stub = np.asarray(rs["take stub 2"][1]) + rs["take stub 2"][3] @ o
    assert np.allclose(stub, tight["exit"] + tight["d"] * tr.h_stub_at, atol=1e-9)
    drawn = np.asarray(rs["pull"][1]) + rs["pull"][3] @ o
    assert np.allclose(drawn - stub, tight["d"] * tr.h_pull, atol=1e-9)
    assert L.heading_xy(rs["take stub 2"][3][:, 2]) == pytest.approx(tr.h_stub_yaw) and L.heading_xy(rs["pull"][3][:, 2]) == pytest.approx(tr.h_pull_yaw)
    # end on to the tail, the right's fingers end short of the left's on the head (bounding boxes, both ways the
    # left's wrist may have squared the head)
    for hand in (tr.h_tight[2], tr.h_tight[2] + 11.0):
        lrot = L.rot_z(hand) @ r1
        left_boxes = L.finger_boxes(tight["head"] - lrot @ L.HOLD_O, lrot, L.HEAD[1])
        for c, rr, h in L.finger_boxes(rs["take stub 2"][1], rs["take stub 2"][3], L.W):
            pts = np.array([c + rr @ (np.array(sg) * h) for sg in itertools.product((-1, -0.5, 0, 0.5, 1), repeat=3)])
            assert min(L.box_distance(pts, lb) for lb in left_boxes) >= 0.004
    assert tr.h_stub_at + tr.h_pull >= L.lift_marks(b.tie_l)[0] + 0.004  # far enough for the full mark
    assert tr.h_out + 0.006 + L.HEAD[2] <= tr.h_gft - 0.0137 - 0.005  # the left's pads stop short of the entry
    # within reach all the way
    arms = {arm: ArmPlan(ws, BASES[arm]) for arm, ws in M.plan_waypoints(plan).items()}
    for k in range(1, int(L.plan_seconds(plan) * M.CONTROL_HZ) + 1):
        for a in arms.values():
            a.action7(k / M.CONTROL_HZ)
    assert max(a.worst_ik_mm for a in arms.values()) < 1.0 and max(a.worst_ik_deg for a in arms.values()) < 1.0
    assert L.plan_seconds(plan) * M.CONTROL_HZ < 2400


@pytest.mark.parametrize("gain, stall, why", [(1.0, None, "mark"), (1.0, 0.120, "tight"), (0.0, None, "stuck"),
                                              (1.0, 0.250, "time")])
def test_the_held_draw_closed_loop_takes_the_head_where_it_is_and_stops_the_draw(gain, stall, why):
    tr = L.trial("h37-hold")
    b = L.BUILDS[tr.build]
    plan = M.demo_plan(tr)
    try:
        st = M.DEMO["steering"]
        when = {arm: {} for arm in plan}
        for arm, ws in plan.items():
            t = 0.0
            for w in ws:
                when[arm].setdefault(w.label, t)
                t += w.duration
        wp = lambda arm, label: next(w for w in plan[arm] if w.label == label)  # noqa: E731
        step = lambda t: int(round(t * M.CONTROL_HZ)) // L.STEER_EVERY * L.STEER_EVERY  # noqa: E731
        o = L.hang_pick(b)[2]
        # the head 12 mm from where the plan has it, its strap's way out 8 degrees round
        base = np.asarray(wp("left", "take head 2").pos) + np.asarray(wp("left", "take head 2").rot) @ L.HOLD_O
        root0 = L.heading_xy(np.asarray(wp("left", "take head 2").rot)[:, 2])
        head = base + np.array([0.010, -0.006, 0.003])
        hr = L.rot_z(root0 + 8.0 - 90.0) @ L.RY_ROLL @ np.asarray(b.head_rot, float)
        poly = np.array([head + hr @ L.ROOT + hr[:, 0] * k * L.LINK for k in range(b.n_links + 1)])
        tie = _Tie(head, L.matrix_to_quat_xyzw(hr), poly)
        st.update(step(when["left"]["to the head 2"] + 0.5), 0.034, np.zeros(3), state=tie)
        w = wp("left", "take head 2")
        assert np.allclose(np.asarray(w.pos) + np.asarray(w.rot) @ L.HOLD_O, head, atol=1e-6)
        assert L.heading_xy(np.asarray(w.rot)[:, 2]) == pytest.approx(root0 + 8.0, abs=1e-6)
        pre = wp("left", "to the head 2")
        assert np.allclose(np.asarray(w.pos) - np.asarray(pre.pos), 0.045 * np.asarray(w.rot)[:, 2], atol=1e-6)
        # the head 3 mm further along the fingers than planned: the carry puts the head, not the hand, at its place
        slid = np.asarray(w.rot) @ np.array([0.0, 0.0, 0.003])
        M.SEEN["tcp"] = {"left": np.asarray(w.pos) - slid, "right": np.zeros(3)}
        st.update(step(when["left"]["hold head 3"] + 0.2), 0.034, np.zeros(3), state=tie)
        tight = L.hang_tight(tr, b, L.TOP + tr.h_low)
        c = wp("left", "at the cords")
        assert np.allclose(np.asarray(c.pos) + np.asarray(c.rot) @ (L.HOLD_O + [0.0, 0.0, 0.003]), tight["head"], atol=1e-6)
        # at the cords the head has turned 11 degrees in the fingers: the wrist turns by that, about the head's place
        off = _Tie(tight["head"], L.matrix_to_quat_xyzw(L.rot_z(-11.0) @ tight["rot"]), poly)
        st.update(step(when["left"]["set head"] + 0.2), 0.034, np.zeros(3), state=off)
        c = wp("left", "at the cords")
        assert L.heading_xy(np.asarray(c.rot)[:, 2]) == pytest.approx(tr.h_tight[2] + 11.0, abs=1e-6)
        assert np.allclose(np.asarray(c.pos) + np.asarray(c.rot) @ (L.HOLD_O + [0.0, 0.0, 0.003]), tight["head"], atol=1e-6)
        assert np.allclose(wp("left", "set head").rot, L.rot_z(tr.h_tight[2]) @ L.RY_ROLL @ L.hang_pick(b)[1])
        assert st.reads["square"] == {"root_deg": pytest.approx(tr.h_tight[2] - 11.0), "hand_deg": pytest.approx(tr.h_tight[2]),
                                      "hand_to_deg": pytest.approx(tr.h_tight[2] + 11.0)}
        # at the cords, the head 2 mm and 5 degrees off: the second take and the draw follow the real exit and slot
        hr2 = L.rot_z(5.0) @ tight["rot"]
        head2 = tight["head"] + np.array([-0.002, 0.001, 0.0])
        tie2 = _Tie(head2, L.matrix_to_quat_xyzw(hr2), poly)
        st.update(step(when["right"]["to the stub 2"] + 0.5), 0.028, np.zeros(3), state=tie2)
        d = hr2 @ np.array([0.0, 0.0, -1.0])
        take = wp("right", "take stub 2")
        assert np.allclose(np.asarray(take.pos) + np.asarray(take.rot) @ o,
                           head2 + hr2 @ np.array([L.SLOT_C, 0.0, L.EXIT_Z]) + d * tr.h_stub_at, atol=1e-6)
        grip0 = np.asarray(take.pos, float) + np.asarray(take.rot) @ o  # the strap's point in the right's fingers
        end = np.asarray(wp("right", "pull").pos, float) + np.asarray(wp("right", "pull").rot) @ o
        assert np.allclose(end - grip0, d * tr.h_pull, atol=1e-6)
        assert L.heading_xy(np.asarray(take.rot)[:, 2]) == pytest.approx(tr.h_stub_yaw)  # end on to the tail
        # the draw: the tail follows the hand (or stalls, or never moves)
        t0, dur = when["right"]["pull"], wp("right", "pull").duration
        full = L.lift_marks(b.tie_l)[0]
        start = grip0.copy()
        M.SEEN["tcp"] = None  # no reading of the hand's real place: it eases back from where the draw stopped
        stopped, hand, sides = None, 0.0, []
        for k in range(step(t0), step(t0 + dur + 0.5) + 1):
            through = 0.028 + gain * hand if stall is None else min(0.028 + gain * hand, stall)
            st.update(k, through, np.zeros(3), state=tie2)
            assert np.allclose(wp("right", "pull").pos, wp("right", "take stub 2").pos)  # both ends of the blend: the
            assert np.allclose(wp("right", "pull").rot, wp("right", "take stub 2").rot)  # hand is there
            at = np.asarray(wp("right", "pull").pos, float) + np.asarray(wp("right", "pull").rot) @ o  # the grip's place
            yaw = L.heading_xy(np.asarray(wp("right", "pull").rot)[:, 2])
            assert yaw == pytest.approx(tr.h_stub_yaw + (tr.h_pull_yaw - tr.h_stub_yaw) * min(max(hand, 0.0) / L.HOLD_TURN, 1.0),
                                        abs=1.5) or st.why  # it turns from end on to its drawing yaw over HOLD_TURN
            hand = float((at - start) @ d)
            sides.append(float((at - start) @ np.array([-d[1], d[0], 0.0])))
            if stopped is None and st.why:
                stopped = (hand, through)
        assert st.why == why and stopped is not None
        hold_tcp = np.asarray(wp("right", "pull hold").pos, float)
        hold = hold_tcp + np.asarray(wp("right", "pull hold").rot) @ o
        gone = float(np.linalg.norm(hold - start))
        assert stopped[0] - L.HOLD_EASE - 0.004 <= gone <= stopped[0] + 0.004 and abs(sides[-1]) < 1e-9
        if why != "time":  # stopped before its time was up, it has eased back along the draw since (round 46)
            assert gone < stopped[0] - 0.002
        if why == "mark":
            assert full + st.HELD_PAST <= stopped[1] <= full + st.HELD_PAST + 0.004 and max(abs(v) for v in sides) < 1e-9
        elif why == "tight":
            assert stopped[1] == pytest.approx(stall) and (stall - 0.028) / gain - L.HOLD_EASE <= gone <= (stall - 0.028) / gain + st.TIGHT + 0.004
            reach = math.tan(math.radians(st.WORK_DEG)) * (tr.h_stub_at + stall - 0.028)
            assert max(sides) == pytest.approx(min(reach, 0.025), abs=0.002) and min(sides) == pytest.approx(-min(reach, 0.025), abs=0.002)
        elif why == "time":  # it stalled too late to be worked for long: the draw ends on its line when its time is up
            assert stopped[1] == pytest.approx(stall) and max(sides) > 0.02
        else:
            assert gone <= st.TIGHT + 0.004 and max(sides) == pytest.approx(0.006, abs=0.001)  # worked, never drawn
        assert np.allclose(np.asarray(wp("right", "back off stub").pos) - hold_tcp, (0.0, 0.0, 0.04), atol=1e-9)
    finally:
        M.stop_demo()
        M._new_trial()


def test_a_tie_whose_free_head_has_cocked_on_a_tight_loop_still_reads_as_threaded():
    """Round 48 (job 133): the strap comes into the mouth on a slant, the chord between two joints crossing the
    entry's plane outside the passage's window; with the tail out of the exit that crossing is still its way in."""
    hr = np.eye(3)
    hc = np.array([0.37, -0.01, L.TOP + 0.13])
    root = hc + hr @ L.ROOT
    # from the root round a small loop on the entry side, in over the head's root-side corner, out of the exit
    ring = [root + np.array([0.006, 0.0, 0.004]), hc + np.array([0.014, 0.0, 0.012]), hc + np.array([0.020, 0.0, 0.020]),
            hc + np.array([0.012, 0.0, 0.026]), hc + np.array([0.006, 0.0, 0.016]), hc + np.array([0.00527, 0.0, 0.00332]),
            hc + np.array([0.00072, 0.0, -0.00103]), hc + np.array([L.SLOT_C, 0.0, L.EXIT_Z - 0.001])]
    poly = np.vstack([root, *ring, ring[-1] + np.array([0.0, 0.0, -0.29])])
    th = L.threading(poly, hc, L.matrix_to_quat_xyzw(hr))
    assert th.entered and th.through == pytest.approx(0.291, abs=5e-4) and th.loop is not None
    strict = [c for c in L._crossings(L.to_head(poly, hc, L.matrix_to_quat_xyzw(hr)), L.ENTRY_Z) if L._in_passage(c[2])]
    assert not strict  # the passage's own window misses it
    # with no tail out of the exit the wide reading is not used: a strap lying across the head's face is not threaded
    short = poly[:-3]
    assert not L.threading(short, hc, L.matrix_to_quat_xyzw(hr)).entered
