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


def test_a_snug_tie_scores_full_through_score():
    from labkit import CableTieState

    near = L.tie_near_build(L.BUILDS["jig"])
    poly = L.synthetic_tie(near)
    s = CableTieState(pos=np.asarray(near.head_pos), quat_xyzw=L.matrix_to_quat_xyzw(near.head_rot),
                      lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=poly.min(0), high=poly.max(0), polyline=poly,
                      pawl_angle=1.0, link_speed_max=0.0)
    M.SEEN["hand_clear"] = 0.05
    assert M.score({"tie": s}) == 1.0
    s2 = CableTieState(**{**s.__dict__, "link_speed_max": 0.02})
    assert M.score({"tie": s2}) == 0.0
    M.SEEN["hand_clear"] = 0.01
    assert M.score({"tie": s}) == 0.0
    M._new_trial()


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
        assert M.BUILD == "jig" and M.TIE.fixed and M.PHYS == "heavier"  # round 7: what scored 1.0
    importlib.reload  # noqa: B018 - keep the import used


def test_task_json_matches_the_code():
    import json

    card = json.loads((HERE / "task.json").read_text(encoding="utf-8"))
    for key in ("title", "category", "instruction", "setup", "scoring", "run_limit", "fidelity", "scores"):
        assert card.get(key), key
    assert card["instruction"] == M.INSTRUCTION
    assert card["category"] == "Societal relatability"

    def tail_mm(slack):  # the submitted start: the head held in the clip, so snug is judged round the head there
        return f"{1000 * (L.snug_through(L.BUILDS[M.BUILD]) - slack):.1f}"

    assert f"at least {tail_mm(L.SNUG_SLACK)} mm" in card["scoring"]
    assert f"at least {tail_mm(L.LOOSE_SLACK)} mm" in card["scoring"]
    assert f"{round(1000 * L.CLEAR / 10)} cm" in card["scoring"] and "1 cm/s" in card["scoring"]
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
    wall while the tail goes in, lets it onto the strap at LATCH_AT out, and brakes from then on."""
    import labkit

    labkit._SPECS["tie"] = M.TIE
    try:
        M._new_trial()
        M.set_ratchet("latch")
        s = labkit.start_state(M.TIE)
        M._ratchet(None, s, 0.0)
        assert labkit.held("tie", "pawl") == 0.0 and M.SEEN["opened"]
        M._ratchet(None, s, L.LATCH_AT - 1 * L.MM)
        M._ratchet(None, s, L.LATCH_AT - 3 * L.MM)  # backing out before the latch: no brake, the pawl stays clear
        assert labkit.held("tie", "pawl") == 0.0 and not M.SEEN["brake_on"]
        M._ratchet(None, s, L.LATCH_AT + 1 * L.MM)
        assert labkit.held("tie", "pawl") is None and M.SEEN["latched"] and not M.SEEN["opened"]
        M._ratchet(None, s, L.LATCH_AT + 0.4 * L.MM)  # now it backs out: the brake
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
    cand = L.trial(M.TRIAL)
    assert M.TRIAL == "m-demo" and cand is L.CANDIDATE and cand in L.KEPT
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
THREADS = [t.name for t in L.CHECKED if t.stage == "thread"]


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


SLIDE_FIRST = [t.name for t in L.CHECKED if t.stage == "thread" and t.b_slide_first]


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
