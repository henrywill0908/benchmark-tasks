"""Pure checks of tie_logic (NumPy only): the cords, the tie's spawn shapes, the pawl's wedge, the threading
measurements and the draft scorer on synthetic ties, the probe plans' structure and finger clearances.

    python -m pytest -q test_logic.py test_task.py
"""

import math

import numpy as np
import pytest

import tie_logic as L

JIG = L.BUILDS["jig"]
FREE = L.BUILDS["free"]
NEAR = L.tie_near_build(JIG)
Q = L.matrix_to_quat_xyzw(JIG.head_rot)


def _score(poly, build=NEAR, clear=0.05, speed=0.0):
    return L.task_score(poly, build.head_pos, L.matrix_to_quat_xyzw(build.head_rot), clear, speed)


# -- the room ------------------------------------------------------------------------------------------------------------
def test_the_cords_are_gathered_with_small_gaps_and_dont_overlap():
    for i, a in enumerate(L.CORDS):
        for b in L.CORDS[i + 1:]:
            gap = np.linalg.norm(np.subtract(a.xy, b.xy)) - (a.diameter + b.diameter) / 2
            assert 0.2 * L.MM < gap < 0.4 * L.MM, (a.name, b.name, gap)
    assert np.allclose(np.mean([c.xy for c in L.CORDS], axis=0), L.BUNDLE, atol=1e-9)
    assert 0.040 < L.bundle_perimeter() < 0.046


def test_the_cords_hang_in_front_of_the_tv_and_wall_over_the_power_strip():
    for c in L.CORDS:
        assert c.xy[0] + c.diameter / 2 < L.TV["x"][0] < L.WALL["x"][0]
        assert L.STRIP["x"][0] < c.xy[0] < L.STRIP["x"][1]
    assert L.CORD_Z == (L.STRIP["z"][1], L.TV["z"][0])


# -- the tie -------------------------------------------------------------------------------------------------------------
def test_the_tie_is_200_mm_of_equal_links_within_physx_limits():
    for b in (JIG, FREE):
        ch = np.linalg.norm(np.diff(b.points, axis=0), axis=1)
        assert len(b.points) == L.N_LINKS + 1 and np.ptp(ch) < 1e-9 and abs(ch.sum() - L.TIE_L) < 1e-9
        assert np.allclose(b.points[0], L.ROOT) and np.max(np.abs(b.points[:, 1])) < 1e-12
    assert 2 + L.N_LINKS <= 64
    assert abs(L.STRAP_K - L.EI / L.LINK) < 1e-12 and 0.05e-3 < L.LINK_MASS < 0.1e-3


def test_the_jig_loop_ends_4_mm_short_of_the_slot_on_its_line_pointing_in():
    pts = JIG.points
    assert np.allclose(pts[-1], [L.SLOT_C, 0.0, L.ENTRY_Z + L.TIP_GAP], atol=1e-7)
    u = pts[-1] - pts[-2]
    assert np.allclose(u / np.linalg.norm(u), [0.0, 0.0, -1.0], atol=1e-6)


def test_the_jig_loop_clears_the_cords_and_the_head():
    ring = L.cord_ring_h(JIG, 0.0)
    pts = np.vstack([JIG.points, (JIG.points[:-1] + JIG.points[1:]) / 2])
    for p in pts:
        assert np.min(np.linalg.norm(ring - p[[0, 2]], axis=1)) - L.T / 2 > 2.5 * L.MM
    for p in pts[1:]:  # only the root touches the head
        assert np.any(np.abs(p) > L.HEAD / 2 + 0.5 * L.MM)
    hb = np.maximum(np.abs(ring) - L.HEAD[[0, 2]] / 2, 0.0)
    assert np.min(np.linalg.norm(hb, axis=1)) > 3 * L.MM


def test_the_jig_loop_starts_round_all_three_cords():
    ph = JIG.points[:, [0, 2]]
    for c in L.CORDS:
        p = (np.array([*c.xy, JIG.head_pos[2]]) - np.asarray(JIG.head_pos)) @ JIG.head_rot
        assert L._point_in_poly(p[[0, 2]], ph)


def test_the_free_tie_lies_flat_on_the_console_clear_of_the_cords_and_strip():
    poly = L.start_polyline(FREE)
    assert np.allclose(poly[:, 2], FREE.head_pos[2] + L.ROOT[2])
    assert abs(FREE.head_pos[2] - L.HEAD[2] / 2 - L.TOP) < 0.5 * L.MM
    assert abs(poly[0, 2] - L.T / 2 - L.TOP) < 0.5 * L.MM
    assert np.max(poly[:, 0]) + L.W / 2 < L.STRIP["x"][0] - 0.05


def test_the_pawl_wedges_one_way_and_clears_the_passage_when_held_open():
    a = L.pawl_contact_deg()
    assert 45.0 < a < math.degrees(L.PAWL_REST)
    assert math.tan(math.radians(90.0 - a)) < L.TIE_FRICTION  # self-locking against backing out
    tip = L.pawl_tip(L.PAWL_REST)
    assert L.PASSAGE_X[0] < tip[0] < L.PASSAGE_X[1] and L.EXIT_Z < tip[2] < L.ENTRY_Z
    open_face = L.PAWL_HINGE[0] - L.PAWL_R  # held at 0: the plate flat against the passage's +x wall
    assert open_face - L.PASSAGE_X[0] > L.T + 2 * L.MM
    assert L.PAWL_HINGE[0] + L.PAWL_R <= L.PASSAGE_X[1] + 1e-12


def test_the_passage_takes_the_tip_within_about_1_6_mm():
    tol_x = (L.PASSAGE_X[1] - L.PASSAGE_X[0] - L.TIP_T) / 2
    tol_y = L.PASSAGE_Y - L.TIP_W / 2
    assert 1.3 * L.MM < tol_x < 1.8 * L.MM and 1.3 * L.MM < tol_y < 1.8 * L.MM


def test_rot_axes_is_a_right_handed_rotation():
    for a, c in (((1, 0, 0), (0, 0, 1)), ((0, -0.57, -0.82), (-1, 0, 0)), ((0, 1, 0), (0, 0, 1))):
        r = L.rot_axes(a, c)
        assert np.allclose(r.T @ r, np.eye(3)) and np.isclose(np.linalg.det(r), 1.0)
        assert np.allclose(r[:, 2], np.asarray(a) / np.linalg.norm(a))


def test_quaternion_round_trip():
    for r in (JIG.head_rot, FREE.head_rot, L.rot_down(30)):
        assert np.allclose(L.quat_xyzw_to_matrix(L.matrix_to_quat_xyzw(r)), r, atol=1e-12)


# -- measuring and scoring -----------------------------------------------------------------------------------------------
def test_the_open_start_is_not_closed_and_scores_zero():
    for b in (JIG, FREE):
        q = L.matrix_to_quat_xyzw(b.head_rot)
        th = L.threading(L.start_polyline(b), b.head_pos, q)
        assert th.through == 0.0 and not th.closed
        assert L.task_score(L.start_polyline(b), b.head_pos, q, 1.0, 0.0) == 0.0


def test_a_snug_tie_against_the_cords_scores_full():
    poly = L.synthetic_tie(NEAR)
    th = L.threading(poly, NEAR.head_pos, Q)
    assert th.closed and len(th.enclosed) == 3 and th.slack <= L.SNUG_SLACK
    assert abs(th.loop + L.HEAD[2] + th.through - L.TIE_L) < 0.1 * L.MM  # the slot's run is 5.6 mm, give or take a chord
    assert _score(poly) == 1.0


def test_the_tail_length_a_person_measures_matches_the_scorer():
    poly = L.synthetic_tie(NEAR)
    th = L.threading(poly, NEAR.head_pos, Q)
    assert (th.through >= L.TIE_L - L.HEAD[2] - (L.bundle_perimeter() - L.HEAD_CLOSURE + L.SNUG_SLACK)) == (
        th.slack <= L.SNUG_SLACK)


@pytest.mark.parametrize("grow, want", [(0.002, 1.0), (0.004, 0.5), (0.006, 0.25)])
def test_slack_earns_partial_credit(grow, want):
    assert _score(L.synthetic_tie(NEAR, grow)) == want


def test_no_credit_while_a_gripper_is_near_or_the_tie_is_moving():
    poly = L.synthetic_tie(NEAR)
    assert _score(poly, clear=0.029) == 0.0
    assert _score(poly, speed=0.011) == 0.0
    assert _score(poly, clear=0.030, speed=0.009) == 1.0


def test_a_loop_round_two_cords_or_fed_in_backwards_scores_zero():
    assert _score(L.synthetic_tie(NEAR, skip=("cord_phone",))) == 0.0
    assert _score(L.synthetic_tie(NEAR, backwards=True)) == 0.0


def test_a_tie_that_slid_down_the_cords_gets_a_quarter():
    poly = L.synthetic_tie(NEAR)
    drop = np.array([0.0, 0.0, -(NEAR.head_pos[2] - L.TOP - 0.05)])
    low = L.Build("low", True, tuple(np.asarray(NEAR.head_pos) + drop), NEAR.head_rot, NEAR.points)
    assert L.task_score(poly + drop, low.head_pos, Q, 0.05, 0.0) == 0.25


def test_a_tip_short_of_the_exit_is_not_closed():
    poly = L.synthetic_tie(NEAR)
    th = L.threading(poly, NEAR.head_pos, Q)
    ph = L.to_head(poly, NEAR.head_pos, Q)
    # pull the strap back along itself until only 1 mm of tail is out: rebuild with a longer loop
    for grow in np.linspace(0.0, 0.03, 61):
        th = L.threading(L.synthetic_tie(NEAR, grow), NEAR.head_pos, Q)
        if th.through < L.TAIL_MIN:
            assert not th.closed and _score(L.synthetic_tie(NEAR, grow)) == 0.0
            break
    else:
        pytest.skip("never short enough")
    assert ph.shape == (L.N_LINKS + 1, 3)


def test_snug_is_judged_against_the_tightest_loop_the_cords_allow_round_the_head():
    """Round 2: the jig's rigid cords stand off the head, so a loop hugging the bundle can't close to its perimeter;
    snug is the loop within SNUG_SLACK of the tightest one round the head where it is (full credit is reachable)."""
    assert L.min_slack(JIG) > L.SNUG_SLACK
    assert _score(L.synthetic_tie(JIG), build=JIG) == 1.0
    q = L.matrix_to_quat_xyzw(JIG.head_rot)
    assert abs(L.min_loop_at(JIG.head_pos, q) - L.min_loop(JIG)) < 0.5 * L.MM


def test_hand_clearance_measures_to_the_strap_and_the_head():
    poly = L.start_polyline(JIG)
    far = np.array([[0.0, 0.0, 1.5]])
    assert L.hand_clearance(far, poly, JIG.head_pos, Q) > 0.3
    assert L.hand_clearance(poly[10:11] + [0, 0, 0.01], poly, JIG.head_pos, Q) == pytest.approx(0.01 - L.W / 2, abs=1e-4)
    assert L.hand_clearance(np.array([JIG.head_pos]), poly, JIG.head_pos, Q) == 0.0


# -- the probe's plans ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", L.TRIAL_NAMES)
def test_every_plan_has_both_arms_ending_together_then_a_watch(name):
    plan = L.trial_plan(L.trial(name))
    assert set(plan) == {"left", "right"}
    totals = {arm: sum(s[0] for s in steps) for arm, steps in plan.items()}
    assert abs(totals["left"] - totals["right"]) < 1e-9
    for steps in plan.values():
        assert steps[-1][4] == "watch" and all(s[0] > 0 for s in steps)
        assert all(s[2] in L.FINGER_GAP for s in steps)


def _times(steps):
    out, t = [], 0.0
    for s in steps:
        out.append((t, t + s[0], s))
        t += s[0]
    return out


@pytest.mark.parametrize("name", [t.name for t in L.CHECKED if t.insert and t.pull and not t.handover])
def test_the_right_hand_reaches_for_the_tail_only_after_the_left_has_backed_off(name):
    plan = L.trial_plan(L.trial(name))
    left_done = next(t1 for t0, t1, s in _times(plan["left"]) if s[4] == "back off")
    first = next(t0 for t0, t1, s in _times(plan["right"]) if s[4] == "to the tail")
    assert first >= left_done


@pytest.mark.parametrize("name", [t.name for t in L.CHECKED if t.pull and not t.handover])
def test_the_pull_strokes_draw_the_strap_past_snug(name):
    tr = L.trial(name)
    plan = L.trial_plan(tr)
    d = L.jig_frame()["through"]
    pulls = [s for s in plan["right"] if s[4].startswith("pull ")]
    starts = [s for s in plan["right"] if s[4].startswith("pinch ")]
    drawn = sum(float(np.dot(np.subtract(p[1], a[1]), d)) for p, a in zip(pulls, starts))
    out = L.PROTRUDE3 if (tr.insert and tr.strokes >= 3) else L.PROTRUDE
    assert drawn == pytest.approx(L.snug_through(JIG) - out + tr.extra_pull, abs=1e-6)
    assert all(np.dot(np.subtract(p[1], a[1]), d) <= 0.0601 for p, a in zip(pulls, starts))


@pytest.mark.parametrize("name", [t.name for t in L.CHECKED if t.insert and not t.naive and not t.handover])
def test_the_insert_ends_with_the_tail_out_by_protrude(name):
    tr = L.trial(name)
    f = L.jig_frame()
    steps = {s[4]: s for s in L.trial_plan(tr)["left"]}
    d = L._yawed(f["through"], tr.tilt_deg)  # pushed along the slot's axis, or the trial's tilted approach
    moved = np.dot(np.subtract(steps["insert"][1], steps["align"][1]), d) + np.dot(
        np.subtract(steps["insert 2"][1], steps["regrip"][1]), d)
    assert moved == pytest.approx(L.ALIGN_GAP + L.HEAD[2] + L.PROTRUDE, abs=1e-9)
    if tr.strokes >= 3:  # round 4's third push brings PROTRUDE3 out
        third = np.dot(np.subtract(steps["insert 3"][1], steps["regrip 2"][1]), d)
        assert third == pytest.approx(L.PROTRUDE3 - L.PROTRUDE, abs=1e-9)


def _gap(arm, kind, fingers):
    if fingers != "closed":
        return L.FINGER_GAP[fingers] * L.MM
    return L.T if kind == "flat" else L.W


@pytest.mark.parametrize("name", [t.name for t in L.CHECKED if L.BUILDS[t.build].fixed and (t.insert or t.pull)])
def test_the_fingers_clear_the_head_and_the_cords_near_the_tie(name):
    """A fat envelope of the fingers (boxes round yam.urdf's finger meshes) at every pose near the tie: clear of the
    head's box by 2 mm and of the rigid cords by 0.5 mm (the closest: the left hand letting go after the first
    stroke)."""
    tr = L.trial(name)
    obs = L.obstacle_points(JIG)
    for arm, steps in L.trial_plan(tr).items():
        kind = tr.insert if arm == "left" else tr.pull
        for s in steps:
            if isinstance(s[3], str) or not L.near_tie(s[4]):
                continue
            boxes = L.finger_boxes(s[1], s[3], _gap(arm, kind, s[2]))
            assert min(L.box_distance(obs["head"], b) for b in boxes) > 2.0 * L.MM, (arm, s[4])
            assert min(L.box_distance(obs["cords"], b) for b in boxes) > 0.5 * L.MM, (arm, s[4])
            assert min(L.box_distance(obs["stand"], b) for b in boxes) > 2.0 * L.MM, (arm, s[4])


@pytest.mark.parametrize("name", [t.name for t in L.CHECKED if t.insert and t.depth])
def test_the_fingers_still_clear_the_head_and_cords_at_full_closed_loop_depth(name):
    """Round 2's probe pushes the second stroke up to depth_cap further along the slot's way: the fingers must still
    clear the head by 2 mm and the cords by 0.5 mm (fat envelope) there."""
    tr = L.trial(name)
    obs = L.obstacle_points(L.BUILDS[tr.build])
    d = L.jig_frame(L.BUILDS[tr.build])["through"]
    labels = L.HANDOVER_DEPTH_LABELS if tr.handover else L.DEPTH_LABELS
    for s in L.trial_plan(tr)["left"]:
        if s[4] in labels + ("let go",):
            boxes = L.finger_boxes(np.asarray(s[1]) + d * tr.depth_cap, s[3], _gap("left", tr.insert, s[2]))
            assert min(L.box_distance(obs["head"], b) for b in boxes) > 2.0 * L.MM, (name, s[4])
            assert min(L.box_distance(obs["cords"], b) for b in boxes) > 0.5 * L.MM, (name, s[4])
            if obs["stand"].size:
                assert min(L.box_distance(obs["stand"], b) for b in boxes) > 2.0 * L.MM, (name, s[4])


def test_the_pick_grips_the_head_with_the_pad_tips_off_the_console():
    plan = L.trial_plan(L.Trial("pick", "free", "", insert=None, pull=None, pick=True, head_fingers="head_light"))
    grip = next(s for s in plan["right"] if s[4] == "grip")
    tips = np.asarray(grip[1]) + grip[3][:, 2] * L.PAD_PAST_TCP
    assert L.TOP + 0.5 * L.MM < tips[2] < FREE.head_pos[2] - 1.0 * L.MM
    assert np.allclose(np.abs(grip[3][:, 0]), np.abs(FREE.head_rot[:, 1]), atol=1e-9)  # closing across the head


def test_trial_names_are_unique_and_known_builds():
    assert len(set(L.TRIAL_NAMES)) == len(L.TRIAL_NAMES)
    assert all(t.build in L.BUILDS and t.ratchet in L.RATCHETS for t in L.TRIALS)
    with pytest.raises(KeyError):
        L.trial("nope")


# -- round 1's wider sweep ----------------------------------------------------------------------------------------------
def test_the_tight_build_is_the_jig_with_a_slot_close_to_a_real_ties():
    t = L.BUILDS["tight"]
    assert t.fixed and t.head_pos == JIG.head_pos and np.allclose(t.points, JIG.points)
    assert L.W / 2 < t.passage_y < L.PASSAGE_Y and 1.2 * L.MM < t.passage_y - L.TIP_W / 2 < 1.4 * L.MM
    near = L.tie_near_build(t)
    poly = L.synthetic_tie(near)  # down the slot's centre: through the narrow slot too
    assert L.task_score(poly, near.head_pos, Q, 0.05, 0.0, t.passage_y) == 1.0
    off = poly + JIG.head_rot @ np.array([0.0, 4.0 * L.MM, 0.0])  # 4 mm across: past the narrow slot (+1 mm allowance)
    q = L.matrix_to_quat_xyzw(JIG.head_rot)
    assert not L.threading(off, near.head_pos, q, t.passage_y).closed
    assert L.threading(off, near.head_pos, q).closed  # the wide jig slot still takes it


def _insert_trial(**kw):  # round 1's generic insert (lines the strap up, then pushes): its knobs still plan right
    return L.Trial("x", "jig", "", **{"pull": None, "watch_s": 2.0, "insert": "flat", **kw})


def test_the_robustness_repeats_change_only_where_the_strap_is_taken():
    base = {s[4]: s for s in L.trial_plan(_insert_trial(aim_off=(0.0, 3.0 * L.MM)))["left"]}
    for dz in (2 * L.MM, -2 * L.MM):
        rep = {s[4]: s for s in L.trial_plan(_insert_trial(grip_shift=(dz, 0.0)))["left"]}
        nominal = {s[4]: s for s in L.trial_plan(_insert_trial())["left"]}
        assert np.allclose(np.subtract(rep["grip"][1], nominal["grip"][1]), [0, 0, dz])
        assert np.allclose(rep["align"][1], nominal["align"][1]) and np.allclose(rep["insert 2"][1], nominal["insert 2"][1])
    assert base["align"][1] != base["grip"][1]


def test_the_tilted_approach_is_10_degrees_off_the_slot_axis():
    st = {s[4]: s for s in L.trial_plan(_insert_trial(tilt_deg=10.0))["left"]}
    push = np.subtract(st["insert"][1], st["align"][1])
    ang = math.degrees(math.acos(np.dot(push / np.linalg.norm(push), L.jig_frame()["through"])))
    assert ang == pytest.approx(10.0, abs=1e-6)


def test_the_tug_pulls_the_strap_along_itself_against_the_held_head():
    b = FREE
    for fingers in ("head", "head_light"):
        plan = L.trial_plan(L.Trial("x", "free", "", insert=None, pull=None, pick=True, tug=True,
                                    head_fingers=fingers, watch_s=2.0))
        left = {s[4]: s for s in plan["left"]}
        right = {s[4]: s for s in plan["right"]}
        assert right["grip"][2] == fingers and left["grip"][2] == "closed"
        pull = np.subtract(left["tug"][1], left["grip"][1])
        assert np.allclose(pull, b.head_rot[:, 0] * L.TUG)
        t_hold = sum(s[0] for s in plan["right"][: [s[4] for s in plan["right"]].index("hold up") + 1])
        t_grip = sum(s[0] for s in plan["left"][: [s[4] for s in plan["left"]].index("grip")])
        assert t_grip >= t_hold - 1.0  # the left takes the strap once the head is up
        tips = np.asarray(left["grip"][1]) + left["grip"][3][:, 2] * L.PAD_PAST_TCP
        assert tips[2] > L.TOP + 0.03
        assert all(s[4] in ("watch", "wait") or s[2] == fingers or s[4] in ("to the head", "descend")
                   for s in plan["right"] if s[4] not in ("grip", "lift", "hold up"))


def test_the_light_squeezes_are_lighter():
    assert 0 < L.FINGER_GAP["light"] < L.T / L.MM and L.FINGER_GAP["head"] < L.FINGER_GAP["head_light"] < L.HEAD[1] / L.MM


# -- round 3: the straight run --------------------------------------------------------------------------------------
def test_the_jig_loop_ends_in_a_straight_run_down_the_slot():
    assert L.straight_run(JIG.points) >= 40.0 * L.MM


@pytest.mark.parametrize("name", [t.name for t in L.CHECKED if t.insert == "edge" and t.build != "wrap"])
def test_the_inserting_hand_never_turns_and_only_moves_along_the_slot(name):
    """Round 2: fingers closed on the strap's edges can't turn it (it pivots between the pads); a hand that turned
    while lining up swung the tip 10-16 mm off the slot. Round 3's hand takes the straight run and only pushes."""
    tr = L.trial(name)
    d = L.jig_frame()["through"]
    steps = [s for s in L.trial_plan(tr)["left"] if s[4] in ("in", "grip", "align", "insert", "insert hold", "open",
                                                              "slide back", "regrip", "insert 2", "insert 2 hold",
                                                              "open 2", "slide back 2", "regrip 2", "insert 3",
                                                              "insert 3 hold")]
    for s in steps[1:]:
        assert np.allclose(s[3], steps[0][3], atol=1e-12), (name, s[4])
        off = np.subtract(s[1], steps[0][1])
        assert np.linalg.norm(off - (off @ d) * d) < 1e-9, (name, s[4])


@pytest.mark.parametrize("name", [t.name for t in L.CHECKED if t.insert == "edge"])
def test_the_pads_take_the_straight_run_only(name):
    tr = L.trial(name)
    pad_half = 13.7 * L.MM  # the fingers' width along the strap near the TCP (finger_boxes)
    assert tr.grip_from_tip + pad_half <= L.straight_run(JIG.points) + 6.5 * L.MM  # at most into the turn's first link


# -- round 5: the hand-over ------------------------------------------------------------------------------------------
HANDOVERS = [t.name for t in L.CHECKED if t.handover]


def _timeline(steps):
    out, t = {}, 0.0
    for s in steps:
        out.setdefault(s[4], (t, t + s[0], s))
        t += s[0]
    return out


@pytest.mark.parametrize("name", HANDOVERS)
def test_the_left_lets_go_only_after_the_right_holds_the_tail(name):
    """Round 4: every time the inserting hand let go to regrip, the strap sprang back out of the slot. In the hand-over
    the strap is always held: the left lets go only once the right's fingers have closed on the tail."""
    plan = L.trial_plan(L.trial(name))
    left, right = _timeline(plan["left"]), _timeline(plan["right"])
    assert right["pinch 1"][1] <= left["let go"][0]
    assert right["to the tail"][0] >= left["insert"][1]  # the right comes in once the push is done
    if L.trial(name).guide:  # round 10: the left stays as a loose ring round the strap until the first pull is done
        assert left["unguide"][0] >= right["pull 1"][1] - 1e-9
        assert [s[2] for s in plan["left"] if s[4] in ("guide", "guide hold")] == ["loose", "loose"]
    else:
        assert right["pull 1"][0] >= left["back off"][1]  # and pulls only once the left has left
    assert [s[2] for s in plan["left"] if s[4] in ("grip", "align", "insert", "insert hold")] == ["closed"] * 4


@pytest.mark.parametrize("name", HANDOVERS)
def test_the_one_push_brings_handover_out_and_the_one_pull_draws_past_snug(name):
    tr = L.trial(name)
    f = L.jig_frame(L.BUILDS[tr.build], tr.grip_from_tip)
    d = f["through"]
    left, right = _timeline(L.trial_plan(tr)["left"]), _timeline(L.trial_plan(tr)["right"])
    push = np.dot(np.subtract(left["insert"][2][1], left["align"][2][1]), d)
    assert push == pytest.approx(L.TIP_GAP + L.HEAD[2] + L.HANDOVER_OUT, abs=1e-9)
    took = "bite" if tr.rebite else "pinch 1"  # round 9: draw rebite, open, slide back to where it pinched, bite
    if tr.rebite:
        assert np.allclose(right["bite"][2][1], right["pinch 1"][2][1])
        assert np.linalg.norm(np.subtract(right["draw"][2][1], right["pinch 1"][2][1])) == pytest.approx(tr.rebite)
    aim = np.array([tr.pull_aim_x, 0.0, 0.0])  # round 11: the pull's line may be shifted sideways
    drawn = tr.rebite + np.linalg.norm(np.subtract(right["pull 1"][2][1], right[took][2][1]) - aim) + np.linalg.norm(
        np.subtract(right["pull 2"][2][1], right["pull 1"][2][1]))
    assert drawn == pytest.approx(L.snug_through(L.BUILDS[tr.build]) - L.HANDOVER_OUT + tr.extra_pull, abs=1e-6)
    first = np.subtract(right["pull 1"][2][1], right[took][2][1]) - aim
    assert np.allclose(first / np.linalg.norm(first), d)  # straight out of the slot first


@pytest.mark.parametrize("name", [n for n in HANDOVERS if L.trial(n).build != "wrap"])
def test_the_push_starts_from_the_straight_run_and_never_turns(name):
    tr = L.trial(name)
    f = L.jig_frame(L.BUILDS[tr.build], tr.grip_from_tip)
    assert f["grip_dir"] @ f["through"] > np.cos(np.radians(0.5))
    assert tr.grip_from_tip + 13.7 * L.MM <= L.straight_run(L.BUILDS[tr.build].points) + 6.5 * L.MM
    steps = [s for s in L.trial_plan(tr)["left"] if s[4] in ("in", "grip", "align", "insert", "insert hold")]
    for s in steps:
        assert np.allclose(s[3], steps[0][3])


@pytest.mark.parametrize("name", HANDOVERS)
def test_the_two_hands_clear_each_other_at_the_hand_over(name):
    """At the hand-over both hands hold the strap a few mm either side of the head: their fingers (fat envelope) must
    stay 2 mm apart, with the push at its full closed-loop depth."""
    tr = L.trial(name)
    d = L.jig_frame(L.BUILDS[tr.build])["through"]
    left = {s[4]: s for s in L.trial_plan(tr)["left"]}
    right = {s[4]: s for s in L.trial_plan(tr)["right"]}
    lb = L.finger_boxes(np.asarray(left["insert hold"][1]) + d * tr.depth_cap, left["insert hold"][3], L.W)
    rb = L.finger_boxes(right["pinch 1"][1], right["pinch 1"][3], L.W)
    g = np.linspace(-1, 1, 5)
    for box in lb:  # sample the left boxes' surfaces against the right boxes
        c, r, h = box
        pts = np.array([c + r @ (np.array([a, b_, s_]) * h) for a in g for b_ in g for s_ in (-1.0, 1.0)]
                       + [c + r @ (np.array([a, s_, b_]) * h) for a in g for b_ in g for s_ in (-1.0, 1.0)]
                       + [c + r @ (np.array([s_, a, b_]) * h) for a in g for b_ in g for s_ in (-1.0, 1.0)])
        for other in rb:
            assert L.box_distance(pts, other) > 2.0 * L.MM, name


# -- round 7: the wrap start --------------------------------------------------------------------------------------------
WRAPS = [t.name for t in L.CHECKED if t.build == "wrap"]
WRAP = L.BUILDS["wrap"]


def test_the_wrap_start_is_the_jig_loop_round_the_back_then_a_straight_tail_clear_of_everything():
    w, j = WRAP.points, JIG.points
    assert len(w) == len(j) and np.allclose(w[: L.WRAP_CUT + 1], j[: L.WRAP_CUT + 1])
    seg = np.diff(w, axis=0)
    assert np.allclose(np.linalg.norm(seg, axis=1), L.LINK)
    u = seg[L.WRAP_CUT - 1:] / L.LINK
    assert np.allclose(u, u[0], atol=1e-9)  # straight from the cut to the tip (nylon's rest shape)
    ring = L.cord_ring_h(WRAP, 0.0)
    pts = np.vstack([w, (w[:-1] + w[1:]) / 2])
    for p in pts:
        assert np.min(np.linalg.norm(ring - p[[0, 2]], axis=1)) - L.T / 2 > 2.5 * L.MM
    world = L.start_polyline(WRAP)
    assert world[:, 0].max() < L.WALL["x"][0] - 0.03 and world[-1, 0] < L.BUNDLE[0] - 0.05  # the tail toward the robot


def _winding_deg(points, build):
    c = np.mean([(np.array([*cd.xy, build.head_pos[2]]) - np.asarray(build.head_pos)) @ build.head_rot
                 for cd in L.CORDS], axis=0)[[0, 2]]
    a = np.unwrap([math.atan2(*(p[[2, 0]] - c)) for p in points])
    return abs(np.degrees(a[-1] - a[0]))


def test_the_wrap_start_goes_only_partway_round_the_cords():
    """The wrap tie isn't round the cords yet: the robot has to bring the tail round the front (about 90 degrees more
    than it starts with); the jig's open loop already goes nearly all the way."""
    assert _winding_deg(JIG.points, JIG) > 300.0
    assert _winding_deg(WRAP.points, WRAP) < _winding_deg(JIG.points, JIG) - 60.0


@pytest.mark.parametrize("name", WRAPS)
def test_the_swing_ends_where_the_jig_starts_then_the_jig_hand_over_follows(name):
    tr = L.trial(name)
    plan = L.trial_plan(tr)
    w = L.wrap_swing(tr, WRAP)
    settle = next(s for s in plan["left"] if s[4] == "wrap settle")
    assert np.allclose(settle[1], w["end"], atol=1e-9) and np.allclose(settle[3], w["rot1"], atol=1e-9)
    jig = L._handover_plan(tr, JIG)
    tail = [s for s in plan["left"] if not s[4].startswith("wrap") and s[4] not in ("to the strap", "in", "grip")]
    ref = [s for s in jig["left"] if s[4] not in ("to the strap", "in", "grip")]
    tail = tail[: len(ref)]  # trial_plan pads the shorter arm with a wait at the end
    assert [s[4] for s in tail] == [s[4] for s in ref]
    for a, b in zip(tail, ref):
        assert a[0] == pytest.approx(b[0]) and (a[1] is None or np.allclose(a[1], b[1]))
    assert 60.0 < abs(np.degrees(w["phi"])) < 100.0
    # the hand never lets go during the swing
    assert all(s[2] == "closed" for s in plan["left"] if s[4].startswith("wrap"))


@pytest.mark.parametrize("name", WRAPS)
def test_the_swing_moves_the_grip_round_the_pivot_in_small_steps(name):
    tr = L.trial(name)
    w = L.wrap_swing(tr, WRAP)
    steps = [s for s in L.trial_plan(tr)["left"] if s[4].startswith("wrap ") and s[4][5:].isdigit()]
    r0 = np.linalg.norm(w["taken"][:2] - w["pivot"])
    prev = w["taken"]
    for s in steps:
        p = np.asarray(s[1])
        assert np.linalg.norm(p[:2] - w["pivot"]) == pytest.approx(r0, abs=1e-9)
        assert np.linalg.norm(p - prev) < 3.5 * L.MM  # about 7 degrees a step on a 19 mm radius
        prev = p


def test_tail_yaw_error_turns_the_tail_onto_the_slot():
    """Round 8: tail_yaw_error is the turn that lines the tail's end up with the slot's way, and steer_pose turns the
    hand about the strap in its fingers (that point stays put)."""
    d = np.array([0.0, -1.0, 0.0])
    for deg in (-12.0, 0.0, 7.0):
        a = math.radians(deg)
        u = np.array([math.sin(-a), -math.cos(-a), 0.0])  # d turned by -a
        poly = np.array([k * 0.00625 * u for k in range(10)])
        assert L.tail_yaw_error(poly, d) == pytest.approx(a, abs=1e-9)
    rot = L.rot_axes(np.array([0.0, 1.0, 0.0]), L.EDGE_UP)
    pos = np.array([0.36, 0.05, 0.92])
    p2, r2 = L.steer_pose(pos, rot, [0.0, 0.0, 0.0], math.radians(10.0), L.GRIP_DEPTH)
    assert np.allclose(pos + rot[:, 2] * L.GRIP_DEPTH, p2 + r2[:, 2] * L.GRIP_DEPTH)
    assert np.allclose(r2 @ r2.T, np.eye(3))


# -- round 13: the clip stand ---------------------------------------------------------------------------------------------
def test_the_clip_holds_the_head_top_and_bottom_and_leaves_the_slot_and_root_open():
    b = L.BUILDS["wrap"]
    parts = {n: (k, g) for n, k, g in L.stand_parts(b)}
    hp = np.asarray(b.head_pos)
    half = np.abs(b.head_rot) @ (L.HEAD / 2)
    lo, hi = hp - half, hp + half
    for n in ("clip_bottom", "clip_top", "clip_back"):
        c_lo, c_hi = parts[n][1]
        assert c_lo[1] >= lo[1] - 1e-12 and c_hi[1] <= hi[1] + 1e-12  # inside the slot's faces: entry and exit open
        assert c_hi[0] <= hi[0] + 1e-12  # nothing past the +x face, where the strap leaves
        gap = np.max(np.maximum(c_lo - hi, lo - c_hi))  # the clip is out of the head's box
        assert gap == pytest.approx(L.CLIP_GAP, abs=1e-9), n
    assert parts["clip_bottom"][1][1][2] < lo[2] and parts["clip_top"][1][0][2] > hi[2]
    rod = parts["stand_rod"][1]
    assert rod[2][0] == pytest.approx(L.TOP + L.STAND_BASE["h"]) and rod[2][1] == pytest.approx(parts["clip_bottom"][1][0][2])
    base_lo, base_hi = parts["stand_base"][1]
    assert base_hi[0] < L.STRIP["x"][0]  # in front of the power strip
    assert L.stand_parts(L.BUILDS["free"]) == []


def test_the_strap_never_starts_in_the_clip_stand():
    for name in ("jig", "wrap", "tight"):
        b = L.BUILDS[name]
        poly = L.start_polyline(b)
        for n, kind, geo in L.stand_parts(b):
            if kind == "box":
                lo, hi = geo
                q = np.maximum(np.maximum(lo - poly, poly - hi), 0.0)
                assert np.min(np.linalg.norm(q, axis=1)) > L.W / 2, (name, n)
            else:
                xy, r, (z0, z1) = geo
                sel = poly[(poly[:, 2] > z0 - L.W) & (poly[:, 2] < z1 + L.W)]
                assert not len(sel) or np.min(np.linalg.norm(sel[:, :2] - xy, axis=1)) > r + L.T, (name, n)


def test_the_eased_end_pulls_past_snug_then_backs_off_along_the_pull_before_letting_go():
    for tr in L.CHECKED:
        if not tr.handover or tr.build != "jig":
            continue
        right = _timeline(L.trial_plan(tr)["right"])
        if tr.ease <= 0.0:
            assert "ease" not in right
            continue
        p2, eased = np.asarray(right["pull 2"][2][1]), np.asarray(right["ease"][2][1])
        back = p2 - eased
        assert np.linalg.norm(back) == pytest.approx(tr.ease, abs=1e-9)
        line = p2 - np.asarray(right["pull 1"][2][1])
        assert float(back @ line) > 0.999 * np.linalg.norm(back) * np.linalg.norm(line)  # straight back along the pull
        for label in ("hold snug", "release 2"):
            assert np.allclose(right[label][2][1], eased)  # held and let go where it eased to
        assert tr.ease <= tr.extra_pull + L.snug_through(L.BUILDS[tr.build]) - L.HANDOVER_OUT  # never back past the start
