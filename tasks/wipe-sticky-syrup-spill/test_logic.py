"""Checks of syrup_logic (pure Python): the spill, the spray rule, the rub rule, the foul rule, the score, and the dry
run of every demo variant through the tracker.

    python -m pytest -q test_logic.py test_task.py
"""

import math

import numpy as np
import pytest

import dryrun
import syrup_logic as L

TOWEL = dryrun.towel_nodes(L.TOWEL_C)
UP = np.array([0.0, 0.0, 0.0, 1.0])


def inputs(trigger=0.0, nodes=TOWEL, tips=(), pos=L.BOTTLE_START, quat=UP, grippers=((0.2, -0.25, 1.0),)):
    return L.Inputs(bottle_pos=np.asarray(pos, float), bottle_quat=np.asarray(quat, float), trigger=trigger,
                    nodes=None if nodes is None else np.asarray(nodes, float), tips=[np.asarray(t, float) for t in tips],
                    grippers=[np.asarray(g, float) for g in grippers])


def aimed(target):
    """The bottle's pose (pos, quat) when the hand aims at ``target``."""
    tcp, R, _ = L.aim_hand(target, L.SPRAY_TILT, L.SPRAY_HEIGHT)
    bR = L.bottle_rot_of_hand(R)
    return tcp - bR @ np.array([L.GRIP_X, 0.0, L.GRIP_Z]), L.matrix_to_quat_xyzw(bR)


def test_the_puddle_is_an_irregular_blob_about_10_cm_across():
    assert L.N_PUDDLE == 19
    cells = np.array([L.cell_centre(i, j) for i, j in L.PUDDLE])
    span = cells.max(0) - cells.min(0) + L.CELL
    assert 0.09 <= span[0] <= 0.11 and 0.09 <= span[1] <= 0.11
    s = L.start_cells()
    assert (s == L.STICKY).sum() == L.N_PUDDLE and (s == L.WET).sum() == 0


def test_the_spray_lands_where_the_hand_aims():
    for target in L.spray_targets():
        pos, q = aimed(target)
        (hx, hy), r, rng = L.spray_patch(pos, q)
        assert math.hypot(hx - target[0], hy - target[1]) < 1e-6
        assert 0.12 < rng < 0.20 and 0.02 < r < 0.032


def test_an_upright_bottle_sprays_nothing_onto_the_table():
    assert L.spray_patch(L.BOTTLE_START, UP) is None


def test_the_tilted_bottle_stays_clear_of_the_table():
    for target in L.spray_targets():
        pos, q = aimed(target)
        R = L.quat_xyzw_to_matrix(q)
        body = np.array([[x, y, z] for x in (-L.BODY_R, L.BODY_R) for y in (-L.BODY_R, L.BODY_R) for z in (0, L.BODY_H)])
        assert ((pos + body @ R.T)[:, 2] > L.TOP + 0.02).all()


def test_a_squeeze_sprays_once_until_the_trigger_comes_back():
    t = L.Tracker()
    pos, q = aimed(L.SPILL_C)
    for trig in (0.0, 0.009, 0.010, 0.009, 0.006, 0.009, 0.004, 0.009):
        t.step(inputs(trigger=trig, pos=pos, quat=q))
    assert t.sprays == 2  # the second press didn't re-arm (6 mm); the third did (4 mm)
    assert (t.cells == L.WET).sum() > 0


def test_a_carry_squeeze_never_sprays():
    """The dry run's spring model: the "carry" grip leaves the trigger short of REARM_AT, the "squeeze" bottoms it."""
    rest = L.FRONT_X - L.BACK_X
    travel = lambda gap: dryrun.FINGER_K / 2 * (rest - gap * L.MM) / (L.TRIGGER_RATE + dryrun.FINGER_K / 2)
    assert travel(L.FINGER_GAP["carry"]) < L.REARM_AT
    assert travel(L.FINGER_GAP["squeeze"]) > L.TRIGGER_TRAVEL


def press(t, start_xy, end_xy, steps=20, nodes0=TOWEL, follow=1.0):
    """Press a fingertip on the towel at ``start_xy`` and move it to ``end_xy``, the towel following."""
    nodes = np.asarray(nodes0, float).copy()
    a, b = np.asarray(start_xy, float), np.asarray(end_xy, float)
    for k in range(steps + 1):
        tip = np.r_[a + (b - a) * k / steps, L.TOP + L.TOWEL_SIZE[2]]
        if k:
            nodes[:, :2] += follow * (b - a) / steps
        t.step(inputs(nodes=nodes, tips=[tip]))
    return nodes


def towel_at(xy):
    return dryrun.towel_nodes(xy)


def blank(i=6, j=6, m=0.0, w=0.0):
    """A tracker with an empty field but for square (i, j): syrup m and water w on its four fine cells."""
    t = L.Tracker()
    t.m[:] = 0.0
    t.w[:] = 0.0
    t.m[2 * i:2 * i + 2, 2 * j:2 * j + 2] = m
    t.w[2 * i:2 * i + 2, 2 * j:2 * j + 2] = w
    return t


def square_m(t, i=6, j=6):
    return float(t.square_m()[i, j])


def test_soft_syrup_comes_off_in_a_few_passes_and_dried_syrup_smears_along():
    c = L.cell_centre(6, 6)
    t = blank(m=1.0, w=1.0)  # dissolved
    nodes = press(t, c + [0.0, -0.02], c + [0.0, 0.02], nodes0=towel_at(c + [0.0, -0.02]))
    assert square_m(t) > L.CLEAN_M  # one 4 cm stroke isn't enough
    for rep in range(4):
        a, b = (c + [0.0, 0.02], c + [0.0, -0.02]) if rep % 2 == 0 else (c + [0.0, -0.02], c + [0.0, 0.02])
        nodes = press(t, a, b, nodes0=nodes)
    assert square_m(t) < L.CLEAN_M and t.load > 3.0  # the towel took it
    t2 = blank(m=1.0, w=0.0)  # dried
    press(t2, c + [0.0, -0.01], c + [0.0, 0.05], steps=30, nodes0=towel_at(c + [0.0, -0.01]))
    assert square_m(t2) > L.CLEAN_M and square_m(t2, 6, 7) > L.CLEAN_M  # smeared onward, not cleared
    assert t2.load < 0.5  # the dry towel took almost none


def test_a_fingertip_off_the_table_or_not_on_cloth_doesnt_rub():
    c = L.cell_centre(6, 6)
    t = blank(m=1.0, w=1.0)
    nodes = towel_at(c)
    high = np.r_[c, L.TOP + L.PRESS_Z + 0.005]
    t.step(inputs(nodes=nodes, tips=[high]))
    moved = nodes.copy()
    moved[:, :2] += 0.01
    t.step(inputs(nodes=moved, tips=[high]))
    assert t.rubbed_total == 0.0


def test_bare_fingers_on_syrup_foul_the_gripper_but_not_through_the_towel():
    c = L.cell_centre(6, 6)
    t = L.Tracker()
    t.step(inputs(nodes=towel_at(c), tips=[np.r_[c, L.TOP + L.TOWEL_SIZE[2]]]))
    assert not t.facts.fouled
    t.step(inputs(nodes=TOWEL, tips=[np.r_[c, L.TOP + 0.002]]))
    assert t.facts.fouled and L.task_score(t.facts) == 0.0


def test_the_score_levels():
    f = L.Facts(puddle_clean=4, dirty=15)
    assert L.task_score(f) == 0.0
    f.puddle_clean = 5
    assert L.task_score(f) == 0.25
    f.puddle_clean = 10
    assert L.task_score(f) == 0.5
    f.puddle_clean, f.dirty = 19, 2  # the puddle clean but smears left
    assert L.task_score(f) == 0.75
    f.dirty, f.bottle_upright, f.hands_clear, f.still = 0, True, True, True
    assert L.task_score(f) == 1.0
    for k in ("bottle_upright", "hands_clear", "still"):
        g = L.Facts(**{**f.__dict__, k: False})
        assert L.task_score(g) == 0.75, k
    for k in ("bottle_on_table", "towel_on_table"):
        g = L.Facts(**{**f.__dict__, k: False})
        assert L.task_score(g) == 0.0, k
    assert L.task_score(L.Facts(**{**f.__dict__, "fouled": True})) == 0.0


def test_hands_on_the_bottle_or_the_towel_are_not_clear():
    t = L.Tracker()
    grip, _ = L.hand_for_bottle(np.asarray(L.BOTTLE_START))
    t.step(inputs(grippers=[grip]))
    assert not t.facts.hands_clear
    t.step(inputs(grippers=[np.r_[L.TOWEL_C, L.TOP + 0.01]]))
    assert not t.facts.hands_clear
    t.step(inputs(grippers=[(0.2, -0.25, 1.0)]))
    assert t.facts.hands_clear


def test_the_first_frame_shows_the_dried_puddle_only():
    t = L.Tracker()
    ch = dict(t.visual_changes())
    assert {k for k, v in ch.items() if v} == {f"s_{i}_{j}" for i, j in L.PUDDLE}
    assert t.visual_changes() == []
    shown, colors = t.field_look()
    assert shown.sum() == 4 * L.N_PUDDLE and colors.shape == (L.FN, L.FN, 3)
    assert (colors[shown] < 0.5).all()  # dried syrup: dark amber


@pytest.mark.parametrize("variant", sorted(L.VARIANTS))
@pytest.mark.parametrize("follow", [0.5, 1.0])
def test_every_variant_scores_what_it_should_in_the_dry_run(variant, follow):
    r = dryrun.run(variant, follow)
    assert r["score"] == L.EXPECTED[variant], r["log"]


@pytest.mark.parametrize("pattern", ["strokes", "zigzag", "circles"])
def test_the_patterns_finish_inside_the_run_limit(pattern):
    """2,400 steps (2 minutes), with the towel taking 80% of the fingertips' motion (round 1 measured 81%)."""
    r = dryrun.run(pattern, 0.8)
    assert r["seconds"] <= 110.0


def test_a_dry_wipe_spreads_the_syrup():
    r = dryrun.run("dry_wipe", 1.0)
    assert r["log"]["smeared"] > 10 and r["log"]["puddle_clean"] == 0


def test_cloth_trembling_under_a_still_fingertip_is_no_rub():
    c = L.cell_centre(6, 6)
    t = blank(m=1.0, w=1.0)
    tip = np.r_[c, L.TOP + L.TOWEL_SIZE[2]]
    for k in range(40):  # the cloth shivers 2 mm back and forth; the hand doesn't move
        t.step(inputs(nodes=towel_at(c + [0.002 * (-1) ** k, 0.0]), tips=[tip]))
    assert t.rubbed_total == 0.0 and square_m(t) == pytest.approx(1.0)


def test_a_fingertip_resting_on_the_towel_unpressed_doesnt_rub():
    c = L.cell_centre(6, 6)
    t = blank(m=1.0, w=1.0)
    press(t, c + [0.0, -0.02], c + [0.0, 0.02], nodes0=towel_at(c + [0.0, -0.02]))
    assert t.rubbed_total > 0  # pressed (tip at the towel's top, 4 mm): rubs
    t2 = blank(m=1.0, w=1.0)
    a, b = c + [0.0, -0.02], c + [0.0, 0.02]
    nodes = towel_at(a)
    for k in range(21):  # the same stroke with the fingertip 7 mm up (resting on the towel, not pressing it)
        tip = np.r_[a + (b - a) * k / 20, L.TOP + 0.007]
        if k:
            nodes = nodes.copy()
            nodes[:, :2] += (b - a) / 20
        t2.step(inputs(nodes=nodes, tips=[tip]))
    assert t2.rubbed_total == 0.0


def test_a_bottle_tipped_past_level_sprays_nothing():
    t = L.Tracker()
    pos, q = aimed(L.SPILL_C)
    flipped = L.matrix_to_quat_xyzw(L.quat_xyzw_to_matrix(q) @ L.rx(180.0))
    for trig in (0.0, 0.010):
        t.step(inputs(trigger=trig, pos=pos, quat=flipped))
    assert t.sprays == 1 and t.w.sum() == 0.0


def test_the_towel_takes_the_spray_where_it_lies_and_a_damp_towel_cleans_dried_syrup():
    t = L.Tracker()
    pos, q = aimed(L.SPILL_C)
    over = towel_at(L.SPILL_C)  # the towel lying over the middle of the spill
    for trig in (0.0, 0.010):
        t.step(inputs(trigger=trig, pos=pos, quat=q, nodes=over))
    assert t.damp() and t.towel_hits == 1
    under = np.linalg.norm(L.FINE_CENTRES - np.asarray(L.SPILL_C), axis=2) < 0.01
    assert t.w[under].sum() == 0.0  # the cells under the towel stayed dry
    t2 = blank(m=1.0, w=0.0)
    t2.towel_water = 15.0  # a well-damped towel on dried syrup
    c = L.cell_centre(6, 6)
    nodes = towel_at(c + [0.0, -0.02])
    for rep in range(14):
        a, b = (c + [0.0, -0.02], c + [0.0, 0.02]) if rep % 2 == 0 else (c + [0.0, 0.02], c + [0.0, -0.02])
        nodes = press(t2, a, b, nodes0=nodes)
    assert t2.facts.puddle_clean >= 0 and t2.square_m().sum() * 4 < 2.0  # most of it lifted, slowly


def test_poking_the_syrup_then_cleaning_it_all_still_scores_0():
    r = dryrun.run("poke_then_wipe", 0.8)
    assert r["log"]["fouled"] and r["score"] == 0.0


def test_syrup_is_conserved_between_the_table_and_the_towel():
    t = L.Tracker()
    total = t.m.sum()
    c = L.cell_centre(6, 6)
    t.w[:] = 1.0
    nodes = towel_at(c + [0.0, -0.04])
    for rep in range(6):
        a, b = (c + [0.0, -0.04], c + [0.0, 0.04]) if rep % 2 == 0 else (c + [0.0, 0.04], c + [0.0, -0.04])
        nodes = press(t, a, b, nodes0=nodes)
    assert t.m.sum() + t.load + t.off_grid == pytest.approx(total, rel=1e-9)


def test_a_loaded_towel_leaves_a_film_and_a_clean_one_does_not():
    c = L.cell_centre(2, 2)  # a clean square away from the spill
    for load, expect_film in ((0.0, False), (L.LOAD_CAP, True)):
        t = blank(m=0.0)
        t.load = load
        press(t, c + [0.0, -0.02], c + [0.0, 0.02], nodes0=towel_at(c + [0.0, -0.02]))
        assert (t.m.max() > 0.005) == expect_film


def test_a_dry_wipe_leaves_streaks_beyond_the_spill():
    r = dryrun.run("dry_wipe", 0.8)
    assert r["log"]["smeared"] >= 8 and r["log"]["puddle_clean"] == 0 and r["log"]["towel_load"] < 15
