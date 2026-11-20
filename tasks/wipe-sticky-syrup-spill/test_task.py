"""Checks of the task file with the official yam_arena package (no GPU, no Isaac Sim): it loads, its objects are where
syrup_logic says, the instruction names only the goal, every demo pose is reachable with yam_arena's own IK, and the
YAM finger envelope fits round the sprayer's head where the demo needs it.

    python -m pytest -q test_logic.py test_task.py
"""

import numpy as np
import pytest
from yam_arena.task_file import load

import armcheck

HERE = __import__("pathlib").Path(__file__).parent
TASK = load(HERE / "wipe_sticky_syrup_spill.py")
M = TASK.module
L = M.L
MM = 1e-3


def test_the_task_file_loads_with_unique_names():
    assert TASK.success is not None and TASK.score is not None and TASK.demo is not None
    assert len(set(TASK.names)) == len(TASK.names)
    for o in TASK.objects:
        o._check()


def test_the_instruction_states_the_goal_only():
    text = M.INSTRUCTION.lower()
    assert "syrup" in text and "table" in text
    for technique in ("spray", "water", "wet", "wipe", "towel", "cloth", "first", "soak", "scrub"):
        assert technique not in text


def test_the_docstring_names_every_object_the_policy_sees():
    doc = TASK.docs.lower()
    for word in ("syrup", "spray", "bottle", "trigger", "towel", "table"):
        assert word in doc


def test_nothing_scores_at_the_start():
    assert M.score({}) == 0.0 and not M.success({})


def test_the_syrup_layer_is_a_field_of_fine_tiles_with_a_patch_fallback():
    f = M.SPILL
    assert f.n == L.FN and f.tile == L.FINE and len(f.shown0) == L.FN ** 2
    parts = {p.name for p in M._patches()}
    assert len(parts) == 2 * L.GRID_N ** 2 and {f"s_{i}_{j}" for i, j in L.PUDDLE} <= parts


def test_the_bottle_matches_syrup_logic():
    b = M.BOTTLE
    assert b.trigger_rest == L.TRIGGER_REST and b.travel == L.TRIGGER_TRAVEL and b.mass == L.BOTTLE_MASS
    assert abs(b.pos[2] - L.TOP) < 2 * MM
    # the open fingers span the head and the nozzle with room, and the grip is narrower than "wide"
    assert L.NOZZLE[0] - L.BACK_X + 2 * 5 * MM <= L.FINGER_GAP["open"] * MM
    assert L.FINGER_GAP["squeeze"] * MM < L.FRONT_X - L.TRIGGER_TRAVEL - L.BACK_X < L.FINGER_GAP["carry"] * MM + 8 * MM
    # the trigger, squeezed home, stays clear of the shroud
    assert L.TRIGGER_REST[0] - L.TRIGGER[0] / 2 - L.TRIGGER_TRAVEL > L.SHROUD[2][0] + L.SHROUD[1][0] / 2 + 5 * MM


@pytest.mark.parametrize("variant", ["strokes", "zigzag", "circles", "outside_in", "upright_squeeze"])
def test_every_demo_pose_is_reachable(variant):
    for arm, w in armcheck.worst(variant).items():
        assert w["pos_mm"] < 1.0 and w["rot_deg"] < 1.0, (arm, w)


# -- the YAM finger envelope (the same boxes as the earlier tasks' checks: pads, then the finger's wider body) ---------
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


def bottle_boxes():
    """The bottle's solids in its own frame (centre, half size): body, shroud, head, nozzle."""
    return [((0.0, 0.0, L.BODY_H / 2), (L.BODY_R, L.BODY_R, L.BODY_H / 2)),
            (L.SHROUD[2], np.asarray(L.SHROUD[1]) / 2), (L.HEAD[2], np.asarray(L.HEAD[1]) / 2),
            ((L.NOZZLE[0] - 0.003, 0.0, L.NOZZLE[2]), (0.003, 0.0045, 0.0045))]


def gap_to_bottle(points, squeezed=0.0):
    """Smallest clearance (m) from points (bottle frame, bottle at the origin upright) to the bottle's solids, the
    trigger excluded (the fingers press it); negative inside."""
    p = np.asarray(points)
    best = 1.0
    for c, h in bottle_boxes():
        d = np.abs(p - np.asarray(c)) - np.asarray(h)
        outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
        inside = np.minimum(d.max(1), 0.0)
        best = min(best, float((outside + inside).min()))
    return best


def test_the_open_fingers_come_down_over_the_head_clear_of_it():
    steps = {s[4]: s for s in L.take_bottle(np.zeros(3) + [0.0, 0.0, 0.0])}
    for label in ("over the sprayer's head", "down either side of the head"):
        s = steps[label]
        pts = np.vstack([box_samples(b) for b in finger_boxes(s[1], s[3], L.FINGER_GAP["open"] * MM)])
        assert gap_to_bottle(pts) > 4 * MM, label


def test_the_closed_grip_touches_only_the_trigger_and_the_shroud():
    tcp, R = L.hand_for_bottle(np.zeros(3), squeezed=0.0)
    gap = L.FRONT_X - L.BACK_X
    pts = np.vstack([box_samples(b) for b in finger_boxes(tcp, R, gap)])
    # pads may touch the shroud's back face; nothing else of the bottle (body, head) within 3 mm
    body_head = [b for b in bottle_boxes() if b[0] != L.SHROUD[2]]
    p = np.asarray(pts)
    for c, h in body_head:
        d = np.abs(p - np.asarray(c)) - np.asarray(h)
        assert np.linalg.norm(np.maximum(d, 0.0), axis=1).min() > 3 * MM, c


def test_the_first_frame_look_is_set_at_spawn():
    """The layer spawns as the first frame must show it: the dried puddle only, no mist (round 1's first frame showed
    every patch: the look was set after the reset had rendered)."""
    shown, colors = L.Tracker().field_look()
    assert M.SPILL.shown0 == tuple(bool(v) for v in shown.ravel())
    vis = {p.name: p.visible for p in M._patches()}
    assert {n for n, v in vis.items() if v} == {f"s_{i}_{j}" for i, j in L.PUDDLE}
    mist = next(p for p in M.BOTTLE.housing_parts if p.name == "mist")
    assert not mist.visible


def test_the_wiping_arm_clears_the_standing_bottle():
    """Round 2: the left arm knocked the bottle over while scrubbing the spill's right edge. With the spill moved 8 cm
    left, the arm's collision hulls clear the bottle by 2 cm over every scrub pose and stay off it at every touch-up
    the demo makes (syrup_logic.touch_up_reachable: within 9 cm of the spill's middle)."""
    import math

    import hulls
    from yam_arena import ik
    from yam_arena.constants import READY
    from yam_arena.station import BASES

    B = np.asarray(L.BOTTLE_START, float)
    n = 16
    body = np.array([(L.BODY_R * math.cos(2 * math.pi * k / n), L.BODY_R * math.sin(2 * math.pi * k / n), z)
                     for z in (0.0, L.BODY_H) for k in range(n)])
    bottle = [hulls.part("body", body + B)] + [hulls.box(nm, B + np.asarray(c), np.eye(3), sz)
                                               for nm, sz, c in (L.SHROUD, L.HEAD)]
    q = np.array(READY, float)
    worst_scrub, worst_touch = 1.0, 1.0
    scrub = [p for pat in ("strokes", "zigzag", "circles") for p in L.scrub_path(pat)]
    touch = [tuple(c) for c in L.CENTRES.reshape(-1, 2) if L.touch_up_reachable(c)]
    for k, p in enumerate(scrub + touch):
        q, pe, _ = ik.solve(L.press_tcp(*p), ik.matrix_to_quat_xyzw(L.PRESS_ROT), q, base_pos=BASES["left"], iters=150)
        if k % 3:
            continue
        d = hulls.min_dist(hulls.arm_parts("left", q, 0.0), bottle, cap=0.08)[0]
        if k < len(scrub):
            worst_scrub = min(worst_scrub, d)
        else:
            worst_touch = min(worst_touch, d)
    assert worst_scrub > 0.02 and worst_touch > 0.0, (worst_scrub, worst_touch)


def test_the_towel_starts_clear_of_the_spill_and_the_bottle():
    t0, half = np.asarray(L.TOWEL_C), np.asarray(L.TOWEL_SIZE[:2]) / 2
    for i, j in L.PUDDLE:
        c = L.cell_centre(i, j)
        assert max(abs(c[0] - t0[0]) - half[0], abs(c[1] - t0[1]) - half[1]) > L.CELL / 2 + 0.01
    b = np.asarray(L.BOTTLE_START[:2])
    assert max(abs(b[0] - t0[0]) - half[0], abs(b[1] - t0[1]) - half[1]) > L.BODY_R + 0.05




def test_the_demo_is_the_recorded_trial_and_passes_the_runners_limits():
    """demo() replays demo_actions.csv (the full-score round-3 trial) through kit_replay; every row passes the runner's
    guards unchanged, and the replay ends inside the run limit."""
    import kit_replay

    d = TASK.demo()
    r = d["left"][0]
    assert d["right"][0] is r and len(r.rows) == 1228
    hold = kit_replay.hold_steps(kit_replay.DEFAULT_HOLD_S, 20)
    assert kit_replay.stop_step(len(r.rows), hold) < 2400
    assert kit_replay.check_actions(r.rows, 20)["rows"] == 1228
