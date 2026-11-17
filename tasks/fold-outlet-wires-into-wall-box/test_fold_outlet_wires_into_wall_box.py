"""Offline checks of round 13 (the copper rule, real sizes, kinked 8 in wires, ready-for-screws scoring). No simulator.
    PYTHONPATH=<yam-isaac-lab-arena> python -m pytest -q test_round13.py
Rounds 1-12's tests (other sizes, other scoring) are kept in ../orig for the record. No arm reach checks here.
"""
import math

import numpy as np
import pytest

import copper
import outlet_logic as L
import fold_outlet_wires_into_wall_box as task

UPRIGHT = np.array([0.0, 0.0, 0.0, 1.0])
MOUNTED = np.array([L.WALL_X, *L.OPEN_C])  # device origin with the strap's back face on the wall
H = task.HARNESS


def _inside_wires():
    return {"hot": np.array([[L.WALL_X + 0.010, L.OPEN_C[0] - 0.01, L.OPEN_C[1]], [L.WALL_X + 0.04, L.OPEN_C[0], L.OPEN_C[1]]])}


# -- the copper rule ---------------------------------------------------------------------------------------------------------
def test_copper_yields_at_59_and_is_fully_plastic_near_100_n_mm():
    assert copper.M_Y == pytest.approx(0.0592, abs=0.0005) and copper.M_P == pytest.approx(0.1005, abs=0.0005)
    kap, m = copper.moment_curve(hard=0.0)
    assert m[-1] == pytest.approx(copper.M_P, rel=0.01)  # no hardening: the fully plastic moment at a tight bend
    kp, flow = copper.flow_table()
    assert flow[0] == pytest.approx(copper.M_Y) and np.all(np.diff(flow) >= -1e-12) and np.all(np.diff(kp) >= 0)
    assert np.interp(50.0, kp, flow) > 1.5 * copper.M_P  # work hardening: a 20 mm radius bend takes over 150 N mm


def test_an_elastic_bend_springs_back_and_a_plastic_one_stays():
    kp, flow = copper.flow_table()
    k, s = copper.EI / L.SEGMENT, L.SEGMENT
    groups = np.array([[0, 1]])
    small = 0.5 * copper.M_Y / k
    rest, acc, cap = copper.plastic_step([small, 0.0], [0.0, 0.0], [0.0], groups, k, s, kp, flow)
    assert np.allclose(rest, 0.0) and acc[0] == 0.0 and cap[0] == pytest.approx(copper.M_Y)
    q, rest, acc = np.array([0.20, 0.0]), np.zeros(2), np.zeros(1)
    for _ in range(50):
        rest, acc, cap = copper.plastic_step(q, rest, acc, groups, k, s, kp, flow)
    held = k * (q[0] - rest[0])  # the moment it settles at is the flow moment for the bend it has been through
    assert held == pytest.approx(cap[0], rel=0.02) and cap[0] > copper.M_P and rest[0] > 0.17
    assert rest[1] == 0.0  # bent about one axis only


def test_a_diagonal_bend_yields_at_the_same_moment_as_a_straight_one():
    kp, flow = copper.flow_table()
    k, s, groups = copper.EI / L.SEGMENT, L.SEGMENT, np.array([[0, 1]])
    a = 1.2 * copper.M_Y / k / math.sqrt(2)  # each axis under yield on its own, the two together 20% over
    rest, acc, _ = copper.plastic_step([a, a], [0.0, 0.0], [0.0], groups, k, s, kp, flow)
    assert acc[0] > 0 and rest[0] == pytest.approx(rest[1]) and rest[0] > 0


# -- the wires and the harness -------------------------------------------------------------------------------------------------
def test_eight_inch_wires_in_61_links_with_a_half_link_at_each_end():
    assert L.wire_length(L.WIRES[0]) == pytest.approx(0.203) and L.wire_length(L.WIRES[1]) == pytest.approx(0.203)
    assert L.links_needed() == 61 <= 64
    for w in L.wires():
        pts = L.start_polyline(w)
        ch = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        assert len(pts) == L.segments(w) + 1 and np.allclose(pts[0], w.root) and np.allclose(pts[-1], L.terminal_world(w))
        assert ch[0] == pytest.approx(L.SEGMENT / 2, rel=0.02) and ch[-1] == pytest.approx(L.SEGMENT / 2, rel=0.02)
        assert np.all(ch[1:-1] <= L.SEGMENT * 1.001) and np.all(ch[1:-1] > 0.8 * L.SEGMENT)  # a link across a fold cuts its corner
        assert float(np.sum(ch)) == pytest.approx(L.wire_length(w), rel=0.03)


def test_the_harness_uses_the_copper_rule():
    H._check()
    assert H.rigid_root and H.segment == pytest.approx(L.SEGMENT) and H.flow_m[0] == pytest.approx(copper.M_Y)
    assert H.stiffness == pytest.approx(copper.EI / L.SEGMENT)
    joints = sum(w.segments - 1 for w in H.wires)
    assert H.joint_count() == joints == 19 + 19 + 18 and H.dof_count() == 2 * joints
    for w in H.wires:
        assert H.link_lengths(w)[0] == pytest.approx(L.SEGMENT / 2, rel=0.02)


def test_the_state_reads_the_links_back_into_the_same_wires():
    s = H.state_at()
    for w in L.wires():
        assert np.allclose(s.polylines[w.name], L.start_polyline(w), atol=1e-6)


def test_the_wires_start_apart_and_clear_of_the_outlet_and_the_box():
    ws = L.wires()
    dense = {}
    for w in ws:
        p = L.start_polyline(w)
        dense[w.name] = (p[:-1, None, :] + np.linspace(0, 1, 10)[None, :, None] * (p[1:] - p[:-1])[:, None, :]).reshape(-1, 3)
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            d = np.min(np.linalg.norm(dense[ws[i].name][:, None] - dense[ws[j].name][None], axis=2))
            assert d > ws[i].radius + ws[j].radius + 0.001, (ws[i].name, ws[j].name)
    for w in ws:  # only the links that don't collide with the outlet may be within 1 mm of it
        p = L.start_polyline(w)
        n = len(p) - 1
        for k in range(n - L.END_FILTER):
            seg = p[k] + np.linspace(0, 1, 25)[:, None] * (p[k + 1] - p[k]) - L.DEVICE_START
            pad = w.radius + 0.001
            near = ((seg[:, 0] > -L.STRAP_T - L.FACE[0] - pad) & (seg[:, 0] < L.BODY[0] + pad)
                    & (np.abs(seg[:, 1]) < L.BODY[1] / 2 + pad) & (np.abs(seg[:, 2]) < L.STRAP_H / 2 + pad))
            assert not near.any(), (w.name, k)
        d = dense[w.name]
        box = d[d[:, 0] > L.WALL_X]  # inside the box: clear of its walls and back by 2 mm or more
        assert L.BOX_W / 2 - np.abs(box[:, 1] - L.OPEN_C[0]).max() - w.radius > 0.002
        assert L.BOX_H / 2 - np.abs(box[:, 2] - L.OPEN_C[1]).max() - w.radius > 0.002
        assert L.WALL_X + L.BOX_D - box[:, 0].max() - w.radius > 0.002


def test_the_outlet_fits_the_box_and_the_grip_clears_the_screws():
    assert L.BOX_H / 2 - L.BOSS_P - L.BODY[2] / 2 > 0.003  # the body passes between the bosses
    assert L.BOX_W / 2 - L.TERMINAL_Y - L.INSULATED_R > 0.003  # a wire fits between the body's side and the box
    assert L.STRAP_H / 2 - L.BOX_H / 2 - L.BOX_WALL > 0.004  # the strap's ends land on the wall, above and below the box
    tips = L.GRIP + (L.PAD_TIP, 0.0, 0.0)
    assert 0 < tips[0] < L.WIRES[0].terminal[0] - L.INSULATED_R and L.WIRES[0].terminal[2] - tips[2] > 0.02


def test_each_wire_keeps_two_old_folds_and_they_start_stiff():
    kp, flow = copper.flow_table()
    for w in L.wires():
        h = L.start_hardening(w)
        assert len(h) == L.joints(w)
        sharp = np.degrees(h * L.SEGMENT)
        assert np.sum(sharp > 35) >= 2 and sharp.max() < 80  # two folds, each sharp but no hairpin
        assert np.interp(h.max(), kp, flow) > 2.0 * copper.M_P  # a fold starts over twice as hard to bend as fresh wire
        assert np.interp(np.median(h), kp, flow) < 1.8 * copper.M_P  # the runs between the folds much less so
    assert len(H.hardened0) == H.joint_count()
    assert np.allclose(H.hardened0, np.concatenate([L.start_hardening(w) for w in L.wires()]))


# -- scoring: ready for its screws -------------------------------------------------------------------------------------------------
def test_ready_for_its_screws_scores_one_even_7_mm_off_the_wall():
    assert L.task_score(MOUNTED, UPRIGHT, _inside_wires(), hand_clear=0.05, clear_s=2.0) == 1.0
    assert L.task_score(MOUNTED - (0.007, 0, 0), UPRIGHT, _inside_wires(), hand_clear=0.05, clear_s=2.0) == 1.0  # sprung back
    assert L.task_score(MOUNTED - (0.016, 0, 0), UPRIGHT, _inside_wires(), hand_clear=0.05, clear_s=2.0) == 0.0  # no screw reaches


def test_off_its_holes_scores_half():
    assert L.task_score(MOUNTED + (0, 0, 0.004), UPRIGHT, _inside_wires(), hand_clear=0.05, clear_s=2.0) == 0.5


def test_any_wire_outside_the_box_scores_nothing():
    showing = {"hot": np.array([[L.WALL_X - 0.003, L.OPEN_C[0], L.OPEN_C[1]]])}
    over_edge = {"ground": np.array([[L.WALL_X + 0.001, L.OPEN_C[0], L.OPEN_C[1] + L.BOX_H / 2 + 0.004]])}  # under the strap
    deep_outside_ok = {"hot": np.array([[L.WALL_X + 0.02, L.OPEN_C[0], L.OPEN_C[1]]])}
    assert L.wires_outside(showing) == ["hot"] and L.wires_outside(over_edge) == ["ground"]
    assert L.wires_outside(deep_outside_ok) == []
    for bad in (showing, over_edge):
        assert L.task_score(MOUNTED, UPRIGHT, bad, hand_clear=0.05, clear_s=2.0) == 0.0


def test_full_credit_needs_it_let_go_and_still_for_two_seconds():
    # in place but held, pressed, just released or still moving: half, never full (a held outlet springs out when let go)
    assert L.task_score(MOUNTED, UPRIGHT, _inside_wires(), hand_clear=0.02, clear_s=0.0) == 0.0   # a gripper on it
    assert L.task_score(MOUNTED, UPRIGHT, _inside_wires(), hand_clear=0.05, clear_s=1.9) == 0.0   # let go 1.9 s ago
    assert L.task_score(MOUNTED, UPRIGHT, _inside_wires(), hand_clear=0.05, speed=0.02, clear_s=2.0) == 0.0  # moving
    assert L.task_score(MOUNTED, UPRIGHT, _inside_wires(), hand_clear=0.05, clear_s=2.0) == 1.0


def test_a_gripper_holding_the_wires_in_is_not_let_go():
    wires = _inside_wires()
    on_wire = wires["hot"][0] + (0.0, 0.0, 0.005)       # a fingertip 5 mm from a wire, inside the box mouth
    far = np.array([L.WALL_X - 0.10, L.OPEN_C[0], L.OPEN_C[1]])
    assert L.wire_clearance([on_wire], wires) == pytest.approx(0.005, abs=1e-6)
    assert L.wire_clearance([far], wires) > 0.03


def test_fallen_over_scores_nothing():
    tipped = np.array([0.0, math.sin(math.radians(10)), 0.0, math.cos(math.radians(10))])  # 20 degrees about y
    assert L.task_score(MOUNTED, tipped, _inside_wires(), hand_clear=0.05, clear_s=2.0) == 0.0


def test_doing_nothing_scores_zero():
    s = H.state_at()
    assert L.task_score(s.pos, s.quat_xyzw, s.polylines, hand_clear=1.0, clear_s=10.0) == 0.0


# -- the first launch's trials ---------------------------------------------------------------------------------------------------
def test_round_13_runs_three_pushes_with_no_wire_handling_then_the_tuck():
    names = [t.name for t in L.TRIALS]
    assert names[:3] == ["e1-naive", "e1-careful", "e1-tilt"] and all(not t.crest for t in L.TRIALS[:3])
    assert all(t.crest for t in L.TRIALS[3:] if t.name not in ("t-tip", "u-low", "u-low60")) and {t.ground for t in L.TRIALS} == {"side15"}
    for t in L.TRIALS:
        plan = L.trial_plan(t)
        assert set(plan) == {"left", "right"} and L.plan_seconds(plan) < 70
        assert sum(s[0] for s in plan["left"]) == pytest.approx(sum(s[0] for s in plan["right"]))
        assert set(task.plan_waypoints(plan)) == {"left", "right"}
    assert all(L.plan_seconds(L.trial_plan(t)) < 70 for t in L.TRIALS)  # each trial well under the 2 minute run limit


def test_the_live_presses_aim_at_each_loop_and_keep_the_plans_timing():
    tops = {"hot": np.array([L.WALL_X - 0.026, L.OPEN_C[0] - 0.010, L.OPEN_C[1] + 0.077]), "neutral": None,
            "ground": np.array([L.WALL_X - 0.040, L.OPEN_C[0] + 0.035, L.OPEN_C[1] + 0.045])}
    names = ("hot", "neutral", "ground")
    live, fixed = L.crest_plan_at(tops, names), L.crest_plan(names)
    assert [s_[0] for s_ in live] == [s_[0] for s_ in fixed] and len(live) == 16
    onto, press = live[1], live[2]
    tip_on = np.array(onto[1]) + L.PAD_TIP * (L.crest_rotation(0.0) @ np.array([0.0, 0.0, 1.0]))
    tip_in = np.array(press[1]) + L.PAD_TIP * (L.crest_rotation(0.0) @ np.array([0.0, 0.0, 1.0]))
    assert np.allclose(tip_on, tops["hot"] + (-0.008, 0.0, 0.010), atol=1e-9)     # behind the loop's top, just above it
    assert tip_in[0] == pytest.approx(L.WALL_X + 0.012) and tip_in[2] <= L.OPEN_C[1] + 0.016 + 1e-9     # past the lip
    sweep = tip_in - tip_on                                                     # toward the wall and down, end face first
    assert sweep[0] > 0 and sweep[2] < 0 and sweep @ (L.crest_rotation(0.0) @ np.array([0.0, 0.0, 1.0])) > 0.8 * np.linalg.norm(sweep)
    cross = tip_on + sweep * (L.WALL_X - tip_on[0]) / sweep[0]                  # where the tip's centre crosses the wall face
    assert cross[2] + 0.008 < L.OPEN_C[1] + L.BOX_H / 2 - 0.004                 # its top edge under the lip there
    assert L.crest_start(0) == 0 and L.crest_start(2) == pytest.approx(14.0)
    assert tip_in[2] < L.OPEN_C[1] + L.BOX_H / 2 - 0.008                       # the fingertips' top edge under the lip
    far_y = np.array(live[12][1])[1]                                            # the ground's press is kept inside the mouth
    assert abs(far_y - L.OPEN_C[0]) <= L.BOX_W / 2 - 0.010 + 1e-9
    for t_ in L.TUCKS:
        plan = L.trial_plan(t_)
        assert sum(s_[0] for s_ in plan["left"]) == pytest.approx(sum(s_[0] for s_ in plan["right"]))
        assert set(task.plan_waypoints(plan)) == {"left", "right"}


def test_the_run_can_be_stopped_after_each_piece():
    whole = L.trial_plan(L.trial("f1-tuck"))
    ends = []
    for k in (1, 2, 3, 4, 5):
        tr = L.trial(f"p{k}")
        plan = L.trial_plan(tr)
        end = L.piece_end(tr, whole)
        ends.append(end)
        for arm in ("left", "right"):
            assert sum(s_[0] for s_ in plan[arm]) == pytest.approx(end + 2.0)
            assert plan[arm][-1][4].startswith("hold still") and plan[arm][-1][1] == plan[arm][-2][1]
        assert set(task.plan_waypoints(plan)) == {"left", "right"}
    assert ends == sorted(ends) and ends[1] - ends[0] == pytest.approx(7.5) and ends[2] - ends[1] == pytest.approx(6.5)
    assert L.trial_plan(L.trial("p1"))["right"][-2][4] == "lower it for the left hand"   # piece 1: held low at the stand-off
    assert L.trial_plan(L.trial("p5"))["right"][-2][4] == "grip hold"                     # piece 5: pushed home, still held
    live = L.cut_steps([whole["left"][0]] + L.crest_plan_at({}, ("hot", "neutral", "ground")), ends[1])
    assert live[-2][4] == "lift up and away" and sum(s_[0] for s_ in live) == pytest.approx(ends[1] + 2.0)


def test_loop_tops_finds_the_highest_point_in_front_of_the_wall():
    line = np.array([[L.WALL_X + 0.03, 0, 1.0], [L.WALL_X - 0.01, 0, 1.02], [L.WALL_X - 0.03, 0, 1.05], [L.WALL_X - 0.05, 0, 1.01]])
    tops = L.loop_tops({"hot": line, "neutral": line[:1]}, ("hot", "neutral"))
    assert np.allclose(tops["hot"], line[2]) and tops["neutral"] is None


def test_the_tilt_leads_with_the_top_edge():
    steps = L.trial_path(L.trial("e1-tilt"))
    assert steps[0][3] > 0 and steps[1][3] > 0 and len(steps[2]) == 3  # tipped, tipped, then level
    assert (L.roty(steps[0][3]) @ np.array([0.0, 0.0, 1.0]))[0] > 0  # a positive pitch moves the top toward the wall


def test_the_lean_pushes_a_loop_sideways_to_the_centre_and_the_tipped_push_keeps_its_timing():
    y, z = float(L.OPEN_C[0]), float(L.OPEN_C[1])
    tops = {"neutral": np.array([L.WALL_X - 0.021, y + 0.024, z + 0.050]), "hot": np.array([L.WALL_X - 0.023, y - 0.003, z + 0.054])}
    plan = L.lean_plan_at(tops, ("neutral", "hot"))
    ax = L.crest_rotation(0.0) @ np.array([0.0, 0.0, 1.0])
    beside, lean = (np.array(plan[k][1]) + L.PAD_TIP * ax for k in (1, 2))
    assert beside[1] == pytest.approx(y + 0.042) and lean[1] == pytest.approx(y + 0.015)   # from outside the loop to 15 mm
    assert beside[0] == pytest.approx(lean[0]) and beside[0] < L.WALL_X - 0.010              # in front of the wall throughout
    assert beside[2] == pytest.approx(lean[2]) and beside[2] == pytest.approx(z + 0.042)     # level, just under the loop's top
    assert plan[6][1] == plan[7][1]                                                          # the hot is near the centre: left alone
    assert [s_[0] for s_ in plan] == [s_[0] for s_ in L.crest_plan_at(tops, ("neutral", "hot"))]
    for name in ("t-tip", "t-lean", "t-lean-press"):
        t_ = L.trial(name)
        tp = L.trial_plan(t_)
        assert sum(s_[0] for s_ in tp["left"]) == pytest.approx(sum(s_[0] for s_ in tp["right"]))
        assert set(task.plan_waypoints(tp)) == {"left", "right"}
    tp = L.trial_plan(L.trial("t-lean"))
    grips = {s_[4]: s_[3] for s_ in tp["right"]}
    assert grips["push to the mouth"] == "pitch:15" and grips["centre it at the mouth"] == "pitch:15"
    assert grips["push it home"] == "forward" and grips["carry to the stand-off"] == "forward"
    live = L.live_plan(L.trial("t-lean"), {"neutral": tops["neutral"]})
    assert live[2][4] == "lean the neutral loop toward the centre" and len(live) == 6


def test_the_low_approach_comes_in_under_the_lip_then_lifts():
    tp = L.trial_plan(L.trial("u-low"))
    by = {s_[4]: s_ for s_ in tp["right"]}
    z = float(L.OPEN_C[1])
    low, up = np.array(by["push in, still low"][1]), np.array(by["lift it to the opening"][1])
    assert up[2] - low[2] == pytest.approx(0.048) and up[0] == pytest.approx(low[0])          # straight up, 48 mm
    assert "carry to the stand-off" not in by and "lower it for the left hand" not in by
    labels = [s_[4] for s_ in tp["right"]]
    assert labels.index("lift it to the opening") < labels.index("centre it at the mouth") < labels.index("push it home")
    assert sum(s_[0] for s_ in tp["left"]) == pytest.approx(sum(s_[0] for s_ in tp["right"]))
    assert set(task.plan_waypoints(tp)) == {"left", "right"} and L.plan_seconds(tp) < 70
    assert not {"carry in low", "push in, still low", "lift it to the opening"} & set(L.AIM_LABELS)   # not steered while low


def test_the_firm_moves_aim_past_the_wire_and_keep_the_plans_timing():
    y, z = float(L.OPEN_C[0]), float(L.OPEN_C[1])
    top_n = np.array([L.WALL_X - 0.028, y + 0.023, z + 0.036])
    top_h = np.array([L.WALL_X - 0.030, y - 0.005, z + 0.038])
    t_ = L.trial("g-fold")
    plan = L.live_plan(t_, [top_n, top_h, top_n])
    assert [s_[4] for s_ in plan][2::5][:3] == ["lean the neutral loop toward the centre", "sweep the hot loop down and into the box",
                                                 "sweep the neutral loop down and into the box"]
    assert [s_[0] for s_ in plan] == [s_[0] for s_ in L.crest_plan(("neutral", "hot", "neutral"))]   # the right hand's timing holds
    ax = L.crest_rotation(0.0) @ np.array([0.0, 0.0, 1.0])
    tip = lambda k: np.array(plan[k][1]) + L.PAD_TIP * ax  # noqa: E731
    assert tip(2)[1] == pytest.approx(y + L.LEAN_TO + L.LEAN_HALF - 0.045)       # the lean, aimed 45 mm past
    soft = L.crest_plan_at({"hot": top_h}, ("hot",))
    end_soft = np.array(soft[2][1]) + L.PAD_TIP * ax
    assert np.linalg.norm(tip(7) - end_soft) == pytest.approx(0.040)             # the sweep, aimed 40 mm further on its line
    assert np.allclose(np.cross(tip(7) - tip(6), end_soft - tip(6)), 0.0, atol=1e-9)
    for name in ("g-fold", "g-fold-70", "g-run"):
        tp = L.trial_plan(L.trial(name))
        assert sum(s_[0] for s_ in tp["left"]) == pytest.approx(sum(s_[0] for s_ in tp["right"]))
        assert set(task.plan_waypoints(tp)) == {"left", "right"} and L.plan_seconds(tp) < 70
    low = {s_[4]: s_ for s_ in L.trial_plan(L.trial("g-fold"))["right"]}["lower it for the left hand"]
    std = {s_[4]: s_ for s_ in L.trial_plan(L.trial("s-press"))["right"]}["lower it for the left hand"]
    assert std[1][2] - low[1][2] == pytest.approx(0.055 - 0.028)


def test_an_along_sweep_runs_on_the_fingers_own_axis_through_the_loops_top():
    y, z = float(L.OPEN_C[0]), float(L.OPEN_C[1])
    top = np.array([L.WALL_X - 0.029, y - 0.008, z + 0.022])
    ax = L.crest_rotation(0.0) @ np.array([0.0, 0.0, 1.0])
    plan = L.crest_plan_at({"hot": top}, ("hot",), over=0.040, along=True)
    behind, end = (np.array(plan[k][1]) + L.PAD_TIP * ax for k in (1, 2))
    assert np.allclose(behind, top - 0.015 * ax)                                   # 15 mm short of the top, on the axis
    assert np.allclose(np.cross(end - behind, ax), 0.0, atol=1e-9) and (end - behind) @ ax > 0   # and straight on along it
    assert end[0] == pytest.approx(L.WALL_X + 0.012 + 0.040 * ax[0])              # 12 mm inside, then 40 mm past
    high = np.array([L.WALL_X - 0.023, y - 0.005, z + 0.056])                      # a loop above the lip: the old line
    old = L.crest_plan_at({"hot": high}, ("hot",), over=0.040, along=True)
    assert old == L.crest_plan_at({"hot": high}, ("hot",), over=0.040)
    for name in ("h-fold", "h-run"):
        tp = L.trial_plan(L.trial(name))
        assert sum(s_[0] for s_ in tp["left"]) == pytest.approx(sum(s_[0] for s_ in tp["right"]))
        assert set(task.plan_waypoints(tp)) == {"left", "right"} and L.plan_seconds(tp) < 70


def test_a_wires_end_at_its_terminal_is_judged_with_the_outlet_not_as_loose_wire():
    y, z = float(L.OPEN_C[0]), float(L.OPEN_C[1])
    inside = [[L.WALL_X + 0.040, y, z + 0.030], [L.WALL_X + 0.020, y, z], [L.WALL_X + 0.002, y + 0.015, z - 0.020]]
    end_proud = inside + [[L.WALL_X - 0.0025, y + 0.020, z - 0.030]]        # only the terminal end 2.5 mm in front
    assert L.wires_outside({"ground": end_proud}) == []
    link_proud = inside[:2] + [[L.WALL_X - 0.0025, y + 0.015, z - 0.020], [L.WALL_X - 0.006, y + 0.020, z - 0.030]]
    assert L.wires_outside({"ground": link_proud}) == ["ground"]              # the point before it is judged as before
    off_outline = inside + [[L.WALL_X - 0.0025, y + L.BOX_W / 2 + 0.004, z]]  # an end outside the opening's outline still fails
    assert L.wires_outside({"ground": off_outline}) == ["ground"]


def test_the_demo_hands_the_policy_exactly_the_target_it_works_out_each_step():
    from yam_arena.demo import ArmPlan
    from yam_arena.station import BASES

    way = task.demo()
    assert set(way) == {"left", "right"} and task._DEMO["tr"].name == task.DEMO_TRIAL
    lengths = {arm: sum(w.duration for w in ws) for arm, ws in way.items()}
    assert lengths["left"] == pytest.approx(lengths["right"]) and (lengths["right"] + 3) * 20 <= 0.75 * 2400
    policy = {arm: ArmPlan(list(ws), BASES[arm]) for arm, ws in way.items()}   # what yam_scripted builds from demo()
    rng = np.random.default_rng(0)
    for step in range(2, int(round(lengths["right"] * 20)) + 40):              # every control step, and past the end
        t = step / 20
        for arm in ("left", "right"):
            pos, rot, finger, _ = task._DEMO["plan"][arm].target(t)            # the plan as written ...
            first = t < way[arm][0].duration
            pos = np.asarray(pos, float) + (0.0 if first else rng.uniform(-0.02, 0.02, 3))   # ... plus a correction
            task._play(arm, t, pos, rot, finger)
            got = policy[arm].target(t)
            assert np.allclose(got[0], pos, atol=1e-9) and np.allclose(got[1], rot, atol=1e-9)
            assert got[2] == pytest.approx(float(finger))
    task._demo_control(None)                                                    # offline: no simulator, nothing to do
    assert task._DEMO["error"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", "-rA", __file__]))
