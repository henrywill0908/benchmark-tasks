"""A short piece of off-white wall stands on the table in front of the robot, a little to its right. Set into it,
flush with its face, is an empty blue single-gang electrical box with a threaded screw boss at its top and bottom. An
ivory wall outlet on a steel mounting strap hangs out of the box on its wires, about 11 cm in front of the wall: a
black wire to its right side, a white wire to its left side and a bare copper wire to the left side, lower down. The
wires are stiff solid copper, about 20 cm long, still kinked where they were folded, and come out of the top back of the
box.
"""

# yam_arena hands the docstring above to the policy as task documentation, so it describes only what is on the table.
#
# The numbers (sizes, the wires' start shape, the scoring limits and the scripted demo's moves) are in outlet_logic.py;
# copper.py is how a 12 AWG copper conductor bends; labkit.py builds the outlet and its wires as one articulation.
# The scripted demo is closed loop (it reads where the wires and the outlet are as it goes): scripted_demo.py runs it.
# Development switches, left at their defaults for the task:
#   OUTLET_STYLE=d6|alternate   joint style (two-axis joints, or alternating revolute joints)
#   OUTLET_ARMATURE=5e-5        per-joint armature (kg m^2): larger is steadier, and slower to move
#   OUTLET_STIFFNESS_SCALE=1.0  multiplies copper's joint stiffness
#   OUTLET_HARD=1.0             scales copper's work hardening (0: none)

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # labkit and outlet_logic sit next to this file

import numpy as np  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

import copper  # noqa: E402
import outlet_logic as L  # noqa: E402
from labkit import Harness, HarnessPart, HarnessWire, Pose, forward_grip, on_reset, on_step, ready_pose, state  # noqa: E402

SCENE = "station"
assert abs(TABLE_TOP[SCENE] - L.TOP) < 1e-9, "outlet_logic.TOP must match the station's table top"
CONTROL_HZ = L.CONTROL_HZ
SUCCESS_HOLD_S = 1.0
INSTRUCTION = "Put the outlet back in its wall box, ready for its screws, with every wire inside the box. Then let go."
PROBE = os.environ.get("OUTLET_PROBE", "straight")
LOG = bool(os.environ.get("OUTLET_LOG"))  # print the measurements behind the score twice a second

DRYWALL, BLUE, IVORY, STEEL = (0.90, 0.90, 0.87), (0.15, 0.33, 0.72), (0.93, 0.91, 0.84), (0.66, 0.67, 0.70)

# -- the wall, with a hole for the box -------------------------------------------------------------------------------------
_y, _z = L.OPEN_C
_hy, _hz = L.BOX_W / 2 + L.BOX_WALL, L.BOX_H / 2 + L.BOX_WALL  # the hole fits the box's outside
_wx = L.WALL_X + L.WALL_T / 2
_lo, _hi = L.TOP, L.TOP + L.WALL_H


def _wall(name, y0, y1, z0, z1) -> Box:
    return Box(name, pos=(_wx, (y0 + y1) / 2, (z0 + z1) / 2), size=(L.WALL_T, y1 - y0, z1 - z0), color=DRYWALL)


WALL = [
    _wall("wall_right", L.WALL_Y[0], _y - _hy, _lo, _hi),
    _wall("wall_left", _y + _hy, L.WALL_Y[1], _lo, _hi),
    _wall("wall_below", _y - _hy, _y + _hy, _lo, _z - _hz),
    _wall("wall_above", _y - _hy, _y + _hy, _z + _hz, _hi),
]

# -- the box: five walls and two mounting bosses, its rim flush with the wall face ----------------------------------------
_bx = L.WALL_X + L.BOX_D / 2
_t = L.BOX_WALL
BOX = [
    Box("box_back", pos=(L.WALL_X + L.BOX_D + _t / 2, _y, _z), size=(_t, 2 * _hy, 2 * _hz), color=BLUE),
    Box("box_top", pos=(_bx, _y, _z + L.BOX_H / 2 + _t / 2), size=(L.BOX_D, 2 * _hy, _t), color=BLUE),
    Box("box_bottom", pos=(_bx, _y, _z - L.BOX_H / 2 - _t / 2), size=(L.BOX_D, 2 * _hy, _t), color=BLUE),
    Box("box_right", pos=(_bx, _y - L.BOX_W / 2 - _t / 2, _z), size=(L.BOX_D, _t, L.BOX_H), color=BLUE),
    Box("box_left", pos=(_bx, _y + L.BOX_W / 2 + _t / 2, _z), size=(L.BOX_D, _t, L.BOX_H), color=BLUE),
    Box("boss_top", pos=(_bx, _y, _z + L.BOX_H / 2 - L.BOSS_P / 2), size=(L.BOX_D, L.BOSS_W, L.BOSS_P), color=BLUE),
    Box("boss_bottom", pos=(_bx, _y, _z - L.BOX_H / 2 + L.BOSS_P / 2), size=(L.BOX_D, L.BOSS_W, L.BOSS_P),
        color=BLUE),
]

# -- the outlet on its wires (labkit Harness) ------------------------------------------------------------------------------
_ear_z = L.STRAP_MID_H / 2 + L.EAR_H / 2
PARTS = (
    HarnessPart("strap", (L.STRAP_T, L.BODY[1], L.STRAP_MID_H), (-L.STRAP_T / 2, 0.0, 0.0), STEEL),
    HarnessPart("ear_top", (L.STRAP_T, L.STRAP_W, L.EAR_H), (-L.STRAP_T / 2, 0.0, _ear_z), STEEL),
    HarnessPart("ear_bottom", (L.STRAP_T, L.STRAP_W, L.EAR_H), (-L.STRAP_T / 2, 0.0, -_ear_z), STEEL),
    HarnessPart("body", L.BODY, (L.BODY[0] / 2, 0.0, 0.0), IVORY),
    HarnessPart("face", L.FACE, (-L.STRAP_T - L.FACE[0] / 2, 0.0, 0.0), IVORY),
)


def _link_mass(w: L.WireRun) -> float:
    copper = np.pi * (L.D12 / 2) ** 2 * L.SEGMENT * L.COPPER_DENSITY
    insulation = np.pi * (w.radius**2 - (L.D12 / 2) ** 2) * L.SEGMENT * 1400.0  # PVC/nylon
    return float(copper + max(0.0, insulation))


_f = lambda key, default: float(os.environ.get(key, default))  # noqa: E731


def _torque_cap() -> float | None:
    v = os.environ.get("OUTLET_TORQUE_CAP", "").strip()
    return None if not v else L.yield_moment() if v == "yield" else float(v)


GROUND = os.environ.get("OUTLET_GROUND", "side15")  # the candidate task's build (outlet_logic.BUILDS)


HARD = _f("OUTLET_HARD", 1.0)  # scales copper's work hardening (1: handbook annealed copper; 0: none)


def build_harness(ground: str = GROUND) -> Harness:
    kp, flow = copper.flow_table(HARD)
    scale = _f("OUTLET_STIFFNESS_SCALE", 1.0)
    return Harness(
        "outlet", pos=tuple(L.ANCHOR), size=(0.01, 0.01, 0.01), color=IVORY,
        wires=tuple(HarnessWire(w.name, tuple(np.asarray(w.root) - L.ANCHOR), w.terminal, L.segments(w), w.radius,
                                w.color, _link_mass(w),
                                points=tuple(tuple(float(v) for v in p) for p in L.start_polyline(w) - L.ANCHOR))
                    for w in L.wires(ground)),  # pre-bent: each wire's relaxed S (outlet_logic.start_polyline)
        parts=PARTS, device_at=tuple(L.DEVICE_START - L.ANCHOR), device_mass=L.DEVICE_MASS,
        stiffness=L.joint_stiffness(L.SEGMENT) * scale,
        damping=0.03, armature=_f("OUTLET_ARMATURE", 5e-5),
        yield_angle=L.yield_angle(L.SEGMENT) * _f("OUTLET_YIELD_SCALE", 1.0),
        max_bend=L.MAX_BEND, style=os.environ.get("OUTLET_STYLE", "d6"), torque_cap=_torque_cap(),
        # Round 13, the copper rule (copper.py): a half link clamped at each end, a joint per SEGMENT of wire, yield at
        # copper's flow moment (59 N mm at first, about 100 N mm once fully plastic, more as it work-hardens), and the
        # drive torque capped at that moment each control step. Rounds 1-12 yielded at 59 N mm for ever, and a careful
        # push then seated the outlet with no wire handling.
        segment=L.SEGMENT, flow_kp=tuple(float(v) for v in kp), flow_m=tuple(float(v) * scale for v in flow),
        rigid_root=True, end_filter=L.END_FILTER,
        hardened0=tuple(float(v) for w in L.wires(ground) for v in L.start_hardening(w)),  # the old folds start stiff
    )


HARNESS = build_harness()

OBJECTS = [*WALL, *BOX, HARNESS]


# -- scoring ---------------------------------------------------------------------------------------------------------------
# score() sees object states only, so an on_step hook records each step how close the nearest gripper point (TCP and
# pad tips, both arms) comes to the outlet's outline: credit needs the outlet let go, 3 cm clear (outlet_logic.CLEAR).
SEEN = {"hand_clear": float("inf"), "clear_s": 0.0, "error": None}


@on_reset
def _new_trial() -> None:
    SEEN["hand_clear"], SEEN["clear_s"], SEEN["error"] = float("inf"), 0.0, None


def _gripper_points(environment) -> list:
    from labkit import _np
    from yam_arena.runtime import pad_tips_world, tcp_world

    base = environment.unwrapped
    origin = _np(base.scene.env_origins)[0]
    out = []
    for key in ("left_robot", "right_robot"):
        robot = base.scene[key]
        out += [np.asarray(tcp_world(robot), float) - origin,
                *np.atleast_2d(np.asarray(pad_tips_world(robot), float) - origin)]
    return out


@on_step
def _step(environment) -> None:
    if environment is None:
        return
    try:
        s = state(environment, "outlet")
        pts = _gripper_points(environment)
        # clear of the outlet AND of every wire: a gripper pressing the wires into the box has not let go
        SEEN["hand_clear"] = min(L.hand_clearance(pts, s.pos, s.quat_xyzw), L.wire_clearance(pts, s.polylines))
        still = float(np.linalg.norm(s.lin_vel)) <= L.STILL_SPEED
        # how long it has been let go and still, without a break (one control step at a time)
        SEEN["clear_s"] = SEEN["clear_s"] + 1.0 / CONTROL_HZ if SEEN["hand_clear"] >= L.CLEAR and still else 0.0
    except Exception as err:  # noqa: BLE001 - never credit an outlet whose grippers couldn't be checked
        SEEN["hand_clear"], SEEN["clear_s"], SEEN["error"] = 0.0, 0.0, f"{type(err).__name__}: {err}"
    if LOG:  # OUTLET_LOG=1: the measurements behind the score, twice a second
        try:
            from labkit import trial_time

            n = int(round(trial_time() * CONTROL_HZ))
            if n % 10 == 0:
                s = state(environment, "outlet")
                front = {k: round(1000 * float(L.WALL_X - np.min(np.asarray(v)[:-1, 0])), 1) for k, v in s.polylines.items()}
                print("OUTLET " + json.dumps({
                    "t": round(n / CONTROL_HZ, 2), "device_mm": [round(1000 * float(v), 1) for v in s.pos],
                    "strap_gap_mm": [round(1000 * float(v), 1) for v in L.strap_gaps(s.pos, s.quat_xyzw)],
                    "tab_off_mm": [round(1000 * float(v), 1) for v in L.tab_offsets(s.pos, s.quat_xyzw)],
                    "loose_wire_front_mm": front, "wires_outside": L.wires_outside(s.polylines),
                    "hand_clear_mm": round(1000 * min(float(SEEN["hand_clear"]), 9.999), 1), "clear_s": round(SEEN["clear_s"], 2),
                    "score": score({"outlet": s}), "demo_error": _DEMO.get("error") if _DEMO else None,
                    "check_error": SEEN["error"]}), flush=True)
        except Exception as err:  # noqa: BLE001 - logging must never stop a trial
            print(f"OUTLET log error: {type(err).__name__}: {err}", flush=True)


def score(objects) -> float:
    """Ready for its screws. 0 unless the outlet stands upright (within 15 degrees), every strap corner is within 15 mm
    of the wall and every wire is inside the box: none more than 2 mm in front of the wall face (each wire's end at
    its terminal aside: that is part of the outlet), none looped over the box's edge or trapped under the strap. Then 1.0 only if both screw tabs are within 3 mm of the box's screw holes and
    it has been let go: both grippers at least 3 cm from the outlet and from every wire, and the outlet at rest (under
    1 cm/s), for the last 2 s. Let go and in place but off its holes: 0.5. Still held, pressed or moving: 0."""
    s = objects.get("outlet")
    if s is None:
        return 0.0
    speed = float(np.linalg.norm(s.lin_vel)) if getattr(s, "lin_vel", None) is not None else 0.0
    return L.task_score(s.pos, s.quat_xyzw, s.polylines, SEEN["hand_clear"], speed, SEEN["clear_s"])


def success(objects) -> bool:
    return score(objects) >= 1.0


# -- the probe's scripted moves ----------------------------------------------------------------------------------------------
def _waypoint(arm: str, step):
    seconds, pos, fingers, grip, label = step
    # "grip": part-closed on the outlet's body (outlet_logic.GRIP_FINGER), so the squeeze stays under the fingers'
    # 50 N limit and they hold it firmly sideways (round 3: fully closed, both fingers sat at the limit).
    finger = {"open": FINGER_OPEN, "closed": FINGER_CLOSED, "grip": L.GRIP_FINGER, "grip9": L.GRIP_FIRM}[fingers]
    if grip == "ready":
        w = ready_pose(arm, duration=seconds, label=label)
        w.finger = finger
        return w
    rot = forward_grip(0.0)
    if grip.startswith("pitch:"):  # a forward grip tipped about world y (outlet_logic.roty)
        rot = L.roty(float(grip.split(":", 1)[1])) @ rot
    elif grip.startswith("yaw:"):  # a forward grip turned about the vertical (the left hand's poke)
        rot = forward_grip(float(grip.split(":", 1)[1]))
    elif grip.startswith("crest:"):  # round 12: the left hand's press, fingertips forward and 30 degrees down
        rot = L.crest_rotation(float(grip.split(":", 1)[1]))
    return Pose(seconds, pos, finger, label, rot=rot)


def plan_waypoints(plan: dict) -> dict:
    """A plan in outlet_logic's step format (probe_plan, trial_plan) as yam_arena waypoints."""
    return {arm: [_waypoint(arm, s) for s in steps] for arm, steps in plan.items()}


# -- the scripted demo -----------------------------------------------------------------------------------------------------
# Closed loop: it reads where the wires and the outlet are as it goes (a scripted demo may read the scene; it never moves
# an object or edits the simulation). The benchmark's yam_scripted policy plays demo()'s waypoints open loop, so the loop
# is closed here: demo() hands the policy waypoint objects, and after every control step _demo_control() works out
# where each hand should be on the next step and writes that into the two waypoints the policy is interpolating between.
# The moves and the control rules are the ones scripted_demo.py ran to build the demo (outlet_logic's trial DEMO_TRIAL).
DEMO_TRIAL = "j-run"
_DEMO: dict = {}


def demo():
    """The scripted demo's plan, per arm, for the yam_scripted policy (see _demo_control)."""
    import copy

    from yam_arena.demo import ArmPlan
    from yam_arena.station import BASES

    tr = L.trial(DEMO_TRIAL)
    steps = L.trial_plan(tr)
    way = plan_waypoints(steps)
    _DEMO.clear()
    _DEMO.update(
        tr=tr, steps=steps, way=way,
        plan={arm: ArmPlan(copy.deepcopy(ws), BASES[arm]) for arm, ws in way.items()},  # the plan as written
        g={"twist": 0.0, "x": None, "frozen_s": 0.0, "started": False}, corr=np.zeros(3), corr_l=np.zeros(3),
        face_snap=None, error=None)
    return way


def _play(arm: str, t: float, pos, rot, finger) -> None:
    """Make the policy's plan for ``arm`` give exactly (pos, rot, finger) at time ``t``: both waypoints it is
    interpolating between are set to that target. (During a plan's first step there is no earlier waypoint, and
    nothing is corrected then.)"""
    acc, ws = 0.0, _DEMO["way"][arm]
    for i, w in enumerate(ws):
        if t < acc + w.duration:
            if i > 0:
                for v in (ws[i - 1], w):
                    v.pos, v.rot, v.finger = tuple(float(c) for c in pos), np.asarray(rot, float), float(finger)
            return
        acc += w.duration
    if ws:  # past the end: the plan holds its last waypoint
        ws[-1].pos, ws[-1].rot, ws[-1].finger = tuple(float(c) for c in pos), np.asarray(rot, float), float(finger)


@on_step
def _demo_control(environment) -> None:
    if environment is None or not _DEMO:
        return
    d = _DEMO
    try:
        import dataclasses

        from labkit import _np, rolled, trial_time
        from yam_arena.runtime import tcp_world

        tr, g, corr, corr_l, steps_plan = d["tr"], d["g"], d["corr"], d["corr_l"], d["steps"]
        step = int(round(trial_time() * CONTROL_HZ)) + 1  # the control step the policy is about to command
        t = step / CONTROL_HZ
        right, left = d["plan"]["right"], d["plan"]["left"]
        fast = getattr(tr, "steer", "slow") == "fast"
        every = L.FAST_EVERY if fast else 10
        # -- the right hand: keep the held outlet centred on the opening, and hold the push back until it is
        planned = right.target(t)
        label = planned[3]
        st = state(environment, "outlet")
        dev = np.asarray(st.pos, float)
        roll = L.roll_deg(st.quat_xyzw)
        if getattr(tr, "steer", "slow") != "none" and step % every == 0:
            if label in L.AIM_LABELS:
                gain, cy, cz = (L.FAST_GAIN, L.FAST_Y, L.FAST_Z) if fast else (L.AIM_GAIN, L.AIM_Y, L.AIM_Z)
                corr[1] = np.clip(corr[1] + gain * (L.OPEN_C[0] - dev[1]), -cy, cy)
                corr[2] = np.clip(corr[2] + gain * (L.OPEN_C[1] - dev[2]), *cz)
                if fast:
                    g["twist"] = float(np.clip(g["twist"] - L.TWIST_GAIN * roll, -L.TWIST_CAP, L.TWIST_CAP))
            elif label in L.FACE_LABELS:
                if d["face_snap"] is None:
                    d["face_snap"] = dev.copy()
                corr[1] = np.clip(d["face_snap"][1] - L.OPEN_C[0], -L.AIM_Y, L.AIM_Y)
                corr[2] = np.clip(d["face_snap"][2] - L.OPEN_C[1], *L.AIM_Z)
        if getattr(tr, "gate", False):  # every step: the push's x, held back while the outlet is off
            plan_x = float(planned[0][0])
            if label in L.GATED_LABELS:
                g["started"] = True
            if g["started"] and label in L.AIM_LABELS:
                off = max(abs(dev[1] - L.OPEN_C[0]), abs(dev[2] - L.OPEN_C[1]))
                aligned = off <= L.GATE_MM / 1000 and abs(roll) <= L.GATE_DEG
                if g["x"] is None:
                    g["x"] = plan_x
                if label in L.GATED_LABELS and not aligned and g["frozen_s"] < L.GATE_MAX_S:
                    g["frozen_s"] += 1.0 / CONTROL_HZ
                else:  # catch up with the plan, no faster than the gate's speed
                    g["x"] = min(plan_x, g["x"] + L.GATE_SPEED / CONTROL_HZ) if plan_x > g["x"] else plan_x
            elif g["started"]:
                g["x"] = None  # the gripped push is over: the plan's own x again
        # -- the left hand: aim each wire move at where that wire's loop is, just before the move starts
        aimed = g.get("aimed", 0)
        if getattr(tr, "live", False) and tr.crest and aimed < len(tr.crest) and t >= (
                steps_plan["left"][0][0] + L.crest_start(aimed) - (0.3 if aimed == 0 else 0.05)):
            tops = L.loop_tops(st.polylines, tuple(set(tr.crest)))
            aims = g.setdefault("aims", [None] * len(tr.crest))
            for k_ in range(aimed, len(tr.crest)):
                aims[k_] = tops[tr.crest[k_]]
            whole = L.trial_plan(dataclasses.replace(tr, pieces=0)) if getattr(tr, "pieces", 0) else steps_plan
            left_new = [whole["left"][0]] + L.live_plan(tr, aims)
            rest_s = L.plan_seconds(whole) - sum(s_[0] for s_ in left_new)
            if rest_s > 1e-9:
                left_new.append((rest_s, None, "open", "ready", "wait"))
            new_way = plan_waypoints({"left": left_new})["left"]
            if [w.duration for w in new_way] != [w.duration for w in d["way"]["left"]]:
                raise ValueError("the re-aimed left plan's step times differ from the plan the policy holds")
            left.waypoints = new_way
            g["aimed"] = aimed + 1
        # -- the left hand: follow the wire sideways during a sweep; a lean stops once the loop is near the centre
        if getattr(tr, "track", False):
            lab_l = str(left.target(t)[3])
            if lab_l.startswith(("sweep the ", "lean the ")):
                base_ = environment.unwrapped
                tcp_l = np.asarray(tcp_world(base_.scene["left_robot"]), float) - _np(base_.scene.env_origins)[0]
                tip_l = tcp_l + L.PAD_TIP * (L.crest_rotation(0.0) @ np.array([0.0, 0.0, 1.0]))
                wire_l = np.asarray(st.polylines[lab_l.split()[2]], float)
                if lab_l.startswith("sweep the "):  # keep the fingertips on the wire's own y
                    near = wire_l[np.hypot(wire_l[:, 0] - tip_l[0], wire_l[:, 2] - tip_l[2]) < 0.025]
                    if len(near):
                        corr_l[1] = float(np.clip(corr_l[1] + 0.3 * (float(np.mean(near[:, 1])) - tip_l[1]), -0.03, 0.03))
                elif g.get("lean_hold") is None:
                    top_l = L.loop_tops({"w": wire_l}, ("w",))["w"]
                    if top_l is None or abs(float(top_l[1]) - float(L.OPEN_C[0])) <= L.LEAN_TO + 0.002:
                        g["lean_hold"] = float(tcp_l[1])
            elif lab_l != "hold it there":
                corr_l[:] = 0.0
                g["lean_hold"] = None
        # -- hand the policy the two targets for the step it is about to command
        r_pos, r_rot, r_finger, _ = right.target(t)
        r_pos = np.asarray(r_pos, float) + corr
        if g["x"] is not None:
            r_pos[0] = g["x"]
        if g["twist"]:
            r_rot = rolled(np.asarray(r_rot, float), g["twist"])
        _play("right", t, r_pos, r_rot, r_finger)
        l_pos, l_rot, l_finger, _ = left.target(t)
        l_pos = np.asarray(l_pos, float) + (corr_l if getattr(tr, "track", False) else 0.0)
        if g.get("lean_hold") is not None:
            l_pos[1] = g["lean_hold"]
        _play("left", t, l_pos, l_rot, l_finger)
    except Exception as err:  # noqa: BLE001 - a demo fault must not take the trial down with it; score() will show it
        d["error"] = f"{type(err).__name__}: {err}"


def probe_demo():
    """An early open-loop development variant (OUTLET_PROBE: hang, straight, bend, ...), which scripted_demo.py's
    single-variant mode plays. It is NOT the task's demo and does not reach full score: the demo is closed loop
    (demo() above, with _demo_control)."""
    return {arm: [_waypoint(arm, s) for s in steps] for arm, steps in L.probe_plan(PROBE).items()}
