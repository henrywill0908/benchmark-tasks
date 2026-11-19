"""A steel shaft, 20 mm across, runs from front to back over the bench about 30 cm in front of the robot, 45 mm above
the bench top, clamped at its far end in a blue support block, with a 30 mm shoulder collar near that end. A 25 mm keyseat
is cut in its top just in front of the shoulder. A 60 mm aluminium pulley (a 12 mm rim in front, a 36 mm boss behind)
sits loose on the shaft's front half, turned so its keyway and the set screw in its boss face up and to the left. A
steel key, 6 x 6 x 25 mm, lies on a green mat to the right of the shaft. A red T-handle driver with a 2.5 mm hex bit
stands in a grey cup to the left, its handle across the top of the cup.
"""

# yam_arena hands the docstring above to the policy as task documentation, so it describes only what is on the table.
#
# Task code (pulley_logic.Tracker, every control step) stands in for what PhysX can't do: the key's press fit in its
# keyseat, the hub's keyway sliding over the key (only lined up, only with the key home), the set screw's thread and
# the driver's clutch. pulley_logic.py has every number. Settings:
#   DEMO_VARIANT=full|no_key|proud|no_turn|leave_driver|turn_err3|turn_err6   which scripted run demo() plays
#   PULLEY_DEBUG=1   prints the tracker's state (one JSON line) every half second

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # labkit and pulley_logic sit next to this file

import numpy as np  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

import pulley_logic as L  # noqa: E402
from labkit import (Container, KeyedPulley, Part, Pose, Slider, Tool, arc_yz, hold_joint, on_reset,  # noqa: E402
                    on_step, ready_pose, release_joint, state, write_body_pose, x_cylinder_mesh, x_prism_mesh)
from labkit import cylinder_mesh  # noqa: E402

SCENE = "station"
assert abs(TABLE_TOP[SCENE] - L.TOP) < 1e-9, "pulley_logic.TOP must match the station's bench top"
CONTROL_HZ = L.CONTROL_HZ
SUCCESS_HOLD_S = 1.0
INSTRUCTION = "Fit the pulley to the shaft and lock it with its set screw."
VARIANT = os.environ.get("DEMO_VARIANT", "full")
DEBUG = bool(os.environ.get("PULLEY_DEBUG"))

STEEL, DARK_STEEL, ALU, ALU_DARK = (0.62, 0.63, 0.66), (0.42, 0.43, 0.46), (0.80, 0.81, 0.83), (0.66, 0.67, 0.70)
BLUE, GREEN, RED, BLACK, GREY = (0.16, 0.30, 0.62), (0.20, 0.55, 0.30), (0.75, 0.10, 0.10), (0.06, 0.06, 0.07), \
    (0.55, 0.56, 0.58)
MM = L.MM


def _shaft_parts() -> tuple:
    """The shaft in the pulley's frame (origin on the axis at the pulley's starting rear face): round where it is round,
    and in the keyseat's length three convex pieces that leave the 6.4 mm slot (its floor 1 mm under the drawing's, for
    the press-fit pad that stands in it; 1.5 mm longer at the free end and its top edges broken L.LEAD_IN: the
    simulator's allowances, pulley_logic.SEAT_X_SIM)."""
    o = L.REAR0
    x_end, x_s0, x_s1, x_far = L.SHAFT_END - o, L.SEAT_X_SIM[0] - o, L.SEAT_X_SIM[1] - o, L.SHAFT_FAR - o
    r, w, floor = L.SHAFT_R, L.SEAT_W_SIM / 2, L.SEAT_FLOOR_R - 1.0 * MM
    a_side = math.atan2(floor, math.sqrt(r * r - floor * floor))  # where the floor's line meets the round
    c = L.LEAD_IN
    a_edge = math.atan2(math.sqrt(r * r - (w + c) ** 2), w + c)  # where the broken edge meets the round
    low = arc_yz(r, math.pi - a_side, 2 * math.pi + a_side, 40)  # the round below the floor's line
    wall_top = math.sqrt(r * r - w * w) - c  # the slot's wall stops c under the round, then the lead-in
    side_pos_y = [(w, floor), (math.sqrt(r * r - floor * floor), floor), *arc_yz(r, a_side, a_edge, 8)[1:],
                  (w, wall_top)]
    side_neg_y = [(-y, z) for y, z in reversed(side_pos_y)]
    return (
        Part("shaft_front", x_cylinder_mesh(r, x_end, x_s0), color=STEEL, collides=True, roughness=0.3),
        Part("seat_low", x_prism_mesh(low, x_s0, x_s1), color=STEEL, collides=True, roughness=0.3),
        Part("seat_left", x_prism_mesh(side_pos_y, x_s0, x_s1), color=STEEL, collides=True, roughness=0.3),
        Part("seat_right", x_prism_mesh(side_neg_y, x_s0, x_s1), color=STEEL, collides=True, roughness=0.3),
        Part("shaft_back", x_cylinder_mesh(r, x_s1, x_far), color=STEEL, collides=True, roughness=0.3),
        Part("collar", x_cylinder_mesh(L.COLLAR_R, L.COLLAR_X[0] - o, L.COLLAR_X[1] - o), color=STEEL,
             collides=True, roughness=0.3),
    )


def _support_boxes() -> tuple:
    """The shaft support block and its foot, in the pulley's frame."""
    o = L.REAR0
    x0, x1 = L.SUPPORT_X
    h = L.AX_Z + 0.02 - L.TOP
    return (("support_block", (x1 - x0, 0.06, h), ((x0 + x1) / 2 - o, 0.0, L.TOP + h / 2 - L.AX_Z), BLUE),
            ("foot", (0.07, 0.10, 0.008), ((x0 + x1) / 2 - o, 0.0, L.TOP + 0.004 - L.AX_Z), BLUE))


def _wheel_parts() -> tuple:
    """The pulley in its own frame (origin on the axis at its rear face, x toward the shoulder): the rim in front, the
    boss behind, the keyway's mouth on the front face (dark) and the set screw's head on the boss (black)."""
    rim0, rim1 = -L.HUB_L, -L.HUB_L + L.RIM_W
    roof = L.KEYWAY_ROOF_R
    keyway = [(-L.SEAT_W / 2, L.BORE_R - 0.5 * MM), (L.SEAT_W / 2, L.BORE_R - 0.5 * MM), (L.SEAT_W / 2, roof),
              (-L.SEAT_W / 2, roof)]
    bore = arc_yz(L.BORE_R + 0.6 * MM, 0.0, 2 * math.pi, 33)[:-1]
    return (
        Part("rim", x_cylinder_mesh(L.RIM_R, rim0, rim1, 48), color=ALU, collides=True, roughness=0.35),
        Part("boss", x_cylinder_mesh(L.BOSS_R, rim1, 0.0, 40), color=ALU_DARK, collides=True, roughness=0.35),
        Part("bore_face", x_prism_mesh(bore, rim0 - 0.4 * MM, rim0 - 0.1 * MM), color=DARK_STEEL),
        Part("keyway_face", x_prism_mesh(keyway, rim0 - 0.6 * MM, rim0 - 0.2 * MM), color=BLACK),
        Part("screw", cylinder_mesh(2.5 * MM, 0.8 * MM, 12), at=(-L.SCREW_FROM_REAR, 0.0, L.BOSS_R + 0.1 * MM),
             color=BLACK),
    )


PULLEY = KeyedPulley(
    "pulley", pos=(L.REAR0, L.AX_Y, L.AX_Z), size=(L.SHAFT_FAR - L.SHAFT_END, 0.06, 0.06), color=STEEL,
    base_parts=_shaft_parts(), base_boxes=_support_boxes(), wheel_parts=_wheel_parts(), wheel_mass=L.PULLEY_MASS,
    slide_limits=L.SLIDE_LIMITS, turn_start=math.radians(L.PHI0), slide_damping=L.SLIDE_DAMPING,
    turn_damping=L.TURN_DAMPING)

# The press fit: a hidden pad in the keyseat that a key rests on 1.5 mm proud until pressed (task code holds the pad
# until it gives PAD_GIVE, then lets it drop to the floor).
PAD_TOP0 = L.AX_Z + L.SEAT_FLOOR_R + L.PAD_RISE
PAD = Slider(
    "pad", pos=(L.SEAT_CX_SIM, L.AX_Y, L.AX_Z - 4.0 * MM), size=(0.004, 0.002, 0.001), color=STEEL,
    panel=(L.SEAT_X_SIM[1] - L.SEAT_X_SIM[0] - 0.4 * MM, L.SEAT_W_SIM - 0.4 * MM, 1.0 * MM), panel_mass=0.005,
    panel_at=(0.0, 0.0, PAD_TOP0 - 0.5 * MM - (L.AX_Z - 4.0 * MM)), axis="z", limits=(-L.PAD_RISE, 0.0), start=0.0,
    damping=2.0, balanced=True, panel_visible=False, frame_collides=False, no_collide_with=("pulley/wheel",))

MAT_BOX = Box("key_mat", pos=(L.KEY_MAT[0], L.KEY_MAT[1], L.TOP + L.MAT[2] / 2), size=L.MAT, color=GREEN)
KEY = Tool("key", pos=L.KEY_START, size=tuple(L.KEY_SIM), color=DARK_STEEL, mass=L.KEY_MASS,
           no_collide_with=("pulley/wheel",))
CUP = Container("driver_stand", pos=(L.HOLDER[0], L.HOLDER[1], L.TOP),
                size=(L.CUP_IN + 2 * L.CUP_WALL, L.CUP_IN + 2 * L.CUP_WALL, L.CUP_H), wall=L.CUP_WALL, color=GREY)
DRIVER = Tool("driver", pos=L.BAR_START, size=tuple(L.BAR), color=RED, mass=L.DRIVER_MASS,
              boxes=(("shank", tuple(L.SHANK), (0.0, 0.0, -(L.BAR[2] + L.SHANK[2]) / 2), DARK_STEEL),),
              no_collide_with=("pulley/wheel",))

OBJECTS = [PULLEY, PAD, MAT_BOX, KEY, CUP, DRIVER]

# -- task code, every control step -----------------------------------------------------------------------------------
TRACK = {"t": L.Tracker()}


@on_reset
def _new_trial() -> None:
    TRACK["t"] = L.Tracker()


def _robot_points(environment) -> tuple[list, list]:
    """Every gripper point (TCP and pad tips, both arms) and each arm's (TCP, finger gap), task frame."""
    from labkit import _np
    from yam_arena.runtime import pad_tips_world, tcp_world

    base = environment.unwrapped
    origin = _np(base.scene.env_origins)[0]
    pts, tcps = [], []
    for key in ("left_robot", "right_robot"):
        robot = base.scene[key]
        tcp = np.asarray(tcp_world(robot), float) - origin
        pts += [tcp, *np.atleast_2d(np.asarray(pad_tips_world(robot), float) - origin)]
        q = _np(robot.data.joint_pos)[0]
        names = list(robot.joint_names)
        fingers = [abs(float(q[i])) for i, n in enumerate(names) if n.endswith(("joint7", "joint8"))]
        tcps.append((tcp, sum(fingers) if fingers else 0.0))
    return pts, tcps


def _apply(environment, a: L.Actions) -> None:
    if a.hold_pad is True:
        hold_joint(environment, "pad", 0.0)
    elif a.hold_pad is False:
        release_joint(environment, "pad")
    for joint, value in (("slide", a.slide_hold), ("turn", a.turn_hold)):
        if value is L.KEEP:
            continue
        if value is None:
            release_joint(environment, "pulley", joint)
        else:
            hold_joint(environment, "pulley", float(value), joint)
    if a.key_pose is not None:
        write_body_pose(environment, "key", *a.key_pose)
    if a.driver_pose is not None:
        pos, quat, spin = a.driver_pose
        write_body_pose(environment, "driver", pos, quat, (0.0, 0.0, 0.0), spin)


@on_step
def _step(environment) -> None:
    if environment is None:
        return
    t = TRACK["t"]
    try:
        p, k, d, pad = (state(environment, n) for n in ("pulley", "key", "driver", "pad"))
        pts, tcps = _robot_points(environment)
        x = L.Inputs(slide=p.slide, turn=p.turn, key_pos=k.pos, key_quat=k.quat_xyzw, driver_pos=d.pos,
                     driver_quat=d.quat_xyzw, driver_ang_vel=d.ang_vel, pad=pad.position, grippers=pts, tcps=tcps)
        _apply(environment, t.step(x))
    except Exception as err:  # noqa: BLE001 - never credit a run whose state couldn't be read
        t.facts.hand_clear, t.facts.still = 0.0, False
        t.log["error"] = f"{type(err).__name__}: {err}"
        print(f"[key-pulley] step error: {t.log['error']}", flush=True)
    if DEBUG and t.steps % (CONTROL_HZ // 2) == 0:
        print("[key-pulley] " + json.dumps(t.log, default=str), flush=True)


def score(objects) -> float:
    """0 after a latched failure (the key cocked under the hub, the key or the driver off the bench). Otherwise: 1.0
    with the key home, the pulley keyed on it against the shoulder, the set screw tightened there until the driver's
    clutch slipped, the driver out of the screw and on the bench, every gripper point 2 cm from the pulley, key and
    driver, and nothing moving more than 5 mm over the last second; 0.75 locked like that but short of the rest; 0.5
    keyed and against the shoulder, screw not tight; 0.25 the key home only; else 0."""
    return L.task_score(TRACK["t"].facts)


def success(objects) -> bool:
    return score(objects) >= 1.0


# -- the scripted demo -------------------------------------------------------------------------------------------------
FINGERS = {"open": FINGER_OPEN, "closed": FINGER_CLOSED,
           **{k: -g * MM / 2 for k, g in L.FINGER_GAP.items() if k not in ("open", "closed")}}


def _waypoint(arm: str, step):
    seconds, pos, fingers, rot, label = step
    if isinstance(rot, str) or pos is None:
        w = ready_pose(arm, duration=seconds, label=label)
        w.finger = FINGERS[fingers]
        return w
    return Pose(seconds, tuple(float(v) for v in pos), FINGERS[fingers], label, rot=np.asarray(rot, float))


def plan_waypoints(plan: dict) -> dict:
    return {arm: [_waypoint(arm, s) for s in steps] for arm, steps in plan.items()}


def demo():
    """Open loop: the right hand sets the key in its seat and presses it home; the left turns the pulley's keyway onto
    the key, pushes the pulley on to the shoulder, takes the driver from its stand, tightens the set screw in two
    strokes until the clutch slips, and puts the driver back. DEMO_VARIANT picks a degenerate run instead."""
    return plan_waypoints(L.demo_plan(VARIANT))
