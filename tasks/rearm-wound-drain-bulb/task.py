"""At a nursing-skills bedside: a training manikin lies on a bed beyond a grey side rail, head on a pillow to the robot's
left, in a light-blue gown with a blanket over the legs. A clear drain tube runs from a dressing on the manikin's near side, under
the rail, into a white oval suction bulb about 10 cm tall that hangs loose from a strap on the rail, in front of the
robot and a little to its right. The bulb has a small port on top with a white cap on a short tether; the cap stands
open. A measuring cup with a little pink fluid in it stands on the bedside table to the robot's left.
"""

# yam_arena hands the docstring above to the policy as task documentation, so it describes only what is on the table:
# the technique (squeeze the bulb flat, then cap it while it is still flat) is what the task tests, and stays out.
#
# How the bulb works here (labkit HangingBulb, one articulation). It hangs free: it swings about a point 16 cm above
# its centre against a soft spring (its strap and tubing), so the robot has to steady it; pulled more than 3 cm from
# where it hangs, the drain has been tugged and the trial scores 0. Its two faces are invisible paddles on sprung
# slides (about 10 N to hold flat); the white oval you see is a mesh that task code flattens each step to the gap
# between them. The cap is a hinged lid on the port whose tether springs it open: it seats only when pressed within 2
# degrees of flat, and task code then holds it there (a push-fit plug grips the port). When it seats, task code holds
# both faces where they are, both ways (labkit.hold_joint): a sealed bulb can neither spring open (the trapped vacuum)
# nor be squeezed further (the trapped air). The seal is final for the trial.
# drain_logic.py holds the numbers and the scoring; it imports nothing from the simulator, so tests run anywhere.

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # labkit and drain_logic sit next to this file

import numpy as np  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

import drain_logic as D  # noqa: E402
from labkit import (  # noqa: E402
    Container, HangingBulb, Latch, Part, Pose, Prop, cylinder_mesh, ellipsoid_mesh, forward_grip, held, hold_joint,
    on_reset, on_step, ready_pose, release_joint, scale_part, state, tube_mesh,
)

SCENE = "station"
TOP = TABLE_TOP[SCENE]
assert abs(TOP - D.TOP) < 1e-9, "drain_logic.TOP must match the station's table top"
CONTROL_HZ = D.CONTROL_HZ
SUCCESS_HOLD_S = 1.0
INSTRUCTION = "The training manikin's drain bulb has just been emptied. Put it back on suction without tugging on the drain tube."

WHITE, BULB_WHITE, GREY = (0.95, 0.95, 0.94), (0.93, 0.95, 0.96), (0.55, 0.57, 0.60)
GOWN, SKIN, SHEET, TUBE, FLUID = (0.60, 0.77, 0.88), (0.93, 0.74, 0.62), (0.96, 0.96, 0.95), (0.84, 0.89, 0.91), \
    (0.86, 0.52, 0.48)
YAW_DEG = math.degrees(D.YAW)
X, Y = D.BULB_XY

# -- the bulb: one hanging assembly (labkit HangingBulb) -----------------------------------------------------------------
# Hanging point SWING_DROP above the bulb's centre; swings against a soft spring (strap and tubing); two invisible sprung
# faces; a push-fit cap on a sprung tether. Its visible parts ride on its core, so they swing with it.
BULB = HangingBulb(
    "drain_bulb", pos=tuple(D.HANG_POINT), yaw=D.YAW, size=(2 * D.HALF_W, 2 * D.HALF_T, 2 * D.HALF_H),
    color=BULB_WHITE, drop=D.SWING_DROP, swing_spring=D.SWING_SPRING, swing_damping=0.05, max_swing=D.MAX_SWING,
    core_mass=0.04, paddle=D.PADDLE, face_at=D.HALF_T - D.PADDLE[1] / 2, travel=D.TRAVEL, spring=D.SPRING,
    face_damping=2.0, face_mass=0.02, port=D.PORT, port_z=D.PORT_Z - D.BULB_Z, port_color=WHITE,
    cap_mesh=cylinder_mesh(D.PLUG_R, D.PLUG_T, 20), cap_size=(2 * D.PLUG_R, 2 * D.PLUG_R, D.PLUG_T),
    cap_pivot=(D.PLUG_PIVOT[1], D.PLUG_PIVOT[2]), cap_at=D.PLUG_AT, cap_mass=D.PLUG_MASS,
    cap_limits=(0.0, D.PLUG_OPEN), cap_start=D.PLUG_OPEN, cap_rest=D.PLUG_OPEN, cap_spring=D.PLUG_SPRING,
    cap_damping=0.002, cap_latch=Latch(detent=D.PLUG_SEATED, tol=D.PLUG_UNSEATED), cap_color=(0.98, 0.98, 0.98),
    parts=(
        Part("shell", ellipsoid_mesh(D.HALF_W, D.HALF_T, D.HALF_H, n=28), at=(0.0, 0.0, -D.SWING_DROP),
             color=BULB_WHITE, roughness=0.2),
        Part("strap", tube_mesh(D.strap_path_core(), 0.004), color=WHITE, roughness=0.7),
        Part("tail", tube_mesh(D.tail_path_core(), 0.003), color=TUBE, roughness=0.2),
    ),
)

# -- the bedside --------------------------------------------------------------------------------------------------------
RAIL = Box("bed_rail", pos=D.RAIL["pos"], size=D.RAIL["size"], color=GREY)
POSTS = [Box(f"rail_post_{i}", pos=p["pos"], size=p["size"], color=GREY) for i, p in enumerate(D.POSTS)]
MATTRESS = Box("mattress_sheet", pos=D.MATTRESS["pos"], size=D.MATTRESS["size"], color=SHEET)
_EXIT = tuple(D.DRAIN_EXIT - np.asarray(D.MANIKIN_ORIGIN))
HAIR, PILLOW, BLANKET, FOLD = (0.30, 0.22, 0.16), (0.97, 0.97, 0.97), (0.92, 0.89, 0.80), (0.96, 0.94, 0.88)


def _arm(side: str, sign: float) -> tuple:
    """One arm lying along the manikin's side: gown sleeve, elbow, forearm and hand (sign -1: the robot's side)."""
    shoulder, elbow = (sign * 0.20, 0.30, 0.10), (sign * 0.22, 0.12, 0.07)
    wrist, hand = (sign * 0.21, -0.04, 0.05), (sign * 0.20, -0.08, 0.045)
    return (
        Part(f"{side}_sleeve", tube_mesh([shoulder, elbow], 0.040, n=12), color=GOWN, roughness=0.8),
        Part(f"{side}_elbow", ellipsoid_mesh(0.036, 0.036, 0.036, n=12), at=elbow, color=SKIN, roughness=0.6),
        Part(f"{side}_forearm", tube_mesh([elbow, wrist], 0.033, n=12), color=SKIN, roughness=0.6),
        Part(f"{side}_hand", ellipsoid_mesh(0.030, 0.055, 0.020, n=12), at=hand, color=SKIN, roughness=0.6),
    )


def _legs() -> tuple:
    """The legs under a blanket folded down below the drain site: a flat drape to the mattress edges, thighs and shins
    as rounded ridges, and the feet pointing up and tenting it (the manikin lies on its back). All of it ends
    inside the mattress (manikin frame y -0.635; the mattress and table end at -0.65)."""
    parts = [
        Part("blanket_drape", ellipsoid_mesh(0.24, 0.26, 0.016), at=(0.0, -0.385, 0.008), color=BLANKET,
             roughness=0.95),
        Part("blanket_mound", ellipsoid_mesh(0.19, 0.245, 0.055, n=28), at=(0.0, -0.385, 0.035), color=BLANKET,
             roughness=0.95),
        Part("blanket_fold", ellipsoid_mesh(0.215, 0.032, 0.024, n=24), at=(0.0, -0.18, 0.082), color=FOLD,
             roughness=0.95),
        Part("blanket_toes", ellipsoid_mesh(0.11, 0.045, 0.05, n=20), at=(0.0, -0.592, 0.045), color=BLANKET,
             roughness=0.95),
    ]
    for side, x in (("near", -0.08), ("far", 0.08)):
        parts += [
            Part(f"{side}_thigh", ellipsoid_mesh(0.068, 0.15, 0.05, n=16), at=(x, -0.31, 0.047), color=BLANKET,
                 roughness=0.95),
            Part(f"{side}_shin", ellipsoid_mesh(0.05, 0.12, 0.04, n=16), at=(0.9 * x, -0.49, 0.04),
                 color=BLANKET, roughness=0.95),
            Part(f"{side}_foot", ellipsoid_mesh(0.042, 0.036, 0.06, n=20), at=(0.85 * x, -0.595, 0.06),
                 color=BLANKET, roughness=0.95),
        ]
    return tuple(parts)


# A training manikin lying on the bed, head on a pillow toward the robot's left, in a gown, a blanket over the legs folded
# down below the drain site. Visual only: nothing here is within the arms' working area or collides.
MANIKIN = Prop("manikin", pos=D.MANIKIN_ORIGIN, size=(0.01, 0.01, 0.01), parts=(
    Part("pillow", ellipsoid_mesh(0.17, 0.11, 0.05), at=(0.0, 0.50, 0.05), color=PILLOW, roughness=0.9),
    Part("head", ellipsoid_mesh(0.075, 0.095, 0.085), at=(0.0, 0.47, 0.12), color=SKIN, roughness=0.6),
    Part("hair", ellipsoid_mesh(0.072, 0.070, 0.070), at=(0.0, 0.50, 0.13), color=HAIR, roughness=0.9),
    Part("nose", ellipsoid_mesh(0.012, 0.016, 0.014, n=10), at=(0.0, 0.44, 0.203), color=SKIN, roughness=0.6),
    Part("neck", tube_mesh([(0.0, 0.40, 0.10), (0.0, 0.35, 0.10)], 0.045, n=12), color=SKIN, roughness=0.6),
    Part("shoulders", ellipsoid_mesh(0.21, 0.08, 0.085), at=(0.0, 0.30, 0.10), color=GOWN, roughness=0.8),
    Part("chest", ellipsoid_mesh(0.18, 0.22, 0.10), at=(0.0, 0.10, 0.10), color=GOWN, roughness=0.8),
    Part("hips", ellipsoid_mesh(0.18, 0.12, 0.09), at=(0.0, -0.12, 0.09), color=GOWN, roughness=0.8),
    *_arm("near", -1.0), *_arm("far", 1.0),
    *_legs(),
    Part("dressing", ellipsoid_mesh(0.012, 0.035, 0.03), at=_EXIT, color=WHITE, roughness=0.9),
))
LINES = Prop("drain_lines", pos=(0.0, 0.0, 0.0), size=(0.01, 0.01, 0.01), parts=(
    Part("tubing", tube_mesh(D.tubing_path(), 0.003), color=TUBE, roughness=0.2),
))
CUP = Container("measuring_cup", pos=(*D.CUP_XY, TOP + 0.0015), size=(0.07, 0.07, 0.08), wall=0.003,
                color=(0.90, 0.93, 0.95), label=(0.30, 0.45, 0.70), band=(0.050, 0.054))
CUP_FLUID = Prop("cup_fluid", pos=(*D.CUP_XY, TOP + 0.003 + 0.0125), size=(0.01, 0.01, 0.01), parts=(
    Part("fluid", cylinder_mesh(0.030, 0.025, 20), color=FLUID, roughness=0.1),
))

OBJECTS = [BULB, RAIL, *POSTS, MATTRESS, MANIKIN, LINES, CUP, CUP_FLUID]

# -- the trial: seal, hold, score ---------------------------------------------------------------------------------------
TRIAL = {"t": D.DrainTrial()}


@on_reset
def _new_trial() -> None:
    TRIAL["t"] = D.DrainTrial()


def _points(environment) -> list:
    """Each gripper's TCP and pad tips, in the frame object states use (env 0's origin subtracted)."""
    from labkit import _np
    from yam_arena.runtime import pad_tips_world, tcp_world

    base = environment.unwrapped
    origin = _np(base.scene.env_origins)[0]
    out = []
    for key in ("left_robot", "right_robot"):
        robot = base.scene[key]
        out += [tcp_world(robot) - origin, pad_tips_world(robot) - origin]
    return out


FACES = ("squeeze_left", "squeeze_right")


def _faces(environment, bulb) -> tuple[float, float]:
    """The faces' positions: where the seal holds them while sealed (the bulb can't move then), else the joints'."""
    out = []
    for joint, now in zip(FACES, (bulb.squeeze_left, bulb.squeeze_right)):
        sealed = held("drain_bulb", joint)
        out.append(sealed if sealed is not None else now)
    return out[0], out[1]


def advance(bulb, left_pos: float, right_pos: float, points) -> str | None:
    """One step of the trial's memory from the bulb's state (what the on_step hook does; tests call it directly)."""
    return TRIAL["t"].step(bulb.cap_angle, bulb.cap_latched, left_pos, right_pos, points, bulb.centre, bulb.mouth)


@on_step
def _step(environment) -> None:
    """Every control step, before scoring: update the seal and the tug check, hold the sealed bulb and its cap, and
    draw the bulb's thickness."""
    if environment is None:
        return
    bulb = state(environment, "drain_bulb")
    left, right = _faces(environment, bulb)
    action = advance(bulb, left, right, _points(environment))
    if action == "seal":
        hold_joint(environment, "drain_bulb", 0.0, joint="cap_hinge")  # a push-fit plug grips the port
        for joint in FACES:
            hold_joint(environment, "drain_bulb", joint=joint)
    elif action == "unseal":
        for joint in FACES:
            release_joint(environment, "drain_bulb", joint)
    squash = D.thickness(left, right) / (2 * D.HALF_T)
    scale_part(environment, "drain_bulb", "core/shell", (1.0, squash, 1.0),
               at=(0.0, (left + right) / 2, -D.SWING_DROP))


def score(objects) -> float:
    """0 if the bulb was ever pulled more than 3 cm from where it hangs (the drain tugged), a gripper pad touched the
    open port's mouth, or the cap isn't seated at the end. 0.25 if it was seated
    while the bulb was less than 70% flat (no suction). 0.5 if seated flat but a gripper is still at the bulb, or the
    bulb ended less than 60% flat. 1.0 if seated while at least 70% flat, still at least 60% flat, both grippers
    away from the bulb."""
    return TRIAL["t"].score()


def success(objects) -> bool:
    """Full marks; must hold for SUCCESS_HOLD_S."""
    return TRIAL["t"].done()


# -- the scripted demo ----------------------------------------------------------------------------------------------------
def _waypoint(arm: str, step) -> Waypoint:
    seconds, pos, fingers, grip, label = step
    finger = FINGER_OPEN if fingers == "open" else FINGER_CLOSED
    if grip == "ready":
        w = ready_pose(arm, duration=seconds, label=label)
        w.finger = finger
        return w
    if grip == "forward":
        return Pose(seconds, pos, finger, label, rot=forward_grip(YAW_DEG))
    return Waypoint(seconds, pos, finger, label, yaw=D.YAW)


def demo():
    """Left arm: forward grip, fingers either side of the bulb, squeeze it flat and hold it. Right arm: closed fingers,
    push the open cap over onto the port, press it home, lift away. Then the left arm lets go and backs out. Open
    loop: drain_logic.demo_plan() has the numbers."""
    return {arm: [_waypoint(arm, s) for s in steps] for arm, steps in D.demo_plan().items()}
