"""On the station table, about 37 cm in front of the robot and 12 cm to its left, a spill of dark maple syrup about
10 cm across has dried tacky on the table top. To the robot's right, 26 cm out, stands a clear 500 mL trigger spray
bottle of warm water, its nozzle and trigger facing away from the robot. Nearer the robot and further to its left
lies a folded white cotton bar towel, flat.
"""

# yam_arena hands the docstring above to the policy as task documentation, so it describes only what is on the table.
#
# Task code (syrup_logic.Tracker, every control step) stands in for what PhysX can't do: liquids and adhesion. The spill
# is a 1 cm field of syrup and water; a trigger squeeze past FIRE_AT wets the field where the nozzle's line meets the
# table; the towel's cloth, pressed on the table by a fingertip and moving, picks syrup up (readily when it is soft)
# and drags it along (a smear, when it is dry), and leaves a film once it is loaded. The syrup's look follows the field:
# a colour field of 1 cm tiles (default) or, with SYRUP_LOOK=patches, one round patch per 2 cm square. syrup_logic.py
# has every number. Settings:
#   DEMO_VARIANT=strokes|outside_in|circles|zigzag|dry_wipe|spray_only|bare_fingers|upright_squeeze   demo() plays
#   SYRUP_DEBUG=1   prints the tracker's state (one JSON line) every half second
#   SYRUP_LOOK=field|patches   how the syrup is drawn (the score is the same)

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # labkit and syrup_logic sit next to this file

import numpy as np  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

import kit_replay  # noqa: E402  (plays demo_actions.csv through yam_scripted)
import syrup_logic as L  # noqa: E402
from labkit import (Cloth, ColorField, Part, Pose, Prop, SprayBottle, _mesh, cylinder_mesh, on_reset,  # noqa: E402
                    on_reset_env, on_step, ready_pose, set_field, set_part_visible, state, x_cylinder_mesh)

SCENE = "station"
assert abs(TABLE_TOP[SCENE] - L.TOP) < 1e-9, "syrup_logic.TOP must match the station's table top"
CONTROL_HZ = L.CONTROL_HZ
SUCCESS_HOLD_S = 1.0
INSTRUCTION = "Clean the dried syrup off the table."
VARIANT = os.environ.get("DEMO_VARIANT", "strokes")
DEBUG = bool(os.environ.get("SYRUP_DEBUG"))
LOOK = os.environ.get("SYRUP_LOOK", "field")
if LOOK not in ("field", "patches"):
    raise ValueError(f"SYRUP_LOOK must be field or patches, not {LOOK!r}")

SYRUP, SYRUP_WET = (0.28, 0.12, 0.03), (0.58, 0.33, 0.10)  # dark maple amber; thinned by water, lighter and glossy
PLASTIC, WHITE, BLACK, TERRY = (0.62, 0.78, 0.92), (0.95, 0.95, 0.92), (0.08, 0.08, 0.09), (0.96, 0.96, 0.94)
SPRAYER, TRIGGER_RED, MIST = (0.12, 0.33, 0.72), (0.80, 0.12, 0.10), (0.90, 0.94, 0.97)

# -- the spill: a visual layer of round patches, one sticky and one wet per cell (only one, or none, shows) ----------
# Each patch's radius is a little over half a cell's diagonal (14.1 mm), varied per cell, so the patches of
# neighbouring cells merge into one puddle with an uneven edge.


def _patch_r(i: int, j: int) -> float:
    return 0.0143 + 0.0015 * (((i * 7 + j * 13) % 5) / 4.0)


def _patches() -> tuple:
    parts = []
    for i in range(L.GRID_N):
        for j in range(L.GRID_N):
            cx, cy = L.cell_centre(i, j) - np.asarray(L.SPILL_C)
            r = _patch_r(i, j)
            parts.append(Part(f"s_{i}_{j}", cylinder_mesh(r, L.SYRUP_T, 16), at=(cx, cy, L.SYRUP_T / 2 + 2e-4),
                              color=SYRUP, roughness=0.2, visible=(i, j) in L.PUDDLE))
            parts.append(Part(f"w_{i}_{j}", cylinder_mesh(r, L.SYRUP_T, 16),
                              at=(cx, cy, L.SYRUP_T / 2 + 3e-4), color=SYRUP_WET, roughness=0.05, visible=False))
    return tuple(parts)


def _cone_x(length: float, r_end: float, n: int = 16) -> tuple:
    """A cone along +x from its tip at the origin to a disc of ``r_end`` at ``length`` (visual: the spray's mist)."""
    pts = [(0.0, 0.0, 0.0)] + [(length, r_end * np.cos(2 * np.pi * k / n), r_end * np.sin(2 * np.pi * k / n))
                               for k in range(n)]
    faces = [(0, 1 + (k + 1) % n, 1 + k) for k in range(n)] + [tuple(range(1, n + 1))]
    return _mesh(pts, faces)


def _spill():
    if LOOK == "patches":
        return Prop("syrup", pos=(L.SPILL_C[0], L.SPILL_C[1], L.TOP),
                    size=(L.GRID_N * L.CELL, L.GRID_N * L.CELL, 0.002), color=SYRUP, parts=_patches())
    shown, colors = L.Tracker().field_look()
    return ColorField("syrup", pos=(L.SPILL_C[0], L.SPILL_C[1], L.TOP), size=(L.FN * L.FINE, L.FN * L.FINE, 0.002),
                      color=SYRUP, n=L.FN, tile=L.FINE, shown0=tuple(bool(v) for v in shown.ravel()),
                      colors0=tuple(tuple(float(c) for c in rgb) for rgb in colors.reshape(-1, 3)))


SPILL = _spill()

# -- the spray bottle ---------------------------------------------------------------------------------------------------
BOTTLE = SprayBottle(
    "spray_bottle", pos=(L.BOTTLE_START[0], L.BOTTLE_START[1], L.TOP + 0.001), size=(2 * L.BODY_R, 2 * L.BODY_R, 0.22), color=PLASTIC, mass=L.BOTTLE_MASS,
    body_r=L.BODY_R, body_h=L.BODY_H,
    housing_boxes=((L.SHROUD[0], L.SHROUD[1], L.SHROUD[2], SPRAYER), (L.HEAD[0], L.HEAD[1], L.HEAD[2], SPRAYER)),
    housing_parts=(Part("bottle", cylinder_mesh(L.BODY_R, L.BODY_H, 32), at=(0.0, 0.0, L.BODY_H / 2), color=PLASTIC,
                        roughness=0.1),
                   Part("label", cylinder_mesh(L.BODY_R + 0.0006, 0.06, 32), at=(0.0, 0.0, 0.06), color=WHITE),
                   Part("shoulder", cylinder_mesh(0.022, 0.012, 24), at=(0.0, 0.0, L.BODY_H + 0.004), color=PLASTIC,
                        roughness=0.1),
                   Part("collar", cylinder_mesh(0.017, 0.010, 24), at=(-0.004, 0.0, L.BODY_H + 0.012), color=BLACK),
                   Part("nozzle", x_cylinder_mesh(0.0045, L.HEAD[2][0] + L.HEAD[1][0] / 2 - 0.001, L.NOZZLE[0], 12),
                        at=(0.0, 0.0, L.NOZZLE[2]), color=BLACK),
                   Part("mist", _cone_x(L.MIST_LEN, L.MIST_R), at=L.NOZZLE, color=MIST, roughness=0.9,
                        visible=False)),
    trigger=L.TRIGGER, trigger_rest=L.TRIGGER_REST, trigger_color=TRIGGER_RED, travel=L.TRIGGER_TRAVEL,
    rate=L.TRIGGER_RATE, damping=L.TRIGGER_DAMPING, com=L.BOTTLE_COM)

# -- the towel: true cloth (the merged napkin task's model and settings), one sheet as thick as the folded towel -------
_T = L.TOWEL_SIZE
TOWEL = Cloth("towel", pos=(L.TOWEL_C[0], L.TOWEL_C[1], L.TOP + _T[2] / 2 + 0.001), size=_T, color=TERRY,
              quarter=False, self_collide=True, edge=L.TOWEL_EDGE, thickness=_T[2],
              density=L.TOWEL_MASS / (_T[0] * _T[1] * _T[2]), rest_offset=_T[2] / 2, contact_offset=_T[2] / 2 + 0.004,
              bend=500.0, youngs=1e6, damping=1.0, bend_damping=5.0, friction=1.5)

OBJECTS = [SPILL, BOTTLE, TOWEL]

# -- task code, every control step ---------------------------------------------------------------------------------------
TRACK = {"t": L.Tracker()}


@on_reset
def _new_trial() -> None:
    TRACK["t"] = L.Tracker()


@on_reset_env
def _starting_look(environment) -> None:
    """The dried puddle only, no mist: set just before each reset (so its first frame shows it) and again after it
    (labkit.on_reset_env). The parts also spawn this way, for the first trial."""
    if LOOK == "patches":
        for i in range(L.GRID_N):
            for j in range(L.GRID_N):
                set_part_visible(environment, "syrup", f"s_{i}_{j}", (i, j) in L.PUDDLE)
                set_part_visible(environment, "syrup", f"w_{i}_{j}", False)
    else:
        set_field(environment, "syrup", *L.Tracker().field_look())
    set_part_visible(environment, "spray_bottle", "housing/mist", False)
    t = TRACK["t"]
    t.shown = L.start_cells()
    t.field_changed = False
    t.mist_shown = False


def _robot_points(environment) -> tuple[list, list]:
    """(each finger's pad tip, both arms, in a fixed order; every gripper point: on each finger's pad face at its tip,
    at the TCP and 2 cm behind it, and the TCP), task frame. From the gripper body's pose and the two finger joints,
    as the merged napkin task's labkit reads them (round 1 used one centreline point per arm, which missed a single
    open finger)."""
    from labkit import _np
    from yam_arena.constants import PAD_TIP_PAST_TCP, TCP_OFFSET
    from yam_arena.urdf import quat_xyzw_to_matrix

    base = environment.unwrapped
    origin = _np(base.scene.env_origins)[0]
    tips, pts = [], []
    for key in ("left_robot", "right_robot"):
        robot = base.scene[key]
        pose = _np(robot.data.body_link_pose_w)[0][list(robot.body_names).index("gripper")]
        jp, jn = _np(robot.data.joint_pos)[0], list(robot.joint_names)
        half_gap = max(abs(float(jp[jn.index(j)])) for j in ("joint7", "joint8") if j in jn)
        r = quat_xyzw_to_matrix(pose[3:7])
        for along in (PAD_TIP_PAST_TCP, 0.0, -0.02):
            centre = pose[:3] + r @ (np.array(TCP_OFFSET) + [0.0, 0.0, along]) - origin
            for side in (1.0, -1.0):
                q = centre + r[:, 0] * side * half_gap
                pts.append(q)
                if along == PAD_TIP_PAST_TCP:
                    tips.append(q)
            if along == 0.0:
                pts.append(centre)
    return tips, pts


def _step_dt(environment) -> float:
    dt = getattr(getattr(environment, "unwrapped", environment), "step_dt", None)
    try:
        return float(dt) if dt else L.DT
    except (TypeError, ValueError):
        return L.DT


@on_step
def _step(environment) -> None:
    if environment is None:
        return
    t = TRACK["t"]
    try:
        b, w = state(environment, "spray_bottle"), state(environment, "towel")
        tips, pts = _robot_points(environment)
        x = L.Inputs(bottle_pos=b.pos, bottle_quat=b.quat_xyzw, trigger=b.travel, nodes=w.nodes, tips=tips,
                     grippers=pts, dt=_step_dt(environment))
        changes = t.step(x)
        if LOOK == "patches":
            for part, show in changes:
                set_part_visible(environment, "syrup", part, show)
        elif t.field_changed:
            set_field(environment, "syrup", *t.field_look())
        if t.mist_on() != t.mist_shown:  # the spray's mist shows for MIST_S after each squeeze
            t.mist_shown = t.mist_on()
            set_part_visible(environment, "spray_bottle", "housing/mist", t.mist_shown)
    except Exception as err:  # noqa: BLE001 - never credit a run whose state couldn't be read
        t.facts.hands_clear, t.facts.still = False, False
        t.log["error"] = f"{type(err).__name__}: {err}"
        print(f"[syrup] step error: {t.log['error']}", flush=True)
    if DEBUG and t.steps % (CONTROL_HZ // 2) == 0:
        print("[syrup] " + json.dumps(t.log, default=str), flush=True)


def score(objects) -> float:
    """Read on 2 cm squares (19 start under the spill; clean: mean syrup under CLEAN_M). 0 when a gripper touched the
    syrup without the towel under it, or the bottle or the towel left the table, or under 5 of the 19 squares are
    clean; 0.25 at least 5; 0.5 at least 10; 0.75 all 19; 1.0 all 19 and no syrup smeared on any other square, the
    bottle standing upright on the table, both grippers 2 cm clear of the bottle and the towel, and nothing moving more
    than 5 mm over the last second."""
    return L.task_score(TRACK["t"].facts)


def success(objects) -> bool:
    return score(objects) >= 1.0


# -- the demo ------------------------------------------------------------------------------------------------------------
# demo_actions.csv is the 14 joint targets the runner executed at each control step of a full-score trial (round 3 on
# the GPU, 20 Nov 2026: the right hand's spray plan and the left's closed-loop wipe, syrup_logic.Wiper, which reads
# the towel's nodes). The standard command replays it row by row (kit_replay); replayed open loop in the same launch it
# scored 1.0 again, in rounds 2 and 3. DEMO_VARIANT picks an open-loop plan instead (development only).
DEMO_SHA256 = "3a8e54049958bb1a48eeb32bfe84c8e39158ca2e263bf71feb1858168fc9b1c5"

def _finger(name: str) -> float:
    if name == "open":
        return FINGER_OPEN
    if name == "closed":
        return FINGER_CLOSED
    return -L.FINGER_GAP[name] * L.MM / 2


def _waypoint(arm: str, step):
    seconds, pos, fingers, rot, label = step
    if isinstance(rot, str) or pos is None:
        w = ready_pose(arm, duration=seconds, label=label)
        w.finger = _finger(fingers)
        return w
    return Pose(seconds, tuple(float(v) for v in pos), _finger(fingers), label, rot=np.asarray(rot, float))


def plan_waypoints(plan: dict) -> dict:
    return {arm: [_waypoint(arm, s) for s in steps] for arm, steps in plan.items()}


def demo():
    """The recorded full-score trial: the right hand takes the spray bottle, sprays the spill in a 3 x 3 pattern and
    stands it back; the left presses the towel onto the spill and scrubs it clean in back-and-forth strokes, then lets
    go. With DEMO_VARIANT set, an open-loop plan instead (syrup_logic.demo_plan)."""
    if os.environ.get("DEMO_VARIANT"):
        return plan_waypoints(L.demo_plan(VARIANT))
    return kit_replay.replay_demo(HERE / "demo_actions.csv", expect_sha256=DEMO_SHA256)
