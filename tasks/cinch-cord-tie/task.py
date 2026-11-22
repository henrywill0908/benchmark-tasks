"""A TV hangs on the wall just behind the console in front of the robot. Three cords hang from the TV's bottom edge
straight down to a white power strip on the console, gathered in a bundle about 38 cm in front of the robot: a black TV
power cord (7 mm), a black HDMI cable (7.3 mm) and a white phone charger cable (3.5 mm). A black nylon cable tie (a
200 mm by 4.8 mm strap; its head 8 by 9 by 5.6 mm with a slot through it) is loosely looped round the cords 17 cm above
the console. Its head is held in a blue clip on a steel post just in front of the cords, the slot facing left. The
strap leaves the head toward the wall, goes round the back of the cords and comes forward on the left; its last 6 cm
run straight toward the slot, the tip stopping 4 mm short of it.
"""

# yam_arena hands the docstring above to the policy as task documentation, so it describes only what is on the table.
#
# The submitted task is the default build, jig: the tie looped round the cords, its head welded where the clip (drawn
# only, task.stand_decor) holds it. tie_logic.py has every number; demo() plays probe trial m-demo with the probe's
# closed loop (Steering). Settings, for the probe (../../../cinch-cord-tie-gpu/probe.py); the defaults are the task:
#   TIE_BUILD=jig|wrap|free|tight   jig: the submitted start. wrap: the strap straight behind the cords (the robot wraps
#                          it). free: flat on the console. tight: the jig with the slot narrowed to 5.6 mm.
#   TIE_RATCHET=latch|cam|brake|open   latch: the pawl held clear until LATCH_AT is out, then braked whenever the
#                          strap backs out. cam: the sprung pawl alone. brake: braked from the start. open: no ratchet.
#   TIE_PHYS=heavier        the tie's physics preset (tie_logic.PHYS: base, heavy, heavier)
#   TIE_TRIAL=m-demo       which probe trial demo() plays (tie_logic.TRIALS and KEPT)
# The cords are rigid and already gathered (fixed cylinders).

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # labkit and tie_logic sit next to this file

import numpy as np  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box, Cylinder  # noqa: E402

import tie_logic as L  # noqa: E402
from labkit import (CableTie, Harness, HarnessPart, HarnessWire, Pose, hold_joint, on_reset, on_step,  # noqa: E402
                    ready_pose, release_joint, set_tie_yield, state)

SCENE = "station"
assert abs(TABLE_TOP[SCENE] - L.TOP) < 1e-9, "tie_logic.TOP must match the station's table top"
CONTROL_HZ = L.CONTROL_HZ
SUCCESS_HOLD_S = 1.0
INSTRUCTION = "Close the cable tie snugly around the three cords, then let go of it."
BUILD = os.environ.get("TIE_BUILD", "jig")  # round 7: the start that has scored 1.0 (wrap: the robot wraps it too)
PHYS = os.environ.get("TIE_PHYS", "heavier")  # round 6 scored 1.0 on it; tie_logic.PHYS: the tie's masses, damping and contact skin
TRIAL = os.environ.get("TIE_TRIAL", "m-demo")

WHITE, BLACK, DRYWALL = (0.94, 0.94, 0.92), (0.06, 0.06, 0.07), (0.90, 0.90, 0.87)


def _box(name, span, color) -> Box:
    (x0, x1), (y0, y1), (z0, z1) = span["x"], span["y"], span["z"]
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0), color=color)


ROOM = [_box("wall", L.WALL, DRYWALL), _box("tv", L.TV, BLACK), _box("power_strip", L.STRIP, WHITE)]
CORDS = [Cylinder(c.name, pos=(c.xy[0], c.xy[1], (L.CORD_Z[0] + L.CORD_Z[1]) / 2), color=c.color,
                  radius=c.diameter / 2, height=L.CORD_Z[1] - L.CORD_Z[0]) for c in L.CORDS]
STEEL, CLIP_BLUE = (0.62, 0.63, 0.65), (0.10, 0.32, 0.78)  # a bare steel rod; a blue plastic clip (shows the black head)


def stand_decor(build: str) -> tuple:
    """The clip stand that holds the head (fixed builds only; tie_logic.stand_parts), as the tie's visual-only decor
    in the head frame. Round 13 spawned it as colliders: their contact skin, 0.3 mm off the head, changed the
    pre-looped start's fresh trial from 1.0 to 0.25. Drawn only, the physics is that of the welded head (what the clip
    does); the arm and finger checks still keep clear of it (armcheck, tie_logic.finger_boxes)."""
    b = L.BUILDS[build]
    hp, rot = np.asarray(b.head_pos, float), np.asarray(b.head_rot, float)
    if not np.allclose(np.abs(rot), np.round(np.abs(rot)), atol=1e-9):
        raise ValueError("the clip stand is drawn for a head square to the world axes")
    out = []
    for name, kind, geo in L.stand_parts(b):
        if kind == "box":
            lo, hi = geo
            at = rot.T @ ((lo + hi) / 2 - hp)
            size = np.abs(rot.T) @ (hi - lo)
            out.append((name, "box", tuple(float(v) for v in at), tuple(float(v) for v in size),
                        STEEL if name.startswith("stand") else CLIP_BLUE))
        else:
            xy, r, (z0, z1) = geo
            at = rot.T @ (np.array([xy[0], xy[1], (z0 + z1) / 2]) - hp)
            axis = "XYZ"[int(np.argmax(np.abs(rot.T @ np.array([0.0, 0.0, 1.0]))))]
            out.append((name, "cyl", tuple(float(v) for v in at), (float(r), float(z1 - z0), axis), STEEL))
    return tuple(out)


TIE_BLACK = (0.06, 0.06, 0.07)  # black nylon: shows against the off-white wall (round 1's white tie barely did)


def build_tie(build: str = BUILD, phys: str = PHYS) -> CableTie:
    b = L.BUILDS[build]
    ph = L.PHYS[phys]
    k = ph["mass_scale"] / L.MASS_SCALE  # tie_logic's masses are at MASS_SCALE
    return CableTie(
        "tie", pos=b.head_pos, size=tuple(L.HEAD), color=TIE_BLACK, head_color=TIE_BLACK,
        quat=tuple(float(v) for v in L.matrix_to_quat_xyzw(b.head_rot)),
        points=tuple(tuple(float(v) for v in p) for p in b.points), width=L.W, thick=L.T, tip_width=L.TIP_W,
        tip_thick=L.TIP_T, link_mass=L.LINK_MASS * k, head_size=tuple(L.HEAD), head_mass=L.HEAD_MASS * k,
        passage_x=L.PASSAGE_X, passage_y=b.passage_y, stiffness=L.STRAP_K * ph.get("k_scale", 1.0),
        damping=ph["joint_damping"],
        armature=ph["armature"], max_bend=L.MAX_BEND, friction=L.TIE_FRICTION, fixed=b.fixed,
        pawl_hinge=tuple(L.PAWL_HINGE), pawl_len=L.PAWL_LEN, pawl_radius=L.PAWL_R, pawl_width=min(L.PAWL_WIDTH, 2 * b.passage_y - 0.4 * L.MM),
        pawl_rest=L.PAWL_REST, pawl_mass=L.PAWL_MASS * k, pawl_spring=L.PAWL_SPRING * k, pawl_damping=L.PAWL_DAMPING * k,
        hold_stiffness=L.BRAKE_STIFFNESS, contact_offset=ph["offset"], body_linear_damping=ph["body_lin"],
        body_angular_damping=ph["body_ang"], yield_angle=math.radians(L.YIELD_DEG), decor=stand_decor(build))


def build_cords() -> Harness:
    """Round 19, the lift start: the three cords as one Harness, anchored where they hang by the wall
    (tie_logic.LIFT_ANCHOR_Z), lying along the console into the power strip (the harness's device, free on the
    console). Each cord's start shape (tie_logic.lift_cord_points) is its relaxed shape."""
    anchor = np.array([L.LIFT_WALL_X, L.LIFT_HANG_Y, L.LIFT_ANCHOR_Z])
    strip = L.lift_strip_centre()
    wires = []
    for name, r, off, kg_m, color in L.LIFT_CORDS:
        pts = L.lift_cord_points(off, r)
        seg = float(np.linalg.norm(pts[1] - pts[0]))
        wires.append(HarnessWire(name, tuple(float(v) for v in pts[0] - anchor),
                                 tuple(float(v) for v in pts[-1] - strip), L.LIFT_SEGMENTS, r, color, kg_m * seg,
                                 points=tuple(tuple(float(v) for v in p) for p in pts - anchor)))
    s = L.LIFT_STRIP
    size = (s["x"][1] - s["x"][0], s["y"][1] - s["y"][0], s["z"][1] - s["z"][0])
    return Harness("cords", pos=tuple(float(v) for v in anchor), size=(0.01, 0.01, 0.01), color=BLACK,
                   wires=tuple(wires), parts=(HarnessPart("strip", size, (0.0, 0.0, 0.0), WHITE),),
                   device_at=tuple(float(v) for v in strip - anchor), device_mass=L.LIFT_STRIP_MASS,
                   stiffness=L.LIFT_K, damping=0.004, armature=5e-5, yield_angle=L.LIFT_YIELD,
                   max_bend=math.radians(90.0), style="d6", end_filter=2, wire_friction=0.5)


def hang_rods() -> list:
    """The cords from the TV's underside down to where the jointed part starts: fixed rods, out of the arms' way."""
    out = []
    for name, r, off, _, color in L.LIFT_CORDS:
        z0, z1 = L.LIFT_ANCHOR_Z, L.TV["z"][0]
        out.append(Cylinder(f"{name}_hang", pos=(L.LIFT_WALL_X, L.LIFT_HANG_Y - off, (z0 + z1) / 2), color=color,
                            radius=r, height=z1 - z0))
    return out


TIE = build_tie()
if BUILD in L.LIFT_BUILDS:  # round 19: the cords lie on the console (a Harness ending on the strip); no fixed rods, no
    #                            strip box (round 24: lift25 and lift30, the same with longer ties)
    CORDS_HARNESS = build_cords()
    OBJECTS = [*ROOM[:2], *hang_rods(), CORDS_HARNESS, TIE]
else:
    CORDS_HARNESS = None
    OBJECTS = [*ROOM, *CORDS, TIE]

# -- the ratchet and the release check (every control step) --------------------------------------------------------------
# score() sees object states only, so this hook records, each step, how close the nearest gripper point (TCP and pad
# tips, both arms) comes to the tie, and runs the ratchet's task-code part (TIE_RATCHET).
RATCHET = {"mode": os.environ.get("TIE_RATCHET", "latch"), "latch_at": L.LATCH_AT}
# Stillness from displacement, not velocity: round 2 (17 Nov) showed PhysX reporting 25-50 mm/s for a held tie whose
# head and tip stood still to 0.1 mm. The hook keeps the last second of the tie's points (head and strap joints);
# "still" is the farthest any of them moved over that second, per second (what a person sees on a 1 s video).
TRAIL: list = []
SEEN = {}


def set_ratchet(mode: str) -> None:
    if mode not in L.RATCHETS:
        raise ValueError(f"ratchet must be one of {L.RATCHETS}, got {mode!r}")
    RATCHET["mode"] = mode


def set_latch(at: float) -> None:
    """Latch mode: hold the pawl clear until this much tail (m) is out (the probe sets it per trial)."""
    RATCHET["latch_at"] = float(at)


def set_plastic(yield_deg: float | None) -> None:
    """The strap's yield angle per joint (deg), or None for a purely elastic strap (the probe sets it per trial)."""
    set_tie_yield("tie", None if yield_deg is None else math.radians(yield_deg))


@on_reset
def _new_trial() -> None:
    SEEN.clear()
    TRAIL.clear()
    SEEN.update(hand_clear=float("inf"), drift_speed=None, error=None, through=0.0, s_max=0.0, brake_on=False, brake_s=0.0, brakes=0,
                releases=0, opened=False, latched=False, captured_step=None, steps=0)


_new_trial()


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


def _ratchet(environment, s, through: float) -> None:
    mode = RATCHET["mode"]
    if mode == "open" and not SEEN["opened"]:  # the pawl turned flat against the passage's wall, out of the way
        hold_joint(environment, "tie", 0.0, "pawl")
        SEEN["opened"] = True
    if mode == "latch" and not SEEN["latched"]:  # round 6: the pawl clear of the passage until LATCH_AT is out
        if max(SEEN["s_max"], through) < RATCHET.get("latch_at", L.LATCH_AT):
            if not SEEN["opened"]:
                hold_joint(environment, "tie", 0.0, "pawl")
                SEEN["opened"] = True
        else:  # onto the strap's face on its own spring; from now on, the brake
            release_joint(environment, "tie", "pawl")
            SEEN.update(opened=False, latched=True)
    braking = mode == "brake" or (mode == "latch" and SEEN["latched"])
    if braking and through > 0.0:  # round 5: only while tail is out (round 4: braking an empty slot blocked the
        #                            next push)
        if not SEEN["brake_on"] and through <= SEEN["s_max"] - L.BRAKE_SLIP:
            hold_joint(environment, "tie", min(s.pawl_angle + math.radians(L.BRAKE_DEG), L.PAWL_REST), "pawl")
            SEEN.update(brake_on=True, brake_s=through, brakes=SEEN["brakes"] + 1)
        elif SEEN["brake_on"] and through > SEEN["brake_s"] + L.BRAKE_RELEASE:
            release_joint(environment, "tie", "pawl")
            SEEN.update(brake_on=False, releases=SEEN["releases"] + 1)
    SEEN["s_max"] = max(SEEN["s_max"], through)


@on_step
def _step(environment) -> None:
    if environment is None:
        return
    SEEN["steps"] += 1
    try:
        s = state(environment, "tie")
        th = L.threading(s.polyline, s.pos, s.quat_xyzw, TIE.passage_y)
        SEEN["through"] = th.through
        if th.through > 0.0 and SEEN["captured_step"] is None:
            SEEN["captured_step"] = SEEN["steps"]
        _ratchet(environment, s, th.through)
        SEEN["hand_clear"] = L.hand_clearance(_gripper_points(environment), s.polyline, s.pos, s.quat_xyzw)
        pts = np.vstack([np.asarray(s.pos, float)[None, :], np.asarray(s.polyline, float)])
        TRAIL.append(pts)
        window = int(round(SUCCESS_HOLD_S * CONTROL_HZ))
        del TRAIL[:-window - 1]
        if len(TRAIL) > window:
            span = len(TRAIL) - 1
            SEEN["drift_speed"] = float(np.max(np.linalg.norm(TRAIL[-1] - TRAIL[0], axis=1))) * CONTROL_HZ / span
        steering = DEMO["steering"]
        if steering is not None:  # the demo's closed loop, for the next control step (this hook runs after each)
            r = L.quat_xyzw_to_matrix(s.quat_xyzw)
            steering.update(SEEN["steps"] + 1, th.through, r @ np.array([th.tip_h[0] - L.SLOT_C, th.tip_h[1], 0.0]),
                            state=s)
    except Exception as err:  # noqa: BLE001 - never credit a tie whose grippers couldn't be checked
        SEEN["hand_clear"], SEEN["error"] = 0.0, f"{type(err).__name__}: {err}"


def score(objects) -> float:
    """0 unless every gripper point is at least 3 cm from the tie, the tie is at rest (no point of it moved more than
    1 cm over the last second; before a second has passed, its velocities under 1 cm/s) and it is closed
    round all three cords (the tail at least 2 mm out of the head's far side, every cord through the loop). Then 1.0 if
    snug (the loop at most 8 mm longer than the tightest the cords allow round the head) with the head at least 10 cm
    above the console; 0.5 within 25 mm; 0.25 otherwise."""
    s = objects.get("tie")
    if s is None or getattr(s, "polyline", None) is None:
        return 0.0
    drift = SEEN.get("drift_speed")  # the simulator's own record of how far it moved over the last second
    speed = drift if drift is not None else max(float(np.linalg.norm(s.lin_vel)), float(getattr(s, "link_speed_max", 0.0)))
    return L.task_score(s.polyline, s.pos, s.quat_xyzw, SEEN["hand_clear"], speed, TIE.passage_y)


def success(objects) -> bool:
    return score(objects) >= 1.0


# -- the probe's scripted moves ---------------------------------------------------------------------------------------------
FINGERS = {"open": FINGER_OPEN, "closed": FINGER_CLOSED,  # every other name: finger joint = -half the gap
           **{k: -g * L.MM / 2 for k, g in L.FINGER_GAP.items() if k not in ("open", "closed")}}


def _waypoint(arm: str, step):
    seconds, pos, fingers, rot, label = step
    finger = FINGERS[fingers]
    if isinstance(rot, str) or pos is None:
        w = ready_pose(arm, duration=seconds, label=label)
        w.finger = finger
        return w
    return Pose(seconds, tuple(float(v) for v in pos), finger, label, rot=np.asarray(rot, float))


def plan_waypoints(plan: dict) -> dict:
    """A tie_logic plan (trial_plan) as yam_arena waypoints."""
    return {arm: [_waypoint(arm, s) for s in steps] for arm, steps in plan.items()}


class Steering:
    """The probe's closed loop on the inserting (left) hand, run from this task's step hook so that yam_scripted plays
    exactly what the probe does. Every L.STEER_EVERY control steps: while the hand is lining up or pushing and no tail
    is out yet, it shifts sideways onto the tip's measured offset from the slot's centre line (at most steer_cap); while
    it pushes, it goes further along the slot until L.HANDOVER_OUT of tail is out (at most depth_cap). The shift moves
    every one of the hand's waypoints, so ArmPlan's target is the plan's plus the shift, as the probe computes it. A
    scripted demo may read the simulator's state; a policy under test gets only the cameras and its joints."""

    def __init__(self, tr: L.Trial, poses: list, right: list | None = None) -> None:
        if tr.steer_yaw:
            raise ValueError("the demo's steering shifts the hand; it does not turn it (steer_yaw)")
        b = L.BUILDS[tr.build]
        self.tr, self.poses = tr, poses
        self.base = [np.asarray(w.pos, float) for w in poses]
        self.d_dir = L.jig_frame(b)["through"] if b.fixed else np.zeros(3)
        self.corr, self.depth_mm = np.zeros(3), 0.0
        # round 16: the pulling (right) hand's tighten, along its last stroke, until tr.tighten_slack short of snug
        self.right = right if right is not None else []
        self.right_base = [np.asarray(w.pos, float) for w in self.right]
        self.pull_dir, self.tighten_mm, self.tighten_to = np.zeros(3), 0.0, 0.0
        if tr.tighten > 0.0:
            at = {w.label: np.asarray(w.pos, float) for w in self.right}
            if not {"pull 1", "pull 2", *L.TIGHTEN_LABELS} <= set(at):
                raise ValueError("tighten needs the hand-over plan's pull 1, pull 2 and tighten")
            v = at["pull 2"] - at["pull 1"]
            self.pull_dir = v / np.linalg.norm(v)
            self.tighten_to = L.snug_through(b) - tr.tighten_slack
        # round 18: the pulling hand's unstick, its target held to UNSTICK_LEAD ahead of a stalled strap (L.UNSTICK_*)
        self.back_mm, self.side_mm, self.through0, self.back_dir = 0.0, 0.0, None, np.zeros(3)
        if tr.unstick:
            at = {w.label: np.asarray(w.pos, float) for w in self.right}
            if not {"hold the tail", *L.UNSTICK_LABELS} <= set(at):
                raise ValueError("unstick needs the hand-over plan's hold the tail, pull 1 and pull 2")
            self.grip_at = at["hold the tail"]

    def label(self, t: float, poses: list | None = None) -> str:
        """The hand's step label at time t: the same walk, and the same float arithmetic, as ArmPlan's target."""
        for w in (self.poses if poses is None else poses):
            if t < w.duration:
                return w.label
            t -= w.duration
        return "done"

    def update(self, step: int, through: float, tip_err, state=None) -> None:
        """Before control step ``step`` (1-based, t = step / CONTROL_HZ), from the state after the step before:
        ``through`` the tail out (m), ``tip_err`` the tip's offset from the slot's centre line (world, m). ``state``
        (the tie's) is for ThreadSteering; this loop doesn't need it."""
        if step % L.STEER_EVERY:
            return
        tr, label = self.tr, self.label(step / CONTROL_HZ)
        if tr.steer and label in L.STEER_LABELS and through <= 0.0:
            self.corr = np.clip(self.corr - L.STEER_GAIN * np.asarray(tip_err, float), -tr.steer_cap, tr.steer_cap)
        if tr.depth and label in (L.HANDOVER_DEPTH_LABELS if tr.handover else L.DEPTH_LABELS):
            target = L.HANDOVER_OUT if tr.handover else L.DEPTH_TARGET.get(label, L.PROTRUDE)
            self.depth_mm = float(np.clip(self.depth_mm + 1000 * L.DEPTH_GAIN * max(target - through, 0.0), 0.0,
                                          1000 * tr.depth_cap))
        shift = self.corr + self.d_dir * self.depth_mm / 1000.0
        for w, p in zip(self.poses, self.base):
            w.pos = tuple(float(v) for v in p + shift)
        if not self.right:
            return
        t = step / CONTROL_HZ
        lab, moved = self.label(t, self.right), False
        if tr.tighten > 0.0 and lab in L.TIGHTEN_LABELS and through < self.tighten_to:
            self.tighten_mm = float(min(self.tighten_mm + 1000 * L.TIGHTEN_RATE * L.STEER_EVERY / CONTROL_HZ,
                                        1000 * tr.tighten))
            moved = True
        if tr.unstick:
            moved = self._unstick(t, lab, through) or moved
        if moved:
            tight = self.pull_dir * self.tighten_mm / 1000.0
            free = -self.back_dir * self.back_mm / 1000.0 + np.array([self.side_mm / 1000.0, 0.0, 0.0])
            for w, p in zip(self.right, self.right_base):
                if w.label in L.TIGHTEN_SHIFT_LABELS or w.label in L.UNSTICK_SHIFT_LABELS:
                    v = p + (tight if w.label in L.TIGHTEN_SHIFT_LABELS else 0.0)
                    w.pos = tuple(float(x) for x in (v + free if w.label in L.UNSTICK_SHIFT_LABELS else v))

    def planned(self, t: float) -> np.ndarray:
        """The pulling hand's unshifted target position at t: ArmPlan.target's walk and smoothstep (positions only; the
        first waypoint, which starts from the arm's ready pose, is never asked for)."""
        prev = self.right_base[0]
        for w, p in zip(self.right, self.right_base):
            if t < w.duration:
                s = min(max(t / w.duration, 0.0), 1.0)
                return prev + s * s * (3 - 2 * s) * (p - prev)
            t -= w.duration
            prev = p
        return prev

    def _unstick(self, t: float, lab: str, through: float) -> bool:
        """While the hand pulls and less than UNSTICK_ZONE is out: hold its target to UNSTICK_LEAD ahead of the strap's
        advance (along the way the plan has gone from where it took the tail), and while held, move it sideways. Past the
        zone, or after the pulls, let both go at their rates. True when anything changed."""
        tr, dt = self.tr, L.STEER_EVERY / CONTROL_HZ
        before = (self.back_mm, self.side_mm)
        if lab in L.UNSTICK_LABELS:
            if self.through0 is None:
                self.through0 = through
            v = self.planned(t) - self.grip_at
            adv = float(np.linalg.norm(v))
            if adv > 1e-6:
                self.back_dir = v / adv
            if through < L.UNSTICK_ZONE:
                self.back_mm = max(1000 * (adv - (through - self.through0) - L.UNSTICK_LEAD), 0.0)
                if self.back_mm > 0.0:
                    step = 1000 * L.UNSTICK_SIDE_RATE * dt
                    cap = 1000 * tr.unstick_side
                    self.side_mm = float(max(self.side_mm - step, cap) if cap < 0 else min(self.side_mm + step, cap))
                return (self.back_mm, self.side_mm) != before
        if self.through0 is not None:  # past the zone or the pulls: ease off
            self.back_mm = max(self.back_mm - 1000 * L.UNSTICK_CATCHUP * dt, 0.0)
            self.side_mm = float(np.sign(self.side_mm) * max(abs(self.side_mm) - 1000 * L.UNSTICK_SIDE_RATE * dt, 0.0))
        return (self.back_mm, self.side_mm) != before


class ThreadSteering:
    """Round 20, stage B's closed loop (the probe's and the demo's, run from the step hook): every L.STEER_EVERY control
    steps, from the tie's state, (1) while the left comes down to hold the strap, its waypoints move onto the strap's
    point tr.b_hold_s from the head; (2) while the right comes round behind the head (before its fingers can touch it:
    round 22), onto the head's centre; (3) while
    the left comes down to the tail, onto the strap's point tr.b_gft from its tip (sideways only: it lies on the
    console); (4) while it lines the tip up and pushes, sideways onto the tip's offset from the slot's centre line (at
    L.B_STEER_LO..L.B_STEER_CAP, only until tail is out), and on along the slot until tr.b_out is out (at most L.B_DEPTH_CAP).
    Each shift is frozen once that hand closes; (1)-(3) move a hand at most L.B_SNAP (round 21). A scripted demo may
    read the simulator's state; a policy under test gets only the cameras and its joints."""

    GROUPS = {"strap": ("left", L.THREAD_STRAP_LABELS, 2), "head": ("right", L.THREAD_HEAD_LABELS, 3),
              "tail": ("left", L.THREAD_TAIL_LABELS, 2), "steer": ("left", L.THREAD_STEER_LABELS, 4)}

    def __init__(self, tr: L.Trial, left: list, right: list) -> None:
        self.tr, self.arms = tr, {"left": left, "right": right}
        self.base = {arm: [np.asarray(w.pos, float) for w in ws] for arm, ws in self.arms.items()}
        self.targets = L.thread_targets(tr)
        self.shift = {k: np.zeros(3) for k in self.GROUPS}
        self.want = {k: np.zeros(3) for k in L.B_SNAP}  # round 21: each snap before its cap (the probe logs both)
        self.corr, self.depth_mm = np.zeros(3), 0.0
        self.tighten_mm = self.back_mm = self.side_mm = 0.0  # the probe logs these for every demo loop
        self.d_dir = self.targets["through"]

    @staticmethod
    def label(t: float, poses: list) -> str:
        for w in poses:
            if t < w.duration:
                return w.label
            t -= w.duration
        return "done"

    def update(self, step: int, through: float, tip_err, state=None) -> None:
        if step % L.STEER_EVERY or state is None:
            return
        t = step / CONTROL_HZ
        lab = {arm: self.label(t, ws) for arm, ws in self.arms.items()}
        poly = np.asarray(state.polyline, float)
        tr = self.tr
        if lab["left"] in L.THREAD_STRAP_LABELS[:2]:
            self._snap("strap", L.point_at_arc(poly, tr.b_hold_s) - self.targets["strap"])
        if lab["right"] in L.THREAD_HEAD_LABELS[:2]:  # round 22: frozen from "onto head" (job 96: a fingertip pushed
            self._snap("head", np.asarray(state.pos, float) - self.targets["regrasp"])  # the head and the snap chased it)
        if lab["left"] in L.THREAD_TAIL_LABELS[:2]:
            self._snap("tail", L.point_at_arc(poly, L.arc_length(poly) - tr.b_gft) - self.targets["tail"])
        if lab["left"] in L.THREAD_STEER_LABELS[:4]:
            if through <= 0.0:
                self.corr = np.clip(self.corr - L.STEER_GAIN * np.asarray(tip_err, float), L.B_STEER_LO,
                                    L.B_STEER_CAP)
            if lab["left"] in L.THREAD_PUSH_LABELS:
                self.depth_mm = float(np.clip(self.depth_mm + 1000 * L.DEPTH_GAIN * max(tr.b_out - through, 0.0), 0.0,
                                              1000 * L.B_DEPTH_CAP))
            self.shift["steer"] = self.corr + self.d_dir * self.depth_mm / 1000.0
        for key, (arm, labels, _) in self.GROUPS.items():
            for w, p in zip(self.arms[arm], self.base[arm]):
                if w.label in labels:
                    w.pos = tuple(float(v) for v in p + self.shift[key])

    def _snap(self, key: str, v) -> None:
        """Shift group ``key`` onto offset ``v``, within L.B_SNAP[key]: further off, the hand goes only that far."""
        self.want[key] = np.asarray(v, float)
        lo, hi = L.B_SNAP[key]
        self.shift[key] = np.clip(self.want[key], lo, hi)

    def log(self) -> dict:
        return {k: [round(1000 * float(x), 2) for x in v] for k, v in self.shift.items()} | {
            "depth_mm": round(self.depth_mm, 2)} | {
            f"{k}_want": [round(1000 * float(x), 2) for x in v] for k, v in self.want.items()}


DEMO = {"steering": None}  # the demo's closed loop while a demo plan plays (demo_plan sets it, stop_demo clears it)


def demo_plan(tr: L.Trial) -> dict:
    """Trial ``tr``'s plan as waypoints, its inserting hand on the closed loop (:class:`Steering`)."""
    plan = plan_waypoints(L.trial_plan(tr))
    if tr.stage == "thread":
        DEMO["steering"] = ThreadSteering(tr, plan["left"], plan["right"])
        return plan
    DEMO["steering"] = Steering(tr, plan["left"], plan["right"]) if (tr.steer or tr.depth or tr.tighten or
                                                                     tr.unstick) else None
    return plan


def stop_demo() -> None:
    DEMO["steering"] = None


def demo():
    """The scripted demo (yam_scripted plays it): probe trial TIE_TRIAL, by default m-demo (1.0 fresh in job 73), with
    the probe's closed loop on the inserting hand and the pulling hand's tighten (Steering)."""
    return demo_plan(L.trial(TRIAL))
