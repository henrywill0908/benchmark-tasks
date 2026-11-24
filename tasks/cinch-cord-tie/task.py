"""A TV hangs on the wall behind the console in front of the robot. Three cords hang straight down from the TV's
bottom edge into a white power strip lying on the console against the wall: a black TV power cord (7 mm), a black
HDMI cable (7.3 mm) and a white phone charger cable (3.5 mm), gathered in a bundle. A black nylon cable tie (a 370 mm
by 4.8 mm strap; its head 8 by 9 by 5.6 mm with a slot through it) lies flat on the console in front of the power
strip, to the right of the cords, its strap along the console's width with the head at the right-hand end. The strap
carries two white marks, 290 mm and 270 mm from its tip.
"""

# yam_arena hands the docstring above to the policy as task documentation, so it describes only what is on the table.
#
# The task is the default build, hang37 (round 38): the cords hanging under the TV, the tie lying free in front of
# them, no stand, nothing holding its head. tie_logic.py has every number. demo() plays the whole tie (h37-tie).
# Settings, for the probe (../../../cinch-cord-tie-gpu/probe.py); the defaults are the task:
#   TIE_BUILD=hang37|hang30|lift25|lift|lift30|jig|wrap|free|tight   hang37: the task's start. hang30: the same with a
#                          300 mm tie (45 mm short for the wrap). lift25, lift, lift30: rounds 19-30's start, the cords
#                          lying on the console. jig: the first submission's start (the tie looped round the cords, its
#                          head welded where a clip held it). wrap, free, tight: earlier probes on the hanging cords.
#   TIE_TALK=1             print the tail's length, the cords through the loop and the score once a second
#   TIE_RATCHET=latch|cam|brake|open   latch: the pawl held clear until LATCH_AT is out, then braked whenever the
#                          strap backs out. cam: the sprung pawl alone. brake: braked from the start. open: no ratchet.
#   TIE_PHYS=heavier        the tie's physics preset (tie_logic.PHYS: base, heavy, heavier)
#   TIE_TRIAL=h37-tie      which probe trial demo() plays (tie_logic.TRIALS and KEPT)

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
from labkit import (CableTie, Harness, HarnessPart, HarnessWire, Part, Pose, Prop, cylinder_mesh,  # noqa: E402
                    hold_joint, on_reset, on_step, ready_pose, release_joint, set_tie_yield, state)

SCENE = "station"
assert abs(TABLE_TOP[SCENE] - L.TOP) < 1e-9, "tie_logic.TOP must match the station's table top"
CONTROL_HZ = L.CONTROL_HZ
SUCCESS_HOLD_S = 1.0
INSTRUCTION = "Close the cable tie snugly around the three cords, then let go of it."
BUILD = os.environ.get("TIE_BUILD", "hang37")  # round 38: the hanging cords and a free 370 mm tie (jig: the first
#                                                 submission's pre-looped start, its head welded)
PHYS = os.environ.get("TIE_PHYS", L.HANG_PHYS if BUILD in L.HANG_BUILDS else "heavier")  # tie_logic.PHYS: the tie's
#   masses, damping, contact skin, friction by contact and the slot's lead-in (the earlier starts: heavier)
TRIAL = os.environ.get("TIE_TRIAL", L.KEYS_START if BUILD in L.LIFT_BUILDS else L.HANG_DEMO.get(BUILD, "m-demo"))
TALK = os.environ.get("TIE_TALK", "") not in ("", "0")
MARK_WHITE = (0.95, 0.95, 0.93)

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
        armature=ph["armature"], max_bend=L.MAX_BEND, friction=ph.get("friction", L.TIE_FRICTION),
        friction_combine=ph.get("combine", "average"), head_friction=ph.get("head_friction"),
        lead_in=ph.get("lead_in", 0.0), lead_in_walls=ph.get("lead_in_walls", ("back", "front", "side_pos", "side_neg")),
        lead_in_mass=ph.get("lead_in_mass", True), pawl_friction=ph.get("pawl_friction"), fixed=b.fixed,
        pawl_hinge=tuple(L.PAWL_HINGE), pawl_len=L.PAWL_LEN, pawl_radius=L.PAWL_R, pawl_width=min(L.PAWL_WIDTH, 2 * b.passage_y - 0.4 * L.MM),
        pawl_rest=L.PAWL_REST, pawl_mass=L.PAWL_MASS * k, pawl_spring=L.PAWL_SPRING * k, pawl_damping=L.PAWL_DAMPING * k,
        hold_stiffness=L.BRAKE_STIFFNESS, contact_offset=ph["offset"], body_linear_damping=ph["body_lin"],
        body_angular_damping=ph["body_ang"], yield_angle=math.radians(L.YIELD_DEG), decor=stand_decor(build),
        marks=tuple((b.tie_l - m, MARK_WHITE) for m in L.lift_marks(b.tie_l)) if build in L.LIFT_BUILDS + L.HANG_BUILDS else ())


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


def build_bundle(phys: str = PHYS) -> list:
    """The hanging cords. With a physics preset that gives them their own friction (tie_logic.PHYS, cord_friction):
    one fixed prop of three upright cylinders with that friction; else yam_arena's own cylinders."""
    ph = L.PHYS[phys]
    if "cord_friction" not in ph:
        return list(CORDS)
    mid, tall = (L.CORD_Z[0] + L.CORD_Z[1]) / 2, L.CORD_Z[1] - L.CORD_Z[0]
    at = np.mean([c.xy for c in L.CORDS], axis=0)
    parts = tuple(Part(c.name, cylinder_mesh(c.diameter / 2, tall, 24), at=(c.xy[0] - at[0], c.xy[1] - at[1], 0.0),
                       color=c.color, collides=True) for c in L.CORDS)
    return [Prop("cord_bundle", pos=(float(at[0]), float(at[1]), mid), size=(0.02, 0.02, tall), parts=parts,
                 friction=ph["cord_friction"], friction_combine="min")]


TIE = build_tie()
if BUILD in L.LIFT_BUILDS:  # round 19: the cords lie on the console (a Harness ending on the strip); no fixed rods, no
    #                            strip box (round 24: lift25 and lift30, the same with longer ties)
    CORDS_HARNESS = build_cords()
    OBJECTS = [*ROOM[:2], *hang_rods(), CORDS_HARNESS, TIE]
else:
    CORDS_HARNESS = None
    OBJECTS = [*ROOM, *build_bundle(), TIE]

# -- the ratchet and the release check (every control step) --------------------------------------------------------------
# score() sees object states only, so this hook records, each step, how close the nearest gripper point (TCP and pad
# tips, both arms) comes to the tie, and runs the ratchet's task-code part (TIE_RATCHET).
RATCHET = {"mode": os.environ.get("TIE_RATCHET", "latch"),
           "latch_at": L.LIFT_LATCH_AT if BUILD in L.LIFT_BUILDS + L.HANG_BUILDS else L.LATCH_AT}
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


def _tcps(environment) -> dict:
    """Each arm's TCP (world, env frame): the closed loop reads how far the head has slipped in the right's fingers."""
    from labkit import _np
    from yam_arena.runtime import tcp_world

    base = environment.unwrapped
    origin = _np(base.scene.env_origins)[0]
    return {arm: np.asarray(tcp_world(base.scene[f"{arm}_robot"]), float) - origin for arm in ("left", "right")}


def _ratchet(environment, s, through: float) -> None:
    mode = RATCHET["mode"]
    if mode == "open" and not SEEN["opened"]:  # the pawl turned flat against the passage's wall, out of the way
        hold_joint(environment, "tie", 0.0, "pawl")
        SEEN["opened"] = True
    if mode == "latch" and not SEEN["latched"]:  # round 6: the pawl clear of the passage until LATCH_AT is out
        if max(SEEN.get("s_best", 0.0), SEEN["s_max"], through) < RATCHET.get("latch_at", L.LATCH_AT):
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
            # Round 41 (job 126): let go under load, the strap ran back 1.3 mm within one control step and the pawl
            # was off it (40 degrees; on the strap it sits at 47 to 57) when the brake came on: aimed 3 degrees past
            # where the pawl then was, the brake held it clear and the whole tail ran out. It aims from where the
            # pawl last sat on the strap (the most closed of the last few steps).
            seat = max([s.pawl_angle, math.radians(L.BRAKE_SEAT), *SEEN.get("pawl_seen", ())])
            hold_joint(environment, "tie", min(seat + math.radians(L.BRAKE_DEG), L.PAWL_REST), "pawl")
            SEEN.update(brake_on=True, brake_s=through, brakes=SEEN["brakes"] + 1, brake_at=min(seat + math.radians(L.BRAKE_DEG), L.PAWL_REST))
        elif SEEN["brake_on"]:
            if through < SEEN["brake_s"] - L.BRAKE_SLIP and SEEN.get("brake_at", L.PAWL_REST) < L.PAWL_REST:
                # Round 45 (jobs 126, 129): with nylon's friction on the pawl the brake's 3 degrees did not stop a
                # strap the loop was pushing back out; while it still backs out the brake closes further, BRAKE_MORE
                # at a time (a real pawl's tooth does not let go at all).
                SEEN["brake_at"] = min(SEEN.get("brake_at", 0.0) + math.radians(L.BRAKE_MORE), L.PAWL_REST)
                hold_joint(environment, "tie", SEEN["brake_at"], "pawl")
                SEEN["brake_more"] = SEEN.get("brake_more", 0) + 1
            # Round 37 (job 121): a strap dragged back through the braked pawl (10 mm, by the loop pulled across the
            # cords) could never come forward again: the brake waited for the tail to pass where it had first braked,
            # and the best pull so far, s_max, re-braked it at once. A ratchet that has skipped holds where it now
            # is: the brake's mark follows the strap back, and letting go starts the best pull afresh from there.
            SEEN["brake_s"] = min(SEEN["brake_s"], through)
            if through > SEEN["brake_s"] + L.BRAKE_RELEASE:
                release_joint(environment, "tie", "pawl")
                SEEN.update(brake_on=False, releases=SEEN["releases"] + 1, s_max=through)
    SEEN["s_max"] = max(SEEN["s_max"], through)
    SEEN["s_best"] = max(SEEN.get("s_best", 0.0), through)
    if through > 0.0 and not SEEN["brake_on"] and not SEEN["opened"]:
        SEEN["pawl_seen"] = (*SEEN.get("pawl_seen", ())[-3:], float(s.pawl_angle))


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
        if CORDS_HARNESS is not None:  # the lying cords: which of them pass through the loop (for TIE_TALK and the
            #                            probe; score() reads the cords itself). Never in the closed loop's way.
            try:
                SEEN["enclosed"] = L.enclosed_lift(s.polyline, s.pos, s.quat_xyzw,
                                                   state(environment, "cords").polylines, TIE.passage_y)
            except Exception as err:  # noqa: BLE001
                SEEN["enclosed"], SEEN["enclosed_error"] = (), f"{type(err).__name__}: {err}"
        pts = np.vstack([np.asarray(s.pos, float)[None, :], np.asarray(s.polyline, float)])
        TRAIL.append(pts)
        window = int(round(SUCCESS_HOLD_S * CONTROL_HZ))
        del TRAIL[:-window - 1]
        if len(TRAIL) > window:
            span = len(TRAIL) - 1
            SEEN["drift_speed"] = float(np.max(np.linalg.norm(TRAIL[-1] - TRAIL[0], axis=1))) * CONTROL_HZ / span
        steering = DEMO["steering"]
        if steering is not None:
            try:
                SEEN["tcp"] = _tcps(environment)
            except Exception:  # noqa: BLE001 - the closed loop does without
                SEEN["tcp"] = None
        if steering is not None:  # the demo's closed loop, for the next control step (this hook runs after each)
            r = L.quat_xyzw_to_matrix(s.quat_xyzw)
            steering.update(SEEN["steps"] + 1, th.through, r @ np.array([th.tip_h[0] - L.SLOT_C, th.tip_h[1], 0.0]),
                            state=s)
        if TALK and SEEN["steps"] % CONTROL_HZ == 0:
            try:
                _talk(s, th)
            except Exception:  # noqa: BLE001 - a status line must never cost the trial its score
                pass
    except Exception as err:  # noqa: BLE001 - never credit a tie whose grippers couldn't be checked
        SEEN["hand_clear"], SEEN["error"] = 0.0, f"{type(err).__name__}: {err}"


def _talk(s, th) -> None:
    """TIE_TALK=1: one line a second for a person driving the arms (the Station page shows only the score)."""
    full, half = L.lift_marks(L.arc_length(s.polyline))
    drift = SEEN.get("drift_speed")
    moving = "?" if drift is None else f"{1000 * drift:.1f}"
    print(f"tie: t {SEEN['steps'] / CONTROL_HZ:5.1f} s  tail out {1000 * th.through:6.1f} mm (half mark "
          f"{1000 * half:.1f}, full mark {1000 * full:.1f})  cords in the loop {len(SEEN.get('enclosed', ()))}"
          f"  hands {1000 * SEEN['hand_clear']:5.1f} mm off  moving {moving} mm/s  latched {SEEN.get('latched')}",
          flush=True)


def score(objects) -> float:
    """0 unless every gripper point is at least 3 cm from the tie, the tie is at rest (no point of it moved more than
    1 cm over the last second; before a second has passed, its velocities under 1 cm/s) and it is closed
    round all three cords (the tail at least 2 mm out of the head's far side, every cord through the loop). Then 1.0
    if the strap's full mark has come out of the head, 0.5 if its half mark has, 0.25 otherwise (tie_logic.hang_score
    on the task's hanging cords, lift_score on the lying cords of rounds 19-30). On the first submission's start
    (TIE_BUILD=jig): 1.0 if snug (the loop at most 8 mm longer than the tightest the cords allow round the head) with
    the head at least 10 cm above the console; 0.5 within 25 mm; 0.25 otherwise."""
    s = objects.get("tie")
    if s is None or getattr(s, "polyline", None) is None:
        return 0.0
    drift = SEEN.get("drift_speed")  # the simulator's own record of how far it moved over the last second
    speed = drift if drift is not None else max(float(np.linalg.norm(s.lin_vel)), float(getattr(s, "link_speed_max", 0.0)))
    if CORDS_HARNESS is not None:  # the lying cords: closed round all three, then by the marks on the strap
        polys = getattr(objects.get("cords"), "polylines", None)
        if not polys:
            return 0.0
        return L.lift_score(s.polyline, s.pos, s.quat_xyzw, polys, SEEN["hand_clear"], speed, TIE.passage_y)
    if BUILD in L.HANG_BUILDS:  # round 31: the hanging cords without the stand, scored by the marks too
        return L.hang_score(s.polyline, s.pos, s.quat_xyzw, SEEN["hand_clear"], speed, TIE.passage_y)
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


class CinchSteering(ThreadSteering):
    """Round 26, the scripted finish's closed loop (tie_logic._cinch_tail), run from the step hook like ThreadSteering,
    whose three snaps (the strap hold, the head from behind, the tail's pick) it keeps. Then, from the tie's state:
    (1) while the left's pinch closes again over the bundle ("settle"), the tail's aim sets how far the wrist turns on
    (L.cinch_turn) and where "turn tail" ends, the tip L.CINCH_BACK short of the entry of the head as it will face it;
    (2) at "settle 2" the aim is read again and fixes the head's pitch (L.cinch_pitch; the right turns to it in "face
    tail"), the tail's vector from the fingers to the tip, and so the line-up and the push; (3) while lining up and
    pushing, the left shifts onto the tip's offset from the slot's line (until tail is out) and, at "push hold", on
    along the slot if less than L.CINCH_OUT is out; (4) as the right lets go of the head, its grip on the tail moves
    onto the strap's real point below the exit and the pull's line onto the slot's real axis; (5) as the left lets go
    of the strap, its grip moves onto the head's real centre; (6) during the pull the right's target stays at most
    L.CINCH_LEAD ahead of the strap's advance and stops once the tail is CINCH_PAST past the full mark. Every move is at
    most L.CINCH_SNAP from the plan. A scripted demo may read the simulator's state; a policy under test can't."""

    PAST = 0.004  # the pull stops with the tail this far past the full mark (it rests a link's joint back: 3.1 mm)

    def __init__(self, tr: L.Trial, left: list, right: list) -> None:
        super().__init__(tr, left, right)
        self.rot0 = {arm: [None if getattr(w, "rot", None) is None else np.asarray(w.rot, float) for w in ws]
                     for arm, ws in self.arms.items()}
        self.owner, self.when = {}, {}
        for arm, ws in self.arms.items():  # a "wait" holds the step before it: it moves with that step
            own, last, t, when = [], None, 0.0, {}
            for w in ws:
                last = last if w.label == "wait" else w.label
                own.append(last)
                when.setdefault(w.label, t)
                t += w.duration
            self.owner[arm], self.when[arm] = own, when
        self.hc0 = np.asarray(tr.b_head, float)  # the head's centre as the right is told to hold it
        self.frame = L.cinch_frame(self.hc0, tr.b_insert_pitch)
        self.phi = tr.b_feed_phi + tr.b_turn
        self.turn_deg, self.pitch = float(tr.b_turn), float(tr.b_insert_pitch)
        self.aims = [None, None]
        self.tail_vec = self.frame["d"] * tr.b_gft  # from the strap's point in the left's fingers to the tip
        self.steer, self.depth = np.zeros(3), 0.0
        self.grip = None  # the right's (TCP, rotation) on the tail
        self.pull_dir = self.frame["d"].copy()
        self.through0, self.freeze, self.back = None, None, 0.0
        self.moved = {}

    # -- helpers -------------------------------------------------------------------------------------------------
    def _set(self, arm: str, labels, pos, rot=None) -> None:
        for w, own in zip(self.arms[arm], self.owner[arm]):
            if own in labels and w.pos is not None:
                w.pos = tuple(float(v) for v in pos)
                if rot is not None:
                    w.rot = np.asarray(rot, float)

    def _base(self, arm: str, label: str):
        i = next(k for k, w in enumerate(self.arms[arm]) if w.label == label)
        return self.base[arm][i], self.rot0[arm][i]

    def _near(self, arm: str, label: str, pos, key: str) -> np.ndarray:
        """``pos`` at most L.CINCH_SNAP from the plan's ``label``; the move asked for is kept for the log."""
        v = np.asarray(pos, float) - self._base(arm, label)[0]
        self.moved[key] = [round(1000 * float(x), 1) for x in v]
        n = float(np.linalg.norm(v))
        return self._base(arm, label)[0] + (v if n <= L.CINCH_SNAP else v * (L.CINCH_SNAP / n))

    def _snap(self, key: str, v) -> None:
        if key != "tail":
            return super()._snap(key, v)
        self.want[key] = np.asarray(v, float)  # the tail's pick: the guess of where it lies is rougher here
        self.shift[key] = np.clip(self.want[key], [-0.006, -0.045, 0.0], [0.006, 0.045, 0.0])

    # -- the steps ---------------------------------------------------------------------------------------------------
    def _plan_turn(self, tip, aim: float, hc) -> None:
        tr = self.tr
        at, rot = self._base("left", "settle")
        held = at + L.B_TAIL_DEPTH * rot[:, 2]
        g = float(np.clip(np.linalg.norm(tip - held), tr.b_gft - 0.02, tr.b_gft + 0.02))
        self.aims[0] = round(float(aim), 1)
        self.turn_deg = L.cinch_turn(tr, aim)
        self.phi = tr.b_feed_phi + self.turn_deg
        a = math.radians(aim + self.turn_deg)
        u = np.array([0.0, -math.cos(a), -math.sin(a)])
        self.pitch = L.cinch_pitch(aim + self.turn_deg)
        self.frame = L.cinch_frame(hc, self.pitch)
        self.tail_vec = u * g
        r = L.thread_left_rot(self.phi)
        pos = self.frame["entry"] - self.frame["d"] * L.CINCH_BACK - self.tail_vec - L.B_TAIL_DEPTH * r[:, 2]
        self._set("left", ("turn tail", "settle 2"), self._near("left", "turn tail", pos, "turn"), r)
        self._feed()

    def _commit(self, tip, u, aim: float, hc) -> None:
        tr = self.tr
        w = next(x for x in self.arms["left"] if x.label == "settle 2")
        held = np.asarray(w.pos, float) + L.B_TAIL_DEPTH * np.asarray(w.rot, float)[:, 2]
        v = np.asarray(tip, float) - held
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            self.tail_vec = v * (float(np.clip(n, tr.b_gft - 0.02, tr.b_gft + 0.02)) / n)
        self.aims[1] = round(float(aim), 1)
        self.pitch = L.cinch_pitch(aim)
        self.frame = L.cinch_frame(hc, self.pitch)
        tilt = L.B_GRIP_TILT + self.pitch
        self._set("right", ("face tail", "release head"), L.back_tcp(self.hc0, tilt), L.back_rot(tilt))
        self._feed()

    def _feed(self) -> None:
        f, r = self.frame, L.thread_left_rot(self.phi)
        off = self.tail_vec + L.B_TAIL_DEPTH * r[:, 2]
        align = f["entry"] - f["d"] * L.B_TIP_GAP + self.steer - off
        pushed = f["entry"] + f["d"] * (L.HEAD[2] + L.CINCH_OUT + self.depth) + self.steer - off
        self._set("left", ("align",), self._near("left", "align", align, "align"), r)
        pushed = self._near("left", "push", pushed, "push")
        self._set("left", ("push", "push hold", "let go strap 2"), pushed, r)
        self._set("left", ("up from strap 2",), pushed - 0.035 * r[:, 2], r)

    def _tail_grip(self, state, through: float) -> None:
        tr = self.tr
        rm = L.quat_xyzw_to_matrix(state.quat_xyzw)
        d = rm @ np.array([0.0, 0.0, -1.0])
        f = {"exit": np.asarray(state.pos, float) + rm @ np.array([L.SLOT_C, 0.0, L.EXIT_Z]), "d": d}
        pos, rot = L.cinch_tail_grip(f, tr.b_tail_at, tr.b_tail_tilt)
        along = tr.b_tail_at + L.CINCH_PAD
        poly = np.asarray(state.polyline, float)
        if through >= along + 0.004:  # the strap's own point that far below the exit (the tail may hang off the axis)
            pos = L.point_at_arc(poly, L.arc_length(poly) - through + along) - L.GRIP_DEPTH * rot[:, 2]
        pos = self._near("right", "to the tail 2", pos, "tail_grip")
        self.grip, self.pull_dir = (pos, rot), d / max(float(np.linalg.norm(d)), 1e-9)
        self._pull_targets()

    def _pull_targets(self) -> None:
        tr = self.tr
        pos, rot = self.grip
        lift = np.array([0.0, 0.0, tr.b_pull_lift])
        back = -self.pull_dir * self.back
        self._set("right", ("to the tail 2", "take tail 2"), pos, rot)
        self._set("right", ("raise tail",), pos + lift + back, rot)
        drawn = pos + lift + self.pull_dir * tr.b_pull + back
        self._set("right", ("pull", "pull hold", "let go tail 2"), drawn, rot)
        self._set("right", ("back off tail 2",), drawn - 0.05 * rot[:, 2], rot)

    def _head_grip(self, state) -> None:
        tr = self.tr
        pos, rot = L.cinch_head_grip({"hc": np.asarray(state.pos, float)}, tr.b_head_phi)
        pos = self._near("left", "down to head", pos, "head_grip")
        up = np.array([0.0, 0.0, 1.0])
        self._set("left", ("over head",), pos - 0.045 * rot[:, 2], rot)
        self._set("left", ("down to head", "grip head 2"), pos, rot)
        self._set("left", ("raise head", "hold head"), pos + tr.b_pull_lift * up, rot)
        self._set("left", ("lower head", "let go head 2"), pos + 0.010 * up, rot)
        self._set("left", ("up from head 2",), pos + 0.010 * up - 0.050 * rot[:, 2], rot)

    def _pull(self, t: float, lab: str, through: float) -> None:
        tr = self.tr
        if self.grip is None:
            self.grip = self._base("right", "take tail 2")
        if self.through0 is None:
            self.through0 = through
        w = next(x for x in self.arms["right"] if x.label == "pull")
        s = min(max((t - self.when["right"]["pull"]) / w.duration, 0.0), 1.0)
        planned = tr.b_pull * s * s * (3 - 2 * s) if lab == "pull" else tr.b_pull
        hand = planned - self.back  # how far along the pull the hand's target is now
        target = L.lift_marks(L.BUILDS[tr.build].tie_l)[0] + self.PAST
        if self.freeze is None and through >= target:
            self.freeze = hand
        allowed = self.freeze if self.freeze is not None else (through - self.through0) + L.CINCH_LEAD
        self.back = max(planned - allowed, 0.0)
        self._pull_targets()

    def update(self, step: int, through: float, tip_err, state=None) -> None:
        if step % L.STEER_EVERY or state is None:
            return
        t = step / CONTROL_HZ
        lab = {arm: self.label(t, ws) for arm, ws in self.arms.items()}
        poly = np.asarray(state.polyline, float)
        tr = self.tr
        # rounds 20-22's snaps, as ThreadSteering applies them
        if lab["left"] in L.THREAD_STRAP_LABELS[:2]:
            self._snap("strap", L.point_at_arc(poly, tr.b_hold_s) - self.targets["strap"])
        if lab["right"] in L.THREAD_HEAD_LABELS[:2]:
            self._snap("head", np.asarray(state.pos, float) - self.targets["regrasp"])
        if lab["left"] in L.THREAD_TAIL_LABELS[:2]:
            self._snap("tail", L.point_at_arc(poly, L.arc_length(poly) - tr.b_gft) - self.targets["tail"])
        for key in ("strap", "head", "tail"):
            arm, labels, _ = self.GROUPS[key]
            for w, p in zip(self.arms[arm], self.base[arm]):
                if w.label in labels:
                    w.pos = tuple(float(v) for v in p + self.shift[key])
        tip, u, aim = L.tail_aim(poly)
        hc = np.asarray(state.pos, float)
        if lab["left"] == "settle":
            self._plan_turn(tip, aim, hc)
        elif lab["left"] == "settle 2":
            self._commit(tip, u, aim, hc)
        elif lab["left"] in L.CINCH_LEFT_STEER:
            if through <= 0.0:
                self.steer = np.clip(self.steer - L.STEER_GAIN * np.asarray(tip_err, float), -L.CINCH_STEER_CAP,
                                     L.CINCH_STEER_CAP)
            if lab["left"] == "push hold" and through < L.CINCH_OUT - 0.002:
                self.depth = float(np.clip(self.depth + L.DEPTH_GAIN * (L.CINCH_OUT - through), 0.0, L.CINCH_DEPTH_CAP))
            self._feed()
        if lab["right"] == "release head":
            self._tail_grip(state, through)
        if lab["left"] in ("let go strap 2", "up from strap 2"):
            self._head_grip(state)
        if lab["right"] in ("pull", "pull hold"):
            self._pull(t, lab["right"], through)

    def log(self) -> dict:
        return super().log() | {
            "aim_deg": self.aims, "turn_deg": round(self.turn_deg, 1), "pitch_deg": round(self.pitch, 1),
            "tail_vec_mm": [round(1000 * float(x), 1) for x in self.tail_vec],
            "cinch_steer_mm": [round(1000 * float(x), 2) for x in self.steer], "cinch_depth_mm": round(1000 * self.depth, 2),
            "pull_back_mm": round(1000 * self.back, 1), "pull_from_mm": None if self.through0 is None else round(
                1000 * self.through0, 1), "pull_frozen": self.freeze is not None, "moved_mm": dict(self.moved)}


class WedgeSteering(CinchSteering):
    """Round 29's closed loop (tie_logic._wedge_tail), run from the step hook like ThreadSteering, whose three snaps
    (the strap hold, the head from behind, the tail's pick: that one carries through the arch, the lift and "settle")
    it keeps. Then, from the tie's state: (1) at "settle" the tail's vector from the left's fingers to the tip gives
    the angle the strap locked at in the pinch, and so the wrist's tilt that points the tail down the slot of the head
    as the right really holds it (L.wedge_feed): "over" and everything after are rewritten; (2) at "settle 2", over
    the bundle with the wrist turned, the same is read again and the way down rewritten once more; (3) from "above
    head" to "push hold" the left shifts onto the tip's offset from the slot's line (until tail is out), and during
    the push on along the slot until tr.b_out is out (at most L.B_DEPTH_CAP)."""

    GROUPS = dict(ThreadSteering.GROUPS, tail=("left", L.WEDGE_TAIL_LABELS, 2))

    def __init__(self, tr: L.Trial, left: list, right: list) -> None:
        super().__init__(tr, left, right)
        self.entry = np.asarray(self.targets["entry"], float)
        self.tail_vec, self.vec_phi = L.wedge_tail_vec(tr), 90.0
        self.phi_over = L.wedge_feed(tr, self.entry, self.d_dir, self.tail_vec, 90.0)["phi"]
        self.reads = [None, None]

    def _feed(self, over: bool = False) -> dict:
        f = L.wedge_feed(self.tr, self.entry, self.d_dir, self.tail_vec, self.vec_phi, self.corr, self.depth_mm / 1000.0)
        if over:
            self.phi_over = f["phi"]
            self._set("left", ("over", "settle 2"), f["over"], f["rot"])
        self._set("left", ("above head",), f["above head"], f["rot"])
        self._set("left", ("align",), f["align"], f["rot"])
        self._set("left", ("push", "push hold", "let go tail"), f["push"], f["rot"])
        self._set("left", ("back off tail",), f["back off tail"], f["rot"])
        return f

    def _read(self, i: int, label: str, phi_at: float, state) -> None:
        tr = self.tr
        w = next(x for x in self.arms["left"] if x.label == label)
        held = np.asarray(w.pos, float) + L.B_TAIL_DEPTH * np.asarray(w.rot, float)[:, 2]
        v = np.asarray(state.polyline, float)[-1] - held
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            self.tail_vec = v * (float(np.clip(n, tr.b_gft - 0.015, tr.b_gft + 0.015)) / n)
            self.vec_phi = float(phi_at)
        rm = L.quat_xyzw_to_matrix(state.quat_xyzw)
        self.entry = np.asarray(state.pos, float) + rm @ np.array([L.SLOT_C, 0.0, L.ENTRY_Z])
        self.d_dir = rm @ np.array([0.0, 0.0, -1.0])
        f = self._feed(over=(i == 0))
        self.reads[i] = {"offset_deg": round(f["offset"], 1), "phi_deg": round(f["phi"], 1),
                         "tail_mm": round(1000 * n, 1), "slot_deg": round(L.heading_deg(self.d_dir), 1)}

    def update(self, step: int, through: float, tip_err, state=None) -> None:
        if step % L.STEER_EVERY or state is None:
            return
        t = step / CONTROL_HZ
        lab = {arm: self.label(t, ws) for arm, ws in self.arms.items()}
        poly = np.asarray(state.polyline, float)
        tr = self.tr
        if lab["left"] in L.THREAD_STRAP_LABELS[:2]:
            self._snap("strap", L.point_at_arc(poly, tr.b_hold_s) - self.targets["strap"])
        if lab["right"] in L.THREAD_HEAD_LABELS[:2]:
            self._snap("head", np.asarray(state.pos, float) - self.targets["regrasp"])
        if lab["left"] in L.WEDGE_TAIL_LABELS[:2]:
            self._snap("tail", L.point_at_arc(poly, L.arc_length(poly) - tr.b_gft) - self.targets["tail"])
        for key in ("strap", "head", "tail"):
            arm, labels, _ = self.GROUPS[key]
            for w, p in zip(self.arms[arm], self.base[arm]):
                if w.label in labels:
                    w.pos = tuple(float(v) for v in p + self.shift[key])
        if lab["left"] == "settle":
            self._read(0, "settle", 90.0, state)
        elif lab["left"] == "settle 2":
            self._read(1, "settle 2", self.phi_over, state)
        elif lab["left"] in L.WEDGE_FEED_LABELS[:4]:
            if through <= 0.0:
                self.corr = np.clip(self.corr - L.STEER_GAIN * np.asarray(tip_err, float), L.B_STEER_LO, L.B_STEER_CAP)
            if lab["left"] in L.THREAD_PUSH_LABELS:
                self.depth_mm = float(np.clip(self.depth_mm + 1000 * L.DEPTH_GAIN * max(tr.b_out - through, 0.0), 0.0,
                                              1000 * L.B_DEPTH_CAP))
            self.shift["steer"] = self.corr
            self._feed()
        if tr.b_draw:  # round 30
            if lab["right"] in ("release head", "back off head", "to the tail 2"):
                self._tail_grip(state, through)
            elif lab["right"] in ("pull", "pull hold"):
                self._pull(t, lab["right"], through)

    # -- round 30: the draw --------------------------------------------------------------------------------------
    def _tail_grip(self, state, through: float) -> None:
        """The right's grip on the tail, onto the head's real exit and the strap's real point below it."""
        tr = self.tr
        rm = L.quat_xyzw_to_matrix(state.quat_xyzw)
        f = {"exit": np.asarray(state.pos, float) + rm @ np.array([L.SLOT_C, 0.0, L.EXIT_Z]),
             "d": rm @ np.array([0.0, 0.0, -1.0])}
        pos, rot = L.cinch_tail_grip(f, tr.b_tail_at, tr.b_tail_tilt)
        along = tr.b_tail_at + L.CINCH_PAD
        poly = np.asarray(state.polyline, float)
        if through >= along + 0.004:
            pos = L.point_at_arc(poly, L.arc_length(poly) - through + along) - L.GRIP_DEPTH * rot[:, 2]
        self.grip = (self._near("right", "take tail 2", pos, "tail_grip"), rot)
        self._pull_targets()

    def _pull_targets(self, travel: float | None = None) -> None:
        tr = self.tr
        pos, rot = self.grip
        self._set("right", ("to the tail 2",), pos - 0.035 * rot[:, 2], rot)
        self._set("right", ("onto tail 2", "take tail 2"), pos, rot)
        drawn = pos + L.draw_dir(tr) * (tr.b_pull if travel is None else travel)
        self._set("right", ("pull", "pull hold", "let go tail 2"), drawn, rot)
        self._set("right", ("back off tail 2",), drawn - 0.05 * rot[:, 2], rot)

    def _pull(self, t: float, lab: str, through: float) -> None:
        """The draw stops where the hand is once the full mark is PAST out of the exit, or once the tie is tight: the
        hand 12 mm further on with neither the tail nor the head following."""
        tr = self.tr
        if self.grip is None:
            self.grip = self._base("right", "take tail 2")
        w = next(x for x in self.arms["right"] if x.label == "pull")
        s = min(max((t - self.when["right"]["pull"]) / w.duration, 0.0), 1.0)
        sm = s * s * (3 - 2 * s) if lab == "pull" else 1.0
        hand = tr.b_pull * sm  # how far the plan has the hand by now
        if self.through0 is None:
            self.through0, self.mark = through, (hand, through)
        if through > self.mark[1] + 0.001:
            self.mark = (hand, through)
        target = L.lift_marks(L.BUILDS[tr.build].tie_l)[0] + self.PAST
        tight = through > self.through0 + 0.020 and hand - self.mark[0] >= 0.012
        if self.freeze is None and (through >= target or tight):
            self.freeze, self.why = hand, "mark" if through >= target else "tight"
        if self.freeze is None:
            self._pull_targets()
        else:  # the waypoint the blend reaches the frozen place from: the hand stays where it stopped
            self.back = tr.b_pull - self.freeze
            self._pull_targets(self.freeze / max(sm, 1e-3))

    def log(self) -> dict:
        return ThreadSteering.log(self) | {
            "wedge": list(self.reads), "tail_vec_mm": [round(1000 * float(x), 1) for x in self.tail_vec],
            "phi_over_deg": round(self.phi_over, 1), "pull_from_mm": None if self.through0 is None else round(
                1000 * self.through0, 1), "pull_stopped": getattr(self, "why", None),
            "pull_back_mm": round(1000 * self.back, 1), "moved_mm": dict(self.moved)}


class HangSteering:
    """Rounds 31-32, the hanging cords' closed loop (tie_logic._hang_plan, _hang_thread), run from the step hook. From
    the tie's state: (1) while the left comes down onto the tail, its grip goes onto the strap's real point tr.h_gft
    from the tip and turns to close across the strap as it really runs there; (2) at "settle 3" the tail's real
    vector from the left's fingers to the tip and the head's real slot give the turn of the left's wrist toward the
    slot's line (at most TURN, and the tail no further round than tr.h_tail_max: its wrist joint) and the places of
    the line-up and the push, the tip on the entry's real centre, the push along the tail's own line; (3) from
    "above head" to "push hold" the left shifts onto the tip's offset from the slot's line (until tail is out), and
    during the push on along its line until tr.h_out is out (at most DEPTH); (4) as the
    right lets go of the head, its grip on the tail goes onto the strap's real point below the real exit; (5) the
    draw stops where the hand is once the full mark is PAST out, or once the tie is tight (the hand 12 mm on with no
    more tail coming). Every move is at most SNAP (the second take: SNAP2) from the plan. A scripted demo may read
    the simulator's state; a policy under test can't."""

    SNAP, SNAP2, TURN, DEPTH, STEER_CAP, PAST = 0.030, 0.060, 35.0, 0.006, 0.012, 0.004
    PIVOT_OK, SLIP, EASE, SLOT_PAST = 12.0, 0.003, 0.018, 5.0  # round 33, drawing the strap taut: stop once the head
    #   lies within PIVOT_OK degrees of the strap's line from the cords, or has slipped SLIP in the right's fingers.
    #   Round 35 (job 119: the hand's target stayed 25 mm out past where the taut strap let the hand go, so it pulled
    #   on through the turn and the push, the head would not turn with it, and the push blew the simulator up): the
    #   hand stops where it really is, then brings the head EASE back toward the cords: slack. The turn back aims
    #   the slot SLOT_PAST degrees round of the tail's planned line (the strap sat 5 degrees round in the left's grip).

    def __init__(self, tr: L.Trial, left: list, right: list) -> None:
        self.tr, self.arms = tr, {"left": left, "right": right}
        self.b = L.BUILDS[tr.build]
        self.o = L.hang_pick(self.b)[2]
        self.base, self.owner, self.when = {}, {}, {}
        for arm, ws in self.arms.items():
            base, own, when, last, t = {}, [], {}, None, 0.0
            for w in ws:
                last = last if w.label == "wait" else w.label
                own.append(last)
                when.setdefault(w.label, t)
                if w.pos is not None:
                    base.setdefault(w.label, (np.asarray(w.pos, float), np.asarray(w.rot, float)))
                t += w.duration
            self.base[arm], self.owner[arm], self.when[arm] = base, own, when
        self.corr, self.depth_mm, self.tighten_mm, self.back_mm, self.side_mm = np.zeros(3), 0.0, 0.0, 0.0, 0.0
        self.moved, self.reads = {}, {}
        self.aim = None  # (rotation, tail vector, entry, way through) once read at "settle 3"
        self.taut, self.slip0 = None, None  # round 33: where the draw at the front stopped
        self.through0, self.mark, self.freeze, self.why, self.leg = None, None, None, None, None
        self.r1 = L.RY_ROLL @ L.hang_pick(self.b)[1]  # a hand pointing level at the wall, its fingers above and below
        self.o_left = np.asarray(L.HOLD_O, float)  # round 40: the head in the left's fingers (read once it holds it)
        self.held = None  # the draw with the head held: (where the hand started, where it was to end, where it stopped)

    label = staticmethod(ThreadSteering.label)

    def _set(self, arm: str, labels, pos, rot=None) -> None:
        for w, own in zip(self.arms[arm], self.owner[arm]):
            if own in labels and w.pos is not None:
                w.pos = tuple(float(v) for v in pos)
                if rot is not None:
                    w.rot = np.asarray(rot, float)

    def _near(self, arm: str, label: str, pos, key: str, cap: float) -> np.ndarray:
        base = self.base[arm][label][0]
        v = np.asarray(pos, float) - base
        self.moved[key] = [round(1000 * float(x), 1) for x in v]
        k = float(np.linalg.norm(v))
        return base + (v if k <= cap else v * (cap / k))

    def _tail(self, poly):
        """The strap's real point tr.h_gft from the tip and its level direction there (toward the tip)."""
        n = L.arc_length(poly)
        g = self.tr.h_gft
        d = L.point_at_arc(poly, n - g + 0.012) - L.point_at_arc(poly, n - g - 0.012)
        d[2] = 0.0
        k = float(np.linalg.norm(d))
        return L.point_at_arc(poly, n - g), (None if k < 1e-6 else d / k)

    def _take(self, poly) -> None:
        p, d = self._tail(poly)
        if d is None:
            return
        rot = L.face_rot(d)
        at = self._near("left", "take tail 2", L.face_at(p, rot), "tail", self.SNAP)
        self._set("left", ("to the tail 2",), at + np.array([0.0, 0.0, 0.06]), rot)
        self._set("left", ("down to the tail 2", "take tail 2", "hold tail"), at, rot)

    def _read_aim(self, state) -> None:
        tr = self.tr
        w = next(x for x in self.arms["left"] if x.label == "settle 3")
        rot0 = np.asarray(w.rot, float)
        grip = np.asarray(w.pos, float) + L.GRIP_DEPTH * rot0[:, 2]
        tv = np.asarray(state.polyline, float)[-1] - grip
        rm = L.quat_xyzw_to_matrix(state.quat_xyzw)
        entry = np.asarray(state.pos, float) + rm @ np.array([L.SLOT_C, 0.0, L.ENTRY_Z])
        d = rm @ np.array([0.0, 0.0, -1.0])
        want = (L.heading_xy(d) - L.heading_xy(tv) + 180.0) % 360.0 - 180.0
        hand = L.heading_xy(rot0[:, 1])  # the way the left's hand holds the strap (the tail, but for what it slipped)
        room = (tr.h_tail_max - hand) % 360.0  # how much further round its wrist can go
        room = room if room < 180.0 else 0.0
        turn = float(np.clip(want, -self.TURN, min(self.TURN, room)))
        rz = L.rot_z(turn)
        tv = rz @ tv
        line = np.array([tv[0], tv[1], 0.0])
        self.aim = (rz @ rot0, tv, entry, line / float(np.linalg.norm(line)))
        self.reads["aim"] = {"turn_deg": round(turn, 1), "wanted_deg": round(want, 1),
                             "tail_mm": round(1000 * float(np.linalg.norm(tv)), 1),
                             "tail_deg": round(L.heading_xy(tv) % 360.0, 1), "slot_deg": round(L.heading_xy(d) % 360.0, 1),
                             "hand_deg": round((hand + turn) % 360.0, 1),
                             "slot_dz": round(float(d[2]), 3), "tail_dz_mm": round(1000 * float(tv[2]), 1)}
        self._feed()

    def _feed(self) -> None:
        tr = self.tr
        rot, tv, e, d = self.aim
        back = tv + L.GRIP_DEPTH * rot[:, 2]  # from the TCP to the tip
        pushed = e + d * (L.HEAD[2] + tr.h_out + self.depth_mm / 1000.0) + self.corr - back
        self._set("left", ("above head",), e - d * (L.B_TIP_GAP + L.B_ABOVE) + self.corr - back, rot)
        self._set("left", ("align",), e - d * L.B_TIP_GAP + self.corr - back, rot)
        self._set("left", ("push", "push hold", "let go tail 3"), pushed, rot)
        self._set("left", ("back off tail 3",), pushed + np.array([0.0, 0.0, 0.004]), rot)

    def _tauten(self, t: float, state) -> None:
        """Round 33: the right draws the head out along the strap's line until the strap is taut."""
        rb = self.base["right"]
        p0, (p1, rot) = rb["hold head"][0], rb["tauten"]
        w = next(x for x in self.arms["right"] if x.label == "tauten")
        s = min(max((t - self.when["right"]["tauten"]) / w.duration, 0.0), 1.0)
        sm = s * s * (3 - 2 * s)
        now = p0 + (p1 - p0) * sm
        if self.taut is None:
            rm = L.quat_xyzw_to_matrix(state.quat_xyzw)
            leg, length = L.hang_leg(state.pos)
            gap = (L.heading_xy(rm[:, 0]) - leg + 180.0) % 360.0 - 180.0
            tcp = (SEEN.get("tcp") or {}).get("right")
            slip = None if tcp is None else float(np.linalg.norm(np.asarray(state.pos, float) - (tcp + rot @ self.o)))
            if self.slip0 is None:
                self.slip0 = slip
            slipped = 0.0 if slip is None or self.slip0 is None else slip - self.slip0
            if sm > 0.05 and (abs(gap) <= self.PIVOT_OK or slipped >= self.SLIP or s >= 1.0):
                step = (p1 - p0) / max(float(np.linalg.norm(p1 - p0)), 1e-9)
                self.taut = now if tcp is None else np.asarray(tcp, float)  # where the hand really is
                stop = np.asarray(state.pos, float) - step * self.EASE - rot @ self.o  # the head EASE back: slack
                self._set("right", ("ease back", "open head", "yaw out", "close head", "settle 2b", "turn head",
                                    "hold head 2", "release head"), stop)
                self.reads["taut"] = {"drawn_mm": round(1000 * float(np.linalg.norm(now - p0)), 1),
                                      "why": "turned" if abs(gap) <= self.PIVOT_OK else "slip" if slipped >= self.SLIP
                                      else "end", "gap_deg": round(gap, 1), "leg_mm": round(1000 * length, 1),
                                      "slip_mm": round(1000 * slipped, 2)}
        if self.taut is not None:  # the waypoint the blend reaches the stopped place from: the hand stays there
            self._set("right", ("tauten",), p0 + (self.taut - p0) / max(sm, 1e-3))

    def _regrip(self, state) -> None:
        """Round 34: the right's second grip on the head goes onto the head's real centre (let go, it may shift)."""
        rot = self.base["right"]["close head"][1]
        pos = self._near("right", "close head", np.asarray(state.pos, float) - rot @ self.o, "head 2", self.SNAP2)
        self._set("right", ("yaw out", "close head", "settle 2b"), pos, rot)
        self.reads["regrip"] = {"head_mm": [round(1000 * float(x), 1) for x in state.pos]}

    def _turn_head(self, state) -> None:
        """Round 33: how far the right turns its hand back so the head's slot lies along the tail's line."""
        tr = self.tr
        rm = L.quat_xyzw_to_matrix(state.quat_xyzw)
        d = rm @ np.array([0.0, 0.0, -1.0])
        want = (L.heading_xy(d) - (tr.h_tail_deg + self.SLOT_PAST) + 180.0) % 360.0 - 180.0
        yaw = tr.h_carry_yaw + tr.h_regrip_yaw  # how far out the hand is turned now
        turn = float(np.clip(want, 0.0, yaw))
        rot = L.rot_z(yaw - turn) @ self.r1
        w = next(x for x in self.arms["right"] if x.label == "settle 2b")
        head = np.asarray(w.pos, float) + np.asarray(w.rot, float) @ self.o  # the head's centre as the hand holds it
        self._set("right", ("turn head", "hold head 2", "release head"), head - rot @ self.o, rot)
        self.reads["turn"] = {"turn_deg": round(turn, 1), "wanted_deg": round(want, 1),
                              "slot_deg": round(L.heading_xy(d) % 360.0, 1)}

    def _stub(self, state, through: float) -> None:
        tr = self.tr
        rm = L.quat_xyzw_to_matrix(state.quat_xyzw)
        d = rm @ np.array([0.0, 0.0, -1.0])
        at = np.asarray(state.pos, float) + rm @ np.array([L.SLOT_C, 0.0, L.EXIT_Z]) + d * tr.h_stub_at
        poly = np.asarray(state.polyline, float)
        if through >= tr.h_stub_at + 0.006:  # the strap's own point that far below the exit
            at = L.point_at_arc(poly, L.arc_length(poly) - through + tr.h_stub_at)
        rot = np.asarray(next(x for x in self.arms["right"] if x.label == "release head").rot, float)
        pos = self._near("right", "take stub", at - rot @ self.o, "stub", self.SNAP2)
        self._set("right", ("to the stub", "take stub"), pos, rot)
        self.reads["stub"] = {"through_mm": round(1000 * through, 1), "tail_deg": round(L.heading_xy(d), 1)}

    def _head_take(self, state) -> None:
        """Round 40: the left's grip on the head goes onto the head's real centre, its hand pointing the strap's real
        way out of the head (level), its pads flush with the exit face."""
        rm = L.quat_xyzw_to_matrix(state.quat_xyzw)
        rot = L.rot_z(L.heading_xy(rm[:, 0])) @ self.r1
        at = self._near("left", "take head 2", np.asarray(state.pos, float) - rot @ L.HOLD_O, "head 3", self.SNAP2)
        self._set("left", ("to the head 2",), at - 0.045 * rot[:, 2], rot)
        self._set("left", ("onto the head", "take head 2", "hold head 3"), at, rot)
        self.reads["head 3"] = {"root_deg": round(L.heading_xy(rm[:, 0]), 1),
                                "head_mm": [round(1000 * float(x), 1) for x in state.pos]}

    def _carry(self, state) -> None:
        """Round 40: where the head really sits in the left's fingers (it may have shifted as they closed): the carry
        puts the head, not the hand, at its place by the cords."""
        tcp = (SEEN.get("tcp") or {}).get("left")
        if tcp is None:
            return
        w = next(x for x in self.arms["left"] if x.label == "hold head 3")
        o = np.asarray(w.rot, float).T @ (np.asarray(state.pos, float) - np.asarray(tcp, float))
        d = o - L.HOLD_O
        k = float(np.linalg.norm(d))
        self.o_left = L.HOLD_O + (d if k <= 0.006 else d * (0.006 / k))
        tight = L.hang_tight(self.tr, self.b, L.TOP + self.tr.h_low)
        rot = self.base["left"]["carry head"][1]
        at = tight["head"] - rot @ self.o_left
        self._set("left", ("carry head", "set head", "square head", "at the cords", "let go head 2"), at)
        self._set("left", ("back off head 2",), at - 0.045 * rot[:, 2])
        self.reads["carry"] = {"head_in_hand_mm": [round(1000 * float(x), 1) for x in o]}

    def _square(self, state) -> None:
        """Round 43: at the cords the head has turned in the left's fingers (job 126: 11 degrees); the wrist turns by
        what it is off its heading (to tr.h_square_max at most), about the head's own place."""
        tr = self.tr
        rm = L.quat_xyzw_to_matrix(state.quat_xyzw)
        real = L.heading_xy(rm[:, 0])
        want = tr.h_tight[2]
        hand = L.heading_xy(self.base["left"]["carry head"][1][:, 2])
        to = float(np.clip(hand + (want - real + 180.0) % 360.0 - 180.0, hand - 30.0, tr.h_square_max))
        rot = L.rot_z(to) @ self.r1
        tight = L.hang_tight(tr, self.b, L.TOP + tr.h_low)
        at = tight["head"] - rot @ self.o_left
        self._set("left", ("square head", "at the cords", "let go head 2"), at, rot)
        self._set("left", ("back off head 2",), at - 0.045 * rot[:, 2], rot)
        self.reads["square"] = {"root_deg": round(real, 1), "hand_deg": round(hand, 1), "hand_to_deg": round(to, 1)}

    def _stub2(self, state, through: float) -> None:
        """Round 40: the right's second take of the tail, onto the strap's real point below the real exit, and the
        draw from there along the slot's real line (level)."""
        tr = self.tr
        rm = L.quat_xyzw_to_matrix(state.quat_xyzw)
        d = rm @ np.array([0.0, 0.0, -1.0])
        at = np.asarray(state.pos, float) + rm @ np.array([L.SLOT_C, 0.0, L.EXIT_Z]) + d * tr.h_stub_at
        poly = np.asarray(state.polyline, float)
        if through >= tr.h_stub_at + 0.006:  # the strap's own point that far below the exit
            at = L.point_at_arc(poly, L.arc_length(poly) - through + tr.h_stub_at)
        rot = self.base["right"]["take stub 2"][1]
        pos = self._near("right", "take stub 2", at - rot @ self.o, "stub 2", self.SNAP2)
        self._set("right", ("to the stub 2", "take stub 2"), pos)
        line = np.array([d[0], d[1], 0.0])
        line = line / max(float(np.linalg.norm(line)), 1e-9)
        grip = pos + rot @ self.o  # the strap's point in its fingers; the draw is this point's, whatever way the hand turns
        end = grip + line * tr.h_pull - self.base["right"]["pull"][1] @ self.o
        self._set("right", ("pull", "pull hold", "let go stub"), end)
        self._set("right", ("back off stub",), end + np.array([0.0, 0.0, 0.04]))
        self.held = {"from": grip, "to": grip + line * tr.h_pull, "stop": None, "mark": None, "t0": None}
        self.reads["stub 2"] = {"through_mm": round(1000 * through, 1), "line_deg": round(L.heading_xy(line) % 360.0, 1)}

    HELD_PAST = 0.012  # the held draw stops with the full mark this far out: 302 mm drawn, as job 132's draw ended
    #                    (301.5 mm; let go, it rested at 296.5)
    WORK_ON, TIGHT, WORK_S, WORK_T, WORK_DEG, CREEP = 0.008, 0.030, 4.8, 1.6, 15.0, 0.005
    # Round 43, the held draw. The hand goes along the slot's line at the plan's speed while tail comes. Once it is
    # WORK_ON on with no more tail (job 126: a strap joint at the slot's mouth, the next link's end on the head's
    # face) it works the tail from side to side, WORK_DEG either way about the exit every WORK_T seconds, creeping on
    # at CREEP m/s to TIGHT at most, as a person works a tie that sticks. WORK_S of that with no tail: the tie is
    # tight (or, with under 20 mm drawn, stuck) and the draw ends there.

    def _pull_held(self, step: int, lab: str, through: float) -> None:
        """Round 40, the draw with the head held: paced by the tail (above), it ends where the hand is once the full
        mark is PAST out (or tr.h_pull_to), once the tie is tight, or at the end of its reach or its time."""
        h, tr = self.held, self.tr
        if h is None or lab not in ("pull", "pull hold") or (lab == "pull hold" and h["t0"] is None):
            return
        if lab == "pull hold" and h["stop"] is None:  # out of time: it ends on the line, where the hand has got to
            h["stop"] = h["at"] = h["from"] + h["line"] * h["p"]
            self.why = "time"
            self.reads["pull"] = {"why": "time", "through_mm": round(1000 * through, 1), "hand_mm": round(1000 * h["p"], 1),
                                  "worked_s": round(h["work"], 1)}
        if lab == "pull hold" and h["t0"] is not None:  # eased back before it lets go: the pawl takes the strap
            if "real" not in h:  # from where the hand really is (job 133: 24 mm behind its place, so 12 mm back from
                #                  its place still pulled, and let go at that pull the strap ran back 9 mm)
                tcp = (SEEN.get("tcp") or {}).get("right")
                rot = L.rot_z(h.get("yaw", tr.h_stub_yaw)) @ self.r1
                real = h["stop"] if tcp is None else np.asarray(tcp, float) + rot @ self.o
                h["real"] = h["stop"] + h["line"] * float(np.clip((real - h["stop"]) @ h["line"], -0.060, 0.0))
            k = min(max((step / CONTROL_HZ - self.when["right"]["pull hold"]) / 1.0, 0.0), 1.0)
            h["at"] = h["stop"] + (h["real"] - h["line"] * L.HOLD_EASE - h["stop"]) * k
        if h["t0"] is None:
            line = h["to"] - h["from"]
            h.update(t0=through, mark=(0.0, through), p=0.0, work=0.0, step=step, far=float(np.linalg.norm(line)),
                     line=line / max(float(np.linalg.norm(line)), 1e-9), at=np.array(h["from"], float), yaw=tr.h_stub_yaw)
            self.through0 = through
        dt, h["step"] = (step - h["step"]) / CONTROL_HZ, step
        if h["stop"] is None:
            if through > h["mark"][1] + 0.001:
                h["mark"], h["work"] = (h["p"], through), 0.0
            gained, on = through - h["t0"], h["p"] - h["mark"][0]
            target = tr.h_pull_to or L.lift_marks(self.b.tie_l)[0] + self.HELD_PAST
            w = next(x for x in self.arms["right"] if x.label == "pull")
            side = 0.0
            if on < self.WORK_ON:
                h["p"] += h["far"] / w.duration * dt
            else:
                h["work"] += dt
                h["p"] += self.CREEP * dt if on < self.TIGHT else 0.0
                reach = float(np.clip(math.tan(math.radians(self.WORK_DEG)) * (tr.h_stub_at + gained), 0.006, 0.025))
                side = reach * math.sin(2.0 * math.pi * h["work"] / self.WORK_T)
            perp = np.array([-h["line"][1], h["line"][0], 0.0])
            h["at"] = h["from"] + h["line"] * h["p"] + perp * side
            h["yaw"] = tr.h_stub_yaw + (tr.h_pull_yaw - tr.h_stub_yaw) * min(h["p"] / L.HOLD_TURN, 1.0)
            why = ("mark" if through >= target else ("tight" if gained >= 0.020 else "stuck") if h["work"] >= self.WORK_S
                   else "end" if h["p"] >= h["far"] else None)
            if why:
                h["stop"], self.why = h["from"] + h["line"] * h["p"], why
                h["at"] = h["stop"]
                self.back_mm = round(1000 * (h["far"] - h["p"]), 1)
                self.reads["pull"] = {"why": why, "through_mm": round(1000 * through, 1),
                                      "hand_mm": round(1000 * h["p"], 1), "worked_s": round(h["work"], 1)}
        rot = L.rot_z(h.get("yaw", tr.h_stub_yaw)) @ self.r1
        tcp = h["at"] - rot @ self.o
        self._set("right", ("take stub 2", "pull", "pull hold", "let go stub"), tcp, rot)  # both ends of the blend:
        self._set("right", ("back off stub",), tcp + np.array([0.0, 0.0, 0.04]), rot)     # the hand is where h has it

    def _pull(self, t: float, lab: str, through: float, head) -> None:
        """The draw (both its legs): stop where the hand is once the full mark is PAST out, or the tie is tight."""
        rb = self.base["right"]
        pa = np.asarray(next(x for x in self.arms["right"] if x.label == "take stub").pos, float)
        pb, pc = rb["pull out"][0], rb["pull"][0]
        if self.leg is None or lab != self.leg[0]:  # a new leg: where it starts (the stop may have moved it)
            w = next(x for x in self.arms["right"] if x.label == lab)
            start = pa if lab == "pull out" else np.asarray(next(x for x in self.arms["right"] if x.label == "pull out").pos, float)
            self.leg = (lab, self.when["right"][lab], w.duration, start)
        _, t0, dur, start = self.leg
        s = min(max((t - t0) / dur, 0.0), 1.0)
        sm = s * s * (3 - 2 * s)
        end = {"pull out": pb, "pull": pc}.get(lab, pc)
        now = start + (end - start) * sm if lab in ("pull out", "pull") else pc
        hand = float(np.linalg.norm(now - pa))  # how far the plan has the hand from where it took the tail
        if self.through0 is None:
            self.through0, self.mark = through, (hand, through)
        if through > self.mark[1] + 0.001:
            self.mark = (hand, through)
        target = L.lift_marks(self.b.tie_l)[0] + self.PAST
        tight = through > self.through0 + 0.020 and hand - self.mark[0] >= 0.012
        if self.freeze is None and lab in ("pull out", "pull") and (through >= target or tight):
            self.freeze, self.why = (lab, now), "mark" if through >= target else "tight"
            self.back_mm = round(1000 * float(np.linalg.norm(pc - now)), 1)
        if self.freeze is not None:
            at, stop = self.freeze
            rot = rb["pull"][1]
            if lab == at:  # the waypoint the blend reaches the stopped place from: the hand stays there
                self._set("right", (lab,), start + (stop - start) / max(sm, 1e-3), rot)
            later = ("pull", "pull hold", "let go stub") if at == "pull out" else ("pull hold", "let go stub")
            self._set("right", later, stop, rot)
            self._set("right", ("back off stub",), stop + np.array([0.0, 0.0, 0.04]), rot)

    def update(self, step: int, through: float, tip_err, state=None) -> None:
        if state is not None and self.tr.h_held and self.held is not None:  # the held draw: every control step
            self._pull_held(step, self.label(step / CONTROL_HZ, self.arms["right"]), through)
        if step % L.STEER_EVERY or state is None:
            return
        tr = self.tr
        t = step / CONTROL_HZ
        lab = {arm: self.label(t, ws) for arm, ws in self.arms.items()}
        poly = np.asarray(state.polyline, float)
        if lab["left"] in L.HANG_LEFT_TAIL[:2]:
            self._take(poly)
        if not tr.h_thread:
            return
        if lab["left"] == "settle 3":
            self._read_aim(state)
        elif lab["left"] in L.HANG_LEFT_THREAD[2:] and self.aim is not None:
            if through <= 0.0:
                self.corr = np.clip(self.corr - L.STEER_GAIN * np.asarray(tip_err, float), -self.STEER_CAP, self.STEER_CAP)
            if lab["left"] in L.THREAD_PUSH_LABELS:
                self.depth_mm = float(np.clip(self.depth_mm + 1000 * L.DEPTH_GAIN * max(tr.h_out - through, 0.0), 0.0,
                                              1000 * self.DEPTH))
            self._feed()
        if lab["right"] == "tauten":
            self._tauten(t, state)
        elif lab["right"] in L.HANG_RIGHT_REGRIP[:2]:
            self._regrip(state)
        elif lab["right"] == "settle 2b":
            self._turn_head(state)
        if tr.h_held and lab["left"] in L.HANG_LEFT_HEAD[:2]:
            self._head_take(state)
        elif tr.h_held and lab["left"] == "hold head 3":
            self._carry(state)
        elif tr.h_held and lab["left"] == "set head":
            self._square(state)
        if lab["right"] in L.HANG_RIGHT_STUB[:2]:
            self._stub(state, through)
        elif tr.h_held:
            if lab["right"] == "to the stub 2":
                self._stub2(state, through)
        elif lab["right"] in L.HANG_PULL[:3]:
            self._pull(t, lab["right"], through, state.pos)

    def log(self) -> dict:
        return {"moved_mm": dict(self.moved), "reads": dict(self.reads),
                "steer_mm": [round(1000 * float(x), 2) for x in self.corr], "depth_mm": round(self.depth_mm, 2),
                "pull_from_mm": None if self.through0 is None else round(1000 * self.through0, 1),
                "pull_stopped": self.why, "pull_back_mm": self.back_mm}


DEMO = {"steering": None}  # the demo's closed loop while a demo plan plays (demo_plan sets it, stop_demo clears it)


def demo_plan(tr: L.Trial) -> dict:
    """Trial ``tr``'s plan as waypoints, its inserting hand on the closed loop (:class:`Steering`)."""
    plan = plan_waypoints(L.trial_plan(tr))
    if tr.stage == "hang":
        DEMO["steering"] = HangSteering(tr, plan["left"], plan["right"])
        return plan
    if tr.stage == "thread":
        DEMO["steering"] = (WedgeSteering if tr.b_wedge else CinchSteering if tr.b_cinch else ThreadSteering)(
            tr, plan["left"], plan["right"])
        return plan
    DEMO["steering"] = Steering(tr, plan["left"], plan["right"]) if (tr.steer or tr.depth or tr.tighten or
                                                                     tr.unstick) else None
    return plan


def stop_demo() -> None:
    DEMO["steering"] = None


def demo():
    """The scripted demo (yam_scripted plays it): probe trial TIE_TRIAL, by default the whole tie on the hanging
    cords (h37-tie: both hands lift the tie and turn it on edge, the right passes it behind the cords, the left takes
    the tail in front, the right brings the head round, the left feeds the tip through the slot, the right draws the
    tail until the full mark is out, both let go), with its closed loop (HangSteering)."""
    return demo_plan(L.trial(TRIAL))
