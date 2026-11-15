"""Geometry, scoring logic and the demo plan for rearm-wound-drain-bulb, as plain numbers.

Pure Python (numpy only): nothing here imports the simulator, so the scoring can be tested anywhere and the demo's
reach checked against the arm's URDF. task.py builds the scene and the demo from these numbers.

World frame: x forward from the arms, y to the robot's left, z up (m). The arm bases stand at x 0, y +-0.305.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TOP = 0.7475  # the station's table top (yam_arena.task_file.TABLE_TOP["station"])
CONTROL_HZ = 20

# -- the bulb ----------------------------------------------------------------------------------------------------------
# A 100 mL closed-suction bulb hanging upright from the bed's side rail, its flat faces toward the robot's sides.
# It is turned 20 degrees (YAW) so the left arm's forward grip meets it square: checked on the URDF, 14 Nov 2026.
BULB_XY = (0.22, -0.12)
YAW = math.radians(-20.0)
HALF_W, HALF_T, HALF_H = 0.032, 0.024, 0.050  # half width (local x), half thickness round (local y), half height (z)
BULB_Z = TOP + 0.025 + HALF_H  # bottom 2.5 cm above the bedside table
TRAVEL = 0.016  # each face moves this far in when squeezed flat: 48 mm thick round, 16 mm flat
PADDLE = (0.060, 0.003, 0.080)  # each face's collider (local x, y, z): what the fingers press
SPRING = 10.0 / TRAVEL  # N/m per face: about 10 N to hold it flat (silicone bulbs take roughly 10-20 N)

# The drainage port on the bulb's top, and its tethered plug (a hinged cap; angle 0 = seated, positive = open).
PORT = (0.012, 0.012, 0.010)  # base box
PORT_Z = BULB_Z + HALF_H + PORT[2] / 2 - 0.002  # sunk 2 mm into the bulb's top
PORT_TOP = PORT_Z + PORT[2] / 2
PLUG_R, PLUG_T = 0.008, 0.006  # cap: 16 mm across, 6 mm thick
PLUG_PIVOT = (0.0, -PORT[1] / 2, PORT[2] / 2)  # base frame: the port's top edge on the local -y side
PLUG_AT = (0.0, PLUG_R, PLUG_T / 2)  # the cap's centre from the pivot when seated: its rim on the hinge line
PLUG_OPEN = 2.0  # rad: open, leaning past upright onto its stop
# The tether springs the cap back open (rest = PLUG_OPEN), as a real plug's strap does: its own weight can't close it,
# so it seats only while something presses it flat. At flat the spring pushes up with PLUG_SPRING * PLUG_OPEN, about
# 17 times the cap's weight torque and a fingertip push of about 0.25 N. Once seated, task code holds it there (a
# push-fit plug grips the port), so a seated cap stays seated.
PLUG_SPRING = 0.002  # N m/rad
PLUG_MASS = 0.003  # kg
PLUG_SEATED = 0.035  # rad: pressed within 2 degrees of flat on the port counts as seated (the hinge's latch detent)
PLUG_UNSEATED = 0.12  # rad: lifted past this, it would be unseated again (the latch's tolerance; held caps aren't)

# The bulb hangs free: it swings about a point SWING_DROP above its centre, against a soft spring standing in for its
# strap and tubing. A 1 N sideways push moves its centre about 1.7 cm; pressing down on the cap acts through the
# hanging point and doesn't swing it. Pulled more than TUG_LIMIT from where it hangs, the drain has been tugged.
SWING_DROP = 0.16  # m: the hanging point is this far above the bulb's centre
SWING_SPRING = 1.5  # N m/rad
MAX_SWING = 0.7  # rad: the strap's reach
TUG_LIMIT = 0.03  # m: the bulb's centre pulled further than this from where it hangs = the drain tugged (sticky 0)
HANG_POINT = None  # set below, after CENTRE

# Scoring thresholds.
ARM_AT = 0.7  # squeeze needed at the moment the plug seats
END_AT = 0.6  # squeeze still needed at the end (the hold gives about 1 mm to 20 N)
LET_GO_R = 0.06  # m: a gripper point within this of the bulb's vertical axis, between its bottom and above the plug,
LET_GO_Z = (BULB_Z - HALF_H - 0.02, PORT_TOP + 0.04)  # is still at the bulb
MOUTH_R, MOUTH_Z = 0.006, 0.004  # a pad tip this close to the open port's mouth touches inside it
OPEN_FOR_TOUCH = 0.35  # rad: the plug counts as open (the mouth exposed) above this


def local_to_world(v) -> np.ndarray:
    """A vector in the bulb's frame (yawed) as a world vector."""
    c, s = math.cos(YAW), math.sin(YAW)
    x, y, z = v
    return np.array([c * x - s * y, s * x + c * y, z])


U = local_to_world((1.0, 0.0, 0.0))  # the bulb's local x: the left arm's approach direction
V = local_to_world((0.0, 1.0, 0.0))  # the bulb's local y: the squeeze axis, the left face's outward normal
CENTRE = np.array([BULB_XY[0], BULB_XY[1], BULB_Z])  # where the bulb hangs at rest
MOUTH = np.array([BULB_XY[0], BULB_XY[1], PORT_TOP])
HANG_POINT = CENTRE + np.array([0.0, 0.0, SWING_DROP])


def squeeze(left_pos: float, right_pos: float) -> float:
    """How flat the bulb is, 0 (round) to 1 (flat): the mean of the two faces' travel. The left face (local +y)
    moves to negative positions when pushed in, the right face (local -y) to positive ones."""
    return (max(0.0, -left_pos) + max(0.0, right_pos)) / (2 * TRAVEL)


def thickness(left_pos: float, right_pos: float) -> float:
    """The bulb's thickness (m) between its faces."""
    return 2 * HALF_T - max(0.0, -left_pos) - max(0.0, right_pos)


def at_bulb(point, centre=CENTRE) -> bool:
    """A gripper point near enough the bulb (centred at ``centre`` now) to be holding it, or pressing its plug."""
    p, c = np.asarray(point, float), np.asarray(centre, float)
    dz = p[2] - c[2]
    return bool(math.hypot(p[0] - c[0], p[1] - c[1]) < LET_GO_R and
                LET_GO_Z[0] - BULB_Z < dz < LET_GO_Z[1] - BULB_Z)


def touches_mouth(point, mouth=MOUTH) -> bool:
    """A pad tip at the port's mouth (where it is now)."""
    p, m = np.asarray(point, float), np.asarray(mouth, float)
    return bool(math.hypot(p[0] - m[0], p[1] - m[1]) < MOUTH_R and abs(p[2] - m[2]) < MOUTH_Z)


def pulled(centre) -> bool:
    """The bulb's centre more than TUG_LIMIT across from where it hangs: the drain has been tugged."""
    c = np.asarray(centre, float)
    return bool(math.hypot(c[0] - CENTRE[0], c[1] - CENTRE[1]) > TUG_LIMIT)


@dataclass
class DrainTrial:
    """One trial's memory, updated every control step by :meth:`step` (task.py's on_step hook, or a test)."""

    seated: bool = False  # the plug was seated at the last step
    armed: bool = False  # seated while the bulb was squeezed at least ARM_AT, and not unseated since
    seal_squeeze: float | None = None  # how flat the bulb was when the plug last seated
    contaminated: bool = False  # a pad tip touched the open port's mouth (latched)
    tugged: bool = False  # the bulb was pulled more than TUG_LIMIT from where it hangs (latched)
    squeeze_now: float = 0.0
    points: list = field(default_factory=list)  # the grippers' points (TCP and pad tip of each), last step
    centre: np.ndarray = field(default_factory=lambda: CENTRE.copy())  # the bulb's centre, last step

    def step(self, plug_angle: float, plug_latched: bool, left_pos: float, right_pos: float, points,
             centre=CENTRE, mouth=MOUTH) -> str | None:
        """Advance one control step (``centre`` and ``mouth`` where the swinging bulb is now). Returns "seal" when
        the plug has just seated (hold the bulb where it is), "unseal" when it has just come off, else None."""
        self.squeeze_now = squeeze(left_pos, right_pos)
        self.points = [np.asarray(p, float) for p in points]
        self.centre = np.asarray(centre, float)
        if plug_angle > OPEN_FOR_TOUCH and any(touches_mouth(p, mouth) for p in self.points):
            self.contaminated = True
        if pulled(self.centre):
            self.tugged = True
        action = None
        if plug_latched and not self.seated:
            self.seal_squeeze, self.armed, action = self.squeeze_now, self.squeeze_now >= ARM_AT, "seal"
        elif not plug_latched and self.seated:
            self.seal_squeeze, self.armed, action = None, False, "unseal"
        self.seated = bool(plug_latched)
        return action

    def let_go(self) -> bool:
        return not any(at_bulb(p, self.centre) for p in self.points)

    def done(self) -> bool:
        """Full marks: seated while squeezed, still flat, both grippers away, port never touched, drain never
        tugged."""
        return (not self.contaminated and not self.tugged and self.seated and self.armed
                and self.squeeze_now >= END_AT and self.let_go())

    def score(self) -> float:
        if self.contaminated or self.tugged or not self.seated:
            return 0.0
        if not self.armed:
            return 0.25  # plugged while round: looks finished, no suction
        if self.done():
            return 1.0
        return 0.5  # armed, but still held (or the hold slipped below END_AT)


# -- the rest of the scene (fixed) -------------------------------------------------------------------------------------
RAIL_X = 0.33  # the bed's side rail runs along y behind the bulb; the manikin lies beyond it
RAIL = dict(size=(0.025, 0.62, 0.030), pos=(RAIL_X, -0.05, TOP + 0.145))
POSTS = [dict(size=(0.025, 0.025, 0.130), pos=(RAIL_X, y, TOP + 0.065)) for y in (-0.34, 0.24)]
MATTRESS = dict(size=(0.60, 1.20, 0.006), pos=(0.65, -0.05, TOP + 0.003))
MANIKIN_ORIGIN = (0.62, 0.0, TOP + 0.006)
DRAIN_EXIT = np.array([0.455, -0.12, TOP + 0.085])  # the dressing on the manikin's flank, nearest the robot
CUP_XY = (0.12, 0.36)  # the measuring cup the drain was just emptied into, on the bedside table


TAIL = 0.02  # m of tubing below the bulb that swings with it


def world_to_core(p) -> np.ndarray:
    """A world point in the bulb's core frame at rest (origin the hanging point, turned YAW)."""
    d = np.asarray(p, float) - HANG_POINT
    c, s = math.cos(YAW), math.sin(YAW)
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]])


def tubing_path() -> list:
    """The fixed run of drain tubing (world): from the dressing, under the rail, to just below the bulb."""
    below = CENTRE - np.array([0.0, 0.0, HALF_H - 0.004 + TAIL])
    return [DRAIN_EXIT, np.array([0.41, -0.12, TOP + 0.06]), np.array([0.35, -0.12, TOP + 0.03]),
            np.array([0.28, -0.12, TOP + 0.012]), below]


def tail_path_core() -> list:
    """The last bit of tubing, into the bulb's bottom (core frame: it swings with the bulb)."""
    bottom = np.array([0.0, 0.0, -SWING_DROP - HALF_H + 0.004])
    return [bottom - np.array([0.0, 0.0, TAIL]), bottom]


def strap_path_core() -> list:
    """The holder strap from the bulb's back shoulder up to its clip on the rail (core frame: it swings with the
    bulb; its far end stays within a few mm of the clip for the small swings that don't count as a tug)."""
    shoulder = np.array([0.9 * HALF_W, 0.0, -SWING_DROP + 0.55 * HALF_H])
    clip_world = np.array([RAIL_X - 0.013, BULB_XY[1] + (HALF_W * 0.9) * U[1], RAIL["pos"][2]])
    return [shoulder, world_to_core(clip_world)]


# -- the demo plan (world positions; rotations named, not computed, so this stays free of the simulator) ---------------
# Left arm: forward grip turned YAW, fingers closing along V: it squeezes the bulb and holds it. Right arm: straight
# down, closed fingers, turned YAW: it pushes the open plug over, then presses it flat. Positions are TCP targets.
# PAD_TIP is how far the pad tips stand past the TCP (yam_arena.constants.PAD_TIP_PAST_TCP).
PAD_TIP = 0.013
SQUEEZED_AT = 4.5  # s: the left arm has the bulb flat by then


def _tip(at, z) -> tuple:
    """TCP of a straight-down grip whose pad tips are at ``at`` (x, y) and height ``z``."""
    return (float(at[0]), float(at[1]), float(z + PAD_TIP))


def demo_plan() -> dict:
    """{"left": [...], "right": [...]}, each step (seconds, TCP (x, y, z), fingers "open"/"closed", grip, label);
    grip is "forward" (turned YAW), "down" (turned YAW) or "ready" (the arm's ready pose; its TCP is not used)."""
    c = CENTRE
    port = MOUTH[:2]
    side = port - 0.035 * V[:2]  # beside the open plug, on the side it leans toward
    cap = port + (PLUG_PIVOT[1] + PLUG_AT[1]) * V[:2]  # above the seated cap's centre: the sweep ends here
    away = port - 0.075 * V[:2]  # off to the right arm's side: straight up out of reach would hit a joint limit
    right = [
        (2.5, _tip(side, PORT_TOP + 0.05), "closed", "down", "above the open plug"),
        (SQUEEZED_AT - 2.5, _tip(side, PORT_TOP + 0.05), "closed", "down", "wait for the squeeze"),
        # The open cap stands up to about 14 mm above the port's top: sweeping at 9 mm catches its upper half and tips
        # it over. The tether springs it back open, so the fingers stop on top of it (over its middle, the cap about 25
        # degrees from flat under them) and press straight down without letting go of it. The pad tips never come
        # within MOUTH_Z of the mouth: the cap (6 mm) is always between them.
        (1.0, _tip(side, PORT_TOP + 0.009), "closed", "down", "down beside the plug"),
        (1.5, _tip(cap, PORT_TOP + 0.009), "closed", "down", "push the cap over onto the port"),
        # The cap needs only about 0.25 N, and the bulb hangs free: press 1.2 cm past the seated cap (the arm gives
        # about 8 mm), not the 2.5 cm a stiff button needs, so the push stays light.
        (1.2, _tip(cap, PORT_TOP + PLUG_T - 0.012), "closed", "down", "press the cap home"),
        (0.8, _tip(cap, PORT_TOP + PLUG_T - 0.012), "closed", "down", "hold"),
        (0.6, _tip(cap, PORT_TOP + 0.035), "closed", "down", "lift off the cap"),
        (1.0, _tip(away, PORT_TOP + 0.035), "closed", "down", "move away to the side"),
        (1.5, None, "open", "ready", "back to ready"),
    ]
    pressed = sum(s[0] for s in right[:-1])  # the right arm is clear of the bulb by then
    left = [
        (2.0, tuple(c - 0.10 * U), "open", "forward", "in front of the bulb"),
        (1.5, tuple(c), "open", "forward", "fingers either side"),
        (SQUEEZED_AT - 3.5, tuple(c), "closed", "forward", "squeeze flat"),
        (pressed - SQUEEZED_AT, tuple(c), "closed", "forward", "hold flat while the plug goes in"),
        (0.8, tuple(c), "open", "forward", "let go"),
        (1.2, tuple(c - 0.12 * U), "open", "forward", "back out"),
        (1.5, None, "open", "ready", "back to ready"),
    ]
    return {"left": left, "right": right}


def demo_seconds() -> float:
    return max(sum(s[0] for s in steps) for steps in demo_plan().values())
