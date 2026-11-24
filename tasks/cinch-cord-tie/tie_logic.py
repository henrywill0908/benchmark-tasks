"""cinch-cord-tie: every number, the geometry, the draft scorer and the probe's plans. NumPy only (no Isaac Sim, no
yam_arena), so tests and the GPU runner can read it before the simulator starts.

Frames. World: x forward (away from the arms), y left, z up, metres; the arm bases at x 0, y +-0.305, z 0.75. The
tie's own frame, the HEAD frame ("_h"): origin at the head's centre; x_h is the way the strap leaves the head (the
root), y_h is across the strap's width, z_h is the slot's axis. The tail goes into the slot on the +z_h face (the
entry) and comes out of the -z_h face (the exit). The strap only bends about y_h (no twist), so the loop lies in the
head's x_h-z_h plane and closes toward +z_h.

Builds (one simulator launch each; BUILDS):
- "jig": probe round 1. The head is welded in the air beside the cords (a jig standing in for the hand that holds it);
  the strap spawns as a relaxed open loop round the cords, its tip 4 mm short of the slot. The cords hang straight down
  from the TV to the power strip, rigid and already gathered.
- "free": the tie lies flat on the console, free.
- "wrap" (round 7): the head held where the jig holds it, the strap slipped round the back of the cords and its last
  69 mm straight (nylon's rest shape), sticking out toward the robot. The robot swings the tail round the front of the
  bundle to the slot itself, then threads and cinches it as on the jig.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# -- the room ---------------------------------------------------------------------------------------------------------
TOP = 0.7475  # the station's table top: the TV console's top
CONTROL_HZ = 20
MM = 0.001

BUNDLE = np.array([0.380, 0.000])  # where the cords hang (x, y)
Z_TIE = TOP + 0.170  # the height the jig holds the tie at
CORD_Z = (TOP + 0.040, TOP + 0.340)  # power strip top to TV bottom
TV = dict(x=(0.420, 0.460), y=(-0.30, 0.30), z=(TOP + 0.340, TOP + 0.420))
WALL = dict(x=(0.460, 0.473), y=(-0.30, 0.30), z=(TOP, TOP + 0.420))
STRIP = dict(x=(0.3725, 0.4275), y=(-0.14, 0.14), z=(TOP, TOP + 0.040))  # the power strip, along y under the cords


@dataclass(frozen=True)
class Cord:
    name: str
    diameter: float
    color: tuple
    xy: tuple  # its axis (x, y), world


def _cords() -> tuple:
    """Three household cords, gathered: TV power (7.0 mm) and HDMI (7.3 mm) side by side, the phone charger (3.5 mm)
    nestled behind them, 0.3 mm gaps."""
    gap = 0.3 * MM
    ra, rb, rc = 3.50 * MM, 3.65 * MM, 1.75 * MM
    a = np.array([0.0, -(ra + gap / 2)])
    b = np.array([0.0, rb + gap / 2])
    da, db = ra + rc + gap, rb + rc + gap
    # c touches a and b: |c - a| = da, |c - b| = db, behind them (x > 0)
    d = np.linalg.norm(b - a)
    t = (da**2 - db**2 + d**2) / (2 * d)
    along = (b - a) / d
    c = a + t * along + math.sqrt(max(da**2 - t**2, 0.0)) * np.array([1.0, 0.0])
    off = (a + b + c) / 3
    return (Cord("cord_tv", 2 * ra, (0.05, 0.05, 0.05), tuple(BUNDLE + a - off)),
            Cord("cord_hdmi", 2 * rb, (0.12, 0.12, 0.13), tuple(BUNDLE + b - off)),
            Cord("cord_phone", 2 * rc, (0.93, 0.93, 0.92), tuple(BUNDLE + c - off)))


CORDS = _cords()

# -- the cable tie: 200 mm x 4.8 mm x 1.2 mm nylon 6/6 ------------------------------------------------------------------
TIE_L = 0.200  # strap length, head face to tip
N_LINKS = 32
LINK = TIE_L / N_LINKS  # 6.25 mm
W, T = 4.8 * MM, 1.2 * MM  # strap width and thickness
TIP_W, TIP_T = 3.0 * MM, 0.9 * MM  # the last link: the tapered tip
NYLON_DENSITY = 1140.0
EI = 1.0e-3  # N m^2: E about 1.5 GPa (conditioned nylon 6/6) x I = W T^3 / 12 = 6.9e-13 m^4
MASS_SCALE = 2.0  # sim links weigh twice the real strap (PhysX steadiness); a free strap sags more than a real one
LINK_MASS = W * T * LINK * NYLON_DENSITY * MASS_SCALE  # about 0.08 g
STRAP_K = EI / LINK  # N m/rad per joint
STRAP_DAMPING = 1.0e-3
# Round 1 (17 Nov): a tie touching fingers or its own slot chattered at 35-60 mm/s and never came to rest (untouched it
# was still). Presets for round 2, chosen per simulator launch with TIE_PHYS: heavier links and heads (the outlet's
# 0.5 g wire links were steady), more joint and body damping, a thicker contact skin. EI stays nylon's in all of them,
# so the pull a snug loop needs doesn't change.
PHYS = {
    "base": dict(mass_scale=2.0, joint_damping=1.0e-3, body_lin=0.2, body_ang=1.0, offset=0.4e-3, armature=5e-6),
    "heavy": dict(mass_scale=6.0, joint_damping=4.0e-3, body_lin=2.0, body_ang=6.0, offset=0.8e-3, armature=2e-5),
    "heavier": dict(mass_scale=15.0, joint_damping=1.0e-2, body_lin=5.0, body_ang=15.0, offset=0.8e-3, armature=5e-5),
    # Round 19: heavier's links weigh 15 times a real strap's at real stiffness, so a tie held by its head flops onto the
    # console like a shoelace (job 65, f-pick: head lifted 60 mm, tip still on the console) where real nylon stands out
    # (a 194 mm cantilever sags about 6 mm). k_scale raises the strap's stiffness toward its weight; damping and
    # armature grow with it so the joints stay as steady.
    "stiff5": dict(mass_scale=15.0, joint_damping=3.2e-2, body_lin=5.0, body_ang=15.0, offset=0.8e-3, armature=1e-4,
                   k_scale=5.0),
    "stiff15": dict(mass_scale=15.0, joint_damping=7.7e-2, body_lin=5.0, body_ang=15.0, offset=0.8e-3, armature=2e-4,
                    k_scale=15.0),
    # Round 38 (job 123): the threaded tie would not draw. With nothing holding the head, the pull on the tail is
    # answered by the strap's root, 4.6 mm to one side of the slot: the head cocks, the strap bears on the slot's
    # two ends, and at friction 0.9 that bearing holds more than the pull (1.5 times it), however hard the pull. Nylon
    # on nylon is 0.2 to 0.4. These are heavier with the tie's friction nylon's, and "min" so that it stays that on
    # the cords and the pads (average would bring it back up to 0.7).
    "slick3": dict(mass_scale=15.0, joint_damping=1.0e-2, body_lin=5.0, body_ang=15.0, offset=0.8e-3, armature=5e-5,
                   friction=0.3, combine="min"),
    "slick5": dict(mass_scale=15.0, joint_damping=1.0e-2, body_lin=5.0, body_ang=15.0, offset=0.8e-3, armature=5e-5,
                   friction=0.5, combine="min"),
    # Friction by contact: the strap as grippy on the pads as before (0.9, averaged), nylon on nylon in the slot and
    # on the pawl (0.25), nylon on a PVC jacket round the cords (0.35).
    "nylon": dict(mass_scale=15.0, joint_damping=1.0e-2, body_lin=5.0, body_ang=15.0, offset=0.8e-3, armature=5e-5,
                  head_friction=0.25, cord_friction=0.35),
    # Round 41: nylon with a lead-in on the slot's mouth (job 126: the held draw fed 6 mm, then stalled with a
    # link's end at the mouth). Job 127 (0.6 mm) never got to the draw: the head slid 0.5 mm along the right's pads
    # while the left bent the strap in (37.5 s), was held by its end from there and came out on the way round the
    # cords. Every run before held there; this one is the same but for the lead-in's few cubic millimetres.
    "nylonlead": dict(mass_scale=15.0, joint_damping=1.0e-2, body_lin=5.0, body_ang=15.0, offset=0.8e-3, armature=5e-5,
                      head_friction=0.25, cord_friction=0.35, lead_in=0.5e-3),
    # nylongrip (round 48, not the task's, never run): the pawl's plate with a friction of its own, 1.2 (at 0.25 it
    # cannot wedge, and a tail let go under load runs back until the loop is slack: job 133, 10 mm).
    "nylongrip": dict(mass_scale=15.0, joint_damping=1.0e-2, body_lin=5.0, body_ang=15.0, offset=0.8e-3, armature=5e-5,
                      head_friction=0.25, cord_friction=0.35, lead_in=0.5e-3, pawl_friction=1.2),
    # Round 44. Job 128's lead-in (0.5 mm) lost the head at the same step as job 127's (0.6): not chance. The pads
    # touch the same faces either way; what the lead-in changes outside the slot is the head's centre of mass and
    # inertia, which PhysX works out from the colliders. nylonlead now pins them to the plain head's (labkit,
    # lead_in_mass). nylonback: a lead-in on the back wall only, the wall the pawl presses the strap on, not pinned.
    "nylonback": dict(mass_scale=15.0, joint_damping=1.0e-2, body_lin=5.0, body_ang=15.0, offset=0.8e-3, armature=5e-5,
                      head_friction=0.25, cord_friction=0.35, lead_in=0.5e-3, lead_in_walls=("back",), lead_in_mass=False),
}
STRAP_ARMATURE = 5.0e-6
MAX_BEND = math.radians(100.0)  # per joint, either way from the start shape (round 6: 60 left the D-loop's 55 degree
#                                    turn 5 degrees from its stop when pulled straight through the head)
TIE_FRICTION = 0.9

HEAD = np.array([8.0, 9.0, 5.6]) * MM  # x_h, y_h, z_h
HEAD_MASS = 0.25e-3 * MASS_SCALE
PASSAGE_X = (-2.6 * MM, 1.4 * MM)  # the slot's through-passage across x_h (4.0 mm; a real slot is about 1.6)
PASSAGE_Y = 3.2 * MM  # half its width across y_h (6.4 mm; real about 5.2)
SLOT_C = (PASSAGE_X[0] + PASSAGE_X[1]) / 2  # -0.6 mm: the passage's centre line x_h
ENTRY_Z, EXIT_Z = HEAD[2] / 2, -HEAD[2] / 2
ROOT = np.array([HEAD[0] / 2, 0.0, -HEAD[2] / 2 + T / 2])  # the strap leaves the +x_h face, flush with the bottom

# The pawl: a capsule tongue hinged near the entry on the passage's +x_h side, pointing down toward the strap. Its
# spring turns it into the passage (toward PAWL_REST); the strap, pushed against the back wall, stops it at
# pawl_contact_deg(). Pulled back, the strap drags the tongue's tip toward it: past 45 degrees from the slot axis it
# wedges (self-locking for friction above tan(90 - angle)). Pulled through, it drags the tip away: free.
PAWL_HINGE = np.array([1.1 * MM, 0.0, 2.4 * MM])
PAWL_LEN = 2.6 * MM  # hinge to the tip sphere's centre
PAWL_R = 0.3 * MM
PAWL_WIDTH = 5.6 * MM
PAWL_REST = math.radians(75.0)  # angle from -z_h toward -x_h; the joint's limit and the spring's rest
PAWL_MASS = 0.1e-3
PAWL_SPRING = 5.0e-4  # N m/rad
PAWL_DAMPING = 2.0e-5
# Round 3 (17 Nov, r3-5502a16-29): with 0.5 N m/rad and 2 degrees the braked strap still slid from 11 mm out to 0 in
# under a second (the loop springs back toward its spawn shape); a strap that left the slot within one step was never
# braked at all. Round 4: ten times stiffer, aimed further in, engaged at the first 0.2 mm of backing out.
BRAKE_STIFFNESS = 5.0  # N m/rad: the pawl's stiffness while task code brakes it ("brake" ratchet)
BRAKE_DEG = 3.0  # how far past where it is the brake aims the pawl
BRAKE_SEAT = 55.0  # deg: the brake aims from at least here, where the pawl sits on a strap that lies on the slot's
#                    wall (job 131: drawn hard the strap rode high, the pawl at 27, and a brake aimed 3 past that
#                    let 6 mm run back; job 132: from 52 it held only once it had closed to 58, 5 mm on)
BRAKE_MORE = 2.0  # deg: ... and for each further BRAKE_SLIP it backs out, the brake closes this much more
BRAKE_SLIP = 0.2 * MM  # the strap backing out this far below its best pull engages the brake
BRAKE_RELEASE = 0.4 * MM  # ... and pulled this far past where it braked lets it go
RATCHETS = ("cam", "brake", "open", "latch")
# Round 6 "latch": the pawl held flat against the passage's far wall while the tail goes in, released onto the strap's
# face once LATCH_AT is out, then the brake. Round 5: 3 of 5 pushes blew up where the tip met the sprung pawl (the tip
# flung 20-45 mm back, 18-33 mm slip in the grip); the 2 that got past it threaded 21 mm.
LATCH_AT = 16.0 * MM
LIFT_LATCH_AT = 8.0 * MM  # the lying-cords task: the pawl drops once 8 mm of tail is out, about where a real tie's first
#                           teeth reach its pawl (the tapered lead is smooth); by then the pawl, 1 mm inside the entry,
#                           comes down on the strap's face 14 mm behind the tip, not head-on
# Round 16 also tried a "catch" (the brake on once the strap stood still for 1 s): held at a fixed angle the pawl can't
# turn in and wedge, so a let-go ran back as the latch's did, and three pulls under it slipped 46-102 mm. Removed.
# Round 6: nylon takes a set. Its yield strain (about 3.5 %) on a 1.2 mm strap is a curvature of about 58 /m: 0.36 rad
# over a 6.25 mm link. Round 5's pull jammed (then NaN) when the D-loop's 55 degree turn, a permanent kink in an
# elastic strap, reached the slot.
YIELD_DEG = 20.0


def pawl_contact_deg(face_x: float = PASSAGE_X[0] + T) -> float:
    """The pawl angle (deg) at which its tip sphere touches a strap face at x_h = ``face_x``."""
    s = (PAWL_HINGE[0] - PAWL_R - face_x) / PAWL_LEN
    return float(np.degrees(np.arcsin(np.clip(s, -1.0, 1.0))))


def pawl_tip(angle_rad: float) -> np.ndarray:
    """The tip sphere's centre (head frame) at pawl angle ``angle_rad``."""
    return PAWL_HINGE + PAWL_LEN * np.array([-math.sin(angle_rad), 0.0, -math.cos(angle_rad)])


# -- rotations ----------------------------------------------------------------------------------------------------------
def rot_axes(approach, closing) -> np.ndarray:
    """A TCP rotation from where the gripper points (its z) and the way its fingers close (its x); yam_arena's grips
    are built the same way (grip_rotation(0): approach -z, closing -y)."""
    z = np.asarray(approach, float)
    z = z / np.linalg.norm(z)
    x = np.asarray(closing, float)
    x = x - (x @ z) * z
    x = x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def rot_down(yaw_deg: float) -> np.ndarray:
    """yam_arena's straight-down grip turned ``yaw_deg`` about the vertical (equals grip_rotation(radians(yaw)))."""
    y = math.radians(yaw_deg)
    return rot_axes((0.0, 0.0, -1.0), (math.sin(y), -math.cos(y), 0.0))


def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat_xyzw(r) -> np.ndarray:
    r = np.asarray(r, float)
    tr = np.trace(r)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


# The head's orientation in each build: columns x_h, y_h, z_h in world.
R_JIG = np.column_stack([(1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)])  # loop flat, round the hanging cords
R_FREE = np.column_stack([(0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)])  # lying flat, strap toward +y


# -- spawn shapes -------------------------------------------------------------------------------------------------------
TIP_GAP = 4.0 * MM  # the jig's tip stops this short of the entry face, on the slot's centre line
# Where the jig puts the cords' centroid in the head frame: beside the head, inside the open loop's corner (3 mm above
# the root run, 5 mm past the head's +x face, 9 mm from the tail's run down into the slot), as a hand holding the head
# against the bundle would. Pulled snug, the loop closes on the cords next to the head.
JIG_BUNDLE_H = np.array([14.0, 0.0, 9.6]) * MM


def _teardrop_curve(b: float, r: float):
    """Arc-length parametrised curve (head frame, in the x_h-z_h plane): from ROOT straight along +x_h for ``b``, then
    270 degrees round a circle of radius ``r`` toward +z_h, then straight along -z_h forever."""
    arc = 1.5 * math.pi * r
    c = ROOT + np.array([b, 0.0, r])

    def at(s: float) -> np.ndarray:
        if s <= b:
            return ROOT + np.array([s, 0.0, 0.0])
        if s <= b + arc:
            th = (s - b) / r
            return c + r * np.array([math.sin(th), 0.0, -math.cos(th)])
        end = c + r * np.array([-1.0, 0.0, 0.0])
        return end + (s - b - arc) * np.array([0.0, 0.0, -1.0])

    return at


def _chord_walk(curve, n: int, length: float, s_max: float) -> np.ndarray:
    """n + 1 points on ``curve`` with every chord exactly ``length``."""
    pts, s = [curve(0.0)], 0.0
    for _ in range(n):
        lo, hi = s, s + 1.6 * length
        while np.linalg.norm(curve(hi) - pts[-1]) < length and hi < s_max:
            hi += length
        for _ in range(60):
            mid = (lo + hi) / 2
            if np.linalg.norm(curve(mid) - pts[-1]) < length:
                lo = mid
            else:
                hi = mid
        s = (lo + hi) / 2
        p = curve(s)
        p = pts[-1] + (p - pts[-1]) * (length / np.linalg.norm(p - pts[-1]))
        pts.append(p)
    return np.array(pts)


def _teardrop(b: float, r: float) -> np.ndarray:
    return _chord_walk(_teardrop_curve(b, r), N_LINKS, LINK, 2 * TIE_L)


D_TURN_R = 6.0 * MM  # the D-loop's small turn, from running back along the top into the run down to the slot
D_TOP = 2.0 * MM  # the straight along the loop's top between the big and the small turn
# Round 5: a tight small turn leaves a 62 mm straight run into the slot (44 in round 3-4), so one push from 50 mm back
# brings 22 mm of tail out without letting go (round 4: every regrip let the strap spring back out of the slot).


def _dloop_curve(b: float, r1: float):
    """Arc-length curve (head frame, x_h-z_h plane): from ROOT straight along +x_h for ``b``, a half turn of radius
    ``r1`` toward +z_h (now running back along -x_h), D_TOP straight, a quarter turn of D_TURN_R down toward -z_h, then
    straight along -z_h forever: the run that points into the slot."""
    r2, c = D_TURN_R, D_TOP
    c1 = ROOT + np.array([b, 0.0, r1])
    a1 = math.pi * r1
    top0 = c1 + np.array([0.0, 0.0, r1])  # the top of the half turn, heading -x_h
    c2 = top0 + np.array([-c, 0.0, -r2])
    a2 = 0.5 * math.pi * r2
    down0 = c2 + np.array([-r2, 0.0, 0.0])  # heading -z_h

    def at(s_: float) -> np.ndarray:
        if s_ <= b:
            return ROOT + np.array([s_, 0.0, 0.0])
        s_ -= b
        if s_ <= a1:
            th = s_ / r1
            return c1 + r1 * np.array([math.sin(th), 0.0, -math.cos(th)])
        s_ -= a1
        if s_ <= c:
            return top0 + np.array([-s_, 0.0, 0.0])
        s_ -= c
        if s_ <= a2:
            th = s_ / r2
            return c2 + r2 * np.array([-math.sin(th), 0.0, math.cos(th)])
        s_ -= a2
        return down0 + np.array([0.0, 0.0, -s_])

    return at


def _dloop(b: float, r1: float) -> np.ndarray:
    return _chord_walk(_dloop_curve(b, r1), N_LINKS, LINK, 2 * TIE_L)


def jig_points() -> np.ndarray:
    """The jig's start shape (head frame, N_LINKS + 1 points): a relaxed open D-shaped loop round the cords whose last
    45 mm runs straight down the slot's centre line, its tip TIP_GAP short of the entry face. Round 2 (17 Nov) showed
    why: fingers closed on the strap's edges can't turn it (it pivots between the pads), so a hand lining a curved
    strap up swung the tip 10-16 mm off the slot. On this loop the hand takes the straight run and only ever pushes
    along it. Solved for the root's straight run b and the big turn's radius r1."""
    goal = np.array([SLOT_C, ENTRY_Z + TIP_GAP])
    b, r1 = 0.0034, 0.0389
    for _ in range(30):
        p = _dloop(b, r1)[-1][[0, 2]]
        err = p - goal
        if np.max(np.abs(err)) < 1e-9:
            break
        jac = np.zeros((2, 2))
        for j, (db, dr) in enumerate(((1e-6, 0.0), (0.0, 1e-6))):
            jac[:, j] = (_dloop(b + db, r1 + dr)[-1][[0, 2]] - p) / 1e-6
        b, r1 = np.array([b, r1]) - np.linalg.solve(jac, err)
    return _dloop(b, r1)


def straight_run(points: np.ndarray, tol_deg: float = 0.5) -> float:
    """How far back from the tip (m) a polyline runs straight (every link within ``tol_deg`` of the last one)."""
    seg = np.diff(points, axis=0)
    u = seg / np.linalg.norm(seg, axis=1)[:, None]
    last = u[-1]
    n = 0
    for v in u[::-1]:
        if math.degrees(math.acos(np.clip(v @ last, -1.0, 1.0))) > tol_deg:
            break
        n += 1
    return n * LINK


def free_points(n_links: int = N_LINKS) -> np.ndarray:
    """The task's start shape: straight along +x_h (head frame)."""
    return ROOT + np.outer(np.arange(n_links + 1) * LINK, [1.0, 0.0, 0.0])


def _loop_centre(pts: np.ndarray) -> np.ndarray:
    """The centre of the circle through the loop's quarter points (head frame)."""
    a, b, c = pts[8], pts[16], pts[24]
    x1, z1, x2, z2, x3, z3 = a[0], a[2], b[0], b[2], c[0], c[2]
    d = 2 * (x1 * (z2 - z3) + x2 * (z3 - z1) + x3 * (z1 - z2))
    ux = ((x1**2 + z1**2) * (z2 - z3) + (x2**2 + z2**2) * (z3 - z1) + (x3**2 + z3**2) * (z1 - z2)) / d
    uz = ((x1**2 + z1**2) * (x3 - x2) + (x2**2 + z2**2) * (x1 - x3) + (x3**2 + z3**2) * (x2 - x1)) / d
    return np.array([ux, 0.0, uz])


@dataclass(frozen=True)
class Build:
    name: str
    fixed: bool  # the head welded (the jig) or free
    head_pos: tuple  # world
    head_rot: np.ndarray = field(repr=False)
    points: np.ndarray = field(repr=False)  # head frame, one more than the strap's links (N_LINKS but for lift25/30)
    why: str = ""
    passage_y: float = PASSAGE_Y  # half the slot's width (the "tight" build narrows it)

    @property
    def n_links(self) -> int:
        return len(self.points) - 1

    @property
    def tie_l(self) -> float:
        """The strap's length, head face to tip (TIE_L but for the longer ties of round 24)."""
        return self.n_links * LINK


def tip_from_head(b: Build) -> float:
    """The straight strap's tip from the head's centre, along x_h (TIP_FROM_HEAD for a 200 mm tie)."""
    return HEAD[0] / 2 + b.tie_l


def _jig() -> Build:
    pts = jig_points()
    hull_c = np.mean([c.xy for c in CORDS], axis=0)
    pos = np.array([hull_c[0], hull_c[1], Z_TIE]) - R_JIG @ JIG_BUNDLE_H
    return Build("jig", True, tuple(float(v) for v in pos), R_JIG, pts,
                 "head welded beside the cords, strap a relaxed open loop round them, tip 4 mm short of the slot")


def _free() -> Build:
    pos = np.array([0.300, -0.160, TOP + HEAD[2] / 2 + 0.3 * MM])
    return Build("free", False, tuple(float(v) for v in pos), R_FREE, free_points(),
                 "the tie lying flat on the console, free (the task's start)")


WRAP_CUT = 21  # wrap: the jig loop's points up to this one (round the back of the cords), then straight


def wrap_points() -> np.ndarray:
    """The wrap start shape (head frame): the jig's loop as far as the back of the cords (point WRAP_CUT, where the
    strap comes back toward the robot past their +y side), then straight on, the way it was going, to the tip."""
    p = jig_points()
    v = (p[WRAP_CUT] - p[WRAP_CUT - 1]) / np.linalg.norm(p[WRAP_CUT] - p[WRAP_CUT - 1])
    tail = [p[WRAP_CUT] + k * LINK * v for k in range(1, N_LINKS - WRAP_CUT + 1)]
    return np.vstack([p[: WRAP_CUT + 1], tail])


def _wrap() -> Build:
    j = _jig()
    return Build("wrap", True, j.head_pos, j.head_rot, wrap_points(),
                 "head held as on the jig; the strap slipped behind the cords, its tail straight, toward the robot")


TIGHT_Y = 2.8 * MM  # the tight slot: 5.6 mm wide, 0.4 mm a side on the 4.8 mm strap, 1.3 mm on the 3 mm tip


def _tight() -> Build:
    j = _jig()
    return Build("tight", True, j.head_pos, j.head_rot, j.points,
                 "the jig with the slot narrowed to 5.6 mm, close to a real tie's", passage_y=TIGHT_Y)


# -- the lift start (round 19) ------------------------------------------------------------------------------------------
# The user's start: "one arm lifts the cord, the other slides the tie under it." The three cords come out under the TV,
# hang down by the wall, turn onto the console and run along it, parallel to the wall, as a loose bundle into the power
# strip at the left. In the sim they are labkit's Harness "cords": three jointed cords anchored LIFT_ANCHOR_Z up (drawn
# as fixed rods above that, up to the TV), ending on the strip, a free rigid body on the console. The tie lies flat on
# the console to the front right, strap toward +y.
LIFT_WALL_X = 0.445  # the cords hang down here, under the TV (its front is at 0.420, the wall at 0.460)
LIFT_HANG_Y = 0.100  # on the left: the left arm reaches the console at y >= 0, the right at y <= 0
LIFT_ANCHOR_Z = TOP + 0.150
LIFT_BEND_R = 0.015  # off the wall onto the console (in x-z), then straight toward the robot into the strip
LIFT_STRIP = dict(x=(0.195, 0.250), y=(LIFT_HANG_Y - 0.140, LIFT_HANG_Y + 0.140), z=(TOP, TOP + 0.040))
LIFT_SEGMENTS = 20  # links per cord: 3 x 20 + anchor + strip = 62 (PhysX: 64)
LIFT_CORDS = (  # name, radius, offset across the bundle (+: left of the way to the strip, i.e. -y), kg/m, colour
    ("power", 3.50 * MM, -5.40 * MM, 0.045, (0.05, 0.05, 0.05)),
    ("hdmi", 3.65 * MM, 1.95 * MM, 0.055, (0.09, 0.09, 0.10)),
    ("charger", 1.75 * MM, 7.55 * MM, 0.015, (0.93, 0.93, 0.92)),
)
LIFT_K = 0.08  # N m/rad per cord joint: a PVC cord's EI, 2-3e-3 N m^2, over a 17 mm link (soft side)
LIFT_YIELD = 0.03  # rad: a cord keeps some of a bend
LIFT_STRIP_MASS = 0.35
LIFT_AT = np.array([0.300, LIFT_HANG_Y])  # where the left hand pinches the bundle and lifts it
LIFT_H = 0.060
SLIDE_XY = np.array([0.330, LIFT_HANG_Y])  # the strap goes under the bundle here, 30 mm wall-ward of the pinch
#                                            (further out the right forearm, carrying the tie low, comes within 2 cm of the wall)
SLIDE_DIR = np.array([0.0, 1.0])  # ... along +y, the way the tie's strap points as it lies (no turn)
SLIDE_TIP = (LIFT_HANG_Y - 0.030, LIFT_HANG_Y + 0.040)  # the tip (y): 20 mm short of the bundle, then 30 mm past it
LIFT_TIE_HEAD = (SLIDE_XY[0], -0.240)
TIP_FROM_HEAD = HEAD[0] / 2 + TIE_L  # the straight strap's tip from the head's centre, along x_h
BACK_OFF_X = 0.220  # both hands come back toward the robot, up, before ready (the return otherwise passes the TV)

# -- the lift start, stage B (round 20) -----------------------------------------------------------------------------------
# Stage A's pick is a down grip on the flat head: its palm and 27 mm-wide fingers sit over the slot's entry, so that hand
# can't hold the head for threading, and a flat head can't be gripped from behind. So the hands change grips first: the
# right lifts the tie by its head, the left holds the strap B_HOLD_S out (its fingers 55 mm clear of the right's), the
# right lets go and takes the head again from behind (pointing +y, fingers on its sides: both slot faces free). Then
# stage A's lift and slide, the head pitched down so the strap leaves it downward; the left lowers the bundle onto the
# strap; the right levels the head beside the bundle, slot vertical; the left takes the tail where it lies past the
# bundle, brings it up and over while turning from a down grip to pointing -y (the tail turns from +y to straight down),
# and pushes the tip down through the slot. Reach (yam_arena IK, 22 Nov): the left's down grip closing -x reaches
# y -0.06..0.12 at any height; its level -y grip z >= 0.08 near y 0.06; the right's grip from behind tilted 15-45
# degrees down reaches y -0.16..0.06 at z 0.02-0.08 (level only above about 0.05; round 23: y 0.07 at z 0.033 tilted 5).
# Round 23: low down, the left's level -y grip brings its upper arm (link3) within 3-9 mm of the power strip at x
# 0.33-0.345 (20 mm needs its TCP 75 mm up); tilted 10 degrees down it keeps 26 mm, its wrist higher. So the tail is held
# 10 degrees off vertical (b_phi) and the head pitched to match (b_insert_pitch -10: the slot along the tail).
# Round 23 (job 97): the grip from behind held the head on the repeat and on stiff15 but lost it to small loads: its
# pads' top edge was level with the head's top corner and the fingertips 1.2 mm past its front (b-thread: the head sat
# 2.3 mm high and closing squeezed it out upward; stiff15: lifting the tail raised it 2.8 mm out over the pads; the repeat:
# the strap, too short to reach the left, pulled it 1.7 mm toward the fingertips). Now 4 mm of pad over it, 3.2 past it.
B_GRIP_TILT = 15.0  # deg: the right's grip from behind is taken tilted this far down on the level head
B_BACK_IN = 0.005  # ... the head's centre this far along the approach from the TCP (fingertips 8 mm past it; round 22: 7)
B_BACK_UP = 0.006  # ... and this far across, up (its pads' top edge 4 mm over the head's top corner; round 22: 10, 0)
B_TAIL_DEPTH = 0.0125  # the left takes the tail off the console fingertips 0.3 mm up: the strap 12.5 mm past its TCP
B_TIP_GAP = 0.004  # the tip lines up this far before the entry
B_ABOVE = 0.005  # ... coming in from this much further (round 23: 15 took 10 mm more strap than there is to spare)
B_STEER_CAP = np.array([0.012, 0.006, 0.006])  # the left's shift onto the tip's offset from the slot's centre line,
#   at most (x, y, z; along y its fingers near the right's)
B_STEER_LO = np.array([-0.012, -0.002, -0.006])  # round 23: toward the right (-y) only 2 mm: further, the left's
#   fingertips meet the right's finger behind its pad whatever the heights (round 23's grip, 1 mm short at -6)
B_DEPTH_CAP = 0.003  # ... and its extra push, at most (its pads stay 2 mm over the right's)
# round 21: ThreadSteering's snaps onto the real strap, head and tail move a hand at most this far (lo, hi, world m);
# round 20's were open and drove the left 57 mm, then 203 mm, into the right once the tie had fallen. Round 22: the head
# may be 12 mm up (job 96: on the heavier strap it rose 7.7 mm once the left alone held it; capped at 6, the right's
# fingers came in under it and never closed on it)
B_SNAP = {"strap": (np.array([-0.006, -0.012, -0.020]), np.array([0.006, 0.012, 0.005])),
          "head": (np.array([-0.006, -0.008, -0.010]), np.array([0.006, 0.008, 0.012])),
          "tail": (np.array([-0.006, -0.025, 0.0]), np.array([0.006, 0.025, 0.0]))}
THREAD_STRAP_LABELS = ("to the strap", "down to the strap", "hold strap", "let go strap", "up from strap")
THREAD_HEAD_LABELS = ("around", "behind head", "onto head", "grip head")
THREAD_TAIL_LABELS = ("to the tail", "down to the tail", "take tail", "lift tail")
THREAD_STEER_LABELS = ("above head", "align", "push", "push hold", "let go tail", "back off tail")
THREAD_PUSH_LABELS = ("push", "push hold")


def lift_cord_path(offset: float, radius: float, n: int = 2000) -> np.ndarray:
    """One cord's centre line, dense, world: down by the wall from LIFT_ANCHOR_Z, a quarter bend onto the console
    (heading -x), then straight along the console into the strip's back face. It lies on the console
    (z = TOP + radius + 0.2 mm), ``offset`` across the bundle (+: -y)."""
    z = TOP + radius + 0.2 * MM
    y0 = LIFT_HANG_Y - offset
    bx = LIFT_WALL_X - LIFT_BEND_R
    hang = np.column_stack([np.full(n, LIFT_WALL_X), np.full(n, y0), np.linspace(LIFT_ANCHOR_Z, z + LIFT_BEND_R, n)])
    th = np.linspace(0.0, math.pi / 2, n)
    bend = np.column_stack([bx + LIFT_BEND_R * np.cos(th), np.full(n, y0), z + LIFT_BEND_R - LIFT_BEND_R * np.sin(th)])
    run = np.column_stack([np.linspace(bx, LIFT_STRIP["x"][1], n), np.full(n, y0), np.full(n, z)])
    return np.vstack([hang, bend[1:], run[1:]])


def equal_chords(path: np.ndarray, segments: int) -> np.ndarray:
    """``segments`` + 1 points on ``path`` (dense polyline) with equal chords, from its first point to its last."""
    path = np.asarray(path, float)
    seg = np.diff(path, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])

    def at(s):
        i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1))
        f = (s - cum[i]) / max(cum[i + 1] - cum[i], 1e-15)
        return path[i] + f * seg[i]

    def walk(c):
        """Arc positions of the points at chord c (None once past the end)."""
        out, s = [0.0], 0.0
        for _ in range(segments):
            p0, lo, hi = at(s), s, cum[-1]
            if np.linalg.norm(at(hi) - p0) < c:
                return out, False
            for _ in range(60):  # the chord grows along the path: bisect for the point at distance c
                mid = (lo + hi) / 2
                if np.linalg.norm(at(mid) - p0) < c:
                    lo = mid
                else:
                    hi = mid
            s = hi
            out.append(s)
        return out, True

    lo, hi = 0.0, cum[-1] / segments * 1.0001
    for _ in range(80):
        c = (lo + hi) / 2
        pos, ok = walk(c)
        if ok and pos[-1] < cum[-1]:
            lo = c
        else:
            hi = c
    pos, _ = walk(lo)
    pts = np.array([at(s) for s in pos])
    pts[-1] = path[-1]
    return pts


def lift_cord_points(offset: float, radius: float) -> np.ndarray:
    return equal_chords(lift_cord_path(offset, radius), LIFT_SEGMENTS)


def _nearest_on(poly, xy):
    """The point of polyline ``poly`` nearest ``xy`` horizontally, and that horizontal distance."""
    p = np.asarray(poly, float)
    a, ab = p[:-1], np.diff(p, axis=0)
    den = np.maximum(np.einsum("ij,ij->i", ab[:, :2], ab[:, :2]), 1e-18)
    t = np.clip(np.einsum("ij,ij->i", np.asarray(xy, float)[:2] - a[:, :2], ab[:, :2]) / den, 0.0, 1.0)
    q = a + t[:, None] * ab
    d = np.linalg.norm(q[:, :2] - np.asarray(xy, float)[:2], axis=1)
    i = int(np.argmin(d))
    return q[i], float(d[i])


def cord_height_at(polys: dict, xy) -> float:
    """The highest cord's height (world z) where it passes nearest ``xy`` (horizontally)."""
    return max(float(_nearest_on(p, xy)[0][2]) for p in polys.values())


def strap_under_cords(strap, polys: dict, radii: dict, margin: float = 3.0 * MM) -> tuple:
    """Where the strap passes a cord (a strap point within the cord's radius + ``margin`` of it, horizontally): how
    many such points, and the strap's height over the cord there, least and most (m; negative: under it)."""
    dz = []
    for name, p in polys.items():
        for pt in np.asarray(strap, float):
            q, d = _nearest_on(p, pt)
            if d < radii[name] + margin:
                dz.append(float(pt[2] - q[2]))
    return (len(dz), min(dz), max(dz)) if dz else (0, None, None)


def lift_strip_centre() -> np.ndarray:
    s = LIFT_STRIP
    return np.array([sum(s["x"]) / 2, sum(s["y"]) / 2, sum(s["z"]) / 2])


def _lift(name: str = "lift", n_links: int = N_LINKS) -> Build:
    pos = np.array([LIFT_TIE_HEAD[0], LIFT_TIE_HEAD[1], TOP + HEAD[2] / 2 + 0.3 * MM])
    return Build(name, False, tuple(float(v) for v in pos), R_FREE, free_points(n_links),
                 f"the cords lying on the console in a loose bundle; a {n_links * LINK * 1000:.0f} mm tie flat beside "
                 "them (round 19; round 24: 250 and 300 mm)")


# Round 24: longer ties for the lift start (250 and 300 mm, 40 and 48 links; standard cord-tie lengths). Job 98 showed why
# 200 mm can't thread here: the left's fingers close across the strap's width, the axis it bends about, so they hold
# its angle by friction alone (12 N edge-on). Brought over the bundle and down onto the slot, the strap has to turn
# from heading -y to heading down above the fingers; with 14 mm of strap to spare it turned at the one joint where it
# left the pads (100 degrees, the joint's stop) and the rest slipped in the fingers: the tip pointed 33 degrees below
# level, 20 mm off the slot line and 30 mm above it. A smooth turn above the fingers takes about 50 mm more strap than
# the tightest path, and bends the strap about 0.04 N m where it leaves the pads, a quarter of what the pads hold.
LIFT_BUILDS = ("lift", "lift25", "lift30")
BUILDS = {"jig": _jig(), "free": _free(), "tight": _tight(), "wrap": _wrap(), "lift": _lift(),
          "lift25": _lift("lift25", 40), "lift30": _lift("lift30", 48)}


# -- the clip stand (round 13): what holds the head in the fixed builds ---------------------------------------------------
# The jig welds the head where it starts; the scene shows what does that in a room: a small spring clip gripping the
# head's top and bottom faces from the robot's side, on a 5 mm steel rod standing on a small base on the console, in
# front of the power strip. The slot's two faces (+-y) and the strap's root (+x face) stay open. CLIP_GAP keeps the
# welded head out of contact with it. Round 13's offline check: the fingers' fat envelope stays 3.2 mm or more from it.
CLIP_GAP = 0.3 * MM
CLIP_JAW = 1.5 * MM  # each jaw's thickness, over the head's top and under its bottom
CLIP_BACK = 2.0 * MM  # the clip's back, behind the head's -x face (the robot's side)
STAND_R = 2.5 * MM  # the rod
STAND_BASE = dict(dx=(-25.0 * MM, 5.0 * MM), dy=(-15.0 * MM, 15.0 * MM), h=4.0 * MM)  # about the rod's axis


def stand_parts(build: Build) -> list:
    """The clip stand of a fixed build ([] for a free one), as (name, kind, geometry): a box as its (lo, hi) world
    corners, the rod as ((x, y), radius, (z0, z1))."""
    if not build.fixed:
        return []
    hp = np.asarray(build.head_pos, float)
    half = np.abs(np.asarray(build.head_rot, float)) @ (HEAD / 2)  # the head's half extents along world x, y, z
    lo, hi = hp - half, hp + half
    g, j, k = CLIP_GAP, CLIP_JAW, CLIP_BACK
    zb = (lo[2] - g - j, lo[2] - g)
    zt = (hi[2] + g, hi[2] + g + j)
    x0 = lo[0] - g - k
    b = STAND_BASE
    return [
        ("clip_bottom", "box", (np.array([x0, lo[1], zb[0]]), np.array([hi[0], hi[1], zb[1]]))),
        ("clip_top", "box", (np.array([x0, lo[1], zt[0]]), np.array([hi[0], hi[1], zt[1]]))),
        ("clip_back", "box", (np.array([x0, lo[1], zb[0]]), np.array([lo[0] - g, hi[1], zt[1]]))),
        ("stand_rod", "cyl", (hp[:2].copy(), STAND_R, (TOP + b["h"], zb[0]))),
        ("stand_base", "box", (np.array([hp[0] + b["dx"][0], hp[1] + b["dy"][0], TOP]),
                               np.array([hp[0] + b["dx"][1], hp[1] + b["dy"][1], TOP + b["h"]]))),
    ]


def stand_points(build: Build, rod_top: float = 0.06) -> np.ndarray:
    """Surface samples of the clip and the rod's top ``rod_top`` (what fingers working at the head could touch)."""
    pts = []
    for name, kind, geo in stand_parts(build):
        if kind == "box":
            lo, hi = geo
            if name == "stand_base":
                continue
            g = [np.linspace(lo[i], hi[i], 7) for i in range(3)]
            for a in g[0]:
                for b_ in g[1]:
                    for c in g[2]:
                        p = np.array([a, b_, c])
                        if np.any(np.isclose(p, lo) | np.isclose(p, hi)):
                            pts.append(p)
        else:
            xy, r, (z0, z1) = geo
            for z in np.linspace(max(z0, z1 - rod_top), z1, 25):
                for t in np.linspace(0, 2 * math.pi, 24, endpoint=False):
                    pts.append([xy[0] + r * math.cos(t), xy[1] + r * math.sin(t), z])
    return np.array(pts, float).reshape(-1, 3)


def world(build: Build, p_h) -> np.ndarray:
    return np.asarray(build.head_pos) + np.asarray(p_h, float) @ build.head_rot.T


def start_polyline(build: Build) -> np.ndarray:
    return world(build, build.points)


# -- measuring a tie (what the scorer and the probe read) -----------------------------------------------------------------
def to_head(points, head_pos, head_quat) -> np.ndarray:
    r = quat_xyzw_to_matrix(head_quat)
    return (np.asarray(points, float) - np.asarray(head_pos, float)) @ r


def _in_passage(p, slack: float = 1.0 * MM, passage_y: float = PASSAGE_Y) -> bool:
    return bool(PASSAGE_X[0] - slack <= p[0] <= PASSAGE_X[1] + slack and abs(p[1]) <= passage_y + slack)


def _crossings(ph: np.ndarray, z: float) -> list:
    """(segment index, fraction, point) wherever the polyline crosses the plane z_h = z going down (-z_h)."""
    out = []
    for i in range(len(ph) - 1):
        a, b = ph[i], ph[i + 1]
        if a[2] > z >= b[2]:
            f = (a[2] - z) / (a[2] - b[2])
            out.append((i, f, a + f * (b - a)))
    return out


def _arc(ph: np.ndarray, i0: int, f0: float, i1: int, f1: float) -> float:
    """Arc length along the polyline from (segment i0, fraction f0) to (i1, f1)."""
    seg = np.linalg.norm(np.diff(ph, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return float((cum[i1] + f1 * seg[i1]) - (cum[i0] + f0 * seg[i0]))


@dataclass
class Threading:
    entered: bool  # the strap crosses the entry face inside the passage
    through: float  # m of strap past the exit face (the tail out of the head); 0 if it doesn't come out
    loop: float | None  # m of strap from the root to the entry face, or None
    slack: float | None  # loop + the closure through the head - the bundle's snug perimeter
    slack_rel: float | None  # loop - the tightest loop the cords allow round this head pose (what scores)
    tip_h: np.ndarray  # the tip's point (head frame)
    tip_off_mm: tuple  # (x_h, y_h) of the tip from the slot's centre line, mm
    enclosed: tuple  # which cords pass through the loop
    closed: bool


HEAD_CLOSURE = float(np.linalg.norm(ROOT[[0, 2]] - np.array([SLOT_C, ENTRY_Z])))


def _cross2(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(map(tuple, points))

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and _cross2(np.subtract(h[-1], h[-2]), np.subtract(p, h[-2])) <= 0:
                h.pop()
            h.append(p)
        return h

    lower, upper = half(pts), half(reversed(pts))
    return np.array(lower[:-1] + upper[:-1])


def bundle_perimeter(offset: float = T / 2) -> float:
    """The perimeter of the cords' convex hull grown by ``offset`` (the strap's centre line hugging the bundle)."""
    ring = []
    for c in CORDS:
        for a in np.linspace(0, 2 * math.pi, 180, endpoint=False):
            ring.append(np.asarray(c.xy) + (c.diameter / 2 + offset) * np.array([math.cos(a), math.sin(a)]))
    h = _hull(np.array(ring))
    return float(np.sum(np.linalg.norm(np.diff(np.vstack([h, h[:1]]), axis=0), axis=1)))


SNUG_SLACK = 8.0 * MM  # full credit: at most this much strap beyond hugging the bundle (round 1 calibrates it)
LOOSE_SLACK = 25.0 * MM
TAIL_MIN = 2.0 * MM  # "closed": the tip out of the exit face at least this far
SLID_Z = TOP + 0.100  # a head below this has slid down the cords
CLEAR = 0.030  # every gripper point this far from the tie
STILL = 0.010  # m/s


def tail_snug() -> float:
    """The tail length (ruler, head face to tip) that leaves SNUG_SLACK: what a person measures."""
    return TIE_L - (HEAD[2]) - (bundle_perimeter() - HEAD_CLOSURE + SNUG_SLACK)


def _point_in_poly(p, poly) -> bool:
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def enclosed_cords(ph: np.ndarray, entry, head_pos, head_quat) -> tuple:
    """The cords whose axes pass through the loop: the polyline from the root to the entry point, closed through
    the head, in the head's x_h-z_h plane. A cord's axis meets that plane where y_h = 0, within its hanging length."""
    if entry is None:
        return ()
    i, f, p = entry
    loop = np.vstack([ph[: i + 1], p[None, :]])[:, [0, 2]]
    r = quat_xyzw_to_matrix(head_quat)
    out = []
    for c in CORDS:
        base = (np.array([c.xy[0], c.xy[1], CORD_Z[0]]) - np.asarray(head_pos, float)) @ r
        d = np.array([0.0, 0.0, 1.0]) @ r
        if abs(d[1]) < 0.3:
            continue
        t = -base[1] / d[1]
        if not 0.0 <= t <= CORD_Z[1] - CORD_Z[0]:
            continue
        q = base + t * d
        if _point_in_poly((q[0], q[2]), loop):
            out.append(c.name)
    return tuple(out)


def threading(polyline, head_pos, head_quat, passage_y: float = PASSAGE_Y) -> Threading:
    ph = to_head(polyline, head_pos, head_quat)
    tip = ph[-1]
    exits = [c for c in _crossings(ph, EXIT_Z) if _in_passage(c[2], passage_y=passage_y)]
    entries = [c for c in _crossings(ph, ENTRY_Z) if _in_passage(c[2], passage_y=passage_y)]
    entry = None
    through = 0.0
    if exits:
        i, f, _ = exits[-1]
        before = [e for e in entries if (e[0], e[1]) < (i, f)]
        if not before:
            # Round 48 (job 133): a tight loop let go of cocks the free head, and the strap then comes into the
            # mouth on a slant over the lead-in; the chord between two of its joints crossed the entry's plane
            # 4.7 mm from the slot's line, outside the passage's window, and a tie drawn 297 mm through its head
            # read as not threaded. With the tail out of the exit the strap is in the slot: its way in is its
            # last crossing of the entry's plane within the head's face.
            wide = [e for e in _crossings(ph, ENTRY_Z) if abs(e[2][0]) <= HEAD[0] / 2 + 2.0 * MM
                    and abs(e[2][1]) <= HEAD[1] / 2 and (e[0], e[1]) < (i, f)]
            before = wide[-1:]
        entry = before[-1] if before else None
        through = _arc(ph, i, f, len(ph) - 2, 1.0)
    elif entries:
        entry = entries[-1]
    loop = None if entry is None else _arc(ph, 0, 0.0, entry[0], entry[1])
    slack = None if loop is None else loop + HEAD_CLOSURE - bundle_perimeter()
    enc = enclosed_cords(ph, entry, head_pos, head_quat) if exits else ()
    closed = bool(through >= TAIL_MIN and entry is not None and len(enc) == len(CORDS))
    slack_rel = None if loop is None else loop - min_loop_at(head_pos, head_quat)
    return Threading(entered=entry is not None, through=through, loop=loop, slack=slack, slack_rel=slack_rel, tip_h=tip,
                     tip_off_mm=(round((tip[0] - SLOT_C) / MM, 2), round(tip[1] / MM, 2)), enclosed=enc, closed=closed)


def _seg_dist(p, a, b) -> float:
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-18), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def hand_clearance(points, polyline, head_pos, head_quat) -> float:
    """The smallest distance from any gripper point to the strap's centre line (less half its width) or the head's
    box."""
    poly = np.asarray(polyline, float)
    best = float("inf")
    r = quat_xyzw_to_matrix(head_quat)
    for p in points:
        p = np.asarray(p, float)
        d = min(_seg_dist(p, poly[i], poly[i + 1]) for i in range(len(poly) - 1)) - W / 2
        q = np.abs((p - np.asarray(head_pos)) @ r) - HEAD / 2
        d_head = float(np.linalg.norm(np.maximum(q, 0.0)) + min(np.max(q), 0.0))
        best = min(best, d, d_head)
    return max(best, 0.0)


def task_score(polyline, head_pos, head_quat, hand_clear: float, speed: float, passage_y: float = PASSAGE_Y) -> float:
    """0 unless every gripper point is 3 cm clear of the tie and it is at rest (< 1 cm/s) and closed round all three
    cords. Then 1.0 snug (the loop at most SNUG_SLACK longer than the tightest the cords allow round the head where it
    is) with the head still up the hanging run; 0.5 within LOOSE_SLACK; 0.25 otherwise."""
    if not (hand_clear >= CLEAR and speed < STILL):
        return 0.0
    th = threading(polyline, head_pos, head_quat, passage_y)
    if not th.closed:
        return 0.0
    up = float(head_pos[2]) >= SLID_Z
    if th.slack_rel is not None and th.slack_rel <= SNUG_SLACK and up:
        return 1.0
    if th.slack_rel is not None and th.slack_rel <= LOOSE_SLACK and up:
        return 0.5
    return 0.25


# -- the lying-cords start's score (round 26) -------------------------------------------------------------------------
# The reviewed task scored the tail's length to 0.1 mm against 134.8 and 117.8 mm. Here the strap carries two marks,
# and the score is which of them has come out of the head. The loop comes to rest only where a strap joint sits at
# the pawl, so the tail rests at whole links: on the 250 mm tie 187.5 mm out is the tightest the three cords should
# allow (the 200 mm tie's 137.5 on the same three cords, 5 mm of slack), 193.8 if they pack closer, then 181.3, 175.0
# and so on. Each mark sits halfway between two rests, 3.1 mm from either: read by eye, not to a fraction of a mm.
MARK_FULL_BACK = 10.5 * LINK  # the full mark, 65.6 mm from the head end of the strap: 184.4 mm from the tip on the
#                               250 mm tie
MARK_HALF_BACK = 13.5 * LINK  # the half mark, 84.4 mm: 165.6 mm from the tip (three more links of slack, 19 mm)


def lift_marks(tie_l: float) -> tuple:
    """(full, half): how much tail is out of the head's exit face when each mark is just out of it (m)."""
    full, half = MARKS_BACK.get(int(round(tie_l / LINK)), (MARK_FULL_BACK, MARK_HALF_BACK))
    return tie_l - full, tie_l - half


MARKS_BACK = {59: (59 * LINK - 0.290, 59 * LINK - 0.270)}  # links in the strap: (full, half) marks' distances from
#   the strap's head end, where a tie of that length has its own (else MARK_FULL_BACK, MARK_HALF_BACK). The 370 mm
#   tie of the hanging cords: marks 290 mm and 270 mm from the tip, whole millimetres on a steel rule (round 47 had
#   them at 300 and 280). With the 290 mm mark at the head's face 73 mm of strap is left in the loop (hugging the
#   three cords takes 36). Round 48: with the head in the left's fingers, 11 mm off the cords, the loop comes taut
#   on them at about 300 mm drawn; drawn on, the arms and the grips give, and let go the tail runs back to where
#   the loop is slack again (jobs 131, 132, 133: drawn 309.3, 301.5, 307.5 mm, at rest 303.2, 296.5, 297.1). A full
#   mark at 300 mm sat inside that scatter; at 290 mm the three runs clear it by 6.5 mm or more.


def lift_score(polyline, head_pos, head_quat, cord_polys: dict, hand_clear: float, speed: float,
               passage_y: float = PASSAGE_Y) -> float:
    """The lying-cords start. 0 unless every gripper point is 3 cm clear of the tie, it is at rest (< 1 cm/s) and it
    is closed round all three cords (the tail at least 2 mm out of the head's far side, every cord through the loop).
    Then 1.0 if the full mark is out of the head, 0.5 if the half mark is, 0.25 otherwise."""
    if not (hand_clear >= CLEAR and speed < STILL):
        return 0.0
    th = threading(polyline, head_pos, head_quat, passage_y)
    if th.through < TAIL_MIN or not th.entered:
        return 0.0
    if len(enclosed_lift(polyline, head_pos, head_quat, cord_polys, passage_y)) != len(LIFT_CORDS):
        return 0.0
    full, half = lift_marks(arc_length(polyline))
    return 1.0 if th.through >= full else 0.5 if th.through >= half else 0.25


# -- the probe's plans ----------------------------------------------------------------------------------------------------
# A plan: {"left": [step...], "right": [step...]}, each step (seconds, pos, fingers, rot, label); rot a 3x3 matrix or
# "ready". Finger names (task.py maps them): open (94 mm), wide (14 mm gap), loose (6 mm: let go of the strap without
# swinging a finger into the cords), closed (pads together: 3 N a side on a
# flat 1.2 mm strap, 12 N edge-on on its 4.8 mm width), head (1 mm gap on the 9 mm head: about 20 N a side).
FINGER_GAP = {"open": 94.0, "wide": 14.0, "loose": 6.0, "closed": 0.0, "light": 0.6, "head": 1.0, "head_light": 5.0,
              "cords": 30.0, "head_open": 24.0, "soft": 4.0, "soft3": 3.2, "soft5": 4.4}
# soft, soft3, soft5 (round 26): the left's pinch on the strap's 4.8 mm width eased to about 2 N, 4 N and 1 N a side
# (closed: 12), so the strap can pivot in it about the fingers' closing axis while it is carried over the bundle.
# light: half the squeeze on a flat strap (0.3 mm short of it a side: about 1.5 N). head_light: about 10 N a side on
# the head instead of 20. head_open (round 22): the right's approach to the head from behind, 7.5 mm a side of the 9 mm
# head (wide leaves 2.5: job 96's hand lagged 4 mm sideways coming in and its fingertip pushed the head 10 mm).
PAD_PAST_TCP = 0.013
PROTRUDE = 18.0 * MM  # the second push leaves this much tail out of the exit
PROTRUDE3 = 30.0 * MM  # round 4: a third push leaves this much (round 3's pulling hand found 9 mm, mostly the narrow
#                        tip link, and slipped off after 4 mm)
FIRST_OUT = 3.0 * MM  # the first stroke stops with the tip this far out of the exit
REGRIP = 20.0 * MM  # between strokes the open hand slides this far back along the (now straight) strap
GRIP_FROM_TIP = 50.0 * MM  # where the inserting hand holds the strap: on the D-loop's straight run into the slot
GRIP_DEPTH = 8.0 * MM  # the strap sits this far past the TCP (round 3: deeper, fingertips 5 mm past the strap)
TIP_STANDOFF = 3.0 * MM  # the end-on pull's fingertips stop this short of the exit face
EDGE_UP = np.array([0.0, 0.0, 1.0])
STEER_LABELS = ("align", "insert", "insert hold")
FLAT_TILT = 35.0  # deg: the flat grip's tilt from straight down
STEER_EVERY = 2  # control steps (0.1 s)
STEER_GAIN = 0.6
STEER_CAP = 6.0 * MM
# Round 8 (yaw steering): round 7's wrap ended with the tail turned about 10 degrees in the edge grip, so the tip sat
# 9-10 mm off the slot's line, more than the 6 mm cap could shift, and never went in. The probe now also turns the
# inserting hand about the vertical through the strap in its fingers until the last YAW_LOOK of the tail points along
# the slot's way.
YAW_LOOK = 12.0 * MM
YAW_GAIN = 0.5
YAW_CAP = math.radians(15.0)
# Closed-loop depth (round 2): in round 1 the strap slid 6-7 mm along the fingers on every push, so the second push
# keeps going, up to DEPTH_CAP further, until the probe measures PROTRUDE out of the exit.
DEPTH_LABELS = ("insert 2", "insert 2 hold", "insert 3", "insert 3 hold")
DEPTH_TARGET = {"insert 2": PROTRUDE, "insert 2 hold": PROTRUDE, "insert 3": PROTRUDE3, "insert 3 hold": PROTRUDE3}
# Round 5 (hand-over): the single push is closed-loop on "insert" too, to HANDOVER_OUT out.
HANDOVER_OUT = 22.0 * MM
HANDOVER_DEPTH_LABELS = ("insert", "insert hold")
PULL_STRAIGHT = 60.0 * MM  # the hand-over pull runs straight out of the slot this far, then turns pull_angle
DEPTH_GAIN = 0.5
DEPTH_CAP = 12.0 * MM
# Round 16: the demo's pulling hand on a closed loop too. Round 15's pulls fell short (a6-slow, a6-deep: the pads slipped
# 6 mm along the tail in a stall and 6 mm past snug ended 132 mm out) or, 30 mm past snug, slid the pads off the tip
# (round 14). Now the last stroke goes extra_pull past snug, then "tighten": the hand goes on along its pull at
# TIGHTEN_RATE while less than tr.tighten_slack short of snug is out (at most tr.tighten further). "pull 2" shifts with
# it so ArmPlan's interpolation (which starts at the previous waypoint's goal) can't lag.
# Round 17: the loop comes to rest only where a strap joint sits at the pawl: on the jig 137.5 mm out (5.3 mm slack, 1.0)
# or a link (6.25 mm) back, 131.3 (11.5 mm, 0.5). Every let-go in jobs 38-71 at 137.9 mm or more ended at 137.5; every
# one at 137.3 or less fell to 131.3 or further, brake or no brake (one exception, e-slow: 136.4 -> 136.1). Round 16
# tightened to 137.3 (5.5 mm short of snug): the stall just before that joint passes the pawl. Its gain on a 0.1 mm
# shortfall barely moved the hand, which sat 10 mm behind its target, held by the strap. So: past the joint, at a rate.
TIGHTEN_LABELS = ("tighten",)
TIGHTEN_SHIFT_LABELS = ("pull 2", "tighten", "hold snug", "release 2", "clear 2")
TIGHTEN_SLACK = 4.5 * MM  # 138.3 mm out on the jig
TIGHTEN_RATE = 12.0 * MM  # per second, while short
TIGHTEN_S = 2.0
JIG_REST = 137.5 * MM  # where the jig's loop rests, its last joint at the pawl (jobs 38-73)
# Round 18 "unstick": 4 of round 17's 12 pulls (none of them a fresh launch's first) stalled at once, 25.6 mm out (a
# strap joint on the slot's entry, the strap against its far wall). The pulling hand holds only the strap's last 7 mm:
# it ran on until it was 21 mm ahead of the strap (its pads slip at about 20 mm) and slid off the tip, 0.25. With
# unstick, while less than UNSTICK_ZONE is out the hand's target never leads the strap's advance by more than
# UNSTICK_LEAD, and while it is held back it moves sideways, at most tr.unstick_side (-: toward the robot; pull 2's
# 20 degree turn toward the robot freed m-demo's stall at 67 mm). Past the zone the hold eases off at UNSTICK_CATCHUP.
UNSTICK_LABELS = ("pull 1", "pull 2")
UNSTICK_SHIFT_LABELS = ("hold the tail", "pull 1", "pull 2", "tighten", "hold snug", "release 2", "clear 2")
UNSTICK_ZONE = 40.0 * MM
UNSTICK_LEAD = 12.0 * MM
UNSTICK_SIDE_RATE = 8.0 * MM  # per second
UNSTICK_CATCHUP = 20.0 * MM  # per second
ALIGN_GAP = 4.0 * MM  # the tip lines up this far short of the entry (round 1's 2 mm let it slip in early)


@dataclass(frozen=True)
class Trial:
    name: str
    build: str
    why: str
    ratchet: str = "brake"
    insert: str | None = "edge"  # edge (horizontal grip, fingers on the strap's edges) or flat (tilted from above)
    pull: str | None = "edge"  # edge (end-on, then side strokes), flat (down-grip strokes) or None
    steer: bool = False
    aim_off: tuple = (0.0, 0.0)  # world (dx, dz) added to the inserting hand's aim, m
    extra_pull: float = 30.0 * MM  # the last stroke goes this far past snug
    watch_s: float = 4.0
    pick: bool = False  # free build: the right hand picks the tie up by its head
    tug: bool = False  # free build, after the pick: the left hand takes the strap and pulls against the held head
    head_fingers: str = "head"  # the holding hand's squeeze on the head
    grip_shift: tuple = (0.0, 0.0)  # the inserting hand takes the strap (dz world, further from the tip), m: the plan
    #                                 after it doesn't know (robustness repeats)
    insert_s: float = 2.5  # the first push's duration (the second takes 0.8 of it)
    tilt_deg: float = 0.0  # the strap comes at the slot this far off its axis, in the loop's plane
    grip_depth: float = GRIP_DEPTH  # how far past the TCP the strap sits in the pads
    insert_fingers: str = "closed"  # the inserting hand's squeeze
    naive: bool = False  # the control: no lining up, the strap pushed along wherever it pointed when taken
    phys: str = "base"  # the tie's physics preset (PHYS); one simulator launch per (build, phys)
    depth: bool = True  # the probe extends the second push until PROTRUDE is measured out
    depth_cap: float = 8.0 * MM  # ... by at most this (the fingers' clearance to the head sets it: 4 mm left)
    strokes: int = 3  # pushes in the insert (round 4: three, so 30 mm of tail is out for the pulling hand)
    handover: bool = False  # round 5: one push, the right hand takes the tail before the left lets go, one pull
    pull_angle: float = 20.0  # deg: the pull turns this far toward the robot after PULL_STRAIGHT (reach)
    yield_deg: float | None = YIELD_DEG  # the strap's yield angle per joint (None: purely elastic)
    wrap_s: float = 4.0  # wrap build: the swing round the front of the cords takes this long
    wrap_over: float = 0.0  # deg: ... and goes this far past the slot's line first, then back (the set settles)
    grip_from_tip: float = GRIP_FROM_TIP
    steer_cap: float = STEER_CAP  # round 8: how far the steering may shift the inserting hand
    steer_yaw: bool = False  # round 8: also turn it about the strap in its fingers (YAW_*)
    rebite: float = 0.0  # round 9: the pulling hand draws this far, opens, slides back to the exit and takes a deep bite
    guide: bool = False  # round 10: the inserting hand stays, fingers loose round the strap, until the first pull is done
    latch_mm: float = LATCH_AT  # round 11: the pawl is held clear until this much tail is out
    pull_aim_x: float = 0.0  # round 11: the pull's targets shifted this far along world x (+: toward the pawl's side)
    ease: float = 0.0  # round 15: after the last pull the hand eases this far back along it before holding and letting go
    demo: bool = False  # round 15: the probe plays task.demo_plan() (the task's own closed loop) instead of steering itself
    tighten: float = 0.0  # round 16: after the last stroke the hand goes on (closed loop) until snug, at most this far
    hold_s: float = 1.0  # round 16: how long the pulling hand holds the snug loop before it lets go
    tighten_slack: float = TIGHTEN_SLACK  # round 17: the tighten goes on until less than this short of snug is out
    unstick: bool = False  # round 18: the pulling hand never far ahead of a stalled strap (UNSTICK_*)
    unstick_side: float = -8.0 * MM  # ... and moves this far sideways (world x) while held back
    stage: str = ""  # round 19, the lift start: "lift" (left only), "hold" (right only) or "slide" (both)
    lift_h: float = LIFT_H  # how high the left lifts the bundle
    lift_yaw: float = 0.0  # the left's down grip: fingers close across the bundle (along y)
    turn_deg: float = 0.0  # the right may turn the held tie about the vertical (it lies pointing the slide's way)
    slide_h: float = 0.030  # the head's underside this far above the console while the strap slides under
    # round 20, stage B ("thread"): where the hands change grips, slide and thread (tie_logic B_*, thread_targets)
    b_regrasp: tuple = (0.330, -0.130, TOP + 0.048)  # the head's centre while the right changes grip
    b_hold_s: float = 0.050  # the left holds the strap this far along it from the head's face meanwhile (round 20:
    #                          76 mm; the heavier strap lets the head drop about 3 mm here, 11 mm there)
    b_hold_phi: float = 45.0  # deg: ... its grip pointing -y tilted this far down (round 20's down grip, 90, put both
    #                           forearms within 3 mm: job 95's left ran into the right and the head fell)
    b_slide_pitch: float = 15.0  # deg: the head pitched down while the strap slides (it leaves the head downward)
    b_slide_z: float = TOP + 0.035  # the head's centre while it slides
    b_head: tuple = (0.330, 0.070, TOP + 0.033)  # the head's centre for the threading (round 23: closer to the bundle and
    #                                              lower; round 22's 0.045, 0.060 needed 29-88 mm more strap than the
    #                                              tie has: strap_slack. Lower, the right's finger base meets the console)
    b_insert_pitch: float = -10.0  # deg: the head's pitch then (0: level, the slot vertical; round 23: -10, its entry
    #                                tilted toward the robot so the slot runs along the left's tail, b_phi)
    b_lower: float = 0.010  # the left lowers the bundle to this above its pinch, then lets go
    b_gft: float = 0.036  # the left takes the tail this far from its tip (round 23: 50 left too little strap; under 34
    #                       its pads come down onto the right's at the end of the push)
    b_phi: float = 10.0  # deg: the left's tilt below level (pointing -y) while threading (0: the tail straight down;
    #                      round 23: 10, its upper arm 26 mm over the power strip, not 3)
    b_lift: tuple = (-0.060, 0.050)  # the tail's lift from where it was taken (world y, z: round 23 swings it toward
    #                                  the bundle as it rises; round 22 lifted it 60 mm straight up, 20-30 mm of strap)
    b_over: tuple = (0.330, 0.105, TOP + 0.070)  # the strap's point in the left's fingers as it goes over the bundle
    #                                              (round 22: 0.330, 0.125, 0.120, 60 mm short)
    b_out: float = 4.0 * MM  # the push leaves this much tail out of the exit (round 23: 22 brought the left's pads down
    #                          onto the right's, now 4 mm deeper on the head; B_DEPTH_CAP adds up to 3 more)
    b_hold_after: float = 1.0  # the right holds the head this long after the left has let go
    b_slide_first: bool = False  # round 24 (the longer ties): stage A's slide first, with the pick's down grip, to the
    #                              head at b_regrasp's y; then the grips change there, clear of the cords; then the right
    #                              carries the head to b_head (the strap slides on under the cords) and the left threads
    b_slide_s: float = 4.5  # ... the slide's duration then (224 mm on a 300 mm tie)
    b_lift_s: float = 1.5  # the left's lift of the tail, its move over the bundle and down to above the head (s)
    b_lift_phi: float = 90.0  # round 25: the left's tilt (thread_left_rot) at the top of the tail's lift. Job 101: lifted
    #                           at 90 the tail turned 45-50 degrees tip-up in the pinch (the strap behind levers it) and
    #                           then followed the wrist to the degree; tilted with it as it rises, it shouldn't turn
    b_over_s: float = 2.5
    b_above_s: float = 2.5
    # round 26, the cinch (Trial.b_cinch, _cinch_tail): the tail carried over the bundle on an eased pinch, the head
    # turned to face it, one push, then the right takes the tail, the left the head, and the right pulls
    b_cinch: bool = False
    b_wait_pitch: float = -10.0  # deg: the head's pitch while the tail comes over (level, the right's hand low and
    #                              behind it: the tail swings over above it); it turns to b_insert_pitch after
    b_top: tuple = (0.330, 0.115, TOP + 0.115)  # the strap's point in the left's fingers at the top of the carry, over
    #                                             the bundle: the tail points straight up there and tips over
    h_z: float = 0.140  # round 31, the hanging cords (stage "hang"): the height the tie is lifted and turned at
    h_low: float = 0.090  # ... and the height the left takes its tail at (higher, its upper arm nears the TV)
    h_first: float = 0.030  # ... the left takes the lying tail this far from its tip for the lift and the turn
    h_roll_s: float = 4.0
    h_lane: float = 0.340  # ... the x the right carries the ruler along, in front of the cords
    h_tip_short: float = 0.025  # ... its tip this far short of the cords before the pass behind them
    h_poke_s: float = 4.0
    h_root_back: float = 0.030  # ... the pass ends with the strap's root this far short of the point behind the cords
    h_swing: float = 58.0  # deg: ... the tail swung this far round toward the robot, about that point (less, the
    #                        left's wrist starts too far round for the wrap that follows; more, the head nears the wall)
    h_swing_s: float = 4.5
    h_gft: float = 0.066  # ... the left takes the tail this far from its tip (coming in at h_tail_deg its fingers
    #                       end the push 39 mm from the head's centre, 22 mm past the right's fingertips)
    h_hold: float = 3.0  # ... and holds the tie alone this long
    h_thread: bool = False  # round 32: on from there: the head round to the front, thread, hand over, draw (_hang_thread)
    h_feed: tuple = (0.305, 0.100, 205.0)  # ... the left brings the tail's grip here (x, y), the tail heading this (deg)
    h_feed_s: float = 3.0
    h_tilt: float = -30.0  # ... its hand leaning this far along the strap (negative: wrist toward the tail's tip)
    h_round_s: float = 2.0  # ... each leg of the head's way round the cords
    h_head_at: tuple = (0.288, 0.033)  # ... the head where it is threaded, in front of the cords: 100 mm of strap
    #                                    from their right side, leaving the head 25 degrees right of straight at the wall
    h_d_deg: float = 245.0  # ... the slot's way through there, heading (deg from +x, level)
    h_tail_deg: float = 236.0  # ... the tail's heading as the left threads it (further round, its wrist joint is at
    #                            its stop; nearer the cords, its forearm is at them)
    h_tail_max: float = 239.0  # ... and the furthest round the closed loop turns the left's hand
    h_out: float = 0.024  # ... the push leaves this much tail out of the exit (more, the left's pads reach the right's)
    h_stub_at: float = 0.014  # ... the right takes the tail this far below the exit (the middle of its pads)
    h_clear: tuple = (0.250, 0.150, 0.060)  # ... the left's hand goes here (x, y, up) before it goes back to ready
    h_carry_fingers: str = "head_light"  # ... the right's squeeze on the head while it carries it round (light: the
    #                                      head turns between its fingers to trail the strap)
    h_push_s: float = 4.0
    h_carry_yaw: float = 0.0  # round 33: the right's hand keeps this turn (deg, toward the tail's side) while it
    #                           carries the head round, so it has that much to turn the head back with afterwards
    h_tauten: float = 0.0  # ... at the front it draws the head out along the strap's line, at most this far (0: not)
    h_tauten_s: float = 5.0
    h_taut_at: float = 0.030  # ... the plan's guess of where the strap is taut
    h_turn_head: float = 0.0  # deg: ... then turns its hand back this far (the closed loop: until the slot lies along
    #                           the tail's line)
    h_regrip_yaw: float = 0.0  # deg, round 34: ... before that it lets go of the head, turns its hand out this far and
    #                            takes the head again (0: not; job 120: let go, the bowed strap flings the head back
    #                            behind the cords)
    h_turn_fingers: str = ""  # round 36: the right's squeeze from the turn back on ("": as it carried the head)
    h_pull_from: float = -0.040  # ... the right brings the tail to (h_lane, this y) at the first carry's height
    #                              (the draw's first leg: round 37, away from the cords all the way; job 121: brought
    #                              across the front of them, the loop pulled 10 mm of tail back out through the pawl),
    h_out_s: float = 3.0
    h_pull: float = 0.250  # ... then draws it this far toward itself (-y)
    h_held: bool = False  # round 40: the left holds the head for the draw. Jobs 123, 124: with nothing holding the
    #                       head the pull on the tail is answered by the strap's root, 4.6 mm beside the slot; the head
    #                       cocks 27 degrees, the strap bears on the slot's two ends and stays put (12 N pulled the
    #                       tail out of the right's fingers). So, as a person does: the right takes the tail's stub (it
    #                       holds the tie meanwhile), the left lets go of the strap and takes the head from its free
    #                       face, level, its pads above and below, flush with the exit face (HOLD_O); the right lets
    #                       go; the left carries the head to the cords (h_tight); the right takes the stub again and
    #                       draws it along the slot's line until the full mark is out. Both let go.
    h_tight: tuple = (0.0115, 0.0100, 10.0)  # the head's place for the draw: the cords' middle this far along the
    #                       strap's way out of the head and along the slot's way in (hugging them: 9.2, 5.9 mm), and
    #                       the heading (deg) of the strap's way out of the head there (0: at the wall). Round 43
    #                       (job 128): 10 degrees round toward the cords. The loose loop comes to the slot along
    #                       the cords' front; with the head square to the wall (it sat 11 degrees the other way) the
    #                       loop came in from the far side of the slot's line, lay on the edge of the slot's wall
    #                       there, and each strap joint that reached the mouth butted on the head's face (the draw
    #                       stalled at the first, 38 mm, at 10 N). Pushed in by the left the strap came in on the
    #                       near side, clear of that edge, and five joints went through.
    h_square_max: float = 28.0  # deg: as far as the left's wrist turns to put the head on that heading (its reach)
    h_carry_s: float = 4.0
    h_pull_to: float = 0.0  # m of tail at which the held draw stops (0: the full mark and PAST more)
    h_stub_yaw: float = 90.0  # deg: the right's hand for its second take of the tail: end on to it (round 45), so
    #                           its fingers end by the left's on the head instead of crossing them, ...
    h_pull_yaw: float = 58.0  # ... turning to this over the first HOLD_TURN of the draw (end on, the draw is out of reach)
    h_stand_by: tuple = (0.330, -0.120)  # where the right waits (its grip's place) while the left carries the head
    h_pull_s: float = 9.0
    b_wedge: bool = False  # round 29: the tail is locked tip-down in the pinch before threading (_wedge_tail)
    b_push_s: float = 2.5  # ... the push takes this long
    b_draw: bool = False  # round 30: then the right takes the tail below the head and draws it out (tr.b_pull)
    b_pull_up: float = 0.10  # ... rising this much per unit drawn
    b_hold_cords: bool = False  # ... while the left pins the bundle where it pinched it to lift it
    b_arch: float = 0.035  # ... the pinched tail is pushed this far toward the bundle along the console, then lifted
    b_keys: bool = False  # the plan ends with the right holding the head by the bundle and the left holding the tail,
    #                       lifted: a person takes over from there on the keyboard (the Station's script_first)
    b_soft: str = "soft"  # the left's fingers while it carries the tail over the bundle (FINGER_GAP)
    b_feed: tuple = (0.330, 0.138, TOP + 0.057)  # the strap's point in the left's fingers when it comes down past
    #                                              the far side of the bundle, the tail reaching back over it to the head
    b_feed_phi: float = 10.0  # deg: the left's tilt as it arrives (thread_left_rot): wound back, so it can turn on
    #                           and still lean away from the right's hand on the head (30 with a 37-degree turn put
    #                           the two grippers 0.6 mm apart at the end of the push, offline)
    b_turn: float = 27.0  # deg: the plan's turn of the wrist there (the closed loop sets the real one from the tail;
    #                       the model's tail arrives 10 degrees above level)
    b_pull: float = 0.175  # the right draws the tail up to this far, straight out along the slot (the closed loop
    #                        stops it once the full mark is 4 mm out of the head)
    b_pull_s: float = 6.0
    b_pull_lift: float = 0.050  # both hands raise the tie this far first (the pull runs 17 degrees down; the bundle
    #                             comes up into the loop)
    b_tail_at: float = 0.008  # the right's pads are centred this far plus half a pad along the tail from the exit face
    b_tail_tilt: float = 40.0  # deg: the right takes the tail pointing +y tilted this far down (back_rot): from its own
    #                            side, its hand leaning away from the left's on the head
    b_head_phi: float = 55.0  # deg: the left takes the head pointing -y tilted this far down (thread_left_rot): its
    #                           hand leaning back over the bundle, away from the right's


def _seconds(steps) -> float:
    return float(sum(s[0] for s in steps))


def plan_seconds(plan: dict) -> float:
    return max(_seconds(v) for v in plan.values())


def _pad(plan: dict) -> dict:
    """Both arms end together: the shorter one holds its last pose."""
    total = plan_seconds(plan)
    out = {}
    for arm, steps in plan.items():
        steps = list(steps)
        left = total - _seconds(steps)
        if left > 1e-9:
            if steps:
                s = steps[-1]
                steps.append((left, s[1], s[2], s[3], "wait"))
            else:
                steps.append((left, None, "open", "ready", "wait"))
        out[arm] = steps
    return out


def _ready(seconds: float = 2.0, fingers: str = "open", label: str = "ready") -> tuple:
    return (seconds, None, fingers, "ready", label)


def _wait(steps: list, until: float) -> None:
    """Hold the last pose until time ``until``."""
    left = until - _seconds(steps)
    if left > 1e-9:
        s = steps[-1] if steps else _ready(0.0)
        steps.append((left, s[1], s[2], s[3], "wait"))


def _along(poly: np.ndarray, d_from_tip: float):
    """The point ``d_from_tip`` back from the tip along a polyline, and the strap's direction there (toward the tip)."""
    seg = np.diff(poly, axis=0)
    lens = np.linalg.norm(seg, axis=1)
    left = d_from_tip
    for i in range(len(seg) - 1, -1, -1):
        if left <= lens[i]:
            u = seg[i] / lens[i]
            return poly[i + 1] - left * u, u
        left -= lens[i]
    return poly[0], seg[0] / lens[0]


def tail_yaw_error(poly: np.ndarray, through: np.ndarray, look: float = YAW_LOOK) -> float:
    """rad: the turn about the vertical that would bring the tail's last ``look`` (toward the tip) onto ``through``."""
    _, u = _along(poly, look)
    return math.atan2(u[0] * through[1] - u[1] * through[0], float(u[:2] @ through[:2]))


def steer_pose(pos, rot, corr, yaw: float, grip_depth: float):
    """The inserting hand's steered pose: turned by ``yaw`` about the vertical through the strap in its fingers
    (grip_depth past the TCP), then shifted by ``corr``."""
    pos, rot = np.asarray(pos, float), np.asarray(rot, float)
    if yaw:
        strap = pos + rot[:, 2] * grip_depth
        rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        pos, rot = strap + rz @ (pos - strap), rz @ rot
    return pos + np.asarray(corr, float), rot


def jig_frame(build: Build | None = None, grip_from_tip: float = GRIP_FROM_TIP) -> dict:
    """World points the jig plans aim at: the slot's entry and exit centres, the way the tail goes through (world),
    the strap's out-of-loop normal at the inserting grip."""
    b = build or BUILDS["jig"]
    entry = world(b, (SLOT_C, 0.0, ENTRY_Z))
    exit_ = world(b, (SLOT_C, 0.0, EXIT_Z))
    through = b.head_rot @ np.array([0.0, 0.0, -1.0])
    poly = start_polyline(b)
    grip, u = _along(poly, grip_from_tip)
    centre = world(b, _loop_centre(b.points))
    out = grip - centre
    out = out - (out @ EDGE_UP) * EDGE_UP
    return {"entry": entry, "exit": exit_, "through": through, "grip": grip, "grip_dir": u,
            "out": out / np.linalg.norm(out)}


def _flat_rot(u) -> np.ndarray:
    """A grip on the strap's faces where it runs along ``u`` (horizontal, toward the tip): fingers closing square to it
    (horizontal), the gripper pointing down and FLAT_TILT toward where the strap goes. A straight-down grip at the tie's
    height is out of reach (yam_arena IK misses by 23-30 mm near the cords); tilted 20-80 degrees it reaches."""
    t = math.radians(FLAT_TILT)
    u = np.asarray(u, float)
    approach = np.array([u[0] * math.sin(t), u[1] * math.sin(t), -math.cos(t)])
    return rot_axes(approach, -np.cross(EDGE_UP, u))  # this closing sign is the one both arms reach


def _yawed(v, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.asarray(v, float)


def _insert_steps(tr: Trial, b: Build) -> list:
    """The left hand: take the strap GRIP_FROM_TIP back from its tip, line the tip up 2 mm before the entry on the
    slot's line, push until the tip is FIRST_OUT past the exit, open, slide back REGRIP along the now straight strap,
    take it again and push until PROTRUDE is out, let go, back off, ready. Trial knobs: grip_shift (taken off where the
    plan thinks), tilt_deg (the strap comes at the slot off its axis), insert_s (speed), grip_depth, insert_fingers,
    naive (no lining up: pushed along wherever the strap pointed where it was taken)."""
    gft = tr.grip_from_tip + tr.grip_shift[1]
    f = jig_frame(b, gft)
    g, u, out = f["grip"], f["grip_dir"], f["out"]
    d = _yawed(f["through"], tr.tilt_deg)  # the way the strap is pushed in
    aim = np.array([tr.aim_off[0], 0.0, tr.aim_off[1]])
    depth, fing = tr.grip_depth, tr.insert_fingers
    if tr.insert == "flat":  # fingers on the strap's faces, the gripper pointing down and toward the head
        rot_g, rot_a = _flat_rot(u), _flat_rot(d)
    else:  # edge: approach square to the strap from outside the loop, fingers closing on its edges (vertical)
        rot_g, rot_a = rot_axes(-out, EDGE_UP), rot_axes(-np.cross(d, EDGE_UP), EDGE_UP)
    at_g = g - rot_g[:, 2] * depth  # the strap ``depth`` past the TCP, between the pads
    nominal = g - rot_g[:, 2] * GRIP_DEPTH  # the pre-grasp doesn't move with grip_depth (it stays reachable)
    pre = nominal + EDGE_UP * 0.015 if tr.insert == "flat" else nominal - rot_g[:, 2] * 0.025  # flat: in from above
    taken = at_g + np.array([0.0, 0.0, tr.grip_shift[0]])  # where the hand really takes it
    p1 = ALIGN_GAP + HEAD[2] + FIRST_OUT  # stroke 1: the tip just out of the exit
    if tr.naive:  # the control: hold still, then push along the strap's own direction where it was taken
        rot_a, d = rot_g, u
        g_a = taken
    elif tr.insert == "edge" and abs(tr.tilt_deg) < 1e-9 and float(u @ d) > math.cos(math.radians(0.5)):
        # round 3: the strap already runs down the slot's line where it is taken; the hand never turns, it only
        # pushes along (an edge grip can't turn the strap: round 2's line-up swung the tip 10-16 mm)
        rot_a = rot_g
        g_a = taken + np.array([tr.aim_off[0], 0.0, tr.aim_off[1]])
    else:
        tip_a = f["entry"] - d * ALIGN_GAP  # aligned: strap straight from the hand along the line, tip short of it
        g_a = tip_a - d * gft - rot_a[:, 2] * depth + aim
    e1 = g_a + d * p1
    g2 = e1 - d * REGRIP  # slide back along the strap, open, and take it again
    e2 = g2 + d * (PROTRUDE - FIRST_OUT)  # stroke 2: PROTRUDE out
    g3 = e2 - d * REGRIP
    e3 = g3 + d * (PROTRUDE3 - PROTRUDE)  # stroke 3 (round 4): PROTRUDE3 out, the hand 8 mm further back than stroke 2
    # Edge grips open wide (14 mm) to let go: round 3's 6 mm opening dragged the strap out of the slot as the hand left.
    rel = "wide" if tr.insert == "edge" else "loose"
    steps = [
        (2.5, tuple(pre), "wide", rot_g, "to the strap"),
        (1.5, tuple(taken), "wide", rot_g, "in"),
        (1.0, tuple(taken), fing, rot_g, "grip"),
        (2.0, tuple(g_a), fing, rot_a, "align"),
        (tr.insert_s, tuple(e1), fing, rot_a, "insert"),
        (0.5, tuple(e1), fing, rot_a, "insert hold"),
        (0.6, tuple(e1), rel, rot_a, "open"),
        (1.2, tuple(g2), rel, rot_a, "slide back"),
        (0.8, tuple(g2), fing, rot_a, "regrip"),
        (0.8 * tr.insert_s, tuple(e2), fing, rot_a, "insert 2"),
        (0.4, tuple(e2), fing, rot_a, "insert 2 hold"),
    ]
    last = e2
    if tr.strokes >= 3:
        steps += [
            (0.6, tuple(e2), rel, rot_a, "open 2"),
            (1.2, tuple(g3), rel, rot_a, "slide back 2"),
            (0.8, tuple(g3), fing, rot_a, "regrip 2"),
            (0.8 * tr.insert_s, tuple(e3), fing, rot_a, "insert 3"),
            (0.4, tuple(e3), fing, rot_a, "insert 3 hold"),
        ]
        last = e3
    steps += [
        (0.8, tuple(last), rel, rot_a, "let go"),
        (1.2, tuple(last + (EDGE_UP * 0.015 if tr.naive else -rot_a[:, 2] * 0.04)), rel, rot_a, "back off"),
    ]
    if tr.pull:
        steps.append(_ready(2.0))
    return steps


def _handover_plan(tr: Trial, b: Build) -> dict:
    """Round 5's mechanism. Left: take the straight run GRIP_FROM_TIP back from the tip and push it straight down the
    slot until HANDOVER_OUT is out (the probe extends the push closed-loop), and keep holding. Right: come at the tail
    end-on, slide over it, close. Only then does the left open wide and leave. Right: pull straight out PULL_STRAIGHT,
    then on along a line turned pull_angle toward the robot (reach) to extra_pull past the tightest loop; hold, let go,
    leave. The strap is never let go of until a hand or the snug loop holds it (round 4: every regrip let it spring out
    of the slot)."""
    f = jig_frame(b, tr.grip_from_tip)
    g, u, d = f["grip"], f["grip_dir"], f["through"]
    rot = rot_axes(-f["out"], EDGE_UP)  # the left hand from the front, fingers on the strap's edges
    if float(u @ d) < math.cos(math.radians(0.5)):
        raise ValueError("the hand-over grip must sit on the straight run into the slot")
    rot = rot_axes(-np.cross(d, EDGE_UP), EDGE_UP)
    taken = g - rot[:, 2] * tr.grip_depth
    pre = taken - rot[:, 2] * 0.025
    push = TIP_GAP + HEAD[2] + HANDOVER_OUT
    e = taken + np.array([tr.aim_off[0], 0.0, tr.aim_off[1]]) + d * push
    left = [
        (2.5, tuple(pre), "wide", rot, "to the strap"),
        (1.5, tuple(taken), "wide", rot, "in"),
        (1.0, tuple(taken), "closed", rot, "grip"),
        (0.8, tuple(taken), "closed", rot, "align"),
        (tr.insert_s + 1.5, tuple(e), "closed", rot, "insert"),
    ]
    t_pushed = sum(s[0] for s in left)
    # the right hand: end-on onto the tail, the fingertips TIP_STANDOFF short of the exit face
    x0 = f["exit"]
    end_on = rot_axes(-d, EDGE_UP)
    at = x0 + d * (TIP_STANDOFF + PAD_PAST_TCP)
    right = [(t_pushed + 0.3, None, "open", "ready", "wait for the push"),
             (2.5, tuple(at + d * 0.04), "wide", end_on, "to the tail"),
             (1.5, tuple(at), "wide", end_on, "slide on"),
             (0.8, tuple(at), "closed", end_on, "pinch 1")]
    t_pinched = sum(s[0] for s in right)
    left.append((t_pinched - t_pushed + 0.3, tuple(e), "closed", rot, "insert hold"))
    if tr.guide:  # round 10: open to a loose ring round the strap and stay: it feeds the head straight while pulled
        left += [(0.8, tuple(e), "wide", rot, "let go"), (0.5, tuple(e), "loose", rot, "guide")]
        t_left_gone = sum(s[0] for s in left)
    else:
        left += [(0.8, tuple(e), "wide", rot, "let go"),
                 (1.2, tuple(e - rot[:, 2] * 0.04), "wide", rot, "back off"),
                 _ready(2.0)]
        t_left_gone = sum(s[0] for s in left[:-1])
    right.append((t_left_gone - t_pinched, tuple(at), "closed", end_on, "hold the tail"))
    need = snug_through(b) - HANDOVER_OUT + tr.extra_pull
    a = math.radians(tr.pull_angle)
    turned = np.array([d[0] * math.cos(a) - d[1] * math.sin(a), d[0] * math.sin(a) + d[1] * math.cos(a), 0.0])
    if turned[0] > 0:  # turn toward the robot (-x)
        turned = np.array([d[0] * math.cos(-a) - d[1] * math.sin(-a), d[0] * math.sin(-a) + d[1] * math.cos(-a), 0.0])
    if tr.rebite > 0.0:  # round 9: the first bite is the narrow tip link; draw a little, then take the full strap
        right += [(1.5, tuple(at + d * tr.rebite), "closed", end_on, "draw"),
                  (0.5, tuple(at + d * tr.rebite), "wide", end_on, "open"),
                  (1.2, tuple(at), "wide", end_on, "slide back"),
                  (0.8, tuple(at), "closed", end_on, "bite")]
        need -= tr.rebite
    aim = np.array([tr.pull_aim_x, 0.0, 0.0])
    p1 = at + aim + d * min(PULL_STRAIGHT, need)
    p2 = p1 + turned * max(need - PULL_STRAIGHT, 0.0)
    rot2 = rot_axes(-turned, EDGE_UP)
    right += [(3.0, tuple(p1), "closed", end_on, "pull 1"),
              (3.5, tuple(p2), "closed", rot2, "pull 2")]
    if tr.ease > 0.0:  # round 15: off the stall, so the strap is let go at low tension, in the pads, braked
        p2 = p2 - turned * tr.ease
        right.append((0.8, tuple(p2), "closed", rot2, "ease"))
    if tr.tighten > 0.0:  # round 16: task.Steering carries the hand on along the pull until the loop is snug
        right.append((TIGHTEN_S, tuple(p2), "closed", rot2, "tighten"))
    right += [(tr.hold_s, tuple(p2), "closed", rot2, "hold snug"),
              (0.6, tuple(p2), "wide", rot2, "release 2"),
              (1.0, tuple(p2 + turned * 0.03), "wide", rot2, "clear 2"),
              _ready(2.0)]
    if tr.guide:  # the guide leaves once the first pull is done (before the loop closes on the cords)
        t_pulled = 0.0
        for s in right:
            t_pulled += s[0]
            if s[4] == "pull 1":
                break
        left += [(max(t_pulled - sum(s[0] for s in left), 0.1), tuple(e), "loose", rot, "guide hold"),
                 (0.6, tuple(e), "wide", rot, "unguide"),
                 (1.2, tuple(e - rot[:, 2] * 0.04), "wide", rot, "back off"),
                 _ready(2.0)]
    return {"left": left, "right": right}


WRAP_STEPS = 12


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap_swing(tr: Trial, b: Build) -> dict:
    """The wrap's swing: where the left hand takes the straight tail (GRIP_FROM_TIP from the tip, fingers on its edges,
    from the +y side), and the turn about the vertical (angle, pivot) that carries that grip onto the jig's grip on the
    slot's line. Returns taken, rot0, phi (rad), pivot (x, y) and the jig frame."""
    j = BUILDS["jig"]
    f = jig_frame(j, tr.grip_from_tip)
    d = f["through"]
    g0, u0 = _along(start_polyline(b), tr.grip_from_tip)
    rot0 = rot_axes(-np.cross(u0, EDGE_UP), EDGE_UP)
    rot1 = rot_axes(-np.cross(d, EDGE_UP), EDGE_UP)
    t0 = g0 - rot0[:, 2] * tr.grip_depth
    t1 = f["grip"] - rot1[:, 2] * tr.grip_depth
    phi = math.atan2(u0[0] * d[1] - u0[1] * d[0], float(u0 @ d))
    r = _rz(phi)[:2, :2]
    pivot = np.linalg.solve(np.eye(2) - r, t1[:2] - r @ t0[:2])
    return {"taken": t0, "rot0": rot0, "phi": phi, "pivot": pivot, "frame": f, "end": t1, "rot1": rot1}


def _wrap_plan(tr: Trial, b: Build) -> dict:
    """Round 7's whole tie. Left: take the straight tail from the +y side, then swing it (and turn with it) about the
    vertical through the bend behind it, round the front of the cords, until it lies on the slot's line where the jig
    left it, wrap_over past and back if asked; then round 5-6's hand-over from "align" on: push, the right takes the
    tail, the left leaves, one pull, hold, let go."""
    w = wrap_swing(tr, b)
    t0, rot0, phi, c = w["taken"], w["rot0"], w["phi"], w["pivot"]

    def at(a):
        p = c + _rz(a)[:2, :2] @ (t0[:2] - c)
        return np.array([p[0], p[1], t0[2]]), _rz(a) @ rot0

    pre = t0 - rot0[:, 2] * 0.025
    left = [(2.5, tuple(pre), "wide", rot0, "to the strap"),
            (1.5, tuple(t0), "wide", rot0, "in"),
            (1.0, tuple(t0), "closed", rot0, "grip")]
    over = math.radians(tr.wrap_over) * (1.0 if phi >= 0 else -1.0)
    for k in range(1, WRAP_STEPS + 1):
        p, r = at((phi + over) * k / WRAP_STEPS)
        left.append((tr.wrap_s / WRAP_STEPS, tuple(p), "closed", r, f"wrap {k}"))
    if over:
        p, r = at(phi)
        left.append((1.0, tuple(p), "closed", r, "wrap back"))
    p, r = at(phi)
    left.append((1.0, tuple(p), "closed", r, "wrap settle"))
    added = sum(s[0] for s in left)
    hp = _handover_plan(tr, BUILDS["jig"])
    removed = 0.0
    for s in hp["left"]:
        if s[4] in ("to the strap", "in", "grip"):
            removed += s[0]
    left += [s for s in hp["left"] if s[4] not in ("to the strap", "in", "grip")]
    right = list(hp["right"])
    s0 = right[0]
    right[0] = (s0[0] + added - removed, s0[1], s0[2], s0[3], s0[4])
    return {"left": left, "right": right}


def cord_ring_h(build: Build, offset: float = T / 2, n: int = 90) -> np.ndarray:
    """Points round each cord where it crosses the loop's plane, grown by ``offset``, in the head frame's (x_h, z_h)."""
    out = []
    for c in CORDS:
        p = (np.array([c.xy[0], c.xy[1], build.head_pos[2]]) - np.asarray(build.head_pos)) @ build.head_rot
        for a in np.linspace(0, 2 * math.pi, n, endpoint=False):
            out.append(p[[0, 2]] + (c.diameter / 2 + offset) * np.array([math.cos(a), math.sin(a)]))
    return np.array(out)


def min_loop_at(head_pos, head_quat) -> float:
    """The tightest loop (root to entry, m) the cords allow round a head at this pose: a taut string from the root round
    every cord (where its axis crosses the loop's plane) to the slot's entry. 1e9 if the cords don't cross the plane."""
    r = quat_xyzw_to_matrix(head_quat)
    ring = []
    for c in CORDS:
        base = (np.array([c.xy[0], c.xy[1], CORD_Z[0]]) - np.asarray(head_pos, float)) @ r
        d = np.array([0.0, 0.0, 1.0]) @ r
        if abs(d[1]) < 0.3:
            return 1e9
        q = base + (-base[1] / d[1]) * d
        for a in np.linspace(0, 2 * math.pi, 48, endpoint=False):
            ring.append(q[[0, 2]] + (c.diameter / 2 + T / 2) * np.array([math.cos(a), math.sin(a)]))
    h = _hull(np.vstack([ring, ROOT[[0, 2]], [SLOT_C, ENTRY_Z]]))
    return float(np.sum(np.linalg.norm(np.diff(np.vstack([h, h[:1]]), axis=0), axis=1))) - HEAD_CLOSURE


def min_loop(build: Build) -> float:
    """The tightest loop the cords allow, root to entry (m): a taut string from the root round the cords to the slot's
    entry, i.e. the perimeter of the convex hull of the cords, the root and the entry, less the closure through the
    head. With the head held against the bundle (the task), slack at this loop is close to zero; in the jig, where the
    rigid cords stand 3 to 9 mm off the head, it is the least slack round 1 can reach."""
    pts = np.vstack([cord_ring_h(build), ROOT[[0, 2]], [SLOT_C, ENTRY_Z]])
    h = _hull(pts)
    return float(np.sum(np.linalg.norm(np.diff(np.vstack([h, h[:1]]), axis=0), axis=1))) - HEAD_CLOSURE


def min_slack(build: Build) -> float:
    return min_loop(build) + HEAD_CLOSURE - bundle_perimeter()


def snug_through(build: Build | None = None) -> float:
    """The tail out of the exit (m) when the loop is as tight as the cords allow."""
    return TIE_L - HEAD[2] - min_loop(build or BUILDS["jig"])


def _pull_steps(tr: Trial, f: dict) -> list:
    """The right hand: strokes along the tail's way out until it has pulled extra_pull past snug. Edge: first end-on
    (fingers on the tail's edges, coming straight at the exit), then from the front; flat: down grips."""
    x0, d = f["exit"], f["through"]
    out = PROTRUDE3 if (tr.insert and tr.strokes >= 3) else PROTRUDE
    need = snug_through(BUILDS[tr.build]) - out + tr.extra_pull  # strap to draw through, m
    strokes = []
    left_ = need
    while left_ > 1e-6:
        strokes.append(min(0.060, left_))
        left_ -= strokes[-1]
    steps = []
    side = rot_axes((1.0, 0.0, 0.0), EDGE_UP)
    flat = _flat_rot(-d)  # pointing down and back toward the head, closing on the tail's faces
    for k, pull in enumerate(strokes):
        if k == 0:  # end-on: the pads slide over the tail's end (only PROTRUDE is out: no room for a side grip)
            rot = rot_axes(-d, EDGE_UP)
            at = x0 + d * (TIP_STANDOFF + PAD_PAST_TCP)
            pre = at + d * 0.04
            steps += [(2.5, tuple(pre), "wide", rot, "to the tail"), (1.5, tuple(at), "wide", rot, "slide on")]
        else:
            q = x0 + d * 0.020  # the tail 20 mm out of the head
            rot = side if tr.pull == "edge" else flat
            at = q - rot[:, 2] * GRIP_DEPTH
            pre = q - rot[:, 2] * 0.04
            steps += [(2.0 if k else 2.5, tuple(pre), "wide", rot, f"to the tail {k + 1}"),
                      (1.0, tuple(at), "wide", rot, f"in {k + 1}")]
        steps += [(0.8, tuple(at), "closed", rot, f"pinch {k + 1}"),
                  (3.0 + 0.5 * (k == len(strokes) - 1), tuple(at + d * pull), "closed", rot, f"pull {k + 1}"),
                  (0.6, tuple(at + d * pull), "wide", rot, f"release {k + 1}")]
        away = at + d * pull - rot[:, 2] * 0.02 + np.array([-0.01, 0.0, 0.01])
        steps.append((1.0, tuple(away), "wide", rot, f"clear {k + 1}"))
    steps.append(_ready(2.0))
    return steps


LIFT = 0.060  # the pick raises the head this far
TUG_AT = 100.0 * MM  # the tug takes the strap this far from the head's face (nearer, it is out of the left's reach)
TUG = 25.0 * MM  # and pulls this far along it, against the held head


def _lift_plan(tr: Trial, b: Build) -> dict:
    """Round 19, stage A of the lift start. Left ("lift", "slide"): pinch the bundle on its run along the wall, lift it.
    Right ("hold", "slide"): pick the tie up by its head, turn it so the strap points at the wall, and ("slide") carry
    it so the strap goes under the lifted bundle, or ("hold") hold it up to see how the strap stands out. Then the left
    lowers the bundle onto the strap and lets go, and the right lets go of the tie."""
    left, right = [], []
    up = np.array([0.0, 0.0, 1.0])
    pinch = np.array([LIFT_AT[0], LIFT_AT[1], TOP + 1.0 * MM + PAD_PAST_TCP])  # fingertips 1 mm off the console
    lrot = rot_down(tr.lift_yaw)
    if tr.stage in ("lift", "slide"):
        left += [(2.5, tuple(pinch + 0.06 * up), "cords", lrot, "to the cords"),
                 (1.5, tuple(pinch), "cords", lrot, "down to the cords"),
                 (1.0, tuple(pinch), "closed", lrot, "pinch"),
                 (2.0, tuple(pinch + tr.lift_h * up), "closed", lrot, "lift cords"),
                 (1.0, tuple(pinch + tr.lift_h * up), "closed", lrot, "hold cords")]
    t_lifted = _seconds(left)
    if tr.stage in ("hold", "slide"):
        head = np.asarray(b.head_pos, float)
        across = b.head_rot[:, 1]  # y_h: fingers close across the head's 9 mm width
        yaw0 = math.degrees(math.atan2(across[0], -across[1])) + 180.0  # the same closing axis, the hand the other
        #                                                                  way round: reaches the slide's end
        rot0, rot1 = rot_down(yaw0), rot_down(yaw0 + tr.turn_deg)
        c, s_ = math.cos(math.radians(tr.turn_deg)), math.sin(math.radians(tr.turn_deg))
        x_h1 = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]) @ b.head_rot[:, 0]  # the strap's way, turned
        drop = np.array([0.0, 0.0, -HEAD[2] / 2 + 0.8 * MM + PAD_PAST_TCP])

        def tcp(head_at, x_h):  # the pick's grip on the head: 2 mm toward its back, pad tips 0.8 mm above its underside
            return np.asarray(head_at, float) - x_h * 2.0 * MM + drop

        at = tcp(head, b.head_rot[:, 0])
        hf = tr.head_fingers
        right += [(2.5, tuple(at + 0.05 * up), "wide", rot0, "to the head"),
                  (1.5, tuple(at), "wide", rot0, "descend"),
                  (1.0, tuple(at), hf, rot0, "grip"),
                  (1.5, tuple(at + 0.03 * up), hf, rot0, "raise tie")]
        z_head = TOP + HEAD[2] / 2 + (0.08 if tr.stage == "hold" else tr.slide_h)
        start = np.array([SLIDE_XY[0], SLIDE_TIP[0] - tip_from_head(b) * x_h1[1], z_head])
        end = np.array([SLIDE_XY[0], SLIDE_TIP[1] - tip_from_head(b) * x_h1[1], z_head])
        right.append((2.5, tuple(tcp(start, x_h1)), hf, rot1, "carry tie"))
        if tr.stage == "hold":
            right += [(4.0, tuple(tcp(start, x_h1)), hf, rot1, "hold tie"),
                      (1.5, tuple(tcp(start, x_h1) - 0.05 * up), hf, rot1, "lower tie")]
        else:
            _wait(right, t_lifted)
            right += [(3.0, tuple(tcp(end, x_h1)), hf, rot1, "slide under"),
                      (1.0, tuple(tcp(end, x_h1)), hf, rot1, "hold tie")]
            _wait(left, _seconds(right))
        last = right[-1]
        tie_done = _seconds(right)
    else:
        tie_done = 0.0
    if left:  # down onto the strap (or the console), let go, away
        low = pinch + (tr.slide_h + 0.020) * up if tr.stage == "slide" else pinch
        if tr.stage == "lift":
            left.append((2.0, tuple(pinch + tr.lift_h * up), "closed", lrot, "hold cords"))
        left += [(2.0, tuple(low), "closed", lrot, "lower cords"),
                 (0.8, tuple(low), "cords", lrot, "let go cords"),
                 (1.5, tuple(low + 0.06 * up), "cords", lrot, "clear cords"),
                 (1.5, (BACK_OFF_X, float(low[1]), TOP + 0.15), "cords", lrot, "back off"),  # elbow clear of the TV
                 _ready(2.0)]
    if right:
        _wait(right, max(_seconds(left) - 3.5, tie_done))  # the left is clear of the bundle by now
        p = np.asarray(right[-1][1], float)
        right += [(0.8, tuple(p), "wide", right[-1][3], "let go tie"),
                  (1.5, tuple(p + 0.05 * up), "wide", right[-1][3], "clear tie"),
                  (1.5, (BACK_OFF_X, float(p[1]), TOP + 0.15), "wide", right[-1][3], "back off"),
                  _ready(2.0)]
    return {"left": left, "right": right}


def head_rot_pitched(pitch_deg: float) -> np.ndarray:
    """The tie's head as it lies (R_FREE: strap toward +y, slot vertical) pitched ``pitch_deg`` down about the x axis
    (its strap's root lower)."""
    p = math.radians(pitch_deg)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(p), math.sin(p)], [0.0, -math.sin(p), math.cos(p)]])
    return rx @ R_FREE


def back_rot(tilt_deg: float) -> np.ndarray:
    """The right's grip from behind the head: pointing +y, tilted ``tilt_deg`` down, fingers closing along +x."""
    t = math.radians(tilt_deg)
    return rot_axes((0.0, math.cos(t), -math.sin(t)), (1.0, 0.0, 0.0))


def back_tcp(head_c, tilt_deg: float) -> np.ndarray:
    """The right's TCP holding the head (centre head_c) from behind with its hand tilted ``tilt_deg`` down. The grip is
    taken at B_GRIP_TILT on a level head, so the head's pitch is then tilt_deg - B_GRIP_TILT."""
    r = back_rot(tilt_deg)
    return np.asarray(head_c, float) - B_BACK_IN * r[:, 2] + B_BACK_UP * r[:, 1]


def thread_left_rot(phi_deg: float) -> np.ndarray:
    """The left's grip on the strap: pointing -y tilted ``phi_deg`` down (90: straight down), fingers closing along -x
    on its edges. The strap runs along its y axis, the tip's side toward +y_tcp: +y world at 90, straight down at 0."""
    p = math.radians(phi_deg)
    return rot_axes((0.0, -math.cos(p), -math.sin(p)), (-1.0, 0.0, 0.0))


def arc_length(poly) -> float:
    return float(np.sum(np.linalg.norm(np.diff(np.asarray(poly, float), axis=0), axis=1)))


def point_at_arc(poly, s: float) -> np.ndarray:
    """The point ``s`` along a polyline from its first point (clamped to its ends)."""
    poly = np.asarray(poly, float)
    seg = np.diff(poly, axis=0)
    lens = np.linalg.norm(seg, axis=1)
    s = float(np.clip(s, 0.0, float(lens.sum())))
    for i, ln in enumerate(lens):
        if s <= ln or i == len(lens) - 1:
            return poly[i] + seg[i] * (s / max(ln, 1e-15))
        s -= ln
    return poly[-1]


def thread_targets(tr: Trial) -> dict:
    """Stage B's planned points (world): the head's centre while the hands change grips, the strap point the left
    holds then, the tail point it takes off the console (the strap guessed straight down at 45 degrees from the
    levelled head to the console, then flat), the slot's entry centre and the way through while threading."""
    g = np.asarray(tr.b_regrasp, float)
    strap = g + R_FREE @ ROOT + R_FREE[:, 0] * tr.b_hold_s
    hc = np.asarray(tr.b_head, float)
    hr = head_rot_pitched(tr.b_insert_pitch)
    root = hc + hr @ ROOT
    h = float(root[2] - TOP)
    tip_y = float(root[1]) + h + (BUILDS[tr.build].tie_l - math.sqrt(2.0) * h)
    tail = np.array([SLIDE_XY[0], tip_y - tr.b_gft, TOP + T / 2])
    if tr.b_cinch:  # round 26: the head is held pitched steeply, the strap leaving it downward
        tail = cinch_tail_guess(tr)
    return {"regrasp": g, "strap": strap, "tail": tail, "head": hc, "head_rot": hr,
            "entry": hc + hr @ np.array([SLOT_C, 0.0, ENTRY_Z]), "through": hr @ np.array([0.0, 0.0, -1.0])}


# -- stage B's strap budget (round 23) ---------------------------------------------------------------------------------
# Job 97: with the head held 55 mm from the bundle and 32 mm over it, and the tail held 50 mm from its tip 54 mm over
# the slot, the tightest strap from the head's face under and round the lowered bundle to the left's fingers is 201-218
# mm; there were 150. The left dragged the cords 18 mm up and pulled the head out of the right's fingers toward the
# bundle (on the stiff strap already while it lifted the tail straight up, which alone takes 20-30 mm). So every step
# the left holds the tail must leave strap to spare: strap_slack() >= B_SLACK_MIN, the cords lying low or propped up.
B_SLACK_MIN = 0.010
B_TURN_SLACK = 0.050  # round 24: from align to the end of the push, room for the strap to turn from over the bundle
#   down onto the slot above the left's fingers (job 98: 14 mm, it folded where it left the pads); about 50 mm more
#   than the tightest path makes a 20 mm-radius turn there
B_CORDS_Z = (None, 0.017)  # the lowered cords' centres: on the strap on the console (None), or 17 mm up (job 97)
B_HOLD_LABELS = ("lift tail", "over", "above head", "align", "push", "push hold")  # the left holds the tail from its
#                                                                                    pick to these, in turn


def lift_bundle_ring(zc: float | None = None) -> np.ndarray:
    """The lowered bundle's cross-section at the slide line (world y, z): each cord grown by half the strap's
    thickness, its centre ``zc`` over the console (None: lying on the strap on the console), plus the console under the
    bundle's two sides (the strap passes under the cords, never through the console)."""
    pts = []
    for _, r, off, *_ in LIFT_CORDS:
        c = np.array([LIFT_HANG_Y - off, TOP + (r + T if zc is None else zc)])
        for a in np.linspace(0.0, 2 * math.pi, 72, endpoint=False):
            pts.append(c + (r + T / 2) * np.array([math.cos(a), math.sin(a)]))
    ys = [p[0] for p in pts]
    pts += [np.array([min(ys), TOP + T / 2]), np.array([max(ys), TOP + T / 2])]
    return np.array(pts)


def strap_need(root, held, zc: float | None = None) -> float:
    """The tightest strap (y-z) from its root at the head under and round the bundle to the point the left holds: the
    hull of the three walked counter-clockwise from the root (down, under the cords, up their far side, over them) to the
    held point. A held point inside that hull (tucked against the bundle) adds its distance from the nearest corner."""
    root, held = np.asarray(root, float), np.asarray(held, float)
    h = _hull(np.vstack([lift_bundle_ring(zc), root, held]))  # counter-clockwise, from the lowest y
    i = int(np.argmin(np.linalg.norm(h - root, axis=1)))
    k = int(np.argmin(np.linalg.norm(h - held, axis=1)))
    out = 0.0
    while i != k:
        j = (i + 1) % len(h)
        out += float(np.linalg.norm(h[j] - h[i]))
        i = j
    return out + float(np.linalg.norm(h[k] - held))


def strap_slack(tr: Trial, held) -> float:
    """The strap to spare (m) with the left holding the strap's point tr.b_gft from its tip at ``held`` (world) and
    the right holding the head where it threads: the worse of the two cord heights."""
    tg = thread_targets(tr)
    root = (tg["head"] + tg["head_rot"] @ ROOT)[1:]
    return min((BUILDS[tr.build].tie_l - tr.b_gft) - strap_need(root, np.asarray(held, float)[1:], zc)
               for zc in B_CORDS_Z)


def tail_held_points(plan_left: list, n: int = 6) -> list:
    """(label, strap point) the left holds, from the pick through B_HOLD_LABELS, ``n`` points along each move (the
    strap sits B_TAIL_DEPTH along its approach from its TCP; the plan's moves blend the TCP straight)."""
    steps = [s for s in plan_left if s[4] in ("take tail",) + B_HOLD_LABELS]
    pts = [np.asarray(s[1], float) + B_TAIL_DEPTH * np.asarray(s[3], float)[:, 2] for s in steps]
    out = []
    for i in range(1, len(steps)):
        for f in np.linspace(0.0, 1.0, n + 1)[1:]:
            out.append((steps[i][4], pts[i - 1] + f * (pts[i] - pts[i - 1])))
    return out


def enclosed_lift(polyline, head_pos, head_quat, polys: dict, passage_y: float = PASSAGE_Y) -> tuple:
    """The lift start's cords (polylines, world) that pass through the tie's loop: the strap from the root to where it
    enters the slot, closed through the head, in the head's x_h-z_h plane; a cord passes where it crosses y_h = 0."""
    ph = to_head(polyline, head_pos, head_quat)
    exits = [c for c in _crossings(ph, EXIT_Z) if _in_passage(c[2], passage_y=passage_y)]
    entries = [c for c in _crossings(ph, ENTRY_Z) if _in_passage(c[2], passage_y=passage_y)]
    if exits:
        before = [e for e in entries if (e[0], e[1]) < (exits[-1][0], exits[-1][1])]
        entry = before[-1] if before else None
    else:
        entry = entries[-1] if entries else None
    if entry is None:
        return ()
    i, _, p = entry
    loop = np.vstack([ph[: i + 1], p[None, :]])[:, [0, 2]]
    out = []
    for name, cord in polys.items():
        c = to_head(cord, head_pos, head_quat)
        for a, b in zip(c[:-1], c[1:]):
            if (a[1] > 0.0) != (b[1] > 0.0):
                q = a + (b - a) * (a[1] / (a[1] - b[1]))
                if _point_in_poly((q[0], q[2]), loop):
                    out.append(name)
                    break
    return tuple(out)


def _thread_plan(tr: Trial, b: Build) -> dict:
    """Round 20 (23), stage B of the lift start (THREAD_* above): change grips, lift and slide, level the head, take the
    tail, over the bundle, thread it down through the slot, let go. The closed loop (task.ThreadSteering) moves the
    waypoints named in THREAD_STRAP_LABELS, THREAD_HEAD_LABELS, THREAD_TAIL_LABELS and THREAD_STEER_LABELS onto where
    the strap, the head, the tail and the tip really are."""
    if tr.b_slide_first:
        return _thread_slide_first_plan(tr, b)
    up = np.array([0.0, 0.0, 1.0])
    tg = thread_targets(tr)
    left, right = [], []
    hf = tr.head_fingers
    # the right picks the tie up by its head (stage A's down grip) and lifts it to where the grips change
    head0 = np.asarray(b.head_pos, float)
    across = b.head_rot[:, 1]
    rot0 = rot_down(math.degrees(math.atan2(across[0], -across[1])) + 180.0)
    drop = np.array([0.0, 0.0, -HEAD[2] / 2 + 0.8 * MM + PAD_PAST_TCP])

    def down_tcp(c):  # stage A's grip: 2 mm toward the head's back, pad tips 0.8 mm above its underside
        return np.asarray(c, float) - b.head_rot[:, 0] * 2.0 * MM + drop

    g = tg["regrasp"]
    at = down_tcp(head0)
    right += [(2.5, tuple(at + 0.05 * up), "wide", rot0, "to the head"),
              (1.5, tuple(at), "wide", rot0, "descend"),
              (1.0, tuple(at), hf, rot0, "grip"),
              (1.5, tuple(at + 0.03 * up), hf, rot0, "raise tie"),
              (2.0, tuple(down_tcp(g)), hf, rot0, "to regrasp")]
    # the left holds the strap b_hold_s out from the head's face, fingers on its edges, pointing -y tilted b_hold_phi
    # down (round 20's down grip, 90, put both forearms within 3 mm of each other: armcheck.arms_clearance)
    lh = thread_left_rot(tr.b_hold_phi)
    l_at = tg["strap"] - GRIP_DEPTH * lh[:, 2]
    l_pre = l_at - 0.015 * lh[:, 2]
    _wait(left, _seconds(right))
    left += [(2.5, tuple(l_pre), "wide", lh, "to the strap"),
             (1.2, tuple(l_at), "wide", lh, "down to the strap"),
             (0.8, tuple(l_at), "closed", lh, "hold strap")]
    # the right lets go, comes round behind the head and takes it again from behind
    _wait(right, _seconds(left))
    rb = back_rot(B_GRIP_TILT)
    bt = back_tcp(g, B_GRIP_TILT)
    # (round 22: its fingers open 24 mm round the head, not 14)
    right += [(0.6, tuple(down_tcp(g)), "wide", rot0, "release pick"),
              (1.0, tuple(down_tcp(g) + 0.03 * up), "wide", rot0, "up from head"),
              (1.5, tuple(bt - 0.035 * rb[:, 2] - 0.02 * rb[:, 1]), "head_open", rb, "around"),  # low: the forearm, the TV
              (1.5, tuple(bt - 0.025 * rb[:, 2]), "head_open", rb, "behind head"),
              (1.2, tuple(bt), "head_open", rb, "onto head"),
              (1.0, tuple(bt), hf, rb, "grip head")]
    _wait(left, _seconds(right))
    left += [(0.6, tuple(l_at), "wide", lh, "let go strap"),
             (1.0, tuple(l_pre), "wide", lh, "up from strap")]
    t_freed = _seconds(left)  # round 22: the right holds still until the left is off the strap (job 96: carrying while
    #                           the left let go, the head squeezed out of the new grip)
    # stage A's lift
    pinch = np.array([LIFT_AT[0], LIFT_AT[1], TOP + 1.0 * MM + PAD_PAST_TCP])
    crot = rot_down(tr.lift_yaw)
    left += [(2.5, tuple(pinch + 0.06 * up), "cords", crot, "to the cords"),
             (1.5, tuple(pinch), "cords", crot, "down to the cords"),
             (1.0, tuple(pinch), "closed", crot, "pinch"),
             (2.0, tuple(pinch + tr.lift_h * up), "closed", crot, "lift cords"),
             (1.0, tuple(pinch + tr.lift_h * up), "closed", crot, "hold cords")]
    t_lifted = _seconds(left)
    # the right pitches the head down, carries it to the slide's start, waits for the lift, slides the strap under
    ts = B_GRIP_TILT + tr.b_slide_pitch
    rs = back_rot(ts)
    hc = tg["head"]
    s0 = np.array([SLIDE_XY[0], SLIDE_TIP[0] - tip_from_head(b), tr.b_slide_z])
    s1 = np.array([SLIDE_XY[0], hc[1], tr.b_slide_z])
    _wait(right, t_freed)
    right.append((2.5, tuple(back_tcp(s0, ts)), hf, rs, "carry tie"))
    _wait(right, t_lifted)
    right += [(4.0, tuple(back_tcp(s1, ts)), hf, rs, "slide under"),
              (0.5, tuple(back_tcp(s1, ts)), hf, rs, "hold tie")]
    # the left lowers the bundle onto the strap and lets go
    _wait(left, _seconds(right))
    low = pinch + tr.b_lower * up
    left += [(2.0, tuple(low), "closed", crot, "lower cords"),
             (0.8, tuple(low), "cords", crot, "let go cords"),
             (1.5, tuple(low + 0.06 * up), "cords", crot, "clear cords")]
    # the right levels the head beside the bundle and raises it: slot vertical (b_insert_pitch)
    _wait(right, _seconds(left))
    ti = B_GRIP_TILT + tr.b_insert_pitch
    ri = back_rot(ti)
    r_at = back_tcp(hc, ti)
    right.append((2.5, tuple(r_at), hf, ri, "level head"))  # round 22: 2.5 s (was 1.5)
    _thread_tail(tr, tg, left, right, r_at, ri)
    return {"left": left, "right": right}


def _thread_tail(tr: Trial, tg: dict, left: list, right: list, r_at, ri) -> None:
    """Stage B's end, the head levelled (the right at ``r_at``, ``ri``): the left takes the tail where it lies past
    the bundle, b_gft from its tip, lifts it, goes over the bundle turning so the tail points down along the slot,
    lines the tip up on the slot's line and pushes it through; then both let go."""
    up = np.array([0.0, 0.0, 1.0])
    lr90 = thread_left_rot(90.0)
    rl = thread_left_rot(tr.b_lift_phi)  # round 25: tilted as the tail rises, about the strap in its pads (not its TCP)
    _wait(left, _seconds(right))
    t_at = tg["tail"] + B_TAIL_DEPTH * up
    left += [(2.5, tuple(t_at + 0.05 * up), "wide", lr90, "to the tail"),
             (1.2, tuple(t_at), "wide", lr90, "down to the tail"),
             (1.0, tuple(t_at), "closed", lr90, "take tail"),
             (tr.b_lift_s, tuple(t_at + np.array([0.0, tr.b_lift[0], tr.b_lift[1]])
                                 + B_TAIL_DEPTH * (lr90[:, 2] - rl[:, 2])), "closed", rl, "lift tail")]
    # up and over the bundle, turning so the tail points down along the slot, then the tip onto the slot's line
    ro, rt = thread_left_rot(45.0), thread_left_rot(tr.b_phi)
    left.append((tr.b_over_s, tuple(np.asarray(tr.b_over, float) - B_TAIL_DEPTH * ro[:, 2]), "closed", ro, "over"))
    e, d, w = tg["entry"], tg["through"], rt[:, 1]

    def l_tcp(tip):  # the left's TCP with the tip at ``tip``: the strap b_gft back along it, B_TAIL_DEPTH in the pads
        return np.asarray(tip, float) - tr.b_gft * w - B_TAIL_DEPTH * rt[:, 2]

    pushed = l_tcp(e + d * (HEAD[2] + tr.b_out))
    left += [(tr.b_above_s, tuple(l_tcp(e - d * (B_TIP_GAP + B_ABOVE))), "closed", rt, "above head"),
             (1.5, tuple(l_tcp(e - d * B_TIP_GAP)), "closed", rt, "align"),
             (2.5, tuple(pushed), "closed", rt, "push"),
             (1.0, tuple(pushed), "closed", rt, "push hold"),
             (0.8, tuple(pushed), "wide", rt, "let go tail"),
             (1.2, tuple(pushed - 0.04 * rt[:, 2]), "wide", rt, "back off tail"),
             _ready(2.0)]
    # the right holds the head until the left has gone, then lets go and backs off along its approach
    _wait(right, _seconds(left) - 2.0 + tr.b_hold_after)
    right += [(0.6, tuple(r_at), "wide", ri, "let go head"),
              (1.2, tuple(r_at - 0.04 * ri[:, 2]), "wide", ri, "back off head"),
              _ready(2.0)]


B_CLEAR_Y = 0.200  # round 24: the left waits over the far side of the bundle while the right carries the head past
#                    where it held the strap (waiting there, its fingertip came 0.6 mm from the right's hand)


def _thread_slide_first_plan(tr: Trial, b: Build) -> dict:
    """Round 24, stage B for the longer ties (Trial.b_slide_first). Round 23's order (grips changed first, then the
    slide with the grip from behind) can't take a longer strap: held up at b_regrasp it reaches into the cords. So
    stage A's lift and slide come first, the right still in the pick's down grip, carrying the head to b_regrasp's y
    (the tip about 145 mm past the bundle on a 300 mm tie); the left lowers the bundle onto the strap and lets go; the
    grips change there (the head raised to b_regrasp, 140 mm clear of the cords); the right carries the head to b_head
    (the strap slides on under the lowered cords: they weigh it down about 0.1 N, it buckles past 1 N); then the tail
    as in round 23, over the bundle high (b_over) so the strap trails above the left's fingers as they come down."""
    up = np.array([0.0, 0.0, 1.0])
    tg = thread_targets(tr)
    left, right = [], []
    hf = tr.head_fingers
    head0 = np.asarray(b.head_pos, float)
    across = b.head_rot[:, 1]
    rot0 = rot_down(math.degrees(math.atan2(across[0], -across[1])) + 180.0)
    drop = np.array([0.0, 0.0, -HEAD[2] / 2 + 0.8 * MM + PAD_PAST_TCP])

    def down_tcp(c):  # stage A's grip: 2 mm toward the head's back, pad tips 0.8 mm above its underside
        return np.asarray(c, float) - b.head_rot[:, 0] * 2.0 * MM + drop

    # stage A: the left lifts the bundle; the right picks the tie up by its head and slides the strap under
    pinch = np.array([LIFT_AT[0], LIFT_AT[1], TOP + 1.0 * MM + PAD_PAST_TCP])
    crot = rot_down(tr.lift_yaw)
    left += [(2.5, tuple(pinch + 0.06 * up), "cords", crot, "to the cords"),
             (1.5, tuple(pinch), "cords", crot, "down to the cords"),
             (1.0, tuple(pinch), "closed", crot, "pinch"),
             (2.0, tuple(pinch + tr.lift_h * up), "closed", crot, "lift cords"),
             (1.0, tuple(pinch + tr.lift_h * up), "closed", crot, "hold cords")]
    t_lifted = _seconds(left)
    g = tg["regrasp"]
    z_slide = TOP + HEAD[2] / 2 + tr.slide_h
    s0 = np.array([SLIDE_XY[0], SLIDE_TIP[0] - tip_from_head(b), z_slide])
    s1 = np.array([SLIDE_XY[0], g[1], z_slide])
    at = down_tcp(head0)
    right += [(2.5, tuple(at + 0.05 * up), "wide", rot0, "to the head"),
              (1.5, tuple(at), "wide", rot0, "descend"),
              (1.0, tuple(at), hf, rot0, "grip"),
              (1.5, tuple(at + 0.03 * up), hf, rot0, "raise tie"),
              (2.0, tuple(down_tcp(s0)), hf, rot0, "carry tie")]
    _wait(right, t_lifted)
    right += [(tr.b_slide_s, tuple(down_tcp(s1)), hf, rot0, "slide under"),
              (0.5, tuple(down_tcp(s1)), hf, rot0, "hold tie")]
    # the left lowers the bundle onto the strap, lets go and clears it
    _wait(left, _seconds(right))
    low = pinch + tr.b_lower * up
    left += [(2.0, tuple(low), "closed", crot, "lower cords"),
             (0.8, tuple(low), "cords", crot, "let go cords"),
             (1.5, tuple(low + 0.06 * up), "cords", crot, "clear cords")]
    # the grips change at b_regrasp: the right raises the head, the left holds the strap b_hold_s out, the right lets
    # go and takes the head again from behind, the left lets go
    _wait(right, _seconds(left))
    right.append((1.5, tuple(down_tcp(g)), hf, rot0, "to regrasp"))
    lh = thread_left_rot(tr.b_hold_phi)
    l_at = tg["strap"] - GRIP_DEPTH * lh[:, 2]
    l_pre = l_at - 0.015 * lh[:, 2]
    _wait(left, _seconds(right))
    left += [(2.5, tuple(l_pre), "wide", lh, "to the strap"),
             (1.2, tuple(l_at), "wide", lh, "down to the strap"),
             (0.8, tuple(l_at), "closed", lh, "hold strap")]
    _wait(right, _seconds(left))
    rb = back_rot(B_GRIP_TILT)
    bt = back_tcp(g, B_GRIP_TILT)
    right += [(0.6, tuple(down_tcp(g)), "wide", rot0, "release pick"),
              (1.0, tuple(down_tcp(g) + 0.03 * up), "wide", rot0, "up from head"),
              (1.5, tuple(bt - 0.035 * rb[:, 2] - 0.02 * rb[:, 1]), "head_open", rb, "around"),
              (1.5, tuple(bt - 0.025 * rb[:, 2]), "head_open", rb, "behind head"),
              (1.2, tuple(bt), "head_open", rb, "onto head"),
              (1.0, tuple(bt), hf, rb, "grip head")]
    _wait(left, _seconds(right))
    left += [(0.6, tuple(l_at), "wide", lh, "let go strap"),
             (1.0, tuple(l_pre), "wide", lh, "up from strap"),
             (1.5, (SLIDE_XY[0], B_CLEAR_Y, TOP + 0.120), "wide", lh, "clear the head's way")]  # out of the carry's path
    # the right carries the head to the bundle at b_regrasp's height, then levels it there: slot vertical
    _wait(right, _seconds(left))
    hc = tg["head"]
    ti = B_GRIP_TILT + (tr.b_wait_pitch if tr.b_cinch else tr.b_insert_pitch)
    ri = back_rot(ti)
    r_at = back_tcp(hc, ti)
    right += [(2.5, tuple(back_tcp(np.array([hc[0], hc[1], g[2]]), B_GRIP_TILT)), hf, rb, "carry head"),
              (2.5, tuple(r_at), hf, ri, "level head")]
    if tr.b_wedge:
        _wedge_tail(tr, tg, left, right, r_at, ri)
    elif tr.b_keys:  # the left takes the tail where it lies and lifts it, as rounds 23-25 did; then both hold
        up = np.array([0.0, 0.0, 1.0])
        lr90 = thread_left_rot(90.0)
        _wait(left, _seconds(right))
        t_at = tg["tail"] + B_TAIL_DEPTH * up
        left += [(2.5, tuple(t_at + 0.05 * up), "wide", lr90, "to the tail"),
                 (1.2, tuple(t_at), "wide", lr90, "down to the tail"),
                 (1.0, tuple(t_at), "closed", lr90, "take tail"),
                 (tr.b_lift_s, tuple(t_at + np.array([0.0, tr.b_lift[0], tr.b_lift[1]])), "closed", lr90, "lift tail")]
    elif tr.b_cinch:
        _cinch_tail(tr, tg, left, right, r_at, ri)
    else:
        _thread_tail(tr, tg, left, right, r_at, ri)
    return {"left": left, "right": right}


# -- round 26: the cinch ---------------------------------------------------------------------------------------------
# Rounds 23-25 turned the tail the short way (tip down, 90 degrees) while the hand carried it over the bundle. A strap
# looped round the bundle has to turn the long way (tip up, over, back and down: 270 degrees), so the strap behind the
# pinch was wound against itself and levered the tail back (job 101's 47-50 degrees). A 2D model of the strap on a free
# pivot gives job 101's 43 degrees tip-up at the end of the lift. Followed step by step along the carry, it shows the
# way matters: taken straight to a point past the bundle, the strap arches and the tail points away from the head
# (135 degrees off); taken up over the bundle first (115 mm up, where the tail stands straight up and tips over), then
# down past its far side, the tail ends pointing back over the bundle at the head, 10 degrees above level, its tip
# 40 mm over the head. The tail sweeps over the head on the way, so the right holds the head low and level, its hand
# behind it, until the tail is there. So: the left
# carries the tail over on an eased pinch (it pivots), closes, turns its wrist on until the tail aims B_AIM below level
# (closed loop), the right turns the head to face it (pitch 90 minus the tail's angle, at most CINCH_PITCH_MAX: its hand
# straight down), the left lines the tip up and pushes about 34 mm through in one stroke (its fingers stay over the
# bundle, clear of the right's). Then the right lets go of the head (the left's pinch and the latch hold it) and takes
# the tail below the exit; the left lets go of the strap and takes the head from above (its fingers close across the
# head whatever its pitch); both raise the tie; the right draws the tail straight out along the slot; both let go.
B_AIM = 17.0  # deg: the tail's aim below level (pointing -y) to thread; the head's pitch is then 73
CINCH_PITCH_MAX = 75.0  # the right holding the head from behind with its hand straight down (tilt 90: its reach)
CINCH_PITCH_MIN = 30.0
CINCH_OUT = 0.034  # the push leaves this much tail out of the exit (the latch drops at 16)
CINCH_DEPTH_CAP = 0.004  # ... the closed loop going at most this much further, if the push ends short
CINCH_BACK = 0.015  # the tip waits this far short of where the entry will be while the head turns to face it
CINCH_HEAD_IN = 0.004  # the left takes the head with its fingertips this far past the head's centre
CINCH_HEAD_DY = 0.004  # ... and its hand this far toward the bundle (+y) of it
CINCH_STEER_CAP = 0.015  # the left's shift onto the tip's offset from the slot's line, at most
CINCH_SNAP = 0.030  # the closed loop moves a hand at most this far from its planned place
CINCH_TURN = (0.0, 45.0)  # deg: ... and turns the left's wrist on by this much, at least and at most (past 55
#                           degrees of tilt its hand stands beside the right's; below 10 it is out of reach)
CINCH_LEAD = 0.015  # the pulling hand stays at most this far ahead of the strap's advance
CINCH_PAD = 0.0137  # half a pad, across the fingers (finger_boxes)
CINCH_LEFT_STEER = ("align", "push", "push hold")
CINCH_LEFT_HEAD = ("over head", "down to head", "grip head 2", "raise head", "hold head", "lower head", "let go head 2",
                   "up from head 2")
CINCH_RIGHT_TAIL = ("to the tail 2", "take tail 2", "raise tail", "pull", "pull hold", "let go tail 2", "back off tail 2")
CINCH_PULL_LABELS = ("pull", "pull hold", "let go tail 2", "back off tail 2")


def cinch_frame(hc, pitch_deg: float) -> dict:
    """The head held at ``hc`` pitched ``pitch_deg`` (head_rot_pitched): the slot's entry and exit centres and the way
    through (world)."""
    hr = head_rot_pitched(pitch_deg)
    hc = np.asarray(hc, float)
    return {"hc": hc, "rot": hr, "pitch": float(pitch_deg), "d": hr @ np.array([0.0, 0.0, -1.0]),
            "entry": hc + hr @ np.array([SLOT_C, 0.0, ENTRY_Z]), "exit": hc + hr @ np.array([SLOT_C, 0.0, EXIT_Z])}


def cinch_left(f: dict, gft: float, phi_deg: float, back: float):
    """The left's (TCP, rotation) with its wrist at ``phi_deg`` and the tail straight along the slot's line, its tip
    ``back`` short of the entry (negative: that far in): the strap's point in the fingers is gft further back."""
    r = thread_left_rot(phi_deg)
    return f["entry"] - f["d"] * (back + gft) - B_TAIL_DEPTH * r[:, 2], r


def cinch_tail_grip(f: dict, at: float, tilt_deg: float):
    """The right's (TCP, rotation) taking the tail below the exit, pointing +y tilted ``tilt_deg`` down, fingers closing
    along x on its edges: the strap's point ``at`` plus half a pad from the exit face GRIP_DEPTH past its TCP."""
    r = back_rot(tilt_deg)
    return f["exit"] + f["d"] * (at + CINCH_PAD) - GRIP_DEPTH * r[:, 2], r


def cinch_head_grip(f: dict, phi_deg: float):
    """The left's (TCP, rotation) taking the head from the bundle's side, pointing -y tilted ``phi_deg`` down, fingers
    closing along x across the head's 9 mm (whatever its pitch): its fingertips CINCH_HEAD_IN past the head's centre,
    the pads shifted CINCH_HEAD_DY toward the bundle so they stay off the tail's side."""
    r = thread_left_rot(phi_deg)
    return f["hc"] - (PAD_PAST_TCP - CINCH_HEAD_IN) * r[:, 2] + np.array([0.0, CINCH_HEAD_DY, 0.0]), r


def tail_aim(polyline, links: int = 4):
    """(tip, unit direction, degrees below level pointing -y) of the strap's last ``links`` links: 0 when the tail
    points level toward -y, 90 straight down, negative tip-up; near +-180 it points +y (as it lies at the start)."""
    poly = np.asarray(polyline, float)
    u = poly[-1] - poly[-1 - links]
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    return poly[-1], u, math.degrees(math.atan2(-u[2], -u[1]))


def cinch_turn(tr: Trial, aim_deg: float) -> float:
    """How far the left's wrist turns on from tr.b_feed_phi to bring a tail measured at ``aim_deg`` to B_AIM (the closed
    pinch turns the tail with it, round 24), within CINCH_TURN."""
    return float(np.clip(B_AIM - aim_deg, CINCH_TURN[0], CINCH_TURN[1]))


def cinch_pitch(aim_deg: float) -> float:
    """The head's pitch that faces a tail aimed ``aim_deg`` below level, within what the right's hand reaches."""
    return float(np.clip(90.0 - aim_deg, CINCH_PITCH_MIN, CINCH_PITCH_MAX))


def cinch_tail_guess(tr: Trial) -> np.ndarray:
    """Where the strap's point b_gft from the tip lies on the console while the head waits level by the bundle: the
    strap guessed 45 degrees down from its root to the console, then flat toward +y, as thread_targets guesses it
    (job 101: 5 mm off; the closed loop finds the real one)."""
    f = cinch_frame(tr.b_head, tr.b_wait_pitch)
    root = f["hc"] + f["rot"] @ ROOT
    h = float(root[2] - TOP)
    return np.array([SLIDE_XY[0], float(root[1]) + h + BUILDS[tr.build].tie_l - math.sqrt(2.0) * h - tr.b_gft,
                     TOP + T / 2])


def _cinch_tail(tr: Trial, tg: dict, left: list, right: list, r_at, ri) -> None:
    """Round 26's end of stage B (the note above). The head is held at tr.b_head pitched tr.b_insert_pitch (the right at
    ``r_at``, ``ri``). Every step from "settle" on is a plan for the tail arriving as the model says; task.CinchSteering
    rewrites them from the tie's state."""
    up = np.array([0.0, 0.0, 1.0])
    f = cinch_frame(tr.b_head, tr.b_insert_pitch)
    d = f["d"]
    lr90 = thread_left_rot(90.0)

    def at(point, rot):  # the left's TCP with the strap's point ``point`` at its fingertips
        return tuple(np.asarray(point, float) - B_TAIL_DEPTH * rot[:, 2])

    # the pick and a short lift, the pinch closed
    _wait(left, _seconds(right))
    t0 = tg["tail"]
    p1 = t0 + np.array([0.0, tr.b_lift[0], tr.b_lift[1]])
    left += [(2.5, at(t0 + 0.05 * up, lr90), "wide", lr90, "to the tail"),
             (1.2, at(t0, lr90), "wide", lr90, "down to the tail"),
             (1.0, at(t0, lr90), "closed", lr90, "take tail"),
             (tr.b_lift_s, at(p1, lr90), "closed", lr90, "lift tail")]
    # up over the bundle on the eased pinch (the tail stands up and tips over toward the head), down past its far
    # side, the wrist winding back all the way; then closed again
    over, top, feed = (np.asarray(v, float) for v in (tr.b_over, tr.b_top, tr.b_feed))
    ro, rt_, rf = (thread_left_rot(v) for v in (50.0, 35.0, tr.b_feed_phi))  # steeper than 60 up there, the left's
    #                                                                         forearm comes within 17 mm of the TV
    left += [(2.0, at(over, ro), tr.b_soft, ro, "over"),
             (2.0, at(top, rt_), tr.b_soft, rt_, "top"),
             (tr.b_above_s, at(feed, rf), tr.b_soft, rf, "feed"),
             (1.0, at(feed, rf), "closed", rf, "settle")]
    # the wrist turns the tail on to its aim, its tip CINCH_BACK short of the entry; the head turns to face it; line
    # up; push
    phi = tr.b_feed_phi + tr.b_turn
    turn = cinch_left(f, tr.b_gft, phi, CINCH_BACK)
    left += [(1.8, tuple(turn[0]), "closed", turn[1], "turn tail"),
             (0.8, tuple(turn[0]), "closed", turn[1], "settle 2")]
    _wait(right, _seconds(left))
    tf = B_GRIP_TILT + tr.b_insert_pitch
    r_at, ri = back_tcp(f["hc"], tf), back_rot(tf)  # from here the right holds the head turned to face the tail
    right.append((2.5, tuple(r_at), tr.head_fingers, ri, "face tail"))
    _wait(left, _seconds(right))
    pushed = cinch_left(f, tr.b_gft, phi, -(HEAD[2] + CINCH_OUT))
    left += [(1.5, tuple(cinch_left(f, tr.b_gft, phi, B_TIP_GAP)[0]), "closed", turn[1], "align"),
             (3.0, tuple(pushed[0]), "closed", turn[1], "push"),
             (1.0, tuple(pushed[0]), "closed", turn[1], "push hold")]
    # the right lets go of the head (the left's pinch and the latch hold the tie) and takes the tail below the exit
    _wait(right, _seconds(left))
    g_at, g_rot = cinch_tail_grip(f, tr.b_tail_at, tr.b_tail_tilt)
    right += [(0.6, tuple(r_at), "head_open", ri, "release head"),
              (1.6, tuple(g_at), "head_open", g_rot, "to the tail 2"),
              (0.8, tuple(g_at), "closed", g_rot, "take tail 2")]
    # the left lets go of the strap and takes the head from above
    _wait(left, _seconds(right))
    h_at, h_rot = cinch_head_grip(f, tr.b_head_phi)
    left += [(0.6, tuple(pushed[0]), "wide", turn[1], "let go strap 2"),
             (1.0, tuple(pushed[0] - 0.035 * turn[1][:, 2]), "wide", turn[1], "up from strap 2"),
             (1.6, tuple(h_at - 0.045 * h_rot[:, 2]), "head_open", h_rot, "over head"),
             (1.2, tuple(h_at), "head_open", h_rot, "down to head"),
             (0.8, tuple(h_at), "head", h_rot, "grip head 2")]
    # both raise the tie; the right draws the tail straight out along the slot; holds; lets go; then the left
    _wait(right, _seconds(left))
    lift = tr.b_pull_lift * up
    drawn = g_at + lift + d * tr.b_pull
    right += [(1.5, tuple(g_at + lift), "closed", g_rot, "raise tail"),
              (tr.b_pull_s, tuple(drawn), "closed", g_rot, "pull"),
              (1.5, tuple(drawn), "closed", g_rot, "pull hold"),
              (0.6, tuple(drawn), "wide", g_rot, "let go tail 2"),
              (1.2, tuple(drawn - 0.05 * g_rot[:, 2]), "wide", g_rot, "back off tail 2"),
              _ready(2.0)]
    left.append((1.5, tuple(h_at + lift), "head", h_rot, "raise head"))
    left.append((_seconds(right) - 3.2 - _seconds(left), tuple(h_at + lift), "head", h_rot, "hold head"))  # until the
    #                                                                                          right has opened
    left += [(1.5, tuple(h_at + 0.010 * up), "head", h_rot, "lower head"),
             (0.6, tuple(h_at + 0.010 * up), "head_open", h_rot, "let go head 2"),
             (1.2, tuple(h_at + 0.010 * up - 0.050 * h_rot[:, 2]), "head_open", h_rot, "up from head 2"),
             (1.5, (BACK_OFF_X, float(h_at[1]), TOP + 0.15), "wide", h_rot, "back off"),
             _ready(2.0)]


# -- round 29: the wedge -------------------------------------------------------------------------------------------------
# Rounds 24, 25 and 28 (jobs 109, 114 and the cloud run) say the same thing about the closed pinch: taken at the
# fingertips, the strap turns about 48 degrees about the fingers' closing axis and then locks (it wedges diagonally
# between the pads), and from there follows the wrist to the degree. Eased to 1-2 N the pinch drops it (round 28). It
# has always locked tip-up, because lifting pulls the strap behind the fingers down: the tail then points 48 degrees
# above the wrist's line, and with the wrist at its limit that is still 42 degrees short of straight down. Locked the
# other way (the strap behind up between the fingers, the tail 48 degrees below the wrist's line) the tail points
# straight down with the wrist tilted about 58, and the strap behind leaves the fingers upward, into the loop over the
# bundle, whose spring then presses it further into that lock. To lock it that way the left pinches the tail where it
# lies and pushes it toward the bundle along the console: the strap behind arches up, and as the fingers lift, the
# arch turns the strap in the pinch tip-down. At "settle" and again over the bundle ("settle 2") the closed loop
# (task.WedgeSteering) reads the tail's angle off the wrist's and sets the wrist and the way down onto the slot.
WEDGE_DEG = 46.0  # how far the strap turns in the pinch before it locks (46 to 51 in the four runs)
WEDGE_PHI = (5.0, 62.0)  # the left's tilt while threading, at least and at most (steeper, its forearm comes
#   within 20 mm of the TV; flatter, its fingers come down on the right's)
WEDGE_TAIL_LABELS = ("to the tail", "down to the tail", "take tail", "arch", "lift tail", "settle")
WEDGE_FEED_LABELS = ("above head", "align", "push", "push hold", "let go tail", "back off tail")


def heading_deg(v) -> float:
    """A direction's angle in the tie's plane (y, z): 0 pointing +y, 90 up, -90 down, +-180 pointing -y."""
    return math.degrees(math.atan2(float(v[2]), float(v[1])))


def turned(v, deg: float) -> np.ndarray:
    """``v`` turned ``deg`` in the tie's plane (about x: its heading grows by deg)."""
    a = math.radians(deg)
    c, s_ = math.cos(a), math.sin(a)
    return np.array([float(v[0]), c * float(v[1]) - s_ * float(v[2]), s_ * float(v[1]) + c * float(v[2])])


def wedge_phi(d, offset_deg: float) -> float:
    """The left's tilt (thread_left_rot) that points the tail along ``d`` when the strap sits ``offset_deg`` off the
    wrist's line (the tail's heading is the wrist's, phi - 90, plus the offset), within WEDGE_PHI."""
    return float(np.clip(90.0 + heading_deg(d) - offset_deg, WEDGE_PHI[0], WEDGE_PHI[1]))


def wedge_tail_vec(tr: Trial) -> np.ndarray:
    """The plan's tail at "settle" (wrist at 90): from the strap's point in the fingers to the tip, locked tip-down."""
    a = math.radians(-WEDGE_DEG)
    return tr.b_gft * np.array([0.0, math.cos(a), math.sin(a)])


def wedge_feed(tr: Trial, entry, d, tail_vec, phi_at: float, steer=(0.0, 0.0, 0.0), depth: float = 0.0) -> dict:
    """The left's way from over the bundle down through the slot, for a tail that points ``tail_vec`` (from the strap's
    point at its fingertips to the tip) while its wrist is tilted ``phi_at``: the tilt that points that tail along the
    slot (``d``, from the entry at ``entry``), and with it the TCP over the bundle (tr.b_over), with the tip B_ABOVE
    short of the line-up, lined up B_TIP_GAP before the entry, pushed tr.b_out (+ ``depth``) out of the exit, and backed
    off; ``steer`` shifts the last four sideways."""
    entry, d, steer = np.asarray(entry, float), np.asarray(d, float), np.asarray(steer, float)
    off = heading_deg(tail_vec) - (phi_at - 90.0)
    phi = wedge_phi(d, off)
    rot = thread_left_rot(phi)
    back = turned(tail_vec, phi - phi_at) + B_TAIL_DEPTH * rot[:, 2]  # from the TCP to the tip
    pushed = entry + d * (HEAD[2] + tr.b_out + depth) + steer - back
    return {"phi": phi, "offset": off, "rot": rot,
            "over": np.asarray(tr.b_over, float) - B_TAIL_DEPTH * rot[:, 2],
            "above head": entry - d * (B_TIP_GAP + B_ABOVE) + steer - back,
            "align": entry - d * B_TIP_GAP + steer - back,
            "push": pushed, "back off tail": pushed - 0.04 * rot[:, 2]}


def draw_dir(tr: Trial) -> np.ndarray:
    """The way the right draws the tail out: toward itself (-y), rising tr.b_pull_up per unit."""
    v = np.array([0.0, -1.0, tr.b_pull_up])
    return v / float(np.linalg.norm(v))


def _wedge_tail(tr: Trial, tg: dict, left: list, right: list, r_at, ri) -> None:
    """Round 29's end of stage B (the note above), the head held low and level as in rounds 23-25: take the tail, push
    it tr.b_arch toward the bundle along the console, lift it tr.b_lift from there, settle; then over the bundle,
    settle again, and down onto the slot with the wrist set for a tail WEDGE_DEG below its line: line up, push, let
    go. Every step from "over" on is the plan for that lock; the closed loop rewrites them from what the tail does."""
    lr90 = thread_left_rot(90.0)

    def at(point, rot):  # the left's TCP with the strap's point ``point`` at its fingertips
        return tuple(np.asarray(point, float) - B_TAIL_DEPTH * rot[:, 2])

    _wait(left, _seconds(right))
    t0 = tg["tail"]
    low = t0 + np.array([0.0, -tr.b_arch, 0.002])
    held = low + np.array([0.0, tr.b_lift[0], tr.b_lift[1]])
    left += [(2.5, at(t0 + np.array([0.0, 0.0, 0.05]), lr90), "wide", lr90, "to the tail"),
             (1.2, at(t0, lr90), "wide", lr90, "down to the tail"),
             (1.0, at(t0, lr90), "closed", lr90, "take tail"),
             (2.0, at(low, lr90), "closed", lr90, "arch"),
             (tr.b_lift_s, at(held, lr90), "closed", lr90, "lift tail"),
             (1.0, at(held, lr90), "closed", lr90, "settle")]
    f = wedge_feed(tr, tg["entry"], tg["through"], wedge_tail_vec(tr), 90.0)
    rt = f["rot"]
    left += [(tr.b_over_s, tuple(f["over"]), "closed", rt, "over"),
             (0.8, tuple(f["over"]), "closed", rt, "settle 2"),
             (tr.b_above_s, tuple(f["above head"]), "closed", rt, "above head"),
             (1.5, tuple(f["align"]), "closed", rt, "align"),
             (2.5, tuple(f["push"]), "closed", rt, "push"),
             (tr.b_push_s, tuple(f["push"]), "closed", rt, "push"),
             (1.0, tuple(f["push"]), "closed", rt, "push hold")]
    let_go = [(0.8, tuple(f["push"]), "wide", rt, "let go tail"),
              (1.2, tuple(f["back off tail"]), "wide", rt, "back off tail")]
    if not tr.b_draw:
        left += let_go + [_ready(2.0)]
        _wait(right, _seconds(left) - 2.0 + tr.b_hold_after)
        right += [(0.6, tuple(r_at), "wide", ri, "let go head"),
                  (1.2, tuple(r_at - 0.04 * ri[:, 2]), "wide", ri, "back off head"),
                  _ready(2.0)]
        return
    # round 30, the draw: the left's pinch and the latch hold the tie while the right lets go of the head and takes
    # the tail below the exit; the left lets go (and, tr.b_hold_cords, pins the bundle where it pinched it to lift
    # it); the right draws the tail out toward itself, the head turning to follow and coming up against the bundle
    e, d = np.asarray(tg["entry"], float), np.asarray(tg["through"], float)
    gp, gr = cinch_tail_grip({"exit": e + d * HEAD[2], "d": d}, tr.b_tail_at, tr.b_tail_tilt)
    _wait(right, _seconds(left))
    right += [(0.6, tuple(r_at), "head_open", ri, "release head"),
              (1.2, tuple(r_at - 0.04 * ri[:, 2]), "head_open", ri, "back off head"),
              (1.6, tuple(gp - 0.035 * gr[:, 2]), "head_open", gr, "to the tail 2"),
              (1.0, tuple(gp), "head_open", gr, "onto tail 2"),
              (0.8, tuple(gp), "closed", gr, "take tail 2")]
    _wait(left, _seconds(right))
    left += let_go
    if tr.b_hold_cords:
        up = np.array([0.0, 0.0, 1.0])
        low = np.array([LIFT_AT[0], LIFT_AT[1], TOP + 1.0 * MM + PAD_PAST_TCP]) + tr.b_lower * up
        crot = rot_down(tr.lift_yaw)
        left += [(2.5, tuple(low + 0.06 * up), "cords", crot, "to the cords 2"),
                 (1.2, tuple(low), "cords", crot, "down to the cords 2"),
                 (0.8, tuple(low), "closed", crot, "pin cords")]
    else:
        left.append(_ready(2.0))
    _wait(right, _seconds(left) - (0.0 if tr.b_hold_cords else 2.0))
    drawn = gp + draw_dir(tr) * tr.b_pull
    right += [(tr.b_pull_s, tuple(drawn), "closed", gr, "pull"),
              (1.0, tuple(drawn), "closed", gr, "pull hold"),
              (0.8, tuple(drawn), "wide", gr, "let go tail 2"),
              (1.2, tuple(drawn - 0.05 * gr[:, 2]), "wide", gr, "back off tail 2"),
              _ready(2.0)]
    if tr.b_hold_cords:
        _wait(left, _seconds(right) - 2.0)
        left += [(0.8, tuple(low), "cords", crot, "let go cords 2"),
                 (1.2, tuple(low + 0.06 * up), "cords", crot, "clear cords 2"),
                 _ready(2.0)]


# -- round 31: the hanging cords again, the strap held by its faces -----------------------------------------------------
# Rounds 19-29 could not thread on the lying cords: there the loop stands in a vertical plane, both hands pinch the
# strap by its edges (their fingers close across its width, the axis it bends about), and such a pinch is a pivot: the
# strap turns in it about 50 degrees either way, and the tail has to turn 270 degrees round the bundle, the first 180
# of them tipping over backward, further than either wrist can follow (jobs 109, 114, 116; the 2D strap model agrees).
# On the first submission's hanging cords the loop lies level. Turned on edge (a quarter turn about its length) the
# strap can't sag: it stands out like a ruler, a hand reaching down closes on its flat faces, and its direction is that
# wrist's turn about the vertical. No stand, no loop made beforehand: the tie starts flat on the console (300 mm),
# the right takes its head and the left its tail, they lift it straight between them and turn it on edge together
# (lifted by the head alone the heavy strap hangs, and swings into the cords as the hand turns); the left lets go; the
# right, its fingers above and below the head, passes the ruler behind the cords and swings its tail round to the
# front, where the left can reach down to it (behind the cords' plane its upper arm is in the TV: armcheck).
HANG_BUILDS = ("hang30", "hang37")
HANG_DEMO = {"hang30": "h30-take", "hang37": "h37-hold-l"}  # what task.demo() plays on each (hang37 is the task)
HANG_PHYS = "nylonlead"  # the physics preset of the hanging-cords builds (PHYS): the task's  # round 33: 370 mm (job 118: the loop round the cords and both hands takes
#                                      about 45 mm more strap than the 300 mm tie has)
RY_ROLL = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # the quarter turn: what pointed down points
#                                                                           at the wall (+x)
HANG_LEFT_TAIL = ("to the tail 2", "down to the tail 2", "take tail 2", "hold tail")
HANG_PIVOT = np.array([0.3995, 0.0])  # the ruler passes behind the cords here (their backs are at x 0.385) and swings
#                                       about this point


def _hang(name: str = "hang30", n_links: int = 48) -> Build:
    pos = np.array([0.290, -0.170, TOP + HEAD[2] / 2 + 0.3 * MM])  # further out or left, the left's forearm is at
    #   the cords when it takes the tail; nearer or further right, the right's wrist can't turn the head on edge
    return Build(name, False, tuple(float(v) for v in pos), R_FREE, free_points(n_links),
                 f"the hanging cords (the first submission's), no stand: a {n_links * LINK * 1000:.0f} mm tie lying "
                 "flat on the console in front of them, free (round 31)")


BUILDS["hang30"] = _hang()
BUILDS["hang37"] = _hang("hang37", 59)


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def hang_pick(b: Build):
    """The right's pick of the lying head (down grip, fingers across its 9 mm): its TCP, its rotation, and the head's
    centre in the TCP's frame (the same whatever way the hand turns afterwards)."""
    head = np.asarray(b.head_pos, float)
    across = b.head_rot[:, 1]
    rot = rot_down(math.degrees(math.atan2(across[0], -across[1])))
    at = head - b.head_rot[:, 0] * 2.0 * MM + np.array([0.0, 0.0, -HEAD[2] / 2 + 0.8 * MM + PAD_PAST_TCP])
    return at, rot, rot.T @ (head - at)


def hang_ruler(b: Build, swing_deg: float, z: float, root_back: float = 0.030) -> dict:
    """The tie on edge behind the cords, its strap through HANG_PIVOT turned ``swing_deg`` from +y toward the robot
    (the tail's side), its root ``root_back`` short of that point, its centre line ``z`` up: the head's rotation and
    centre, the root, the strap's direction."""
    hr = rot_z(swing_deg) @ RY_ROLL @ np.asarray(b.head_rot, float)
    u = hr[:, 0]
    root = np.array([HANG_PIVOT[0], HANG_PIVOT[1], z]) - u * root_back
    return {"rot": hr, "u": u, "root": root, "head": root - hr @ ROOT}


def face_rot(strap_dir, toward_wall: bool = False) -> np.ndarray:
    """A hand reaching straight down that closes on the faces of a strap standing on edge and running ``strap_dir``
    (level). Its fingers close across the strap either way round (the same grip, the wrist half a turn apart):
    ``toward_wall`` picks the closing direction with +x in it; the left's wrist reaches the other (IK, 23 Nov)."""
    d = np.asarray(strap_dir, float)
    c = np.array([d[1], -d[0], 0.0])
    c = c / float(np.linalg.norm(c))
    return rot_axes((0.0, 0.0, -1.0), c if (c[0] >= 0) == toward_wall else -c)


def face_tcp(point) -> np.ndarray:
    """The TCP of that hand with the strap's centre line at ``point`` GRIP_DEPTH past it: its pad tips 2.4 mm below
    the strap's lower edge."""
    return np.asarray(point, float) + np.array([0.0, 0.0, GRIP_DEPTH])


def _hang_plan(tr: Trial, b: Build) -> dict:
    """Round 31, the first steps on the hanging cords (the note above). Both: take the tie where it lies (the right
    the head, the left the tail tr.h_first from its tip, by its edges), lift it tr.h_z straight between them, turn it
    on edge together. The left lets go. Right: carry the ruler out to tr.h_lane and back until its tip is short of
    the cords, in behind their plane, pass it along +y behind them, swing its tail tr.h_swing degrees round to the
    front, lower it to tr.h_low. Left: reach down and take the tail by its faces, tr.h_gft from the tip. Then the
    right lets go of the head and backs off: the left holds the tie alone for tr.h_hold."""
    up = np.array([0.0, 0.0, 1.0])
    at, r0, o = hang_pick(b)
    r1 = RY_ROLL @ r0
    hf = tr.head_fingers
    z, z_low = TOP + tr.h_z, TOP + tr.h_low
    head0 = np.asarray(b.head_pos, float)
    hr0, hr1 = np.asarray(b.head_rot, float), RY_ROLL @ np.asarray(b.head_rot, float)

    def r_tcp(head_c, rot):
        return tuple(np.asarray(head_c, float) - rot @ o)

    lifted = np.array([head0[0], head0[1], z])
    right = [(2.5, tuple(at + 0.05 * up), "wide", r0, "to the head"),
             (1.5, tuple(at), "wide", r0, "descend"),
             (1.0, tuple(at), hf, r0, "grip"),
             (3.0, r_tcp(lifted, r0), hf, r0, "lift"),
             (tr.h_roll_s, r_tcp(lifted, r1), hf, r1, "turn on edge"),
             (1.0, r_tcp(lifted, r1), hf, r1, "settle")]
    # the left on the tail, in step with the right
    l0 = thread_left_rot(90.0)
    l1 = RY_ROLL @ l0
    along = b.tie_l - tr.h_first
    p0 = head0 + hr0 @ (ROOT + np.array([along, 0.0, 0.0]))  # the strap's point as it lies
    p_up = lifted + hr0 @ (ROOT + np.array([along, 0.0, 0.0]))
    p_edge = lifted + hr1 @ (ROOT + np.array([along, 0.0, 0.0]))

    def l_tcp(point, rot):
        return tuple(np.asarray(point, float) - B_TAIL_DEPTH * rot[:, 2])

    left = [(2.5, l_tcp(p0 + 0.05 * up, l0), "wide", l0, "to the tail"),
            (1.5, l_tcp(p0, l0), "wide", l0, "down to the tail"),
            (1.0, l_tcp(p0, l0), "closed", l0, "take tail"),
            (3.0, l_tcp(p_up, l0), "closed", l0, "lift tail"),
            (tr.h_roll_s, l_tcp(p_edge, l1), "closed", l1, "turn tail"),
            (1.0, l_tcp(p_edge, l1), "closed", l1, "settle"),
            (0.8, l_tcp(p_edge, l1), "wide", l1, "let go tail"),
            (1.5, tuple(np.asarray(l_tcp(p_edge, l1)) - 0.03 * l1[:, 2]), "wide", l1, "clear"),
            _ready(2.0)]
    # the right alone: the ruler round behind the cords, its tail swung to the front
    _wait(right, _seconds(left) - 3.0)
    y_from = HANG_PIVOT[1] - tr.h_tip_short - b.tie_l - HEAD[0] / 2
    f0 = hang_ruler(b, 0.0, z, tr.h_root_back)
    start = np.array([f0["head"][0], y_from, z])
    right += [(2.5, r_tcp((tr.h_lane, lifted[1], z), r1), hf, r1, "out"),
              (2.5, r_tcp((tr.h_lane, y_from, z), r1), hf, r1, "back"),
              (2.0, r_tcp(start, r1), hf, r1, "behind"),
              (tr.h_poke_s, r_tcp(f0["head"], r1), hf, r1, "pass behind")]
    for k in (1, 2, 3):
        f = hang_ruler(b, tr.h_swing * k / 3, z, tr.h_root_back)
        right.append((tr.h_swing_s / 3, r_tcp(f["head"], rot_z(tr.h_swing * k / 3) @ r1), hf, rot_z(tr.h_swing * k / 3) @ r1,
                      "swing"))
    f = hang_ruler(b, tr.h_swing, z_low, tr.h_root_back)
    rs = rot_z(tr.h_swing) @ r1
    right += [(2.0, r_tcp(f["head"], rs), hf, rs, "lower"),
              (1.0, r_tcp(f["head"], rs), hf, rs, "hold tie")]
    # the left takes the tail by its faces, in front of the cords
    tail = f["root"] + f["u"] * (b.tie_l - tr.h_gft)
    lr = face_rot(f["u"])
    _wait(left, _seconds(right) - 3.0)
    left += [(2.5, tuple(face_tcp(tail) + 0.06 * up), "wide", lr, "to the tail 2"),
             (1.5, tuple(face_tcp(tail)), "wide", lr, "down to the tail 2"),
             (1.0, tuple(face_tcp(tail)), "closed", lr, "take tail 2"),
             (1.0, tuple(face_tcp(tail)), "closed", lr, "hold tail")]
    _wait(right, _seconds(left))
    last = np.asarray(right[-1][1], float)
    if not tr.h_thread:
        right += [(0.8, tuple(last), "head_open", rs, "release head"),
                  (1.5, tuple(last - 0.025 * rs[:, 2]), "head_open", rs, "back off head"),
                  (tr.h_hold, tuple(last - 0.025 * rs[:, 2]), "head_open", rs, "look")]
        _wait(left, _seconds(right))
        return {"left": left, "right": right}
    _hang_thread(tr, b, left, right, f, o, r1, lr, tail)
    return {"left": left, "right": right}


def left_face(take_rot, turn_deg: float, tilt_deg: float = 0.0, flipped: bool = False) -> np.ndarray:
    """The left's face grip turned ``turn_deg`` about the vertical from where it took the tail (the tail turns with
    it, counter-clockwise seen from above) and leaning ``tilt_deg`` along the strap (negative: its wrist toward the
    tail's tip; its forearm then stays off the cords). ``flipped``: the same grip with the hand half a turn round
    (its wrist joint then has the tail's last turn to spare). The strap's centre line sits GRIP_DEPTH along its
    approach from the TCP."""
    t = math.radians(tilt_deg if flipped else -tilt_deg)
    lean = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(t), -math.sin(t)], [0.0, math.sin(t), math.cos(t)]])
    return rot_z(turn_deg + (180.0 if flipped else 0.0)) @ np.asarray(take_rot, float) @ lean


def face_at(point, rot) -> tuple:
    """The TCP of a face grip ``rot`` with the strap's centre line at ``point``."""
    return tuple(np.asarray(point, float) - GRIP_DEPTH * np.asarray(rot, float)[:, 2])


def hang_head(tr: Trial, b: Build, z: float) -> dict:
    """The head where it is threaded, in front of the cords (tr.h_head_at), the slot's way through heading
    tr.h_d_deg (level, from +x): its rotation, the entry and exit centres and the way through (world)."""
    hr = rot_z(tr.h_d_deg) @ RY_ROLL @ np.asarray(b.head_rot, float)
    c = np.array([tr.h_head_at[0], tr.h_head_at[1], z])
    return {"rot": hr, "head": c, "entry": c + hr @ np.array([SLOT_C, 0.0, ENTRY_Z]),
            "exit": c + hr @ np.array([SLOT_C, 0.0, EXIT_Z]), "d": hr @ np.array([0.0, 0.0, -1.0])}


def heading_xy(v) -> float:
    """A level direction's heading, degrees from +x toward +y."""
    return math.degrees(math.atan2(float(v[1]), float(v[0])))


HANG_ROUND = ((0.405, -0.045), (0.360, -0.055), (0.315, -0.020))  # the head's way round the cords' right, to the front
HANG_LEFT_THREAD = ("aim", "settle 3", "above head", "align", "push", "push hold")
HANG_LEFT_END = ("let go tail 3", "back off tail 3")
HANG_RIGHT_FRONT = ("tauten", "ease back", "settle 2b", "turn head", "hold head 2")
HANG_RIGHT_REGRIP = ("open head", "yaw out", "close head")
HANG_RIGHT_STUB = ("release head", "to the stub", "take stub")
HANG_PULL = ("pull out", "pull", "pull hold", "let go stub", "back off stub")
HANG_LEFT_HEAD = ("to the head 2", "onto the head", "take head 2", "hold head 3", "carry head", "set head",
                  "square head", "at the cords", "let go head 2", "back off head 2")
HANG_RIGHT_SWAP = ("let go stub 1", "back off stub 1", "stand by", "to the stub 2", "take stub 2")
HOLD_TURN = 0.060  # m of the draw over which the right's hand turns from h_stub_yaw to h_pull_yaw
HOLD_EASE = 0.006  # m the right's hand comes back along the draw, from where it really is, before it lets go (job
#                    131: let go at 8 N or so the strap ran back 6 mm; eased, the pawl takes the strap while the
#                    hand still has it)
HOLD_PAST = 6.0 * MM  # how far the left's pads reach past the head's exit face as it holds the head for the draw
HOLD_O = np.array([0.0, -(13.7 * MM - HEAD[2] / 2 - HOLD_PAST), 11.0 * MM])  # the head's centre in the left's TCP
#   frame for that hold: at its fingertips (as the right holds it), and to one side (the pads are 27.4 mm across,
#   the head 5.6). Round 45 (job 130): with the pads' edge flush with the exit face the head left them sideways at
#   about 2 N, the pull a 150 mm loop takes to be bent smaller (till the loop is on the cords the left answers the
#   whole pull); 6 mm in from the edge it sits on the pads' flat. Past the exit they stay short of the right's pads
#   on the tail; what lies past the entry lies in front of the cords (the cords sit toward the strap's root).


def hang_tight(tr: "Trial", b: Build, z: float) -> dict:
    """The head held at the cords for the draw (tr.h_tight): its rotation and centre, the exit's centre, the way
    through (level), and the heading of the hand that holds it."""
    along, into, deg = tr.h_tight
    hr = rot_z(deg - 90.0) @ RY_ROLL @ np.asarray(b.head_rot, float)  # the way through: a quarter turn right of the root
    mid = np.mean([c.xy for c in CORDS], axis=0)
    head = np.array([mid[0], mid[1], z]) - hr @ np.array([along, 0.0, into])
    return {"rot": hr, "head": head, "exit": head + hr @ np.array([SLOT_C, 0.0, EXIT_Z]),
            "d": hr @ np.array([0.0, 0.0, -1.0]), "hand_deg": deg}
HANG_CORDS_RIGHT = np.array([0.379, -0.009])  # where the strap leaves the cords toward the head (their right side)


def hang_leg(head_xy) -> tuple:
    """From the head (x, y) to where the strap leaves the cords toward it: the heading (deg) and the length."""
    v = HANG_CORDS_RIGHT - np.asarray(head_xy, float)[:2]
    return heading_xy(v), float(np.linalg.norm(v))


def _hang_thread(tr: Trial, b: Build, left: list, right: list, f: dict, o, r1, lr, tail) -> None:
    """Rounds 32-33, on from the left's hold of the tail (hang_ruler ``f``, the left's grip ``lr`` at ``tail``).
    Left: bring the tail in toward the cords, turning it (the strap between bows: strap for the head's way round).
    Right: ease its squeeze, turn its hand to tr.h_carry_yaw, carry the head round the cords' right to the front
    (tr.h_head_at: the strap still slack), then (round 33, tr.h_tauten) draw the head out along the strap's line from
    the cords until it is taut: the pull turns the head between its fingers to trail the strap (job 118: to 8
    degrees of that line, then, pulled on, out of its fingers: the closed loop stops it); ease back; turn the hand
    tr.h_turn_head toward the wall: the head turns with it (job 118), its slot onto the tail's line. Left: turn the
    tail on to tr.h_tail_deg (as far as its wrist goes), line the tip up on the entry, push tr.h_out through along
    the tail's own line (the slot has 26 degrees of play across the strap's thickness). Right: let go of the head,
    take the tail below the exit by its edges, and once the left has let go bring it out to tr.h_lane and draw it
    toward itself. Both let go."""
    up = np.array([0.0, 0.0, 1.0])
    z_low, z = TOP + tr.h_low, TOP + tr.h_z
    light = tr.h_carry_fingers
    rs = rot_z(tr.h_swing) @ r1
    rc = rot_z(tr.h_carry_yaw) @ r1
    head0 = np.asarray(right[-1][1], float) + rs @ o  # the head's centre as the right holds it now
    take_deg = heading_xy(f["u"])
    g = tr.h_gft

    def r_tcp(head_c, rot):
        return tuple(np.asarray(head_c, float) - rot @ o)

    # the left brings the tail in
    fx, fy, fdeg = tr.h_feed
    lf = left_face(lr, fdeg - take_deg, tr.h_tilt)
    feed = np.array([fx, fy, z_low])
    left.append((tr.h_feed_s, face_at(feed, lf), "closed", lf, "feed"))
    # the right carries the head round
    _wait(right, _seconds(left))
    right += [(0.6, r_tcp(head0, rs), light, rs, "ease"),
              (1.5, r_tcp(head0, rc), light, rc, "square")]
    for k, (x, y) in enumerate(HANG_ROUND):
        right.append((tr.h_round_s, r_tcp((x, y, z_low), rc), light, rc, f"round {k + 1}"))
    front = np.array([tr.h_head_at[0], tr.h_head_at[1], z_low])
    right += [(tr.h_round_s, r_tcp(front, rc), light, rc, "to front"),
              (1.0, r_tcp(front, rc), light, rc, "hold head")]
    rh, head_c = rc, front
    if tr.h_tauten > 0.0:  # round 33
        leg_deg, _ = hang_leg(front)
        away = -np.array([math.cos(math.radians(leg_deg)), math.sin(math.radians(leg_deg)), 0.0])
        far = front + away * tr.h_tauten
        head_c = front + away * tr.h_taut_at  # where the plan expects it taut (the closed loop finds out)
        right += [(tr.h_tauten_s, r_tcp(far, rc), light, rc, "tauten"),
                  (0.8, r_tcp(head_c, rc), light, rc, "ease back")]
        yaw = tr.h_carry_yaw
        if tr.h_regrip_yaw > 0.0:  # round 34: let go of the head (the strap holds it there), turn the hand out, take
            #                        it again: that much turn to bring the slot round with
            yaw = tr.h_carry_yaw + tr.h_regrip_yaw
            rg = rot_z(yaw) @ r1
            right += [(0.6, r_tcp(head_c, rc), "wide", rc, "open head"),
                      (1.8, r_tcp(head_c, rg), "wide", rg, "yaw out"),
                      (0.8, r_tcp(head_c, rg), tr.head_fingers, rg, "close head")]
            light = tr.head_fingers
        rg = rot_z(yaw) @ r1
        rh = rot_z(yaw - tr.h_turn_head) @ r1
        light = tr.h_turn_fingers or light  # round 36: the squeeze for the turn and the threading
        right += [(0.8, r_tcp(head_c, rg), light, rg, "settle 2b"),
                  (2.5, r_tcp(head_c, rh), light, rh, "turn head"),
                  (0.8, r_tcp(head_c, rh), light, rh, "hold head 2")]
    # the left turns the tail on toward the slot, lines its tip up on the entry, pushes it through along its own line
    a = math.radians(tr.h_tail_deg)
    d = np.array([math.cos(a), math.sin(a), 0.0])  # the tail's line; the slot's own is turned onto it (round 33) or
    #                                                lies up to 20 degrees further round (round 32)
    hr = rot_z(tr.h_tail_deg if tr.h_tauten > 0.0 else tr.h_d_deg) @ RY_ROLL @ np.asarray(b.head_rot, float)
    e = head_c + hr @ np.array([SLOT_C, 0.0, ENTRY_Z])
    ex = head_c + hr @ np.array([SLOT_C, 0.0, EXIT_Z])
    la = left_face(lr, tr.h_tail_deg - take_deg, tr.h_tilt)

    def tip_at(tip):
        return face_at(np.asarray(tip, float) - d * g, la)

    _wait(left, _seconds(right) - 1.0)
    pushed = e + d * (HEAD[2] + tr.h_out)
    left += [(2.5, tip_at(e - d * (B_TIP_GAP + B_ABOVE)), "closed", la, "aim"),
             (0.8, tip_at(e - d * (B_TIP_GAP + B_ABOVE)), "closed", la, "settle 3"),
             (0.6, tip_at(e - d * (B_TIP_GAP + B_ABOVE)), "closed", la, "above head"),
             (1.5, tip_at(e - d * B_TIP_GAP), "closed", la, "align"),
             (tr.h_push_s, tip_at(pushed), "closed", la, "push"),
             (1.0, tip_at(pushed), "closed", la, "push hold")]
    # the right lets go of the head and takes the tail below the exit, by its edges
    stub = ex + d * tr.h_stub_at  # along the tail's line: the head will have turned onto it
    if tr.h_held:  # round 40: the left holds the head for the draw
        hf = tr.head_fingers
        end = np.asarray(tip_at(pushed), float)
        _wait(right, _seconds(left))
        right += [(0.8, r_tcp(head_c, rh), "wide", rh, "release head"),
                  (1.5, r_tcp(stub, rh), "wide", rh, "to the stub"),
                  (0.8, r_tcp(stub, rh), "closed", rh, "take stub")]
        _wait(left, _seconds(right))

        def l_tcp(head, rot):
            return np.asarray(head, float) - rot @ HOLD_O

        lh = rot_z(heading_xy(hr[:, 0])) @ r1  # level, pointing the strap's way out of the head
        on = l_tcp(head_c, lh)
        left += [(1.0, tuple(end), "open", la, "let go tail 3"),
                 (0.6, tuple(end + 0.004 * up), "open", la, "back off tail 3"),
                 (2.0, (tr.h_clear[0], tr.h_clear[1], z_low + tr.h_clear[2]), "wide", la, "clear 2"),
                 _ready(2.0),
                 (2.5, tuple(on - 0.045 * lh[:, 2]), "head_open", lh, "to the head 2"),
                 (1.5, tuple(on), "head_open", lh, "onto the head"),
                 (1.0, tuple(on), hf, lh, "take head 2"),
                 (0.6, tuple(on), hf, lh, "hold head 3")]
        _wait(right, _seconds(left))
        rw = rot_z(tr.h_stub_yaw) @ r1
        by = np.array([tr.h_stand_by[0], tr.h_stand_by[1], z_low])
        right += [(0.8, r_tcp(stub, rh), "head_open", rh, "let go stub 1"),
                  (1.2, tuple(np.asarray(r_tcp(stub, rh)) - 0.04 * rh[:, 2]), "head_open", rh, "back off stub 1"),
                  (2.0, r_tcp(by, rw), "wide", rw, "stand by")]
        tight = hang_tight(tr, b, z_low)
        lt = rot_z(tight["hand_deg"]) @ r1
        _wait(left, _seconds(right) - 2.0)  # once the right has backed off
        left += [(tr.h_carry_s, tuple(l_tcp(tight["head"], lt)), hf, lt, "carry head"),
                 (0.6, tuple(l_tcp(tight["head"], lt)), hf, lt, "set head"),      # the head's real heading is read,
                 (1.5, tuple(l_tcp(tight["head"], lt)), hf, lt, "square head"),   # and the wrist turns it onto its own
                 (0.8, tuple(l_tcp(tight["head"], lt)), hf, lt, "at the cords")]
        stub2 = tight["exit"] + tight["d"] * tr.h_stub_at
        drawn = stub2 + tight["d"] * tr.h_pull
        _wait(right, _seconds(left) - 0.8)
        rp = rot_z(tr.h_pull_yaw) @ r1
        right += [(2.0, r_tcp(stub2, rw), "wide", rw, "to the stub 2"),
                  (0.8, r_tcp(stub2, rw), "closed", rw, "take stub 2"),
                  (tr.h_pull_s, r_tcp(drawn, rp), "closed", rp, "pull"),
                  (1.5, r_tcp(drawn - tight["d"] * HOLD_EASE, rp), "closed", rp, "pull hold"),  # eased back
                  (0.8, r_tcp(drawn - tight["d"] * HOLD_EASE, rp), "head_open", rp, "let go stub"),
                  (1.2, tuple(np.asarray(r_tcp(drawn - tight["d"] * HOLD_EASE, rp)) + 0.04 * up), "head_open", rp,
                   "back off stub"),
                  _ready(2.0)]
        _wait(left, _seconds(right) - 3.2)  # the left lets go as the right backs off
        at = l_tcp(tight["head"], lt)
        left += [(0.8, tuple(at), "head_open", lt, "let go head 2"),
                 (1.5, tuple(at - 0.045 * lt[:, 2]), "head_open", lt, "back off head 2"),
                 _ready(2.0)]
        _wait(right, _seconds(left))
        _wait(left, _seconds(right))
        return
    _wait(right, _seconds(left))
    right += [(0.8, r_tcp(head_c, rh), "wide", rh, "release head"),  # 2.5 mm a side: wider, its upper finger
              (1.5, r_tcp(stub, rh), "wide", rh, "to the stub"),     # nears the left's
              (0.8, r_tcp(stub, rh), "closed", rh, "take stub")]
    _wait(left, _seconds(right))
    end = np.asarray(tip_at(pushed), float)
    left += [(1.0, tuple(end), "open", la, "let go tail 3"),  # wide open: its fingers clear the strap as it leaves
             (0.6, tuple(end + 0.004 * up), "open", la, "back off tail 3"),  # (back along the strap, its forearm
             #                                                                 nears the cords; higher is out of reach)
             (2.0, (tr.h_clear[0], tr.h_clear[1], z_low + tr.h_clear[2]), "wide", la, "clear 2"),  # out from between the cords
             _ready(2.0)]                                                                  # and the right's arm
    _wait(right, _seconds(left) - 2.0)
    out = np.array([tr.h_lane, tr.h_pull_from, z])
    drawn = out + np.array([0.0, -tr.h_pull, 0.0])
    right += [(tr.h_out_s, r_tcp(out, r1), "closed", r1, "pull out"),
              (tr.h_pull_s, r_tcp(drawn, r1), "closed", r1, "pull"),
              (1.0, r_tcp(drawn, r1), "closed", r1, "pull hold"),
              (0.8, r_tcp(drawn, r1), "head_open", r1, "let go stub"),
              (1.2, tuple(np.asarray(r_tcp(drawn, r1)) + 0.04 * up), "head_open", r1, "back off stub"),
              _ready(2.0)]
    _wait(left, _seconds(right))


def hang_score(polyline, head_pos, head_quat, hand_clear: float, speed: float, passage_y: float = PASSAGE_Y) -> float:
    """The hanging cords without the stand: as lift_score (hands 3 cm clear, the tie at rest, closed round all three
    cords; then the marks), the cords the fixed ones (threading)."""
    if not (hand_clear >= CLEAR and speed < STILL):
        return 0.0
    th = threading(polyline, head_pos, head_quat, passage_y)
    if th.through < TAIL_MIN or not th.entered or len(th.enclosed) != len(CORDS):
        return 0.0
    full, half = lift_marks(arc_length(polyline))
    return 1.0 if th.through >= full else 0.5 if th.through >= half else 0.25


def trial_plan(tr: Trial) -> dict:
    left, right = [], []
    b = BUILDS[tr.build]
    if tr.stage == "hang":
        hp = _hang_plan(tr, b)
        left, right = hp["left"], hp["right"]
    elif tr.stage == "thread":
        hp = _thread_plan(tr, b)
        left, right = hp["left"], hp["right"]
    elif tr.stage:
        hp = _lift_plan(tr, b)
        left, right = hp["left"], hp["right"]
    elif b.fixed and tr.handover:
        hp = _wrap_plan(tr, b) if b.name == "wrap" else _handover_plan(tr, b)
        left, right = hp["left"], hp["right"]
    elif b.fixed:
        f = jig_frame(b)
        if tr.insert:
            left = _insert_steps(tr, b)
        insert_done = next((sum(s[0] for s in left[: i + 1]) for i, s in enumerate(left) if s[4] == "back off"),
                           0.0)
        if tr.pull:
            _wait(right, insert_done + 0.3)
            right += _pull_steps(tr, f)
    elif tr.pick:
        head = np.asarray(b.head_pos)
        across = b.head_rot[:, 1]  # y_h: the head's 9 mm width, fingers close across it
        yaw = math.degrees(math.atan2(across[0], -across[1]))
        rot = rot_down(yaw)
        back = -b.head_rot[:, 0] * 2.0 * MM  # TCP 2 mm toward the head's back, off the strap's root
        at = head + back + np.array([0.0, 0.0, -HEAD[2] / 2 + 0.8 * MM + PAD_PAST_TCP])  # tips 1.1 mm up: more pad on it
        up = at + [0, 0, LIFT]
        hf = tr.head_fingers
        right = [(2.5, tuple(at + [0, 0, 0.05]), "wide", rot, "to the head"),
                 (1.5, tuple(at), "wide", rot, "descend"),
                 (1.0, tuple(at), hf, rot, "grip"),
                 (2.0, tuple(up), hf, rot, "lift"),
                 (3.0, tuple(up), hf, rot, "hold up")]
        if tr.tug:  # the left hand takes the strap TUG_AT out (down grip, fingers on its edges) and pulls along it
            x_h = b.head_rot[:, 0]
            strap = head + [0, 0, LIFT] + b.head_rot @ (ROOT + np.array([TUG_AT, 0.0, 0.0]))
            closing = b.head_rot[:, 1]
            lrot = rot_down(math.degrees(math.atan2(closing[0], -closing[1])))
            lat = strap + np.array([0.0, 0.0, -2.0 * MM])  # pads reach 40 mm up and 12.8 down: droop fits
            left = [(sum(s[0] for s in right) - 0.5, None, "open", "ready", "wait for the lift"),
                    (2.0, tuple(lat + [0, 0, 0.02]), "wide", lrot, "to the strap"),
                    (1.2, tuple(lat), "wide", lrot, "in"),
                    (0.8, tuple(lat), "closed", lrot, "grip"),
                    (3.0, tuple(lat + x_h * TUG), "closed", lrot, "tug"),
                    (1.0, tuple(lat + x_h * TUG), "closed", lrot, "tug hold"),
                    (0.6, tuple(lat + x_h * TUG), "wide", lrot, "let go"),
                    (1.2, tuple(lat + x_h * TUG + [0, 0, 0.04]), "wide", lrot, "back off")]
            _wait(right, sum(s[0] for s in left))
    left = left or [_ready(1.0, label="stay")]
    right = right or [_ready(1.0, label="stay")]
    plan = _pad({"left": left, "right": right})
    for arm in plan:  # then everyone stays clear while the tie is watched
        last = plan[arm][-1]
        plan[arm].append((tr.watch_s, last[1], last[2], last[3], "watch"))
    return plan


KEYS_START = "k25-start"  # the task's demo() until a person's run is recorded: the lying-cords start's first half


def _t(name, build, why, **kw) -> Trial:
    return Trial(name, build, why, **kw)


# Round 1 (b5e1751-24): the tilted grip from above hit the TV; one edge-grip insert threaded 5.7 mm.
# Round 2 (bfe841b-27): edge grips can't turn the strap; "chatter" was velocity noise.
# Round 3 (5502a16-29): fed along a straight run, 5/7 threaded; one ended closed (0.25); the tail slid back out.
# Round 4 (9a363c3, job 32): 5/6 threaded, but the strap sprang back out of the slot every time the hand opened to
# regrip (1 mm out can't be held), the brake left on after it left blocked the next push, and the pulls found nothing.
# Round 5 (30f589b, job 34): the hand-over held; h-heavy pulled the tail from 21 to 67 mm out, then jammed when the
# D-loop's kink reached the slot. 3 of 5 pushes blew up on the sprung pawl. Best: 0.25 (cam).
# Round 6 (6c41635, job 38): the latch got every tip in (5 of 5, 21-23 mm out). l-heavier tied snug: 1.0, the first.
# Rounds 7-10 (jobs 45, 49, 53, 56): the pre-looped start 1.0 every time (c-heavier; also with the re-bite and with the
# guide). The wrap: the swing and a 12 mm steering shift get the tail in, but every pull jams 34-57 mm out, whatever the
# grip (re-bite), a loose guiding hand or a softer strap. Round 10's shape logs: the wrap's tail leaves the exit angled
# toward the robot, pressing the strap to the passage's far wall, and the pawl stands at 54-55 degrees on it.
# Round 11 (1016c3a, job 60): THE WRAP TIED SNUG, 1.0 (k-latch-aim: pawl clear until 100 mm out, the pull aimed 6 mm
# toward the pawl's side; drawn 121 mm, 5.0 mm slack, the pawl at 30 degrees at the end, tied at 29.75 s). Either fix
# alone: 0.25 (stroke 2 slipped 62-103 mm). The jig with the pawl clear to 100 mm went unstable (c-latch, 0.0); the
# jig's 1.0 held (c-heavier).
# Round 12 (12dd75d, job 62): THE WRAP's 1.0 DID NOT REPEAT. k-la (round 11's winner, unchanged) 0.25; every margin
# variant 0.25 or 0 (l45, a4: the tip never got through). The push leaves 11-17 mm out (the strap slides 3-10 mm in the
# left's fingers), the right holds the tail by its last link, and the pull stops at 44, 51, 57 or 63 mm out, one link
# (6.25 mm) apart: a joint catches in the head, the tip slips out of the fingers, the pawl (held clear) lets it slide
# back. The jig's c-heavier: 1.0 again (seventh run in a row); c-la unstable again.
# Round 13 (df5c2b5, job 65): with the clip stand spawned as colliders, the pre-looped start's fresh trial (c-heavier,
# first in its launch, identical to the 0.1 mm in jobs 53-62) scored 0.25: the pawl stood at 52-54 degrees as the pull
# began and wedged; its second trial scored 1.0. The colliders' 5 mm contact skin, 0.3 mm off the head, is enough to
# send the contacts down another path. The wrap with the clip: 0.25, 0.25, 0.0. B1: the gripper holds the head (slip
# 0.0-0.3 mm in all three); the firm squeeze shakes it (head 56-68 mm/s), the light one holds it still (7 mm/s); under
# the 25 mm tug the holding arm gives 10 mm.
# Round 14 (9149eb8, job 67): the clip stand drawn only (no colliders). c-heavier fresh: 1.0 again, identical to the
# 0.1 mm to jobs 53-62 (the physics is back). c-deep 1.0; c-heavier-2 0.5 (the last stroke, 30 mm past snug, slid the
# pads off the 6 mm of tail past them; the strap snapped back 7 mm before the brake held); c-fast, c-slow 0.25 (pull 1
# stalled 3 mm in, 25.6 mm out). Shape logs: at each stall a strap joint sits on the slot's entry plane with the strap
# against the far wall (x_h -2.0 mm, pawl 54 degrees): the loop comes in from the far-wall side, the hand pulls straight
# down the slot, so the strap bends round the far entry edge and a link corner catches there (the wrap's 44/51/57/63 mm
# stalls are the same catch, one link apart). In c-heavier the strap came off that wall as the pull began (pawl 50 -> 31).
# Round 15 (a13325d, job 69): c-demo, fresh (c-heavier played by task.demo(), its closed loop in the task's step hook,
# what a reviewer's yam_scripted run plays): 1.0, identical to c-heavier to the 0.1 mm. Aimed 6 mm, no pull-1 stall in 4
# (aimed 3 mm: 1 of 4; unaimed: 1 of 3). But only c-demo and e-slow scored 1.0: the eased runs were let go with the
# brake off and the strap ran back 5-10 mm inside one control step (0.5); a6-slow and a6-deep slipped 6 mm in a stall at
# 57 mm and their 6 mm past snug ended 132 mm out (0.5).
# Round 16 (945ecbe, job 71): the "catch" ratchet, aimed 6 mm, 6 mm past snug, then tightened to 5.5 mm short of snug:
# 0 of 12 at 1.0. Every tightened run got to 137.3 mm out and, let go, fell a link to 131.3 (0.5) or 127.1 (k-slow,
# k-slow-deep), catch or latch alike (k-latch); k-loose 134.0 -> 126.2. k-deep, k-a9, k-c slipped 46-102 mm in the pulls
# under the catch (0.25); k-ang15 lost the strap in the push (0.0).
# Round 17: the latch again, the tighten past the joint (to 138.3 mm out, 12 mm/s, at most 24 mm on: 30 mm past snug in
# all, c-heavier's), held 1.5 s, let go. m-demo first in its launch (fresh).
# Round 17 (05e9f94, job 73): 7 of 12 at 1.0, m-demo fresh among them (drawn to 138.1, let go at 137.5, ended there:
# 5.1 mm slack). Every run that drew past the joint kept it (no backslide). m-deep, m-slow-deep, m-s3 and m-ang15
# stalled at once, 25.6 mm out, and the pads slid off the tip (0.25); m-a9 stalled at 57 mm and again (0.25). m-c (the
# old candidate, not fresh) 1.0.
# Round 18: m-demo is the candidate (task.TRIAL). The launch replays round 17's order with the unstick on every trial
# after the first. Those that didn't stall should repeat round 17 to the 0.1 mm (it never acts); u-deep, fourth, meets
# m-deep's stall.
WRAP = dict(steer=True, phys="heavier", handover=True, ratchet="latch", depth_cap=4.0 * MM, steer_cap=12.0 * MM)
LA = dict(WRAP, latch_mm=100.0 * MM, pull_aim_x=6.0 * MM)
JIG = dict(steer=True, phys="heavier", handover=True, ratchet="latch", depth_cap=4.0 * MM)
HELD = dict(insert=None, pull=None, pick=True, phys="heavier", head_fingers="head_light")
JIG15 = dict(JIG, extra_pull=6.0 * MM, ease=6.0 * MM)
A3 = dict(JIG15, pull_aim_x=3.0 * MM)
A6 = dict(JIG15, pull_aim_x=6.0 * MM)
FAST, SLOW, DEEP = dict(insert_s=2.3), dict(insert_s=2.7), dict(grip_depth=GRIP_DEPTH + 0.4 * MM)
M17 = dict(JIG, demo=True, extra_pull=6.0 * MM, pull_aim_x=6.0 * MM, tighten=24.0 * MM, hold_s=1.5)
CANDIDATE = _t("m-demo", "jig", "THE SUBMISSION CANDIDATE as task.demo() plays it: the latch, the pull aimed 6 mm, 6 mm "
               "past snug, then tightened closed-loop past the last joint (138.3 mm out); fresh 1.0 in job 73", **M17)
U = dict(M17, unstick=True)
R18 = (
    CANDIDATE,
    _t("u-fast", "jig", "the push 0.2 s quicker; unstick", **dict(U, **FAST)),
    _t("u-slow", "jig", "the push 0.2 s slower; unstick", **dict(U, **SLOW)),
    _t("u-deep", "jig", "the strap 0.4 mm deeper in the pushing hand's pads; unstick (m-deep stalled here)",
       **dict(U, **DEEP)),
    _t("u-fast-deep", "jig", "quicker and deeper; unstick", **dict(U, **FAST, **DEEP)),
    _t("u-slow-deep", "jig", "slower and deeper; unstick", **dict(U, **SLOW, **DEEP)),
    _t("u-a3", "jig", "the pull aimed 3 mm; unstick", **dict(U, pull_aim_x=3.0 * MM)),
    _t("u-a9", "jig", "the pull aimed 9 mm; unstick", **dict(U, pull_aim_x=9.0 * MM)),
    _t("u-s3", "jig", "tightened to 3 mm short of snug; unstick", **dict(U, tighten_slack=3.0 * MM)),
    _t("u-x0", "jig", "the last stroke to snug, the tighten the rest (at most 30 mm); unstick",
       **dict(U, extra_pull=0.0, tighten=30.0 * MM)),
    _t("u-ang15", "jig", "the pull turned 15 degrees toward the robot, not 20; unstick", **dict(U, pull_angle=15.0)),
    _t("u-c", "jig", "the old candidate's pull (unaimed, 30 mm past snug, no tighten) with the unstick",
       **dict(JIG, demo=True, unstick=True)),
)
# Round 19: the lift start, stage A (the user: "one arm lifts the cord, the other slides the tie under it"). New in the
# sim: the cords as jointed cords on the console (Harness), and a strap stiff for its weight (stiff5, stiff15). Each
# launch: does the bundle lift and settle, does a tie held by its head stand out, does its strap go under.
A19 = dict(insert=None, pull=None, ratchet="latch", steer=False, depth=False, watch_s=2.0)
A19_TRIALS = (
    _t("a-lift", "lift", "the left pinches the bundle and lifts it 60 mm, holds, lowers it, lets go (heavier tie)",
       **dict(A19, stage="lift", phys="heavier")),
    _t("a-slide", "lift", "lift, and the right slides the tie (held by its head) under: today's strap, a control",
       **dict(A19, stage="slide", phys="heavier")),
    _t("s5-hold", "lift", "the right holds the tie up by its head, strap toward the wall: strap 5x stiffer",
       **dict(A19, stage="hold", phys="stiff5")),
    _t("s5-slide", "lift", "lift and slide under, strap 5x stiffer", **dict(A19, stage="slide", phys="stiff5")),
    _t("s15-hold", "lift", "held up by the head, strap 15x stiffer (stiffness and weight as real nylon's ratio)",
       **dict(A19, stage="hold", phys="stiff15")),
    _t("s15-slide", "lift", "lift and slide under, strap 15x stiffer", **dict(A19, stage="slide", phys="stiff15")),
)
# Round 19 (job 84): stage A works in all three physics, today's heavier strap included: the cords lift 55 mm and stay
# at least 48.8 mm up while the strap passes; it ends under all three, its tip 35-36 mm past; the strip slides 8.5 mm.
# Round 20: stage B (_thread_plan). demo=True: the probe plays task.demo_plan(), whose ThreadSteering is the closed loop.
# Round 20 (job 95): stage B fell at the change of grips in all three: the left's down grip on the strap and the right's
# on the head put the forearms 3 mm apart, the left pushed the right 57 mm and the tie fell. Round 21: the left holds the
# strap 50 mm out, pointing at the head 45 degrees down (Trial.b_hold_*), the snaps are capped (B_SNAP), and
# armcheck.arms_clearance checks arm against arm.
# Round 21 (job 96): the grips changed in all three, the arms apart; the strap went under the cords on the heavier tie,
# but the right's grip from behind never held the head: its fingers closed to their 1 mm, past it. Heavier: the head
# rose 7.7 mm on the left's strap, the snap's 6 mm cap left the hand under it; one repeat closed on it (1.7 mm high)
# and lost it when the left let go as the carry began. Stiff15: the hand lagged 4 mm sideways coming in, a fingertip
# pushed the head 10 mm (the snap, still live, followed it) and it fell at once. The heavier ones were carried through
# the slide resting on the closed fingers and fell when the head was raised. The tail then lined up on the planned
# slot, 80-85 mm above the fallen head. Round 22: the head snap reads the head before any finger can touch it and up to
# 12 mm high; the fingers open 24 mm round it; the right waits for the left to be off the strap; the raise takes 2.5 s.
B20 = dict(A19, stage="thread", demo=True, watch_s=3.0)
B23_TRIALS = (
    _t("b-thread", "lift", "stage B: change grips, lift, slide, level the head, take the tail, over, thread it down "
       "(heavier tie)", **dict(B20, phys="heavier")),
    _t("b-thread-r", "lift", "the same again in the same launch (not fresh): how repeatable", **dict(B20, phys="heavier")),
    _t("b15-thread", "lift", "the same with the strap 15x stiffer (stiffness and weight as real nylon's ratio)",
       **dict(B20, phys="stiff15")),
)
# Round 23 (job 98): the grip from behind held the head through every step in all three (it moved 1.5 mm in the
# fingers) and the tail dragged the cords 1-3 mm (job 97: 14-54), but no tail went in: the strap folded where it left
# the left's pads as the wrist turned (LIFT_BUILDS above), and on stiff15 it sprang out of them. Round 24: a 300 mm and
# a 250 mm tie, the slide first, and the tail brought over the bundle high so the strap trails above the fingers.
# The grips change at y -0.050 (at -0.010 the forearms came within 2-3 mm, the closed loop's snaps at their caps,
# and 1.8 mm while the left lowered the cords), the left's strap hold 60 mm out pointing 30 degrees down (at 45 its
# wrist came 17 mm from the power cord hanging by the wall); over the bundle 110 mm up (at 130 the wrist came within
# 15 mm of the TV).
B24 = dict(B20, b_slide_first=True, b_regrasp=(0.330, -0.050, TOP + 0.048), b_hold_s=0.060, b_hold_phi=30.0,
           b_lift=(-0.120, 0.080), b_over=(0.330, 0.105, TOP + 0.110))
R24_TRIALS = (
    _t("l30-thread", "lift30", "stage B on a 300 mm tie: lift and slide (down grip), change grips clear of the cords, "
       "carry the head to the bundle, take the tail, over it high, thread it down (heavier tie)",
       **dict(B24, phys="heavier")),
    _t("l30-thread-r", "lift30", "the same again in the same launch (not fresh): how repeatable",
       **dict(B24, phys="heavier")),
    _t("l25-thread", "lift25", "the same on a 250 mm tie", **dict(B24, phys="heavier")),
)
# Round 24 (job 101): the start held in all three (strap under, head slip 0.1 mm) but no tail went in. The tail turned
# 45-50 degrees tip-up in the left's pinch as it was lifted (the strap behind, hanging to the console, levers it) and
# from then on followed the wrist to within 2 degrees through 80 degrees of turning, so it lined up 16-19 mm off the
# slot's line, about 35 degrees down where 80 was planned. Round 25: the wrist tilts as the tail rises (b_lift_phi), so
# the pinch holds the tail at the angle the strap behind takes anyway and the later turns start from no offset. A tilt
# as steep as job 101's 48 degrees is out of the left's reach there, and 25 degrees or more brought its forearm within
# 11-15 mm of the cords hanging by the wall: so the lift is shorter, 40 mm toward the bundle and 70 mm up, where the
# strap behind runs about 26 degrees down to the bundle (job 101's lift: 57 degrees by the chord, 45-50 measured), and
# the wrist tilts 25 degrees about the strap in its pads (forearm 23 mm from the TV; 28 came within 18); a second
# trial tilts it 15, to see whether the tail follows the wrist or the strap.
B25 = dict(B24, b_lift=(-0.040, 0.070), b_lift_phi=115.0)
# Round 26 (the cinch, above): the head waits level 30 mm from the bundle and 33 mm up; the tail is taken 76 mm from
# its tip, lifted 45 mm, carried on the eased pinch up over the bundle (115 mm up) and down to 28 mm past it, 57 mm up;
# the wrist turns it on, the head turns to pitch 73 (root down, entry toward the bundle); one push to 34 mm out;
# hand-over; pull along the slot until the full mark is 4 mm out.
B26 = dict(B24, b_cinch=True, b_gft=0.076, b_head=(0.330, 0.060, TOP + 0.033), b_insert_pitch=90.0 - B_AIM,
           b_lift=(-0.030, 0.045), b_over=(0.330, 0.150, TOP + 0.100), watch_s=4.0, latch_mm=LIFT_LATCH_AT)
W29 = dict(B24, b_wedge=True, b_gft=0.046, b_out=0.013, b_over=(0.330, 0.130, TOP + 0.100), b_arch=0.055,
           b_lift=(-0.020, 0.050), b_lift_s=2.0, watch_s=3.0, latch_mm=LIFT_LATCH_AT, phys="heavier")
#   the arch and the lift leave about 50 mm of strap to spare (strap_slack): lifted 70 mm straight up from a 35 mm
#   arch there are 9-15, and a strap pulled straight is the tip-up lock again
TRIALS = (  # round 29: lock the tail tip-down in the pinch, then thread it down as rounds 23-25 planned
    _t("w25-arch", "lift25", "the wedge on the 250 mm tie: pinch the tail 46 mm from its tip, push it 55 mm toward the "
       "bundle along the console (the strap behind arches up), lift 50 mm leaning 20 mm further in, read the lock, "
       "wrist set from it, over the bundle, read again, down onto the slot, push 13 mm through (latch at 8), let go",
       **W29),
    _t("w30-arch", "lift30", "the same on the 300 mm tie (50 mm more strap in the loop)", **W29),
    _t("w30-deep", "lift30", "the same with a deeper arch (75 mm, the lift leaning 30 mm in: about 75 mm to spare; "
       "not fresh)", **dict(W29, b_arch=0.075, b_lift=(-0.030, 0.050))),
)
W30 = dict(W29, b_draw=True, b_gft=0.068, b_out=CINCH_OUT, b_push_s=4.0, b_head=(0.330, 0.070, TOP + 0.045),
           b_insert_pitch=15.0, b_over=(0.330, 0.130, TOP + 0.115), b_lift=(-0.020, 0.065), b_pull=0.270, b_pull_s=9.0,
           b_tail_tilt=40.0, watch_s=3.0)
#   34 mm of tail out for the right's pads (27 mm across, below the head): the tail taken 68 mm from its tip, the head
#   held 45 mm up so the tail hangs clear of the console; the 300 mm tie for the strap that takes. The head pitched
#   15 degrees toward the left (rounds 23-29: -10, as near as the right's wrist came to a tip-up tail): the wrist then
#   threads tilted about 31 instead of 56, its forearm 20 mm and more off the TV and its fingers off the right's
D30 = (  # round 30: round 29's threading, pushed 34 mm through, then the draw
    _t("w30-draw", "lift30", "thread as w30-arch but 34 mm through (tail taken 68 mm from its tip, head 45 mm up); the "
       "right lets go of the head, takes the tail below it and draws it 270 mm toward itself (stopping once the full "
       "mark is 4 mm out); both let go", **W30),
    _t("w30-draw-pin", "lift30", "the same with the left pinning the bundle (where it pinched it to lift it) during "
       "the draw", **dict(W30, b_hold_cords=True)),
)
TRIALS = TRIALS + D30
H31 = dict(A19, stage="hang", demo=True, phys="heavier", latch_mm=LIFT_LATCH_AT, watch_s=2.0)
HANG_TRIALS = (  # round 31: the first steps on the hanging cords
    _t("h30-take", "hang30", "the hanging cords, no stand: both hands lift the 300 mm tie and turn it on edge (the "
       "strap a ruler); the right passes it behind the cords and swings its tail to the front; the left takes the "
       "tail by its faces; the right lets go and the left holds the tie alone 3 s", **H31),
    _t("h30-take-r", "hang30", "the same again in the same launch (not fresh): how repeatable", **H31),
    _t("h30-tie", "hang30", "the whole tie on the hanging cords: as h30-take, then the head round the cords to the "
       "front, the left aims the tail down the slot and pushes 34 mm through, the right takes the tail below the "
       "head and draws it out; both let go", **dict(H31, h_thread=True, watch_s=3.0)),
    _t("h30-tie-firm", "hang30", "the same with the right's full squeeze on the head all the way round (does the head "
       "still turn to trail the strap, or the strap kink at its root?)",
       **dict(H31, h_thread=True, watch_s=3.0, h_carry_fingers="head")),
)
TRIALS = TRIALS + HANG_TRIALS
H33 = dict(H31, h_thread=True, watch_s=3.0, h_low=0.140, h_carry_yaw=58.0, h_tauten=0.050, h_taut_at=0.0,
           h_turn_head=25.0, h_head_at=(0.270, 0.026), h_feed=(0.300, 0.100, 205.0), h_push_s=6.0,
           h_pull_from=-0.200, h_out_s=6.0, h_pull=0.200, h_pull_s=8.0)
#   round 35 (job 119: taut with the head 134 mm from the cords): the head to the front 115 mm from them, slack; out
#   along the strap's line until taut (about 19 mm on); back the same (the closed loop: from where the head really is)
#   on the 370 mm tie, everything at 140 mm (the right's hand reaches every turn there, the left's as at 90): the head
#   to the front with 100 mm of strap from the cords, slack; out along that line until taut (expected near 127 mm);
#   the hand, still turned 58 degrees as it swung the ruler, turns back up to 58 to bring the slot onto the tail's line
HANG33_TRIALS = (
    _t("h37-tie", "hang37", "the whole tie on the hanging cords with a 370 mm tie: as h30-tie, but the right keeps its "
       "hand turned while it carries the head round, draws the strap taut at the front (the head turns to trail "
       "it), then turns its hand back until the slot lies along the tail's line; thread, hand over, draw", **H33),
    _t("h37-tie-r", "hang37", "the same again in the same launch (not fresh): how repeatable", **H33),
)
H34 = dict(H33, h_carry_yaw=0.0, h_regrip_yaw=45.0, h_turn_head=25.0)
HANG34_TRIALS = (
    _t("h37-regrip", "hang37", "as h37-tie, but the right carries the head round with its hand square to the wall "
       "(as round 32 did, which got it to the front), draws the strap taut, then lets go of the head, turns its "
       "hand 45 degrees out, takes the head again and turns back until the slot lies along the tail's line", **H34),
)
HANG36_TRIALS = (
    _t("h37-firm", "hang37", "as h37-tie with the right's full squeeze on the head from the turn back on (the head "
       "turns with the hand, and stays put while the tail is pushed through)", **dict(H33, h_turn_fingers="head")),
)
HANG38_TRIALS = (
    _t("h37-slick3", "hang37", "h37-tie with the tie's friction 0.3 on everything it touches (nylon; 0.9 so far): "
       "does the tail draw now, with nothing holding the head, and do the grips before it still hold", **dict(H33, phys="slick3")),
    _t("h37-slick5", "hang37", "the same at friction 0.5", **dict(H33, phys="slick5")),
    _t("h37-nylon", "hang37", "h37-tie with friction by contact: the strap on the pads as before (0.9), in the slot "
       "and on the pawl 0.25, on the cords 0.35", **dict(H33, phys="nylon")),
)
H40 = dict(H33, h_held=True, h_out=0.034, h_stub_at=0.024, h_pull=0.290, h_pull_s=10.0)  # 34 mm pushed through (24):
#   room on the stub between the right's pads and the left's on the head
HANG40_TRIALS = (
    _t("h37-hold", "hang37", "the whole tie with the left holding the head for the draw: the right takes the tail's "
       "stub, the left lets go of the strap and takes the head from its free face, the right lets go, the left "
       "carries the head to the cords, the right takes the stub again and draws it along the slot's line until the "
       "full mark is out; both let go", **H40),
    _t("h37-hold-n", "hang37", "the same with friction by contact (slot and pawl 0.25, cords 0.35)",
       **dict(H40, phys="nylon")),
    _t("h37-hold-max", "hang37", "h37-hold drawn until no more tail comes: how tight the loop gets on the cords",
       **dict(H40, h_pull=0.310, h_pull_to=0.340, h_pull_s=20.0)),
    _t("h37-hold-n-max", "hang37", "the same with friction by contact", **dict(H40, phys="nylon", h_pull=0.310, h_pull_to=0.340, h_pull_s=20.0)),
    _t("h37-hold-l", "hang37", "THE DEMO (round 46): the whole tie on the task's physics (friction by contact, the "
       "lead-in on the slot's mouth), the left holding the head for the draw, the draw stopped once the full mark "
       "is 12 mm out and eased before the right lets go", **dict(H40, phys="nylonlead", h_pull=0.320, h_pull_s=21.0,
                                                                watch_s=8.0)),  # job 132: 290 mm of draw ended at
    #   301.5 mm of tail (the hand runs 20 to 30 mm ahead of the tail it draws); and the run ended while the tie,
    #   let go, was still sliding down the cords onto the power strip
    _t("h37-hold-l-max", "hang37", "the same with a 0.5 mm lead-in on the slot's mouth as well",
       **dict(H40, phys="nylonlead", h_pull=0.320, h_pull_to=0.340, h_pull_s=26.0)),
    _t("h37-hold-b-max", "hang37", "the same with the lead-in on the slot's back wall only",
       **dict(H40, phys="nylonback", h_pull=0.310, h_pull_to=0.340, h_pull_s=20.0)),
)
TRIALS = TRIALS + HANG33_TRIALS + HANG34_TRIALS + HANG36_TRIALS + HANG38_TRIALS + HANG40_TRIALS
C26 = (  # round 28: the scripted finish with the tail carried over on an eased pinch (the pinch dropped it)
    _t("c25-soft", "lift25", "the cinch on the 250 mm tie: lift, slide, change grips, the head pitched to face the "
       "bundle, the tail carried over on an eased pinch (about 2 N a side), wrist and head turned to line up (closed "
       "loop), one push, the right takes the tail and the left the head, pull along the slot", **dict(B26, phys="heavier")),
    _t("c25-soft3", "lift25", "the same with the pinch eased less (about 4 N a side; not fresh)",
       **dict(B26, b_soft="soft3", phys="heavier")),
    _t("c25-soft5", "lift25", "the same with the pinch eased more (about 1 N a side; not fresh)",
       **dict(B26, b_soft="soft5", phys="heavier")),
)
K26 = (_t(KEYS_START, "lift25", "the start a person finishes on the keyboard: the left lifts the cords, the right "
          "slides the 250 mm tie under them, the grips change, and the right holds the head by the bundle (round 24's "
          "plan up to there, as jobs 98 and 101 played it); the left takes the tail 70 mm from its tip and lifts it "
          "50 mm, moving 50 mm toward the bundle so the strap behind stays slack",
          **dict(B24, b_keys=True, b_gft=0.070, b_lift=(-0.050, 0.050), b_lift_s=2.0, phys="heavier", watch_s=0.2,
                 latch_mm=LIFT_LATCH_AT)),)  # the probe sets each trial's latch: the task's own
R25_TRIALS = (
    _t("l25-tilt", "lift25", "round 24's l25-thread with a shorter lift of the tail (40 mm toward the bundle, 70 up) and "
       "the left's wrist tilted 25 degrees as it rises, along the strap behind the pinch, so the tail points down along "
       "the slot when the wrist turns to thread", **dict(B25, phys="heavier")),
    _t("l25-tilt105", "lift25", "the same with the wrist tilted 15 degrees (not fresh): does the tail follow the wrist or "
       "the strap behind", **dict(B25, b_lift_phi=105.0, phys="heavier")),
    _t("l30-tilt", "lift30", "l25-tilt on the 300 mm tie", **dict(B25, phys="heavier")),
)
# Round 18 ran late (job 78, 8/12, m-demo 1.0 again, the demo through inspect-robots 1.0) and round 19 (4df53b9) carried
# its sweep again by mistake (the same launch: the same results). The unstick ended every early stall (stroke-1 slip
# 54-58 mm -> under 2 mm); what's left is a short first stroke that the second can't make up (rest at 131 or 128 mm).
# Kept by name whether or not in this round's sweep: the submission candidate (task.TRIAL plays it), the earlier
# candidate (demo.py's offline checks use its push) and round 15's check that task.demo() plays it as the probe did.
KEPT = (CANDIDATE,
        _t("c-heavier", "jig", "the earlier candidate: the pre-looped start (fresh 1.0 in jobs 53-69)", **JIG),
        _t("c-demo", "jig", "the candidate as task.demo() plays it (its own closed loop, what yam_scripted runs); "
           "fresh in job 69: c-heavier's 1.0 to the 0.1 mm", **dict(JIG, demo=True)),
        *R18[1:],  # round 18's unstick trials: the tests still check the unstick on them
        *A19_TRIALS,  # round 19's stage A: the tests still check its plans
        *B23_TRIALS,  # round 23's stage B on the 200 mm tie: likewise
        *R24_TRIALS,  # round 24's untilted lift of the tail
        *R25_TRIALS,  # round 25's tilted lift
        *K26,  # the keys' start (task.demo() plays it)
        *C26)  # round 28's eased-pinch carry
TRIAL_NAMES = tuple(t.name for t in TRIALS)
CHECKED = TRIALS + tuple(t for t in KEPT if t not in TRIALS)  # every trial the offline checks cover


def trial(name: str) -> Trial:
    for t in TRIALS + KEPT:
        if t.name == name:
            return t
    raise KeyError(f"no trial {name!r}; known: {', '.join(TRIAL_NAMES)}")


# -- finger clearance (offline check of the plans) ------------------------------------------------------------------------
# The YAM's fingers as boxes in the gripper frame (x: closing, y: across, z: approach; origin the TCP), from yam.urdf's
# collision meshes (slices of tip_left/tip_right with the fingers closed): near the tip (z -5..+12.8 mm) each finger
# is 6.4 mm thick and +-13.7 mm across; z -20..-5 mm, 10 mm and +-20.1 mm; z -40..-20 mm, 18 mm and +-27.6 mm.
# ``gap`` is the pad-to-pad distance. A bounding envelope: the real fingers taper inside it.
def finger_boxes(tcp, rot, gap: float) -> list:
    out = []
    for side in (1.0, -1.0):
        for (x0, x1), half_y, (z0, z1) in (((0.0, 6.4), 13.7, (-5.0, 12.8)), ((1.0, 10.0), 20.1, (-20.0, -5.0)),
                                           ((2.2, 18.0), 27.6, (-40.0, -20.0))):
            lo = np.array([gap / 2 + x0 * MM, -half_y * MM, z0 * MM])
            hi = np.array([gap / 2 + x1 * MM, half_y * MM, z1 * MM])
            c = (lo + hi) / 2
            c[0] *= side
            out.append((np.asarray(tcp, float) + np.asarray(rot, float) @ c, np.asarray(rot, float), (hi - lo) / 2))
    return out


def box_distance(points, box) -> float:
    """The smallest distance from ``points`` (n, 3) to a box (centre, rotation, half sizes); negative inside."""
    c, r, h = box
    q = np.abs((np.asarray(points, float) - c) @ r) - h
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return float(np.min(outside + inside))


def obstacle_points(build: Build) -> dict:
    """Surface samples of what the fingers must not touch: the head's box and the cords near the tie's height."""
    hx, hy, hz = HEAD / 2
    g = np.linspace(-1, 1, 7)
    face = [(a, b) for a in g for b in g]
    pts = []
    for ax in range(3):
        for sgn in (-1.0, 1.0):
            for a, b in face:
                v = [0.0, 0.0, 0.0]
                v[ax] = sgn
                others = [i for i in range(3) if i != ax]
                v[others[0]], v[others[1]] = a, b
                pts.append(np.array(v) * [hx, hy, hz])
    head = world(build, np.array(pts))
    cords = []
    for c in CORDS:
        for z in np.linspace(build.head_pos[2] - 0.05, build.head_pos[2] + 0.05, 21):
            for a in np.linspace(0, 2 * math.pi, 24, endpoint=False):
                cords.append([c.xy[0] + c.diameter / 2 * math.cos(a), c.xy[1] + c.diameter / 2 * math.sin(a), z])
    return {"head": head, "cords": np.array(cords), "stand": stand_points(build)}


NEAR_LABELS = ("in", "grip", "align", "insert", "insert hold", "slide on", "open", "slide back", "regrip",
               "insert 2", "insert 2 hold", "let go")


def near_tie(label: str) -> bool:
    return label in NEAR_LABELS or label.split(" ")[0] in ("in", "pinch", "pull", "wrap")


# -- synthetic ties (tests and the fake-simulator dry run) -------------------------------------------------------------------
def _path_curve(path: np.ndarray):
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    def at(s: float) -> np.ndarray:
        if s >= cum[-1]:
            u = (path[-1] - path[-2]) / seg[-1]
            return path[-1] + (s - cum[-1]) * u
        i = int(np.searchsorted(cum, s, side="right") - 1)
        return path[i] + (s - cum[i]) / seg[i] * (path[i + 1] - path[i])

    return at, float(cum[-1])


def _loop_path_h(build: Build, grow: float, skip: tuple = ()) -> np.ndarray:
    """A loop (head frame, x_h-z_h) from the root round the cords (each grown by ``grow``) to the slot's entry,
    counter-clockwise, along the convex hull of the cords, the root and the entry."""
    ring = []
    for c in CORDS:
        if c.name in skip:
            continue
        p = (np.array([c.xy[0], c.xy[1], build.head_pos[2]]) - np.asarray(build.head_pos)) @ build.head_rot
        for a in np.linspace(0, 2 * math.pi, 120, endpoint=False):
            ring.append(p[[0, 2]] + (c.diameter / 2 + T / 2 + grow) * np.array([math.cos(a), math.sin(a)]))
    root, entry = ROOT[[0, 2]], np.array([SLOT_C, ENTRY_Z])
    h = [tuple(v) for v in _hull(np.vstack([ring, root, entry]))]  # counter-clockwise
    i0 = int(np.argmin([np.linalg.norm(np.subtract(v, root)) for v in h]))
    h = h[i0:] + h[:i0]
    i1 = int(np.argmin([np.linalg.norm(np.subtract(v, entry)) for v in h]))
    loop = np.array(h[: i1 + 1])
    return np.column_stack([loop[:, 0], np.zeros(len(loop)), loop[:, 1]])


def synthetic_tie(build: Build, grow: float = 0.0, skip: tuple = (), backwards: bool = False) -> np.ndarray:
    """A tie closed round the cords (world polyline, N_LINKS + 1 points, equal chords): its loop follows the cords'
    hull grown by ``grow`` (0: as tight as they allow), then down the slot's centre line and out of the exit (the tail
    is whatever is left of the strap). ``skip``: cords the loop leaves out. ``backwards``: the tail fed in from
    the exit side (wrong way)."""
    loop = _loop_path_h(build, grow, skip)
    zdir = 1.0 if backwards else -1.0
    start, stop = (EXIT_Z, ENTRY_Z) if backwards else (ENTRY_Z, EXIT_Z)
    slot = np.array([[SLOT_C, 0.0, start], [SLOT_C, 0.0, stop], [SLOT_C, 0.0, stop + zdir * 0.30]])
    path = np.vstack([loop, slot])
    at, _ = _path_curve(path)
    pts = _chord_walk(at, N_LINKS, LINK, 2 * TIE_L)
    return world(build, pts)


def tie_near_build(build: Build) -> Build:
    """``build`` with its head moved against the cords (the cords' centroid 9.2 mm and 5.9 mm from its centre), as a
    hand holding the head on the bundle leaves it: here the tightest loop leaves under SNUG_SLACK."""
    hull_c = np.mean([c.xy for c in CORDS], axis=0)
    pos = np.array([hull_c[0], hull_c[1], build.head_pos[2]]) - build.head_rot @ (np.array([9.2, 0.0, 5.9]) * MM)
    return Build(build.name + "-near", build.fixed, tuple(float(v) for v in pos), build.head_rot, build.points, build.why,
                 build.passage_y)
