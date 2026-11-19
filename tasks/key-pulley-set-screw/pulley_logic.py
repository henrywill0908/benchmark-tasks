"""Every number and every rule of key-pulley-set-screw, in plain Python (no Isaac Sim), so the offline tests and task.py
share one source.

Frame: the task file's (metres, x forward from the midpoint between the arm bases, y left, z up; TOP is the bench top).
The shaft's axis runs along +x at y 0, AX_Z high. The pulley turns about +x by its phase PHI (rad); phase 0 puts its
keyway and set screw straight up, where the key sits in the shaft's keyseat.

Parts, as specified for the real trial (ISO/R 773 / DIN 6885 A parallel key for a 20 mm shaft):
- shaft: steel, 20 mm (h7), keyseat 6 N9 wide, 3.5 mm deep (t1), 25.5 mm long, 1 mm short of the shoulder;
- key: steel, 6 x 6 x 25 mm, square ends;
- pulley: aluminium, 60 mm rim 12 mm wide on a 36 mm boss, 28 mm overall, bore 20 H7, keyway 6 JS9, 2.8 mm deep (t2);
- set screw: M5 x 0.8 hexagon socket cup point (ISO 4029), 2.5 mm socket, in the boss over the keyway;
- driver: a T-handle preset-torque driver with a 2.5 mm hex bit, its clutch set to 2.0 N m (it slips past that).

The simulator can't model a 6 N9 interference fit, threads or a clutch, so task code does (task.py, the rules below):
the key's press fit (a pad under it that gives at PRESS_N), the hub's slide over the key (allowed only with the
keyway lined up and the key fully home), the screw's travel (0.8 mm a turn while the driver's bit is in its socket) and
the driver's clutch (the screw stops turning SNUG_DEG past touching the key).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MM = 1e-3
DEG = math.pi / 180
TOP = 0.7475  # the station's bench top (yam_arena.task_file.TABLE_TOP["station"])
CONTROL_HZ = 20

# -- the shaft and its fixture ---------------------------------------------------------------------------------------
AX_Y = 0.0
AX_Z = TOP + 0.045  # the shaft's axis: 45 mm over the bench (15 mm under the rim)
SHAFT_R = 10.0 * MM
SHAFT_END = 0.200  # x of the shaft's free end (toward the robot)
SHOULDER_X = 0.315  # x of the shoulder's face: the pulley's rear face stops here
COLLAR_R = 15.0 * MM  # the shoulder collar (a 30 mm step on the shaft)
COLLAR_X = (SHOULDER_X, SHOULDER_X + 0.008)
SUPPORT_X = (0.375, 0.405)  # the support block clamping the shaft's far end
SHAFT_FAR = 0.395  # x of the shaft's far end (inside the support block)
SEAT_X = (0.2885, 0.3140)  # the keyseat (25.5 mm long, 1 mm short of the shoulder)
SEAT_W = 6.0 * MM  # 6 N9 on the drawing
SEAT_W_SIM = 6.4 * MM  # the simulator's slot: 0.2 mm a side for PhysX contact (stated in task.json)
SEAT_FLOOR_R = SHAFT_R - 3.5 * MM  # t1 = 3.5 mm: the floor 6.5 mm above the axis
SEAT_CX = (SEAT_X[0] + SEAT_X[1]) / 2
# The simulator's slot is longer than the drawing's at the free end (1.5 mm) and its top edges are broken LEAD_IN at
# about 30 deg: the arm places a key to about a millimetre, and a key end or side caught on a square lip sits perched
# (GPU round 1: every key stopped 3.4 mm high, one end on the slot's end). Stated in task.json.
SEAT_X_SIM = (SEAT_X[0] - 1.5 * MM, SEAT_X[1])
SEAT_CX_SIM = (SEAT_X_SIM[0] + SEAT_X_SIM[1]) / 2
LEAD_IN = 0.6 * MM

# -- the key and its press fit ---------------------------------------------------------------------------------------
KEY = np.array([25.0, 6.0, 6.0]) * MM  # x along the shaft, y across, z up when seated
KEY_SIM = np.array([24.6, 6.0, 6.0]) * MM
KEY_MASS = 0.020  # kg (a steel 6x6x25 key is 7 g; 20 g keeps PhysX steady in the grip, stated in task.json)
KEY_HOME_TOP_R = SEAT_FLOOR_R + KEY[2]  # 12.5 mm over the axis when home
KEY_SLACK_X = (SEAT_X_SIM[1] - SEAT_X_SIM[0] - KEY_SIM[0]) / 2  # how far a home key's centre can sit off the middle
PAD_RISE = 1.5 * MM  # a key set in the seat rests this far proud until pressed home
PRESS_N = 10.0  # N: the push that seats it (a 6 N9 key goes in by hand pressure or a light tap)
PAD_HOLD_STIFFNESS = 20000.0  # N/m: labkit.HOLD_STIFFNESS, the held pad's spring
PAD_GIVE = PRESS_N / PAD_HOLD_STIFFNESS  # the pad's sag at PRESS_N: past it, task code lets the pad drop
HOME_TOL = 0.3 * MM  # home: the key's top within this of KEY_HOME_TOP_R
LATERAL_TOL = 0.6 * MM  # and its centre within this of the keyseat's centre line (the sim slot allows 0.2)
YAW_TOL = 3.0  # deg
KEY_MAT = (0.255, -0.150)  # the green mat the key lies on (x, y)
MAT = (0.08, 0.08, 0.002)

# -- the pulley -----------------------------------------------------------------------------------------------------
RIM_R = 30.0 * MM
RIM_W = 12.0 * MM
BOSS_R = 18.0 * MM
HUB_L = 28.0 * MM  # the pulley overall, rim at its front (toward the robot), boss behind
BORE_R = 10.0 * MM
KEYWAY_ROOF_R = BORE_R + 2.8 * MM  # t2 = 2.8 mm: 0.3 mm over a home key's top
PULLEY_MASS = 0.15  # kg
REAR0 = 0.262  # x of the pulley's rear face at the start (its keyway clear of the keyseat, 26.5 mm short)
SLIDE_LIMITS = (SHAFT_END + HUB_L - REAR0, SHOULDER_X - REAR0)  # the slide joint, m from the start
PHI0 = -65.0  # deg about +x: the pulley starts with its keyway and screw turned 65 deg off the top, toward +y
CAPTURE_DEG = 4.0  # within this, a push slides the keyway onto the key (the key's chamfer and the hand's give)
SEATED_TOL = 0.5 * MM  # seated: the rear face within this of the shoulder
SLIDE_DAMPING = 2.0  # N s/m
TURN_DAMPING = 0.002  # N m s/rad
FORCE_JAM_MM = 1.0  # a hub held at the key's end and pushed this far into it (about 20 N) cocks a proud key: jammed

# -- the set screw and the driver's clutch --------------------------------------------------------------------------
SCREW_FROM_REAR = 8.0 * MM  # the screw's axis, from the pulley's rear face (on the boss, behind the rim)
PITCH = 0.8 * MM  # M5
SOCKET_DEPTH = 2.5 * MM
TIP_START_R = KEYWAY_ROOF_R + 0.3 * MM  # the reset backs the screw out until its tip is 0.3 mm above the keyway roof
SNUG_DEG = 30.0  # past first touch, the turn that reaches the clutch's 2.0 N m (estimated: see task.json)
SCREW_BACK_LIMIT = -720.0  # deg: it can be backed out two more turns


def screw_tip_r(psi_deg: float) -> float:
    """The screw tip's distance from the shaft's axis after turning ``psi_deg`` in from its start."""
    return TIP_START_R - PITCH * psi_deg / 360.0


def touch_deg(keyed: bool) -> float:
    """How far the screw turns from its start before its tip touches: the key's top when the pulley is keyed, else the
    shaft (its keyway over bare shaft)."""
    stop = KEY_HOME_TOP_R if keyed else SHAFT_R
    return (TIP_START_R - stop) / PITCH * 360.0


TOUCH_KEYED = touch_deg(True)  # 270 deg
TOUCH_BARE = touch_deg(False)  # 1395 deg

# The driver: T-handle (crossbar along its x), shank down its -z, bit tip DRIVER_TIP below the crossbar's centre.
BAR = np.array([80.0, 16.0, 16.0]) * MM
SHANK = np.array([5.0, 5.0, 60.0]) * MM  # the shank's collider (a 2.5 mm hex bit in a 5 mm shank)
DRIVER_TIP = -(BAR[2] / 2 + SHANK[2])  # the bit's tip, z in the driver's frame
DRIVER_MASS = 0.06
HOLDER = (0.235, 0.150)  # the driver's stand (x, y): a cup the shank hangs in, the crossbar on its rim
CUP_H = 0.065
CUP_IN = 14.0 * MM
CUP_WALL = 3.0 * MM
ENGAGE_LATERAL = 2.0 * MM  # the bit drops into the socket when its tip is within this of the screw's axis,
ENGAGE_TILT = 10.0  # deg, the shank within this of the axis,
ENGAGE_ABOVE = 2.0 * MM  # and the tip no higher than this above the socket's mouth
ENGAGE_BELOW = 8.0 * MM  # a tip this far below the mouth is resting on the screw's face (the boss doesn't collide with
#                          the bit in the simulator): it drops in once it lines up (round 2's no_key reached -5.5 mm)
LIFT_OUT = 5.0 * MM  # a hand holding the crossbar this far above where the seated bit puts it pulls the bit out

# -- scoring -------------------------------------------------------------------------------------------------------
HAND_CLEAR = 20.0 * MM
DRIVER_OUT = 20.0 * MM  # the bit's tip at least this far from the socket's mouth
STILL_MM_S = 5.0  # mm/s over the last second
RUN_LIMIT_STEPS = 3000  # 150 s at 20 Hz


# -- geometry -------------------------------------------------------------------------------------------------------
def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def up_dir(phi: float) -> np.ndarray:
    """The pulley's keyway/screw direction (unit, world) at phase ``phi`` (rad)."""
    return rot_x(phi) @ np.array([0.0, 0.0, 1.0])


def rear_x(slide: float) -> float:
    return REAR0 + slide


def screw_mouth(slide: float, phi: float) -> np.ndarray:
    """Where the screw's socket opens on the boss (world)."""
    return np.array([rear_x(slide) - SCREW_FROM_REAR, AX_Y, AX_Z]) + BOSS_R * up_dir(phi)


def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat_xyzw(m) -> np.ndarray:
    m = np.asarray(m, float)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# -- the key ---------------------------------------------------------------------------------------------------------
@dataclass
class KeyFit:
    in_slot: bool  # its bottom below the shaft's surface, within the slot
    home: bool
    top_r: float  # its top's height over the axis (m)
    front_x: float  # x of its end toward the robot
    lateral: float
    yaw_deg: float


def key_fit(pos, quat_xyzw) -> KeyFit:
    """Where the key is relative to the keyseat, from its pose."""
    R = quat_xyzw_to_matrix(quat_xyzw)
    p = np.asarray(pos, float)
    half = KEY_SIM / 2
    corners = np.array([p + R @ (half * np.array(s)) for s in
                        [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]])
    top_r = float(corners[:, 2].max() - AX_Z)
    bottom_r = float(corners[:, 2].min() - AX_Z)
    lateral = float(abs(p[1] - AX_Y))
    long_axis = R[:, 0]
    yaw = math.degrees(math.atan2(abs(long_axis[1]), abs(long_axis[0])))
    pitch = math.degrees(math.asin(min(1.0, abs(long_axis[2]))))
    within_x = SEAT_X_SIM[0] - 1.0 * MM <= corners[:, 0].min() and corners[:, 0].max() <= SEAT_X_SIM[1] + 1.0 * MM
    slot_top = math.sqrt(SHAFT_R ** 2 - (SEAT_W_SIM / 2) ** 2)  # where the slot's walls meet the round
    in_slot = bool(within_x and lateral <= SEAT_W_SIM / 2 and bottom_r < slot_top and yaw < 15.0 and pitch < 15.0)
    home = bool(in_slot and top_r <= KEY_HOME_TOP_R + HOME_TOL and lateral <= LATERAL_TOL and yaw <= YAW_TOL
                and pitch <= YAW_TOL and abs(p[0] - SEAT_CX_SIM) <= KEY_SLACK_X + 0.2 * MM)
    return KeyFit(in_slot, home, top_r, float(corners[:, 0].min()), lateral, yaw)


def key_home_pose(x: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The key's centre and quaternion when fully home, at ``x`` along the slot (its middle if None)."""
    x = SEAT_CX_SIM if x is None else float(np.clip(x, SEAT_CX_SIM - KEY_SLACK_X, SEAT_CX_SIM + KEY_SLACK_X))
    return np.array([x, AX_Y, AX_Z + SEAT_FLOOR_R + KEY[2] / 2]), np.array([0.0, 0.0, 0.0, 1.0])


# -- the hub's slide over the key ----------------------------------------------------------------------------------
@dataclass
class Gate:
    """What task code does with the pulley's slide this step. ``block_at``: hold the slide here (m), or None;
    ``snap``: turn the pulley to phase 0 and hold it there (keyed); ``jam``: the key is cocked (latched failure)."""

    block_at: float | None = None
    snap: bool = False
    jam: bool = False
    keyed: bool = False


def gate(slide: float, phi: float, key: KeyFit, screw_psi: float, block_held: float | None) -> Gate:
    """The keyway meets the key. ``block_held``: where the slide was held last step (None if free); the slide reads
    past it when the hand pushes against the hold."""
    if not key.in_slot:
        return Gate()
    stop = key.front_x - 0.5 * MM - REAR0  # the slide at which the rear face meets the key's end
    lined_up = abs(math.degrees(wrap(phi))) <= CAPTURE_DEG
    screw_clear = screw_tip_r(screw_psi) >= KEYWAY_ROOF_R - 0.05 * MM  # a screw run into the keyway stops the key
    if slide < stop - (2.0 * MM if block_held is not None else 0.0):  # a held hub reads a little short: keep holding
        return Gate()
    if block_held is None and slide > stop + 3.0 * MM:  # already over the key (it was let through earlier)
        return Gate(keyed=lined_up and key.home, snap=lined_up and key.home)
    if lined_up and key.home and screw_clear:
        return Gate(snap=True, keyed=slide > stop + 1.0 * MM)
    # A proud key under a lined-up keyway: the hub's keyway rides onto its end. Held there (it can't pass), and if the
    # hand keeps pushing (the held slide reads FORCE_JAM_MM past the hold, about 20 N), the key cocks: jammed.
    jam = (not key.home) and lined_up and block_held is not None and slide >= block_held + FORCE_JAM_MM * MM
    return Gate(block_at=stop, jam=bool(jam))


# -- the screw and the clutch ---------------------------------------------------------------------------------------
@dataclass
class Screw:
    """The set screw's turn from its start (deg, + is in) and the driver's clutch. ``slipped`` latches when the clutch
    gives; ``tight_where`` records (slide, keyed) at that moment."""

    psi: float = 0.0
    slipped: bool = False
    tight_where: tuple | None = None

    def drive(self, turn_deg: float, keyed: bool, slide: float) -> None:
        """The driver's bit turned ``turn_deg`` in its socket this step (+ tightens: clockwise seen from outside)."""
        stop = touch_deg(keyed) + SNUG_DEG
        want = self.psi + turn_deg
        if want >= stop:
            if want > stop + 1.0:  # the clutch slipped: the screw is at 2.0 N m
                self.slipped = True
                self.tight_where = (float(slide), bool(keyed))
            self.psi = stop
        else:
            self.psi = max(SCREW_BACK_LIMIT, want)
            if self.psi < stop - 10.0:  # backed off: no longer tight
                self.slipped, self.tight_where = False, None

    @property
    def tight(self) -> bool:
        return self.slipped


def seated(slide: float) -> bool:
    return rear_x(slide) >= SHOULDER_X - SEATED_TOL


# -- the driver's bit in the socket -------------------------------------------------------------------------------------
def driver_tip(pos, quat_xyzw) -> tuple[np.ndarray, np.ndarray]:
    """The bit's tip (world) and the shank's direction toward the tip (unit)."""
    R = quat_xyzw_to_matrix(quat_xyzw)
    return np.asarray(pos, float) + R @ np.array([0.0, 0.0, DRIVER_TIP]), -R[:, 2]


def engages(tip, shank_dir, mouth, up) -> bool:
    """Does the bit drop into the socket? ``up`` points out of the socket."""
    d = np.asarray(tip, float) - np.asarray(mouth, float)
    along = float(d @ up)
    lateral = float(np.linalg.norm(d - along * up))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(-np.asarray(shank_dir) @ up)))))
    return lateral <= ENGAGE_LATERAL and tilt <= ENGAGE_TILT and -ENGAGE_BELOW <= along <= ENGAGE_ABOVE


def seated_driver_pose(quat_xyzw, mouth, up) -> tuple[np.ndarray, np.ndarray, float]:
    """The driver's pose with its bit fully in the socket and its shank on the screw's axis, keeping its twist about
    the axis. Returns (position, quaternion, twist in rad about ``up``)."""
    R = quat_xyzw_to_matrix(quat_xyzw)
    up = np.asarray(up, float)
    bar = R[:, 0] - (R[:, 0] @ up) * up  # the crossbar, flattened onto the plane across the axis
    if np.linalg.norm(bar) < 1e-6:
        bar = np.cross(up, [1.0, 0.0, 0.0])
    bar /= np.linalg.norm(bar)
    Rn = np.column_stack([bar, np.cross(up, bar), up])
    tip = np.asarray(mouth, float) - SOCKET_DEPTH * up
    pos = tip - Rn @ np.array([0.0, 0.0, DRIVER_TIP])
    ref = np.cross(up, [0.0, 0.0, 1.0]) if abs(up[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    ref = ref - (ref @ up) * up
    ref /= np.linalg.norm(ref)
    twist = math.atan2(float(np.cross(ref, bar) @ up), float(ref @ bar))
    return pos, matrix_to_quat_xyzw(Rn), twist


# -- the score -------------------------------------------------------------------------------------------------------
@dataclass
class Facts:
    """What score() needs, gathered each step by task.py's hook."""

    key_home: bool = False
    keyed: bool = False
    seated: bool = False
    tight: bool = False  # the clutch slipped with the pulley keyed and against the shoulder, and the screw still there
    driver_out: bool = False  # bit out of the socket by DRIVER_OUT and the driver on the bench
    hand_clear: float = 0.0  # m, nearest gripper point to the key, pulley and driver
    still: bool = False
    failed: str | None = None  # latched: "jam", "key off the bench", "driver off the bench"
    notes: dict = field(default_factory=dict)


def task_score(f: Facts) -> float:
    """0 after a latched failure (a key cocked under the hub, the key or driver dropped off the bench). Otherwise 1.0
    when the key is home, the pulley is on it against the shoulder, the set screw tight (the clutch slipped) there, the
    driver out of the socket and on the bench, every gripper point 2 cm clear and everything still; 0.75 when it is
    locked like that but not the rest; 0.5 keyed and seated, screw not tight; 0.25 key home only; else 0."""
    if f.failed:
        return 0.0
    if not f.key_home:
        return 0.0
    if not (f.keyed and f.seated):
        return 0.25
    if not f.tight:
        return 0.5
    if f.driver_out and f.hand_clear >= HAND_CLEAR and f.still:
        return 1.0
    return 0.75


# -- the scripted demo -----------------------------------------------------------------------------------------------
# A plan: {"left": [step...], "right": [step...]}; a step is (seconds, TCP position, fingers, rotation 3x3 or "ready",
# label). Finger names (task.py maps a gap g to finger joints -g/2): open 94 mm, wide 30 mm, closed 0.
FINGER_GAP = {"open": 94.0, "wide": 30.0, "closed": 0.0}
PAD_TIP = 12.8 * MM  # the finger pads reach this far past the TCP along the approach (yam fingers)
PAD_HALF = 6.4 * MM  # a closed finger pair is this thick each side of the TCP along the closing axis


def rot_axes(approach, closing) -> np.ndarray:
    """A TCP rotation from where the gripper points (its z) and the way its fingers close (its x); yam_arena's grips
    are built the same way (grip_rotation(0): approach -z, closing -y)."""
    z = np.asarray(approach, float) / np.linalg.norm(approach)
    x = np.asarray(closing, float)
    x = x - (x @ z) * z
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def rot_down(yaw_deg: float) -> np.ndarray:
    """yam_arena's straight-down grip turned ``yaw_deg`` about the vertical (equals grip_rotation(radians(yaw)))."""
    y = math.radians(yaw_deg)
    return rot_axes((0.0, 0.0, -1.0), (math.sin(y), -math.cos(y), 0.0))


def P(*xyz) -> tuple:
    return tuple(float(v) for v in xyz)


# where things are
KEY_START = P(KEY_MAT[0], KEY_MAT[1], TOP + MAT[2] + KEY[2] / 2)
BAR_START = P(HOLDER[0], HOLDER[1], TOP + CUP_H + BAR[2] / 2)
GRIP_KEY_BELOW_TOP = 3.0 * MM  # the right hand holds the key's top 3 mm (its fingertips stay clear of the shaft)
BAR_GRIP_ABOVE = 6.0 * MM  # the TCP 6 mm over the crossbar's centre (fingertips 1.2 mm over its underside)
RIM_GRIP_R = BOSS_R + 3.0 * MM + PAD_TIP  # the TCP's radius when the fingertips are 3 mm off the boss
TURN_ARC = (-45.0, 20.0)  # deg: where the left hand takes the rim and lets go (65 = -PHI0; it reaches -80..+20)
PRESS_PAST = 25.0 * MM  # closed fingertips aim this far into the key's top: the arm's give makes about 15 N
PUSH_PAST = 10.0 * MM  # the push aims this far past the seat (the arm gives under load)
STROKE = (60.0, -120.0)  # deg of wrist yaw: each tightening stroke (clockwise seen from above)
STROKES = 2


def _key_plan(variant: str) -> list:
    tcp_on_mat = KEY_START[2] + KEY[2] / 2 - GRIP_KEY_BELOW_TOP + PAD_TIP
    x, y = KEY_START[:2]
    above_seat = (SEAT_CX_SIM, AX_Y)
    rest_top = AX_Z + SEAT_FLOOR_R + PAD_RISE + KEY[2]  # the key's top while it rests on the pad
    tcp_place = rest_top + 0.5 * MM - GRIP_KEY_BELOW_TOP + PAD_TIP
    tcp_press = AX_Z + KEY_HOME_TOP_R + PAD_TIP - PRESS_PAST
    R = rot_down(0.0)  # fingers close across the key's 6 mm width (along y)
    s = [(2.0, P(x, y, tcp_on_mat + 0.05), "open", R, "over the key"),
         (1.5, P(x, y, tcp_on_mat), "open", R, "down to the key"),
         (1.0, P(x, y, tcp_on_mat), "closed", R, "grip the key"),
         (1.5, P(x, y, TOP + 0.12), "closed", R, "lift"),
         (2.5, P(*above_seat, TOP + 0.12), "closed", R, "over the keyseat"),
         (2.0, P(*above_seat, tcp_place + 0.02), "closed", R, "down"),
         (2.0, P(*above_seat, tcp_place), "closed", R, "set it in the seat"),
         (1.0, P(*above_seat, tcp_place), "wide", R, "let go"),
         (1.0, P(*above_seat, tcp_place + 0.015), "wide", R, "up")]
    if variant != "proud":
        s += [(1.0, P(*above_seat, tcp_place + 0.015), "closed", R, "close the fingers"),
              (1.5, P(*above_seat, tcp_press + PRESS_PAST + 2.0 * MM), "closed", R, "onto the key"),
              (2.0, P(*above_seat, tcp_press), "closed", R, "press it home"),
              (1.0, P(*above_seat, tcp_press), "closed", R, "hold"),
              (1.5, P(*above_seat, TOP + 0.10), "closed", R, "up")]
    else:
        s += [(1.5, P(*above_seat, TOP + 0.10), "wide", R, "up (not pressed)")]
    s += [(2.0, None, "open", "ready", "back to ready")]
    return s


def rim_grip(theta_deg: float, slide: float = 0.0, standoff: float = 0.0) -> tuple:
    """The left hand on the rim at angle ``theta_deg`` (from the top, about +x): its TCP and rotation, fingers closing
    along x across the rim's 12 mm, pointing at the axis; ``standoff`` backs it out radially."""
    u = up_dir(math.radians(theta_deg))
    x = rear_x(slide) - HUB_L + RIM_W / 2
    tcp = np.array([x, AX_Y, AX_Z]) + (RIM_GRIP_R + standoff) * u
    return P(*tcp), rot_axes(-u, (-1.0, 0.0, 0.0))  # closing toward -x: the left wrist reaches it (+x it can't)


def _turn_plan(err_deg: float = 0.0) -> list:
    a0, a1 = TURN_ARC[0], TURN_ARC[1] + err_deg
    s = []
    p, R = rim_grip(a0, standoff=0.04)
    s.append((2.5, p, "open", R, "over the rim"))
    p, R = rim_grip(a0)
    s.append((2.0, p, "open", R, "onto the rim"))
    s.append((1.0, p, "closed", R, "take the rim"))
    n = 10
    for i in range(1, n + 1):
        p, R = rim_grip(a0 + (a1 - a0) * i / n)
        s.append((0.5, p, "closed", R, f"turn the pulley ({i}/{n})"))
    s.append((1.0, p, "open", R, "let go of the rim"))
    p, R = rim_grip(a1 - 15.0, standoff=0.015)  # out and back over the top: radially at +20 is out of reach
    s.append((1.0, p, "open", R, "back off"))
    s.append((1.5, P(p[0], p[1], AX_Z + RIM_R + 0.06), "open", rot_down(-90.0), "up"))
    return s


def _push_plan() -> list:
    R = rot_down(-90.0)  # fingers close along x (yaw -90: the left wrist can't do +90 here): closed, a 12.8 mm block
    z = AX_Z + 12.0 * MM + PAD_TIP  # fingertips 2 mm over the shaft's top, on the rim's upper face
    x0 = REAR0 - HUB_L - PAD_HALF - 5.0 * MM
    x1 = SHOULDER_X - HUB_L - PAD_HALF + PUSH_PAST
    return [(2.0, P(x0, AX_Y, z + 0.05), "closed", R, "in front of the pulley"),
            (1.5, P(x0, AX_Y, z), "closed", R, "down"),
            (6.0, P(x1, AX_Y, z), "closed", R, "push it on to the shoulder"),
            (1.0, P(x1, AX_Y, z), "closed", R, "hold"),
            (1.0, P(x1 - 0.02, AX_Y, z), "closed", R, "back off"),
            (1.5, P(x1 - 0.02, AX_Y, z + 0.06), "open", R, "up")]


def bar_tcp(bar_centre) -> tuple:
    return P(bar_centre[0], bar_centre[1], bar_centre[2] + BAR_GRIP_ABOVE)


def _driver_plan(leave_in: bool = False) -> list:
    mouth = screw_mouth(SLIDE_LIMITS[1], 0.0)
    seated_bar = mouth + np.array([0.0, 0.0, -SOCKET_DEPTH - DRIVER_TIP])
    y0, y1 = STROKE
    s = [(2.0, bar_tcp(np.add(BAR_START, (0, 0, 0.04))), "wide", rot_down(0.0), "over the driver"),
         (1.5, bar_tcp(BAR_START), "wide", rot_down(0.0), "down to its handle"),
         (1.0, bar_tcp(BAR_START), "closed", rot_down(0.0), "grip the handle"),
         (2.0, bar_tcp(np.add(BAR_START, (0, 0, 0.10))), "closed", rot_down(0.0), "lift it out"),
         (3.0, bar_tcp(seated_bar + (0, 0, 0.02)), "closed", rot_down(y0), "over the screw"),
         (2.0, bar_tcp(seated_bar + (0, 0, 0.004)), "closed", rot_down(y0), "bit to the socket"),
         (1.0, bar_tcp(seated_bar + (0, 0, 0.004)), "closed", rot_down(y0), "line it up"),
         (1.0, bar_tcp(seated_bar), "closed", rot_down(y0), "bit in")]
    for k in range(STROKES):
        n = 9
        for i in range(1, n + 1):
            yaw = y0 + (y1 - y0) * i / n
            s.append((0.7, bar_tcp(seated_bar), "closed", rot_down(yaw), f"tighten, stroke {k + 1} ({i}/{n})"))
        if k < STROKES - 1:
            s += [(0.8, bar_tcp(seated_bar), "wide", rot_down(y1), "let go of the handle"),
                  (1.0, bar_tcp(seated_bar + (0, 0, 0.022)), "wide", rot_down(y1), "up off it"),
                  (2.0, bar_tcp(seated_bar + (0, 0, 0.022)), "wide", rot_down(y0), "turn the wrist back"),
                  (1.0, bar_tcp(seated_bar), "wide", rot_down(y0), "down round the handle"),
                  (0.8, bar_tcp(seated_bar), "closed", rot_down(y0), "grip it again")]
    if leave_in:
        s += [(0.8, bar_tcp(seated_bar), "wide", rot_down(y1), "let go of the handle"),
              (1.0, bar_tcp(seated_bar + (0, 0, 0.022)), "wide", rot_down(y1), "up off it"),
              (1.5, bar_tcp(seated_bar + (-0.04, 0.03, 0.035)), "wide", rot_down(-90.0), "away from the pulley")]
    else:
        back = -60.0  # the yaw the driver goes home at
        s += [(1.5, bar_tcp(seated_bar + (0, 0, 0.02)), "closed", rot_down(y1), "pull the bit out"),
              (1.5, bar_tcp(seated_bar + (-0.04, 0.03, 0.035)), "closed", rot_down(-90.0), "away from the pulley"),
              (2.5, bar_tcp(np.add(BAR_START, (0, 0, 0.10))), "closed", rot_down(back), "back over its stand"),
              (1.0, bar_tcp(np.add(BAR_START, (0, 0, 0.075))), "closed", rot_down(back), "over the cup"),
              (1.0, bar_tcp(np.add(BAR_START, (0, 0, 0.075))), "closed", rot_down(back), "line it up"),
              (1.0, bar_tcp(np.add(BAR_START, (0, 0, 0.045))), "closed", rot_down(back), "shank into the cup"),
              (1.0, bar_tcp(np.add(BAR_START, (0, 0, 0.02))), "closed", rot_down(back), "down"),
              (1.0, bar_tcp(np.add(BAR_START, (0, 0, 0.003))), "closed", rot_down(back), "handle onto the rim"),
              (1.0, bar_tcp(np.add(BAR_START, (0, 0, 0.003))), "wide", rot_down(back), "let go"),
              (1.5, bar_tcp(np.add(BAR_START, (0, 0, 0.05))), "wide", rot_down(back), "up")]
    s += [(2.0, None, "open", "ready", "back to ready")]
    return s


VARIANTS = {
    "full": "the full-score demo",
    "no_key": "no key: the pulley turned, pushed home and screwed down on bare shaft (expect 0)",
    "proud": "the key set in its seat but not pressed home, then the pulley pushed on (expect 0: the key cocks)",
    "no_turn": "the keyway left 65 deg off the key: the push stops at the key's end (expect 0.25)",
    "leave_driver": "everything, but the driver left in the screw (expect 0.75)",
    "turn_err3": "the pulley turned 3 deg short of the key (inside the 4 deg capture: expect 1.0)",
    "turn_err6": "the pulley turned 6 deg short of the key (outside the capture: expect 0.25)",
}


def demo_plan(variant: str = "full") -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"unknown demo variant {variant!r}; one of {sorted(VARIANTS)}")
    right = [] if variant == "no_key" else _key_plan(variant)
    t_right = sum(s[0] for s in right)
    left = [(max(1.0, t_right), None, "open", "ready", "wait for the key")]
    if variant != "no_turn":
        err = {"turn_err3": -3.0, "turn_err6": -6.0}.get(variant, 0.0)
        left += _turn_plan(err)
    left += _push_plan()
    left += _driver_plan(leave_in=variant == "leave_driver")
    if not right:
        right = [(1.0, None, "open", "ready", "stay at ready")]
    return {"left": left, "right": right}


def plan_seconds(plan: dict) -> float:
    return max(sum(s[0] for s in steps) for steps in plan.values())


# -- distances (gripper clearance) ------------------------------------------------------------------------------------
def dist_box(points, centre, R, half) -> float:
    """Smallest distance from ``points`` (n, 3) to a box (centre, rotation, half sizes); negative inside."""
    q = np.abs((np.atleast_2d(np.asarray(points, float)) - np.asarray(centre, float)) @ np.asarray(R, float)) \
        - np.asarray(half, float)
    out = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    return float(np.min(out + np.minimum(np.max(q, axis=1), 0.0)))


def dist_x_cylinder(points, x0: float, x1: float, r: float) -> float:
    """Smallest distance from ``points`` to a solid cylinder about the shaft's axis from x0 to x1."""
    p = np.atleast_2d(np.asarray(points, float))
    radial = np.hypot(p[:, 1] - AX_Y, p[:, 2] - AX_Z) - r
    axial = np.maximum(x0 - p[:, 0], p[:, 0] - x1)
    out = np.hypot(np.maximum(radial, 0.0), np.maximum(axial, 0.0))
    return float(np.min(np.where((radial < 0) & (axial < 0), np.maximum(radial, axial), out)))


def driver_boxes(pos, quat_xyzw) -> list:
    R = quat_xyzw_to_matrix(quat_xyzw)
    p = np.asarray(pos, float)
    shank_c = p + R @ np.array([0.0, 0.0, -(BAR[2] + SHANK[2]) / 2])
    return [(p, R, BAR / 2), (shank_c, R, SHANK / 2)]


def hand_clearance(points, slide: float, key_pos, key_quat, driver_pos, driver_quat) -> float:
    """The nearest gripper point's distance to the pulley, the key and the driver (m)."""
    if points is None or len(points) == 0:
        return float("inf")
    rear = rear_x(slide)
    d = [dist_x_cylinder(points, rear - HUB_L, rear - HUB_L + RIM_W, RIM_R),
         dist_x_cylinder(points, rear - HUB_L + RIM_W, rear, BOSS_R),
         dist_box(points, key_pos, quat_xyzw_to_matrix(key_quat), KEY_SIM / 2)]
    d += [dist_box(points, c, R, h) for c, R, h in driver_boxes(driver_pos, driver_quat)]
    return min(d)


# -- the per-step bookkeeping (task.py's on_step hook applies its actions) ------------------------------------------------
@dataclass
class Inputs:
    slide: float
    turn: float
    key_pos: np.ndarray
    key_quat: np.ndarray
    driver_pos: np.ndarray
    driver_quat: np.ndarray
    driver_ang_vel: np.ndarray
    pad: float  # the pad's joint (m, 0 up, -PAD_RISE down)
    grippers: list  # gripper points (TCPs and pad tips), world
    tcps: list  # each arm's (TCP, finger gap in m), world


KEEP = "keep"


@dataclass
class Actions:
    hold_pad: bool | None = None  # True: hold the pad up; False: let it go; None: leave it
    slide_hold: object = KEEP  # a value to hold the slide at, None to release it, KEEP to leave it as it is
    turn_hold: object = KEEP
    key_pose: tuple | None = None  # (pos, quat): write the key there, at rest
    driver_pose: tuple | None = None  # (pos, quat, ang_vel)


class Tracker:
    """Everything task code does each control step, as pure logic: the key's press fit and weld, the hub's gate over
    the key, the driver's bit in the socket, the screw and its clutch, the locks, and the facts the score reads."""

    def __init__(self) -> None:
        self.weld_x: float | None = None
        self.steps = 0
        self.pad_released = False
        self.key_welded = False
        self.block_held: float | None = None
        self.turn_held = False
        self.locked = False
        self.engaged = False
        self.twist_prev = 0.0
        self.screw = Screw()
        self.keyed = False
        self.failed: str | None = None
        self.jam_slide: float | None = None
        self.trail: list = []
        self.facts = Facts()
        self.log: dict = {}

    def step(self, x: Inputs) -> Actions:
        a = Actions()
        self.steps += 1
        if self.steps == 1:
            a.hold_pad = True
        # the key's press fit: the held pad gives PAD_GIVE under PRESS_N, then drops; a home key is welded there
        if not self.pad_released and x.pad <= -PAD_GIVE:
            self.pad_released, a.hold_pad = True, False
        fit = key_fit(x.key_pos, x.key_quat)
        if not self.key_welded and fit.home:
            self.key_welded, self.weld_x = True, float(x.key_pos[0])
        raw_fit = fit
        key_pos, key_quat = np.asarray(x.key_pos, float), np.asarray(x.key_quat, float)
        if self.key_welded:
            key_pos, key_quat = key_home_pose(self.weld_x)
            a.key_pose = (key_pos, key_quat)
            fit = key_fit(key_pos, key_quat)
        # the hub over the key
        if self.failed is None:
            g = gate(x.slide, x.turn, fit, self.screw.psi, self.block_held)
            if g.jam:
                self.failed, self.jam_slide = "jam", x.slide
            if g.block_at is not None:
                if self.block_held is None or abs(self.block_held - g.block_at) > 1e-6:
                    a.slide_hold = g.block_at
                self.block_held = g.block_at
            elif self.block_held is not None and not self.locked:
                a.slide_hold, self.block_held = None, None
            off_key = (not fit.in_slot) or x.slide < fit.front_x - REAR0 - 3.0 * MM
            if g.snap and not self.turn_held:
                a.turn_hold, self.turn_held = 0.0, True
            elif not g.snap and self.turn_held and not self.locked and g.block_at is None and off_key:
                a.turn_hold, self.turn_held = None, False  # pulled back off the key: free to turn again
            self.keyed = bool(fit.home and self.turn_held and x.slide > fit.front_x - REAR0 + 0.5 * MM)
        if self.failed == "jam":
            a.slide_hold = self.jam_slide
        # the driver's bit and the screw
        up = up_dir(x.turn)
        mouth = screw_mouth(x.slide, x.turn)
        tip, shank = driver_tip(x.driver_pos, x.driver_quat)
        d_tip = tip - mouth
        bit = (float(d_tip @ up), float(np.linalg.norm(d_tip - (d_tip @ up) * up)),
               math.degrees(math.acos(max(-1.0, min(1.0, float(-np.asarray(shank) @ up))))))
        if not self.engaged and engages(tip, shank, mouth, up):
            self.engaged = True
            self.twist_prev = seated_driver_pose(x.driver_quat, mouth, up)[2]
        if self.engaged:
            pos, quat, twist = seated_driver_pose(x.driver_quat, mouth, up)
            # pulled out: a hand closed on the handle (gap under the bar's 16 mm plus 4) has lifted it LIFT_OUT
            pulling = [t for t, gap in x.tcps if gap < BAR[1] + 4.0 * MM
                       and np.linalg.norm(np.asarray(t, float) - pos) < 0.03
                       and float((np.asarray(t, float) - pos) @ up) > BAR_GRIP_ABOVE + LIFT_OUT]
            if pulling:
                self.engaged = False
            else:
                turned = -math.degrees(wrap(twist - self.twist_prev))  # + tightens: clockwise seen from outside
                self.twist_prev = twist
                self.screw.drive(turned, self.keyed, x.slide)
                spin = float(np.asarray(x.driver_ang_vel, float) @ up)
                a.driver_pose = (pos, quat, up * spin)
                tip = pos + quat_xyzw_to_matrix(quat) @ np.array([0.0, 0.0, DRIVER_TIP])
        # the tight screw locks the pulley where it is
        if self.screw.tight and not self.locked:
            self.locked = True
            a.slide_hold, a.turn_hold = x.slide, x.turn
        elif self.locked and not self.screw.tight:
            self.locked = False
            a.slide_hold = self.block_held
            a.turn_hold = 0.0 if self.turn_held else None
        # latched failures
        if self.failed is None:
            if x.key_pos[2] < TOP - 0.02:
                self.failed = "key off the bench"
            elif x.driver_pos[2] < TOP - 0.02:
                self.failed = "driver off the bench"
        # stillness over the last second: the key, a rim point, the driver
        rim = np.array([x.slide, RIM_R * math.cos(x.turn), RIM_R * math.sin(x.turn)])
        self.trail.append(np.vstack([key_pos, rim, np.asarray(x.driver_pos, float)]))
        del self.trail[:-CONTROL_HZ - 1]
        still = len(self.trail) > CONTROL_HZ and \
            float(np.max(np.linalg.norm(self.trail[-1] - self.trail[0], axis=1))) <= STILL_MM_S * MM
        tw = self.screw.tight_where
        f = self.facts
        f.key_home = self.key_welded
        f.keyed = self.keyed
        f.seated = seated(x.slide)
        f.tight = bool(self.screw.tight and tw is not None and tw[1] and seated(tw[0]))
        f.driver_out = bool(not self.engaged and np.linalg.norm(tip - mouth) >= DRIVER_OUT
                            and x.driver_pos[2] > TOP - 0.01)
        f.hand_clear = hand_clearance(x.grippers, x.slide, key_pos, key_quat, x.driver_pos, x.driver_quat)
        f.still = bool(still)
        f.failed = self.failed
        self.log = dict(step=self.steps, slide_mm=round(x.slide * 1e3, 2), turn_deg=round(math.degrees(x.turn), 2),
                        pad_mm=round(x.pad * 1e3, 2), key_top_mm=round(fit.top_r * 1e3, 2), key_in_slot=fit.in_slot,
                        key_home=f.key_home, keyed=f.keyed, seated=f.seated, block=self.block_held,
                        engaged=self.engaged, psi=round(self.screw.psi, 1), slipped=self.screw.slipped,
                        tight=f.tight, driver_out=f.driver_out, hand_clear_mm=round(min(f.hand_clear, 9.99) * 1e3, 1),
                        still=f.still, failed=self.failed, score=task_score(f),
                        key_dx_mm=round((float(x.key_pos[0]) - SEAT_CX_SIM) * 1e3, 2),
                        key_dy_mm=round((float(x.key_pos[1]) - AX_Y) * 1e3, 2),
                        key_z_mm=round((float(x.key_pos[2]) - AX_Z) * 1e3, 2), key_yaw=round(raw_fit.yaw_deg, 2),
                        key_raw_top_mm=round(raw_fit.top_r * 1e3, 2),
                        driver_mm=[round((float(v) - o) * 1e3, 1) for v, o in zip(x.driver_pos, (0.0, 0.0, TOP))],
                        bit_along_mm=round(bit[0] * 1e3, 1), bit_lat_mm=round(bit[1] * 1e3, 1),
                        bit_tilt=round(bit[2], 1),
                        tcp_mm=[[round(float(v) * 1e3, 1) for v in t] + [round(g * 1e3, 1)] for t, g in x.tcps])
        return a
