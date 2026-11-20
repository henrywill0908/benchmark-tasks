"""Every number and rule of wipe-sticky-syrup-spill, in plain Python (no simulator), so the offline tests, the dry run,
the GPU probe and task.py share one source.

The scene (task frame: x forward from the arm bases, y to the robot's left, z up; m):

- a spill of dried maple syrup on the table, tracked as a GRID_N x GRID_N grid of CELL squares centred on SPILL_C.
  PUDDLE marks the cells the spill starts on; the others start clean and only get syrup if it is smeared onto them.
- a trigger spray bottle of warm water standing at BOTTLE_START, its nozzle pointing forward (+x). Its trigger is a
  sprung slide; squeezed past FIRE_AT it sprays once, and must be let out past REARM_AT before it sprays again.
- a folded bar towel lying flat at TOWEL_C.

What task code stands in for (PhysX has no adhesion and no liquids): a field of FN x FN fine cells (FINE, 1 cm)
over the spill, each holding syrup m (1.0: the dried puddle's middle) and water w.
- Spray: a squeeze past FIRE_AT wets the fine cells in the spray's patch where the nozzle's line meets the table
  (within SPRAY_RANGE), most in the middle. Water softens syrup: softness = w / (W_SAT m), from 0 (dried, tacky) to 1
  (dissolved). A bottle tipped past level sprays nothing (the dip tube draws air). Where the towel lies in the patch it
  takes the water instead.
- Wipe: where a fingertip presses the towel onto the table (within PRESS_Z, cloth under it) and moves, the cloth
  under it moves a distance r (the smaller of the two movements) over the fine cells within WIPE_R. On each cell the
  cloth picks up syrup (rate KP: little when dried, a lot when soft; a thin film comes off a damp cloth easily) and
  drags some along onto the next cells in the wipe's direction (rate KD: a lot when dried, which is a smear; a little
  when soft), with the water that thins it. The towel holds what it picks up; the fuller it is, the less it picks up
  and the more film it leaves behind (K_DEP). It soaks up water, and a damp towel wets the cells it rubs.
- Bare fingers on syrup (any fingertip within FOUL_Z of the table over a fine cell with CLEAN_M of syrup, no towel
  under it) foul the gripper: the trial scores 0.
- The score reads the field on a GRID_N x GRID_N grid of 2 cm squares (each 2 x 2 fine cells): a square is clean when
  its mean syrup is under CLEAN_M (a dry tissue pressed on it picks up nothing). PUDDLE marks the 19 squares the
  spill starts on.

Score (task_score): 0 if the gripper was fouled, or the bottle or the towel left the table; else 0.25 / 0.5 when a
quarter / half of the PUDDLE cells are clean; 0.75 when all are clean; 1.0 when every cell of the grid is clean (no
smear left), the bottle stands upright on the table, both grippers are 2 cm clear of the bottle and the towel, and
nothing has moved more than STILL_MM in the last STILL_S.

Layout (round 3): the spill sits 8 cm further to the robot's left than in rounds 1-2, so the wiping (left) arm never
reaches across to its right, where the bottle stands (round 2: the arm knocked the bottle over; with the spill here its
hulls clear the bottle by 44 mm over the scrubs, offline); the towel lies turned, 18 x 14 cm, behind and left of it.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

CONTROL_HZ = 20
DT = 1.0 / CONTROL_HZ
MM = 1e-3
TOP = 0.7475  # the station's table top (yam_arena.task_file.TABLE_TOP["station"]); task.py asserts it

# -- the spill ---------------------------------------------------------------------------------------------------------
CELL = 0.02
GRID_N = 13  # squares: 26 cm a side, room for smears round the 10 cm spill
SPILL_C = (0.37, 0.12)
# The puddle: an irregular blob about 10 x 10 cm in the middle 5 x 5 cells (row i along x from the robot outward,
# column j along y from the robot's right to its left).
PUDDLE_ROWS = (".XXX.", "XXXXX", "XXXXX", ".XXXX", "..XX.")
PUDDLE = frozenset((i + 4, j + 4) for i, row in enumerate(PUDDLE_ROWS) for j, c in enumerate(row) if c == "X")
N_PUDDLE = len(PUDDLE)
SYRUP_T = 0.0015  # the visual layer's thickness (m)
CLEAN, STICKY, WET = 0, 1, 2

# -- rules ---------------------------------------------------------------------------------------------------------------
FIRE_AT = 0.0085  # trigger travel (m) that sprays once
REARM_AT = 0.005  # ... and that it must come back past before the next spray
SPRAY_RANGE = (0.06, 0.40)  # nozzle to table along the spray's line (m)
SPRAY_DOWN_MIN = 0.25  # the spray's line must point at least this far down (its unit vector's -z)
SPRAY_R0, SPRAY_R_PER_M = 0.010, 0.10  # patch radius = R0 + range * R_PER_M (about 2.3 cm at 13 cm)
# The smear model (see the docstring). Rates are per metre of pressed-cloth travel over a cell.
FINE = 0.01
FN = 2 * GRID_N
CLEAN_M = 0.04  # a square with less syrup than this (mean, in units of the dried puddle) is clean
W_SAT = 0.5  # water per unit of syrup that softens it fully
SPRAY_W = 1.0  # water one spray leaves in its patch's middle
W_MAX = 3.0
KP_DRY, KP_WET = 1.0, 21.0  # pick-up: a soft cell is clean after about 15 cm (2.5 passes); a dried one barely gives
KD_DRY, KD_WET = 25.0, 4.0  # drag: dried syrup smears along; dissolved syrup mostly comes away with the cloth
KP_FILM, FILM_M = 25.0, 0.15  # a thin film comes off a damp cloth easily
K_ABS, K_GIVE = 3.0, 2.0  # the cloth soaks up water; a damp one gives some to what it rubs
TOWEL_WATER_SAT, TOWEL_DAMP = 20.0, 0.5
K_DEP = 1.0  # film a full towel leaves behind
LOAD_CAP = 12.0 * 19  # syrup a towel can hold: three spills' worth
OFF_GRID_MAX = 0.2  # syrup smeared off the tracked field beyond this counts as one more dirty square
TABLE_TINT = (0.80, 0.68, 0.47)  # the oak table, for blending a thin film's colour
SYRUP_DARK, SYRUP_THIN = (0.28, 0.12, 0.03), (0.62, 0.38, 0.12)
MIST_S, MIST_LEN, MIST_R = 0.4, 0.11, 0.028  # the spray's visible mist: how long it shows, its cone's size (m)
RUB_STEP_MAX = 0.02  # never credit more than this per control step (a cloth jump is not a wipe)
PRESS_Z = 0.0055  # a fingertip this close to the table top (m) presses the 4 mm towel under it (round 1: pressing
# fingertips sat 2.6 to 3.4 mm above the table, lifted ones 7 mm)
UNDER_R = 0.015  # towel nodes within this (x, y) of a fingertip and below it: the fingertip is on the towel
PAD_R = 0.020  # towel nodes within this of a pressing fingertip are the pressed cloth
NODE_LOW = 0.012  # pressed cloth: nodes within this of the table top
WIPE_R = 0.022  # cells whose centre lies within this of the pressed cloth's centre are rubbed
FOUL_Z = 0.006  # a bare fingertip this close to the table over syrup fouls the gripper
CLEAR_GAP = 0.02  # let go: every gripper point this far from the bottle and the towel
STILL_S, STILL_MM = 1.0, 5.0
UPRIGHT_DEG = 15.0
TABLE_X, TABLE_Y = (-0.18, 0.90), (-0.64, 0.64)  # the station table's top, 1 cm in from its edges (task frame)

# -- the bottle (its frame: origin at the base's centre, x the nozzle's direction, z up) ----------------------------------
BOTTLE_START = (0.26, -0.10, TOP)
BOTTLE_MASS = 0.27  # 70 g bottle + 200 mL of water
BOTTLE_COM = (0.0, 0.0, 0.05)
BODY_R, BODY_H = 0.034, 0.13
# colliders (name, size, centre) in the bottle frame
SHROUD = ("shroud", (0.036, 0.026, 0.060), (-0.006, 0.0, 0.160))  # x -0.024..0.012, z 0.13..0.19
HEAD = ("head", (0.076, 0.030, 0.030), (0.014, 0.0, 0.205))  # x -0.024..0.052, z 0.19..0.22
NOZZLE = (0.057, 0.0, 0.207)  # the nozzle's tip
SPRAY_DIR = (1.0, 0.0, 0.0)
TRIGGER = (0.010, 0.018, 0.040)  # its box
TRIGGER_REST = (0.040, 0.0, 0.165)  # its centre at rest: front face x 0.045, z 0.145..0.185
TRIGGER_TRAVEL = 0.010  # it slides back (-x) this far
TRIGGER_RATE = 3000.0  # N/m
TRIGGER_DAMPING = 5.0
BACK_X = SHROUD[2][0] - SHROUD[1][0] / 2  # -0.024: where the rear finger presses
FRONT_X = TRIGGER_REST[0] + TRIGGER[0] / 2  # 0.045: the trigger's face at rest
GRIP_Z = TRIGGER_REST[2]  # the fingers close at the trigger's middle height
GRIP_X = (BACK_X + FRONT_X) / 2  # the TCP's x in the bottle frame (trigger out)

# -- the towel ------------------------------------------------------------------------------------------------------------
TOWEL_C = (0.23, 0.28)
TOWEL_SIZE = (0.18, 0.14, 0.004)  # a cotton bar towel folded once: 18 x 14 cm, two layers
TOWEL_MASS = 0.040
TOWEL_EDGE = 0.01

# -- fingers (gap between the pads, mm) --------------------------------------------------------------------------------------
FINGER_GAP = {"open": 93.9, "closed": 0.0, "wide": 85.0,
              # the gap the fingers are told to close to on the bottle (the trigger's spring and the finger drive
              # settle between); "carry" leaves the trigger well short of REARM_AT, "squeeze" bottoms it.
              "carry": 62.0, "squeeze": 45.0}


def cell_centre(i: int, j: int) -> np.ndarray:
    return np.array([SPILL_C[0] + (i - (GRID_N - 1) / 2) * CELL, SPILL_C[1] + (j - (GRID_N - 1) / 2) * CELL])


CENTRES = np.array([[cell_centre(i, j) for j in range(GRID_N)] for i in range(GRID_N)])  # (N, N, 2)


def fine_centre(a: int, b: int) -> np.ndarray:
    return np.array([SPILL_C[0] + (a - (FN - 1) / 2) * FINE, SPILL_C[1] + (b - (FN - 1) / 2) * FINE])


FINE_CENTRES = np.array([[fine_centre(a, b) for b in range(FN)] for a in range(FN)])  # (FN, FN, 2)


def start_syrup() -> np.ndarray:
    """The dried puddle: 1.0 on its squares' fine cells, 0.7 on those at its edge (a puddle thins at its rim)."""
    m = np.zeros((FN, FN))
    for i, j in PUDDLE:
        m[2 * i:2 * i + 2, 2 * j:2 * j + 2] = 1.0
    inside = m > 0
    edge = inside.copy()
    edge[1:-1, 1:-1] = inside[1:-1, 1:-1] & ~(inside[:-2, 1:-1] & inside[2:, 1:-1] & inside[1:-1, :-2] & inside[1:-1, 2:])
    m[edge] = 0.7
    return m


# -- geometry -------------------------------------------------------------------------------------------------------------
def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tilt_deg(q) -> float:
    """How far the bottle's z axis is from vertical."""
    return math.degrees(math.acos(max(-1.0, min(1.0, quat_xyzw_to_matrix(q)[2, 2]))))


def spray_patch(pos, quat) -> tuple[np.ndarray, float, float] | None:
    """(hit point x, y on the table; patch radius; range) for a spray from the bottle at (pos, quat), or None if the
    spray's line doesn't meet the table within SPRAY_RANGE."""
    R = quat_xyzw_to_matrix(quat)
    p = np.asarray(pos, float) + R @ np.asarray(NOZZLE, float)
    d = R @ np.asarray(SPRAY_DIR, float)
    if -d[2] < SPRAY_DOWN_MIN:
        return None
    r = (p[2] - (TOP + SYRUP_T)) / -d[2]
    if not SPRAY_RANGE[0] <= r <= SPRAY_RANGE[1]:
        return None
    hit = p + r * d
    return hit[:2], SPRAY_R0 + SPRAY_R_PER_M * r, r


def rx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def ry(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def matrix_to_quat_xyzw(R) -> np.ndarray:
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w])


# Gripper rotations: columns are the gripper's x (the fingers' closing axis), y and z (pointing) in the task frame.
def grip_rot(approach, closing) -> np.ndarray:
    a = np.asarray(approach, float) / np.linalg.norm(approach)
    c = np.asarray(closing, float) / np.linalg.norm(closing)
    return np.column_stack([c, np.cross(a, c), a])


# The right hand holds the bottle from its right side: pointing left (+y), level, the fingers closing along the nozzle
# (one on the trigger's face, one on the back of the shroud). Tipping the nozzle down is a turn of the hand about its
# own pointing axis (ry), which these arms reach near the middle of the table (yam_arena IK, offline).
SIDE_GRIP = grip_rot((0.0, 1.0, 0.0), (1.0, 0.0, 0.0))


def bottle_rot_of_hand(hand_rot) -> np.ndarray:
    """The bottle's rotation when the hand that holds it is at ``hand_rot`` (the grip fixes one to the other)."""
    return np.asarray(hand_rot, float) @ SIDE_GRIP.T


def hand_for_bottle(bottle_pos, bottle_R=np.eye(3), squeezed: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(TCP, rotation) of the right hand holding the bottle at (pos, R) with the trigger ``squeezed`` in (m)."""
    R = np.asarray(bottle_R, float)
    grip = np.array([(BACK_X + FRONT_X - squeezed) / 2, 0.0, GRIP_Z])
    return np.asarray(bottle_pos, float) + R @ grip, R @ SIDE_GRIP


def aim_hand(target_xy, tilt_deg_: float, height: float) -> tuple[np.ndarray, np.ndarray, float]:
    """(TCP, hand rotation, range) of the right hand that sprays at ``target_xy`` with the nozzle tipped ``tilt_deg_``
    down and the TCP ``height`` above the table (trigger out)."""
    R = ry(tilt_deg_)
    grip = np.array([GRIP_X, 0.0, GRIP_Z])
    nozzle_from_tcp = R @ (np.asarray(NOZZLE, float) - grip)
    d = R @ np.asarray(SPRAY_DIR, float)
    tcp_z = TOP + height
    p_z = tcp_z + nozzle_from_tcp[2]
    r = (p_z - (TOP + SYRUP_T)) / -d[2]
    hit_from_tcp = nozzle_from_tcp[:2] + r * d[:2]
    tcp = np.array([target_xy[0] - hit_from_tcp[0], target_xy[1] - hit_from_tcp[1], tcp_z])
    return tcp, R @ SIDE_GRIP, r


# The left hand wipes as the napkin task's hands pull cloth: fingers closed side by side (closing along y), leaning
# LEAN forward, fingertips pressed into the towel.
LEAN = 30.0
_A = np.array([math.sin(math.radians(LEAN)), 0.0, -math.cos(math.radians(LEAN))])
PRESS_ROT = grip_rot(_A, (0.0, -1.0, 0.0))
PAD_TIP_PAST_TCP = 0.013  # yam_arena.constants.PAD_TIP_PAST_TCP
PRESS_DEPTH = 0.016  # the fingertips are aimed this far below the towel's top (the arm gives about 8 mm)


def press_tcp(x: float, y: float, depth: float = PRESS_DEPTH) -> np.ndarray:
    """The left TCP that puts the fingertips ``depth`` below the towel's top at (x, y)."""
    tip = np.array([x, y, TOP + TOWEL_SIZE[2] - depth])
    return tip - PAD_TIP_PAST_TCP * _A


def hover_tcp(x: float, y: float, up: float = 0.05) -> np.ndarray:
    return press_tcp(x, y, -up)


# -- the per-step tracker ------------------------------------------------------------------------------------------------
@dataclass
class Inputs:
    bottle_pos: np.ndarray
    bottle_quat: np.ndarray
    trigger: float  # travel (m, 0 out)
    nodes: np.ndarray | None  # the towel's nodes (n, 3)
    tips: list  # fingertip points (each finger's pad tip, both arms, in a fixed order), task frame
    grippers: list  # every gripper point (TCPs, pads, tips), for the let-go check
    dt: float = DT  # this control step's length (s): the environment's step_dt


@dataclass
class Facts:
    fouled: bool = False
    bottle_on_table: bool = True
    bottle_upright: bool = True
    towel_on_table: bool = True
    puddle_clean: int = 0
    dirty: int = N_PUDDLE  # squares not clean anywhere in the grid
    smeared: int = 0  # squares that were clean at the start and have syrup now
    hands_clear: bool = False
    still: bool = False


def softness(m: np.ndarray, w: np.ndarray) -> np.ndarray:
    """0 (dried, tacky) to 1 (dissolved): the water on a cell against what it takes to soften its syrup. A cell with
    no syrup to speak of counts as soft only if it is wet."""
    return np.where(m > 1e-6, np.clip(w / (W_SAT * np.maximum(m, 1e-9)), 0.0, 1.0), (w > 1e-6).astype(float))


@dataclass
class Tracker:
    m: np.ndarray = field(default_factory=start_syrup)  # syrup on each fine cell (1.0: the dried puddle's middle)
    w: np.ndarray = field(default_factory=lambda: np.zeros((FN, FN)))  # water on each fine cell (one spray's middle: 1)
    load: float = 0.0  # syrup the towel has taken up (cell units)
    towel_water: float = 0.0  # water the towel holds (cell units)
    off_grid: float = 0.0  # syrup smeared off the tracked field (counted as left on the table: never cleaned)
    armed: bool = True
    sprays: int = 0
    wet_hits: int = 0
    prev_nodes: np.ndarray | None = None
    hist: deque = field(default_factory=deque)  # (t, bottle position, towel middle) over the last STILL_S
    prev_tips: list | None = None
    t: float = 0.0
    towel_hits: int = 0
    facts: Facts = field(default_factory=Facts)
    steps: int = 0
    rubbed_total: float = 0.0
    log: dict = field(default_factory=dict)
    shown: np.ndarray | None = None  # square states as last drawn (the patch look)
    field_changed: bool = True  # the fine field changed since the field look was last drawn
    fired_step: int = -10**6  # the control step of the last spray
    mist_shown: bool = False

    # -- what the field means for the 2 cm squares (the score and the patch look) ----------------------------------------
    def square_m(self) -> np.ndarray:
        """Each square's mean syrup (GRID_N, GRID_N)."""
        return self.m.reshape(GRID_N, 2, GRID_N, 2).mean(axis=(1, 3))

    @property
    def cells(self) -> np.ndarray:
        """Each square's state: CLEAN when its mean syrup is under CLEAN_M, else WET when its syrup is soft (half or
        more), else STICKY."""
        sm = self.square_m()
        soft = softness(self.m, self.w).reshape(GRID_N, 2, GRID_N, 2).mean(axis=(1, 3))
        return np.where(sm < CLEAN_M, CLEAN, np.where(soft >= 0.5, WET, STICKY))

    def mist_on(self) -> bool:
        return self.steps - self.fired_step < int(round(MIST_S * CONTROL_HZ))

    def damp(self) -> bool:
        return self.towel_water > TOWEL_DAMP

    def step(self, x: Inputs) -> list:
        """One control step. Returns the patch look's changes [(part name, show)] (the field look reads
        ``field_look()`` when ``field_changed``)."""
        self.steps += 1
        self.t += float(x.dt)
        f = self.facts
        nodes = None if x.nodes is None else np.asarray(x.nodes, float)
        self._spray(x, nodes)
        tips = [np.asarray(p, float) for p in x.tips]
        if nodes is not None and self.prev_nodes is not None and len(nodes) == len(self.prev_nodes) and \
                self.prev_tips is not None and len(self.prev_tips) == len(tips):
            self._rub(nodes, self.prev_nodes, tips, self.prev_tips)
        self._foul(nodes, tips)
        self.prev_nodes = None if nodes is None else nodes.copy()  # a copy: the caller may move its array
        self.prev_tips = [p.copy() for p in tips]
        # outcome facts
        bp = np.asarray(x.bottle_pos, float)
        f.bottle_on_table = bool(TABLE_X[0] <= bp[0] <= TABLE_X[1] and TABLE_Y[0] <= bp[1] <= TABLE_Y[1]
                                 and bp[2] > TOP - 0.03)
        f.bottle_upright = bool(tilt_deg(x.bottle_quat) <= UPRIGHT_DEG and abs(bp[2] - TOP) < 0.015)
        if nodes is not None and len(nodes):
            c = nodes.mean(0)
            f.towel_on_table = bool(nodes[:, 2].min() > TOP - 0.03 and TABLE_X[0] <= c[0] <= TABLE_X[1]
                                    and TABLE_Y[0] <= c[1] <= TABLE_Y[1])
        cells = self.cells
        f.puddle_clean = int(sum(cells[i, j] == CLEAN for i, j in PUDDLE))
        f.dirty = int((cells != CLEAN).sum()) + (1 if self.off_grid > OFF_GRID_MAX else 0)
        f.smeared = int(sum(1 for i in range(GRID_N) for j in range(GRID_N)
                            if (i, j) not in PUDDLE and cells[i, j] != CLEAN))
        f.hands_clear = self._clear(bp, x.bottle_quat, nodes, x.grippers)
        towel_c = nodes.mean(0) if nodes is not None and len(nodes) else np.zeros(3)
        self.hist.append((self.t, bp.copy(), towel_c.copy()))
        while len(self.hist) > 1 and self.hist[1][0] <= self.t - STILL_S + 1e-9:
            self.hist.popleft()
        t0, b0, c0 = self.hist[0]
        if self.t - t0 >= STILL_S - 1e-9:
            moved = max(float(np.linalg.norm(bp - b0)), float(np.linalg.norm(towel_c - c0)))
            f.still = moved * 1000 < STILL_MM
        else:
            f.still = False
        self.log.update(t=round(self.t, 2), sprays=self.sprays, wet=int((cells == WET).sum()),
                        sticky=int((cells == STICKY).sum()), puddle_clean=f.puddle_clean, dirty=f.dirty,
                        smeared=f.smeared, syrup_left=round(float(self.m.sum()), 2), towel_load=round(self.load, 2),
                        towel_water=round(self.towel_water, 2), damp=self.damp(), towel_hits=self.towel_hits,
                        off_grid=round(self.off_grid, 3), rubbed_mm=round(1000 * self.rubbed_total, 1),
                        trigger_mm=round(1000 * float(x.trigger), 2), fouled=f.fouled,
                        bottle_upright=f.bottle_upright, towel_on_table=f.towel_on_table,
                        hands_clear=f.hands_clear, still=f.still)
        return self.visual_changes()

    # -- the spray ---------------------------------------------------------------------------------------------------
    def _spray(self, x: Inputs, nodes) -> None:
        trig = float(x.trigger)
        if not self.armed:
            if trig <= REARM_AT:
                self.armed = True
            return
        if trig < FIRE_AT:
            return
        self.armed = False
        self.sprays += 1
        self.fired_step = self.steps
        hit = spray_patch(x.bottle_pos, x.bottle_quat)
        if hit is None or quat_xyzw_to_matrix(x.bottle_quat)[2, 2] <= 0.0:  # tipped past level: the dip tube draws air
            self.log["last_spray"] = {"hit_mm": None}
            return
        (hx, hy), rad, rng = hit
        d = np.linalg.norm(FINE_CENTRES - np.array([hx, hy]), axis=2)
        g = np.where(d <= rad, np.exp(-0.5 * (d / (rad / 1.5)) ** 2), 0.0)  # the patch: most water in its middle
        covered = np.zeros_like(g, dtype=bool)
        on_towel = 0.0
        if nodes is not None and len(nodes):
            for a, b in zip(*np.nonzero(g > 0)):
                covered[a, b] = bool((np.linalg.norm(nodes[:, :2] - FINE_CENTRES[a, b], axis=1) <= 0.008).any())
            on_towel = float(g[covered].sum())
        if on_towel > 0:  # the towel lying in the patch takes that water
            self.towel_water += SPRAY_W * on_towel
            self.towel_hits += 1
        add = SPRAY_W * np.where(covered, 0.0, g)
        self.w = np.minimum(self.w + add, W_MAX)
        self.wet_hits += int(((add > 0.2) & (self.m >= CLEAN_M)).sum())
        self.field_changed = True
        self.log["last_spray"] = {"hit_mm": [round(1000 * hx, 1), round(1000 * hy, 1)],
                                  "r_mm": round(1000 * rad, 1), "range_mm": round(1000 * rng, 1),
                                  "towel": on_towel > 0}

    # -- the wipe ----------------------------------------------------------------------------------------------------
    def _rub(self, nodes, prev, tips, prev_tips) -> None:
        for tip, tip0 in zip(tips, prev_tips):
            if tip[2] > TOP + PRESS_Z:
                continue
            tip_move = float(np.linalg.norm(tip[:2] - tip0[:2]))
            dxy = np.linalg.norm(nodes[:, :2] - tip[:2], axis=1)
            if not ((dxy <= UNDER_R) & (nodes[:, 2] < tip[2])).any():
                continue
            near = (dxy <= PAD_R) & (nodes[:, 2] < TOP + NODE_LOW)
            if not near.any():
                continue
            move = (nodes[near, :2] - prev[near, :2]).mean(0)
            r = min(float(np.linalg.norm(move)), tip_move, RUB_STEP_MAX)  # cloth that trembles under a still hand
            if r < 1e-4:                                                  # isn't a wipe
                continue
            self.rubbed_total += r
            centre = nodes[near, :2].mean(0)
            self._wipe(np.linalg.norm(FINE_CENTRES - centre, axis=2) <= WIPE_R, r, move / (np.linalg.norm(move) + 1e-12))

    def _wipe(self, foot: np.ndarray, r: float, u: np.ndarray) -> None:
        """The pressed towel moves ``r`` (m) along ``u`` over the fine cells in ``foot``."""
        idx = np.nonzero(foot)
        if not len(idx[0]):
            return
        m, w = self.m[idx], self.w[idx]
        soft = softness(m, w)
        damp = (w > 1e-6) | self.damp()
        kp = KP_DRY + (KP_WET - KP_DRY) * soft + np.where(damp & (m < FILM_M), KP_FILM * (1.0 - m / FILM_M), 0.0)
        kd = KD_DRY + (KD_WET - KD_DRY) * soft
        fill = min(1.0, self.load / LOAD_CAP)
        pick = m * (1.0 - np.exp(-kp * r)) * (1.0 - fill)
        drag = (m - pick) * (1.0 - np.exp(-kd * r))
        wdrag = np.where(m > 1e-9, w * drag / np.maximum(m, 1e-9), 0.0)  # water goes with the syrup it thins
        dep = np.full_like(m, K_DEP * fill * r)  # a loaded towel leaves a film where it passes
        dep = np.minimum(dep, self.load / len(m)) if self.load > 0 else dep * 0.0
        absorb = (w - wdrag) * (1.0 - np.exp(-K_ABS * r))  # the cloth soaks up water...
        give = np.full_like(m, K_GIVE * r * min(1.0, self.towel_water / TOWEL_WATER_SAT))  # ...and a damp one gives some
        self.m[idx] = m - pick - drag + dep
        self.w[idx] = np.minimum(w - wdrag - absorb + give, W_MAX)
        self.load += float(pick.sum() - dep.sum())
        self.towel_water = max(0.0, self.towel_water + float(absorb.sum() - give.sum()))
        # what the cloth drags lands on the next cells along the wipe (split between the x and the y neighbour)
        ax, ay = abs(float(u[0])), abs(float(u[1]))
        sx, sy = (1 if u[0] >= 0 else -1), (1 if u[1] >= 0 else -1)
        for frac, da, db in ((ax / (ax + ay + 1e-12), sx, 0), (ay / (ax + ay + 1e-12), 0, sy)):
            if frac <= 0:
                continue
            a, b = idx[0] + da, idx[1] + db
            ok = (a >= 0) & (a < FN) & (b >= 0) & (b < FN)
            np.add.at(self.m, (a[ok], b[ok]), frac * drag[ok])
            np.add.at(self.w, (a[ok], b[ok]), frac * wdrag[ok])
            self.off_grid += float(frac * drag[~ok].sum())
        self.m[self.m < 1e-6] = 0.0
        self.field_changed = True

    def _foul(self, nodes, tips) -> None:
        for tip in tips:
            if tip[2] > TOP + FOUL_Z:
                continue
            k = np.floor((tip[:2] - (np.asarray(SPILL_C) - FN * FINE / 2)) / FINE).astype(int)
            if not (0 <= k[0] < FN and 0 <= k[1] < FN) or self.m[k[0], k[1]] < CLEAN_M:
                continue
            if nodes is not None and len(nodes):
                dxy = np.linalg.norm(nodes[:, :2] - tip[:2], axis=1)
                if ((dxy <= UNDER_R) & (nodes[:, 2] < tip[2] + 0.002)).any():
                    continue
            self.facts.fouled = True

    def _clear(self, bp, bq, nodes, grippers) -> bool:
        if not grippers:
            return False
        pts = np.asarray(grippers, float).reshape(-1, 3)
        R = quat_xyzw_to_matrix(bq)
        local = (pts - bp) @ R  # into the bottle frame
        # the bottle as boxes: body, shroud, head, trigger (with CLEAR_GAP all round)
        boxes = [((0.0, 0.0, BODY_H / 2), (2 * BODY_R, 2 * BODY_R, BODY_H)), (SHROUD[2], SHROUD[1]), (HEAD[2], HEAD[1]),
                 (TRIGGER_REST, TRIGGER)]
        for c, s in boxes:
            if np.any(np.all(np.abs(local - np.asarray(c)) <= np.asarray(s) / 2 + CLEAR_GAP, axis=1)):
                return False
        if nodes is not None and len(nodes):
            d = np.linalg.norm(pts[:, None, :] - nodes[None, :, :], axis=2)
            if d.min() < CLEAR_GAP:
                return False
        return True

    # -- the looks ---------------------------------------------------------------------------------------------------
    def dirty_centres(self) -> list:
        """(x, y) of every square that still shows syrup: what the cameras see."""
        return [tuple(CENTRES[i, j]) for i, j in zip(*np.nonzero(self.cells != CLEAN))]

    def field_look(self) -> tuple[np.ndarray, np.ndarray]:
        """(shown (FN, FN) bool, colour (FN, FN, 3)) for the field look: a fine cell shows when it has CLEAN_M of
        syrup or more; dark amber where thick, a lighter film where thin, lighter and warmer where dissolved."""
        self.field_changed = False
        thick = np.clip(self.m / 1.0, 0.0, 1.0) ** 0.6
        soft = softness(self.m, self.w)[..., None]
        dry = TABLE_TINT + (np.asarray(SYRUP_DARK) - TABLE_TINT) * thick[..., None]
        wet = TABLE_TINT + (np.asarray(SYRUP_THIN) - TABLE_TINT) * thick[..., None]
        return self.m >= CLEAN_M, dry + (wet - dry) * soft

    def visual_changes(self) -> list:
        """[(part, show)] for every square patch whose look must change (the patch look): s_i_j shows a sticky
        square, w_i_j a wet one."""
        out = []
        prev = self.shown
        now_cells = self.cells
        for i in range(GRID_N):
            for j in range(GRID_N):
                now = now_cells[i, j]
                was = None if prev is None else prev[i, j]
                if was == now:
                    continue
                out.append((f"s_{i}_{j}", bool(now == STICKY)))
                out.append((f"w_{i}_{j}", bool(now == WET)))
        self.shown = now_cells.copy()
        return out


def start_cells() -> np.ndarray:
    """The squares' states at the start (the patch look's first frame)."""
    return Tracker().cells


def task_score(f: Facts) -> float:
    """0 when the gripper touched the syrup bare, or the bottle or the towel is off the table, or under a quarter of the
    puddle is clean; 0.25 a quarter clean; 0.5 half; 0.75 all of it; 1.0 all of it and no syrup anywhere else (smears
    cleaned too), the bottle standing upright on the table, both grippers let go (2 cm from the bottle and the towel)
    and everything still for the last second."""
    if f.fouled or not f.bottle_on_table or not f.towel_on_table:
        return 0.0
    share = f.puddle_clean / N_PUDDLE
    if share >= 1.0:
        if f.dirty == 0 and f.bottle_upright and f.hands_clear and f.still:
            return 1.0
        return 0.75
    if share >= 0.5:
        return 0.5
    if share >= 0.25:
        return 0.25
    return 0.0


# -- the scripted demo ----------------------------------------------------------------------------------------------------
# A step is (seconds, TCP or None for the ready pose, finger name, rotation (3x3) or "ready", label).
SPRAY_TILT = 55.0
SPRAY_HEIGHT = 0.15  # the TCP above the table while spraying (lower and the tipped bottle's base nears the table)
LIFT = 0.05


# The open fingers come straight down over the sprayer's head (they span 93.9 mm; the head and nozzle 81 mm), with the
# TCP DESCEND_DX ahead of its grip x so each finger clears an end by 6.5 mm, then slide back to the grip. (Coming in
# from the bottle's side would need the hand 8 cm further right, where it can't point left: yam_arena IK, offline.)
DESCEND_DX = ((NOZZLE[0] + 0.005) + (BACK_X - 0.005)) / 2 - GRIP_X
ABOVE = 0.08  # the TCP this far over the grip before coming down


def take_bottle(b=None) -> list:
    """Steps to take the bottle at its start: over the head, down, onto the grip, close lightly."""
    b = np.asarray(BOTTLE_START if b is None else b, float)
    grip_p, grip_R = hand_for_bottle(b)
    descend = grip_p + np.array([DESCEND_DX, 0.0, 0.0])
    return [(2.5, descend + np.array([0.0, 0.0, ABOVE]), "open", grip_R, "over the sprayer's head"),
            (1.5, descend, "open", grip_R, "down either side of the head"),
            (0.8, grip_p, "open", grip_R, "back: fingers at the trigger and the shroud"),
            (1.0, grip_p, "carry", grip_R, "close on the trigger lightly")]


def leave_bottle(b=None) -> list:
    """Steps after the bottle stands on its mark: let go, off the grip, up, back to ready."""
    b = np.asarray(BOTTLE_START if b is None else b, float)
    grip_p, grip_R = hand_for_bottle(b)
    descend = grip_p + np.array([DESCEND_DX, 0.0, 0.002])
    return [(0.8, grip_p + np.array([0.0, 0.0, 0.002]), "open", grip_R, "let go"),
            (0.6, descend, "open", grip_R, "fingers off the trigger"),
            (1.2, descend + np.array([0.0, 0.0, ABOVE]), "open", grip_R, "up over the head"),
            (2.0, None, "open", "ready", "back to ready")]


def spray_targets() -> list:
    """Where the sprays aim: a 3 x 3 pattern over the puddle, 3 cm apart, far row first."""
    cx, cy = SPILL_C
    return [(cx + dx, cy + dy) for dx in (0.03, 0.0, -0.03) for dy in (0.03, 0.0, -0.03)]


def plan_spray(targets=None, squeeze: bool = True, put_back: bool = True) -> list:
    """The right hand: take the bottle from its right side, lift it, spray each target (aim, squeeze, let the trigger
    out), turn it upright, stand it back on its mark, let go, back to ready."""
    targets = spray_targets() if targets is None else targets
    b = np.asarray(BOTTLE_START, float)
    grip_p, grip_R = hand_for_bottle(b)
    up = np.array([0.0, 0.0, LIFT])
    steps = take_bottle() + [(1.0, grip_p + up, "carry", grip_R, "lift the bottle")]
    first = True
    for t in targets:
        tcp, R, _ = aim_hand(t, SPRAY_TILT, SPRAY_HEIGHT)
        steps.append((2.5 if first else 0.8, tcp, "carry", R, "aim at the spill"))
        first = False
        if squeeze:
            steps += [(0.5, tcp, "squeeze", R, "squeeze the trigger"), (0.5, tcp, "carry", R, "let the trigger out")]
        else:
            steps.append((1.0, tcp, "carry", R, "hold"))
    if put_back:
        steps += [
            (2.5, grip_p + up, "carry", grip_R, "upright, back over its mark"),
            (1.0, grip_p + np.array([0.0, 0.0, 0.002]), "carry", grip_R, "stand it on the table"),
        ] + leave_bottle()
    return steps


def _scrub_points(pattern: str) -> list:
    """Fingertip (x, y) points of a scrub over the puddle and one cell round it. The rub that clears a cell is about
    6 cm of pressed-cloth travel over it."""
    cx, cy = SPILL_C
    half = 0.065
    if pattern == "strokes":  # back and forth across, rows 2.5 cm apart, far row first, twice
        pts = []
        rows = np.arange(half - 0.01, -half, -0.025)
        for rep in range(2):
            for k, dx in enumerate(rows):
                ys = (cy - half, cy + half) if (k + rep) % 2 == 0 else (cy + half, cy - half)
                pts += [(cx + dx, ys[0]), (cx + dx, ys[1])]
        return pts
    if pattern == "outside_in":  # strokes from the edge toward the middle, round all four sides, twice
        pts = []
        for rep in range(2):
            for off in np.arange(-half + 0.01, half, 0.025):
                pts += [(cx + half, cy + off), (cx, cy + off)]
                pts += [(cx - half, cy + off), (cx, cy + off)]
                pts += [(cx + off, cy + half), (cx + off, cy)]
                pts += [(cx + off, cy - half), (cx + off, cy)]
        return pts
    if pattern == "circles":  # small circles (2.5 cm) marching along rows
        pts = []
        for dx in np.arange(half - 0.015, -half, -0.03):
            for cyy in np.arange(cy - half + 0.015, cy + half, 0.03):
                for a in np.linspace(0, 2 * math.pi, 7)[:-1]:
                    pts.append((cx + dx + 0.025 * math.cos(a), cyy + 0.025 * math.sin(a)))
        return pts + pts[: len(pts) // 2]
    if pattern == "zigzag":  # one continuous serpentine along x, then one along y
        pts = []
        for k, dy in enumerate(np.arange(-half, half + 1e-9, 0.02)):
            xs = (cx - half, cx + half) if k % 2 == 0 else (cx + half, cx - half)
            pts += [(xs[0], cy + dy), (xs[1], cy + dy)]
        for k, dx in enumerate(np.arange(half, -half - 1e-9, -0.02)):
            ys = (cy - half, cy + half) if k % 2 == 0 else (cy + half, cy - half)
            pts += [(cx + dx, ys[0]), (cx + dx, ys[1])]
        return pts
    raise ValueError(f"unknown scrub pattern {pattern}")


SCRUB_SPEED = 0.10  # m/s of the fingertips while pressed


def plan_wipe(pattern: str = "strokes", wait: float = 0.0, towel: bool = True) -> list:
    """The left hand: press the towel near its edge nearest the spill, drag it over the spill, scrub ``pattern``,
    lift off (the towel stays), back to ready. ``towel=False`` scrubs with bare fingertips (a cheat check)."""
    steps = [] if wait <= 0 else [(wait, None, "closed", "ready", "wait")]
    tx, ty = TOWEL_C
    grab = (tx + 0.02, ty - TOWEL_SIZE[1] / 2 + 0.05)  # 5 cm in from the towel's edge nearest the spill
    pts = _scrub_points(pattern)
    if towel:
        steps += [(2.5, hover_tcp(*grab), "closed", PRESS_ROT, "over the towel"),
                  (1.0, press_tcp(*grab), "closed", PRESS_ROT, "press on the towel")]
        here = np.array(grab)
        # drag it so the pressed spot reaches the scrub's first point
        d = float(np.linalg.norm(np.array(pts[0]) - here))
        steps.append((max(1.0, d / 0.06), press_tcp(*pts[0]), "closed", PRESS_ROT, "drag the towel onto the spill"))
    else:
        steps += [(2.5, hover_tcp(*pts[0]), "closed", PRESS_ROT, "over the spill"),
                  (1.0, press_tcp(*pts[0], depth=PRESS_DEPTH - TOWEL_SIZE[2]), "closed", PRESS_ROT,
                   "fingertips on the table")]
    here = np.array(pts[0])
    depth = PRESS_DEPTH if towel else PRESS_DEPTH - TOWEL_SIZE[2]
    for p in pts[1:]:
        d = float(np.linalg.norm(np.array(p) - here))
        steps.append((max(0.25, d / SCRUB_SPEED), press_tcp(*p, depth=depth), "closed", PRESS_ROT, "scrub"))
        here = np.array(p)
    steps += [(0.8, hover_tcp(*here, up=0.06), "closed", PRESS_ROT, "lift off: the towel stays"),
              (2.0, None, "open", "ready", "back to ready")]
    return steps


class Wiper:
    """The closed-loop wiping hand (the probe's and the demo's left arm). It follows the scrub's fingertip path, pressed
    on the towel, at SCRUB_SPEED. The towel creeps under fingertips (it follows them only part of the way), so whenever
    the fingertips have drifted RECENTRE_D from the towel's middle it lifts, moves over the towel's middle (read from the
    towel's nodes: a scripted demo may read object states, as the accepted din-rail-mount demo does) and presses again,
    then carries on along the path from where it is. When the path is done and ``remaining()`` (the cells that still
    show syrup, as the cameras see them) is not empty, it scrubs each of those cells again with short strokes, at most
    ``touch_ups`` times. One call per control step; returns (TCP, rotation, finger gap mm, label)."""

    HOVER_V, PRESS_V, LIFT_V = 0.15, 0.04, 0.08  # m/s
    RECENTRE_D = 0.04
    DWELL_S = 0.3

    def __init__(self, path, start_tcp, towel: bool = True, max_recentres: int = 60, remaining=None,
                 touch_ups: int = 2, poke: bool = False):
        self.path = [np.asarray(p, float) for p in path]
        self.remaining, self.touch_ups, self.touched = remaining, touch_ups, 0
        self.k = 0
        self.p = np.asarray(start_tcp, float).copy()
        self.mode = "poke_go" if poke else "approach" if towel else "bare"
        self.towel = towel
        self.dwell = 0.0
        self.recentres = 0
        self.max_recentres = max_recentres
        self.target_xy = None
        self.depth = PRESS_DEPTH if towel else PRESS_DEPTH - TOWEL_SIZE[2]

    def tip(self) -> np.ndarray:
        return self.p + PAD_TIP_PAST_TCP * PRESS_ROT[:, 2]

    def _toward(self, goal, v) -> bool:
        d = np.asarray(goal, float) - self.p
        n = float(np.linalg.norm(d))
        step = v * DT
        if n <= step:
            self.p = np.asarray(goal, float).copy()
            return True
        self.p = self.p + d * (step / n)
        return False

    @property
    def done(self) -> bool:
        return self.mode == "done"

    def step(self, nodes) -> tuple[np.ndarray, np.ndarray, float, str]:
        c = None if nodes is None or not len(nodes) else np.asarray(nodes, float)[:, :2].mean(0)
        bare_depth = PRESS_DEPTH - TOWEL_SIZE[2]
        if self.mode == "poke_go":  # a bare fingertip onto the spill's middle, then up (the foul check's trial)
            if self._toward(hover_tcp(*SPILL_C, up=0.03), self.HOVER_V):
                self.mode = "poke_down"
            return self.p, PRESS_ROT, 0.0, "over the spill"
        if self.mode == "poke_down":
            if self._toward(press_tcp(*SPILL_C, depth=bare_depth), self.PRESS_V):
                self.mode = "poke_up"
            return self.p, PRESS_ROT, 0.0, "bare fingertips on the syrup"
        if self.mode == "poke_up":
            if self._toward(hover_tcp(*SPILL_C, up=0.03), self.LIFT_V):
                self.mode = "approach"
            return self.p, PRESS_ROT, 0.0, "up off the syrup"
        if self.mode == "bare":  # no towel: straight to the path's start, fingertips on the table
            if self._toward(press_tcp(*self.path[0], depth=self.depth), self.HOVER_V):
                self.mode = "scrub"
            return self.p, PRESS_ROT, 0.0, "fingertips to the spill"
        if self.mode == "approach":
            if self.target_xy is None or c is not None:
                self.target_xy = c if c is not None else np.asarray(TOWEL_C, float)
            if self._toward(hover_tcp(*self.target_xy, up=0.03), self.HOVER_V):
                self.mode = "press"
            return self.p, PRESS_ROT, 0.0, "over the towel's middle"
        if self.mode == "press":
            if self._toward(press_tcp(*self.target_xy, depth=self.depth), self.PRESS_V):
                self.dwell += DT
                if self.dwell >= self.DWELL_S:
                    self.mode, self.dwell = "scrub", 0.0
            return self.p, PRESS_ROT, 0.0, "press on the towel"
        if self.mode == "scrub":
            if self.k >= len(self.path):
                left = [] if self.remaining is None or self.touched >= self.touch_ups else                     [c for c in self.remaining() if touch_up_reachable(c)]
                if left:
                    self.touched += 1
                    for x, y in sorted(left, key=lambda c: (-c[0], c[1])):
                        self.path += [np.array([x, y - 0.025]), np.array([x, y + 0.025]), np.array([x, y - 0.025]),
                                      np.array([x, y + 0.025])]
                    return self.p, PRESS_ROT, 0.0, "touch up what still shows"
                self.mode = "lift"
                return self.p, PRESS_ROT, 0.0, "scrub"
            if self.towel and c is not None and self.recentres < self.max_recentres and \
                    float(np.linalg.norm(self.tip()[:2] - c)) > self.RECENTRE_D:
                self.mode = "recentre"
                self.recentres += 1
                return self.p, PRESS_ROT, 0.0, "lift to take the towel again"
            if self._toward(press_tcp(*self.path[self.k], depth=self.depth), SCRUB_SPEED):
                self.k += 1
            return self.p, PRESS_ROT, 0.0, "scrub"
        if self.mode == "recentre":
            if self._toward(self.p + np.array([0.0, 0.0, 0.03 - (self.p[2] - press_tcp(0, 0)[2])]), self.LIFT_V) or \
                    self.p[2] >= press_tcp(0, 0)[2] + 0.03 - 1e-4:
                self.mode, self.target_xy = "approach", None
            return self.p, PRESS_ROT, 0.0, "lift to take the towel again"
        if self.mode == "lift":
            if self._toward(hover_tcp(*self.tip()[:2], up=0.06), self.LIFT_V):
                self.mode = "done"
            return self.p, PRESS_ROT, 0.0, "lift off: the towel stays"
        return self.p, PRESS_ROT, 0.0, "done"


TOUCH_UP_R = 0.09  # the demo touches up squares within this (x and y) of the spill's middle: further out, toward the
# bottle, the wiping arm's hulls would reach it (test_task)


def touch_up_reachable(xy) -> bool:
    return abs(xy[0] - SPILL_C[0]) <= TOUCH_UP_R + 1e-9 and abs(xy[1] - SPILL_C[1]) <= TOUCH_UP_R + 1e-9


def scrub_path(pattern: str) -> list:
    return _scrub_points(pattern)


def plan_seconds(steps) -> float:
    return float(sum(s[0] for s in steps))


VARIANTS = {
    # full demos, one per scrub pattern
    "strokes": "spray the puddle, then scrub back-and-forth strokes",
    "outside_in": "spray, then strokes from the edges toward the middle",
    "circles": "spray, then small circles",
    "zigzag": "spray, then a serpentine one way and then the other",
    # what must not score
    "dry_wipe": "no spray: scrub with the towel (expect 0, and syrup smeared wider)",
    "spray_only": "spray, no wipe (expect 0)",
    "bare_fingers": "spray, then scrub with bare fingertips, no towel (expect 0: fouled)",
    "upright_squeeze": "squeeze the trigger with the bottle standing upright, then wipe (expect 0: the spray misses)",
    "poke_then_wipe": "spray, touch the syrup with bare fingertips, then wipe it all clean (expect 0: fouled)",
}


def _upright_squeezes() -> list:
    grip_p, grip_R = hand_for_bottle(np.asarray(BOTTLE_START, float))
    steps = take_bottle()
    for _ in range(4):
        steps += [(0.5, grip_p, "squeeze", grip_R, "squeeze the trigger"),
                  (0.5, grip_p, "carry", grip_R, "let the trigger out")]
    return steps + leave_bottle()


def variant(name: str) -> dict:
    """A demo variant: the right hand's open-loop steps, the left hand's scrub pattern (None: it doesn't wipe), whether
    the left wipes through the towel, and when (s) the left starts (once the last spray is done)."""
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name}")
    wait_steps = [(1.0, None, "open", "ready", "wait")]
    if name == "dry_wipe":
        return {"right": wait_steps, "pattern": "strokes", "towel": True, "poke": False, "wipe_after": 0.0}
    if name == "upright_squeeze":
        right = _upright_squeezes()
        return {"right": right, "pattern": "strokes", "towel": True, "poke": False, "wipe_after": plan_seconds(right)}
    right = plan_spray()
    if name == "spray_only":
        return {"right": right, "pattern": None, "towel": True, "poke": False, "wipe_after": 0.0}
    pattern = name if name in ("strokes", "outside_in", "circles", "zigzag") else "strokes"
    return {"right": right, "pattern": pattern, "towel": name != "bare_fingers", "poke": name == "poke_then_wipe",
            "wipe_after": plan_seconds(right[:-6]) + 0.5}


def demo_plan(name: str = "strokes") -> dict:
    """The open-loop form of a variant (the left hand's scrub as fixed waypoints, assuming the towel follows the
    fingertips): what task.demo() hands yam_scripted. The probe and the dry run use the closed-loop Wiper instead."""
    v = variant(name)
    left = [(1.0, None, "open", "ready", "wait")] if v["pattern"] is None else \
        plan_wipe(v["pattern"], wait=v["wipe_after"], towel=v["towel"])
    return {"right": v["right"], "left": left}


EXPECTED = {"strokes": 1.0, "outside_in": 1.0, "circles": 1.0, "zigzag": 1.0, "dry_wipe": 0.0, "spray_only": 0.0,
            "bare_fingers": 0.0, "upright_squeeze": 0.0, "poke_then_wipe": 0.0}
# What the probe and the dry run give each variant's Wiper: no touch-ups for the cheat checks (they would chase smears
# to the run limit and prove nothing more).
TOUCH_UPS = {"dry_wipe": 0, "upright_squeeze": 0, "bare_fingers": 0}
