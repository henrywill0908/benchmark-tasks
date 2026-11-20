"""labkit: extra object kinds, scoring helpers and demo poses for yam_arena task files.

yam_arena's task files spawn single primitives (Box, Cylinder, Sphere) and models from Arena's library. This module
adds, without changing yam_arena:

- :class:`Container`: an open-top square container (bottle, tub, tray, bin), its floor and four walls one body,
  with an optional coloured label band round its middle (visual only). Its ``pos`` is the centre of its floor.
- :class:`Marked`: a solid box with flat coloured marks on chosen faces (a crack, a label, a warning patch). Marks
  are visual only, so a mark on the back face is hidden from a camera that sees only the front.
- :class:`Model`: a 3D model from a USD file (convert STL/OBJ with tools/convert_mesh.py on the GPU machine).
- :class:`Button`: a push button, a loose heavy housing with a cap on top that presses in up to ``travel``
  against a spring (a prismatic joint with a spring drive).
- :class:`Lamp`: a status lamp's lit face, a visual-only disc that stays hidden until any button has been pressed
  ``latch_at`` deep, and goes dark again at every reset. A lamp with a ``group`` lights only when that group is the
  trial's active one (:func:`set_lamp_group`), so a task can show a different answer on different trials.
- :class:`Branch`: a demo step whose waypoints depend on which lamp group lit (what the cameras show), for an
  open-loop demo that must act on an answer it only sees mid-trial.
- :class:`Hinge`: a lid on a fixed base, on a revolute joint (``axis``, ``limits``, optional ``spring``) that starts
  at any angle; with a :class:`Latch`, its state says whether the lid is latched (pressed past a detent near closed,
  and not opened past a tolerance since). For a centrifuge lid, a sharps-container lid, a door.
- :class:`Slider`: a panel on a fixed frame, on a prismatic joint (a fume-hood sash, a drawer front, a cabinet door
  that slides). By default the panel is counterweighted (no gravity), so it stays where it is left.
- :class:`Hand`: a person's open hand, palm up, and forearm, a fixed prop of boxes, with a glove that is on or off
  (per task, or per trial with :func:`set_glove`) and a target zone on the palm (:meth:`Hand.on_palm`,
  :func:`handed_over`) for scoring a hand-over.
- :class:`Towel`: a small cloth, a grid of thin rigid tiles joined at their shared edges by joints that fold and
  twist a little but don't stretch; its state gives every tile's centre, for scoring how much of an area it covers
  (:meth:`TowelState.coverage`).

Timed lamps: :func:`switch_lamp_group` (called from an on_reset hook) lights a lamp group at a chosen time into the
trial, with no button (an alarm panel that turns red 20 s in); the schedule clears at every reset, and
:func:`lit_group` (so a demo :class:`Branch`) sees each switch once it happens. :func:`trial_time` gives the seconds
since the reset, :func:`lamp_group_at` what the schedule shows at a given time.

:func:`trial_seed` gives the seed the current reset used (None if unseeded), for tasks that vary a hidden state.

Importing this module hooks ``yam_arena.scene`` (to spawn these and read their state) and the embodiment's step and
reset (to light the lamps, and to start each trial's bookkeeping through :func:`on_reset` and :class:`Trial`).
``score()`` and ``success()`` then get an ``ObjectState`` for each of these (its body pose; ``low``/``high`` from
its box), a :class:`ButtonState` for a button, and a :class:`HingeState`, :class:`SliderState`, :class:`HandState`
or :class:`TowelState` for the kinds above.

Scoring helpers: :func:`tilt_deg`, :func:`lifted`, :func:`moved`, :func:`inside`, :func:`count_inside`,
:func:`all_resting`, :func:`handed_over` and :class:`Trial` (flags that latch during a trial and clear at reset).
Offline tests play a trial's clock and a lid's angles with :func:`tick` and :func:`hinge_step`, as each control step
does in the simulator.

Demo helpers: a :class:`Pose` waypoint carries its own TCP rotation, which ``yam_arena.demo.ArmPlan`` follows
(see :func:`forward_grip`, :func:`rolled` and :func:`ready_pose`).

Pure Python at import time; Isaac Sim is imported lazily, once yam_arena's scene module loads. Copy this file into
each task folder (tools/new_task.py does) so every pull request is self-contained.

Added for rearm-wound-drain-bulb (14 Nov 2026, offline only; not yet in the kit's template, needs a GPU check):
- :class:`Prop`: a fixed prop made of meshes generated in code (:func:`ellipsoid_mesh`, :func:`cylinder_mesh`,
  :func:`tube_mesh`), each visual only unless ``collides``, plus an optional box collider. A mesh part can be
  rescaled each step with :func:`scale_part` (a bulb that visibly flattens).
- :class:`Slider` options: ``panel_visible`` (False: a collider only), ``frame_collides`` (False: the frame is
  visual only) and ``no_collide_with`` (names of other sliders whose panels this panel passes through).
- :class:`Hinge` option ``lid_mesh``: a generated mesh (convex) in place of the lid's box.
- :func:`on_step`: a task hook run every control step with the environment, before scoring;
  :func:`hold_joint` (alias :func:`hold_slider`) / :func:`release_slider`: hold a slider or hinge where it is with a
  stiff drive (task code standing in for a mechanism: a sealed bulb's trapped vacuum, a push-fit cap's grip), and let
  it go again. Holds clear at reset.
- :class:`Harness` (added 2026-11-15 for outlet-into-box; offline checks only, never run in Isaac Sim): a device on
  stiff wires that stay bent (jointed capsules with elastic-perfectly-plastic joints, :func:`plastic_targets` applied
  every control step). The first GPU run must check that it spawns straight and stable, that the loop-closing
  terminal joints hold, which joint style ("d6" or "alternate") Isaac Lab accepts, and that a bend stays bent.

- :class:`CableTie` (added 2026-11-16 for cinch-cord-tie; offline checks only, never run in Isaac Sim): a nylon cable
  tie as one articulation rooted at its head (welded with ``fixed``, else free and put back at every reset by
  ``_reset_tie_root``, which logs the writer it used in :func:`tie_reset_log`): flat strap links on bending joints, a
  head with a through-passage, and a sprung pawl plate (joint ``pawl``) that :func:`hold_joint` can brake or hold open.
  The first GPU run must check that a free 34-body articulation of 0.08 g links is stable, that the reset puts its root
  back, that the strap slides through the passage, and whether the pawl wedges.

- :class:`SprayBottle` and :class:`Cloth` (added 2026-11-20 for wipe-sticky-syrup-spill; offline checks only): a loose
  trigger spray bottle (an articulation like :class:`Button`: the housing, and the trigger on a sprung slide, joint
  ``trigger``; :class:`SprayState` ``travel``), and true cloth copied from the merged roll-silverware-in-a-napkin
  task's labkit (it ran in Isaac Sim there). :func:`on_reset_env` runs a task hook with the environment after every
  reset (to set the first frame's look), and :func:`set_part_visible` shows or hides one visual part of an object.
  The first GPU run must check that the bottle spawns upright and whole, that squeezing its trigger between the
  fingers moves joint ``trigger`` against its spring, and that the hidden syrup parts stay hidden in the first frame.

Status (2026-11-07): Container, Button and Lamp come from a draft task and pass offline checks only; Marked and
Model are new. None has run in Isaac Sim yet: the first GPU session must check that each one spawns.
Added 2026-11-09 (offline checks only): Lamp ``group`` with :func:`set_lamp_group` / :func:`lit_group`,
:func:`trial_seed`, and the demo :class:`Branch`. The first GPU run of a task that uses them must check that only the
active group's lamp lights, that the reset seed reaches the task, and that a Branch follows the lit lamp.
Added 2026-11-10 (offline checks only; needs a GPU check): :class:`Hinge` with :class:`Latch`, :class:`Slider`,
:class:`Hand`, :class:`Towel` and the timed lamp switch. None has run in Isaac Sim. The first GPU session must check
that each spawns where its start state says (lid angle, panel position, flat towel), that a hinge or slider joint
moves within its limits and its spring pulls toward ``rest`` (not toward ``start``), that the fixed bases stay put,
that the glove shows and hides at reset, that a timed switch lights the lamp at its time, and that the towel is
stable at 240 Hz and can be picked up by an edge.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass

import numpy as np
from yam_arena import demo, ik
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.task_file import Box, ObjectState
from yam_arena.urdf import axis_angle_matrix, quat_xyzw_to_matrix

__all__ = [
    "Bottle", "Branch", "Cloth", "ClothState", "ColorField", "set_field", "SprayBottle", "SprayState", "on_reset_env", "set_part_visible", "Button", "ButtonState", "Container", "Hand", "HandState", "Hinge", "HingeState", "Lamp",
    "Latch", "Mark", "Marked", "Model", "Part", "Pose", "Prop", "Slider", "SliderState", "Towel", "TowelState", "Trial",
    "all_resting", "count_inside", "cylinder_mesh", "ellipsoid_mesh", "forward_grip", "glove_on", "handed_over",
    "HangingBulb", "BulbState", "Harness", "HarnessPart", "HarnessState", "HarnessWire", "harness_rest",
    "CableTie", "CableTieState", "tie_reset_log", "set_tie_yield", "tie_yield", "tie_set",
    "plastic_targets", "KeyedPulley", "KeyedPulleyState", "Tool", "write_body_pose", "x_prism_mesh", "x_cylinder_mesh", "arc_yz", "hinge_step", "hold_joint", "hold_slider", "inside", "release_joint", "lamp_group_at", "lamp_switches", "lifted", "lit_group", "moved", "on_reset",
    "on_step", "ready_pose", "release_slider", "rolled", "scale_part", "set_glove", "set_lamp_group",
    "switch_lamp_group", "tick", "tilt_deg", "trial_seed", "trial_time", "tube_mesh",
]

Color = tuple[float, float, float]
X_AXIS, Y_AXIS, Z_AXIS = np.eye(3)
FACES = ("front", "back", "left", "right", "top")  # front faces the robot (-x); left is +y
PROUD = 0.0006  # how far a visual band or mark stands off its face (m)
AXES = {"x": X_AXIS, "y": Y_AXIS, "z": Z_AXIS}
GRAVITY = 9.81
SKIN = (0.93, 0.74, 0.62)
NITRILE = (0.30, 0.45, 0.90)  # a blue nitrile glove
BASE_MASS = 2.0  # kg: a hinge's or slider's base (welded to the world, so it only steadies the solver)
MAX_LINKS = 64  # PhysX's limit on the links of one articulation


def _box_corners(center_local, size, pos, quat_xyzw) -> np.ndarray:
    """The 8 corners (world) of a box of ``size`` centred at ``center_local`` in a body at ``pos``/``quat``."""
    sx, sy, sz = size
    c = np.asarray(center_local, float)
    local = np.array([[i * sx / 2, j * sy / 2, k * sz / 2] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) + c
    return local @ quat_xyzw_to_matrix(np.asarray(quat_xyzw, float)).T + np.asarray(pos, float)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def _yaw_matrix(yaw: float) -> np.ndarray:
    return axis_angle_matrix(Z_AXIS, yaw)


def _turn(axis: str, angle: float) -> np.ndarray:
    """Rotation by ``angle`` (rad, right-handed) about the body axis ``axis`` ("x", "y" or "z")."""
    return axis_angle_matrix(AXES[axis], angle)


def _union_box(*corner_sets) -> tuple[np.ndarray, np.ndarray]:
    c = np.vstack(corner_sets)
    return c.min(0), c.max(0)


# -- object kinds ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Container(Box):
    """An open-top square container: ``size`` is its outside (x, y, height), ``wall`` its wall and floor thickness,
    ``color`` its body and ``label`` (None: no band) the band round it, from ``band[0]`` to ``band[1]`` above its
    bottom. ``pos`` is its floor's centre. Loose with a ``mass``; fixed without one."""

    wall: float = 0.003
    label: Color | None = None
    band: tuple[float, float] = (0.035, 0.075)

    def _check(self) -> None:
        super()._check()
        if not 0 < 2 * self.wall < min(self.size[:2]) or not self.wall < self.size[2]:
            raise ValueError(f"{self.name}: wall {self.wall} doesn't fit size {self.size}")
        if self.label is not None and not 0 <= self.band[0] < self.band[1] <= self.size[2]:
            raise ValueError(f"{self.name}: band {self.band} must lie within the height {self.size[2]}")

    def corners(self, pos, quat_xyzw) -> np.ndarray:
        """The 8 outside corners, world frame, for a body origin (floor centre) at ``pos``."""
        return _box_corners((0.0, 0.0, self.size[2] / 2 - self.wall / 2), self.size, pos, quat_xyzw)

    def holds(self, state: ObjectState, point, margin: float = 0.0) -> bool:
        """Whether ``point`` (world) is inside this container, as it lies in ``state``."""
        local = quat_xyzw_to_matrix(state.quat_xyzw).T @ (np.asarray(point, float) - state.pos)
        inner_x, inner_y = self.size[0] / 2 - self.wall - margin, self.size[1] / 2 - self.wall - margin
        return bool(abs(local[0]) < inner_x and abs(local[1]) < inner_y and 0 < local[2] < self.size[2] - self.wall)


Bottle = Container  # the name the first draft task used


@dataclass(frozen=True)
class Mark:
    """A flat coloured patch on one face of a :class:`Marked` box: ``size`` (width, height) and ``at`` (offset from
    the face's centre) in the face's own plane. Front/back: width along y, height along z. Left/right: width along x,
    height along z. Top: width along x, height along y."""

    face: str
    size: tuple[float, float]
    color: Color
    at: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class Marked(Box):
    """A solid box with :class:`Mark` patches on its faces. ``pos`` is its centre."""

    marks: tuple[Mark, ...] = ()

    def _check(self) -> None:
        super()._check()
        for m in self.marks:
            if m.face not in FACES:
                raise ValueError(f"{self.name}: mark face {m.face!r} must be one of {FACES}")
            (w, h), (u, v) = m.size, m.at
            fw, fh = self._face_extent(m.face)
            if w <= 0 or h <= 0 or abs(u) + w / 2 > fw / 2 + 1e-9 or abs(v) + h / 2 > fh / 2 + 1e-9:
                raise ValueError(f"{self.name}: a {m.face} mark of {m.size} at {m.at} runs off the face ({fw}, {fh})")

    def _face_extent(self, face: str) -> tuple[float, float]:
        sx, sy, sz = self.size
        return {"front": (sy, sz), "back": (sy, sz), "left": (sx, sz), "right": (sx, sz), "top": (sx, sy)}[face]

    def mark_box(self, m: Mark) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """(size, centre) of a mark's thin visual box, in the body frame."""
        sx, sy, sz = self.size
        (w, h), (u, v) = m.size, m.at
        return {
            "front": ((PROUD, w, h), (-(sx + PROUD) / 2, u, v)),
            "back": ((PROUD, w, h), ((sx + PROUD) / 2, u, v)),
            "left": ((w, PROUD, h), (u, (sy + PROUD) / 2, v)),
            "right": ((w, PROUD, h), (u, -(sy + PROUD) / 2, v)),
            "top": ((w, h, PROUD), (u, v, (sz + PROUD) / 2)),
        }[m.face]

    def corners(self, pos, quat_xyzw) -> np.ndarray:
        return _box_corners((0.0, 0.0, 0.0), self.size, pos, quat_xyzw)


@dataclass(frozen=True)
class Model(Box):
    """A 3D model from a USD file (absolute path in ``usd``), ``scale`` applied. Give ``size`` as its bounding box
    after scaling and ``pos`` as that box's centre; the USD's origin must sit at the box centre (tools/convert_mesh.py
    recentres it). Loose with a ``mass``; fixed without one. ``color`` is ignored: the model keeps its own look."""

    usd: str = ""
    scale: float = 1.0

    def _check(self) -> None:
        super()._check()
        if not self.usd.endswith((".usd", ".usda", ".usdc")):
            raise ValueError(f"{self.name}: usd must be a .usd/.usda/.usdc file path, got {self.usd!r}")
        if not self.scale > 0:
            raise ValueError(f"{self.name}: scale must be positive")

    def corners(self, pos, quat_xyzw) -> np.ndarray:
        return _box_corners((0.0, 0.0, 0.0), self.size, pos, quat_xyzw)


@dataclass(frozen=True)
class Button(Box):
    """A push button: a housing (``size``, ``pos`` its centre, ``mass`` heavy enough not to slide when pressed,
    ``color``) with a cap (``cap`` x, y, z, ``cap_color``) on top that presses in up to ``travel`` against a ``rate``
    N/m spring. Lamps light once the cap has gone in ``latch_at``, and stay lit until the next reset."""

    cap: tuple[float, float, float] = (0.04, 0.04, 0.015)
    cap_color: Color = (0.15, 0.35, 0.85)
    cap_mass: float = 0.01
    travel: float = 0.008
    latch_at: float = 0.0056
    rate: float = 300.0
    damping: float = 4.0

    def _check(self) -> None:
        super()._check()
        if self.mass is None:
            raise ValueError(f"{self.name}: give the housing a mass (it stands loose on the table)")
        if not 0 < self.latch_at <= self.travel:
            raise ValueError(f"{self.name}: latch_at must be within the cap's travel")

    @property
    def cap_top(self) -> float:
        """Height of the cap's top at rest (m)."""
        return self.pos[2] + self.size[2] / 2 + self.cap[2]


@dataclass(frozen=True)
class Lamp(Box):
    """A status lamp's lit face: a visual-only disc (``radius``, ``size[2]`` thick) in ``color``, hidden until a
    button latches. Put the dark lamp under it as an ordinary fixed ``Cylinder``. With a ``group``, it lights only
    if that group is the trial's active one (:func:`set_lamp_group`)."""

    radius: float = 0.015
    group: str | None = None

    def _check(self) -> None:
        super()._check()
        if self.mass is not None:
            raise ValueError(f"{self.name}: a lamp is fixed (no mass)")


@dataclass(frozen=True)
class ButtonState(ObjectState):
    """The housing's pose and how far the cap is pressed in (m, 0 at rest)."""

    travel: float = 0.0


@dataclass(frozen=True)
class SprayBottle(Box):
    """A trigger spray bottle (added 2026-11-20 for wipe-sticky-syrup-spill; offline checks only, never run in Isaac
    Sim): one loose articulation, the bottle a rigid housing and the trigger a sprung slide on it, as :class:`Button`
    is a housing and a sprung cap. ``pos`` is the centre of the bottle's base; its frame's x is the nozzle's direction.
    ``body_r``/``body_h``: the round body (a convex 16-sided collider from the base up). ``housing_boxes`` ((name,
    size, centre, color)) are more colliders on the housing (the shroud, the head); ``housing_parts`` visual
    :class:`Part` meshes. The trigger is a ``trigger`` box (x, y, z) centred at ``trigger_rest``, sliding back (-x) up
    to ``travel`` on joint ``trigger`` against a ``rate`` N/m spring. ``com`` is the centre of mass (housing frame):
    the water sits low. Its state is a :class:`SprayState` (``travel``: how far the trigger is in, m)."""

    body_r: float = 0.034
    body_h: float = 0.13
    housing_boxes: tuple = ()
    housing_parts: tuple = ()
    trigger: tuple[float, float, float] = (0.010, 0.018, 0.040)
    trigger_rest: tuple[float, float, float] = (0.040, 0.0, 0.165)
    trigger_color: Color = (0.10, 0.10, 0.12)
    trigger_mass: float = 0.005
    travel: float = 0.010
    rate: float = 3000.0
    damping: float = 5.0
    com: tuple[float, float, float] = (0.0, 0.0, 0.05)
    friction: tuple[float, float] | None = None

    def _check(self) -> None:
        super()._check()
        if self.mass is None or not self.mass > 0:
            raise ValueError(f"{self.name}: a spray bottle is loose: give it a mass")
        names = [b[0] for b in self.housing_boxes] + [p.name for p in self.housing_parts]
        if len(set(names)) != len(names) or not all(n.isidentifier() for n in names):
            raise ValueError(f"{self.name}: box and part names must be unique identifiers")
        for p in self.housing_parts:
            _check_mesh(f"{self.name}.{p.name}", p.mesh)
        if not self.travel > 0:
            raise ValueError(f"{self.name}: the trigger needs some travel")

    def corners(self, pos, quat_xyzw) -> np.ndarray:
        sets = [_box_corners((0.0, 0.0, self.body_h / 2), (2 * self.body_r, 2 * self.body_r, self.body_h), pos,
                             quat_xyzw)]
        sets += [_box_corners(at, size, pos, quat_xyzw) for _, size, at, _ in self.housing_boxes]
        sets.append(_box_corners(self.trigger_rest, self.trigger, pos, quat_xyzw))
        return np.vstack(sets)

    def state_at(self, pos=None, quat_xyzw=(0.0, 0.0, 0.0, 1.0), travel: float = 0.0, lin_vel=None,
                 ang_vel=None) -> "SprayState":
        p = np.asarray(self.pos if pos is None else pos, float)
        q = np.asarray(quat_xyzw, float)
        c = self.corners(p, q)
        return SprayState(pos=p, quat_xyzw=q, lin_vel=np.zeros(3) if lin_vel is None else np.asarray(lin_vel, float),
                          ang_vel=np.zeros(3) if ang_vel is None else np.asarray(ang_vel, float), low=c.min(0),
                          high=c.max(0), travel=float(travel))


@dataclass(frozen=True)
class ColorField(Box):
    """A flat visual layer (added 2026-11-20 for wipe-sticky-syrup-spill; offline checks only): an ``n`` x ``n`` grid of
    square tiles of side ``tile``, centred on ``pos`` (its z the table top), as ONE USD mesh with no material, so the
    renderer draws each tile in its ``displayColor`` (one colour per face; colours are given in sRGB and stored
    linear, as the renderer reads them: round 3 showed sRGB stored as is comes out washed-out). :func:`set_field` shows or hides each tile
    (a hidden one drops ``sink`` under the table top) and recolours it, every step if need be. ``shown0`` and
    ``colors0`` (flattened, row-major: tile (a, b) is index a * n + b) are the look it spawns with. Nothing collides.
    Round 3 (GPU, 20 Nov 2026): the renderer draws it, and updates made between steps reach the cameras."""

    n: int = 26
    tile: float = 0.01
    lift: float = 0.0004  # a shown tile's height over pos (clear of the table top)
    sink: float = 0.004  # a hidden tile's depth under pos (inside the table top)
    shown0: tuple = ()
    colors0: tuple = ()

    def _check(self) -> None:
        super()._check()
        if self.mass is not None:
            raise ValueError(f"{self.name}: a colour field is fixed (no mass)")
        if self.shown0 and len(self.shown0) != self.n * self.n:
            raise ValueError(f"{self.name}: shown0 needs n * n entries")
        if self.colors0 and len(self.colors0) != self.n * self.n:
            raise ValueError(f"{self.name}: colors0 needs n * n colours")

    def tile_points(self, shown) -> np.ndarray:
        """(n * n * 4, 3) corner points in the field's frame: each tile's four corners (a hair over half a tile, so
        neighbours overlap without a seam), at ``lift`` when shown, ``-sink`` when hidden."""
        n, h = self.n, self.tile * 0.51
        c = (np.arange(n) - (n - 1) / 2) * self.tile
        cx, cy = np.meshgrid(c, c, indexing="ij")
        z = np.where(np.asarray(shown, bool).reshape(n, n), self.lift, -self.sink)
        corners = np.array([(-h, -h), (h, -h), (h, h), (-h, h)])
        pts = np.zeros((n, n, 4, 3))
        pts[..., 0] = cx[..., None] + corners[:, 0]
        pts[..., 1] = cy[..., None] + corners[:, 1]
        pts[..., 2] = z[..., None]
        return pts.reshape(-1, 3)


@dataclass(frozen=True)
class SprayState(ObjectState):
    """The bottle's pose and speeds, and how far its trigger is squeezed in (m, 0 at rest)."""

    travel: float = 0.0



# -- jointed objects: a hinge and a slider ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Latch:
    """A latch on a :class:`Hinge`. It catches once the lid comes within ``detent`` (rad) of closed (angle 0) and
    reports latched until the lid opens past ``tol`` (rad); it does not hold the lid shut. A ``detent`` below 0 means
    the lid must be pressed past flush, into overtravel the hinge's ``limits`` allow below 0, against a ``spring``
    that keeps the lid's own weight from sinking it there (:class:`Hinge` checks the spring is strong enough)."""

    detent: float = 0.03
    tol: float = 0.10

    def step(self, angle: float, was: bool) -> bool:
        """Whether the latch is caught at ``angle``, given whether it was caught at the step before."""
        return bool(angle <= self.detent or (was and angle <= self.tol))


@dataclass(frozen=True)
class Hinge(Box):
    """A lid on a fixed base. The base is a box (``size``, ``pos`` its centre, ``color``; welded in place, so no
    ``mass``); the lid is a box (``lid`` x, y, z, ``lid_color``, ``lid_mass``) on a revolute joint about ``axis``
    ("x", "y" or "z", in the base's frame) through ``pivot`` (base frame; default: the base's back top edge).
    ``lid_at`` is the lid box's centre from the pivot when closed (default: lying on the base's top, reaching forward
    from the pivot). Angle 0 is closed, the geometry as given; a positive angle turns the lid about +``axis``, which
    for the default back-edge y hinge lifts its front. ``limits`` (rad) bound the angle, ``start`` is where each trial
    starts, ``spring`` (N m/rad, None for none) pulls toward ``rest`` (rad: 0 a self-closing lid, an open angle a
    gas strut), ``damping`` (N m s/rad) and ``friction`` (N m) slow it. ``latch`` (a :class:`Latch`) adds a latched
    flag to its state. The lid and base don't collide with each other: the joint limits are the stops."""

    lid: tuple[float, float, float] = (0.12, 0.12, 0.01)
    lid_color: Color = (0.85, 0.85, 0.88)
    lid_mass: float = 0.05
    axis: str = "y"
    pivot: tuple[float, float, float] | None = None
    lid_at: tuple[float, float, float] | None = None
    limits: tuple[float, float] = (0.0, 1.9)
    start: float = 0.0
    spring: float | None = None
    rest: float = 0.0
    damping: float = 0.01
    friction: float = 0.0
    latch: Latch | None = None
    lid_mesh: tuple | None = None  # (points, counts, indices) from a *_mesh helper, centred like the lid box

    def _check(self) -> None:
        super()._check()
        if self.mass is not None:
            raise ValueError(f"{self.name}: a hinge's base is fixed (no mass); the lid's is lid_mass")
        if self.lid_mesh is not None:
            _check_mesh(self.name, self.lid_mesh)
        if self.axis not in AXES:
            raise ValueError(f"{self.name}: axis must be one of {tuple(AXES)}, got {self.axis!r}")
        if any(s <= 0 for s in self.lid) or not self.lid_mass > 0:
            raise ValueError(f"{self.name}: lid size and lid_mass must be positive")
        lo, hi = self.limits
        if not lo <= 0.0 <= hi or not lo < hi:
            raise ValueError(f"{self.name}: limits {self.limits} must run from at most 0 (closed) to above it")
        if not lo <= self.start <= hi:
            raise ValueError(f"{self.name}: start {self.start} is outside the limits {self.limits}")
        if (self.spring is not None and self.spring < 0) or self.damping < 0 or self.friction < 0:
            raise ValueError(f"{self.name}: spring, damping and friction can't be negative")
        if self.latch is None:
            return
        d, tol = self.latch.detent, self.latch.tol
        if not lo <= d < tol <= hi:
            raise ValueError(f"{self.name}: the latch needs limits[0] <= detent < tol <= limits[1], got "
                             f"{self.limits}, detent {d}, tol {tol}")
        if d < 0:
            sink = max(0.0, -self.weight_torque(d))  # the lid's weight, closing it
            hold = (self.spring or 0.0) * (self.rest - d)
            if hold <= 1.5 * sink:
                raise ValueError(f"{self.name}: with the detent below flush, the lid's weight alone would latch it: "
                                 f"give it a spring of over {1.5 * sink / max(self.rest - d, 1e-9):.3f} N m/rad "
                                 f"toward a rest at or above 0")

    @property
    def pivot_point(self) -> np.ndarray:
        """The hinge line's point, base frame."""
        if self.pivot is not None:
            return np.asarray(self.pivot, float)
        return np.array([self.size[0] / 2, 0.0, self.size[2] / 2])

    @property
    def lid_offset(self) -> np.ndarray:
        """The lid box's centre from the pivot when closed, base frame."""
        if self.lid_at is not None:
            return np.asarray(self.lid_at, float)
        return np.array([-self.lid[0] / 2, 0.0, self.lid[2] / 2])

    def lid_pose(self, angle: float) -> tuple[np.ndarray, np.ndarray]:
        """(centre, rotation matrix) of the lid box, world frame, at ``angle``."""
        yaw, turn = _yaw_matrix(self.yaw), _turn(self.axis, angle)
        pivot = np.asarray(self.pos, float) + yaw @ self.pivot_point
        return pivot + yaw @ turn @ self.lid_offset, yaw @ turn

    def lid_corners(self, angle: float) -> np.ndarray:
        centre, rot = self.lid_pose(angle)
        return _box_corners((0.0, 0.0, 0.0), self.lid, centre, ik.matrix_to_quat_xyzw(rot))

    def corners(self, pos, quat_xyzw) -> np.ndarray:
        """The base's 8 corners, world frame."""
        return _box_corners((0.0, 0.0, 0.0), self.size, pos, quat_xyzw)

    def weight_torque(self, angle: float) -> float:
        """The lid's weight as a torque about the hinge at ``angle`` (N m; positive opens it)."""
        arm = _turn(self.axis, angle) @ self.lid_offset  # base frame; yaw doesn't tilt the vertical
        return float(np.cross(arm, (0.0, 0.0, -self.lid_mass * GRAVITY)) @ AXES[self.axis])

    def state_at(self, angle: float, latched: bool | None = None) -> HingeState:
        """The state with the lid at ``angle``. ``latched`` None: as a trial starting at that angle reports it."""
        pos, quat = np.asarray(self.pos, float), _yaw_quat(self.yaw)
        centre, rot = self.lid_pose(angle)
        low, high = _union_box(self.corners(pos, quat), self.lid_corners(angle))
        if latched is None:
            latched = self.latch is not None and self.latch.step(angle, False)
        return HingeState(pos=pos, quat_xyzw=quat, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=low, high=high,
                          angle=float(angle), latched=bool(latched), lid_pos=centre,
                          lid_quat=ik.matrix_to_quat_xyzw(rot))


@dataclass(frozen=True)
class HingeState(ObjectState):
    """The base's pose; ``low``/``high`` bound base and lid. ``angle`` is the lid's (rad, 0 closed), ``latched``
    the latch's flag (always False without a latch), ``lid_pos``/``lid_quat`` the lid box's centre and rotation."""

    angle: float = 0.0
    latched: bool = False
    lid_pos: np.ndarray | None = None
    lid_quat: np.ndarray | None = None


@dataclass(frozen=True)
class Slider(Box):
    """A panel on a fixed frame. The frame is a box (``size``, ``pos`` its centre, ``color``; welded in place, so no
    ``mass``); the panel is a box (``panel``, ``panel_color``, ``panel_mass``) on a prismatic joint along ``axis``
    ("x", "y" or "z", frame axes). ``panel_at`` is the panel's centre from the frame's at position 0 (default: just in
    front of the frame's front face, centred). Position 0 is the geometry as given; ``limits`` (m) bound it,
    ``start`` is where each trial starts, ``spring`` (N/m, None for none) pulls toward ``rest``, ``damping``
    (N s/m) and ``friction`` (N) slow it. ``balanced`` (default) leaves the panel weightless, like a counterweighted
    sash that stays where it's left; without it a vertical panel falls to a limit unless a spring holds it."""

    panel: tuple[float, float, float] = (0.01, 0.30, 0.20)
    panel_color: Color = (0.70, 0.82, 0.92)
    panel_mass: float = 0.3
    axis: str = "z"
    panel_at: tuple[float, float, float] | None = None
    limits: tuple[float, float] = (0.0, 0.20)
    start: float = 0.0
    spring: float | None = None
    rest: float = 0.0
    damping: float = 5.0
    friction: float = 0.0
    balanced: bool = True
    panel_visible: bool = True
    frame_collides: bool = True
    no_collide_with: tuple[str, ...] = ()

    def _check(self) -> None:
        super()._check()
        if self.mass is not None:
            raise ValueError(f"{self.name}: a slider's frame is fixed (no mass); the panel's is panel_mass")
        if self.name in self.no_collide_with:
            raise ValueError(f"{self.name}: no_collide_with names the slider itself")
        if self.axis not in AXES:
            raise ValueError(f"{self.name}: axis must be one of {tuple(AXES)}, got {self.axis!r}")
        if any(s <= 0 for s in self.panel) or not self.panel_mass > 0:
            raise ValueError(f"{self.name}: panel size and panel_mass must be positive")
        lo, hi = self.limits
        if not lo < hi or not lo <= self.start <= hi:
            raise ValueError(f"{self.name}: start {self.start} must lie within limits {self.limits} (low < high)")
        if (self.spring is not None and self.spring < 0) or self.damping < 0 or self.friction < 0:
            raise ValueError(f"{self.name}: spring, damping and friction can't be negative")

    @property
    def panel_offset(self) -> np.ndarray:
        """The panel's centre from the frame's at position 0, frame axes."""
        if self.panel_at is not None:
            return np.asarray(self.panel_at, float)
        return np.array([-(self.size[0] + self.panel[0]) / 2 - 0.002, 0.0, 0.0])

    def panel_center(self, position: float) -> np.ndarray:
        """The panel's centre, world frame, at ``position``."""
        local = self.panel_offset + position * AXES[self.axis]
        return np.asarray(self.pos, float) + _yaw_matrix(self.yaw) @ local

    def corners(self, pos, quat_xyzw) -> np.ndarray:
        """The frame's 8 corners, world frame."""
        return _box_corners((0.0, 0.0, 0.0), self.size, pos, quat_xyzw)

    def state_at(self, position: float) -> SliderState:
        pos, quat = np.asarray(self.pos, float), _yaw_quat(self.yaw)
        centre = self.panel_center(position)
        low, high = _union_box(self.corners(pos, quat), _box_corners((0.0, 0.0, 0.0), self.panel, centre, quat))
        return SliderState(pos=pos, quat_xyzw=quat, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=low, high=high,
                           position=float(position), panel_pos=centre)


@dataclass(frozen=True)
class SliderState(ObjectState):
    """The frame's pose; ``low``/``high`` bound frame and panel. ``position`` is the panel's (m along the axis from
    the geometry as given), ``panel_pos`` its centre."""

    position: float = 0.0
    panel_pos: np.ndarray | None = None


# -- a person's hand -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Hand(Box):
    """A person's open hand, palm up, with its forearm: a fixed prop of boxes (all colliders). ``size`` is the palm
    (length, width, thickness) and ``pos`` its centre; at yaw 0 the fingers point to -x (toward the arms) and the
    forearm runs back along +x, its top level with the palm's. ``finger`` is (length, thickness) of the four fingers,
    ``forearm`` (length, width, height); ``side`` "right" puts the thumb on +y, "left" on -y. ``color`` is the skin.
    The glove (``glove`` its colour) is a visual shell over palm, fingers, thumb and a cuff on the wrist: on at the
    start of every trial if ``gloved``, or set per trial with :func:`set_glove` from an on_reset hook."""

    size: tuple[float, float, float] = (0.09, 0.085, 0.025)
    color: Color = SKIN
    finger: tuple[float, float] = (0.075, 0.018)
    forearm: tuple[float, float, float] = (0.20, 0.065, 0.055)
    side: str = "right"
    glove: Color = NITRILE
    gloved: bool = False

    GAP = 0.003  # between fingers (m)
    CUFF = 0.05  # how far the glove's cuff runs up the forearm (m)

    def _check(self) -> None:
        super()._check()
        if self.mass is not None:
            raise ValueError(f"{self.name}: a hand is a fixed prop (no mass)")
        if self.side not in ("right", "left"):
            raise ValueError(f"{self.name}: side must be 'right' or 'left', got {self.side!r}")
        if any(v <= 0 for v in (*self.finger, *self.forearm)):
            raise ValueError(f"{self.name}: finger and forearm sizes must be positive")
        if self.finger[1] > self.size[2] or self.forearm[2] < self.size[2]:
            raise ValueError(f"{self.name}: fingers must be no thicker than the palm, the forearm no thinner")
        if (self.size[1] - 3 * self.GAP) / 4 <= 0.004:
            raise ValueError(f"{self.name}: the palm is too narrow for four fingers")

    def parts(self) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
        """Each collider box as (centre, size) in the hand's frame (origin the palm's centre, before yaw)."""
        px, py, pz = self.size
        top = pz / 2
        fl, ft = self.finger
        fw = (py - 3 * self.GAP) / 4
        out = {"palm": ((0.0, 0.0, 0.0), (px, py, pz))}
        for k in range(4):
            out[f"finger{k}"] = ((-(px + fl) / 2, -py / 2 + fw / 2 + k * (fw + self.GAP), top - ft / 2), (fl, fw, ft))
        tl, tw = 0.6 * fl, 1.2 * fw
        s = 1.0 if self.side == "right" else -1.0
        out["thumb"] = ((-px / 2 + tl / 2 - 0.01, s * (py / 2 + tw / 2), top - ft / 2), (tl, tw, ft))
        length, width, height = self.forearm
        out["forearm"] = ((px / 2 + length / 2, 0.0, top - height / 2), (length, width, height))
        return out

    def glove_parts(self) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
        """The glove's visual boxes as (centre, size) in the hand's frame: palm, fingers and thumb a shade larger
        than the skin, and a cuff round the start of the forearm."""
        grow = 2 * PROUD
        out = {name: (at, tuple(s + grow for s in size)) for name, (at, size) in self.parts().items()
               if name != "forearm"}
        (_, _, fz), (_, width, height) = self.parts()["forearm"]
        cuff = min(self.CUFF, self.forearm[0])
        out["cuff"] = ((self.size[0] / 2 + cuff / 2, 0.0, fz), (cuff, width + grow, height + grow))
        return out

    def corners(self, pos, quat_xyzw) -> np.ndarray:
        """The corners of every collider box, world frame."""
        return np.vstack([_box_corners(at, size, pos, quat_xyzw) for at, size in self.parts().values()])

    @property
    def palm_top(self) -> float:
        """The palm's top surface height (m)."""
        return float(self.pos[2]) + self.size[2] / 2

    def on_palm(self, point, height: float = 0.06, margin: float = 0.0) -> bool:
        """Whether ``point`` (world) is over the palm (``margin`` in from its edges) and no more than ``height``
        above its top: the target zone for an object handed over."""
        local = _yaw_matrix(self.yaw).T @ (np.asarray(point, float) - np.asarray(self.pos, float))
        hx, hy = self.size[0] / 2 - margin, self.size[1] / 2 - margin
        return bool(abs(local[0]) < hx and abs(local[1]) < hy and self.size[2] / 2 < local[2] <= self.size[2] / 2
                    + height)

    def palm_zone(self, height: float = 0.06, margin: float = 0.0):
        """(centre (x, y), half sizes (x, y) along the hand's own axes, z range) of the target zone; equal to
        :func:`inside`'s region when the hand isn't yawed. Prefer :meth:`on_palm`, which follows the yaw."""
        return ((float(self.pos[0]), float(self.pos[1])), (self.size[0] / 2 - margin, self.size[1] / 2 - margin),
                (self.palm_top, self.palm_top + height))

    def state(self) -> HandState:
        """The hand as the scorer gets it: fixed, with the glove on or off as this trial has it."""
        pos, quat = np.asarray(self.pos, float), _yaw_quat(self.yaw)
        low, high = _union_box(self.corners(pos, quat))
        return HandState(pos=pos, quat_xyzw=quat, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=low, high=high,
                         gloved=glove_on(self))


@dataclass(frozen=True)
class HandState(ObjectState):
    """The palm's centre and the hand's yaw (fixed); ``low``/``high`` bound hand and forearm; ``gloved`` whether
    the glove shows this trial."""

    gloved: bool = False


def handed_over(objects, name: str, hand: Hand, height: float = 0.06, margin: float = 0.0,
                max_speed: float = 0.01) -> bool:
    """Object ``name``'s centre is in ``hand``'s palm zone (:meth:`Hand.on_palm`) and it is still. Whether the
    gripper has let go is the task's to check (an arm still holding it there is not a hand-over)."""
    s = objects[name]
    return hand.on_palm(s.pos, height, margin) and s.resting(max_speed=max_speed)


# -- a towel ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Towel(Box):
    """A small cloth: ``tiles`` (nx, ny) thin rigid tiles in a grid, neighbours joined at their shared edge by a
    joint that folds about the edge (up to ``max_fold`` rad) and twists a little about the line between their
    centres (up to ``max_twist``), with no stretch or shear. ``size`` is the whole towel (x, y, thickness), ``pos``
    its centre lying flat, ``mass`` the whole towel's (required). Each joint has a weak spring back to flat
    (``stiffness`` N m/rad, ``damping``, ``armature`` for stability at 240 Hz). Tiles are ``gap`` smaller than
    their pitch so diagonal neighbours don't touch when flat. It is never jittered at reset (yam_arena moves rigid
    bodies only), so check_task treats it as fixed. At most 64 tiles (PhysX's articulation limit)."""

    size: tuple[float, float, float] = (0.20, 0.20, 0.004)
    color: Color = (0.92, 0.92, 0.95)
    tiles: tuple[int, int] = (6, 6)
    gap: float = 0.001
    max_fold: float = 2.0
    max_twist: float = 0.5
    stiffness: float = 0.002
    damping: float = 0.0005
    armature: float = 2e-6
    friction: float = 0.9
    self_collision: bool = True

    @property
    def free(self) -> bool:
        return False

    def _check(self) -> None:
        super()._check()
        nx, ny = self.tiles
        if self.mass is None:
            raise ValueError(f"{self.name}: give the towel a mass (the whole towel's, kg)")
        if int(nx) != nx or int(ny) != ny or nx < 2 or ny < 2 or nx * ny > MAX_LINKS:
            raise ValueError(f"{self.name}: tiles must be whole numbers, at least 2 x 2 and at most {MAX_LINKS} "
                             f"in all, got {self.tiles}")
        px, py = self.pitch
        if not 0 <= self.gap < min(px, py) / 4 or self.size[2] >= min(px, py) / 2:
            raise ValueError(f"{self.name}: tiles of {px:.3f} x {py:.3f} m need a gap under a quarter and a "
                             f"thickness under half of that")
        if not 0 < self.max_fold <= math.pi or not 0 <= self.max_twist <= math.pi / 2:
            raise ValueError(f"{self.name}: max_fold must be in (0, pi], max_twist in [0, pi/2]")
        if min(self.stiffness, self.damping, self.armature, self.friction) < 0:
            raise ValueError(f"{self.name}: stiffness, damping, armature and friction can't be negative")

    @property
    def pitch(self) -> tuple[float, float]:
        """Centre-to-centre spacing of the tiles along x and y (m)."""
        return self.size[0] / self.tiles[0], self.size[1] / self.tiles[1]

    @property
    def centre_tile(self) -> tuple[int, int]:
        """The tile (i, j) the articulation is rooted at."""
        return self.tiles[0] // 2, self.tiles[1] // 2

    def tile_ids(self) -> list[tuple[int, int]]:
        """Every tile's (i, j), i along x, in the order a state's ``points`` lists them."""
        return [(i, j) for i in range(self.tiles[0]) for j in range(self.tiles[1])]

    @staticmethod
    def tile_name(i: int, j: int) -> str:
        return f"tile_{i}_{j}"

    def edges(self) -> list[tuple[tuple[int, int], tuple[int, int], bool]]:
        """Every pair of neighbouring tiles as (parent, child, in_tree). The tree joints (a spine along x through the
        centre tile and ribs along y off it) form the articulation; the others close the grid's loops as ordinary
        joints."""
        nx, ny = self.tiles
        ic, jc = self.centre_tile
        out = [((i - 1 if i > ic else i + 1, jc), (i, jc), True) for i in range(nx) if i != ic]
        out += [((i, j - 1 if j > jc else j + 1), (i, j), True) for i in range(nx) for j in range(ny) if j != jc]
        out += [((i - 1, j), (i, j), False) for i in range(1, nx) for j in range(ny) if j != jc]
        return out

    def start_points(self) -> np.ndarray:
        """Each tile's starting centre, world frame, (nx * ny, 3) in :meth:`tile_ids` order."""
        nx, ny = self.tiles
        px, py = self.pitch
        local = np.array([((i - (nx - 1) / 2) * px, (j - (ny - 1) / 2) * py, 0.0) for i, j in self.tile_ids()])
        return local @ _yaw_matrix(self.yaw).T + np.asarray(self.pos, float)

    def corners(self, pos, quat_xyzw) -> np.ndarray:
        """The flat towel's 8 corners, world frame."""
        return _box_corners((0.0, 0.0, 0.0), self.size, pos, quat_xyzw)

    def state_at(self, points=None, quats=None, lin_vel=None, ang_vel=None) -> TowelState:
        """The state with tiles at ``points`` (n, 3) turned by ``quats`` (n, 4, xyzw); by default lying flat where it
        starts, still. ``lin_vel``/``ang_vel`` (n, 3) per tile; the state reports the fastest tile's."""
        n = self.tiles[0] * self.tiles[1]
        pts = self.start_points() if points is None else np.asarray(points, float).reshape(n, 3)
        qs = np.tile(_yaw_quat(self.yaw), (n, 1)) if quats is None else np.asarray(quats, float).reshape(n, 4)
        lin = np.zeros((n, 3)) if lin_vel is None else np.asarray(lin_vel, float).reshape(n, 3)
        ang = np.zeros((n, 3)) if ang_vel is None else np.asarray(ang_vel, float).reshape(n, 3)
        px, py = self.pitch
        tile = (px - self.gap, py - self.gap, self.size[2])
        low, high = _union_box(*(_box_corners((0.0, 0.0, 0.0), tile, p, q) for p, q in zip(pts, qs)))
        root = self.tile_ids().index(self.centre_tile)
        return TowelState(pos=pts.mean(0), quat_xyzw=qs[root], lin_vel=lin[np.argmax(np.linalg.norm(lin, axis=1))],
                          ang_vel=ang[np.argmax(np.linalg.norm(ang, axis=1))], low=low, high=high, points=pts,
                          quats=qs, shape=tuple(self.tiles), pitch=self.pitch)


@dataclass(frozen=True)
class TowelState(ObjectState):
    """The towel: ``pos`` the mean of its tile centres, ``quat_xyzw`` the centre tile's, ``lin_vel``/``ang_vel`` the
    fastest tile's (so ``resting()`` means every tile is still), ``low``/``high`` bound every tile. ``points`` are the
    tile centres and ``quats`` their rotations, (nx * ny, 3) and (nx * ny, 4) in :meth:`Towel.tile_ids` order."""

    points: np.ndarray | None = None
    quats: np.ndarray | None = None
    shape: tuple[int, int] = (0, 0)
    pitch: tuple[float, float] = (0.0, 0.0)

    def grid(self) -> np.ndarray:
        """Tile centres as (nx, ny, 3): ``grid()[i, j]`` is tile (i, j)."""
        return self.points.reshape(*self.shape, 3)

    def corner_tiles(self) -> np.ndarray:
        """The four corner tiles' centres: (0, 0), (nx-1, 0), (0, ny-1), (nx-1, ny-1)."""
        g = self.grid()
        return np.array([g[0, 0], g[-1, 0], g[0, -1], g[-1, -1]])

    def fraction_inside(self, center, half_xy, z_range) -> float:
        """The share of tile centres :func:`inside` the region."""
        return float(np.mean([inside(p, center, half_xy, z_range) for p in self.points]))

    def coverage(self, center, half_xy, z_range=None, step: float = 0.004) -> float:
        """The share of the rectangle ``center`` +- ``half_xy`` (x, y; m) that the towel covers seen from above,
        counting only tiles whose centre lies within ``z_range`` (e.g. on the bench, not held up), sampled every
        ``step`` m. Each tile covers its full pitch (gaps included), so a flat towel covers its own footprint."""
        hx, hy = half_xy
        nx, ny = max(1, math.ceil(2 * hx / step)), max(1, math.ceil(2 * hy / step))
        xs = center[0] - hx + (np.arange(nx) + 0.5) * (2 * hx / nx)
        ys = center[1] - hy + (np.arange(ny) + 0.5) * (2 * hy / ny)
        samples = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
        covered = np.zeros(len(samples), bool)
        half_u, half_v = self.pitch[0] / 2 + 1e-9, self.pitch[1] / 2 + 1e-9
        for p, q in zip(self.points, self.quats):
            if z_range is not None and not z_range[0] < p[2] < z_range[1]:
                continue
            rot = quat_xyzw_to_matrix(q)
            uv = np.column_stack([rot[:2, 0], rot[:2, 1]])  # the tile's own x and y, seen from above
            if abs(np.linalg.det(uv)) < 1e-6:  # standing on edge: covers nothing
                continue
            a, b = np.linalg.solve(uv, (samples - p[:2]).T)
            covered |= (np.abs(a) <= half_u) & (np.abs(b) <= half_v)
        return float(covered.mean())


# -- meshes generated in code, and fixed props made of them --------------------------------------------------------------
# A mesh is (points, counts, indices): points as a tuple of (x, y, z) in metres, counts the vertices of each face, and
# indices the faces' vertices in order (counter-clockwise seen from outside). Tuples, so specs stay hashable.


def _mesh(points, faces) -> tuple:
    pts = tuple(tuple(round(float(v), 7) for v in p) for p in points)
    return pts, tuple(len(f) for f in faces), tuple(int(i) for f in faces for i in f)


def _check_mesh(name: str, mesh) -> None:
    if not (isinstance(mesh, tuple) and len(mesh) == 3):
        raise ValueError(f"{name}: a mesh is (points, counts, indices)")
    points, counts, indices = mesh
    if sum(counts) != len(indices) or any(c < 3 for c in counts):
        raise ValueError(f"{name}: the mesh's face counts don't match its indices")
    if indices and (min(indices) < 0 or max(indices) >= len(points)):
        raise ValueError(f"{name}: the mesh indexes a point it doesn't have")


def ellipsoid_mesh(a: float, b: float, c: float, n: int = 20) -> tuple:
    """A closed ellipsoid centred at the origin with semi-axes ``a`` (x), ``b`` (y), ``c`` (z): ``n`` segments round,
    ``n // 2`` rings from pole to pole. Convex."""
    rings = max(3, n // 2)
    pts = [(0.0, 0.0, -c)]
    for i in range(1, rings):
        phi = -math.pi / 2 + math.pi * i / rings
        for j in range(n):
            th = 2 * math.pi * j / n
            pts.append((a * math.cos(phi) * math.cos(th), b * math.cos(phi) * math.sin(th), c * math.sin(phi)))
    pts.append((0.0, 0.0, c))
    top, faces = len(pts) - 1, []
    for j in range(n):
        faces.append((0, 1 + (j + 1) % n, 1 + j))
    for i in range(rings - 2):
        r0, r1 = 1 + i * n, 1 + (i + 1) * n
        for j in range(n):
            k = (j + 1) % n
            faces.append((r0 + j, r0 + k, r1 + k, r1 + j))
    last = 1 + (rings - 2) * n
    for j in range(n):
        faces.append((last + j, last + (j + 1) % n, top))
    return _mesh(pts, faces)


def cylinder_mesh(radius: float, height: float, n: int = 16) -> tuple:
    """A closed upright cylinder centred at the origin (axis z). Convex."""
    h = height / 2
    pts = [(radius * math.cos(2 * math.pi * j / n), radius * math.sin(2 * math.pi * j / n), z) for z in (-h, h)
           for j in range(n)]
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    faces += [(j, (j + 1) % n, n + (j + 1) % n, n + j) for j in range(n)]
    return _mesh(pts, faces)


def tube_mesh(path, radius: float, n: int = 8) -> tuple:
    """A tube of ``radius`` along ``path`` (a list of 3-D points), open at both ends. Not convex: visual only."""
    path = [np.asarray(p, float) for p in path]
    if len(path) < 2:
        raise ValueError("a tube needs at least two points")
    pts, ref = [], np.array([0.0, 0.0, 1.0])
    for i, p in enumerate(path):
        d = path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]
        d /= np.linalg.norm(d) + 1e-12
        if abs(d @ ref) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        u = np.cross(d, ref)
        u /= np.linalg.norm(u)
        v = np.cross(d, u)
        pts += [tuple(p + radius * (math.cos(2 * math.pi * j / n) * u + math.sin(2 * math.pi * j / n) * v))
                for j in range(n)]
    faces = [(i * n + j, i * n + (j + 1) % n, (i + 1) * n + (j + 1) % n, (i + 1) * n + j)
             for i in range(len(path) - 1) for j in range(n)]
    return _mesh(pts, faces)


@dataclass(frozen=True)
class Part:
    """One mesh of a :class:`Prop`: ``mesh`` in the part's frame, placed at ``at`` (prop frame, m), in ``color``.
    ``collides`` makes it a convex-hull collider (only for convex meshes)."""

    name: str
    mesh: tuple
    at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    color: Color = (0.7, 0.7, 0.7)
    collides: bool = False
    roughness: float = 0.5
    visible: bool = True  # False: spawned hidden (shown later with set_part_visible)


@dataclass(frozen=True)
class Prop(Box):
    """A fixed prop built from :class:`Part` meshes made in code (a patient, a bulb's visible body, tubing). ``pos``
    is its origin; ``size`` is a box collider round the origin when ``box_collider`` (else nothing collides but the
    parts marked ``collides``). Visual parts can be rescaled each step (:func:`scale_part`)."""

    parts: tuple = ()
    box_collider: bool = False

    def _check(self) -> None:
        super()._check()
        if self.mass is not None:
            raise ValueError(f"{self.name}: a prop is fixed (no mass)")
        names = [p.name for p in self.parts]
        if len(set(names)) != len(names) or not all(n.isidentifier() for n in names):
            raise ValueError(f"{self.name}: part names must be unique identifiers")
        for p in self.parts:
            _check_mesh(f"{self.name}.{p.name}", p.mesh)


@dataclass(frozen=True)
class HangingBulb(Box):
    """A closed-suction drain bulb hanging free: one articulation, its root welded at ``pos`` (the hanging point,
    above the bulb), turned ``yaw``. Joints, by name:

    - ``swing_u``, ``swing_v``: the bulb swings about the hanging point (about its own x, then y), against a soft
      ``swing_spring`` (N m/rad) standing in for its strap and tubing, within ``max_swing`` (rad).
    - ``squeeze_left``, ``squeeze_right``: its two faces (invisible paddles ``paddle``, at +-``face_at`` on its y axis)
      slide in by up to ``travel`` against ``spring`` (N/m each); left moves to negative positions, right to positive.
    - ``cap_hinge``: the cap (a generated mesh ``cap_mesh`` at ``cap_at`` from the hinge) on the port's edge
      (``cap_pivot``, core frame), sprung toward ``cap_rest`` by ``cap_spring``, within ``cap_limits``, starting at
      ``cap_start``; ``cap_latch`` flags it seated.

    The bulb's own frame ("core") has its origin at the hanging point: the bulb's centre is ``drop`` below it, the
    port (``port`` box) on its top. ``parts`` are extra generated meshes carried by the core (visual unless marked),
    for example the visible bulb, scaled each step with :func:`scale_part` (child path ``core/<part>``)."""

    drop: float = 0.16
    swing_spring: float = 1.5
    swing_damping: float = 0.05
    max_swing: float = 0.7
    core_mass: float = 0.04
    paddle: tuple[float, float, float] = (0.06, 0.003, 0.08)
    face_at: float = 0.0225
    travel: float = 0.016
    spring: float = 625.0
    face_damping: float = 2.0
    face_mass: float = 0.02
    port: tuple[float, float, float] = (0.012, 0.012, 0.010)
    port_z: float = 0.053  # the port box's centre above the bulb's centre
    port_color: Color = (0.95, 0.95, 0.94)
    cap_mesh: tuple | None = None
    cap_size: tuple[float, float, float] = (0.016, 0.016, 0.006)  # bounds of the cap (for its box if no mesh)
    cap_pivot: tuple[float, float] = (-0.006, 0.005)  # (y, z) of the hinge line from the port box's centre
    cap_at: tuple[float, float, float] = (0.0, 0.008, 0.003)  # the cap's centre from the hinge when seated
    cap_mass: float = 0.003
    cap_limits: tuple[float, float] = (0.0, 2.0)
    cap_start: float = 2.0
    cap_rest: float = 2.0
    cap_spring: float = 0.002
    cap_damping: float = 0.002
    cap_latch: Latch = Latch(detent=0.035, tol=0.12)
    cap_color: Color = (0.98, 0.98, 0.98)
    parts: tuple = ()

    def _check(self) -> None:
        super()._check()
        if self.mass is not None:
            raise ValueError(f"{self.name}: the hanging point is fixed (no mass); give core_mass instead")
        lo, hi = self.cap_limits
        if not (lo <= self.cap_start <= hi and lo <= self.cap_latch.detent < self.cap_latch.tol <= hi):
            raise ValueError(f"{self.name}: cap start or latch outside cap_limits {self.cap_limits}")
        if self.cap_mesh is not None:
            _check_mesh(f"{self.name}.cap_mesh", self.cap_mesh)
        for p in self.parts:
            _check_mesh(f"{self.name}.{p.name}", p.mesh)
        if not 0 < self.travel < self.face_at:
            raise ValueError(f"{self.name}: the faces would cross (travel {self.travel}, face_at {self.face_at})")

    JOINTS = ("swing_u", "swing_v", "squeeze_left", "squeeze_right", "cap_hinge")

    def joint_rest(self, joint: str) -> float:
        return self.cap_rest if joint == "cap_hinge" else 0.0

    def joint_spring(self, joint: str) -> float:
        return {"swing_u": self.swing_spring, "swing_v": self.swing_spring, "squeeze_left": self.spring,
                "squeeze_right": self.spring, "cap_hinge": self.cap_spring}[joint]

    @property
    def port_offset(self) -> np.ndarray:
        """The port box's centre in the core frame."""
        return np.array([0.0, 0.0, -self.drop + self.port_z])

    @property
    def cap_hinge_point(self) -> np.ndarray:
        return self.port_offset + np.array([0.0, self.cap_pivot[0], self.cap_pivot[1]])

    def state_at(self, core_pos, core_quat_xyzw, joints: dict, latched: bool | None = None) -> BulbState:
        """The bulb's state from its core pose (world) and joint values (``joints``: name -> value, absolute)."""
        R = quat_xyzw_to_matrix(np.asarray(core_quat_xyzw, float))
        p = np.asarray(core_pos, float)
        centre = p + R @ np.array([0.0, 0.0, -self.drop])
        mouth = p + R @ (self.port_offset + np.array([0.0, 0.0, self.port[2] / 2]))
        cap = float(joints.get("cap_hinge", self.cap_start))
        if latched is None:
            latched = self.cap_latch.step(cap, False)
        half = np.array([self.size[0], self.size[1], self.size[2]]) / 2
        return BulbState(pos=p, quat_xyzw=np.asarray(core_quat_xyzw, float), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                         low=centre - half, high=centre + half, centre=centre, mouth=mouth,
                         squeeze_left=float(joints.get("squeeze_left", 0.0)),
                         squeeze_right=float(joints.get("squeeze_right", 0.0)), cap_angle=cap,
                         cap_latched=bool(latched),
                         swing=(float(joints.get("swing_u", 0.0)), float(joints.get("swing_v", 0.0))))


@dataclass(frozen=True)
class BulbState(ObjectState):
    """A :class:`HangingBulb`: ``pos``/``quat_xyzw`` are its core frame (origin at the hanging point); ``centre`` and
    ``mouth`` (the port's open top) in the world; the faces' and cap's joint values; ``swing`` (u, v) in rad."""

    centre: np.ndarray | None = None
    mouth: np.ndarray | None = None
    squeeze_left: float = 0.0
    squeeze_right: float = 0.0
    cap_angle: float = 0.0
    cap_latched: bool = False
    swing: tuple = (0.0, 0.0)


# -- a device on stiff wires ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HarnessWire:
    """One wire of a :class:`Harness`: from ``root`` (harness frame, i.e. relative to the anchor) to ``terminal``
    (device frame), in ``segments`` rigid capsule links of ``radius``. Straight at the start, unless ``points`` gives
    its start shape: segments + 1 joint points (harness frame), root first, terminal last, all links the same length.
    Whatever the start shape, it is the wire's relaxed shape (zero bend)."""

    name: str
    root: tuple
    terminal: tuple
    segments: int
    radius: float
    color: Color = (0.1, 0.1, 0.1)
    link_mass: float = 0.0005
    points: tuple | None = None


@dataclass(frozen=True)
class HarnessPart:
    """A box of the device's body, in the device frame (visual and collider)."""

    name: str
    size: tuple
    at: tuple
    color: Color


@dataclass(frozen=True)
class Harness(Box):
    """A device (an outlet, a switch) hanging on stiff wires that stay bent: an anchor welded at ``pos`` (where the
    cable's sheath ends), ``wires`` running from it, and a rigid device (``parts``, ``device_mass``) whose origin starts
    at ``device_at`` (harness frame, upright, axes as the world's) with every wire ending on it.

    The wires are elastic-perfectly-plastic, like annealed copper: each joint has a stiff spring (``stiffness`` N m/rad,
    ``damping``, ``armature`` for stability at 240 Hz) toward a rest angle that labkit drags along each control step
    whenever the joint is bent more than ``yield_angle`` past it (:func:`plastic_targets`). Unloaded, a joint springs
    back by at most ``yield_angle`` and keeps the rest of its bend. ``style`` "d6": each joint bends both ways (two
    rotational degrees of freedom, no twist); "alternate": one revolute joint per node, its axis alternating between
    the link's y and z (half the bending resolution in each plane, but plain revolute joints).

    One articulation: anchor -> each wire's links; the first wire's last link -> the device by a fixed joint in the
    tree, the other wires' last links -> the device by fixed joints kept out of it (they close loops). The last
    ``end_filter`` links of each wire don't collide with the device (they are screwed down under its terminals). At
    most 64 links. ``yaw`` must be 0. It is never jittered at reset."""

    wires: tuple = ()
    parts: tuple = ()
    device_at: tuple = (0.0, 0.0, 0.0)
    device_mass: float = 0.07
    stiffness: float = 7.0
    damping: float = 0.03
    armature: float = 5e-5
    yield_angle: float = 0.008
    max_bend: float = math.radians(90.0)
    style: str = "d6"
    end_filter: int = 2
    wire_friction: float = 0.4
    torque_cap: float | None = None  # N m: each wire joint's drive torque limit (copper's yield moment); None = none

    @property
    def free(self) -> bool:
        return False

    def _check(self) -> None:
        super()._check()
        if not self.wires or not self.parts:
            raise ValueError(f"{self.name}: a harness needs wires and device parts")
        if self.yaw != 0.0:
            raise ValueError(f"{self.name}: a harness can't be turned (yaw must be 0)")
        if self.style not in ("d6", "alternate"):
            raise ValueError(f"{self.name}: style must be 'd6' or 'alternate', got {self.style!r}")
        if 2 + sum(w.segments for w in self.wires) > MAX_LINKS:
            raise ValueError(f"{self.name}: {2 + sum(w.segments for w in self.wires)} links, PhysX allows {MAX_LINKS}")
        if self.torque_cap is not None and self.torque_cap <= 0:
            raise ValueError(f"{self.name}: torque_cap must be positive (or None)")
        if min(self.stiffness, self.damping, self.armature, self.yield_angle) < 0 or self.device_mass <= 0:
            raise ValueError(f"{self.name}: stiffness, damping, armature, yield_angle >= 0 and a device mass, please")
        names = [w.name for w in self.wires]
        if len(set(names)) != len(names) or any(w.segments < 2 for w in self.wires):
            raise ValueError(f"{self.name}: wire names must be unique and each wire needs at least 2 links")
        for w in self.wires:
            if w.points is None:
                continue
            pts = np.asarray(w.points, float)
            chords = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            if len(pts) != w.segments + 1 or np.ptp(chords) > 1e-6 or chords.min() <= 0:
                raise ValueError(f"{self.name}/{w.name}: points must be segments + 1 points with equal link lengths")
            if not (np.allclose(pts[0], w.root, atol=1e-6) and np.allclose(pts[-1], self.terminal_local(w), atol=1e-6)):
                raise ValueError(f"{self.name}/{w.name}: points must start at root and end at the terminal")

    @staticmethod
    def link_name(wire: str, i: int) -> str:
        return f"{wire}_{i:02d}"

    def terminal_local(self, wire: HarnessWire) -> np.ndarray:
        """The wire's terminal in the harness frame at the start."""
        return np.asarray(self.device_at, float) + np.asarray(wire.terminal, float)

    def start_points(self, wire: HarnessWire) -> np.ndarray:
        """The wire's joint points at the start, harness frame (segments + 1, 3)."""
        if wire.points is not None:
            return np.asarray(wire.points, float)
        a, b = np.asarray(wire.root, float), self.terminal_local(wire)
        return a + np.linspace(0.0, 1.0, wire.segments + 1)[:, None] * (b - a)

    def link_length(self, wire: HarnessWire) -> float:
        return float(np.mean(np.linalg.norm(np.diff(self.start_points(wire), axis=0), axis=1)))

    def link_frames(self, wire: HarnessWire) -> list:
        """Each link's rotation (3x3) at the start: x along the link, root to terminal; y and z carried from link to
        link with the least turn (no twist), starting level (y horizontal) at the root."""
        pts = self.start_points(wire)
        frames = []
        for i in range(wire.segments):
            u = pts[i + 1] - pts[i]
            u = u / np.linalg.norm(u)
            if not frames:
                side = np.cross(Z_AXIS, u)
                side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-6 else Y_AXIS
            else:
                prev = frames[-1]
                side = prev[:, 1] - (prev[:, 1] @ u) * u  # the last link's y, made square to this link
                side = side / np.linalg.norm(side)
            frames.append(np.column_stack([u, side, np.cross(u, side)]))
        return frames

    def wire_frame(self, wire: HarnessWire) -> np.ndarray:
        """The first link's rotation at the start (every link's, for a straight wire)."""
        return self.link_frames(wire)[0]

    def link_centres(self, wire: HarnessWire) -> np.ndarray:
        """Each link's centre at the start, harness frame (segments, 3)."""
        pts = self.start_points(wire)
        return (pts[:-1] + pts[1:]) / 2

    def dof_count(self) -> int:
        per_joint = 2 if self.style == "d6" else 1
        return per_joint * sum(w.segments for w in self.wires)

    def device_corners(self, pos, quat_xyzw) -> np.ndarray:
        rot = quat_xyzw_to_matrix(quat_xyzw)
        corners = [_box_corners(p.at, p.size, np.zeros(3), (0.0, 0.0, 0.0, 1.0)) for p in self.parts]
        return np.vstack(corners) @ rot.T + np.asarray(pos, float)

    def state_at(self, device_pos=None, device_quat=None, links: dict | None = None, lin_vel=None,
                 ang_vel=None) -> HarnessState:
        """The state with the device at ``device_pos`` / ``device_quat`` (world) and each wire's links at
        ``links[name] = (centres (n, 3), quats (n, 4))`` (world); by default everything where it starts."""
        origin = np.asarray(self.pos, float)
        dpos = origin + np.asarray(self.device_at, float) if device_pos is None else np.asarray(device_pos, float)
        dquat = np.array([0.0, 0.0, 0.0, 1.0]) if device_quat is None else np.asarray(device_quat, float)
        polylines = {}
        for w in self.wires:
            half = self.link_length(w) / 2
            if links is None or w.name not in links:
                c = self.link_centres(w) + origin
                axes = np.array([f[:, 0] for f in self.link_frames(w)])
            else:
                c, q = (np.asarray(v, float) for v in links[w.name])
                axes = np.array([quat_xyzw_to_matrix(qi)[:, 0] for qi in q])
            polylines[w.name] = np.vstack([c - half * axes, c[-1:] + half * axes[-1:]])
        corners = self.device_corners(dpos, dquat)
        return HarnessState(pos=dpos, quat_xyzw=dquat, lin_vel=np.zeros(3) if lin_vel is None else np.asarray(lin_vel),
                            ang_vel=np.zeros(3) if ang_vel is None else np.asarray(ang_vel), low=corners.min(0),
                            high=corners.max(0), polylines=polylines)


@dataclass(frozen=True)
class HarnessState(ObjectState):
    """A harness: ``pos``/``quat_xyzw`` the device's origin and rotation, ``lin_vel``/``ang_vel`` the device's,
    ``low``/``high`` bound the device; ``polylines[wire]`` each wire's joint points from root to terminal (n + 1, 3),
    world."""

    polylines: dict | None = None


def plastic_targets(q, target, theta_y: float) -> np.ndarray:
    """Elastic-perfectly-plastic joints: each rest angle in ``target`` is dragged to within ``theta_y`` of the joint
    angle ``q`` (both vectors, rad)."""
    q, target = np.asarray(q, float), np.asarray(target, float)
    return np.clip(target, q - theta_y, q + theta_y)


@dataclass(frozen=True)
class CableTie(Box):
    """A nylon cable tie (added 2026-11-16 for cinch-cord-tie; offline checks only, never run in Isaac Sim): one
    articulation rooted at its HEAD, whose centre is ``pos`` and whose frame is ``quat`` (xyzw, world). In the head
    frame x points the way the strap leaves the head, y across the strap's width and z along the slot; the tail goes
    into the slot on the head's +z face and out of its -z face.

    The strap is ``len(points) - 1`` flat links (``width`` x ``thick``; the last one ``tip_width`` x ``tip_thick``,
    the tapered tip) on revolute joints ``bend_00``, ``bend_01``, ... about y only (it bends like a ribbon, no twist),
    each a spring of ``stiffness`` N m/rad toward the start shape ``points`` (head frame, equal chords, all at y = 0),
    up to ``max_bend`` either way. The head is four rigid walls round a through-passage ``passage_x`` (lo, hi) by
    +-``passage_y``. Inside it a pawl, a thin plate hinged at ``pawl_hinge`` about y (joint ``pawl``), points down the
    slot toward the strap at angle ``pawl_rest`` from -z (its stop and its spring's rest, ``pawl_spring``): pushed in,
    the strap turns it aside; backing out, it drags the plate's tip toward itself and wedges it (friction, no teeth).
    :func:`hold_joint` on ``pawl`` (stiffness ``hold_stiffness``) lets task code brake it or hold it open.

    ``fixed``: the head is welded where it starts (a jig); otherwise the tie is free and every reset puts it back. At
    most 64 bodies (head, pawl, links)."""

    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    points: tuple = ()
    width: float = 0.0048
    thick: float = 0.0012
    tip_width: float = 0.003
    tip_thick: float = 0.0009
    link_mass: float = 8e-5
    head_size: tuple = (0.008, 0.009, 0.0056)
    head_mass: float = 5e-4
    passage_x: tuple = (-0.0026, 0.0014)
    passage_y: float = 0.0032
    stiffness: float = 0.16
    damping: float = 1e-3
    armature: float = 5e-6
    max_bend: float = math.radians(60.0)
    friction: float = 0.9
    fixed: bool = False
    pawl_hinge: tuple = (0.0011, 0.0, 0.0024)
    pawl_len: float = 0.0026
    pawl_radius: float = 0.0003
    pawl_width: float = 0.0056
    pawl_rest: float = math.radians(75.0)
    pawl_mass: float = 1e-4
    pawl_spring: float = 5e-4
    pawl_damping: float = 2e-5
    hold_stiffness: float = 0.5
    head_color: Color = (0.92, 0.91, 0.86)
    contact_offset: float = 0.0004
    body_linear_damping: float = 0.2  # every tie body's own velocity damping (1/s): calms contact chatter
    body_angular_damping: float = 1.0
    max_depenetration: float = 0.3  # m/s
    yield_angle: float | None = None  # rad: nylon takes a set (see :func:`set_tie_yield`); None = purely elastic

    @property
    def free(self) -> bool:
        return False  # an articulation: yam_arena never jitters it; labkit resets it

    def _check(self) -> None:
        super()._check()
        pts = np.asarray(self.points, float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
            raise ValueError(f"{self.name}: points must be at least 3 (x, y, z) points")
        if 2 + len(pts) - 1 > MAX_LINKS:
            raise ValueError(f"{self.name}: {1 + len(pts)} bodies, PhysX allows {MAX_LINKS}")
        chords = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        if np.ptp(chords) > 1e-6 or chords.min() <= 0:
            raise ValueError(f"{self.name}: points must have equal chords")
        if np.max(np.abs(pts[:, 1])) > 1e-9:
            raise ValueError(f"{self.name}: the strap bends about y only: every point needs y = 0")
        if not np.isclose(np.linalg.norm(self.quat), 1.0, atol=1e-6):
            raise ValueError(f"{self.name}: quat must be a unit quaternion (x, y, z, w)")
        if min(self.stiffness, self.damping, self.armature, self.pawl_spring, self.hold_stiffness) < 0:
            raise ValueError(f"{self.name}: stiffness, damping, armature and springs can't be negative")
        if not (self.passage_x[0] < self.passage_x[1] and self.passage_y > self.width / 2):
            raise ValueError(f"{self.name}: the passage must be wider than the strap")

    @property
    def n_links(self) -> int:
        return len(self.points) - 1

    @staticmethod
    def link_name(i: int) -> str:
        return f"strap_{i:02d}"

    @staticmethod
    def joint_name(i: int) -> str:
        return f"bend_{i:02d}"

    def link_length(self) -> float:
        return float(np.linalg.norm(np.asarray(self.points[1], float) - np.asarray(self.points[0], float)))

    def link_frames(self) -> list:
        """Each link's rotation in the head frame: x along the link (root to tip), y the strap's width (head y)."""
        pts = np.asarray(self.points, float)
        out = []
        for i in range(self.n_links):
            u = (pts[i + 1] - pts[i]) / np.linalg.norm(pts[i + 1] - pts[i])
            out.append(np.column_stack([u, Y_AXIS, np.cross(u, Y_AXIS)]))
        return out

    def link_centres(self) -> np.ndarray:
        pts = np.asarray(self.points, float)
        return (pts[:-1] + pts[1:]) / 2

    def head_rotation(self) -> np.ndarray:
        return quat_xyzw_to_matrix(np.asarray(self.quat, float))

    def corners(self, pos, quat_xyzw) -> np.ndarray:
        return _box_corners((0.0, 0.0, 0.0), self.head_size, pos, quat_xyzw)

    def state_at(self, head_pos=None, head_quat=None, links=None, lin_vel=None, ang_vel=None, pawl=None,
                 link_speed: float = 0.0) -> CableTieState:
        """The state with the head at ``head_pos``/``head_quat`` (world) and the links at ``links = (centres (n, 3),
        quats (n, 4))`` (world); by default where everything starts."""
        hpos = np.asarray(self.pos, float) if head_pos is None else np.asarray(head_pos, float)
        hquat = np.asarray(self.quat, float) if head_quat is None else np.asarray(head_quat, float)
        half = self.link_length() / 2
        if links is None:
            r = self.head_rotation()
            c = self.link_centres() @ r.T + np.asarray(self.pos, float)
            axes = np.array([r @ f[:, 0] for f in self.link_frames()])
        else:
            c, q = (np.asarray(v, float) for v in links)
            axes = np.array([quat_xyzw_to_matrix(qi)[:, 0] for qi in q])
        poly = np.vstack([c - half * axes, c[-1:] + half * axes[-1:]])
        box = np.vstack([self.corners(hpos, hquat), poly])
        return CableTieState(pos=hpos, quat_xyzw=hquat,
                             lin_vel=np.zeros(3) if lin_vel is None else np.asarray(lin_vel, float),
                             ang_vel=np.zeros(3) if ang_vel is None else np.asarray(ang_vel, float), low=box.min(0),
                             high=box.max(0), polyline=poly,
                             pawl_angle=self.pawl_rest if pawl is None else float(pawl),
                             link_speed_max=float(link_speed))


@dataclass(frozen=True)
class CableTieState(ObjectState):
    """A cable tie: ``pos``/``quat_xyzw``/``lin_vel``/``ang_vel`` the head's; ``polyline`` the strap's joint points
    from the root to the tip (n + 1, 3, world); ``pawl_angle`` the pawl's angle from the slot axis (rad);
    ``link_speed_max`` the fastest link's speed (m/s)."""

    polyline: np.ndarray | None = None
    pawl_angle: float = 0.0
    link_speed_max: float = 0.0


# -- a wheel on a shaft (added 2026-11-18 for key-pulley-set-screw; offline checks only) --------------------------------


def x_prism_mesh(poly_yz, x0: float, x1: float) -> tuple:
    """A prism along x from ``x0`` to ``x1`` over the polygon ``poly_yz`` ((y, z) points, counter-clockwise seen from
    +x). Convex when the polygon is."""
    n = len(poly_yz)
    pts = [(x, float(y), float(z)) for x in (x0, x1) for y, z in poly_yz]
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    faces += [(j, (j + 1) % n, n + (j + 1) % n, n + j) for j in range(n)]
    return _mesh(pts, faces)


def arc_yz(radius: float, a0: float, a1: float, n: int = 24) -> list:
    """Points on a circle about the x axis (angle from +y toward +z), from ``a0`` to ``a1`` (rad)."""
    return [(radius * math.cos(a), radius * math.sin(a)) for a in np.linspace(a0, a1, n)]


def x_cylinder_mesh(radius: float, x0: float, x1: float, n: int = 32) -> tuple:
    """A closed cylinder about the x axis from ``x0`` to ``x1``. Convex."""
    return x_prism_mesh(arc_yz(radius, 0.0, 2 * math.pi, n + 1)[:-1], x0, x1)


@dataclass(frozen=True)
class KeyedPulley(Box):
    """A wheel on a fixed shaft: one articulation. ``pos`` is the frame origin, on the shaft's axis (which runs along
    x); the base (welded) carries ``base_parts`` (generated meshes, convex colliders when marked: the shaft, its slot,
    collar and support) and ``base_boxes`` ((name, size, at, color or None for invisible) colliders). A massless
    carriage slides along x on joint ``slide`` (m, 0 at the start, ``slide_limits``), and the wheel turns about x on
    joint ``turn`` (rad, 0 the drawing's phase, starting at ``turn_start``). ``wheel_parts`` are the wheel's meshes in
    its own frame (origin on the axis at the frame origin when ``slide`` is 0). ``size`` is only the base's bounding
    box for the state's low/high. Self-collision is off: the wheel's bore is not modelled, the joints are the fit."""

    base_parts: tuple = ()
    base_boxes: tuple = ()
    wheel_parts: tuple = ()
    wheel_mass: float = 0.15
    wheel_inertia: float = 6e-5  # kg m^2 about every axis (a 60 mm, 150 g pulley: about 6.8e-5 about its axis)
    slide_limits: tuple[float, float] = (-0.03, 0.05)
    turn_start: float = 0.0
    slide_damping: float = 2.0
    turn_damping: float = 0.002

    JOINTS = ("slide", "turn")

    def _check(self) -> None:
        super()._check()
        if self.mass is not None:
            raise ValueError(f"{self.name}: the shaft is fixed (no mass); give wheel_mass")
        lo, hi = self.slide_limits
        if not lo <= 0.0 <= hi or not lo < hi:
            raise ValueError(f"{self.name}: slide_limits {self.slide_limits} must contain 0, the start")
        names = [p.name for p in (*self.base_parts, *self.wheel_parts)] + [b[0] for b in self.base_boxes]
        if len(set(names)) != len(names) or not all(n.isidentifier() for n in names):
            raise ValueError(f"{self.name}: part names must be unique identifiers")
        for p in (*self.base_parts, *self.wheel_parts):
            _check_mesh(f"{self.name}.{p.name}", p.mesh)

    def wheel_pose(self, slide: float, turn: float) -> tuple[np.ndarray, np.ndarray]:
        """The wheel frame's origin and rotation (world)."""
        return np.asarray(self.pos, float) + np.array([slide, 0.0, 0.0]), _turn("x", turn)

    def corners(self, pos, quat_xyzw) -> np.ndarray:
        return _box_corners((0.0, 0.0, 0.0), self.size, pos, quat_xyzw)

    def state_at(self, slide: float = 0.0, turn: float | None = None, slide_vel: float = 0.0,
                 turn_vel: float = 0.0) -> KeyedPulleyState:
        turn = self.turn_start if turn is None else turn
        pos, quat = np.asarray(self.pos, float), np.array([0.0, 0.0, 0.0, 1.0])
        wp, wr = self.wheel_pose(slide, turn)
        c = self.corners(pos, quat)
        return KeyedPulleyState(pos=pos, quat_xyzw=quat, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=c.min(0),
                                high=c.max(0), slide=float(slide), turn=float(turn), slide_vel=float(slide_vel),
                                turn_vel=float(turn_vel), wheel_pos=wp, wheel_quat=ik.matrix_to_quat_xyzw(wr))


@dataclass(frozen=True)
class KeyedPulleyState(ObjectState):
    """The shaft's frame; ``slide`` (m) and ``turn`` (rad) the joints, with their speeds; the wheel frame's pose."""

    slide: float = 0.0
    turn: float = 0.0
    slide_vel: float = 0.0
    turn_vel: float = 0.0
    wheel_pos: np.ndarray | None = None
    wheel_quat: np.ndarray | None = None


@dataclass(frozen=True)
class Tool(Box):
    """A loose rigid tool: its first collider is the box ``size`` round its origin (``color``); ``boxes`` adds more
    ((name, size, at, color) in its frame, colliders). ``quat`` (x, y, z, w) is its starting rotation. Its contacts
    with ``no_collide_with`` (prim paths under the env, "name" or "name/child") are switched off: a key that passes
    through a pulley's hub (task code decides the fit), a bit inside a socket the boss's solid collider fills.
    ``mass`` is required. ``state`` is an ordinary ObjectState."""

    boxes: tuple = ()
    quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    no_collide_with: tuple[str, ...] = ()

    def _check(self) -> None:
        super()._check()
        if self.mass is None or not self.mass > 0:
            raise ValueError(f"{self.name}: a tool is loose: give it a mass")
        names = [b[0] for b in self.boxes]
        if len(set(names)) != len(names) or not all(n.isidentifier() for n in names):
            raise ValueError(f"{self.name}: box names must be unique identifiers")

    def corners(self, pos, quat_xyzw) -> np.ndarray:
        sets = [_box_corners((0.0, 0.0, 0.0), self.size, pos, quat_xyzw)]
        sets += [_box_corners(at, size, pos, quat_xyzw) for _, size, at, _ in self.boxes]
        return np.vstack(sets)


def _strap_frames(points, width_axis) -> list:
    """Each link's rotation: x along the link (root to tip), y the webbing's width, z its face normal. ``width_axis``
    is one vector for every link, or one per link (a webbing that starts twisted)."""
    pts = np.asarray(points, float)
    axes = np.asarray(width_axis, float)
    out = []
    for i in range(len(pts) - 1):
        u = pts[i + 1] - pts[i]
        u = u / np.linalg.norm(u)
        w = axes if axes.ndim == 1 else axes[i]
        w = w - (w @ u) * u
        w = w / np.linalg.norm(w)
        out.append(np.column_stack([u, w, np.cross(u, w)]))
    return out


def _check_strap(name: str, points, width_axis) -> None:
    pts = np.asarray(points, float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise ValueError(f"{name}: points must be at least 2 (x, y, z) points")
    chords = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if np.ptp(chords) > 1e-6 or chords.min() <= 0:
        raise ValueError(f"{name}: points must have equal chords")
    w = np.asarray(width_axis, float)
    u = np.diff(pts, axis=0) / chords[:, None]
    dots = u @ w if w.ndim == 1 else np.einsum("ij,ij->i", u, w)
    if w.ndim == 2 and len(w) != len(u):
        raise ValueError(f"{name}: one width axis per link")
    if np.max(np.abs(dots)) > 1e-6:
        raise ValueError(f"{name}: the width axis must be perpendicular to every link at the start")


@dataclass(frozen=True)
class Webbing(Box):
    """A strap of flat links (added 2026-11-19 for ratchet-strap-tie-down; offline checks only until its GPU round 1):
    one articulation rooted at an end fitting (a hook on its anchor) welded to the world at ``pos`` (frame aligned with
    the world). ``points`` (task frame, equal chords) is the start shape: points[0] where the webbing leaves the
    fitting, then each link's far end. Link frames: x along the link (root to tip), y the webbing's width
    (``width_axis`` at the start, perpendicular to every link), z its face normal. Each joint is a D6 that bends about
    the width and twists about the length, springs of ``stiffness`` toward the start shape (limits ``max_bend``,
    ``max_twist``): no stretch, no edgewise bend. ``fitting_parts``: visual Parts in the fitting's frame. ``size`` is
    only a nominal bounding box."""

    points: tuple = ()
    width_axis: tuple = (0.0, 1.0, 0.0)
    width: float = 0.025
    thick: float = 0.0015
    link_mass: float = 1.5e-3
    stiffness: float = 0.01
    damping: float = 1e-3
    armature: float = 1e-5
    max_bend: float = math.radians(80.0)
    max_twist: float = math.radians(60.0)
    friction: float = 0.8
    fitting_parts: tuple = ()
    contact_offset: float = 0.001
    body_linear_damping: float = 0.3
    body_angular_damping: float = 1.0
    max_depenetration: float = 0.3

    @property
    def free(self) -> bool:
        return False  # an articulation: yam_arena never jitters it

    def _check(self) -> None:
        super()._check()
        _check_strap(self.name, self.points, self.width_axis)
        if len(self.points) > MAX_LINKS:
            raise ValueError(f"{self.name}: {len(self.points)} bodies, PhysX allows {MAX_LINKS}")

    @property
    def n_links(self) -> int:
        return len(self.points) - 1

    @staticmethod
    def link_name(i: int) -> str:
        return f"link_{i:02d}"

    def link_length(self) -> float:
        return float(np.linalg.norm(np.subtract(self.points[1], self.points[0])))

    def link_frames(self) -> list:
        return _strap_frames(self.points, self.width_axis)

    def corners(self, pos, quat_xyzw) -> np.ndarray:
        return np.asarray(self.points, float)

    def state_at(self, links=None, link_speed: float = 0.0) -> "WebbingState":
        """The state with the links at ``links = (centres (n, 3), quats (n, 4))`` (world); by default the start."""
        if links is None:
            poly = np.asarray(self.points, float)
        else:
            c, q = (np.asarray(v, float) for v in links)
            half = self.link_length() / 2
            axes = np.array([quat_xyzw_to_matrix(qi)[:, 0] for qi in q])
            poly = np.vstack([c[:1] - half * axes[:1], c + half * axes])
        return WebbingState(pos=np.asarray(self.pos, float), quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                            lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=poly.min(0), high=poly.max(0), polyline=poly,
                            link_speed_max=float(link_speed))


@dataclass(frozen=True)
class WebbingState(ObjectState):
    """``polyline``: the webbing's joint points, root to tip (n + 1, 3, world); ``link_speed_max`` the fastest link."""

    polyline: np.ndarray | None = None
    link_speed_max: float = 0.0


@dataclass(frozen=True)
class Ratchet(Box):
    """A ratchet buckle on its fixed-end webbing (added 2026-11-19 for ratchet-strap-tie-down): one articulation rooted
    at the hook, welded to the world at ``pos`` (world-aligned). ``points``: the fixed-end webbing's start shape (task
    frame, equal chords): points[0] leaves the hook, points[-1] is the base bar. The housing's frame (x from the base bar
    toward the mandrel, y across the webbing, z the handle's side) starts turned ``housing_rot`` (3x3, row-major tuple),
    its origin ``-base_x`` along its x from the base bar; it hangs off the last link on a D6 like the webbing's (it
    pivots round the base bar). ``housing_boxes`` ((name, size, at, color or None)) are its colliders,
    ``housing_pins`` ((name, radius, length, at)) y-cylinder colliders (the mandrel's half-pins), ``housing_parts``
    visual Parts. The handle turns about the housing's y on joint ``handle`` through ``pivot`` (housing frame): 0 shut,
    opening positive up to ``handle_max`` (rad), damped, on a return spring of ``handle_spring`` (N m/rad) toward shut
    (0: none); ``handle_boxes`` are its colliders in its own frame
    (origin on the pivot). Joint names: the webbing's ``fbend_NN`` D6s, then ``handle``. Self-collision is off (the
    handle's arms run outside the plates; the joints are the fit)."""

    points: tuple = ()
    width_axis: tuple = (1.0, 0.0, 0.0)
    width: float = 0.025
    thick: float = 0.0015
    link_mass: float = 1.5e-3
    stiffness: float = 0.01
    damping: float = 1e-3
    armature: float = 1e-5
    max_bend: float = math.radians(80.0)
    max_twist: float = math.radians(60.0)
    base_bend: float = math.radians(150.0)
    friction: float = 0.8
    housing_rot: tuple = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    base_x: float = -0.045
    housing_mass: float = 0.12
    housing_boxes: tuple = ()
    housing_pins: tuple = ()
    housing_parts: tuple = ()
    pivot: tuple = (0.04, 0.0, 0.0)
    handle_mass: float = 0.05
    handle_boxes: tuple = ()
    handle_damping: float = 0.002
    handle_spring: float = 0.0
    handle_max: float = math.pi
    hold_stiffness: float = 20.0
    hook_parts: tuple = ()
    contact_offset: float = 0.001

    JOINTS = ("handle",)

    @property
    def free(self) -> bool:
        return False

    def _check(self) -> None:
        super()._check()
        _check_strap(self.name, self.points, self.width_axis)
        r = self.rot()
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or np.linalg.det(r) < 0:
            raise ValueError(f"{self.name}: housing_rot must be a rotation")
        if len(self.points) + 2 > MAX_LINKS:
            raise ValueError(f"{self.name}: too many bodies for one articulation")

    def rot(self) -> np.ndarray:
        return np.asarray(self.housing_rot, float).reshape(3, 3)

    @property
    def n_links(self) -> int:
        return len(self.points) - 1

    @staticmethod
    def link_name(i: int) -> str:
        return f"flink_{i:02d}"

    def link_length(self) -> float:
        return float(np.linalg.norm(np.subtract(self.points[1], self.points[0])))

    def link_frames(self) -> list:
        return _strap_frames(self.points, self.width_axis)

    def housing_start(self) -> np.ndarray:
        return np.asarray(self.points[-1], float) - self.rot() @ np.array([self.base_x, 0.0, 0.0])

    def corners(self, pos, quat_xyzw) -> np.ndarray:
        return _box_corners((0.0, 0.0, 0.0), self.size, pos, quat_xyzw)

    def state_at(self, housing_pos=None, housing_quat=None, handle: float = 0.0, handle_vel: float = 0.0,
                 lin_vel=None, ang_vel=None, polyline=None) -> "RatchetState":
        pos = self.housing_start() if housing_pos is None else np.asarray(housing_pos, float)
        quat = ik.matrix_to_quat_xyzw(self.rot()) if housing_quat is None else np.asarray(housing_quat, float)
        c = self.corners(pos, quat)
        return RatchetState(pos=pos, quat_xyzw=np.asarray(quat, float),
                            lin_vel=np.zeros(3) if lin_vel is None else np.asarray(lin_vel, float),
                            ang_vel=np.zeros(3) if ang_vel is None else np.asarray(ang_vel, float), low=c.min(0),
                            high=c.max(0), handle=float(handle), handle_vel=float(handle_vel),
                            polyline=np.asarray(self.points if polyline is None else polyline, float))


@dataclass(frozen=True)
class RatchetState(ObjectState):
    """The housing's pose and speeds; ``handle`` (rad, 0 shut) and its speed; ``polyline`` the fixed-end webbing."""

    handle: float = 0.0
    handle_vel: float = 0.0
    polyline: np.ndarray | None = None


def y_cylinder_mesh(radius: float, length: float, n: int = 20) -> tuple:
    """A closed cylinder about the y axis, centred at the origin. Convex."""
    h = length / 2
    pts = [(radius * math.cos(2 * math.pi * j / n), y, radius * math.sin(2 * math.pi * j / n)) for y in (-h, h)
           for j in range(n)]
    faces = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))]
    faces += [(j, n + j, n + (j + 1) % n, (j + 1) % n) for j in range(n)]
    return _mesh(pts, faces)


_PUSH_LOG: list = []  # which external-force call this Isaac Lab took (the probe logs it)


def push_log() -> list:
    return list(_PUSH_LOG)


def body_com_w(asset, origin) -> tuple[np.ndarray, str]:
    """Every body's centre of mass (env frame) and which field gave it: Isaac Lab's ``body_com_pos_w`` (or the
    position part of ``body_com_pose_w`` / ``body_com_state_w``) where this version has it; else the body frames'
    origins, which is wrong for a body whose colliders are lopsided (the ratchet's housing: about 11 mm)."""
    data = asset.data
    for field_name, cols in (("body_com_pos_w", 3), ("body_com_pose_w", 7), ("body_com_state_w", 13)):
        try:
            v = getattr(data, field_name, None)
        except Exception:  # noqa: BLE001 - a lazily computed field may raise on an old version
            v = None
        if v is None:
            continue
        a = _np(v)
        if a.ndim == 3 and a.shape[-1] == cols:
            return a[0][:, :3] - origin, field_name
    return _np(data.body_pos_w)[0] - origin, "body_pos_w (no COM field)"


def push_bodies(environment, name: str, wrenches: dict) -> str:
    """Apply external loads to bodies of articulation ``name`` until the next call (task frame orientation = world):
    ``wrenches[body] = (force, torque[, at])``, ``at`` the point the force acts through (world, default the body's
    centre of mass). Bodies left out get none. Task code standing in for a mechanism the simulator lacks (a ratchet's
    mandrel winding webbing). Tries Isaac Lab 3's permanent wrench composer, then the older call; returns which (with
    the centre-of-mass source used for forces through ``at``)."""
    if environment is None:
        return "offline"
    import torch

    base = getattr(environment, "unwrapped", environment)
    asset = base.scene[name]
    names = list(asset.body_names)
    n = len(names)
    dev = getattr(asset, "device", None) or base.device
    f = torch.zeros((1, n, 3), dtype=torch.float32, device=dev)
    t = torch.zeros((1, n, 3), dtype=torch.float32, device=dev)
    origin = _np(base.scene.env_origins)[0]
    com, com_src = None, ""
    for body, w in wrenches.items():
        i = names.index(body)
        force, torque = np.asarray(w[0], float), np.asarray(w[1], float)
        if len(w) > 2 and w[2] is not None:  # a force through ``at``: that force at the centre of mass plus its moment
            if com is None:
                com, com_src = body_com_w(asset, origin)
            torque = torque + np.cross(np.asarray(w[2], float) - com[i], force)
        f[0, i] = torch.tensor(force, dtype=torch.float32, device=dev)
        t[0, i] = torch.tensor(torque, dtype=torch.float32, device=dev)
    ids = list(range(n))
    tries = []
    comp = getattr(asset, "permanent_wrench_composer", None)
    if comp is not None:
        def _comp():
            comp.reset()
            comp.add_forces_and_torques(f, t, body_ids=ids, is_global=True)
        tries.append(("permanent_wrench_composer", _comp))
    for fn_name in ("set_external_force_and_torque_index", "set_external_force_and_torque"):
        fn = getattr(asset, fn_name, None)
        if fn is not None:
            tries.append((f"{fn_name}(is_global)", lambda fn=fn: fn(f, t, body_ids=ids, is_global=True)))
    errors = []
    for label, call in tries:
        try:
            call()
        except TypeError as err:
            errors.append(f"{label}: {err}")
            continue
        if com_src:
            label = f"{label}; com {com_src}"
        if not _PUSH_LOG or _PUSH_LOG[-1] != label:
            _PUSH_LOG.append(label)
        return label
    raise RuntimeError(f"no external-force call worked on {name}: {errors or 'none found'}")



@dataclass(frozen=True)
class Cloth(Box):
    """True cloth: a PhysX surface deformable, a triangle mesh whose every node moves, so it drapes, folds and
    wrinkles anywhere (unlike :class:`Towel`, which is rigid tiles). Use it when handling fabric is the skill.

    ``size`` is the cloth as it starts, before ``yaw``: its footprint (x, y) and the height of the stack. ``pos`` is
    the centre of that box. With ``quarter=True`` the cloth is ONE sheet of (2 * size[0], 2 * size[1]) that starts
    folded in half twice, four layers ``layer_gap`` apart: the first fold's two folded edges run along the start
    shape's +x side, the second fold, which wraps all four layers, along its +y side, and the -x and -y sides are
    loose edges. With ``quarter=False`` it is a single flat layer of ``size``. The start shape is also the rest
    shape, so the folds stay as pressed creases.

    ``edge`` is the mesh spacing (m): smaller bends rounder but costs simulation time (nodes = sheet area / edge^2).
    ``thickness``, ``density``, ``youngs``, ``poisson``, ``bend`` and the dampings are the fabric; ``friction`` is its
    static and dynamic friction. ``rest_offset`` is half the gap two layers keep between them (self-collision is on
    for a folded cloth), ``contact_offset`` how far out contacts are generated. It has no ``mass`` (its mass comes
    from ``density``), so yam_arena never jitters it. Status: offline only; it has not run in Isaac Sim."""

    size: tuple[float, float, float] = (0.28, 0.23, 0.010)
    color: Color = (0.95, 0.95, 0.93)
    quarter: bool = True
    edge: float = 0.007
    thickness: float = 0.0006
    layer_gap: float = 0.003
    density: float = 440.0  # kg/m^3: 244 g/m^2 fabric 0.55 mm thick
    youngs: float = 1.0e6  # Pa; Isaac Lab's cloth example value
    poisson: float = 0.25
    bend: float = 1.0e3  # surface bend stiffness. Isaac Lab's example value (1e6) makes a folded cloth spring open like a book
    damping: float = 0.1
    bend_damping: float = 0.1
    friction: float = 0.6
    rest_offset: float = 0.001
    contact_offset: float = 0.004
    iterations: int = 24
    self_collide: bool = False  # a single flat sheet: turn on if it will be folded or rolled onto itself

    def _check(self) -> None:
        super()._check()
        if self.mass is not None:
            raise ValueError(f"{self.name}: a Cloth takes no mass; set density and thickness")
        if not 0.002 <= self.edge <= 0.03:
            raise ValueError(f"{self.name}: edge {self.edge} m is outside 2 to 30 mm")
        if self.quarter and self.layer_gap < 2 * self.rest_offset:
            raise ValueError(f"{self.name}: layer_gap {self.layer_gap} is under twice rest_offset {self.rest_offset}, "
                             "so the layers would start pressed into each other")

    @property
    def sheet(self) -> tuple[float, float]:
        """The unfolded sheet's size (m)."""
        k = 2 if self.quarter else 1
        return k * self.size[0], k * self.size[1]

    @property
    def cells(self) -> tuple[int, int]:
        """Mesh cells along the sheet's two sides (even when folded, so each fold line is a line of nodes)."""
        step = 2 if self.quarter else 1
        return tuple(max(step, step * round(side / self.edge / step)) for side in self.sheet)

    @property
    def node_count(self) -> int:
        return (self.cells[0] + 1) * (self.cells[1] + 1)

    @property
    def sheet_mass(self) -> float:
        """kg, from density, thickness and the sheet's area."""
        return self.density * self.thickness * self.sheet[0] * self.sheet[1]

    def mesh(self, world: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """(nodes (n, 3), triangles (m, 3)) of the start shape. ``world=False`` gives the part frame (before yaw,
        about ``pos``). Node (i, j) of the flat sheet is row ``i * (cells[1] + 1) + j``."""
        (cu, cv), (su, sv) = self.cells, self.sheet
        u = np.linspace(0.0, su, cu + 1)[:, None] * np.ones((1, cv + 1))
        v = np.ones((cu + 1, 1)) * np.linspace(0.0, sv, cv + 1)[None, :]
        if self.quarter:
            hu, hv = su / 2, sv / 2
            a = np.where(np.isclose(u, hu), 0.5, (u > hu).astype(float))  # which side of the first fold (0.5: on it)
            b = np.where(np.isclose(v, hv), 0.5, (v > hv).astype(float))  # which side of the second fold
            layer = np.where(b == 0.5, 1.5, np.where(b == 0.0, a, 3.0 - a))
            x, y, z = np.minimum(u, su - u) - hu / 2, np.minimum(v, sv - v) - hv / 2, (layer - 1.5) * self.layer_gap
            # Round the folds so no two nodes coincide (PhysX rejects such a mesh): the first fold's lines bulge half
            # a gap; on the second fold the outer pair of layers (0 and 3) turns outside the inner pair (1 and 2).
            x = x + np.where(a == 0.5, 0.5 * self.layer_gap, 0.0)
            y = y + np.where(b == 0.5, (1.5 - a) * self.layer_gap, 0.0)
        else:
            x, y, z = u - su / 2, v - sv / 2, np.zeros_like(u)
        nodes = np.stack([x, y, z], -1).reshape(-1, 3)
        idx = np.arange((cu + 1) * (cv + 1)).reshape(cu + 1, cv + 1)
        p, q, r, t = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
        faces = np.concatenate([np.stack([p, q, r], 1), np.stack([p, r, t], 1)])
        if world:
            nodes = nodes @ _yaw_matrix(self.yaw).T + np.asarray(self.pos, float)
        return nodes, faces

    def side_nodes(self, side: str) -> np.ndarray:
        """Rows (into the nodes) of the nodes along one side of the folded start shape, before yaw: ``"+y"`` is the
        fold that wraps all four layers, ``"+x"`` the first fold's edges, ``"-x"`` and ``"-y"`` the loose edges."""
        nodes, _ = self.mesh(world=False)
        axis, sign = "xy".index(side[1]), 1.0 if side[0] == "+" else -1.0
        return np.flatnonzero(sign * nodes[:, axis] > self.size[axis] / 2 - self.edge * 0.6)

    def state_at(self, nodes=None, node_vel=None) -> ClothState:
        """The state with the nodes at ``nodes`` (n, 3), moving at ``node_vel``; by default as it starts, still."""
        pts = self.mesh()[0] if nodes is None else np.asarray(nodes, float).reshape(-1, 3)
        vel = np.zeros_like(pts) if node_vel is None else np.asarray(node_vel, float).reshape(-1, 3)
        fastest = vel[np.argmax(np.linalg.norm(vel, axis=1))]
        return ClothState(pos=pts.mean(0), quat_xyzw=_yaw_quat(self.yaw), lin_vel=fastest, ang_vel=np.zeros(3),
                          low=pts.min(0), high=pts.max(0), nodes=pts, node_vel=vel)


@dataclass(frozen=True)
class ClothState(ObjectState):
    """A cloth: ``nodes`` (n, 3) are its mesh nodes and ``node_vel`` their velocities; ``pos`` is their mean,
    ``low``/``high`` bound them, and ``lin_vel`` is the fastest node's, so ``resting()`` means every node is still."""

    nodes: np.ndarray | None = None
    node_vel: np.ndarray | None = None

    def fraction_inside(self, center, half_xy, z_range) -> float:
        """The share of nodes :func:`inside` the region."""
        return float(np.mean([inside(p, center, half_xy, z_range) for p in self.nodes]))

    def covers(self, point, radius: float = 0.01) -> tuple[bool, bool]:
        """(cloth below, cloth above): whether any node lies within ``radius`` (x, y) of ``point`` and lower, or
        higher, than it. A wrapped object has cloth both below and above all along it."""
        near = self.nodes[np.linalg.norm(self.nodes[:, :2] - np.asarray(point, float)[:2], axis=1) <= radius]
        return bool((near[:, 2] < point[2]).any()), bool((near[:, 2] > point[2]).any())




KitObject = Container | Marked | Model | Button | Lamp | Hinge | Slider | Hand | Towel | Prop | HangingBulb | Harness | CableTie | KeyedPulley | Tool | Webbing | Ratchet | SprayBottle | Cloth | ColorField


# -- scoring helpers ---------------------------------------------------------------------------------------------------


def tilt_deg(quat_xyzw) -> float:
    """Angle between a body's z axis and the vertical (degrees)."""
    x, y, _, _ = quat_xyzw
    return math.degrees(math.acos(min(1.0, max(-1.0, 1.0 - 2.0 * (x * x + y * y)))))


def lifted(state: ObjectState, spec, dz: float = 0.02) -> bool:
    """Risen more than ``dz`` (m) above where ``spec`` starts."""
    return float(state.pos[2]) - float(spec.pos[2]) > dz


def moved(state: ObjectState, spec, d: float = 0.02) -> bool:
    """Shifted more than ``d`` (m) across the table from where ``spec`` starts."""
    return float(np.hypot(state.pos[0] - spec.pos[0], state.pos[1] - spec.pos[1])) > d


def inside(point, center, half_xy, z_range) -> bool:
    """``point`` within ``half_xy`` (x, y half widths, m) of ``center`` (x, y) and between ``z_range`` heights."""
    x, y, z = (float(v) for v in point)
    return bool(abs(x - center[0]) < half_xy[0] and abs(y - center[1]) < half_xy[1] and z_range[0] < z < z_range[1])


def count_inside(objects, names, center, half_xy, z_range) -> int:
    """How many of ``names`` have their centre :func:`inside` the region."""
    return sum(inside(objects[n].pos, center, half_xy, z_range) for n in names)


def all_resting(objects, names, max_speed: float = 0.01, max_spin: float = 0.5) -> bool:
    return all(objects[n].resting(max_speed=max_speed, max_spin=max_spin) for n in names)


# -- demo waypoints with their own rotation ------------------------------------------------------------------------------


@dataclass
class Pose(Waypoint):
    """A demo waypoint whose TCP rotation (3x3, world frame) is ``rot`` instead of a straight-down grip."""

    rot: np.ndarray | None = None


def forward_grip(yaw_deg: float = 0.0) -> np.ndarray:
    """The TCP rotation of a grip pointing horizontally forward (+x, turned ``yaw_deg`` about the vertical), its
    fingers closing along world y: the straight-down grip pitched up 90 degrees."""
    return axis_angle_matrix(Z_AXIS, math.radians(yaw_deg)) @ axis_angle_matrix(Y_AXIS, -math.pi / 2) @ grip_rotation()


def rolled(rot: np.ndarray, deg: float) -> np.ndarray:
    """``rot`` rolled ``deg`` about the world x axis: for a forward grip, a turn of the wrist that tips what it holds
    sideways (positive tips its top toward -y)."""
    return axis_angle_matrix(X_AXIS, math.radians(deg)) @ rot


def ready_pose(arm: str, duration: float = 1.5, label: str = "back to ready") -> Pose:
    """A waypoint at the arm's ready pose. The IK is warm-started, so passing through it also puts the arm back on
    the joint branch a forward grip needs after a straight-down move."""
    from yam_arena import station
    from yam_arena.constants import FINGER_OPEN, READY

    pos, rot = ik.tcp_pose(np.array(READY, dtype=float), np.array(station.BASES[arm]))
    return Pose(duration, tuple(pos), FINGER_OPEN, label, rot=rot)


@dataclass
class Branch:
    """A demo step that plays ``options[key]`` once reached, where ``key`` is the lamp group lit at that moment
    (:func:`lit_group`; ``default`` if none is lit), whether a button or a timed switch (:func:`switch_lamp_group`)
    lit it. Every option is padded to the longest one's duration by holding its last pose, so the plan's length
    doesn't depend on the answer. The choice is made once per trial (demo() builds a fresh Branch every trial)."""

    options: dict
    default: str

    def __post_init__(self) -> None:
        if self.default not in self.options:
            raise ValueError(f"Branch default {self.default!r} is not one of {sorted(self.options)}")
        self._chosen = None

    @property
    def duration(self) -> float:
        return max(sum(w.duration for w in ws) for ws in self.options.values())

    def chosen(self) -> list:
        if self._chosen is None:
            key = lit_group() or self.default
            ws = list(self.options.get(key, self.options[self.default]))
            pad = self.duration - sum(w.duration for w in ws)
            if pad > 1e-9 and ws:
                last = ws[-1]
                ws.append(Pose(pad, last.pos, last.finger, "hold", rot=getattr(last, "rot", None)))
            self._chosen = ws
        return self._chosen


def _flat(waypoints, t: float):
    """The waypoints to walk at time ``t``: a :class:`Branch` is replaced by its chosen option once reached."""
    out, elapsed = [], 0.0
    for w in waypoints:
        if isinstance(w, Branch):
            if t < elapsed:  # not reached yet: don't choose before the arm gets there
                break
            out.extend(w.chosen())
        else:
            out.append(w)
        elapsed += w.duration
    return out


def _plan_target(self, t: float):
    """``ArmPlan.target`` that lets a :class:`Pose` waypoint bring its own rotation, and plays :class:`Branch`."""
    prev_pos, prev_rot, prev_finger = self.start_pos, self.start_rot, self.finger_start
    for w in _flat(self.waypoints, t):
        goal = np.asarray(w.pos, dtype=float)
        rot = w.rot if getattr(w, "rot", None) is not None else grip_rotation(w.yaw)
        if t < w.duration:
            s = ik.smoothstep(t / w.duration)
            finger = prev_finger + s * (w.finger - prev_finger)
            return prev_pos + s * (goal - prev_pos), ik.slerp_matrix(prev_rot, rot, s), finger, w.label
        t -= w.duration
        prev_pos, prev_rot, prev_finger = goal, rot, w.finger
    return prev_pos, prev_rot, prev_finger, "done"


demo.ArmPlan.target = _plan_target

# -- trial bookkeeping ------------------------------------------------------------------------------------------------

_SPECS: dict[str, KitObject] = {}
_RESET: list = []
_RESET_ENV: list = []  # fn(environment) after every reset (on_reset_env)
_BEFORE_RESET_ENV: list = []  # fn(environment) just before every reset after the first (on_reset_env)
_LIT = {"on": False}
_GROUP: dict = {"active": None}
_SEED: dict = {"seed": None}
_CLOCK: dict = {"t": 0.0}
_SWITCHES: list = []  # this trial's timed lamp switches: [at (s), group, applied]
_LATCH: dict = {}  # hinge name -> latched at the last step
_GLOVE: dict = {}  # hand name -> glove on this trial (absent: the Hand's ``gloved``)
_STEP: list = []  # task hooks run every control step: fn(environment)
_HELD: dict = {}  # (object name, joint name) -> the value it is held at
HOLD_STIFFNESS = 20000.0  # N/m: a held slider gives under 1 mm to 20 N
STEP_S = 0.1  # the control period, if the embodiment doesn't say (yam_arena runs at 10 Hz)


def set_lamp_group(group: str | None) -> None:
    """Make ``group`` the lamp group that lights when a button latches this trial (call it from an on_reset hook)."""
    _GROUP["active"] = group


def lit_group() -> str | None:
    """The lamp group lit right now (None before any button latches or timed switch): what the cameras show."""
    return _GROUP["active"] if _LIT["on"] else None


def trial_seed():
    """The seed the current trial's reset used, or None for an unseeded reset."""
    return _SEED["seed"]


def trial_time() -> float:
    """Seconds since this trial's reset (the embodiment's clock, advanced every control step)."""
    return _CLOCK["t"]


def switch_lamp_group(group: str | None, at: float) -> None:
    """At ``at`` seconds into this trial, light lamp ``group`` and put out the rest (``group`` None: put every lamp
    out), with or without a button. Call it from an on_reset hook: every reset clears the schedule. Several calls
    queue several switches (green at 0, red at 20). The lamps change on the first control step at or after ``at``
    and the cameras show it from the step after, so a demo :class:`Branch` that reads it should come at least 0.2 s
    later. A task that lights lamps by timer should give every lamp a ``group``, and not also light them by button."""
    at = float(at)
    if not (math.isfinite(at) and at >= 0):
        raise ValueError(f"switch time must be a finite number of seconds >= 0, got {at!r}")
    _SWITCHES.append([at, group, False])
    _SWITCHES.sort(key=lambda s: s[0])
    tick(_CLOCK["t"])  # one due now (at 0, from the reset hook) applies at once


def lamp_switches() -> list[tuple[float, str | None]]:
    """This trial's timed lamp switches as (time, group), in time order."""
    return [(at, group) for at, group, _ in _SWITCHES]


def lamp_group_at(t: float) -> str | None:
    """The group this trial's timed switches have lit by ``t`` seconds (None before the first, or with none)."""
    lit = None
    for at, group, _ in _SWITCHES:
        if at <= t + 1e-9:
            lit = group
    return lit


def tick(t: float, environment=None) -> bool:
    """Move the trial clock to ``t`` seconds and apply the timed lamp switches now due; True if one applied. Every
    control step calls it in the simulator; offline tests call it to play a trial's clock."""
    _CLOCK["t"] = float(t)
    changed = False
    for switch in _SWITCHES:
        at, group, applied = switch
        if not applied and at <= t + 1e-9:
            switch[2] = changed = True
            _GROUP["active"], _LIT["on"] = group, group is not None
    if changed and environment is not None:
        _show_lamps(environment, _LIT["on"])
    return changed


def hinge_step(spec: Hinge, angle: float) -> HingeState:
    """The hinge's state at ``angle`` (rad), its latch carried over from the last call this trial: what every
    control step does with the joint's angle. Offline tests call it to play a lid's angles through a trial."""
    was = _LATCH.get(spec.name, False)
    latched = spec.latch is not None and spec.latch.step(angle, was)
    _LATCH[spec.name] = latched
    return spec.state_at(angle, latched)


def set_glove(hand: Hand | str, on: bool) -> None:
    """Show or hide ``hand``'s glove this trial (call it from an on_reset hook; every reset restores its
    ``gloved``)."""
    _GLOVE[hand if isinstance(hand, str) else hand.name] = bool(on)


def glove_on(hand: Hand) -> bool:
    """Whether ``hand``'s glove shows this trial."""
    return _GLOVE.get(hand.name, hand.gloved)


def on_reset(fn):
    """Run ``fn()`` after every reset, before the trial's first step."""
    _RESET.append(fn)
    return fn


def on_reset_env(fn):
    """Run ``fn(environment)`` around every reset, to set a trial's starting look (which visual parts show): just
    BEFORE the reset (so the reset's own first frame shows it; round 1 found the after-reset call too late for that
    frame) and again after it, once the :func:`on_reset` hooks have run. A part that must be hidden in the very first
    trial's first frame is spawned hidden (``Part(visible=False)``)."""
    _RESET_ENV.append(fn)
    _BEFORE_RESET_ENV.append(fn)
    return fn


def set_part_visible(environment, name: str, part: str, show: bool) -> None:
    """Show or hide visual part ``part`` of object ``name`` (a :class:`Prop` part, say) in env 0."""
    if environment is not None:
        _set_visible(environment, name, show, part)


def on_step(fn):
    """Run ``fn(environment)`` after every control step, before ``success()`` and ``score()`` see the states. It can
    read states with ``labkit.state(environment, name)`` and act on the scene (:func:`hold_slider`,
    :func:`scale_part`). Offline tests call it with ``environment`` None."""
    _STEP.append(fn)
    return fn


def state(environment, name: str):
    """The current state of kit object ``name`` (what ``score()`` gets for it)."""
    return _state(environment, _SPECS[name])


def _joints(spec) -> tuple:
    """The joint names of a jointed kit object, in the order its docs give them."""
    if isinstance(spec, HangingBulb):
        return HangingBulb.JOINTS
    if isinstance(spec, CableTie):
        return ("pawl",)
    if isinstance(spec, KeyedPulley):
        return KeyedPulley.JOINTS
    if isinstance(spec, Ratchet | Webbing):
        return Ratchet.JOINTS
    return ("hinge",) if isinstance(spec, Hinge) else ("slide",)


def _joint_rest(spec, joint: str) -> float:
    if isinstance(spec, KeyedPulley | Ratchet | Webbing):
        return 0.0
    if isinstance(spec, CableTie):
        return spec.pawl_rest if joint == "pawl" else 0.0
    return spec.joint_rest(joint) if isinstance(spec, HangingBulb) else spec.rest


def _joint_spring(spec, joint: str) -> float:
    if isinstance(spec, KeyedPulley):
        return 0.0  # free to slide and turn: no spring
    if isinstance(spec, Ratchet):
        return spec.handle_spring if joint == "handle" else spec.stiffness
    if isinstance(spec, Webbing):
        return spec.stiffness
    if isinstance(spec, CableTie):
        return spec.pawl_spring if joint == "pawl" else spec.stiffness
    return spec.joint_spring(joint) if isinstance(spec, HangingBulb) else (spec.spring or 0.0)


def _angular(spec, joint: str) -> bool:
    return isinstance(spec, Hinge | CableTie | Ratchet | Webbing) or (isinstance(spec, HangingBulb) and not joint.startswith("squeeze"))         or (isinstance(spec, KeyedPulley) and joint == "turn")


def held(name: str, joint: str | None = None) -> float | None:
    """The value joint ``joint`` of ``name`` is held at (default: its only joint), or None if it is free."""
    return _HELD.get((name, joint or _joints(_SPECS[name])[0]))


def _call_first(obj, names, value) -> None:
    """Call the first of ``names`` that ``obj`` has, with ``value`` by keyword then by position (Isaac Lab 3 renamed
    many writers to ``*_index`` with keyword arguments)."""
    for n in names:
        fn = getattr(obj, n, None)
        if fn is None:
            continue
        for kw in ("stiffness", "target", "position", None):
            try:
                return fn(value) if kw is None else fn(**{kw: value})
            except TypeError:
                continue
    raise RuntimeError(f"none of {names} worked on {type(obj).__name__}: check the Isaac Lab articulation API")


def _apply_drives(environment, name: str) -> None:
    """Write every joint drive of ``name``: held joints stiff at their held value, the rest on their own springs
    toward rest. Whole vectors, in the articulation's joint order, so no joint ids are needed."""
    import torch

    spec = _SPECS[name]
    asset = getattr(environment, "unwrapped", environment).scene[name]
    stiff, goal = [], []
    rest = _PLASTIC.get(name) if isinstance(spec, CableTie) else None
    for i, joint in enumerate(asset.joint_names):
        value = _HELD.get((name, joint))
        driven = _DRIVEN.get((name, joint))
        if value is None and driven is not None:  # a task-set drive (drive_joint)
            stiff.append(driven[0])
            goal.append(driven[1] - _joint_rest(spec, joint))
        elif value is None:
            stiff.append(_joint_spring(spec, joint))
            goal.append(float(rest[i]) if rest is not None and joint.startswith("bend_") else 0.0)
        else:
            hold = spec.hold_stiffness if isinstance(spec, CableTie | Ratchet) else HOLD_STIFFNESS_ANGULAR
            stiff.append(hold if _angular(spec, joint) else HOLD_STIFFNESS)
            goal.append(value - _joint_rest(spec, joint))
    writers = (("write_joint_stiffness_to_sim_index", "write_joint_stiffness_to_sim"),
               ("set_joint_position_target_index", "set_joint_position_target"))
    for names in writers:  # check both exist before changing anything, so a failure leaves the springs as they were
        if not any(hasattr(asset, n) for n in names):
            raise RuntimeError(f"the articulation has none of {names}")
    _call_first(asset, writers[0], torch.tensor([stiff], dtype=torch.float32, device=asset.device))
    _call_first(asset, writers[1], torch.tensor([goal], dtype=torch.float32, device=asset.device))


_PINNED: set = set()  # objects whose drives couldn't be rewritten: their held joints are re-set every step instead
_DRIVEN: dict = {}  # (object name, joint name) -> (stiffness, target): drives set by task code (drive_joint)
_HOLD_LOG: list = []  # how each hold_joint / drive_joint call was carried out (the probe logs it)
_CAPPED: dict = {}  # (object name, joint name) -> what capping its drive's effort did, once per trial
HOLD_STIFFNESS_ANGULAR = 5.0  # N m/rad: a held hinge gives under 1 degree to 0.1 N m


def hold_log() -> list:
    return list(_HOLD_LOG)


def _note_hold(entry: str) -> None:
    if not _HOLD_LOG or _HOLD_LOG[-1] != entry:
        _HOLD_LOG.append(entry)
    del _HOLD_LOG[:-50]


def _joint_value(environment, name: str, joint: str) -> float:
    s = state(environment, name)
    if isinstance(_SPECS[name], KeyedPulley):
        return s.slide if joint == "slide" else s.turn
    if isinstance(_SPECS[name], CableTie):
        return s.pawl_angle
    if isinstance(_SPECS[name], Ratchet):
        return s.handle
    if isinstance(_SPECS[name], HangingBulb):
        return {"squeeze_left": s.squeeze_left, "squeeze_right": s.squeeze_right, "cap_hinge": s.cap_angle,
                "swing_u": s.swing[0], "swing_v": s.swing[1]}[joint]
    return s.angle if isinstance(_SPECS[name], Hinge) else s.position


def hold_joint(environment, name: str, value: float | None = None, joint: str | None = None) -> None:
    """Hold joint ``joint`` of ``name`` (default: its only joint) at ``value`` (m or rad; default where it is now),
    both ways, until :func:`release_joint` or the next reset. First choice: a stiff drive (the joints' stiffness and
    targets rewritten through Isaac Lab's articulation API). If this Isaac Lab doesn't have those writers, the held
    joints are instead put back at their values every control step with ``write_joint_position_to_sim_index``, the
    call labkit's reset already relies on; between steps a spring may move them, so read a held joint with
    :func:`held`. Offline (``environment`` None) it only records the hold."""
    spec = _SPECS[name]
    joint = joint or _joints(spec)[0]
    if value is None:
        value = _joint_value(environment, name, joint)
    _HELD[(name, joint)] = float(value)
    if environment is None:
        return "offline"
    if name not in _PINNED:
        try:
            _apply_drives(environment, name)
            _note_hold(f"hold {name}.{joint}: drive")
            return "drive"
        except (RuntimeError, AttributeError, TypeError) as err:
            print(f"labkit: holding {name} by re-setting its joints each step ({err})", flush=True)
            _PINNED.add(name)
    _pin(environment, name)
    _note_hold(f"hold {name}.{joint}: position rewrite")
    return "position rewrite"


def drive_joint(environment, name: str, joint: str, stiffness: float, target: float,
                effort: float | None = None) -> str:
    """Give joint ``joint`` of ``name`` an implicit spring drive of ``stiffness`` toward ``target`` (rad or m) until the
    next call, :func:`release_joint` or the reset; ``stiffness`` 0 gives it back its own spring. ``effort`` caps the
    drive's force or torque where this Isaac Lab can write the limit. The same writers as :func:`hold_joint`; where
    they are missing (the object is pinned) nothing is driven: re-setting positions would teleport the joint. Returns
    what was done ("drive", "drive, effort capped", "released", "unavailable: ..." or "offline")."""
    key = (name, joint)
    if stiffness <= 0.0:
        _DRIVEN.pop(key, None)
    else:
        _DRIVEN[key] = (float(stiffness), float(target))
    if environment is None:
        return "offline"
    if name in _PINNED:
        out = "unavailable: this Isaac Lab has no drive writers (the object is pinned)"
        _note_hold(f"drive {name}.{joint}: {out}")
        return out
    try:
        _apply_drives(environment, name)
    except (RuntimeError, AttributeError, TypeError) as err:
        out = f"unavailable: {type(err).__name__}: {err}"
        _note_hold(f"drive {name}.{joint}: {out}")
        return out
    out = "drive" if stiffness > 0.0 else "released"
    if effort is not None and stiffness > 0.0:
        if key not in _CAPPED:
            _CAPPED[key] = _cap_effort(environment, name, joint, effort)
        out += _CAPPED[key]
    _note_hold(f"drive {name}.{joint}: {out}")
    return out


def _cap_effort(environment, name: str, joint: str, effort: float) -> str:
    import torch

    asset = getattr(environment, "unwrapped", environment).scene[name]
    try:
        i = list(asset.joint_names).index(joint)
        cur = getattr(asset.data, "joint_effort_limits", None)
        lim = _np(cur)[0].copy() if cur is not None else np.full(len(asset.joint_names), 1e6)
        lim[i] = float(effort)
        value = torch.tensor([lim], dtype=torch.float32, device=asset.device)
        for fn_name in ("write_joint_effort_limit_to_sim_index", "write_joint_effort_limit_to_sim"):
            fn = getattr(asset, fn_name, None)
            for kw in (("limits", "effort_limits", None) if fn is not None else ()):
                try:
                    fn(value) if kw is None else fn(**{kw: value})
                    return ", effort capped"
                except TypeError:
                    continue
        return ", effort not capped (no writer)"
    except Exception as err:  # noqa: BLE001 - the cap is a guard; the target's own cap still holds
        return f", effort not capped ({type(err).__name__})"


def _pin(environment, name: str) -> None:
    """Put the held joints of ``name`` back at their held values, at rest; its other joints are left as they are."""
    spec = _SPECS[name]
    asset = getattr(environment, "unwrapped", environment).scene[name]
    pos = getattr(asset.data.joint_pos, "torch", asset.data.joint_pos).clone()
    vel = getattr(asset.data.joint_vel, "torch", asset.data.joint_vel).clone()
    for i, joint in enumerate(asset.joint_names):
        if (name, joint) in _HELD:
            pos[:, i] = _HELD[(name, joint)] - _joint_rest(spec, joint)
            vel[:, i] = 0.0
    asset.write_joint_position_to_sim_index(position=pos)
    asset.write_joint_velocity_to_sim_index(velocity=vel)


def release_joint(environment, name: str, joint: str | None = None) -> None:
    """Give joint ``joint`` of ``name`` (default: its only joint) back its own spring."""
    joint = joint or _joints(_SPECS[name])[0]
    _HELD.pop((name, joint), None)
    _DRIVEN.pop((name, joint), None)
    if environment is not None and name not in _PINNED:
        _apply_drives(environment, name)


hold_slider, release_slider = hold_joint, release_joint  # the names the first version used


def scale_part(environment, name: str, part: str, scale, at=None) -> None:
    """Rescale part ``part`` of prop ``name`` (x, y, z factors about the part's own origin), and move it to ``at``
    (prop frame) if given. Visual parts only: a collider would not follow. Offline (``environment`` None) it does
    nothing."""
    if environment is None:
        return
    from pxr import Gf, UsdGeom

    prim = _find_prim(environment, name, part)
    if prim is None:
        raise RuntimeError(f"no prim for {name}/{part}")
    xform = UsdGeom.Xformable(prim)
    ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
    (ops.get("xformOp:scale") or xform.AddScaleOp()).Set(Gf.Vec3f(*(float(v) for v in scale)))
    if at is not None:
        (ops.get("xformOp:translate") or xform.AddTranslateOp()).Set(Gf.Vec3d(*(float(v) for v in at)))


@on_reset
def _start_trial() -> None:
    """Clear the last trial's clock, timed switches, latches and gloves (registered first, so a task's own on_reset
    hooks run after it)."""
    _CLOCK["t"] = 0.0
    _SWITCHES.clear()
    _LATCH.clear()
    _GLOVE.clear()
    _HELD.clear()
    _DRIVEN.clear()
    _CAPPED.clear()
    _HOLD_LOG.clear()
    _PINNED.clear()
    _LIT["on"] = False


class Trial(dict):
    """Per-trial flags that start at ``defaults`` and are restored at every reset:
    ``TRIAL = Trial(pressed=False, early=False)``, then ``TRIAL["pressed"] = True`` inside ``score()``."""

    def __init__(self, **defaults):
        super().__init__(defaults)
        self._defaults = dict(defaults)
        on_reset(self.reset)

    def reset(self) -> None:
        self.clear()
        self.update(self._defaults)


# -- Arena objects (Isaac Sim) ----------------------------------------------------------------------------------------


def _look(stage, path: str, color: Color, roughness: float = 0.5) -> str:
    import isaaclab.sim as sim_utils

    from yam_arena.station import linear_rgb

    vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=roughness)
    vis.func(path, vis)
    return path


def _cube(stage, path: str, size, at, look: str | None, physics: str | None = None, offset: float = 0.005) -> None:
    """A box under a body: a collider (contacts from ``offset`` out) if ``physics`` (a physics material) is given,
    else visual only."""
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in at)))
    cube.AddScaleOp().Set(Gf.Vec3f(*(float(v) for v in size)))
    if physics is not None:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        contact = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        contact.CreateContactOffsetAttr(offset)
        contact.CreateRestOffsetAttr(0.0)
        bind_physics_material(path, physics, stage=stage)
    if look is not None:
        bind_visual_material(path, look, stage=stage)


def _mesh_prim(stage, path: str, mesh, at, look: str | None, physics: str | None = None,
               offset: float = 0.003) -> None:
    """A generated mesh under a body or prop, translated to ``at`` with a unit scale op (for :func:`scale_part`): a
    convex-hull collider if ``physics`` is given, else visual only."""
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, Vt

    points, counts, indices = mesh
    geom = UsdGeom.Mesh.Define(stage, path)
    geom.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    geom.CreateFaceVertexCountsAttr(Vt.IntArray(list(counts)))
    geom.CreateFaceVertexIndicesAttr(Vt.IntArray(list(indices)))
    geom.CreateSubdivisionSchemeAttr("none")
    geom.CreateDoubleSidedAttr(True)
    geom.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in at)))
    geom.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    if physics is not None:
        prim = geom.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        contact.CreateContactOffsetAttr(offset)
        contact.CreateRestOffsetAttr(0.0)
        bind_physics_material(path, physics, stage=stage)
    if look is not None:
        bind_visual_material(path, look, stage=stage)


def _prop_parts(prim_path: str, spec: Prop, translation=None, orientation=None):
    """A fixed prop: its generated meshes (visual, or convex colliders when marked) and an optional box collider."""
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    root = _root_xform(stage, prim_path, translation, orientation)
    physics = _grippy(prim_path)
    if spec.box_collider:
        _cube(stage, f"{prim_path}/collider", spec.size, (0.0, 0.0, 0.0), None, physics)
    for p in spec.parts:
        look = _look(stage, f"{prim_path}/Looks/{p.name}", p.color, p.roughness)
        _mesh_prim(stage, f"{prim_path}/{p.name}", p.mesh, p.at, look, physics if p.collides else None)
        if not p.visible:
            from pxr import UsdGeom

            UsdGeom.Imageable(stage.GetPrimAtPath(f"{prim_path}/{p.name}")).MakeInvisible()
    return root.GetPrim()


def _grippy(prim_path: str, friction: tuple[float, float] | None = None) -> str:
    import isaaclab.sim as sim_utils

    from yam_arena.constants import GRIPPY_FRICTION

    static, dynamic = friction or GRIPPY_FRICTION
    physics = f"{prim_path}/Looks/physics"
    grippy = sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic)
    grippy.func(physics, grippy)
    return physics


def _container_parts(prim_path: str, spec: Container) -> None:
    """The four walls (colliders) and the label band (visual) on a container whose floor is already spawned."""
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    sx, sy, h = spec.size
    w = spec.wall
    tall = h - w
    zc = w / 2 + tall / 2
    body = _look(stage, f"{prim_path}/Looks/body", spec.color, 0.4)
    physics = _grippy(prim_path)
    walls = {
        "front": ((w, sy, tall), (-(sx - w) / 2, 0.0, zc)),
        "back": ((w, sy, tall), ((sx - w) / 2, 0.0, zc)),
        "left": ((sx - 2 * w, w, tall), (0.0, (sy - w) / 2, zc)),
        "right": ((sx - 2 * w, w, tall), (0.0, -(sy - w) / 2, zc)),
    }
    for name, (size, at) in walls.items():
        _cube(stage, f"{prim_path}/walls/{name}", size, at, body, physics)
    if spec.label is None:
        return
    label = _look(stage, f"{prim_path}/Looks/label", spec.label, 0.6)
    lo, hi = spec.band
    band_h, band_z = hi - lo, (lo + hi) / 2 - w / 2
    for name, (size, at) in {
        "front": ((PROUD, sy, band_h), (-(sx + PROUD) / 2, 0.0, band_z)),
        "back": ((PROUD, sy, band_h), ((sx + PROUD) / 2, 0.0, band_z)),
        "left": ((sx, PROUD, band_h), (0.0, (sy + PROUD) / 2, band_z)),
        "right": ((sx, PROUD, band_h), (0.0, -(sy + PROUD) / 2, band_z)),
    }.items():
        _cube(stage, f"{prim_path}/band/{name}", size, at, label)


def _mark_parts(prim_path: str, spec: Marked) -> None:
    """The visual patches on a marked box whose body is already spawned."""
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    for i, m in enumerate(spec.marks):
        size, at = spec.mark_box(m)
        _cube(stage, f"{prim_path}/marks/m{i}_{m.face}", size, at, _look(stage, f"{prim_path}/Looks/mark{i}", m.color))


def _button_parts(prim_path: str, spec: Button, translation=None, orientation=None):
    """The button as an articulation: housing body, cap body, and the cap's sprung prismatic joint."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
    physics = _grippy(prim_path)

    def body(path: str, at, mass: float):
        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTranslateOp().Set(Gf.Vec3d(*at))
        UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr(mass)

    housing, cap = f"{prim_path}/housing", f"{prim_path}/cap"
    rest = (0.0, 0.0, spec.size[2] / 2 + spec.cap[2] / 2)
    body(housing, (0.0, 0.0, 0.0), spec.mass)
    _cube(stage, f"{housing}/box", spec.size, (0.0, 0.0, 0.0), _look(stage, f"{prim_path}/Looks/housing", spec.color),
          physics)
    body(cap, rest, spec.cap_mass)
    _cube(stage, f"{cap}/box", spec.cap, (0.0, 0.0, 0.0), _look(stage, f"{prim_path}/Looks/cap", spec.cap_color),
          physics)
    joint = UsdPhysics.PrismaticJoint.Define(stage, f"{cap}/press")
    joint.CreateAxisAttr("Z")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(housing)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(cap)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*rest))
    joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLowerLimitAttr(-spec.travel)
    joint.CreateUpperLimitAttr(0.0)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(spec.rate)
    drive.CreateDampingAttr(spec.damping)
    drive.CreateTargetPositionAttr(0.0)
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(housing))
    art = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(housing))
    art.CreateEnabledSelfCollisionsAttr(False)  # the cap sinks into the housing; the joint limit stops it
    art.CreateSolverPositionIterationCountAttr(32)
    return root.GetPrim()


def _hand_parts(prim_path: str, spec: Hand) -> None:
    """Fingers, thumb and forearm (colliders) and the glove (visual, shown or hidden at reset) on a hand whose palm
    is already spawned."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdGeom

    stage = get_current_stage()
    skin = _look(stage, f"{prim_path}/Looks/skin", spec.color, 0.6)
    physics = _grippy(prim_path)
    UsdGeom.Xform.Define(stage, f"{prim_path}/skin")
    for name, (at, size) in spec.parts().items():
        if name != "palm":  # the palm is the body's own box
            _cube(stage, f"{prim_path}/skin/{name}", size, at, skin, physics)
    glove = _look(stage, f"{prim_path}/Looks/glove", spec.glove, 0.35)
    shell = UsdGeom.Xform.Define(stage, f"{prim_path}/glove")
    for name, (at, size) in spec.glove_parts().items():
        _cube(stage, f"{prim_path}/glove/{name}", size, at, glove)
    if not spec.gloved:
        UsdGeom.Imageable(shell.GetPrim()).MakeInvisible()


def _root_xform(stage, prim_path: str, translation, orientation):
    from pxr import Gf, UsdGeom

    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)  # Isaac Lab 3: (x, y, z, w)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
    return root


def _rigid(stage, path: str, at, rot: np.ndarray, mass: float, gravity: bool = True):
    """A rigid body Xform at ``at`` turned by ``rot`` (3x3) in its parent's frame."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    x, y, z, w = ik.matrix_to_quat_xyzw(rot)
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(float(v) for v in at)))
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(w), float(x), float(y),
                                                                               float(z)))
    prim = xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
    physx = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    if not gravity:
        physx.CreateDisableGravityAttr(True)
    return physx


def _weld_to_world(stage, prim_path: str, body: str) -> None:
    """Weld ``body`` to the world where it stands, making its articulation fixed-base. The articulation root API
    sits on the parent Xform: PhysX treats a fixed joint on a root-API body as an ordinary joint, not a fixed base
    (Isaac Lab's fix_root_link does the same)."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    weld = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/weld")
    weld.CreateBody1Rel().SetTargets([Sdf.Path(body)])
    world = UsdGeom.Xformable(stage.GetPrimAtPath(body)).ComputeLocalToWorldTransform(0)
    t, q = world.ExtractTranslation(), world.ExtractRotationQuat()
    weld.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in t)))
    weld.CreateLocalRot0Attr(Gf.Quatf(float(q.GetReal()), *(float(v) for v in q.GetImaginary())))
    weld.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    weld.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))


def _jointed_parts(prim_path: str, spec: Hinge | Slider, translation=None, orientation=None):
    """A hinge or slider as a fixed-base articulation: the base welded to the world, the lid or panel on a revolute
    ("hinge") or prismatic ("slide") joint whose zero is ``spec.rest``, so a zero drive target is the spring's rest."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdPhysics

    stage = get_current_stage()
    root = _root_xform(stage, prim_path, translation, orientation)
    physics = _grippy(prim_path)
    base = f"{prim_path}/base"
    _rigid(stage, base, (0.0, 0.0, 0.0), np.eye(3), BASE_MASS)
    frame_physics = physics if isinstance(spec, Hinge) or spec.frame_collides else None
    _cube(stage, f"{base}/box", spec.size, (0.0, 0.0, 0.0), _look(stage, f"{prim_path}/Looks/base", spec.color),
          frame_physics)
    hinge = isinstance(spec, Hinge)
    if hinge:
        part, size, color, mass = f"{prim_path}/lid", spec.lid, spec.lid_color, spec.lid_mass
        _rigid(stage, part, spec.pivot_point, _turn(spec.axis, spec.start), mass)
        look = _look(stage, f"{prim_path}/Looks/lid", color)
        if spec.lid_mesh is not None:
            _mesh_prim(stage, f"{part}/mesh", spec.lid_mesh, spec.lid_offset, look, physics)
        else:
            _cube(stage, f"{part}/box", size, spec.lid_offset, look, physics)
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{part}/hinge")
        at0, rot0 = spec.pivot_point, ik.matrix_to_quat_xyzw(_turn(spec.axis, spec.rest))
        lower, upper = (math.degrees(v - spec.rest) for v in spec.limits)  # USD angles are degrees
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateStiffnessAttr((spec.spring or 0.0) * math.pi / 180)  # USD angular gains are per degree
        drive.CreateDampingAttr(spec.damping * math.pi / 180)
    else:
        part, size, color, mass = f"{prim_path}/panel", spec.panel, spec.panel_color, spec.panel_mass
        at = spec.panel_offset + spec.start * AXES[spec.axis]
        _rigid(stage, part, at, np.eye(3), mass, gravity=not spec.balanced)
        look = _look(stage, f"{prim_path}/Looks/panel", color) if spec.panel_visible else None
        _cube(stage, f"{part}/box", size, (0.0, 0.0, 0.0), look, physics)
        if not spec.panel_visible:
            from pxr import UsdGeom

            UsdGeom.Imageable(stage.GetPrimAtPath(f"{part}/box")).MakeInvisible()
        if spec.no_collide_with:  # sibling sliders' panels: pass through each other (one bulb, two faces)
            parent = prim_path.rsplit("/", 1)[0]
            pairs = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(part)).CreateFilteredPairsRel()
            for other in spec.no_collide_with:
                pairs.AddTarget(Sdf.Path(f"{parent}/{other}" if "/" in other else f"{parent}/{other}/panel"))
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"{part}/slide")
        at0, rot0 = spec.panel_offset + spec.rest * AXES[spec.axis], np.array([0.0, 0.0, 0.0, 1.0])
        lower, upper = (v - spec.rest for v in spec.limits)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateStiffnessAttr(spec.spring or 0.0)
        drive.CreateDampingAttr(spec.damping)
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(0.0)
    joint.CreateAxisAttr(spec.axis.upper())
    joint.CreateBody0Rel().SetTargets([Sdf.Path(base)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(part)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in at0)))
    joint.CreateLocalRot0Attr(Gf.Quatf(float(rot0[3]), float(rot0[0]), float(rot0[1]), float(rot0[2])))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLowerLimitAttr(float(lower))
    joint.CreateUpperLimitAttr(float(upper))
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    art = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    art.CreateEnabledSelfCollisionsAttr(False)  # lid and base may touch at the hinge; the joint limits are the stops
    art.CreateSolverPositionIterationCountAttr(32)
    _weld_to_world(stage, prim_path, base)
    return root.GetPrim()


def _bulb_parts(prim_path: str, spec: HangingBulb, translation=None, orientation=None):
    """The hanging bulb as one articulation: anchor (welded at the hanging point) -swing_u-> gimbal -swing_v-> core
    (the bulb: port collider, generated parts), and on the core: two faces (prismatic squeeze_left / squeeze_right)
    and the cap (revolute cap_hinge). Joint zeros are the springs' rests, so zero drive targets are the rests."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    stage = get_current_stage()
    root = _root_xform(stage, prim_path, translation, orientation)
    physics = _grippy(prim_path)
    anchor, gimbal, core = f"{prim_path}/anchor", f"{prim_path}/gimbal", f"{prim_path}/core"
    _rigid(stage, anchor, (0.0, 0.0, 0.0), np.eye(3), BASE_MASS)
    _rigid(stage, gimbal, (0.0, 0.0, 0.0), np.eye(3), 0.005)
    _rigid(stage, core, (0.0, 0.0, 0.0), np.eye(3), spec.core_mass)
    # The anchor and gimbal have no shapes to take inertia from, and the core's only collider is its small port: set
    # their mass properties outright, the core's centre of mass at the bulb's centre (a 40 g oval about 6 x 5 x 10 cm).
    for path, inertia, com in ((anchor, 1e-4, (0.0, 0.0, 0.0)), (gimbal, 1e-6, (0.0, 0.0, 0.0)),
                               (core, 3e-5, (0.0, 0.0, -spec.drop))):
        mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(path))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia, inertia, inertia))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    _cube(stage, f"{core}/port", spec.port, spec.port_offset, _look(stage, f"{prim_path}/Looks/port", spec.port_color),
          physics)
    for p in spec.parts:
        look = _look(stage, f"{prim_path}/Looks/{p.name}", p.color, p.roughness)
        _mesh_prim(stage, f"{core}/{p.name}", p.mesh, p.at, look, physics if p.collides else None)

    def joint(kind, path, axis, body0, body1, at0, rot0, lower, upper, stiffness, damping, linear=False):
        j = (UsdPhysics.PrismaticJoint if linear else UsdPhysics.RevoluteJoint).Define(stage, path)
        j.CreateAxisAttr(axis)
        j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
        j.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in at0)))
        j.CreateLocalRot0Attr(Gf.Quatf(float(rot0[3]), float(rot0[0]), float(rot0[1]), float(rot0[2])))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLowerLimitAttr(float(lower))
        j.CreateUpperLimitAttr(float(upper))
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear" if linear else "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(stiffness if linear else stiffness * math.pi / 180)  # USD angular gains: per degree
        drive.CreateDampingAttr(damping if linear else damping * math.pi / 180)
        drive.CreateTargetPositionAttr(0.0)

    ident = np.array([0.0, 0.0, 0.0, 1.0])
    swing = math.degrees(spec.max_swing)
    joint("swing", f"{gimbal}/swing_u", "X", anchor, gimbal, (0, 0, 0), ident, -swing, swing, spec.swing_spring,
          spec.swing_damping)
    joint("swing", f"{core}/swing_v", "Y", gimbal, core, (0, 0, 0), ident, -swing, swing, spec.swing_spring,
          spec.swing_damping)
    for link, sign, lo, hi in (("face_left", 1.0, -spec.travel, 0.0), ("face_right", -1.0, 0.0, spec.travel)):
        at = (0.0, sign * spec.face_at, -spec.drop)
        path = f"{prim_path}/{link}"
        _rigid(stage, path, at, np.eye(3), spec.face_mass, gravity=False)
        _cube(stage, f"{path}/box", spec.paddle, (0.0, 0.0, 0.0), None, physics)
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{path}/box")).MakeInvisible()
        joint("squeeze", f"{path}/squeeze_{link.split('_')[1]}", "Y", core, path, at, ident, lo, hi, spec.spring,
              spec.face_damping, linear=True)
    cap = f"{prim_path}/cap"
    hinge_at = spec.cap_hinge_point
    _rigid(stage, cap, hinge_at, _turn("x", spec.cap_start), spec.cap_mass)
    look = _look(stage, f"{prim_path}/Looks/cap", spec.cap_color)
    if spec.cap_mesh is not None:
        _mesh_prim(stage, f"{cap}/mesh", spec.cap_mesh, spec.cap_at, look, physics)
    else:
        _cube(stage, f"{cap}/box", spec.cap_size, spec.cap_at, look, physics)
    lo, hi = (math.degrees(v - spec.cap_rest) for v in spec.cap_limits)
    joint("cap", f"{cap}/cap_hinge", "X", core, cap, hinge_at, ik.matrix_to_quat_xyzw(_turn("x", spec.cap_rest)), lo,
          hi, spec.cap_spring, spec.cap_damping)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    art = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    art.CreateEnabledSelfCollisionsAttr(False)  # faces, cap and port overlap by design; the joint limits are the stops
    art.CreateSolverPositionIterationCountAttr(32)
    _weld_to_world(stage, prim_path, anchor)
    return root.GetPrim()


def _filter_contacts(stage, body_path: str, prim_path: str, others) -> None:
    """Switch off contacts between ``body_path`` and each of ``others`` ("name" or "name/child" under the env)."""
    from pxr import Sdf, UsdPhysics

    if not others:
        return
    parent = prim_path.rsplit("/", 1)[0]
    rel = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(body_path)).CreateFilteredPairsRel()
    for other in others:
        rel.AddTarget(Sdf.Path(f"{parent}/{other}"))


def _pulley_parts(prim_path: str, spec: KeyedPulley, translation=None, orientation=None):
    """The shaft (welded base) -slide-> carriage -turn-> wheel, as one fixed-base articulation."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    stage = get_current_stage()
    root = _root_xform(stage, prim_path, translation, orientation)
    physics = _grippy(prim_path)
    base, carriage, wheel = f"{prim_path}/base", f"{prim_path}/carriage", f"{prim_path}/wheel"
    _rigid(stage, base, (0.0, 0.0, 0.0), np.eye(3), BASE_MASS)
    for p in spec.base_parts:
        look = _look(stage, f"{prim_path}/Looks/{p.name}", p.color, p.roughness)
        _mesh_prim(stage, f"{base}/{p.name}", p.mesh, p.at, look, physics if p.collides else None, offset=0.002)
    for name, size, at, color in spec.base_boxes:
        look = None if color is None else _look(stage, f"{prim_path}/Looks/{name}", color)
        _cube(stage, f"{base}/{name}", size, at, look, physics, offset=0.002)
        if color is None:
            UsdGeom.Imageable(stage.GetPrimAtPath(f"{base}/{name}")).MakeInvisible()
    _rigid(stage, carriage, (0.0, 0.0, 0.0), np.eye(3), 0.005, gravity=False)
    _rigid(stage, wheel, (0.0, 0.0, 0.0), _turn("x", spec.turn_start), spec.wheel_mass)
    for path, inertia in ((carriage, 1e-7), (wheel, spec.wheel_inertia)):
        m = UsdPhysics.MassAPI(stage.GetPrimAtPath(path))
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia, inertia, inertia))
        m.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    for p in spec.wheel_parts:
        look = _look(stage, f"{prim_path}/Looks/{p.name}", p.color, p.roughness)
        _mesh_prim(stage, f"{wheel}/{p.name}", p.mesh, p.at, look, physics if p.collides else None, offset=0.002)

    def joint(path, body0, body1, lower, upper, damping, linear):
        j = (UsdPhysics.PrismaticJoint if linear else UsdPhysics.RevoluteJoint).Define(stage, path)
        j.CreateAxisAttr("X")
        j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
        j.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLowerLimitAttr(float(lower))
        j.CreateUpperLimitAttr(float(upper))
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear" if linear else "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(damping if linear else damping * math.pi / 180)
        drive.CreateTargetPositionAttr(0.0)

    lo, hi = spec.slide_limits
    joint(f"{carriage}/slide", base, carriage, lo, hi, spec.slide_damping, True)
    joint(f"{wheel}/turn", carriage, wheel, -720.0, 720.0, spec.turn_damping, False)  # degrees: two turns either way
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    art = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    art.CreateEnabledSelfCollisionsAttr(False)  # the wheel sits on the shaft: the joints are the fit
    art.CreateSolverPositionIterationCountAttr(32)
    _weld_to_world(stage, prim_path, base)
    return root.GetPrim()


def _tool_parts(prim_path: str, spec: Tool) -> None:
    """A tool's extra colliders, on a body whose first box is already spawned, and its contact filters."""
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    physics = _grippy(prim_path)
    for name, size, at, color in spec.boxes:
        _cube(stage, f"{prim_path}/{name}", size, at, _look(stage, f"{prim_path}/Looks/{name}", color, 0.4), physics,
              offset=0.002)
    _filter_contacts(stage, prim_path, prim_path, spec.no_collide_with)


def _towel_parts(prim_path: str, spec: Towel, translation=None, orientation=None):
    """The towel as a floating articulation rooted at its centre tile (``translation``): a tree of edge joints, plus
    loop-closing joints kept out of the articulation."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdPhysics

    stage = get_current_stage()
    root = _root_xform(stage, prim_path, translation, orientation)
    look = _look(stage, f"{prim_path}/Looks/cloth", spec.color, 0.8)
    physics = f"{prim_path}/Looks/physics"
    cloth = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction, dynamic_friction=spec.friction)
    cloth.func(physics, cloth)
    px, py = spec.pitch
    ic, jc = spec.centre_tile
    n = spec.tiles[0] * spec.tiles[1]

    def tile(i: int, j: int) -> str:
        return f"{prim_path}/{spec.tile_name(i, j)}"

    for i, j in spec.tile_ids():
        body = _rigid(stage, tile(i, j), ((i - ic) * px, (j - jc) * py, 0.0), np.eye(3), spec.mass / n)
        body.CreateMaxDepenetrationVelocityAttr(1.0)
        body.CreateLinearDampingAttr(0.5)
        body.CreateAngularDampingAttr(2.0)
        _cube(stage, f"{tile(i, j)}/box", (px - spec.gap, py - spec.gap, spec.size[2]), (0.0, 0.0, 0.0), look,
              physics, offset=0.002)

    quarter = Gf.Quatf(math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
    for (a, b, in_tree) in spec.edges():
        along_x = a[1] == b[1]
        sign = 1.0 if (b[0] - a[0] if along_x else b[1] - a[1]) > 0 else -1.0
        half = np.array([px / 2, 0.0, 0.0]) if along_x else np.array([0.0, py / 2, 0.0])
        joint = UsdPhysics.Joint.Define(stage, f"{tile(*b)}/{'edge' if in_tree else 'loop'}_{a[0]}_{a[1]}")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(tile(*a))])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(tile(*b))])
        frame = Gf.Quatf(1, 0, 0, 0) if along_x else quarter  # joint x along the axis between the two tiles
        joint.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in sign * half)))
        joint.CreateLocalRot0Attr(frame)
        joint.CreateLocalPos1Attr(Gf.Vec3f(*(float(v) for v in -sign * half)))
        joint.CreateLocalRot1Attr(frame)
        if not in_tree:
            joint.CreateExcludeFromArticulationAttr(True)
        prim = joint.GetPrim()
        for dof in ("transX", "transY", "transZ", "rotZ"):  # no stretch, no shear in the cloth's plane
            lim = UsdPhysics.LimitAPI.Apply(prim, dof)
            lim.CreateLowAttr(1.0)
            lim.CreateHighAttr(-1.0)  # low > high locks the axis
        for dof, most in (("rotY", spec.max_fold), ("rotX", spec.max_twist)):  # fold about the edge; twist
            lim = UsdPhysics.LimitAPI.Apply(prim, dof)
            lim.CreateLowAttr(-math.degrees(most))
            lim.CreateHighAttr(math.degrees(most))
            drive = UsdPhysics.DriveAPI.Apply(prim, dof)
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(spec.stiffness * math.pi / 180)
            drive.CreateDampingAttr(spec.damping * math.pi / 180)
            drive.CreateTargetPositionAttr(0.0)

    # The root API on the centre tile itself, so it is the root link (on a plain Xform PhysX picks its own).
    centre = stage.GetPrimAtPath(tile(ic, jc))
    UsdPhysics.ArticulationRootAPI.Apply(centre)
    art = PhysxSchema.PhysxArticulationAPI.Apply(centre)
    art.CreateEnabledSelfCollisionsAttr(spec.self_collision)
    art.CreateSolverPositionIterationCountAttr(32)
    art.CreateSolverVelocityIterationCountAttr(4)
    return root.GetPrim()


def _capsule(stage, path: str, radius: float, length: float, look: str | None, physics: str | None,
             offset: float = 0.001) -> None:
    """A capsule along the body's x axis, ``length`` between its cap centres, as a collider (and visual)."""
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    cap = UsdGeom.Capsule.Define(stage, path)
    cap.CreateAxisAttr("X")
    cap.CreateRadiusAttr(float(radius))
    cap.CreateHeightAttr(float(length))
    if physics is not None:
        UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
        contact = PhysxSchema.PhysxCollisionAPI.Apply(cap.GetPrim())
        contact.CreateContactOffsetAttr(offset)
        contact.CreateRestOffsetAttr(0.0)
        bind_physics_material(path, physics, stage=stage)
    if look is not None:
        bind_visual_material(path, look, stage=stage)


def _quatf(q_xyzw):
    from pxr import Gf

    x, y, z, w = (float(v) for v in q_xyzw)
    return Gf.Quatf(w, x, y, z)


def _harness_parts(prim_path: str, spec: Harness, translation=None, orientation=None):
    """The harness as one fixed-base articulation (see :class:`Harness`). Every joint's zero is the straight start,
    so zero drive targets are the unbent wire."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdPhysics

    stage = get_current_stage()
    root = _root_xform(stage, prim_path, translation, orientation)
    anchor, device = f"{prim_path}/anchor", f"{prim_path}/device"
    _rigid(stage, anchor, (0.0, 0.0, 0.0), np.eye(3), BASE_MASS)
    mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(anchor))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-4, 1e-4, 1e-4))  # no shapes to take inertia from
    _rigid(stage, device, spec.device_at, np.eye(3), spec.device_mass)
    physics = _grippy(prim_path)
    for p in spec.parts:
        _cube(stage, f"{device}/{p.name}", p.size, p.at, _look(stage, f"{prim_path}/Looks/{p.name}", p.color),
              physics, offset=0.002)
    import isaaclab.sim as sim_utils

    wire_physics = f"{prim_path}/Looks/wire_physics"  # insulation on plastic and on itself: less grippy than pads
    mat = sim_utils.RigidBodyMaterialCfg(static_friction=spec.wire_friction, dynamic_friction=spec.wire_friction)
    mat.func(wire_physics, mat)
    ident = np.array([0.0, 0.0, 0.0, 1.0])

    def lock(prim, dofs) -> None:
        for dof in dofs:
            lim = UsdPhysics.LimitAPI.Apply(prim, dof)
            lim.CreateLowAttr(1.0)
            lim.CreateHighAttr(-1.0)  # low > high locks the axis

    def bend_joint(path: str, body0: str, body1: str, pos0, rot0, half: float, k: int) -> None:
        if spec.style == "d6":
            j = UsdPhysics.Joint.Define(stage, path)
        else:
            j = UsdPhysics.RevoluteJoint.Define(stage, path)
            j.CreateAxisAttr("Y" if k % 2 == 0 else "Z")
            j.CreateLowerLimitAttr(-math.degrees(spec.max_bend))
            j.CreateUpperLimitAttr(math.degrees(spec.max_bend))
        j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
        j.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in pos0)))
        j.CreateLocalRot0Attr(_quatf(rot0))
        j.CreateLocalPos1Attr(Gf.Vec3f(-half, 0.0, 0.0))
        j.CreateLocalRot1Attr(_quatf(ident))
        prim = j.GetPrim()
        if spec.style == "d6":
            lock(prim, ("transX", "transY", "transZ", "rotX"))  # no stretch, no twist
            dofs = ("rotY", "rotZ")
            for dof in dofs:
                lim = UsdPhysics.LimitAPI.Apply(prim, dof)
                lim.CreateLowAttr(-math.degrees(spec.max_bend))
                lim.CreateHighAttr(math.degrees(spec.max_bend))
        else:
            dofs = ("angular",)
        for dof in dofs:
            drive = UsdPhysics.DriveAPI.Apply(prim, dof)
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(spec.stiffness * math.pi / 180)  # USD angular gains are per degree
            drive.CreateDampingAttr(spec.damping * math.pi / 180)
            drive.CreateTargetPositionAttr(0.0)

    for n, w in enumerate(spec.wires):
        length, frames = spec.link_length(w), spec.link_frames(w)
        look = _look(stage, f"{prim_path}/Looks/{w.name}", w.color, 0.4)
        links = [f"{prim_path}/{spec.link_name(w.name, i)}" for i in range(w.segments)]
        for i, (path, c) in enumerate(zip(links, spec.link_centres(w))):
            body = _rigid(stage, path, c, frames[i], w.link_mass)
            body.CreateMaxDepenetrationVelocityAttr(0.5)
            body.CreateLinearDampingAttr(0.5)
            body.CreateAngularDampingAttr(2.0)
            _capsule(stage, f"{path}/wire", w.radius, length, look, wire_physics)
            # Joint frames: frame 1 is the link's own; frame 0 in the parent is turned so both agree at the start, so
            # a zero angle is the start shape (a pre-bent wire is relaxed as it spawns).
            if i == 0:
                bend_joint(f"{path}/bend", anchor, path, w.root, ik.matrix_to_quat_xyzw(frames[0]), length / 2, 0)
            else:
                rel = ik.matrix_to_quat_xyzw(frames[i - 1].T @ frames[i])
                bend_joint(f"{path}/bend", links[i - 1], path, (length / 2, 0.0, 0.0), rel, length / 2, i)
        end = UsdPhysics.FixedJoint.Define(stage, f"{links[-1]}/{'terminal' if n == 0 else 'loop_terminal'}")
        end.CreateBody0Rel().SetTargets([Sdf.Path(links[-1])])
        end.CreateBody1Rel().SetTargets([Sdf.Path(device)])
        end.CreateLocalPos0Attr(Gf.Vec3f(length / 2, 0.0, 0.0))
        end.CreateLocalRot0Attr(_quatf(ident))
        end.CreateLocalPos1Attr(Gf.Vec3f(*(float(v) for v in w.terminal)))
        # The device starts upright, so device frame = harness frame: the last link's start rotation.
        end.CreateLocalRot1Attr(_quatf(ik.matrix_to_quat_xyzw(frames[-1])))
        if n > 0:
            end.CreateExcludeFromArticulationAttr(True)
        pairs = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(device)).CreateFilteredPairsRel()
        for path in links[-spec.end_filter:] if spec.end_filter else []:
            pairs.AddTarget(Sdf.Path(path))
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    art = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    art.CreateEnabledSelfCollisionsAttr(True)  # wires push on each other and on the device; jointed neighbours skip
    art.CreateSolverPositionIterationCountAttr(64)
    art.CreateSolverVelocityIterationCountAttr(4)
    _weld_to_world(stage, prim_path, anchor)
    return root.GetPrim()


def _tie_parts(prim_path: str, spec: CableTie, translation=None, orientation=None):
    """The tie as one articulation (see :class:`CableTie`): the head is the root body (welded to the world when
    ``fixed``), the pawl and the first strap link hang off it, each link off the one before. Every joint's zero is the
    start shape, so zero drive targets are the relaxed strap and the pawl at its stop."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdPhysics

    stage = get_current_stage()
    root = _root_xform(stage, prim_path, translation, orientation)
    head = f"{prim_path}/head"
    hb = _rigid(stage, head, (0.0, 0.0, 0.0), np.eye(3), spec.head_mass)
    hb.CreateMaxDepenetrationVelocityAttr(spec.max_depenetration)
    hb.CreateLinearDampingAttr(spec.body_linear_damping)
    hb.CreateAngularDampingAttr(spec.body_angular_damping)
    physics = f"{prim_path}/Looks/nylon"
    mat = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction, dynamic_friction=spec.friction)
    mat.func(physics, mat)
    look_head = _look(stage, f"{prim_path}/Looks/head", spec.head_color, 0.6)
    look_strap = _look(stage, f"{prim_path}/Looks/strap", spec.color, 0.6)
    hx, hy, hz = spec.head_size
    (x0, x1), py, off = spec.passage_x, spec.passage_y, spec.contact_offset
    walls = {  # name: (size, centre), head frame: the passage runs through z
        "back": ((x0 + hx / 2, hy, hz), ((-hx / 2 + x0) / 2, 0.0, 0.0)),
        "front": ((hx / 2 - x1, hy, hz), ((hx / 2 + x1) / 2, 0.0, 0.0)),
        "side_pos": ((x1 - x0, hy / 2 - py, hz), ((x0 + x1) / 2, (hy / 2 + py) / 2, 0.0)),
        "side_neg": ((x1 - x0, hy / 2 - py, hz), ((x0 + x1) / 2, -(hy / 2 + py) / 2, 0.0)),
    }
    for name, (size, at) in walls.items():
        _cube(stage, f"{head}/{name}", size, at, look_head, physics, offset=off)

    def revolute(path, body0, body1, pos0, rot0, pos1, lower, upper):
        j = UsdPhysics.RevoluteJoint.Define(stage, path)
        j.CreateAxisAttr("Y")
        j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
        j.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in pos0)))
        j.CreateLocalRot0Attr(_quatf(ik.matrix_to_quat_xyzw(rot0)))
        j.CreateLocalPos1Attr(Gf.Vec3f(*(float(v) for v in pos1)))
        j.CreateLocalRot1Attr(_quatf((0.0, 0.0, 0.0, 1.0)))
        j.CreateLowerLimitAttr(float(math.degrees(lower)))
        j.CreateUpperLimitAttr(float(math.degrees(upper)))
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateTargetPositionAttr(0.0)
        return drive

    # the pawl: its body frame at the hinge, turned so its -z runs down the plate at the rest angle
    pawl = f"{prim_path}/pawl"
    rest = axis_angle_matrix(Y_AXIS, spec.pawl_rest)
    pb = _rigid(stage, pawl, spec.pawl_hinge, rest, spec.pawl_mass)
    pb.CreateMaxDepenetrationVelocityAttr(spec.max_depenetration)
    length = spec.pawl_len + spec.pawl_radius
    _cube(stage, f"{pawl}/plate", (2 * spec.pawl_radius, spec.pawl_width, length), (0.0, 0.0, -length / 2),
          look_head, physics, offset=off)
    d = revolute(f"{pawl}/pawl", head, pawl, spec.pawl_hinge, rest, (0.0, 0.0, 0.0), -spec.pawl_rest, 0.0)
    d.CreateStiffnessAttr(spec.pawl_spring * math.pi / 180)
    d.CreateDampingAttr(spec.pawl_damping * math.pi / 180)

    frames, centres, half = spec.link_frames(), spec.link_centres(), spec.link_length() / 2
    links = [f"{prim_path}/{spec.link_name(i)}" for i in range(spec.n_links)]
    for i, (path, c) in enumerate(zip(links, centres)):
        body = _rigid(stage, path, c, frames[i], spec.link_mass)
        body.CreateMaxDepenetrationVelocityAttr(spec.max_depenetration)
        body.CreateLinearDampingAttr(spec.body_linear_damping)
        body.CreateAngularDampingAttr(spec.body_angular_damping)
        tip = i == spec.n_links - 1
        size = (2 * half, spec.tip_width if tip else spec.width, spec.tip_thick if tip else spec.thick)
        _cube(stage, f"{path}/box", size, (0.0, 0.0, 0.0), look_strap, physics, offset=off)
        if i == 0:
            drive = revolute(f"{path}/{spec.joint_name(i)}", head, path, spec.points[0], frames[0], (-half, 0.0, 0.0),
                             -spec.max_bend, spec.max_bend)
        else:
            drive = revolute(f"{path}/{spec.joint_name(i)}", links[i - 1], path, (half, 0.0, 0.0),
                             frames[i - 1].T @ frames[i], (-half, 0.0, 0.0), -spec.max_bend, spec.max_bend)
        drive.CreateStiffnessAttr(spec.stiffness * math.pi / 180)
        drive.CreateDampingAttr(spec.damping * math.pi / 180)

    target = root.GetPrim() if spec.fixed else stage.GetPrimAtPath(head)
    UsdPhysics.ArticulationRootAPI.Apply(target)
    art = PhysxSchema.PhysxArticulationAPI.Apply(target)
    art.CreateEnabledSelfCollisionsAttr(True)  # the strap runs through its own head; jointed neighbours skip
    art.CreateSolverPositionIterationCountAttr(64)
    art.CreateSolverVelocityIterationCountAttr(8)
    if spec.fixed:
        _weld_to_world(stage, prim_path, head)
    return root.GetPrim()



def _strap_chain(stage, prim_path: str, link_name, joint_prefix: str, parent: str, parent_rot: np.ndarray, rel_points,
                 frames, spec, physics: str, look: str) -> list:
    """The webbing's links under ``prim_path``, each on a D6 (bend about its width, twist about its length) to the one
    before, the first to ``parent`` (whose rotation in the articulation's frame is ``parent_rot``) at rel_points[0].
    Every joint's zero is the start shape."""
    from pxr import Gf, Sdf, UsdPhysics

    half = float(np.linalg.norm(np.subtract(rel_points[1], rel_points[0]))) / 2
    centres = (np.asarray(rel_points[:-1], float) + np.asarray(rel_points[1:], float)) / 2
    links = [f"{prim_path}/{link_name(i)}" for i in range(len(frames))]
    for i, (path, c) in enumerate(zip(links, centres)):
        body = _rigid(stage, path, c, frames[i], spec.link_mass)
        body.CreateMaxDepenetrationVelocityAttr(getattr(spec, "max_depenetration", 0.3))
        body.CreateLinearDampingAttr(getattr(spec, "body_linear_damping", 0.3))
        body.CreateAngularDampingAttr(getattr(spec, "body_angular_damping", 1.0))
        _cube(stage, f"{path}/box", (2 * half, spec.width, spec.thick), (0.0, 0.0, 0.0), look, physics,
              offset=spec.contact_offset)
        if i == 0:
            body0, pos0, rot0 = parent, rel_points[0], np.asarray(parent_rot, float).T @ frames[0]
        else:
            body0, pos0, rot0 = links[i - 1], (half, 0.0, 0.0), frames[i - 1].T @ frames[i]
        _d6(stage, f"{path}/{joint_prefix}_{i:02d}", body0, path, pos0, rot0, (-half, 0.0, 0.0), spec.max_bend,
            spec.max_twist, spec.stiffness, spec.damping)
    return links


def _d6(stage, path: str, body0: str, body1: str, pos0, rot0, pos1, max_bend: float, max_twist: float,
        stiffness: float, damping: float) -> None:
    """A D6 joint that bends about y and twists about x (both sprung toward zero), everything else locked."""
    from pxr import Gf, Sdf, UsdPhysics

    j = UsdPhysics.Joint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    j.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in pos0)))
    j.CreateLocalRot0Attr(_quatf(ik.matrix_to_quat_xyzw(rot0)))
    j.CreateLocalPos1Attr(Gf.Vec3f(*(float(v) for v in pos1)))
    j.CreateLocalRot1Attr(_quatf((0.0, 0.0, 0.0, 1.0)))
    prim = j.GetPrim()
    for dof in ("transX", "transY", "transZ", "rotZ"):
        lim = UsdPhysics.LimitAPI.Apply(prim, dof)
        lim.CreateLowAttr(1.0)
        lim.CreateHighAttr(-1.0)  # low > high locks the axis
    for dof, most in (("rotY", max_bend), ("rotX", max_twist)):
        lim = UsdPhysics.LimitAPI.Apply(prim, dof)
        lim.CreateLowAttr(-math.degrees(most))
        lim.CreateHighAttr(math.degrees(most))
        drive = UsdPhysics.DriveAPI.Apply(prim, dof)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(stiffness * math.pi / 180)  # USD angular gains are per degree
        drive.CreateDampingAttr(damping * math.pi / 180)
        drive.CreateTargetPositionAttr(0.0)


def _strap_material(prim_path: str, friction: float) -> str:
    import isaaclab.sim as sim_utils

    physics = f"{prim_path}/Looks/webbing_physics"
    mat = sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction)
    mat.func(physics, mat)
    return physics


def _webbing_parts(prim_path: str, spec: Webbing, translation=None, orientation=None):
    """The webbing as one fixed-base articulation: the fitting (welded) and the links."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage = get_current_stage()
    root = _root_xform(stage, prim_path, translation, orientation)
    fit = f"{prim_path}/fitting"
    _rigid(stage, fit, (0.0, 0.0, 0.0), np.eye(3), 0.02)
    m = UsdPhysics.MassAPI(stage.GetPrimAtPath(fit))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-5, 1e-5, 1e-5))
    for p in spec.fitting_parts:
        _mesh_prim(stage, f"{fit}/{p.name}", p.mesh, p.at, _look(stage, f"{prim_path}/Looks/{p.name}", p.color,
                                                                   p.roughness))
    physics = _strap_material(prim_path, spec.friction)
    look = _look(stage, f"{prim_path}/Looks/webbing", spec.color, 0.8)
    rel = np.asarray(spec.points, float) - np.asarray(spec.pos, float)
    _strap_chain(stage, prim_path, spec.link_name, "wbend", fit, np.eye(3), rel, spec.link_frames(), spec, physics,
                 look)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    art = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    art.CreateEnabledSelfCollisionsAttr(True)  # the tail can lie on the strap; jointed neighbours skip
    art.CreateSolverPositionIterationCountAttr(64)
    art.CreateSolverVelocityIterationCountAttr(8)
    _weld_to_world(stage, prim_path, fit)
    return root.GetPrim()


def _ratchet_parts(prim_path: str, spec: Ratchet, translation=None, orientation=None):
    """The ratchet as one fixed-base articulation: hook (welded) -> fixed-end links -> housing -> handle."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    stage = get_current_stage()
    root = _root_xform(stage, prim_path, translation, orientation)
    hook = f"{prim_path}/hook"
    _rigid(stage, hook, (0.0, 0.0, 0.0), np.eye(3), 0.02)
    m = UsdPhysics.MassAPI(stage.GetPrimAtPath(hook))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-5, 1e-5, 1e-5))
    for p in spec.hook_parts:
        _mesh_prim(stage, f"{hook}/{p.name}", p.mesh, p.at, _look(stage, f"{prim_path}/Looks/{p.name}", p.color,
                                                                    p.roughness))
    physics = _strap_material(prim_path, spec.friction)
    metal = _grippy(prim_path, (0.5, 0.4))
    look = _look(stage, f"{prim_path}/Looks/webbing", spec.color, 0.8)
    rel = np.asarray(spec.points, float) - np.asarray(spec.pos, float)
    frames = spec.link_frames()
    links = _strap_chain(stage, prim_path, spec.link_name, "fbend", hook, np.eye(3), rel, frames, spec, physics, look)
    r = spec.rot()
    housing, handle = f"{prim_path}/housing", f"{prim_path}/handle"
    hpos = rel[-1] - r @ np.array([spec.base_x, 0.0, 0.0])
    _rigid(stage, housing, hpos, r, spec.housing_mass)
    for name, size, at, color in spec.housing_boxes:
        _cube(stage, f"{housing}/{name}", size, at, None if color is None else _look(
            stage, f"{prim_path}/Looks/{name}", color, 0.35), metal, offset=spec.contact_offset)
        if color is None:
            UsdGeom.Imageable(stage.GetPrimAtPath(f"{housing}/{name}")).MakeInvisible()
    for name, radius, length, at in spec.housing_pins:
        _mesh_prim(stage, f"{housing}/{name}", y_cylinder_mesh(radius, length), at,
                   _look(stage, f"{prim_path}/Looks/{name}", (0.62, 0.63, 0.66), 0.3), metal, offset=spec.contact_offset)
    for p in spec.housing_parts:
        _mesh_prim(stage, f"{housing}/{p.name}", p.mesh, p.at, _look(stage, f"{prim_path}/Looks/{p.name}", p.color,
                                                                       p.roughness))
    half = spec.link_length() / 2
    _d6(stage, f"{housing}/fbend_{len(links):02d}", links[-1], housing, (half, 0.0, 0.0), frames[-1].T @ r,
        (spec.base_x, 0.0, 0.0), spec.base_bend, spec.max_twist, spec.stiffness, spec.damping)
    pivot = np.asarray(spec.pivot, float)
    _rigid(stage, handle, hpos + r @ pivot, r, spec.handle_mass)
    for name, size, at, color in spec.handle_boxes:
        _cube(stage, f"{handle}/{name}", size, at, None if color is None else _look(
            stage, f"{prim_path}/Looks/h_{name}", color, 0.5), metal, offset=spec.contact_offset)
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{handle}/handle")
    j.CreateAxisAttr("Y")
    j.CreateBody0Rel().SetTargets([Sdf.Path(housing)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(handle)])
    j.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in pivot)))
    j.CreateLocalRot0Attr(_quatf((0.0, 0.0, 0.0, 1.0)))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(_quatf((0.0, 0.0, 0.0, 1.0)))
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(float(math.degrees(spec.handle_max)))
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(spec.handle_spring * math.pi / 180)
    drive.CreateDampingAttr(spec.handle_damping * math.pi / 180)
    drive.CreateTargetPositionAttr(0.0)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    art = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    art.CreateEnabledSelfCollisionsAttr(False)
    art.CreateSolverPositionIterationCountAttr(64)
    art.CreateSolverVelocityIterationCountAttr(8)
    _weld_to_world(stage, prim_path, hook)
    return root.GetPrim()


def _field_parts(prim_path: str, spec: ColorField, translation=None, orientation=None):
    """A colour field: one mesh of n * n tiles (four corners each), no material, displayColor per face."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, UsdGeom, Vt

    stage = get_current_stage()
    root = _root_xform(stage, prim_path, translation, orientation)
    n = spec.n
    shown = spec.shown0 or (False,) * (n * n)
    colors = spec.colors0 or (tuple(spec.color),) * (n * n)
    mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/tiles")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, q)) for q in spec.tile_points(shown)]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * (n * n)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(list(range(4 * n * n))))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    pv = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform)
    pv.Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, c)) for c in _linear(colors)]))
    return root.GetPrim()


_FIELD_LOG: list = []


def _linear(colors) -> np.ndarray:
    """sRGB (0..1) to the linear values displayColor takes (yam_arena.station.linear_rgb, for arrays)."""
    c = np.asarray(colors, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def set_field(environment, name: str, shown, colors) -> None:
    """Show or hide (``shown``, n x n bool) and recolour (``colors``, n x n x 3, sRGB 0..1) the tiles of the colour
    field ``name`` in env 0."""
    if environment is None:
        return
    spec = _SPECS.get(name)
    prim = _find_prim(environment, name, "tiles")
    if spec is None or prim is None:
        if not _FIELD_LOG:
            _FIELD_LOG.append(f"set_field: no field {name!r}")
            print(f"labkit: set_field: no field {name!r}", flush=True)
        return
    from pxr import UsdGeom, Vt

    pts = spec.tile_points(shown).astype(np.float32)
    cols = _linear(colors).astype(np.float32).reshape(-1, 3)
    mesh = UsdGeom.Mesh(prim)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts))
    mesh.GetDisplayColorPrimvar().Set(Vt.Vec3fArray.FromNumpy(cols))


def _spray_parts(prim_path: str, spec: SprayBottle, translation=None, orientation=None):
    """The spray bottle as an articulation: the housing body (round body, shroud, head; visual meshes) and the trigger
    body on its sprung prismatic joint, like :func:`_button_parts`."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    stage = get_current_stage()
    root = _root_xform(stage, prim_path, translation, orientation)
    physics = _grippy(prim_path, spec.friction)
    housing, trig = f"{prim_path}/housing", f"{prim_path}/trigger"
    _rigid(stage, housing, (0.0, 0.0, 0.0), np.eye(3), spec.mass)
    mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(housing))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*(float(v) for v in spec.com)))
    _mesh_prim(stage, f"{housing}/body", cylinder_mesh(spec.body_r, spec.body_h, 16), (0.0, 0.0, spec.body_h / 2),
               None, physics, offset=0.002)
    UsdGeom.Imageable(stage.GetPrimAtPath(f"{housing}/body")).MakeInvisible()  # the collider; the look is a part
    for name, size, at, color in spec.housing_boxes:
        look = None if color is None else _look(stage, f"{prim_path}/Looks/{name}", color, 0.35)
        _cube(stage, f"{housing}/{name}", size, at, look, physics, offset=0.002)
        if look is None:
            UsdGeom.Imageable(stage.GetPrimAtPath(f"{housing}/{name}")).MakeInvisible()
    for p in spec.housing_parts:
        _mesh_prim(stage, f"{housing}/{p.name}", p.mesh, p.at, _look(stage, f"{prim_path}/Looks/{p.name}", p.color,
                                                                        p.roughness))
        if not p.visible:
            UsdGeom.Imageable(stage.GetPrimAtPath(f"{housing}/{p.name}")).MakeInvisible()
    _rigid(stage, trig, spec.trigger_rest, np.eye(3), spec.trigger_mass)
    _cube(stage, f"{trig}/box", spec.trigger, (0.0, 0.0, 0.0), _look(stage, f"{prim_path}/Looks/trigger",
                                                                     spec.trigger_color, 0.4), physics, offset=0.002)
    joint = UsdPhysics.PrismaticJoint.Define(stage, f"{trig}/trigger")
    joint.CreateAxisAttr("X")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(housing)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(trig)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in spec.trigger_rest)))
    joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLowerLimitAttr(-spec.travel)
    joint.CreateUpperLimitAttr(0.0)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(spec.rate)
    drive.CreateDampingAttr(spec.damping)
    drive.CreateTargetPositionAttr(0.0)
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(housing))
    art = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(housing))
    art.CreateEnabledSelfCollisionsAttr(False)  # the trigger slides inside the head's shadow; the joint limit stops it
    art.CreateSolverPositionIterationCountAttr(32)
    return root.GetPrim()


def _cloth_object(spec: Cloth):
    """An Isaac Lab-Arena deformable object for ``spec``: its own folded mesh as a PhysX surface deformable."""
    import isaaclab.sim as sim_utils
    import trimesh
    from isaaclab.sim.spawners.meshes import meshes as mesh_spawners
    from isaaclab.sim.utils import clone, get_current_stage
    from isaaclab_arena.assets.deformable_object import DeformableObject
    from isaaclab_arena.utils.pose import Pose as ArenaPose
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxDeformableBodyPropertiesCfg
    from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg

    from yam_arena.station import linear_rgb

    nodes, faces = spec.mesh(world=False)
    centre = nodes.mean(0)  # Arena resets a deformable's nodes about its prim origin, so put the origin at the centroid

    @clone
    def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
        mesh = trimesh.Trimesh(vertices=nodes - centre, faces=faces, process=False)
        stage = get_current_stage()
        mesh_spawners._spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation, None, stage=stage)
        if spec.quarter:
            # By default PhysX takes a cloth's rest bend angles as flat, so a folded cloth with any stiffness springs
            # open. "restShapeDefault" takes them from the spawn shape instead: the folds become pressed creases.
            from pxr import Sdf, Usd

            done = 0
            for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
                if prim.GetAttribute("omniphysics:restShapePoints").IsValid():
                    attr = prim.GetAttribute("omniphysics:restBendAnglesDefault")
                    if not attr.IsValid():
                        attr = prim.CreateAttribute("omniphysics:restBendAnglesDefault", Sdf.ValueTypeNames.Token, False,
                                                    Sdf.VariabilityUniform)
                    attr.Set("restShapeDefault")
                    done += 1
            print(f"[labkit] cloth {spec.name}: rest bend angles taken from the folded shape on {done} prim(s)", flush=True)
        return stage.GetPrimAtPath(prim_path)

    material = PhysxSurfaceDeformableBodyMaterialCfg(
        density=spec.density, surface_thickness=spec.thickness, poissons_ratio=spec.poisson, youngs_modulus=spec.youngs,
        surface_bend_stiffness=spec.bend, elasticity_damping=spec.damping, bend_damping=spec.bend_damping,
        static_friction=spec.friction, dynamic_friction=spec.friction)
    cfg = sim_utils.MeshRectangleCfg(
        func=spawn, size=(float(spec.sheet[0]), float(spec.sheet[1])), edge_refinement=1.0, physics_material=material,
        deformable_props=PhysxDeformableBodyPropertiesCfg(solver_position_iteration_count=int(spec.iterations),
                                                          linear_damping=0.0,
                                                          self_collision=True if (spec.quarter or spec.self_collide) else None),
        collision_props=[PhysxCollisionCfg(rest_offset=spec.rest_offset, contact_offset=spec.contact_offset)],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.8))
    origin = centre @ _yaw_matrix(spec.yaw).T + np.asarray(spec.pos, float)
    return DeformableObject(name=spec.name, spawner_cfg=cfg,
                            initial_pose=ArenaPose(position_xyz=tuple(float(v) for v in origin),
                                                   rotation_xyzw=tuple(float(v) for v in _yaw_quat(spec.yaw))))



def _arena_object(spec: KitObject):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.shapes import spawn_cuboid
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose as ArenaPose
    from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg, PhysxRigidBodyPropertiesCfg

    from yam_arena.constants import GRIPPY_FRICTION
    from yam_arena.station import linear_rgb

    if isinstance(spec, Cloth):
        return _cloth_object(spec)
    pose = ArenaPose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=tuple(float(v) for v in _yaw_quat(spec.yaw)))
    kind = ObjectType.RIGID if spec.mass is not None else ObjectType.BASE
    body = dict(
        collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=GRIPPY_FRICTION[0],
                                                        dynamic_friction=GRIPPY_FRICTION[1]),
    )
    if spec.mass is not None:
        body.update(rigid_props=PhysxRigidBodyPropertiesCfg(), mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass))

    if isinstance(spec, Container | Marked | Hand):
        parts = _container_parts if isinstance(spec, Container) else _hand_parts if isinstance(spec, Hand) \
            else _mark_parts

        @clone
        def spawn_with_parts(prim_path: str, cfg, *args, **kwargs):
            prim = spawn_cuboid.__wrapped__(prim_path, cfg, *args, **kwargs)  # the body and its first collider
            parts(prim_path, spec)
            return prim

        size = (spec.size[0], spec.size[1], spec.wall) if isinstance(spec, Container) else spec.size
        cfg = sim_utils.CuboidCfg(
            size=size, func=spawn_with_parts,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.4), **body)
        return Object(name=spec.name, object_type=kind, spawner_cfg=cfg, initial_pose=pose)

    if isinstance(spec, Model):
        cfg = sim_utils.UsdFileCfg(usd_path=spec.usd, scale=(spec.scale,) * 3, **{
            k: v for k, v in body.items() if k != "physics_material"})
        return Object(name=spec.name, object_type=kind, spawner_cfg=cfg, initial_pose=pose)

    if isinstance(spec, Lamp):
        cfg = sim_utils.CylinderCfg(
            radius=spec.radius,
            height=spec.size[2],
            visible=False,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.2,
                                                        emissive_color=linear_rgb(spec.color)),
        )
        return Object(name=spec.name, object_type=ObjectType.BASE, spawner_cfg=cfg, initial_pose=pose)

    if isinstance(spec, HangingBulb):

        @configclass
        class BulbSpawnerCfg(SpawnerCfg):
            func: object = clone(lambda prim_path, cfg, *a, translation=None, orientation=None, **k: _bulb_parts(
                prim_path, spec, translation, orientation))

        class BulbObject(Object):
            def _generate_articulation_cfg(self):
                # Implicit actuators are the springs: the swing (strap and tubing), the faces, the cap's tether. Each
                # joint's zero is its rest, so the zero target Isaac Lab holds is the spring's rest.
                acts = {
                    "swing": ImplicitActuatorCfg(joint_names_expr=["swing_.*"], stiffness=spec.swing_spring,
                                                 damping=spec.swing_damping),
                    "squeeze": ImplicitActuatorCfg(joint_names_expr=["squeeze_.*"], stiffness=spec.spring,
                                                   damping=spec.face_damping),
                    "cap": ImplicitActuatorCfg(joint_names_expr=["cap_hinge"], stiffness=spec.cap_spring,
                                               damping=spec.cap_damping),
                }
                cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=acts)
                cfg = self._add_initial_pose_to_cfg(cfg)
                cfg.init_state.joint_pos = {"swing_.*": 0.0, "squeeze_.*": 0.0,
                                            "cap_hinge": spec.cap_start - spec.cap_rest}
                return cfg

        return BulbObject(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=BulbSpawnerCfg(),
                          initial_pose=pose)

    if isinstance(spec, Harness):

        @configclass
        class HarnessSpawnerCfg(SpawnerCfg):
            func: object = clone(lambda prim_path, cfg, *a, translation=None, orientation=None, **k: _harness_parts(
                prim_path, spec, translation, orientation))

        class HarnessObject(Object):
            def _generate_articulation_cfg(self):
                # Every joint is a wire bend: one implicit actuator, its targets the plastic rest angles (zero =
                # straight), which labkit's step hook drags along (_harness_plastic).
                # With torque_cap, a joint's drive can't push back harder than copper's yield moment at any PhysX
                # substep (the plastic rule itself only runs once per control step).
                fields = getattr(ImplicitActuatorCfg, "__dataclass_fields__", {})
                key = "effort_limit_sim" if not fields or "effort_limit_sim" in fields else "effort_limit"
                cap = {} if spec.torque_cap is None else {key: spec.torque_cap}
                act = ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=spec.stiffness, damping=spec.damping,
                                          armature=spec.armature, **cap)
                cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators={"wire": act})
                cfg = self._add_initial_pose_to_cfg(cfg)
                cfg.init_state.joint_pos = {".*": 0.0}
                return cfg

        return HarnessObject(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=HarnessSpawnerCfg(),
                             initial_pose=pose)

    if isinstance(spec, CableTie):

        @configclass
        class TieSpawnerCfg(SpawnerCfg):
            func: object = clone(lambda prim_path, cfg, *a, translation=None, orientation=None, **k: _tie_parts(
                prim_path, spec, translation, orientation))

        class TieObject(Object):
            def _generate_articulation_cfg(self):
                # Implicit actuators are the springs: every strap bend toward the start shape, the pawl toward its stop.
                acts = {"strap": ImplicitActuatorCfg(joint_names_expr=["bend_.*"], stiffness=spec.stiffness,
                                                     damping=spec.damping, armature=spec.armature),
                        "pawl": ImplicitActuatorCfg(joint_names_expr=["pawl"], stiffness=spec.pawl_spring,
                                                    damping=spec.pawl_damping)}
                cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=acts)
                cfg = self._add_initial_pose_to_cfg(cfg)
                cfg.init_state.joint_pos = {".*": 0.0}
                return cfg

        tie_pose = ArenaPose(position_xyz=tuple(float(v) for v in spec.pos),
                             rotation_xyzw=tuple(float(v) for v in spec.quat))
        return TieObject(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=TieSpawnerCfg(),
                         initial_pose=tie_pose)


    if isinstance(spec, Tool):

        @clone
        def spawn_tool(prim_path: str, cfg, *args, **kwargs):
            prim = spawn_cuboid.__wrapped__(prim_path, cfg, *args, **kwargs)  # the body and its first box
            _tool_parts(prim_path, spec)
            return prim

        cfg = sim_utils.CuboidCfg(
            size=spec.size, func=spawn_tool,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.35),
            **{**body, "collision_props": PhysxCollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)})
        tool_pose = ArenaPose(position_xyz=tuple(float(v) for v in spec.pos),
                              rotation_xyzw=tuple(float(v) for v in spec.quat))
        return Object(name=spec.name, object_type=ObjectType.RIGID, spawner_cfg=cfg, initial_pose=tool_pose)

    if isinstance(spec, Webbing | Ratchet):
        parts_fn = _webbing_parts if isinstance(spec, Webbing) else _ratchet_parts

        @configclass
        class StrapSpawnerCfg(SpawnerCfg):
            func: object = clone(lambda prim_path, cfg, *a, translation=None, orientation=None, **k: parts_fn(
                prim_path, spec, translation, orientation))

        class StrapObject(Object):
            def _generate_articulation_cfg(self):
                # Implicit actuators are the springs: every webbing bend and twist toward the start shape; the
                # ratchet's handle is damped, on its light return spring.
                acts = {"webbing": ImplicitActuatorCfg(joint_names_expr=["(?!handle$).*"], stiffness=spec.stiffness,
                                                       damping=spec.damping, armature=spec.armature)}
                if isinstance(spec, Ratchet):
                    acts["handle"] = ImplicitActuatorCfg(joint_names_expr=["handle"], stiffness=spec.handle_spring,
                                                         damping=spec.handle_damping)
                cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=acts)
                cfg = self._add_initial_pose_to_cfg(cfg)
                cfg.init_state.joint_pos = {".*": 0.0}
                return cfg

        strap_pose = ArenaPose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        return StrapObject(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=StrapSpawnerCfg(),
                           initial_pose=strap_pose)

    if isinstance(spec, KeyedPulley):

        @configclass
        class PulleySpawnerCfg(SpawnerCfg):
            func: object = clone(lambda prim_path, cfg, *a, translation=None, orientation=None, **k: _pulley_parts(
                prim_path, spec, translation, orientation))

        class PulleyObject(Object):
            def _generate_articulation_cfg(self):
                # No springs: the slide and turn are free (damped); task code holds them (labkit.hold_joint).
                acts = {"slide": ImplicitActuatorCfg(joint_names_expr=["slide"], stiffness=0.0,
                                                     damping=spec.slide_damping),
                        "turn": ImplicitActuatorCfg(joint_names_expr=["turn"], stiffness=0.0,
                                                    damping=spec.turn_damping)}
                cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=acts)
                cfg = self._add_initial_pose_to_cfg(cfg)
                cfg.init_state.joint_pos = {"slide": 0.0, "turn": spec.turn_start}
                return cfg

        return PulleyObject(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=PulleySpawnerCfg(),
                            initial_pose=pose)

    if isinstance(spec, ColorField):

        @configclass
        class FieldSpawnerCfg(SpawnerCfg):
            func: object = clone(lambda prim_path, cfg, *a, translation=None, orientation=None, **k: _field_parts(
                prim_path, spec, translation, orientation))

        return Object(name=spec.name, object_type=ObjectType.BASE, spawner_cfg=FieldSpawnerCfg(), initial_pose=pose)

    if isinstance(spec, Prop):

        @configclass
        class PropSpawnerCfg(SpawnerCfg):
            func: object = clone(lambda prim_path, cfg, *a, translation=None, orientation=None, **k: _prop_parts(
                prim_path, spec, translation, orientation))

        return Object(name=spec.name, object_type=ObjectType.BASE, spawner_cfg=PropSpawnerCfg(), initial_pose=pose)

    if isinstance(spec, Hinge | Slider):
        joint = "hinge" if isinstance(spec, Hinge) else "slide"

        @configclass
        class JointSpawnerCfg(SpawnerCfg):
            func: object = clone(lambda prim_path, cfg, *a, translation=None, orientation=None, **k: _jointed_parts(
                prim_path, spec, translation, orientation))

        class JointObject(Object):
            def _generate_articulation_cfg(self):
                # The implicit actuator is the spring (stiffness 0 without one); the joint's zero is spec.rest, so
                # the zero target Isaac Lab holds for an undriven joint is the spring's rest.
                act = ImplicitActuatorCfg(joint_names_expr=[joint], stiffness=spec.spring or 0.0,
                                          damping=spec.damping, friction=spec.friction)
                cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators={"joint": act})
                cfg = self._add_initial_pose_to_cfg(cfg)
                cfg.init_state.joint_pos = {joint: spec.start - spec.rest}
                return cfg

        return JointObject(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=JointSpawnerCfg(),
                           initial_pose=pose)

    if isinstance(spec, Towel):

        @configclass
        class TowelSpawnerCfg(SpawnerCfg):
            func: object = clone(lambda prim_path, cfg, *a, translation=None, orientation=None, **k: _towel_parts(
                prim_path, spec, translation, orientation))

        class TowelObject(Object):
            def _generate_articulation_cfg(self):
                act = ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=spec.stiffness, damping=spec.damping,
                                          armature=spec.armature)
                cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators={"cloth": act})
                return self._add_initial_pose_to_cfg(cfg)

        root = spec.start_points()[spec.tile_ids().index(spec.centre_tile)]
        return TowelObject(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=TowelSpawnerCfg(),
                           initial_pose=ArenaPose(position_xyz=tuple(float(v) for v in root),
                                                  rotation_xyzw=tuple(float(v) for v in _yaw_quat(spec.yaw))))

    if isinstance(spec, SprayBottle):

        @configclass
        class SpraySpawnerCfg(SpawnerCfg):
            func: object = clone(lambda prim_path, cfg, *a, translation=None, orientation=None, **k: _spray_parts(
                prim_path, spec, translation, orientation))

        class SprayObject(Object):
            def _generate_articulation_cfg(self):
                # The implicit actuator is the trigger's return spring (target 0: all the way out).
                spring = ImplicitActuatorCfg(joint_names_expr=["trigger"], stiffness=spec.rate, damping=spec.damping)
                cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators={"spring": spring})
                cfg = self._add_initial_pose_to_cfg(cfg)
                cfg.init_state.joint_pos = {"trigger": 0.0}
                return cfg

        return SprayObject(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=SpraySpawnerCfg(),
                           initial_pose=pose)

    @configclass
    class ButtonSpawnerCfg(SpawnerCfg):
        func: object = clone(lambda prim_path, cfg, *a, translation=None, orientation=None, **k: _button_parts(
            prim_path, spec, translation, orientation))

    class ButtonObject(Object):
        def _generate_articulation_cfg(self):
            # The implicit actuator is the spring: its target is the cap's rest stop.
            spring = ImplicitActuatorCfg(joint_names_expr=["press"], stiffness=spec.rate, damping=spec.damping)
            cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators={"spring": spring})
            return self._add_initial_pose_to_cfg(cfg)

    return ButtonObject(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=ButtonSpawnerCfg(),
                        initial_pose=pose)


# -- state and lamps (Isaac Sim) --------------------------------------------------------------------------------------


def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _box_state(spec, pos, quat, lin=np.zeros(3), ang=np.zeros(3)) -> ObjectState:
    corners = spec.corners(pos, quat) if hasattr(spec, "corners") else _box_corners((0, 0, 0), spec.size, pos, quat)
    return ObjectState(pos=np.asarray(pos, float), quat_xyzw=np.asarray(quat, float), lin_vel=np.asarray(lin, float),
                       ang_vel=np.asarray(ang, float), low=corners.min(0), high=corners.max(0))


def start_state(spec) -> ObjectState:
    """The state ``spec`` starts in, at rest: what a fixed object always reports, and what tests can build on."""
    if isinstance(spec, Hinge | Slider):
        return spec.state_at(spec.start)
    if isinstance(spec, KeyedPulley | Webbing | Ratchet):
        return spec.state_at()
    if isinstance(spec, Tool):
        return _box_state(spec, np.asarray(spec.pos, float), np.asarray(spec.quat, float))
    if isinstance(spec, HangingBulb):
        return spec.state_at(np.asarray(spec.pos, float), _yaw_quat(spec.yaw), {})
    if isinstance(spec, Towel):
        return spec.state_at()
    if isinstance(spec, Harness | CableTie):
        return spec.state_at()
    if isinstance(spec, Hand):
        return spec.state()
    if isinstance(spec, Cloth):
        return spec.state_at()
    if isinstance(spec, SprayBottle):
        return spec.state_at(np.asarray(spec.pos, float), _yaw_quat(spec.yaw))
    s = _box_state(spec, np.asarray(spec.pos, float), _yaw_quat(spec.yaw))
    if isinstance(spec, Button):
        half = np.asarray(spec.size, float) / 2
        return ButtonState(pos=s.pos, quat_xyzw=s.quat_xyzw, lin_vel=s.lin_vel, ang_vel=s.ang_vel,
                           low=s.pos - half, high=s.pos + half, travel=0.0)
    return s


def _towel_state(environment, spec: Towel) -> TowelState:
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    names = list(asset.body_names)
    idx = [names.index(spec.tile_name(i, j)) for i, j in spec.tile_ids()]
    return spec.state_at(_np(asset.data.body_pos_w)[0][idx] - origin, _np(asset.data.body_quat_w)[0][idx],
                         _np(asset.data.body_lin_vel_w)[0][idx], _np(asset.data.body_ang_vel_w)[0][idx])


def _harness_state(environment, spec: Harness) -> HarnessState:
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    names = list(asset.body_names)
    pos, quat = _np(asset.data.body_pos_w)[0] - origin, _np(asset.data.body_quat_w)[0]
    links = {}
    for w in spec.wires:
        idx = [names.index(spec.link_name(w.name, i)) for i in range(w.segments)]
        links[w.name] = (pos[idx], quat[idx])
    d = names.index("device")
    return spec.state_at(pos[d], quat[d], links, _np(asset.data.body_lin_vel_w)[0][d],
                         _np(asset.data.body_ang_vel_w)[0][d])


_PLASTIC: dict = {}  # harness name -> its joints' current rest angles (rad), cleared at reset


def harness_rest(name: str) -> np.ndarray | None:
    """The plastic rest angles of harness ``name``'s joints (the permanent bends), or None before its first step."""
    return _PLASTIC.get(name)


def _harness_plastic(environment) -> None:
    """Every control step: drag each harness joint's rest angle to within its yield angle of where the joint is, and
    give the actuators the new rest angles as targets (whole vectors, in the articulation's joint order)."""
    import torch

    base = getattr(environment, "unwrapped", environment)
    for name, spec in _SPECS.items():
        if not isinstance(spec, Harness):
            continue
        asset = base.scene[name]
        q = _np(asset.data.joint_pos)[0]
        rest = _PLASTIC.get(name, np.zeros_like(q))
        new = plastic_targets(q, rest, spec.yield_angle)
        _PLASTIC[name] = new
        if not np.array_equal(new, rest) or name not in _WROTE:
            _call_first(asset, ("set_joint_position_target_index", "set_joint_position_target"),
                        torch.tensor(new[None, :], dtype=torch.float32, device=asset.device))
            _WROTE.add(name)


_WROTE: set = set()
_YIELD: dict = {}  # cable tie name -> yield angle (rad) set while the sim runs, overriding the spec's; None = elastic


def set_tie_yield(name: str, angle: float | None) -> None:
    """Make cable tie ``name``'s strap elastic-perfectly-plastic with yield angle ``angle`` per joint (rad), or purely
    elastic (None), from the next control step. Nylon takes a set: bent past its yield, a strap stays bent, and pulled
    straight through a head, a curl it was spawned with straightens instead of springing back to a kink. Lasts until
    changed (resets keep it, and clear the set it took)."""
    if name in _SPECS and not isinstance(_SPECS[name], CableTie):  # may be set before the scene registers it
        raise KeyError(f"{name!r} is not a cable tie")
    _YIELD[name] = None if angle is None else float(angle)


def tie_yield(name: str) -> float | None:
    return _YIELD.get(name, getattr(_SPECS.get(name), "yield_angle", None))


def _tie_goals(spec, name: str, joint_names) -> list:
    """Every drive target of a cable tie, in ``joint_names`` order (as _apply_drives writes them)."""
    rest = _PLASTIC.get(name)
    out = []
    for i, joint in enumerate(joint_names):
        value = _HELD.get((name, joint))
        if value is not None:
            out.append(value - _joint_rest(spec, joint))
        elif rest is not None and joint.startswith("bend_"):
            out.append(float(rest[i]))
        else:
            out.append(0.0)
    return out


def _tie_plastic(environment) -> None:
    """Every control step, for a cable tie with a yield angle: drag each strap joint's rest angle to within it of where
    the joint is (the harness rule), and give the drives the new targets (held joints keep theirs)."""
    import torch

    base = getattr(environment, "unwrapped", environment)
    for name, spec in _SPECS.items():
        if not isinstance(spec, CableTie):
            continue
        y = tie_yield(name)
        if y is None:
            continue
        asset = base.scene[name]
        q = _np(asset.data.joint_pos)[0]
        bend = np.array([j.startswith("bend_") for j in asset.joint_names])
        rest = _PLASTIC.get(name, np.zeros_like(q))
        new = rest.copy()
        new[bend] = plastic_targets(q[bend], rest[bend], y)
        _PLASTIC[name] = new
        if not np.array_equal(new, rest):
            _call_first(asset, ("set_joint_position_target_index", "set_joint_position_target"),
                        torch.tensor([_tie_goals(spec, name, asset.joint_names)], dtype=torch.float32,
                                     device=asset.device))


def tie_set(name: str) -> np.ndarray | None:
    """The set (plastic rest angles, rad, articulation joint order) cable tie ``name``'s strap has taken this trial."""
    return _PLASTIC.get(name)


UNCAPPED = 1.0e6  # N m: what set_harness_torque_cap writes for "no cap"


def set_harness_torque_cap(environment, name: str, cap: float | None) -> dict:
    """Set every joint drive's torque limit of harness ``name`` while the sim runs (None: no cap), and read it back.
    Returns which writer worked and what the articulation now reports, or raises if no writer worked."""
    import torch

    base = getattr(environment, "unwrapped", environment)
    asset = base.scene[name]
    n = len(asset.joint_names)
    value = torch.full((1, n), UNCAPPED if cap is None else float(cap), dtype=torch.float32, device=asset.device)
    errors = []
    for fn_name in ("write_joint_effort_limit_to_sim_index", "write_joint_effort_limit_to_sim"):
        fn = getattr(asset, fn_name, None)
        if fn is None:
            continue
        for kw in ("limits", "effort_limits", None):
            try:
                fn(value) if kw is None else fn(**{kw: value})
            except TypeError as err:
                errors.append(f"{fn_name}({kw}): {err}")
                continue
            back = None
            for attr in ("joint_effort_limits", "joint_effort_limit"):
                got = getattr(asset.data, attr, None)
                if got is not None:
                    try:  # only a read-back for the log: the write itself already worked
                        back = [float(v) for v in _np(got)[0][:3]]
                    except Exception as err:  # noqa: BLE001
                        back = f"unreadable: {type(err).__name__}"
                    break
            return {"writer": f"{fn_name}({kw or 'positional'})", "set": None if cap is None else float(cap),
                    "reads_back_first3": back}
    raise RuntimeError(f"no effort-limit writer worked on {type(asset).__name__}: {errors or 'none found'}")


def _tie_state(environment, spec: CableTie) -> CableTieState:
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    names = list(asset.body_names)
    pos, quat = _np(asset.data.body_pos_w)[0] - origin, _np(asset.data.body_quat_w)[0]
    lin = _np(asset.data.body_lin_vel_w)[0]
    idx = [names.index(spec.link_name(i)) for i in range(spec.n_links)]
    h = names.index("head")
    q = _np(asset.data.joint_pos)[0]
    pawl = float(q[list(asset.joint_names).index("pawl")]) + spec.pawl_rest
    return spec.state_at(pos[h], quat[h], (pos[idx], quat[idx]), lin[h], _np(asset.data.body_ang_vel_w)[0][h], pawl,
                         float(np.max(np.linalg.norm(lin[idx], axis=1))))


_TIE_RESET: list = []  # what the last resets of a free tie did (the probe logs it)


def tie_reset_log() -> list:
    return list(_TIE_RESET)


def _reset_tie_root(base, asset) -> None:
    """Put a free tie's head back where it starts, at rest: Isaac Lab-Arena's reset may leave an articulation's root
    where the last trial put it. Tries Isaac Lab 3's writers, then the older names; records which worked."""
    import torch

    data = asset.data
    state = getattr(data, "default_root_state", None)
    if state is None:
        _TIE_RESET.append("no default_root_state")
        return
    state = getattr(state, "torch", state).clone()
    origins = base.scene.env_origins
    state[:, :3] += getattr(origins, "torch", origins)[: state.shape[0]]
    pose, vel = state[:, :7], torch.zeros_like(state[:, 7:13])
    done = []
    for names, value, kws in (
            (("write_root_link_pose_to_sim_index", "write_root_pose_to_sim_index", "write_root_link_pose_to_sim",
              "write_root_pose_to_sim"), pose, ("root_pose", "pose", None)),
            (("write_root_com_velocity_to_sim_index", "write_root_velocity_to_sim_index",
              "write_root_com_velocity_to_sim", "write_root_velocity_to_sim"), vel, ("root_velocity", "velocity", None))):
        ok = None
        for n in names:
            fn = getattr(asset, n, None)
            if fn is None:
                continue
            for kw in kws:
                try:
                    fn(value) if kw is None else fn(**{kw: value})
                    ok = f"{n}({kw or 'positional'})"
                    break
                except TypeError:
                    continue
            if ok:
                break
        done.append(ok or f"none of {names[0]}..")
    _TIE_RESET.append(done)


def _webbing_state(environment, spec: Webbing) -> WebbingState:
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins)[0]
    names = list(asset.body_names)
    idx = [names.index(spec.link_name(i)) for i in range(spec.n_links)]
    pos, quat = _np(asset.data.body_pos_w)[0] - origin, _np(asset.data.body_quat_w)[0]
    lin = _np(asset.data.body_lin_vel_w)[0]
    return spec.state_at((pos[idx], quat[idx]), float(np.max(np.linalg.norm(lin[idx], axis=1))))


def _ratchet_state(environment, spec: Ratchet) -> RatchetState:
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins)[0]
    names = list(asset.body_names)
    pos, quat = _np(asset.data.body_pos_w)[0] - origin, _np(asset.data.body_quat_w)[0]
    h = names.index("housing")
    joints = list(asset.joint_names)
    q, v = _np(asset.data.joint_pos)[0], _np(asset.data.joint_vel)[0]
    k = joints.index("handle")
    idx = [names.index(spec.link_name(i)) for i in range(spec.n_links)]
    half = spec.link_length() / 2
    axes = np.array([quat_xyzw_to_matrix(qi)[:, 0] for qi in quat[idx]])
    poly = np.vstack([pos[idx][:1] - half * axes[:1], pos[idx] + half * axes])
    return spec.state_at(pos[h], quat[h], float(q[k]), float(v[k]), _np(asset.data.body_lin_vel_w)[0][h],
                         _np(asset.data.body_ang_vel_w)[0][h], poly)


_CLOTH_NODES: dict[str, int] = {}


def _cloth_state(environment, spec: Cloth) -> ClothState:
    base = environment.unwrapped
    origin = _np(base.scene.env_origins[0])
    nodes = _np(base.arena_world.get_nodal_positions_w(spec.name)[0]) - origin
    vel = _np(base.arena_world.get_nodal_velocities_w(spec.name)[0])
    if spec.name not in _CLOTH_NODES:  # say once what the simulator built, to compare with spec.node_count
        _CLOTH_NODES[spec.name] = len(nodes)
        print(f"[labkit] cloth {spec.name}: {len(nodes)} nodes in the simulator, {spec.node_count} in the mesh", flush=True)
    return spec.state_at(nodes, vel)




def _state(environment, spec: KitObject) -> ObjectState:
    if isinstance(spec, Cloth):
        return _cloth_state(environment, spec)
    if isinstance(spec, Webbing):
        return _webbing_state(environment, spec)
    if isinstance(spec, Ratchet):
        return _ratchet_state(environment, spec)
    if isinstance(spec, Lamp | Hand | ColorField):
        return start_state(spec)
    if isinstance(spec, Towel):
        return _towel_state(environment, spec)
    if isinstance(spec, Harness):
        return _harness_state(environment, spec)
    if isinstance(spec, CableTie):
        return _tie_state(environment, spec)
    base = environment.unwrapped
    if isinstance(spec, HangingBulb):  # the anchor is welded; read the core's pose and every joint
        asset = base.scene[spec.name]
        q = _np(asset.data.joint_pos)[0]
        joints = {n: float(q[i]) + spec.joint_rest(n) for i, n in enumerate(asset.joint_names)}
        pose = _np(asset.data.body_link_pose_w)[0][list(asset.body_names).index("core")]
        was = _LATCH.get(spec.name, False)
        latched = spec.cap_latch.step(joints.get("cap_hinge", spec.cap_start), was)
        _LATCH[spec.name] = latched
        return spec.state_at(pose[:3] - _np(base.scene.env_origins)[0], pose[3:7], joints, latched)
    if isinstance(spec, KeyedPulley):  # the shaft is welded: read the two joints
        asset = base.scene[spec.name]
        names = list(asset.joint_names)
        q, v = _np(asset.data.joint_pos)[0], _np(asset.data.joint_vel)[0]
        i, j = names.index("slide"), names.index("turn")
        return spec.state_at(float(q[i]), float(q[j]), float(v[i]), float(v[j]))
    if isinstance(spec, Hinge | Slider):  # the base is welded in place: only the joint moves
        asset = base.scene[spec.name]
        joint = "hinge" if isinstance(spec, Hinge) else "slide"
        q = float(_np(asset.data.joint_pos)[0][list(asset.joint_names).index(joint)]) + spec.rest
        return hinge_step(spec, q) if isinstance(spec, Hinge) else spec.state_at(q)
    if spec.mass is None:
        return start_state(spec)
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    vel = _np(asset.data.root_vel_w)[0]
    pos, quat = pose[:3] - origin, pose[3:7]
    if isinstance(spec, SprayBottle):
        q = _np(asset.data.joint_pos)[0]
        travel = max(0.0, -float(q[list(asset.joint_names).index("trigger")]))
        return spec.state_at(pos, quat, travel, vel[:3], vel[3:6])
    if not isinstance(spec, Button):
        return _box_state(spec, pos, quat, vel[:3], vel[3:6])
    q = _np(asset.data.joint_pos)[0]
    travel = max(0.0, -float(q[list(asset.joint_names).index("press")]))
    half = np.asarray(spec.size, float) / 2
    return ButtonState(pos=pos, quat_xyzw=quat, lin_vel=vel[:3], ang_vel=vel[3:6], low=pos - half, high=pos + half,
                       travel=travel)


_PRIMS: dict = {}  # (name, child) -> prim, found once


def _find_prim(environment, name: str, child: str = ""):
    """Object ``name``'s prim (or its ``child``) in env 0, or None."""
    if (name, child) in _PRIMS and _PRIMS[(name, child)].IsValid():
        return _PRIMS[(name, child)]
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    root = getattr(environment.unwrapped.scene, "env_prim_paths", ["/World/envs/env_0"])[0]
    prim = stage.GetPrimAtPath(f"{root}/{name}")
    if not prim.IsValid():  # fall back to finding it by name under env 0
        prim = next((p for p in stage.Traverse() if p.GetName() == name and str(p.GetPath()).startswith(root)), prim)
    if prim.IsValid() and child:
        prim = stage.GetPrimAtPath(prim.GetPath().AppendPath(child))
    if not prim.IsValid():
        return None
    _PRIMS[(name, child)] = prim
    return prim


def _set_visible(environment, name: str, show: bool, child: str = "") -> None:
    """Show or hide object ``name``'s prim (or its ``child``) in env 0."""
    from pxr import UsdGeom

    prim = _find_prim(environment, name, child)
    if prim is not None:
        image = UsdGeom.Imageable(prim)
        image.MakeVisible() if show else image.MakeInvisible()


def _show_lamps(environment, on: bool) -> None:
    for name, spec in _SPECS.items():
        if isinstance(spec, Lamp):
            _set_visible(environment, name, on and (spec.group is None or spec.group == _GROUP["active"]))
    _LIT["on"] = on


def _show_gloves(environment) -> None:
    for name, spec in _SPECS.items():
        if isinstance(spec, Hand):
            _set_visible(environment, name, glove_on(spec), "glove")


def _after_step(environment, t: float | None = None) -> None:
    tick(_CLOCK["t"] + STEP_S if t is None else t, environment)  # timed lamp switches
    if environment is not None:
        _harness_plastic(environment)  # before the task's hooks, so they see this step's permanent bends
        _tie_plastic(environment)
    for fn in _STEP:
        fn(environment)
    for name in list(_PINNED):  # holds kept by re-setting the joints (see hold_joint)
        if environment is not None and any(key[0] == name for key in _HELD):
            _pin(environment, name)
    if _LIT["on"]:
        return
    for spec in _SPECS.values():
        if isinstance(spec, Button) and _state(environment, spec).travel >= spec.latch_at:
            _show_lamps(environment, True)
            return


def _set_joint_starts(environment) -> None:
    """Put every hinge and slider at its ``start``. Isaac Lab-Arena's reset leaves their joints at 0 (the spring's
    rest, clamped to the limits): a sharps lid meant to start open started shut, a gas-strut lid meant to start shut
    started open (GPU check, 9 Nov 2026)."""
    base = getattr(environment, "unwrapped", environment)
    for name, spec in _SPECS.items():
        if isinstance(spec, CableTie) and not spec.fixed:
            _reset_tie_root(base, base.scene[name])
        if isinstance(spec, Tool):  # a loose tool back where it starts, at rest
            write_body_pose(environment, name, spec.pos, spec.quat)
        if isinstance(spec, Hinge | Slider | HangingBulb | Harness | CableTie | KeyedPulley | Webbing | Ratchet):
            asset = base.scene[name]
            asset.write_joint_position_to_sim_index(position=asset.data.default_joint_pos.torch.clone())
            asset.write_joint_velocity_to_sim_index(velocity=asset.data.default_joint_vel.torch.clone())
        if isinstance(spec, CableTie) and name in _PLASTIC:  # the strap comes back with no set from the last trial
            import torch

            _PLASTIC.pop(name)
            _call_first(asset, ("set_joint_position_target_index", "set_joint_position_target"),
                        torch.tensor([_tie_goals(spec, name, asset.joint_names)], dtype=torch.float32,
                                     device=asset.device))
        if isinstance(spec, Harness):  # the wires come back straight: no permanent bends from the last trial
            import torch

            n = len(asset.joint_names)
            _PLASTIC[name] = np.zeros(n)  # set outright: joint_pos may still read the last trial until a step runs
            _call_first(asset, ("set_joint_position_target_index", "set_joint_position_target"),
                        torch.zeros((1, n), dtype=torch.float32, device=asset.device))
            _WROTE.add(name)


_SHARP = {"done": False}


def _sharp_render() -> None:
    """KIT_SHARP=1: render the cameras at their full size, once Kit is running. Isaac's default anti-aliasing is DLSS,
    which renders a fraction of the frame and upscales it (soft video); DLAA is the same network at native size.
    Proven on the GPU VM for the drain task, 15 Nov 2026 (aa/op read 3, set to 4). Logged before and after."""
    import os

    if _SHARP["done"] or not os.environ.get("KIT_SHARP"):
        return
    _SHARP["done"] = True
    try:
        import carb
    except ImportError:
        print("[labkit] sharp render: carb not importable; left as is", flush=True)
        return
    s = carb.settings.get_settings()
    keys = ("/rtx/post/aa/op", "/rtx/post/dlss/execMode")
    before = {k: s.get(k) for k in keys}
    s.set_int("/rtx/post/aa/op", 4)  # 4 = DLAA
    s.set_int("/rtx/post/dlss/execMode", 2)  # Quality, should anything still fall back to DLSS
    print(f"[labkit] sharp render (DLAA): before {before}, after { {k: s.get(k) for k in keys} }", flush=True)


_WRITERS: dict = {}  # which pose/velocity writer this Isaac Lab has, found once


def write_body_pose(environment, name: str, pos, quat_xyzw, lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)) -> str:
    """Put loose body ``name`` at ``pos`` (task frame) turned ``quat_xyzw``, moving at ``lin_vel``/``ang_vel``:
    task code standing in for a fit the simulator can't make (a key pressed into its seat, a bit in a screw's socket).
    Tries Isaac Lab 3's writers, then older names; returns the ones used. Offline (``environment`` None) does nothing."""
    if environment is None:
        return "offline"
    import torch

    base = getattr(environment, "unwrapped", environment)
    asset = base.scene[name]
    origin = _np(base.scene.env_origins)[0]
    dev = getattr(asset, "device", None) or base.device
    pose = torch.tensor([[*(np.asarray(pos, float) + origin), *np.asarray(quat_xyzw, float)]], dtype=torch.float32,
                        device=dev)
    vel = torch.tensor([[*np.asarray(lin_vel, float), *np.asarray(ang_vel, float)]], dtype=torch.float32, device=dev)
    used = []
    for kind, names, value, kws in (
            ("pose", ("write_root_link_pose_to_sim_index", "write_root_pose_to_sim_index",
                      "write_root_link_pose_to_sim", "write_root_pose_to_sim"), pose, ("root_pose", "pose", None)),
            ("vel", ("write_root_com_velocity_to_sim_index", "write_root_velocity_to_sim_index",
                     "write_root_com_velocity_to_sim", "write_root_velocity_to_sim"), vel,
             ("root_velocity", "velocity", None))):
        known = _WRITERS.get(kind)
        order = [known] if known else [(n, kw) for n in names for kw in kws]
        for n, kw in order:
            fn = getattr(asset, n, None)
            if fn is None:
                continue
            try:
                fn(value) if kw is None else fn(**{kw: value})
            except TypeError:
                continue
            _WRITERS[kind] = (n, kw)
            used.append(n)
            break
        else:
            raise RuntimeError(f"no {kind} writer worked on {name}: check the Isaac Lab rigid object API")
    return "+".join(used)


def _after_reset(environment) -> None:
    _sharp_render()
    for name, joint in list(_HELD) + list(_DRIVEN):  # a hold or drive from the last trial: springs back first
        release_joint(environment, name, joint)
    _set_joint_starts(environment)
    _show_lamps(environment, False)
    for fn in _RESET:  # labkit's _start_trial first, then the task's hooks (which may schedule lamps, gloves)
        fn()
    _show_lamps(environment, _LIT["on"])  # a timed switch due at 0 lights from the first frame
    _show_gloves(environment)
    for fn in _RESET_ENV:
        fn(environment)


# -- hooking yam_arena ------------------------------------------------------------------------------------------------


def _patch_scene(scene) -> None:
    if getattr(scene, "_labkit", False):
        return
    task_object, object_states = scene.task_object, scene.object_states

    def task_object_hook(spec):
        if isinstance(spec, KitObject):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return task_object(spec)

    def object_states_hook(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = object_states(environment, plain) if plain else {}
        out.update({n: _state(environment, _SPECS[n]) for n in names if n in _SPECS})
        return {n: out[n] for n in names}

    scene.task_object, scene.object_states = task_object_hook, object_states_hook
    scene._labkit = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_labkit", False):
        return
    task_step, reset = cls._task_step, cls.reset

    def task_step_hook(self, *args, **kwargs):
        _after_step(self._env, getattr(self, "_t", None))
        return task_step(self, *args, **kwargs)

    def reset_hook(self, *args, **kwargs):
        env = getattr(self, "_env", None)
        if env is not None:  # set the trial's starting look before reset renders the first frame
            for fn in _BEFORE_RESET_ENV:
                try:
                    fn(env)
                except Exception as err:  # noqa: BLE001 - a look must never stop a reset
                    print(f"labkit: before-reset hook failed: {err!r}", flush=True)
        obs = reset(self, *args, **kwargs)
        _SEED["seed"] = kwargs.get("seed")
        _after_reset(self._env)
        return obs

    cls._task_step, cls.reset = task_step_hook, reset_hook
    cls._labkit = True


class _Patcher(importlib.abc.MetaPathFinder):
    """Patches yam_arena's scene and embodiment modules as they load (they need Isaac Sim, which starts after the
    task file is read)."""

    patches = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.patches:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        exec_module, patch = spec.loader.exec_module, self.patches[fullname]

        def exec_and_patch(module):
            exec_module(module)
            patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    """Patch modules already imported, and the rest as they load. Safe to call twice."""
    for name, patch in _Patcher.patches.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Patcher) for f in sys.meta_path):
        sys.meta_path.insert(0, _Patcher())


install()
