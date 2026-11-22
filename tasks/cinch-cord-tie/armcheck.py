"""Arm-versus-room clearance along a plan's real path (offline, NumPy and yam_arena only).

Round 1 (17 Nov 2026) lost 15 of 17 jig trials because the left arm's tilted approach drove its wrist into the TV:
every waypoint passed yam_arena's IK, but the path between them went through the TV. This replays a plan exactly as
the probe plays it (``ArmPlan.action7`` every control step, IK warm-started), puts yam.urdf's collision meshes
(subsampled, with each link's visual origin applied) at every sampled pose, and measures them against the room: the
TV and wall boxes and the cords' cylinders.

    from armcheck import plan_clearance
    worst = plan_clearance(task_module, plan)   # {arm: (metres, t, label, link, what)}

Round 21: ``arms_clearance`` measures the two arms against each other the same way. Job 95's grip change had the left's
down grip 82 mm from the right's: their forearms (link4) came within 3 mm, the left pushed the right 57 mm and the head
fell, though every finger envelope was 10 mm or more apart.
"""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET  # noqa: S405 - the package's own URDF, read as data
from functools import lru_cache
from pathlib import Path

import numpy as np

ARM_LINKS = ("link3", "link4", "link5", "gripper", "tip_left", "tip_right")
NEAR_CORDS_OK = ("tip_left", "tip_right")  # fingertips work next to the cords: tie_logic.finger_boxes checks those


def _stl(path: Path) -> np.ndarray:
    b = path.read_bytes()
    if b[:5] == b"solid" and b"facet" in b[:400]:
        return np.array([list(map(float, ln.split()[1:4])) for ln in b.decode().splitlines()
                         if ln.strip().startswith("vertex")])
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return a["v"].reshape(-1, 3).astype(float)


@lru_cache(maxsize=1)
def _meshes() -> dict:
    from yam_arena import ik
    from yam_arena.urdf import rpy_matrix

    asset = Path(ik.__file__).parent / "assets" / "yam"
    origin = {}
    for link in ET.parse(asset / "yam.urdf").getroot().findall("link"):  # noqa: S314
        v = link.find("visual")
        if v is not None:
            o = v.find("origin")
            origin[link.get("name")] = (np.array([float(x) for x in o.get("xyz").split()]),
                                        rpy_matrix(np.array([float(x) for x in o.get("rpy", "0 0 0").split()])))
    out = {}
    for name in ARM_LINKS:
        raw = np.vstack([_stl(f) for f in sorted((asset / "collision" / name).glob("*.stl"))])
        xyz, rot = origin[name]
        pts = (rot @ raw.T).T + xyz
        step = max(1, len(pts) // 400)  # an all-pairs distance over every vertex ran 5.6 GB once
        out[name] = np.unique(np.round(pts[::step], 4), axis=0)
    return out


def _box_dist(pts, lo, hi) -> float:
    q = np.maximum(np.maximum(np.asarray(lo) - pts, pts - np.asarray(hi)), 0.0)
    inside = np.all((pts > lo) & (pts < hi), axis=1)
    d = np.linalg.norm(q, axis=1)
    d[inside] = -np.min(np.minimum(pts[inside] - lo, hi - pts[inside]), axis=1) if inside.any() else d[inside]
    return float(d.min())


def room(task_module, build: str | None = None) -> list:
    """(name, kind, geometry) of what the arms must not touch in ``build`` (default: the task module's)."""
    L = task_module.L
    out = []
    for name, span in (("tv", L.TV), ("wall", L.WALL)):
        out.append((name, "box", (np.array([span["x"][0], span["y"][0], span["z"][0]]),
                                  np.array([span["x"][1], span["y"][1], span["z"][1]]))))
    if (build or task_module.BUILD) in ("lift", "lift25", "lift30"):  # round 19: the strip and the cords where
        #                                                                  they start (they move)
        span = L.LIFT_STRIP
        out.append(("strip", "box", (np.array([span["x"][0], span["y"][0], span["z"][0]]),
                                     np.array([span["x"][1], span["y"][1], span["z"][1]]))))
        for name, r, off, _, _ in L.LIFT_CORDS:
            out.append((f"{name}_cord", "poly", (L.lift_cord_points(off, r), r)))
            out.append((f"{name}_hang", "cyl", (np.array([L.LIFT_WALL_X, L.LIFT_HANG_Y - off]), r,
                                                (L.LIFT_ANCHOR_Z, L.TV["z"][0]))))
        return out
    for c in L.CORDS:
        out.append((c.name, "cyl", (np.array(c.xy), c.diameter / 2, L.CORD_Z)))
    out += L.stand_parts(L.BUILDS[build or task_module.BUILD]) if hasattr(L, "stand_parts") else []  # round 13: the clip stand
    return out


def _dist(pts, kind, geo) -> float:
    if kind == "box":
        return _box_dist(pts, *geo)
    if kind == "poly":  # a lying cord: its joint points and radius
        poly, r = geo
        a, b = poly[:-1], poly[1:]
        ab = b - a
        t = np.clip(np.einsum("pij,ij->pi", pts[:, None, :] - a[None], ab) / np.einsum("ij,ij->i", ab, ab), 0.0, 1.0)
        d = np.linalg.norm(pts[:, None, :] - (a[None] + t[..., None] * ab[None]), axis=2)
        return float(d.min() - r)
    xy, r, (z0, z1) = geo
    sel = pts[(pts[:, 2] > z0) & (pts[:, 2] < z1)]
    if not len(sel):
        return float("inf")
    return float(np.min(np.linalg.norm(sel[:, :2] - xy, axis=1)) - r)


def plan_clearance(task_module, plan: dict, every: int = 4, build: str | None = None) -> dict:
    """Replay ``plan`` (tie_logic steps) as the probe does and return, per arm, the closest approach of its links to
    the room: (distance m, t, label, link, what). Fingertips are not measured against the cords."""
    from yam_arena import ik
    from yam_arena.demo import ArmPlan
    from yam_arena.station import BASES
    from yam_arena.urdf import forward_kinematics

    urdf = Path(ik.__file__).parent / "assets" / "yam" / "yam.urdf"
    meshes = _meshes()
    things = room(task_module, build)
    L = task_module.L
    steps = int(L.plan_seconds(plan) * L.CONTROL_HZ)
    out = {}
    for arm, ws in task_module.plan_waypoints(plan).items():
        p = ArmPlan(ws, BASES[arm])
        base = np.asarray(BASES[arm], float)
        worst = (float("inf"), 0.0, "", "", "")
        for k in range(1, steps + 1):
            t = k / L.CONTROL_HZ
            a, label = p.action7(t)
            if k % every:
                continue
            q = {f"joint{i + 1}": float(a[i]) for i in range(6)}
            half_gap = 0.04695 * float(a[6])  # normalised grip 0..1 -> finger joint (m)
            q["joint7"] = q["joint8"] = -half_gap
            poses = forward_kinematics(urdf, q)
            for link in ARM_LINKS:
                pose = poses[link]
                pts = meshes[link] @ pose[:3, :3].T + pose[:3, 3] + base
                for name, kind, geo in things:
                    if link in NEAR_CORDS_OK and (kind in ("cyl", "poly") or name.startswith(("clip", "stand"))):
                        continue  # fingertips at the cords and the clip: tie_logic.finger_boxes checks those
                    d = _dist(pts, kind, geo)
                    if d < worst[0]:
                        worst = (d, t, label, link, name)
        out[arm] = worst
    return out


def _cloud_gap(a: dict, b: dict) -> tuple:
    """The closest approach (m, link of a, link of b) of two arms' link point clouds."""
    best = (float("inf"), "", "")
    for la, pa in a.items():
        for lb, pb in b.items():
            if np.any(np.maximum(pa.min(0), pb.min(0)) - np.minimum(pa.max(0), pb.max(0)) > best[0]):
                continue  # their bounding boxes are further apart than the best so far
            d = float(np.sqrt(((pa[:, None, :] - pb[None]) ** 2).sum(-1)).min())
            if d < best[0]:
                best = (d, la, lb)
    return best


def arms_clearance(task_module, plan: dict, every: int = 4, t0: float = 0.0, t1: float | None = None,
                   shift=None) -> tuple:
    """Replay both arms of ``plan`` together (as :func:`plan_clearance` does, ``shift(arm, waypoint)`` optionally moving
    waypoints as a closed loop might) and return their closest approach between ``t0`` and ``t1``: (metres, t, left's
    label, right's label, left's link, right's link). Meshes are subsampled (about 150 points a link), so read a few mm
    of margin into it."""
    import dataclasses

    from yam_arena import ik
    from yam_arena.demo import ArmPlan
    from yam_arena.station import BASES
    from yam_arena.urdf import forward_kinematics

    urdf = Path(ik.__file__).parent / "assets" / "yam" / "yam.urdf"
    sub = {k: v[:: max(1, len(v) // 150)] for k, v in _meshes().items()}
    L = task_module.L
    arms = {}
    for arm, ws in task_module.plan_waypoints(plan).items():
        if shift is not None:
            ws = [dataclasses.replace(w, pos=tuple(np.asarray(w.pos, float) + shift(arm, w))) for w in ws]
        arms[arm] = ArmPlan(ws, BASES[arm])
    end = L.plan_seconds(plan) if t1 is None else t1
    worst = (float("inf"), 0.0, "", "", "", "")
    for k in range(1, int(end * L.CONTROL_HZ) + 1):
        t = k / L.CONTROL_HZ
        got = {arm: p.action7(t) for arm, p in arms.items()}  # every step: the IK is warm-started
        if k % every or t < t0:
            continue
        clouds = {}
        for arm, (a, _) in got.items():
            q = {f"joint{i + 1}": float(a[i]) for i in range(6)}
            q["joint7"] = q["joint8"] = -0.04695 * float(a[6])
            poses = forward_kinematics(urdf, q)
            base = np.asarray(BASES[arm], float)
            clouds[arm] = {ln: sub[ln] @ poses[ln][:3, :3].T + poses[ln][:3, 3] + base for ln in ARM_LINKS}
        d, la, lb = _cloud_gap(clouds["left"], clouds["right"])
        if d < worst[0]:
            worst = (d, t, got["left"][1], got["right"][1], la, lb)
    return worst
