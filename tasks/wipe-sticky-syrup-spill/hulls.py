"""Exact clearance between the YAM arms' collision hulls (the CoACD hulls yam_arena's URDF lists) and simple solids,
by GJK on point sets: copied from the ratchet-strap kit's armcheck (which checked its demo against the GPU run)."""

from __future__ import annotations

import itertools
import math

import numpy as np
from yam_arena import YAM_URDF
from yam_arena.analysis import link_hulls
from yam_arena.constants import ARM_JOINTS
from yam_arena.station import BASES
from yam_arena.urdf import forward_kinematics

ARM_LINKS = ("link1", "link2", "link3", "link4", "link5", "gripper", "tip_left", "tip_right")


def _closest_on(points):
    k = len(points)
    if k == 1:
        return points[0]
    a = (points[1:] - points[0]).T
    try:
        lam = np.linalg.solve(a.T @ a, -a.T @ points[0])
    except np.linalg.LinAlgError:
        return None
    if lam.min() < -1e-12 or lam.sum() > 1 + 1e-12:
        return None
    return points[0] + a @ lam


def _closest_simplex(s):
    if len(s) == 4:
        p = np.array(s)
        try:
            lam = np.linalg.solve((p[1:] - p[0]).T, -p[0])
            if lam.min() >= -1e-12 and lam.sum() <= 1 + 1e-12:
                return np.zeros(3), list(range(4))
        except np.linalg.LinAlgError:
            pass
    best = None
    for r in range(1, len(s) + 1):
        for idx in itertools.combinations(range(len(s)), r):
            c = _closest_on(np.array([s[i] for i in idx]))
            if c is not None and (best is None or c @ c < best[0] - 1e-18):
                best = (c @ c, c, list(idx))
    return best[1], best[2]


def gjk(a, b, tol: float = 1e-7, iters: int = 60) -> float:
    """The distance between the convex hulls of point sets ``a`` and ``b`` (0 if they overlap)."""
    v = a[0] - b[0]
    s = [v]
    for _ in range(iters):
        if v @ v < 1e-14:
            return 0.0
        w = a[np.argmax(a @ -v)] - b[np.argmax(b @ v)]
        if v @ v - v @ w <= tol * max(v @ v, 1e-12):
            break
        s.append(w)
        v, keep = _closest_simplex(s)
        s = [s[i] for i in keep]
    return float(math.sqrt(v @ v))


def part(name: str, verts) -> tuple:
    v = np.asarray(verts, float)
    return (name, v, v.min(0), v.max(0))


def box(name: str, centre, rot, size) -> tuple:
    c = np.array(list(itertools.product((-1, 1), repeat=3)), float) * (np.asarray(size, float) / 2)
    return part(name, c @ np.asarray(rot, float).T + np.asarray(centre, float))


def _cyl(name: str, centre, rot, radius: float, length: float, n: int = 12) -> tuple:
    pts = [(radius * math.cos(2 * math.pi * k / n), s, radius * math.sin(2 * math.pi * k / n))
           for s in (-length / 2, length / 2) for k in range(n)]
    return part(name, np.asarray(pts) @ np.asarray(rot, float).T + np.asarray(centre, float))


def arm_parts(arm: str, q, finger: float, links=ARM_LINKS) -> list:
    """Each link's hulls posed at joints ``q`` (6) and finger joint ``finger``, task frame."""
    qd = {n: float(v) for n, v in zip(ARM_JOINTS, q)}
    qd["joint7"] = qd["joint8"] = float(finger)
    fk = forward_kinematics(YAM_URDF, qd)
    base = np.asarray(BASES[arm], float)
    hulls = link_hulls()
    out = []
    for link in links:
        t = fk[link]
        for h in hulls.get(link, []):
            out.append(part(f"{arm}.{link}", h.vertices @ t[:3, :3].T + t[:3, 3] + base))
    return out


def min_dist(a: list, b: list, cap: float = 0.05) -> tuple:
    """(distance, part of a, part of b) over two part lists; pairs whose boxes are ``cap`` or more apart are skipped
    (the distance is then reported as ``cap``)."""
    best = (cap, None, None)
    if not a or not b:
        return best
    la, ha = np.array([p[2] for p in a]), np.array([p[3] for p in a])
    lb, hb = np.array([p[2] for p in b]), np.array([p[3] for p in b])
    gap = np.linalg.norm(np.maximum(0.0, np.maximum(lb[None] - ha[:, None], la[:, None] - hb[None])), axis=2)
    ii, jj = np.nonzero(gap < cap)
    for k in np.argsort(gap[ii, jj]):
        i, j = ii[k], jj[k]
        if gap[i, j] >= best[0]:
            break  # the boxes' gap bounds the hulls' distance from below
        d = gjk(a[i][1], b[j][1])
        if d < best[0]:
            best = (d, a[i][0], b[j][0])
            if d == 0.0:
                break
    return best
