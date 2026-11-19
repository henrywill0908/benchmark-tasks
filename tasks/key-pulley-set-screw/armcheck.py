"""Replay a demo plan through yam_arena's own IK, waypoint by waypoint and at points between them, and report the
worst position and rotation errors per arm (no GPU). Used by test_task.py and runnable by hand:

    python armcheck.py [variant]
"""

from __future__ import annotations

import math
import sys

import numpy as np
from yam_arena import ik
from yam_arena.constants import READY
from yam_arena.station import BASES

import pulley_logic as L


def _target(arm: str, step):
    seconds, pos, fingers, rot, label = step
    if isinstance(rot, str) or pos is None:
        p, r = ik.tcp_pose(np.array(READY, dtype=float), np.array(BASES[arm]))
        return np.asarray(p, float), np.asarray(r, float)
    return np.asarray(pos, float), np.asarray(rot, float)


def _slerp_rot(r0, r1, f):
    """Rotation part way from r0 to r1 (axis-angle of the relative turn)."""
    rel = r0.T @ r1
    angle = math.acos(max(-1.0, min(1.0, (np.trace(rel) - 1) / 2)))
    if angle < 1e-9:
        return r0
    axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]) / (2 * math.sin(angle))
    a = f * angle
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return r0 @ (np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k)


def replay(arm: str, steps, between: int = 3) -> list:
    """[(label, position error m, rotation error rad, joints)] for every waypoint and ``between`` points before it."""
    q = np.array(READY, float)
    out = []
    prev = _target(arm, (0, None, "open", "ready", "start"))
    for s in steps:
        tgt = _target(arm, s)
        for i in range(1, between + 2):
            f = i / (between + 1)
            p = prev[0] + f * (tgt[0] - prev[0])
            r = _slerp_rot(prev[1], tgt[1], f)
            q, pe, re = ik.solve(p, ik.matrix_to_quat_xyzw(r), q, base_pos=BASES[arm])
            out.append((s[4] if i == between + 1 else f"{s[4]} (on the way)", float(pe), float(re), q.copy()))
        prev = tgt
    return out


def worst(variant: str = "full") -> dict:
    plan = L.demo_plan(variant)
    res = {}
    for arm, steps in plan.items():
        rows = replay(arm, steps)
        bad = max(rows, key=lambda r: r[1])
        res[arm] = dict(pos_mm=bad[1] * 1e3, rot_deg=math.degrees(max(r[2] for r in rows)), at=bad[0])
    return res


if __name__ == "__main__":
    v = sys.argv[1] if len(sys.argv) > 1 else "full"
    for arm, steps in L.demo_plan(v).items():
        for label, pe, re, q in replay(arm, steps):
            if pe > 1e-3 or re > math.radians(1):
                print(f"{arm}: {label}: {pe * 1e3:.1f} mm, {math.degrees(re):.1f} deg")
    print(worst(v))
