"""A kinematic stand-in for the simulator (no physics, no GPU): plays a demo variant at 20 Hz through
syrup_logic.Tracker exactly as task.py's hook does, so the scorer's verdict on each variant can be checked offline.
Crude on purpose: a hand closed on the bottle's grip carries it and the trigger moves as a spring against the finger
drive; a fingertip pressed on the towel drags it at ``follow`` of its own speed (the napkin task measured about half);
the towel is a rigid sheet. The right hand plays the open-loop spray plan; the left is syrup_logic.Wiper (closed loop
on the towel), as the GPU probe drives them. It checks the rules and the plan's geometry, not the physics.

    python dryrun.py [variant ...]
"""

from __future__ import annotations

import sys

import numpy as np

import syrup_logic as L

FINGER_K = 5000.0  # N/m per finger (about 25 N a side at 5 mm of error, measured on an earlier task)
READY = {"left": (np.array([0.20, 0.25, L.TOP + 0.25]), L.PRESS_ROT),
         "right": (np.array([0.20, -0.25, L.TOP + 0.25]), L.SIDE_GRIP)}
PATTERNS = ("strokes", "outside_in", "circles", "zigzag")


def _timeline(steps, arm):
    out = []
    p0, r0 = READY[arm]
    g0 = L.FINGER_GAP["open"] * L.MM
    for seconds, pos, fingers, rot, label in steps:
        if pos is None or isinstance(rot, str):
            p1, r1 = READY[arm]
        else:
            p1, r1 = np.asarray(pos, float), np.asarray(rot, float)
        g1 = L.FINGER_GAP[fingers] * L.MM
        n = max(1, int(round(seconds * L.CONTROL_HZ)))
        for i in range(1, n + 1):
            f = i / n
            out.append((p0 + f * (p1 - p0), r1 if f > 0.5 else r0, g0 + f * (g1 - g0), label))
        p0, r0, g0 = p1, r1, g1
    return out


def towel_nodes(centre) -> np.ndarray:
    nx, ny = int(round(L.TOWEL_SIZE[0] / L.TOWEL_EDGE)) + 1, int(round(L.TOWEL_SIZE[1] / L.TOWEL_EDGE)) + 1
    xs = np.linspace(-L.TOWEL_SIZE[0] / 2, L.TOWEL_SIZE[0] / 2, nx) + centre[0]
    ys = np.linspace(-L.TOWEL_SIZE[1] / 2, L.TOWEL_SIZE[1] / 2, ny) + centre[1]
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), np.full(X.size, L.TOP + L.TOWEL_SIZE[2] / 2)], 1)


def run(variant: str = "strokes", follow: float = 0.8, verbose: bool = False) -> dict:
    v = L.variant(variant)
    right = _timeline(v["right"], "right")
    t = L.Tracker()
    wiper = None if v["pattern"] is None else L.Wiper(L.scrub_path(v["pattern"]), READY["left"][0],
                                                       towel=v["towel"], remaining=t.dirty_centres,
                                                       touch_ups=L.TOUCH_UPS.get(variant, 2), poke=v.get("poke", False))
    wipe_from = int(round(v["wipe_after"] * L.CONTROL_HZ))
    b_pos, b_R = np.array(L.BOTTLE_START, float), np.eye(3)
    held = False
    towel = np.array(L.TOWEL_C, float)
    nodes = towel_nodes(towel)
    prev_tip = None
    trigger = 0.0
    left_p = READY["left"][0].copy()
    end = None
    k = 0
    for k in range(int(300 * L.CONTROL_HZ)):
        # right hand and the bottle
        p, R, g, label = right[k] if k < len(right) else (READY["right"][0], READY["right"][1],
                                                           L.FINGER_GAP["open"] * L.MM, "ready")
        grip_world = b_pos + b_R @ np.array([L.GRIP_X, 0.0, L.GRIP_Z])
        if not held and g < 0.068 and np.linalg.norm(p - grip_world) < 0.006:
            held = True
        if held:
            g_rest = L.FRONT_X - L.BACK_X
            trigger = float(np.clip(FINGER_K / 2 * (g_rest - g) / (L.TRIGGER_RATE + FINGER_K / 2), 0.0,
                                    L.TRIGGER_TRAVEL))
            b_R = L.bottle_rot_of_hand(R)
            b_pos = p - b_R @ np.array([(L.BACK_X + L.FRONT_X - trigger) / 2, 0.0, L.GRIP_Z])
            if g > 0.075:  # let go
                held, trigger = False, 0.0
                if abs(b_pos[2] - L.TOP) < 0.01:
                    b_pos[2] = L.TOP
        right_p, right_R = p, R
        # left hand and the towel
        if wiper is not None and k >= wipe_from and not wiper.done:
            left_p, _, _, llabel = wiper.step(nodes)
        tip = left_p + L.PAD_TIP_PAST_TCP * L.PRESS_ROT[:, 2]
        on_towel = abs(tip[0] - towel[0]) < L.TOWEL_SIZE[0] / 2 and abs(tip[1] - towel[1]) < L.TOWEL_SIZE[1] / 2
        floor = L.TOP + (L.TOWEL_SIZE[2] if on_towel else 0.0)
        tip_c = np.array([tip[0], tip[1], max(tip[2], floor)])
        pressing = on_towel and tip_c[2] <= L.TOP + L.TOWEL_SIZE[2] + 1e-3
        if prev_tip is not None and pressing and prev_tip[2] <= L.TOP + L.TOWEL_SIZE[2] + 1e-3:
            towel = towel + follow * (tip_c[:2] - prev_tip[:2])
            nodes = towel_nodes(towel)
        prev_tip = tip_c
        right_tip = right_p + L.PAD_TIP_PAST_TCP * right_R[:, 2]
        t.step(L.Inputs(bottle_pos=b_pos, bottle_quat=L.matrix_to_quat_xyzw(b_R), trigger=trigger, nodes=nodes,
                        tips=[tip_c, right_tip], grippers=[left_p, tip_c, right_p, right_tip]))
        if verbose and k % 40 == 0:
            print(k, label, "|", wiper.mode if wiper else "-", t.log)
        done = k >= len(right) and (wiper is None or wiper.done)
        if done and end is None:
            end = k
        if end is not None and k >= end + 2 * L.CONTROL_HZ:
            break
    return {"variant": variant, "follow": follow, "score": L.task_score(t.facts), "steps": k + 1,
            "seconds": (k + 1) / L.CONTROL_HZ, "sprays": t.sprays, "wet_hits": t.wet_hits,
            "recentres": None if wiper is None else wiper.recentres, "log": dict(t.log)}


if __name__ == "__main__":
    for name in sys.argv[1:] or list(L.VARIANTS):
        for fol in (0.5, 0.8):
            r = run(name, fol)
            lg = r["log"]
            print(f"{name:16s} follow {fol}: score {r['score']:.2f} (expect {L.EXPECTED[name]}), {r['seconds']:.0f} s, "
                  f"sprays {r['sprays']}, recentres {r['recentres']}, puddle clean {lg['puddle_clean']}/{L.N_PUDDLE}, "
                  f"dirty {lg['dirty']}, smeared {lg['smeared']}, fouled {lg['fouled']}, rubbed {lg['rubbed_mm']} mm")
