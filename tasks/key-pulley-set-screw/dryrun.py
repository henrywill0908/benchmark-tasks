"""A kinematic stand-in for the simulator (no physics, no GPU): plays a demo plan's waypoints at 20 Hz and moves the key,
the pulley and the driver the way the hands would, through pulley_logic.Tracker exactly as task.py's hook does, so
the scorer's verdict on each demo variant can be checked offline. Crude on purpose: a closed hand near a part carries
it, a pushing finger block pushes the rim's face, the pad sags with what presses on it. It checks the task logic and
the plan's geometry, not the physics (the GPU run does that).

    python dryrun.py [variant]
"""

from __future__ import annotations

import math
import sys

import numpy as np

import pulley_logic as L

DT = 1.0 / L.CONTROL_HZ


def _timeline(steps, arm_start):
    """Per control step: (TCP, rotation, finger gap m, label) by linear interpolation between waypoints."""
    out = []
    p0, r0, g0 = arm_start
    for seconds, pos, fingers, rot, label in steps:
        if pos is None or isinstance(rot, str):
            p1, r1 = arm_start[0], arm_start[1]
        else:
            p1, r1 = np.asarray(pos, float), np.asarray(rot, float)
        g1 = L.FINGER_GAP[fingers] * L.MM
        n = max(1, int(round(seconds * L.CONTROL_HZ)))
        for i in range(1, n + 1):
            f = i / n
            out.append((p0 + f * (p1 - p0), r1 if f > 0.5 else r0, g0 + f * (g1 - g0), label))
        p0, r0, g0 = p1, r1, g1
    return out


def _yaw(R) -> float:
    """Yaw of a straight-down grip (deg)."""
    return math.degrees(math.atan2(R[0, 0], -R[1, 0]))


def run(variant: str = "full", verbose: bool = False) -> dict:
    plan = L.demo_plan(variant)
    home = {"left": (np.array([0.20, 0.25, L.TOP + 0.25]), L.rot_down(0.0), 0.094),
            "right": (np.array([0.20, -0.25, L.TOP + 0.25]), L.rot_down(0.0), 0.094)}
    lines = {arm: _timeline(plan[arm], home[arm]) for arm in plan}
    n = max(len(v) for v in lines.values()) + 3 * L.CONTROL_HZ
    t = L.Tracker()
    key_pos, key_quat = np.array(L.KEY_START), np.array([0.0, 0.0, 0.0, 1.0])
    drv_pos, drv_quat = np.array(L.BAR_START), np.array([0.0, 0.0, 0.0, 1.0])
    slide, turn = 0.0, math.radians(L.PHI0)
    pad, pad_held = 0.0, False
    slide_hold, turn_hold = None, None
    key_in_hand, drv_in_hand, rim_in_hand = None, None, None
    key_welded = False
    drv_pinned = False
    history = []
    for k in range(n):
        st = {arm: (lines[arm][min(k, len(lines[arm]) - 1)]) for arm in lines}
        # -- the hands move things --------------------------------------------------------------------------------
        for arm, (p, R, gap, label) in st.items():
            tip = p + R @ np.array([0.0, 0.0, L.PAD_TIP])
            closed = gap < 0.02
            # the key: picked by a closed hand round it, carried, set down when the hand opens
            if key_in_hand is None and closed and not key_welded and np.linalg.norm(p - key_pos) < 0.03:
                key_in_hand = (arm, key_pos - p)
            if key_in_hand and key_in_hand[0] == arm:
                if not closed:
                    key_in_hand = None
                else:
                    key_pos = p + key_in_hand[1]
            # pressing the key's top with closed fingertips (the hand over the keyseat, the key not held)
            if key_in_hand is None and not key_welded and abs(p[0] - L.SEAT_CX_SIM) < 0.01 and abs(p[1]) < 0.005 \
                    and closed and tip[2] < key_pos[2] + L.KEY[2] / 2:
                sink = key_pos[2] + L.KEY[2] / 2 - tip[2]
                if pad_held:
                    pad = -min(L.PAD_GIVE * 1.2, sink)  # the held pad gives up to its breakaway
                else:
                    pad = -min(L.PAD_RISE, sink + (-pad))
                key_pos = np.array([L.SEAT_CX_SIM, 0.0, L.AX_Z + L.SEAT_FLOOR_R + L.PAD_RISE + pad + L.KEY[2] / 2])
            # the rim: a closed hand on it (fingers across its width) turns it with the hand's angle about x
            on_rim = np.linalg.norm(p - np.array(L.rim_grip(math.degrees(rim_in_hand[1]) if rim_in_hand else 0.0,
                                                               slide)[0])) < 0.5
            if rim_in_hand is None and closed and arm == "left" and "rim" in label and "let go" not in label:
                u = p - np.array([p[0], L.AX_Y, L.AX_Z])
                rim_in_hand = (arm, math.atan2(-u[1], u[2]), turn)
            if rim_in_hand and rim_in_hand[0] == arm:
                if not closed:
                    rim_in_hand = None
                else:
                    u = p - np.array([p[0], L.AX_Y, L.AX_Z])
                    ang = math.atan2(-u[1], u[2])
                    if turn_hold is None:
                        turn = rim_in_hand[2] + (ang - rim_in_hand[1])
            del on_rim
            # the push: a closed finger block in front of the rim's face pushes it along x
            if closed and "push" in label or (closed and label == "hold" and arm == "left" and rim_in_hand is None
                                              and abs(p[2] - (L.AX_Z + 0.012 + L.PAD_TIP)) < 0.002):
                face = p[0] + L.PAD_HALF
                want = face - (L.REAR0 - L.HUB_L)
                if want > slide:
                    slide = min(want, L.SLIDE_LIMITS[1])
            # the driver: picked by its handle, carried; pinned by task code once engaged
            bar_grip = drv_pos + np.array([0.0, 0.0, L.BAR_GRIP_ABOVE])
            if drv_in_hand is None and closed and arm == "left" and np.linalg.norm(p - bar_grip) < 0.012:
                drv_in_hand = (arm, _yaw(R) - math.degrees(math.atan2(L.quat_xyzw_to_matrix(drv_quat)[1, 0],
                                                                      L.quat_xyzw_to_matrix(drv_quat)[0, 0])))
            if drv_in_hand and drv_in_hand[0] == arm:
                if not closed:
                    drv_in_hand = None
                    if not drv_pinned and drv_pos[2] < L.TOP + L.CUP_H + 0.02 and \
                            np.linalg.norm(drv_pos[:2] - np.array(L.HOLDER)) < 0.01:
                        drv_pos = np.array(L.BAR_START)  # it drops back onto the stand's rim
                else:
                    yaw = math.radians(_yaw(R) - drv_in_hand[1])
                    drv_quat = L.matrix_to_quat_xyzw(L.rot_z(yaw))
                    if not drv_pinned:
                        drv_pos = p - np.array([0.0, 0.0, L.BAR_GRIP_ABOVE])
                    else:
                        drv_quat = L.matrix_to_quat_xyzw(L.rot_z(yaw))
        # holds
        if slide_hold is not None:
            slide = min(slide, slide_hold + 1.2e-3 if "push" in st["left"][3] else slide_hold)
        if turn_hold is not None:
            turn = turn_hold
        grippers, tcps = [], []
        for arm, (p, R, gap, label) in st.items():
            tips = [p + R @ np.array([s * (gap / 2 + 3e-3), 0.0, L.PAD_TIP]) for s in (-1, 1)]
            grippers += [p, *tips]
            tcps.append((p, gap))
        x = L.Inputs(slide=slide, turn=turn, key_pos=key_pos, key_quat=key_quat, driver_pos=drv_pos,
                     driver_quat=drv_quat, driver_ang_vel=np.zeros(3), pad=pad, grippers=grippers, tcps=tcps)
        a = t.step(x)
        if a.hold_pad is not None:
            pad_held = a.hold_pad
        if a.slide_hold is not L.KEEP:
            slide_hold = a.slide_hold
            if slide_hold is not None:
                slide = min(slide, slide_hold)
        if a.turn_hold is not L.KEEP:
            turn_hold = a.turn_hold
            if turn_hold is not None:
                turn = turn_hold
        if a.key_pose is not None:
            key_pos, key_quat = np.asarray(a.key_pose[0]), np.asarray(a.key_pose[1])
            key_welded, key_in_hand = True, None
        drv_pinned = a.driver_pose is not None
        if drv_pinned:
            drv_pos, drv_quat = np.asarray(a.driver_pose[0]), np.asarray(a.driver_pose[1])
        elif drv_in_hand is None and drv_pos[2] > L.TOP + L.CUP_H + 0.03:
            pass  # a driver let go in the air would fall; the plan never does that
        history.append(dict(t.log, label=st["left"][3] + " | " + st["right"][3]))
        if verbose and k % 10 == 0:
            print(history[-1])
    return {"score": L.task_score(t.facts), "facts": t.facts, "log": history[-1], "history": history}


EXPECT = {"full": 1.0, "no_key": 0.0, "proud": 0.0, "no_turn": 0.25, "leave_driver": 0.75, "turn_err3": 1.0,
          "turn_err6": 0.25}

if __name__ == "__main__":
    for v in (sys.argv[1:] or L.VARIANTS):
        r = run(v, verbose=False)
        print(f"{v}: {r['score']} (expect {EXPECT[v]})  {r['log']}")
