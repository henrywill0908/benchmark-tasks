"""Offline checks for a harder start than the submitted one: the tie flat on the console, the robot picking it up and
holding the head in one hand (2026-11-16; the submitted demo is task.py's demo(), on the pre-looped start). Not run. It records the demo's plan, the pieces already checked offline, and what round 1 on the GPU must settle first.
A scripted demo may read the simulator's state to plan (the review prompt allows it, and the accepted din-rail-mount
demo does): the insert here steers on the tip's measured pose, as probe trial j-steer does.

The phases, for the task's start (the tie flat on the console, the cords hanging under the TV):

1. PICK (right hand): a down grip across the head's 9 mm width, lift 6 cm. Reachable (probe trial f-pick).
2. REGRIP the head from behind, so both of the slot's faces are free: approach along the strap's root, fingers closing
   across the head's width. This is the hold the round-1 jig stands in for. At the tie's working pose beside the
   cords it is reachable (``hold_pose``, checked in test_demo.py). At the pick spot it is not (yam_arena IK misses by
   9-17 mm), so the regrip has to happen near the cords, with the left hand steadying the strap meanwhile (probe trial
   f-tug measures how much the two grips take). OPEN: carrying the straight 200 mm strap to the cords without it
   striking the wall 8 cm behind them (pointing forward, its tip would reach x 57 cm; the wall is at 46 cm). Candidate:
   carry it pointing left along the console's front, then turn it about the vertical as the left hand bends it round
   the cords.
3. WRAP (left hand): walk the strap round the back of the cords into the open loop the jig spawns (its tip 4 mm short
   of the slot). OPEN: no probe trial covers it yet; it needs flexible cords to be meaningful (round 2).
4. INSERT (left hand): the probe's two strokes (``tie_logic._insert_steps``), steered on the measured tip.
5. PULL (right hand would be holding the head; so the LEFT pulls, or the hands swap): the probe pulls with the right
   hand because the jig holds the head. OPEN: which hand pulls once the right holds the head. The tail exits toward the
   right arm's side; a left-hand end-on grip from that side is the first candidate to check for reach.
6. LET GO, both hands back to ready; the scorer needs 3 cm clearance and 1 s still.

What round 1 decides here: whether the strap feeds and draws through at all (captured, strokes), whether the cam
ratchet holds or the brake is needed (backslide), the capture tolerance (j-off-*, t-off-z15), whether lining up is
needed at all (j-naive: if it also captures, the task's hard part is somewhere else), and how much the holding grip
takes (f-tug).
"""

from __future__ import annotations

import numpy as np

import tie_logic as L

HOLD_DEPTH = 2.0 * L.MM  # how far past the head's back face the pad tips reach


def hold_pose(build: L.Build | None = None) -> tuple:
    """The right hand holding the head from behind at ``build``'s head pose: the TCP behind the head's back face, the
    gripper pointing along the strap's root (x_h), fingers closing across the head's width (y_h). Returns (TCP, R)."""
    b = build or L.BUILDS["jig"]
    x_h, y_h = b.head_rot[:, 0], b.head_rot[:, 1]
    back = np.asarray(b.head_pos) - x_h * L.HEAD[0] / 2
    tcp = back - x_h * (L.PAD_PAST_TCP - HOLD_DEPTH)
    return tcp, L.rot_axes(x_h, y_h)


def hold_clears_the_slot(build: L.Build | None = None) -> dict:
    """How far the holding fingers stay from the slot's entry and exit faces (m): the fingers grip the head's +-y_h
    faces, so the faces at +-z_h must stay open. Uses tie_logic's finger envelope."""
    b = build or L.BUILDS["jig"]
    tcp, rot = hold_pose(b)
    boxes = L.finger_boxes(tcp, rot, L.HEAD[1])  # the pads stop on the head's 9 mm width, not at the commanded gap
    out = {}
    for name, z in (("entry", L.ENTRY_Z), ("exit", L.EXIT_Z)):
        face = L.world(b, [(x, 0.0, z + np.sign(z) * 1.0 * L.MM) for x in np.linspace(L.PASSAGE_X[0], L.PASSAGE_X[1], 5)])
        out[name] = min(L.box_distance(face, box) for box in boxes)
    return out


def pick_steps() -> list:
    """Phase 1 for the right hand, as tie_logic steps (the probe's f-pick)."""
    pick = L.Trial("pick", "free", "", insert=None, pull=None, pick=True, head_fingers="head_light")
    return [s for s in L.trial_plan(pick)["right"] if s[4] not in ("watch", "wait")]


def insert_steps() -> list:
    """Phase 4 for the left hand, as tie_logic steps (the probe's c-heavier push; steering and depth are the
    caller's)."""
    return L._handover_plan(L.trial("c-heavier"), L.BUILDS["jig"])["left"]
