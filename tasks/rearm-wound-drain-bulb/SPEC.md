# TASK_SPEC: rearm-wound-drain-bulb

Status: first GPU run 15 Nov 2026 (Brev L40S): 38 tests pass on the real stack; demo 1.0 in 294 steps; cap-first
0.25; no-squeeze 0.25. Since then (offline only): the cap is push-fit (sprung open by its tether, seats only when
pressed, then held), the manikin is rebuilt as a recognisable body, and the bulb hangs free (it swings; pulled
more than 3 cm, the drain is tugged: 0). Second GPU run 15 Nov 2026 (RTX 4090 VM), all three in: 44 tests pass; demo 1.0 in 262 steps (official run 1.0);
cap-first 0.25; no-squeeze 0.25. Then the blanket's legs and feet were redrawn (visual only). Dream-1 refused the first wording ("...connected directly to a patient..."), so the instruction and setup now name the training manikin (see README). Third GPU run with that wording: 44 tests, demo 1.0 in 262 steps (official run 1.0), cap-first 0.25, no-squeeze 0.25, Dream-1 0.0 over 1,200 steps (not refused). Nothing submitted.
Coordinates in cm from the midpoint between the two arm bases: x forward, y to the robot's left, heights above the
table top. Arm bases at y +30.5 (left) and -30.5 (right).

**Title:** Put a wound-drain bulb back on suction: squeeze it flat and cap it while it is still flat

**Category:** Impact on jobs

**What it tests (1-2 lines):** a two-handed, simultaneous action where the order is the trap: one gripper holds a
springy bulb flat while the other seats its cap, then both let go. A model fails by capping the round bulb (looks
done, no suction), by letting go before the cap is home, or by never capping it.

**Why it matters (1-2 lines):** closed-suction wound drains (Jackson-Pratt type) are emptied and re-armed at least
twice a day, and whenever full, on surgical wards by registered nurses (O*NET 29-1141.00) and licensed practical
nurses (29-2061.00). A bulb capped while expanded gives no suction: fluid collects under the wound, raising the risk
of seroma and infection.

Domain evidence (checked 14 Nov 2026):
- O*NET 29-2061.00 (LPNs), task list: "Assemble and use equipment, such as catheters, tracheotomy tubes, or oxygen
  suppliers" and "Provide basic patient care or treatments, such as ... dressing wounds ...". Drains are not named,
  so the context ties re-arming a drain to these lines rather than claiming a listed task.
- Nursing-skills texts teach it as a standard skill: Open RN *Nursing Skills* 20.13 "Checklist for Drain Management"
  (https://wtcs.pressbooks.pub/nursingskills/chapter/20-13-checklist-for-drain-management/) and *Clinical Procedures
  for Safer Patient Care* 4.7 "Drain Management and Removal" (https://opentextbc.ca/clinicalskills/chapter/4-8-drain-management-and-removal/).
  Re-read both before submitting and quote the step that compresses the bulb before re-capping.
- Headcount: RNs about 3.3 million and LPNs about 0.6 million US jobs (BLS). Confirm the current BLS figures.
- If a reviewer questions the domain, the review prompt makes that "revise", not a final rejection.

**Instruction given to the model (exact text):**

> The training manikin's drain bulb has just been emptied. Put it back on suction without tugging on the drain tube.

The technique is left out: it is the specialist knowledge the physical skill depends on. Fallback if a reviewer
objects: add "squeeze the bulb flat and cap it while it is still flat"; the task stays physical.

**Labels:** one event. Trigger `start`; prompt `good`; scene `good`
(round vs flat is 48 vs 16 mm thick, 9 vs 3 px in the top camera); knowledge `specialist`; response `continue`.

**Closest accepted tasks** (our notes only: the context names none, because the reviewer finds neighbours on its own
and accepted tasks rarely name them): brake bleeding (#1577: the same seal-timing trap, shut the bleeder before the
pedal comes back up), the fingertip oximeter (#1837: steady a hand while working a spring clip), the M.2 drive and
the newborn length board (hold a sprung part flat while the other hand works). Ours differs in the job and in how
it fails: a bulb capped round looks finished.

**Objects** (all fixed or jointed; nothing loose):

| Object | Kind (labkit / yam_arena) | Size | Look | Notes |
|---|---|---|---|---|
| Bulb faces, left and right | 2 x `Slider` (invisible paddles) | 6 x 0.3 x 8 cm each | not drawn | sprung slides, 1.6 cm travel, 10 N at full; pass through each other |
| Bulb body | `Prop` (generated ellipsoid) | 6.4 x 4.8 x 10 cm oval | translucent white | redrawn every step to the faces' gap: the visible squeeze |
| Port and cap | `Hinge` (box base, generated round lid) | port 1.2 cm square, 1 cm tall; cap 1.6 cm round, 6 mm | white | push-fit cap on a tether hinged at the port's edge, sprung open (0.002 N m/rad toward 114 degrees, 17x its weight at flat); seats when pressed within 2 degrees of flat, then held by task code |
| Side rail and 2 posts | `Box` | rail 2.5 x 62 x 3 cm, 13 to 16 cm up | grey | the bulb's strap hangs from it |
| Mattress sheet | `Box` | 60 x 120 cm | white | beyond the rail |
| Training manikin | `Prop` (generated ellipsoids and tubes) | head with hair and nose on a pillow, shoulders, chest, hips, both arms with hands, a blanket over the legs with the feet tenting it, dressing | blue gown, skin, cream blanket | visual only, out of the arms' way |
| Drain tubing and strap | `Prop` (generated tubes) | 6 mm and 8 mm | clear, white | visual only |
| Measuring cup with fluid | `Container` + `Prop` | 7 x 7 x 8 cm, 2.5 cm of fluid | clear, pink | scenery: shows the drain was just emptied |

**Starting layout:** bulb centre at (22, -12), 7.5 cm up, turned 20 degrees so the left arm's forward grip meets its
faces square; cap open; rail along y at x 33; manikin at x 62; cup at (12, 36).

**The default move must be wrong.** On habit a model closes the open cap: the bulb stays round and scores 0.25.

**Mechanism (task code, stated in the fidelity note):** when the cap seats, both faces are held where they are, both
ways (labkit `hold_slider`, a stiff drive): a sealed bulb can neither spring open (trapped vacuum) nor be squeezed
further (trapped air). Lifting the cap releases them. So capping first and squeezing afterwards can't succeed.

**Scoring (numeric):**
- Squeeze: 0 (48 mm thick) to 1 (16 mm). Seated: cap within 2 degrees of flat; unseated past 7 degrees. Sealed
  squeeze: the squeeze when the cap last seated. At the bulb: a gripper's tool point or pad tips within 6 cm of the
  bulb's vertical centre line, from 2 cm below it to 4 cm above the port.
- Gate (0): the bulb's centre pulled more than 3 cm across from where it hangs, at any time (the drain tugged;
  the safety rule, like the accepted oximeter task's "without dragging the hand").
- Gate (0): a pad tip within 6 mm of the port mouth's centre and 4 mm of its height while the cap is open over 20
  degrees (contamination).
- 0 if the cap isn't seated at the end (doing nothing; squeezing without capping). 0.25 sealed under 0.7. 0.5 sealed
  at 0.7 or more but a gripper still at the bulb, or the squeeze fell under 0.6. 1.0 sealed at 0.7 or more, still at
  0.6 or more, both grippers away; success after 1 s.

**Run limit:** 1,200 steps (60 s at 20 Hz). The demo needs 16 s plus the 1 s hold.

**Demo:** left arm, forward grip turned 20 degrees: in front of the bulb, fingers either side, squeeze, hold. Right
arm, straight down, closed fingers: wait for the squeeze, sweep the open cap over at 9 mm above the port and stop on
top of it (its tether pushes it up against the fingers), press it home 2.5 cm past (the arm gives about 8 mm), lift
off, move aside. Then the left arm lets go and
backs out. Reach: every 20 Hz step solved on the URDF within 0.01 mm; joint-limit margin at least 0.038 rad (left
arm backing out), 0.123 rad (right arm).

**Cheats and unintended strategies:**
- Cap first, then squeeze: the seal holds the bulb round: 0.25.
- Squeeze, let go, then cap: round again by then: 0.25.
- Hold the cap half shut so it never latches: not seated: 0.
- Poke the open port: 0 (gate).
- Press the bulb flat against something instead of squeezing: there's nothing behind it in reach (it hangs in front
  of the rail); if a model managed it and capped it, that's the real skill and earns 1.0.

**Overlap check** (all 2,079 PRs' task files, 14 Nov 2026): no drain, suction bulb or re-arming task. Clinical tasks
on the board: 6 accepted, 19 pending, 5 sent back, 39 rejected for good; none overlaps.

**Context, scoring and fidelity:** as in `task.json` (710, 945 and 445 characters).

**Build risks** (check on the first GPU run, in this order):
- Everything spawns: the generated meshes, the invisible paddles, the free hinged cap resting on its stop.
- The left fingers drive both paddles in (thin paddles: fingers mustn't slip past their edges).
- The sweep tips the cap over and it falls onto the port; the press doesn't bounce it off.
- `hold_slider` works: it writes the joint stiffness and target through Isaac Lab's articulation API with fallbacks
  for the 3.0 renames, and raises an error naming the methods if none fit.
- The oval visibly flattens (labkit `scale_part`).

## Review criteria self-check (review prompt v22)
- [x] Importance: real ward job, two large occupations; the hard part is physical and simultaneous; no invented
      event; not a variant of the team's tasks; the two accepted spring-holding tasks are named and distinguished.
- [x] Construct validity: real technique and real objects at full scale (bulb, cap, port, tubing, a wound-care training manikin). The
      deforming bulb is approximated and the fidelity note says how.
- [x] Scoring: numeric; partial credit; every outcome covered; let-go and hold checks.
- [x] Clarity: every object with size and material; fixed layout.
- [x] Feasibility: the scripted demo scores 1.0 in 262 steps (13.1 s) of the 1,200-step limit; Dream-1 scores 0.0.
- [x] Blind: nothing identifying in the text.
