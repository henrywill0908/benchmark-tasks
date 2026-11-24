# Wrap a cable tie round three hanging cords, thread it and pull it snug

Instruction given to the policy: `Close the cable tie snugly around the three cords, then let go of it.`

Three cords hang from the bottom edge of a wall-mounted TV into a power strip on the console. A 370 mm cable tie lies
flat on the console in front of them. Nothing holds the tie, its head or its tip: there is no stand, no clip and no
loop made beforehand. The robot has to pick the tie up, pass it behind the cords, bring both ends round to the front,
feed the tip into the slot in the head, and pull the tail through until the loop is tight, one hand holding the head
while the other pulls.

## Files

- `task.py`: the `yam_arena` task file: the scene, `score()` and `success()`, the instruction and `demo()`.
- `tie_logic.py`: every number, the tie's geometry, the scorer's measurements and the scripted plans (NumPy only).
- `labkit.py`: object kinds yam_arena lacks. This task uses `CableTie` (the tie as one articulation: a strap of 59
  links on bending joints, a head with a slot, a sprung pawl in the slot) and `Prop` (the three cords, fixed, with
  their own friction).
- `armcheck.py`: offline reach and clearance checks for the scripted plans.
- `demo.py`: offline checks from an earlier draft of the start; not used by the task.
- `test_logic.py`, `test_task.py`, `test_demo.py`: the offline checks (no GPU).
- `task.json`: the task's details. `SPEC.md`: the design notes and the probe history behind them.

## Offline check, no GPU

From this folder, with the yam_arena source on `PYTHONPATH`:

```bash
python -m pytest -q -p no:cacheprovider test_logic.py test_task.py test_demo.py
```

The checks cover the scorer on hand-built tie shapes (closed round the cords with each mark in or out of the head,
closed beside the cords, a gripper too close, a moving tie), the tie's start geometry, and the scripted demo through
the arm kinematics (every target within reach, the arms clear of the TV, the wall, the cords and each other).

## Run and score one Dream-1 trial

From the root of this repository:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/cinch-cord-tie/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Close the cable tie snugly around the three cords, then let go of it." \
  --scorer task_score --max-steps 3600 --seed 0 --no-prompt --store-frames --log-dir ~/yam-eval/cinch-dream
```

`task_score` reports 1.0 when `success()` holds for 1 s; otherwise `score()` at the end of the episode, the 0 / 0.25 /
0.5 / 1.0 ladder in `task.json`.

## Run the scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/cinch-cord-tie/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Close the cable tie snugly around the three cords, then let go of it." \
  --scorer task_score --max-steps 3600 --seed 0 --no-prompt --store-frames --log-dir ~/yam-eval/cinch-demo
```

`demo()` plays the whole tie in about 130 s (2,650 steps):

1. Both hands take the tie where it lies (the right the head, the left the tail's end), lift it and turn it on edge,
   where the strap stands out like a ruler. The left lets go.
2. The right, holding the head, carries the strap behind the cords and swings its tail round to the front. The left
   takes the tail there by its faces.
3. The right carries the head round the cords' other side to the front, draws the strap taut on the cords, eases back
   and turns the head until the slot lies along the tail.
4. The left lines the tip up with the slot and pushes 34 mm of tail through.
5. The hands change over. The right lets go of the head and takes the tail where it comes out; the left lets go of
   the strap and takes the head from its free side, level, its pads above and below; the right lets go.
6. The left carries the head to the cords. The right takes the tail again, end on, and draws it along the slot's line
   until the full mark is 12 mm out, the left holding the head against the pull. The right eases back and lets go,
   then the left.

The plan reads the tie's pose as it goes, as a person watching would (where the tail is before the left takes it,
where the slot points before the tip goes in, where the head sits in the fingers, how much tail is out during the
draw). A policy under test gets none of that. Until the loop is down on the cords the hand on the head answers the
whole pull, so the tail does not draw with the head let go: pulled, a free head cocks on the strap and locks.

## Scoring by the marks

The strap carries two white marks, 290 mm and 270 mm from its tip. Once the tie is closed round all three cords,
still, and both grippers are 3 cm away: 1.0 if the centre of the 290 mm mark has come out of the head's far side,
0.5 if the centre of the 270 mm mark has, 0.25 otherwise.

Each mark is a 2.5 mm band across the strap, read against the head's far face: out when the whole band shows beyond
the face or the face cuts it past its middle. With the 290 mm mark at the face 73 mm of strap is left in the loop;
hugging the three cords takes 36 mm. With the 270 mm mark, 93 mm. The marks are minimums: a tie drawn further, down
onto the cords, scores the same.

The marks are read once the tie has been let go, and where they sit comes from how the tie behaves here. With the
head in a gripper's fingers, about 11 mm off the cords, the loop comes taut on the cords at about 300 mm drawn.
Drawn further, the arms and the grips give, and when the hands let go the tail runs back until the loop is slack
again (see the ratchet, below). Three runs of the demo's draw ended at 301.5, 307.5 and 309.3 mm and rested at
296.5, 297.1 and 303.2 mm.

## Run limit, runtime and seeds

- Run limit: 3,600 control steps at 20 Hz (180 s of simulated time), `--max-steps 3600` above.
- Runtime per trial on an RTX 4090: about 0.8 s of wall time per control step with the tie's 59-link strap, so the
  2,650-step demo takes about 37 min and a full 3,600-step trial about 50 min, plus about 30 s to start the simulator.
- Seeds: none matter. `-E object_jitter=0` fixes the layout, and the task never randomises the tie or the cords.

## Settings

The task file reads these environment variables for the probe runs behind it. The defaults are the task; leave them
unset.

- `TIE_BUILD` (default `hang37`): the start. `hang37` is the task. `hang30` is the same start with a 300 mm tie.
  `lift25`, `lift` and `lift30` are an earlier start with the cords lying on the console. `jig`, `wrap`, `free` and
  `tight` are earlier probe starts with the tie already looped round the cords.
- `TIE_PHYS` (default `nylonlead`): the physics preset (`tie_logic.PHYS`): the strap's mass and damping, friction by
  contact, the lead-in on the slot's mouth. `heavier`, `nylon` and the others are earlier probes' presets.
- `TIE_RATCHET` (default `latch`): how code assists the pawl (see below).
- `TIE_TRIAL` (default `h37-hold-l`): which scripted plan `demo()` plays.
- `TIE_TALK` (default off): `1` prints the tail's length, the cords in the loop and the hands' distance once a
  second. It changes nothing in the scene.

## Simulation fidelity

The full note behind `fidelity` in `task.json` (the task board takes 500 characters).

The strap is 59 rigid 6.25 mm links on bending joints with nylon's bending stiffness (EI about 1e-3 N m^2), no twist,
and a permanent set once a joint bends past 20 degrees (nylon yields); the last link is narrower, as the tapered tip.
The joints bend only across the strap's thin side, so a strap held flat droops and a strap held on edge stands out
straight, as a real tie does. For solver stability the links weigh 15 times a real strap's and carry extra damping, so
the flat strap droops more than a real one.

Friction is set by what touches what: the strap on the gripper's pads 0.9 (the pads are rubber), nylon on nylon in the
slot and on the pawl 0.25, nylon on the cords' jackets 0.35. With 0.9 everywhere the strap would not push through the
slot past 23 mm; with 0.3 everywhere the pads lost the strap.

The head is rigid and its slot is modelled 6.4 x 4.0 mm (a real one is about 5.2 x 1.6 mm), so the tip is caught
within about 1.6 mm, as a tapered tip in a chamfered slot is. The slot's mouth has a 0.5 mm lead-in (chamfer) on all
four sides, as a real tie's has. Without it each strap joint that reached the mouth butted on the head's face (the
strap is a chain of square-ended links) and the draw stalled a link at a time. The head's centre of mass and inertia
are set to those of the plain box-walled head: the simulator otherwise works them out from the collision shapes, and
the lead-in's extra shapes shifted them enough to change how the head sat in the fingers.

The ratchet is a sprung pawl plate that wedges against the strap when it backs out (friction, not teeth). Code holds
the pawl clear until 8 mm of tail is out, about where a real tie's first teeth reach its pawl, because a sprung pawl
met head-on flings the tip back in the simulator. Once 0.2 mm of backing out is detected, code also stiffens the pawl
(a little more for each further 0.2 mm), standing in for the teeth, and lets it go again when the tail moves 0.4 mm
forward. A real tie's teeth are about 1 mm apart and hold a loaded tail within that; here the pawl holds by
friction on a nylon face (0.25), which does not hold a tail let go under load: it runs back, 5 to 10 mm in the demo's
runs, until the loop no longer pushes it, and holds from there. The full mark is placed with that in hand.

The three cords are rigid, fixed cylinders from the TV's bottom edge to the power strip, which is fixed too: taut cords
plugged in at both ends, which a tie being closed round them barely moves. So the loop cannot squeeze the cords
together, and "snug" is read from the marks: how much strap is left in the loop. The marks on the strap are drawn
only.

## Third-party assets

None: every shape is generated in code. The robot is yam_arena's YAM.
