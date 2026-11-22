# Thread a cable tie looped round the cords under a TV and pull it snug

Instruction given to the policy: `Close the cable tie snugly around the three cords, then let go of it.`

The tie starts loosely looped round three cords hanging under a TV, its head held in a clip on a post, its tip 4 mm short
of the slot. One hand feeds the tip into the slot, the other takes the tail as it comes out and pulls until the loop is
snug round the cords; the first hand lets go only once the second holds the tail.

## Files

- `task.py`: the `yam_arena` task file: the scene, `score()` and `success()`, the instruction and the scripted
  `demo()`.
- `tie_logic.py`: every number, the tie's geometry, the scorer's measurements and the demo's plans (NumPy only).
- `labkit.py`: object kinds yam_arena lacks. This task uses `CableTie`: the tie as one articulation (a strap of 32
  links on bending joints, a head with a slot, a sprung pawl in the slot).
- `armcheck.py`: offline reach and clearance checks for the demo's plans.
- `demo.py`: offline checks for a harder start (the head held by the robot's hand instead of the clip); not used by
  the submitted demo.
- `test_logic.py`, `test_task.py`, `test_demo.py`: the offline checks (no GPU).
- `task.json`: the task's details. `SPEC.md`: the design notes and the probe history behind them.

## Offline check, no GPU

From this folder, with the yam_arena source on `PYTHONPATH`:

```bash
python -m pytest -q -p no:cacheprovider test_logic.py test_task.py test_demo.py
```

The checks cover the scorer on hand-built tie shapes (the open start, snug, fed in backwards, round only two cords, slid
down the cords, a gripper too close, a moving tie) and the tail length a person measures, the tie's start geometry, the demo's plans through the arm kinematics (every target within reach, the
fingers clear of the cords, the head, the clip and the post), and the demo's closed loop against the probe's.

## Run and score one Dream-1 trial

From the root of this repository:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/cinch-cord-tie/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Close the cable tie snugly around the three cords, then let go of it." \
  --scorer task_score --max-steps 2400 --seed 0 --no-prompt --store-frames --log-dir ~/yam-eval/cinch-dream
```

`task_score` reports 1.0 when `success()` holds for 1 s; otherwise `score()` at the end of the episode, the 0 / 0.25 /
0.5 / 1.0 ladder in `task.json`.

## Run and score the full-score demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/cinch-cord-tie/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Close the cable tie snugly around the three cords, then let go of it." \
  --scorer task_score --max-steps 2400 --seed 0 --no-prompt --store-frames --log-dir ~/yam-eval/cinch-demo
```

The demo is `demo()` in `task.py`: the plan `tie_logic._handover_plan` builds for trial `m-demo`. The left hand takes
the strap's straight run 50 mm back from the tip, edge-on, lines the tip up with the slot and pushes it through. The
right hand comes at the exit end-on, closes on the tail, and only then does the left let go and back away. The right
pulls 60 mm straight out of the slot, then on along a line turned 20 degrees toward the robot, aimed 6 mm toward the
pawl's side of the slot, to 6 mm past where the loop would be snug.

The demo reads the simulator's state: while lining up and pushing, `task.Steering` (run from the task's step hook)
shifts the left hand onto the tip's measured offset from the slot, and extends the push until 22 mm of tail is out.
After the last stroke it carries the right hand on along its pull at 12 mm/s until 138.3 mm of tail is out (at most
24 mm further), past the last strap joint the loop can come to rest on (see `fidelity`); the right then holds 1.5 s,
lets go and backs away. A policy under test gets only the cameras and its joints.

It takes 706 control steps (about 35 s), within the run limit. The scene has no randomness, and the first trial of a
fresh simulator launch repeats exactly from launch to launch: it threads 22.7 mm, draws the tail to 138.1 mm out, and
let go, the tail rests at 137.5 mm, 5.1 mm of slack (1.0). Later trials in the same launch can differ.

## Run limit, runtime and seeds

- Run limit: 2,400 control steps at 20 Hz (120 s of simulated time), `--max-steps 2400` above.
- Runtime per trial on an RTX 4090: about 0.5 s of wall time per control step with the tie's 32-link articulation, so
  a full 2,400-step trial takes about 20 min and the demo about 6 min, plus about 30 s to start the simulator.
- Seeds: none matter. `-E object_jitter=0` fixes the layout, and the task never randomises the tie, cords or clip.
- Tested with inspect-robots 1.0.0, Isaac Sim 6.1.0, Isaac Lab 17.0.2, IsaacLab-Arena 0.3.0, yam_arena 0.1.0,
  PyTorch 2.11.0 (CUDA 12.8), NumPy 2.3.1 and Python 3.12 on an RTX 4090 (driver 595.99).

## Settings

The task file reads three environment variables for the probe runs behind it. The defaults are the submitted task;
leave them unset.

- `TIE_BUILD` (default `jig`): the start. `jig` is the submitted start (looped, head in the clip). `wrap` (the
  strap straight behind the cords; the robot also wraps it), `free` (the tie flat on the console) and `tight` (a
  narrower slot) are harder starts kept for the probe.
- `TIE_PHYS` (default `heavier`): the tie's mass and damping preset (`tie_logic.PHYS`).
- `TIE_RATCHET` (default `latch`): how code assists the pawl (see `fidelity` in `task.json`).

## Simulation fidelity

The full note behind `fidelity` in `task.json` (the task board takes 500 characters).

The strap is 32 rigid 6.25 mm links on bending joints with nylon's bending stiffness (EI about 1e-3 N m^2), no twist, and a permanent set once a joint bends past 20 degrees (nylon yields); the last link is narrower, as the tapered tip. For solver stability the links weigh 15 times a real strap's and carry extra damping. The head is rigid and its slot is modelled 6.4 x 4.0 mm (a real one is about 5.2 x 1.6 mm with a lead-in), so the tip is caught within about 1.6 mm, as a tapered tip in a chamfered slot is. The ratchet is a sprung pawl plate that wedges against the strap when it backs out (friction, not teeth). Code holds the pawl clear until 16 mm of tail is out, because a sprung pawl met head-on flings the tip back in the simulator. Once 0.2 mm of backing out is detected, code also stiffens the pawl, standing in for the teeth. Let go, the loop settles where a link joint meets the pawl, so the tail comes to rest in 6.25 mm steps (a real tie's teeth are much closer together): on this start at 137.5 mm out (1.0) or 131.3 mm (0.5), so the tail has to be drawn past 137.5 mm before it is let go. The clip and its post are drawn only and have no collision: the head is fixed where the clip holds it, because contact shapes that close to the strap changed its path in testing. The cords are rigid fixed cylinders, already gathered; a real bundle gives a little and lets the loop close slightly tighter. Because the strap is a chain of rigid links, a link can catch on the slot's edge partway through the pull. Over eleven scripted runs that varied the demo (the push 0.2 s quicker or slower, the strap 0.4 mm deeper in the fingers, the pull aimed or turned differently, a tighter or later tightening), 6 tied snug and 5 stalled at a link early in the pull, where the fingers slid off the tail.

## Third-party assets

None: every shape is generated in code. The robot is yam_arena's YAM.
