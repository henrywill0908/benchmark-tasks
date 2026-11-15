# Put a wound-drain bulb back on suction

A wound-care training manikin lies in bed with a closed-suction wound drain (Jackson-Pratt type). Its silicone bulb has just been emptied
and hangs loose from the bed rail, round, with its cap open. The robot is told only:

> The training manikin's drain bulb has just been emptied. Put it back on suction without tugging on the drain tube.

Suction comes back only if the bulb is squeezed flat and capped **while it is still flat**: that traps a vacuum. A
bulb capped while round looks finished but draws nothing, and fluid collects under the wound. So the robot has to hold
a springy, swinging bulb flat and steady with one gripper while the other tips the cap over and presses it home, then
let go of both. Capping first, or letting go before the cap is home, silently fails; pulling the bulb more than 3 cm
from where it hangs tugs the drain and scores 0.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the bedside scene, the seal, `score()`, `success()` and `demo()` |
| `drain_logic.py` | Geometry, scoring logic and the demo plan as plain numbers (numpy only) |
| `labkit.py` | The team's shared object kit, with the additions this task needs (listed in its docstring) |
| `test_logic.py` | Every scoring outcome and the demo plan's timing, with numpy and pytest only |
| `test_task.py` | The task file loads, documents agree, nothing leaks the technique, demo reach with yam_arena's IK |
| `SPEC.md` | The design specification and the review self-check |

`task.py` imports `labkit`, which hooks yam_arena's scene and embodiment without changing them, so the task runs with
the standard `yam_arena` embodiment and `task_score` scorer. Written against yam-isaac-lab-arena at `bd62771`.

## How the bulb is simulated

- **The bulb hangs free** (labkit `HangingBulb`, one articulation): it swings about a point 16 cm above its centre
  against a soft spring standing in for its strap and tubing, about 1.7 cm for a 1 N sideways push. Pressing down on
  the cap acts through the hanging point, so it doesn't swing it.
- **The bulb's two faces** are invisible paddles on sprung slides (about 10 N to hold flat, 1.6 cm of travel each).
  The white oval on screen is a mesh that the task redraws every step to the gap between them.
- **The cap** is a 16 mm round push-fit cap on a hinge at the port's edge. Its tether springs it open (as a real
  plug's strap does), so it can't fall shut: it seats only when pressed within 2 degrees of flat, about 0.25 N.
  Task code then holds it there (a push-fit plug grips the port), so a seated cap stays seated.
- **The seal**: when the cap seats, task code holds both faces where they are, both ways (a sealed bulb can neither
  spring open nor be squeezed further).
- **The manikin** is visual only, made of generated shapes: head on a pillow, gown, arms, a blanket over the
  legs folded down below the drain's dressing, the feet tenting it at the end of the mattress.

## Run a trial

On the GPU machine, from this folder, in a shell with the yam_arena environment active
(`source ~/IsaacLab-Arena/yam.env`):

```bash
# 1. Offline checks (seconds; no simulator)
python -m pytest -q test_logic.py test_task.py

# 2. The scripted demo, which should reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The training manikin's drain bulb has just been emptied. Put it back on suction without tugging on the drain tube." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/drain-demo

# 3. Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The training manikin's drain bulb has just been emptied. Put it back on suction without tugging on the drain tube." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/drain-dream

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/drain-*/*.json; do inspect-robots video "$log"; done
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scoring

| What the robot does | Score |
|---|---|
| Squeezes the bulb flat, caps it while flat, lets go of both | 1.0 |
| Squeezes and caps it, but a gripper is still at the bulb at the end | 0.5 |
| Caps the bulb while it is round (or less than 70% flat): no suction | 0.25 |
| Caps first, then squeezes (a sealed bulb can't be squeezed) | 0.25 |
| Squeezes but never caps, or does nothing | 0 |
| A gripper pad touches inside the open port (contamination) | 0 |
| The bulb is pulled more than 3 cm from where it hangs (the drain tugged) | 0 |

Definitions and thresholds are in `task.json` and `drain_logic.py`.

## Runtime and seeds

- Nothing in the scene is loose, so the layout is the same with any seed and any `object_jitter`.
- The demo takes about 16 s of simulated time, plus the 1 s success hold; the run limit is 1,200 steps (60 s at 20 Hz).
- The demo's reach was checked on the arm's URDF with an independent solver (every waypoint and every 20 Hz step
  in between, both arms on one clock): under 0.01 mm of error, at least 0.038 rad from a joint limit.

## Note on Dream-1 and the first instruction wording

The first wording was "The wound drain has just been emptied. Put it back on suction without tugging on the drain.", with the setup calling the manikin a patient. Dream-1 refused it on 15 Nov 2026 before any action: "DA-4000 BAD_REQUEST (malformed): unsafe instruction in this scene, refused: Manipulating medical tubing connected directly to a patient can cause physical harm." The scene was always a training manikin, as in nursing skills labs, and a real-world trial is only safe on one; the instruction now says so and names the drain bulb and its tube. The technique is still left out of it. With this wording Dream-1 was not refused: it ran the full 1,200 steps (60 s) and scored 0.0, moving both arms near their bases without reaching the bulb or the cap.

## Not yet checked in Isaac Sim (check on the first GPU run)

- The bulb, cap, rail, manikin and tubing spawn where they should, and the oval redraws as the faces move.
- The left gripper's squeeze drives both faces in (the paddles are thin; the fingers must not slip past them).
- The right gripper's sweep tips the open cap over and stays on top of it (the tether pushes it back up against the
  fingers), then presses it flat; the hold keeps it seated after the fingers lift.
- First GPU run (15 Nov 2026, the earlier version where the cap fell shut under its weight): demo 1.0 in 294 steps,
  cap-first 0.25, no-squeeze 0.25. Second GPU run (15 Nov 2026, RTX 4090, with the push-fit cap, the new manikin and the free-hanging bulb): 44 tests
  pass, demo 1.0 in 262 steps, cap-first 0.25, no-squeeze 0.25. The legs and feet under the blanket were redrawn
  after it (visual only, no colliders). Third GPU run (15 Nov 2026, RTX 4090, the training-manikin wording): 44 tests
  pass, demo 1.0 in 262 steps (official run 1.0), cap-first 0.25, no-squeeze 0.25; Dream-1 0.0.
- The bulb swings as designed and stays within 3 cm while the left hand steadies it and the right presses the cap
  (the probe logs `bulb_moved_mm` and `swing_deg`).
- The hold (`labkit.hold_slider`) keeps the faces put after the left gripper lets go. It first tries a stiff drive
  (rewriting the joint's stiffness and target); if this Isaac Lab lacks those writers, it prints
  `labkit: holding ... by re-setting its joint each step` and instead puts the faces back every control step with
  `write_joint_position_to_sim_index`, the call labkit's reset already uses. The score and the drawn bulb read the
  sealed thickness either way.
